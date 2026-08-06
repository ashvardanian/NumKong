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
#include "numkong/cast/serial.h" // `nk_partial_load_b32x4_serial_`, `nk_partial_load_b64x2_serial_`, `nk_partial_store_b32x4_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  WASM Relaxed SIMD implementations using 2-wide f64 and 4-wide f32 SIMD.
 *  These require WASM trigonometric kernels from trigonometry/v128relaxed.h.
 */

NK_INTERNAL v128_t nk_haversine_f64x2_v128relaxed_(              //
    v128_t first_latitudes_f64x2, v128_t first_longitudes_f64x2, //
    v128_t second_latitudes_f64x2, v128_t second_longitudes_f64x2) {

    v128_t const earth_radius_f64x2 = wasm_f64x2_splat(NK_EARTH_MEDIATORIAL_RADIUS);
    v128_t const half_f64x2 = wasm_f64x2_splat(0.5);
    v128_t const one_f64x2 = wasm_f64x2_splat(1.0);
    v128_t const two_f64x2 = wasm_f64x2_splat(2.0);

    v128_t latitude_delta_f64x2 = wasm_f64x2_sub(second_latitudes_f64x2, first_latitudes_f64x2);
    v128_t longitude_delta_f64x2 = wasm_f64x2_sub(second_longitudes_f64x2, first_longitudes_f64x2);

    // Haversine terms: sin^2(delta/2)
    v128_t latitude_delta_half_f64x2 = wasm_f64x2_mul(latitude_delta_f64x2, half_f64x2);
    v128_t longitude_delta_half_f64x2 = wasm_f64x2_mul(longitude_delta_f64x2, half_f64x2);
    v128_t sin_latitude_delta_half_f64x2 = nk_f64x2_sin_v128relaxed_(latitude_delta_half_f64x2);
    v128_t sin_longitude_delta_half_f64x2 = nk_f64x2_sin_v128relaxed_(longitude_delta_half_f64x2);
    v128_t sin_squared_latitude_delta_half_f64x2 = wasm_f64x2_mul(sin_latitude_delta_half_f64x2,
                                                                  sin_latitude_delta_half_f64x2);
    v128_t sin_squared_longitude_delta_half_f64x2 = wasm_f64x2_mul(sin_longitude_delta_half_f64x2,
                                                                   sin_longitude_delta_half_f64x2);

    // Latitude cosine product
    v128_t cos_first_latitude_f64x2 = nk_f64x2_cos_v128relaxed_(first_latitudes_f64x2);
    v128_t cos_second_latitude_f64x2 = nk_f64x2_cos_v128relaxed_(second_latitudes_f64x2);
    v128_t cos_latitude_product_f64x2 = wasm_f64x2_mul(cos_first_latitude_f64x2, cos_second_latitude_f64x2);

    // a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    v128_t haversine_term_f64x2 = wasm_f64x2_add(
        sin_squared_latitude_delta_half_f64x2,
        wasm_f64x2_mul(cos_latitude_product_f64x2, sin_squared_longitude_delta_half_f64x2));
    // Clamp haversine_term_f64x2 to [0, 1] to prevent NaN from sqrt of negative values
    // relaxed_min/max: 1 instruction (minpd/maxpd) vs 6-9 (with NaN/signed-zero_f64x2 fixup) on x86.
    // Safe because haversine_term_f64x2 is a product of finite sin/cos values — NaN is impossible.
    v128_t zero_f64x2 = wasm_f64x2_splat(0.0);
    haversine_term_f64x2 = wasm_f64x2_relaxed_max(zero_f64x2, wasm_f64x2_relaxed_min(one_f64x2, haversine_term_f64x2));

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    v128_t sqrt_haversine_f64x2 = wasm_f64x2_sqrt(haversine_term_f64x2);
    v128_t sqrt_complement_f64x2 = wasm_f64x2_sqrt(wasm_f64x2_sub(one_f64x2, haversine_term_f64x2));
    v128_t central_angle_f64x2 = wasm_f64x2_mul(
        two_f64x2, nk_f64x2_atan2_v128relaxed_(sqrt_haversine_f64x2, sqrt_complement_f64x2));

    return wasm_f64x2_mul(earth_radius_f64x2, central_angle_f64x2);
}

NK_PUBLIC void nk_haversine_f64_v128relaxed(        //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        v128_t first_latitudes_f64x2 = wasm_v128_load(a_lats);
        v128_t first_longitudes_f64x2 = wasm_v128_load(a_lons);
        v128_t second_latitudes_f64x2 = wasm_v128_load(b_lats);
        v128_t second_longitudes_f64x2 = wasm_v128_load(b_lons);

        v128_t distances_f64x2 = nk_haversine_f64x2_v128relaxed_(first_latitudes_f64x2, first_longitudes_f64x2,
                                                                 second_latitudes_f64x2, second_longitudes_f64x2);
        wasm_v128_store(results, distances_f64x2);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle tail with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        v128_t distances_f64x2 = nk_haversine_f64x2_v128relaxed_(a_lat_vec.v128, a_lon_vec.v128, b_lat_vec.v128,
                                                                 b_lon_vec.v128);
        result_vec.v128 = distances_f64x2;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

NK_INTERNAL v128_t nk_haversine_f32x4_v128relaxed_(              //
    v128_t first_latitudes_f32x4, v128_t first_longitudes_f32x4, //
    v128_t second_latitudes_f32x4, v128_t second_longitudes_f32x4) {

    v128_t const earth_radius_f32x4 = wasm_f32x4_splat((float)NK_EARTH_MEDIATORIAL_RADIUS);
    v128_t const half_f32x4 = wasm_f32x4_splat(0.5f);
    v128_t const one_f32x4 = wasm_f32x4_splat(1.0f);
    v128_t const two_f32x4 = wasm_f32x4_splat(2.0f);

    v128_t latitude_delta_f32x4 = wasm_f32x4_sub(second_latitudes_f32x4, first_latitudes_f32x4);
    v128_t longitude_delta_f32x4 = wasm_f32x4_sub(second_longitudes_f32x4, first_longitudes_f32x4);

    // Haversine terms: sin^2(delta/2)
    v128_t latitude_delta_half_f32x4 = wasm_f32x4_mul(latitude_delta_f32x4, half_f32x4);
    v128_t longitude_delta_half_f32x4 = wasm_f32x4_mul(longitude_delta_f32x4, half_f32x4);
    v128_t sin_latitude_delta_half_f32x4 = nk_f32x4_sin_v128relaxed_(latitude_delta_half_f32x4);
    v128_t sin_longitude_delta_half_f32x4 = nk_f32x4_sin_v128relaxed_(longitude_delta_half_f32x4);
    v128_t sin_squared_latitude_delta_half_f32x4 = wasm_f32x4_mul(sin_latitude_delta_half_f32x4,
                                                                  sin_latitude_delta_half_f32x4);
    v128_t sin_squared_longitude_delta_half_f32x4 = wasm_f32x4_mul(sin_longitude_delta_half_f32x4,
                                                                   sin_longitude_delta_half_f32x4);

    // Latitude cosine product
    v128_t cos_first_latitude_f32x4 = nk_f32x4_cos_v128relaxed_(first_latitudes_f32x4);
    v128_t cos_second_latitude_f32x4 = nk_f32x4_cos_v128relaxed_(second_latitudes_f32x4);
    v128_t cos_latitude_product_f32x4 = wasm_f32x4_mul(cos_first_latitude_f32x4, cos_second_latitude_f32x4);

    // a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    v128_t haversine_term_f32x4 = wasm_f32x4_add(
        sin_squared_latitude_delta_half_f32x4,
        wasm_f32x4_mul(cos_latitude_product_f32x4, sin_squared_longitude_delta_half_f32x4));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    // relaxed_min/max: 1 instruction (minps/maxps) vs 6-9 (with NaN/signed-zero_f32x4 fixup) on x86.
    // Safe because haversine_term_f32x4 is a product of finite sin/cos values — NaN is impossible.
    v128_t zero_f32x4 = wasm_f32x4_splat(0.0f);
    haversine_term_f32x4 = wasm_f32x4_relaxed_max(zero_f32x4, wasm_f32x4_relaxed_min(one_f32x4, haversine_term_f32x4));

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    v128_t sqrt_haversine_f32x4 = wasm_f32x4_sqrt(haversine_term_f32x4);
    v128_t sqrt_complement_f32x4 = wasm_f32x4_sqrt(wasm_f32x4_sub(one_f32x4, haversine_term_f32x4));
    v128_t central_angle_f32x4 = wasm_f32x4_mul(
        two_f32x4, nk_f32x4_atan2_v128relaxed_(sqrt_haversine_f32x4, sqrt_complement_f32x4));

    return wasm_f32x4_mul(earth_radius_f32x4, central_angle_f32x4);
}

NK_PUBLIC void nk_haversine_f32_v128relaxed(        //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        v128_t first_latitudes_f32x4 = wasm_v128_load(a_lats);
        v128_t first_longitudes_f32x4 = wasm_v128_load(a_lons);
        v128_t second_latitudes_f32x4 = wasm_v128_load(b_lats);
        v128_t second_longitudes_f32x4 = wasm_v128_load(b_lons);

        v128_t distances_f32x4 = nk_haversine_f32x4_v128relaxed_(first_latitudes_f32x4, first_longitudes_f32x4,
                                                                 second_latitudes_f32x4, second_longitudes_f32x4);
        wasm_v128_store(results, distances_f32x4);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle tail with partial loads (n can be 0-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        v128_t distances_f32x4 = nk_haversine_f32x4_v128relaxed_(a_lat_vec.v128, a_lon_vec.v128, b_lat_vec.v128,
                                                                 b_lon_vec.v128);
        result_vec.v128 = distances_f32x4;
        nk_partial_store_b32x4_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  WASM Relaxed SIMD helper for Vincenty's geodesic distance on 2 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL v128_t nk_vincenty_f64x2_v128relaxed_(               //
    v128_t first_latitudes_f64x2, v128_t first_longitudes_f64x2, //
    v128_t second_latitudes_f64x2, v128_t second_longitudes_f64x2) {

    v128_t const equatorial_radius_f64x2 = wasm_f64x2_splat(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    v128_t const polar_radius_f64x2 = wasm_f64x2_splat(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    v128_t const flattening_f64x2 = wasm_f64x2_splat(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    v128_t const convergence_threshold_f64x2 = wasm_f64x2_splat(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    v128_t const one_f64x2 = wasm_f64x2_splat(1.0);
    v128_t const two_f64x2 = wasm_f64x2_splat(2.0);
    v128_t const three_f64x2 = wasm_f64x2_splat(3.0);
    v128_t const four_f64x2 = wasm_f64x2_splat(4.0);
    v128_t const six_f64x2 = wasm_f64x2_splat(6.0);
    v128_t const sixteen_f64x2 = wasm_f64x2_splat(16.0);
    v128_t const epsilon_f64x2 = wasm_f64x2_splat(1e-15);

    // Longitude difference
    v128_t longitude_difference_f64x2 = wasm_f64x2_sub(second_longitudes_f64x2, first_longitudes_f64x2);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    v128_t one_minus_f_f64x2 = wasm_f64x2_sub(one_f64x2, flattening_f64x2);
    v128_t tan_first_f64x2 = wasm_f64x2_div(nk_f64x2_sin_v128relaxed_(first_latitudes_f64x2),
                                            nk_f64x2_cos_v128relaxed_(first_latitudes_f64x2));
    v128_t tan_second_f64x2 = wasm_f64x2_div(nk_f64x2_sin_v128relaxed_(second_latitudes_f64x2),
                                             nk_f64x2_cos_v128relaxed_(second_latitudes_f64x2));
    v128_t tan_reduced_first_f64x2 = wasm_f64x2_mul(one_minus_f_f64x2, tan_first_f64x2);
    v128_t tan_reduced_second_f64x2 = wasm_f64x2_mul(one_minus_f_f64x2, tan_second_f64x2);

    // cos(U) = 1/sqrt(1 + tan^2(U)), sin(U) = tan(U) * cos(U)
    v128_t cos_reduced_first_f64x2 = wasm_f64x2_div(
        one_f64x2,
        wasm_f64x2_sqrt(wasm_f64x2_relaxed_madd(tan_reduced_first_f64x2, tan_reduced_first_f64x2, one_f64x2)));
    v128_t sin_reduced_first_f64x2 = wasm_f64x2_mul(tan_reduced_first_f64x2, cos_reduced_first_f64x2);
    v128_t cos_reduced_second_f64x2 = wasm_f64x2_div(
        one_f64x2,
        wasm_f64x2_sqrt(wasm_f64x2_relaxed_madd(tan_reduced_second_f64x2, tan_reduced_second_f64x2, one_f64x2)));
    v128_t sin_reduced_second_f64x2 = wasm_f64x2_mul(tan_reduced_second_f64x2, cos_reduced_second_f64x2);

    // Initialize lambda_f64x2 and tracking variables
    v128_t lambda_f64x2 = longitude_difference_f64x2;
    v128_t sin_angular_distance_f64x2, cos_angular_distance_f64x2, angular_distance_f64x2;
    v128_t sin_azimuth_f64x2, cos_squared_azimuth_f64x2, cos_double_angular_midpoint_f64x2;

    // Track convergence and coincident points using masks
    v128_t converged_mask_i64x2 = wasm_i64x2_splat(0);
    v128_t coincident_mask_i64x2 = wasm_i64x2_splat(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        if (wasm_i8x16_all_true(converged_mask_i64x2)) break;

        v128_t sin_lambda_f64x2 = nk_f64x2_sin_v128relaxed_(lambda_f64x2);
        v128_t cos_lambda_f64x2 = nk_f64x2_cos_v128relaxed_(lambda_f64x2);

        // sin^2(angular_distance_f64x2) = (cos(U2) * sin(l))^2 + (cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(l))^2
        v128_t cross_term_f64x2 = wasm_f64x2_mul(cos_reduced_second_f64x2, sin_lambda_f64x2);
        v128_t mixed_term_f64x2 = wasm_f64x2_sub(
            wasm_f64x2_mul(cos_reduced_first_f64x2, sin_reduced_second_f64x2),
            wasm_f64x2_mul(wasm_f64x2_mul(sin_reduced_first_f64x2, cos_reduced_second_f64x2), cos_lambda_f64x2));
        v128_t sin_angular_dist_sq_f64x2 = wasm_f64x2_relaxed_madd(cross_term_f64x2, cross_term_f64x2,
                                                                   wasm_f64x2_mul(mixed_term_f64x2, mixed_term_f64x2));
        sin_angular_distance_f64x2 = wasm_f64x2_sqrt(sin_angular_dist_sq_f64x2);

        // Check for coincident points (sin_angular_distance_f64x2 ~ 0)
        coincident_mask_i64x2 = wasm_f64x2_lt(sin_angular_distance_f64x2, epsilon_f64x2);

        // cos(angular_distance_f64x2) = sin(U1) * sin(U2) + cos(U1) * cos(U2) * cos(l)
        cos_angular_distance_f64x2 = wasm_f64x2_relaxed_madd(
            wasm_f64x2_mul(cos_reduced_first_f64x2, cos_reduced_second_f64x2), cos_lambda_f64x2,
            wasm_f64x2_mul(sin_reduced_first_f64x2, sin_reduced_second_f64x2));

        // angular_distance_f64x2 = atan2(sin, cos)
        angular_distance_f64x2 = nk_f64x2_atan2_v128relaxed_(sin_angular_distance_f64x2, cos_angular_distance_f64x2);

        // sin(azimuth) = cos(U1) * cos(U2) * sin(l) / sin(angular_distance_f64x2)
        // Avoid division by zero by using blending
        // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
        // Safe because mask is from comparison (all-ones or all-zeros per lane).
        v128_t safe_sin_angular_i64x2 = wasm_i64x2_relaxed_laneselect(one_f64x2, sin_angular_distance_f64x2,
                                                                      coincident_mask_i64x2);
        sin_azimuth_f64x2 = wasm_f64x2_div(
            wasm_f64x2_mul(wasm_f64x2_mul(cos_reduced_first_f64x2, cos_reduced_second_f64x2), sin_lambda_f64x2),
            safe_sin_angular_i64x2);
        cos_squared_azimuth_f64x2 = wasm_f64x2_relaxed_nmadd(sin_azimuth_f64x2, sin_azimuth_f64x2, one_f64x2);

        // Handle equatorial case: cos^2(a) ~ 0
        v128_t equatorial_mask_f64x2 = wasm_f64x2_lt(cos_squared_azimuth_f64x2, epsilon_f64x2);
        v128_t safe_cos_sq_azimuth_i64x2 = wasm_i64x2_relaxed_laneselect(one_f64x2, cos_squared_azimuth_f64x2,
                                                                         equatorial_mask_f64x2);

        // cos(2sm) = cos(s) - 2 * sin(U1) * sin(U2) / cos^2(a)
        v128_t sin_product_f64x2 = wasm_f64x2_mul(sin_reduced_first_f64x2, sin_reduced_second_f64x2);
        cos_double_angular_midpoint_f64x2 = wasm_f64x2_sub(
            cos_angular_distance_f64x2,
            wasm_f64x2_div(wasm_f64x2_mul(two_f64x2, sin_product_f64x2), safe_cos_sq_azimuth_i64x2));
        cos_double_angular_midpoint_f64x2 = wasm_i64x2_relaxed_laneselect(
            wasm_f64x2_splat(0.0), cos_double_angular_midpoint_f64x2, equatorial_mask_f64x2);

        // C = f/16 * cos^2(a) * (4 + f*(4 - 3*cos^2(a)))
        v128_t correction_factor_f64x2 = wasm_f64x2_mul(
            wasm_f64x2_div(flattening_f64x2, sixteen_f64x2),
            wasm_f64x2_mul(
                cos_squared_azimuth_f64x2,
                wasm_f64x2_relaxed_madd(flattening_f64x2,
                                        wasm_f64x2_relaxed_nmadd(three_f64x2, cos_squared_azimuth_f64x2, four_f64x2),
                                        four_f64x2)));

        // l' = L + (1-C) * f * sin(a) * (s + C * sin(s) * (cos(2sm) + C * cos(s) * (-1 + 2 * cos^2(2sm))))
        v128_t cos_2sm_sq_f64x2 = wasm_f64x2_mul(cos_double_angular_midpoint_f64x2, cos_double_angular_midpoint_f64x2);
        // innermost_f64x2 = -1 + 2 * cos^2(2sm)
        v128_t innermost_f64x2 = wasm_f64x2_relaxed_madd(two_f64x2, cos_2sm_sq_f64x2, wasm_f64x2_splat(-1.0));
        // middle_f64x2 = cos(2sm) + C * cos(s) * innermost_f64x2
        v128_t middle_f64x2 = wasm_f64x2_relaxed_madd(
            wasm_f64x2_mul(correction_factor_f64x2, cos_angular_distance_f64x2), innermost_f64x2,
            cos_double_angular_midpoint_f64x2);
        // inner_f64x2 = C * sin(s) * middle_f64x2
        v128_t inner_f64x2 = wasm_f64x2_mul(wasm_f64x2_mul(correction_factor_f64x2, sin_angular_distance_f64x2),
                                            middle_f64x2);

        // l' = L + (1-C) * f * sin_a * (s + inner_f64x2)
        v128_t lambda_new_f64x2 = wasm_f64x2_relaxed_madd(
            wasm_f64x2_mul(wasm_f64x2_mul(wasm_f64x2_sub(one_f64x2, correction_factor_f64x2), flattening_f64x2),
                           sin_azimuth_f64x2),
            wasm_f64x2_add(angular_distance_f64x2, inner_f64x2), longitude_difference_f64x2);

        // Check convergence: |l - l'| < threshold
        v128_t lambda_diff_f64x2 = wasm_f64x2_sub(lambda_new_f64x2, lambda_f64x2);
        v128_t lambda_diff_abs_f64x2 = wasm_f64x2_abs(lambda_diff_f64x2);
        v128_t newly_converged_f64x2 = wasm_f64x2_lt(lambda_diff_abs_f64x2, convergence_threshold_f64x2);
        converged_mask_i64x2 = wasm_v128_or(converged_mask_i64x2, newly_converged_f64x2);

        // Only update lambda_f64x2 for non-converged lanes
        // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
        // Safe because mask is from comparison (all-ones or all-zeros per lane).
        lambda_f64x2 = wasm_i64x2_relaxed_laneselect(lambda_f64x2, lambda_new_f64x2, converged_mask_i64x2);
    }

    // Final distance calculation
    // u^2 = cos^2(a) * (a^2 - b^2) / b^2
    v128_t a_sq_f64x2 = wasm_f64x2_mul(equatorial_radius_f64x2, equatorial_radius_f64x2);
    v128_t b_sq_f64x2 = wasm_f64x2_mul(polar_radius_f64x2, polar_radius_f64x2);
    v128_t u_squared_f64x2 = wasm_f64x2_div(
        wasm_f64x2_mul(cos_squared_azimuth_f64x2, wasm_f64x2_sub(a_sq_f64x2, b_sq_f64x2)), b_sq_f64x2);

    // A = 1 + u^2/16384 * (4096 + u^2*(-768 + u^2*(320 - 175*u^2)))
    v128_t series_a_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, wasm_f64x2_splat(-175.0), wasm_f64x2_splat(320.0));
    series_a_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, series_a_f64x2, wasm_f64x2_splat(-768.0));
    series_a_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, series_a_f64x2, wasm_f64x2_splat(4096.0));
    series_a_f64x2 = wasm_f64x2_relaxed_madd(wasm_f64x2_div(u_squared_f64x2, wasm_f64x2_splat(16384.0)), series_a_f64x2,
                                             one_f64x2);

    // B = u^2/1024 * (256 + u^2*(-128 + u^2*(74 - 47*u^2)))
    v128_t series_b_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, wasm_f64x2_splat(-47.0), wasm_f64x2_splat(74.0));
    series_b_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, series_b_f64x2, wasm_f64x2_splat(-128.0));
    series_b_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, series_b_f64x2, wasm_f64x2_splat(256.0));
    series_b_f64x2 = wasm_f64x2_mul(wasm_f64x2_div(u_squared_f64x2, wasm_f64x2_splat(1024.0)), series_b_f64x2);

    // Delta-sigma calculation
    v128_t cos_2sm_sq_f64x2 = wasm_f64x2_mul(cos_double_angular_midpoint_f64x2, cos_double_angular_midpoint_f64x2);
    v128_t sin_sq_f64x2 = wasm_f64x2_mul(sin_angular_distance_f64x2, sin_angular_distance_f64x2);
    v128_t term1_f64x2 = wasm_f64x2_relaxed_madd(two_f64x2, cos_2sm_sq_f64x2, wasm_f64x2_splat(-1.0));
    term1_f64x2 = wasm_f64x2_mul(cos_angular_distance_f64x2, term1_f64x2);
    v128_t term2_f64x2 = wasm_f64x2_relaxed_madd(four_f64x2, sin_sq_f64x2, wasm_f64x2_splat(-3.0));
    v128_t term3_f64x2 = wasm_f64x2_relaxed_madd(four_f64x2, cos_2sm_sq_f64x2, wasm_f64x2_splat(-3.0));
    term2_f64x2 = wasm_f64x2_mul(
        wasm_f64x2_mul(wasm_f64x2_div(series_b_f64x2, six_f64x2), cos_double_angular_midpoint_f64x2),
        wasm_f64x2_mul(term2_f64x2, term3_f64x2));
    v128_t delta_sigma_f64x2 = wasm_f64x2_mul(
        series_b_f64x2, wasm_f64x2_mul(sin_angular_distance_f64x2,
                                       wasm_f64x2_add(cos_double_angular_midpoint_f64x2,
                                                      wasm_f64x2_mul(wasm_f64x2_div(series_b_f64x2, four_f64x2),
                                                                     wasm_f64x2_sub(term1_f64x2, term2_f64x2)))));

    // s = b * A * (s - ds)
    v128_t distances_f64x2 = wasm_f64x2_mul(wasm_f64x2_mul(polar_radius_f64x2, series_a_f64x2),
                                            wasm_f64x2_sub(angular_distance_f64x2, delta_sigma_f64x2));

    // Set coincident points to zero
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    distances_f64x2 = wasm_i64x2_relaxed_laneselect(wasm_f64x2_splat(0.0), distances_f64x2, coincident_mask_i64x2);

    return distances_f64x2;
}

NK_PUBLIC void nk_vincenty_f64_v128relaxed(         //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        v128_t first_latitudes_f64x2 = wasm_v128_load(a_lats);
        v128_t first_longitudes_f64x2 = wasm_v128_load(a_lons);
        v128_t second_latitudes_f64x2 = wasm_v128_load(b_lats);
        v128_t second_longitudes_f64x2 = wasm_v128_load(b_lons);

        v128_t distances_f64x2 = nk_vincenty_f64x2_v128relaxed_(first_latitudes_f64x2, first_longitudes_f64x2,
                                                                second_latitudes_f64x2, second_longitudes_f64x2);
        wasm_v128_store(results, distances_f64x2);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle remaining elements with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        v128_t distances_f64x2 = nk_vincenty_f64x2_v128relaxed_(a_lat_vec.v128, a_lon_vec.v128, b_lat_vec.v128,
                                                                b_lon_vec.v128);
        result_vec.v128 = distances_f64x2;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  WASM Relaxed SIMD helper for Vincenty's geodesic distance on 4 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL v128_t nk_vincenty_f32x4_v128relaxed_(               //
    v128_t first_latitudes_f32x4, v128_t first_longitudes_f32x4, //
    v128_t second_latitudes_f32x4, v128_t second_longitudes_f32x4) {

    v128_t const equatorial_radius_f32x4 = wasm_f32x4_splat((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    v128_t const polar_radius_f32x4 = wasm_f32x4_splat((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    v128_t const flattening_f32x4 = wasm_f32x4_splat(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    v128_t const convergence_threshold_f32x4 = wasm_f32x4_splat(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32);
    v128_t const one_f32x4 = wasm_f32x4_splat(1.0f);
    v128_t const two_f32x4 = wasm_f32x4_splat(2.0f);
    v128_t const three_f32x4 = wasm_f32x4_splat(3.0f);
    v128_t const four_f32x4 = wasm_f32x4_splat(4.0f);
    v128_t const six_f32x4 = wasm_f32x4_splat(6.0f);
    v128_t const sixteen_f32x4 = wasm_f32x4_splat(16.0f);
    v128_t const epsilon_f32x4 = wasm_f32x4_splat(1e-7f);

    // Longitude difference
    v128_t longitude_difference_f32x4 = wasm_f32x4_sub(second_longitudes_f32x4, first_longitudes_f32x4);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    v128_t one_minus_f_f32x4 = wasm_f32x4_sub(one_f32x4, flattening_f32x4);
    v128_t tan_first_f32x4 = wasm_f32x4_div(nk_f32x4_sin_v128relaxed_(first_latitudes_f32x4),
                                            nk_f32x4_cos_v128relaxed_(first_latitudes_f32x4));
    v128_t tan_second_f32x4 = wasm_f32x4_div(nk_f32x4_sin_v128relaxed_(second_latitudes_f32x4),
                                             nk_f32x4_cos_v128relaxed_(second_latitudes_f32x4));
    v128_t tan_reduced_first_f32x4 = wasm_f32x4_mul(one_minus_f_f32x4, tan_first_f32x4);
    v128_t tan_reduced_second_f32x4 = wasm_f32x4_mul(one_minus_f_f32x4, tan_second_f32x4);

    // cos(U) = 1/sqrt(1 + tan^2(U)), sin(U) = tan(U) * cos(U)
    v128_t cos_reduced_first_f32x4 = wasm_f32x4_div(
        one_f32x4,
        wasm_f32x4_sqrt(wasm_f32x4_relaxed_madd(tan_reduced_first_f32x4, tan_reduced_first_f32x4, one_f32x4)));
    v128_t sin_reduced_first_f32x4 = wasm_f32x4_mul(tan_reduced_first_f32x4, cos_reduced_first_f32x4);
    v128_t cos_reduced_second_f32x4 = wasm_f32x4_div(
        one_f32x4,
        wasm_f32x4_sqrt(wasm_f32x4_relaxed_madd(tan_reduced_second_f32x4, tan_reduced_second_f32x4, one_f32x4)));
    v128_t sin_reduced_second_f32x4 = wasm_f32x4_mul(tan_reduced_second_f32x4, cos_reduced_second_f32x4);

    // Initialize lambda_f32x4 and tracking variables
    v128_t lambda_f32x4 = longitude_difference_f32x4;
    v128_t sin_angular_distance_f32x4, cos_angular_distance_f32x4, angular_distance_f32x4;
    v128_t sin_azimuth_f32x4, cos_squared_azimuth_f32x4, cos_double_angular_midpoint_f32x4;

    // Track convergence and coincident points using masks
    v128_t converged_mask_i32x4 = wasm_i32x4_splat(0);
    v128_t coincident_mask_i32x4 = wasm_i32x4_splat(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        if (wasm_i8x16_all_true(converged_mask_i32x4)) break;

        v128_t sin_lambda_f32x4 = nk_f32x4_sin_v128relaxed_(lambda_f32x4);
        v128_t cos_lambda_f32x4 = nk_f32x4_cos_v128relaxed_(lambda_f32x4);

        // sin^2(angular_distance_f32x4) = (cos(U2) * sin(l))^2 + (cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(l))^2
        v128_t cross_term_f32x4 = wasm_f32x4_mul(cos_reduced_second_f32x4, sin_lambda_f32x4);
        v128_t mixed_term_f32x4 = wasm_f32x4_sub(
            wasm_f32x4_mul(cos_reduced_first_f32x4, sin_reduced_second_f32x4),
            wasm_f32x4_mul(wasm_f32x4_mul(sin_reduced_first_f32x4, cos_reduced_second_f32x4), cos_lambda_f32x4));
        v128_t sin_angular_dist_sq_f32x4 = wasm_f32x4_relaxed_madd(cross_term_f32x4, cross_term_f32x4,
                                                                   wasm_f32x4_mul(mixed_term_f32x4, mixed_term_f32x4));
        sin_angular_distance_f32x4 = wasm_f32x4_sqrt(sin_angular_dist_sq_f32x4);

        // Check for coincident points (sin_angular_distance_f32x4 ~ 0)
        coincident_mask_i32x4 = wasm_f32x4_lt(sin_angular_distance_f32x4, epsilon_f32x4);

        // cos(angular_distance_f32x4) = sin(U1) * sin(U2) + cos(U1) * cos(U2) * cos(l)
        cos_angular_distance_f32x4 = wasm_f32x4_relaxed_madd(
            wasm_f32x4_mul(cos_reduced_first_f32x4, cos_reduced_second_f32x4), cos_lambda_f32x4,
            wasm_f32x4_mul(sin_reduced_first_f32x4, sin_reduced_second_f32x4));

        // angular_distance_f32x4 = atan2(sin, cos)
        angular_distance_f32x4 = nk_f32x4_atan2_v128relaxed_(sin_angular_distance_f32x4, cos_angular_distance_f32x4);

        // sin(azimuth) = cos(U1) * cos(U2) * sin(l) / sin(angular_distance_f32x4)
        // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
        // Safe because mask is from comparison (all-ones or all-zeros per lane).
        v128_t safe_sin_angular_i32x4 = wasm_i32x4_relaxed_laneselect(one_f32x4, sin_angular_distance_f32x4,
                                                                      coincident_mask_i32x4);
        sin_azimuth_f32x4 = wasm_f32x4_div(
            wasm_f32x4_mul(wasm_f32x4_mul(cos_reduced_first_f32x4, cos_reduced_second_f32x4), sin_lambda_f32x4),
            safe_sin_angular_i32x4);
        cos_squared_azimuth_f32x4 = wasm_f32x4_relaxed_nmadd(sin_azimuth_f32x4, sin_azimuth_f32x4, one_f32x4);

        // Handle equatorial case: cos^2(a) ~ 0
        v128_t equatorial_mask_f32x4 = wasm_f32x4_lt(cos_squared_azimuth_f32x4, epsilon_f32x4);
        v128_t safe_cos_sq_azimuth_i32x4 = wasm_i32x4_relaxed_laneselect(one_f32x4, cos_squared_azimuth_f32x4,
                                                                         equatorial_mask_f32x4);

        // cos(2sm) = cos(s) - 2 * sin(U1) * sin(U2) / cos^2(a)
        v128_t sin_product_f32x4 = wasm_f32x4_mul(sin_reduced_first_f32x4, sin_reduced_second_f32x4);
        cos_double_angular_midpoint_f32x4 = wasm_f32x4_sub(
            cos_angular_distance_f32x4,
            wasm_f32x4_div(wasm_f32x4_mul(two_f32x4, sin_product_f32x4), safe_cos_sq_azimuth_i32x4));
        cos_double_angular_midpoint_f32x4 = wasm_i32x4_relaxed_laneselect(
            wasm_f32x4_splat(0.0f), cos_double_angular_midpoint_f32x4, equatorial_mask_f32x4);

        // C = f/16 * cos^2(a) * (4 + f*(4 - 3*cos^2(a)))
        v128_t correction_factor_f32x4 = wasm_f32x4_mul(
            wasm_f32x4_div(flattening_f32x4, sixteen_f32x4),
            wasm_f32x4_mul(
                cos_squared_azimuth_f32x4,
                wasm_f32x4_relaxed_madd(flattening_f32x4,
                                        wasm_f32x4_relaxed_nmadd(three_f32x4, cos_squared_azimuth_f32x4, four_f32x4),
                                        four_f32x4)));

        // l' = L + (1-C) * f * sin(a) * (s + C * sin(s) * (cos(2sm) + C * cos(s) * (-1 + 2 * cos^2(2sm))))
        v128_t cos_2sm_sq_f32x4 = wasm_f32x4_mul(cos_double_angular_midpoint_f32x4, cos_double_angular_midpoint_f32x4);
        v128_t innermost_f32x4 = wasm_f32x4_relaxed_madd(two_f32x4, cos_2sm_sq_f32x4, wasm_f32x4_splat(-1.0f));
        v128_t middle_f32x4 = wasm_f32x4_relaxed_madd(
            wasm_f32x4_mul(correction_factor_f32x4, cos_angular_distance_f32x4), innermost_f32x4,
            cos_double_angular_midpoint_f32x4);
        v128_t inner_f32x4 = wasm_f32x4_mul(wasm_f32x4_mul(correction_factor_f32x4, sin_angular_distance_f32x4),
                                            middle_f32x4);

        v128_t lambda_new_f32x4 = wasm_f32x4_relaxed_madd(
            wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_sub(one_f32x4, correction_factor_f32x4), flattening_f32x4),
                           sin_azimuth_f32x4),
            wasm_f32x4_add(angular_distance_f32x4, inner_f32x4), longitude_difference_f32x4);

        // Check convergence: |l - l'| < threshold
        v128_t lambda_diff_f32x4 = wasm_f32x4_sub(lambda_new_f32x4, lambda_f32x4);
        v128_t lambda_diff_abs_f32x4 = wasm_f32x4_abs(lambda_diff_f32x4);
        v128_t newly_converged_f32x4 = wasm_f32x4_lt(lambda_diff_abs_f32x4, convergence_threshold_f32x4);
        converged_mask_i32x4 = wasm_v128_or(converged_mask_i32x4, newly_converged_f32x4);

        // Only update lambda_f32x4 for non-converged lanes
        // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
        // Safe because mask is from comparison (all-ones or all-zeros per lane).
        lambda_f32x4 = wasm_i32x4_relaxed_laneselect(lambda_f32x4, lambda_new_f32x4, converged_mask_i32x4);
    }

    // Final distance calculation
    v128_t a_sq_f32x4 = wasm_f32x4_mul(equatorial_radius_f32x4, equatorial_radius_f32x4);
    v128_t b_sq_f32x4 = wasm_f32x4_mul(polar_radius_f32x4, polar_radius_f32x4);
    v128_t u_squared_f32x4 = wasm_f32x4_div(
        wasm_f32x4_mul(cos_squared_azimuth_f32x4, wasm_f32x4_sub(a_sq_f32x4, b_sq_f32x4)), b_sq_f32x4);

    // A = 1 + u^2/16384 * (4096 + u^2*(-768 + u^2*(320 - 175*u^2)))
    v128_t series_a_f32x4 = wasm_f32x4_relaxed_madd(u_squared_f32x4, wasm_f32x4_splat(-175.0f),
                                                    wasm_f32x4_splat(320.0f));
    series_a_f32x4 = wasm_f32x4_relaxed_madd(u_squared_f32x4, series_a_f32x4, wasm_f32x4_splat(-768.0f));
    series_a_f32x4 = wasm_f32x4_relaxed_madd(u_squared_f32x4, series_a_f32x4, wasm_f32x4_splat(4096.0f));
    series_a_f32x4 = wasm_f32x4_relaxed_madd(wasm_f32x4_div(u_squared_f32x4, wasm_f32x4_splat(16384.0f)),
                                             series_a_f32x4, one_f32x4);

    // B = u^2/1024 * (256 + u^2*(-128 + u^2*(74 - 47*u^2)))
    v128_t series_b_f32x4 = wasm_f32x4_relaxed_madd(u_squared_f32x4, wasm_f32x4_splat(-47.0f), wasm_f32x4_splat(74.0f));
    series_b_f32x4 = wasm_f32x4_relaxed_madd(u_squared_f32x4, series_b_f32x4, wasm_f32x4_splat(-128.0f));
    series_b_f32x4 = wasm_f32x4_relaxed_madd(u_squared_f32x4, series_b_f32x4, wasm_f32x4_splat(256.0f));
    series_b_f32x4 = wasm_f32x4_mul(wasm_f32x4_div(u_squared_f32x4, wasm_f32x4_splat(1024.0f)), series_b_f32x4);

    // Delta-sigma calculation
    v128_t cos_2sm_sq_f32x4 = wasm_f32x4_mul(cos_double_angular_midpoint_f32x4, cos_double_angular_midpoint_f32x4);
    v128_t sin_sq_f32x4 = wasm_f32x4_mul(sin_angular_distance_f32x4, sin_angular_distance_f32x4);
    v128_t term1_f32x4 = wasm_f32x4_relaxed_madd(two_f32x4, cos_2sm_sq_f32x4, wasm_f32x4_splat(-1.0f));
    term1_f32x4 = wasm_f32x4_mul(cos_angular_distance_f32x4, term1_f32x4);
    v128_t term2_f32x4 = wasm_f32x4_relaxed_madd(four_f32x4, sin_sq_f32x4, wasm_f32x4_splat(-3.0f));
    v128_t term3_f32x4 = wasm_f32x4_relaxed_madd(four_f32x4, cos_2sm_sq_f32x4, wasm_f32x4_splat(-3.0f));
    term2_f32x4 = wasm_f32x4_mul(
        wasm_f32x4_mul(wasm_f32x4_div(series_b_f32x4, six_f32x4), cos_double_angular_midpoint_f32x4),
        wasm_f32x4_mul(term2_f32x4, term3_f32x4));
    v128_t delta_sigma_f32x4 = wasm_f32x4_mul(
        series_b_f32x4, wasm_f32x4_mul(sin_angular_distance_f32x4,
                                       wasm_f32x4_add(cos_double_angular_midpoint_f32x4,
                                                      wasm_f32x4_mul(wasm_f32x4_div(series_b_f32x4, four_f32x4),
                                                                     wasm_f32x4_sub(term1_f32x4, term2_f32x4)))));

    // s = b * A * (s - ds)
    v128_t distances_f32x4 = wasm_f32x4_mul(wasm_f32x4_mul(polar_radius_f32x4, series_a_f32x4),
                                            wasm_f32x4_sub(angular_distance_f32x4, delta_sigma_f32x4));

    // Set coincident points to zero
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    distances_f32x4 = wasm_i32x4_relaxed_laneselect(wasm_f32x4_splat(0.0f), distances_f32x4, coincident_mask_i32x4);

    return distances_f32x4;
}

NK_PUBLIC void nk_vincenty_f32_v128relaxed(         //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        v128_t first_latitudes_f32x4 = wasm_v128_load(a_lats);
        v128_t first_longitudes_f32x4 = wasm_v128_load(a_lons);
        v128_t second_latitudes_f32x4 = wasm_v128_load(b_lats);
        v128_t second_longitudes_f32x4 = wasm_v128_load(b_lons);

        v128_t distances_f32x4 = nk_vincenty_f32x4_v128relaxed_(first_latitudes_f32x4, first_longitudes_f32x4,
                                                                second_latitudes_f32x4, second_longitudes_f32x4);
        wasm_v128_store(results, distances_f32x4);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        v128_t distances_f32x4 = nk_vincenty_f32x4_v128relaxed_(a_lat_vec.v128, a_lon_vec.v128, b_lat_vec.v128,
                                                                b_lon_vec.v128);
        result_vec.v128 = distances_f32x4;
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
