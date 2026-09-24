/**
 *  @brief SIMD-accelerated Geospatial Distances for NEON.
 *  @file include/numkong/geospatial/neon.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/geospatial.h
 *
 *  @section geospatial_neon_instructions Key NEON Geospatial Instructions
 *
 *      Intrinsic               Instruction     M1 Firestorm    Graviton 3      Graviton 4
 *      vfmaq_f32               FMLA.S (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vfmaq_f64               FMLA.D (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vsqrtq_f32              FSQRT.S (vec)   10c @ V02       10c @ V02       9c @ V02
 *      vsqrtq_f64              FSQRT.D (vec)   13c @ V02       16c @ V02       16c @ V02
 */
#ifndef NK_GEOSPATIAL_NEON_H
#define NK_GEOSPATIAL_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"
#include "numkong/trigonometry/neon.h" // `nk_sin_f64x2_neon_`, `nk_cos_f64x2_neon_`, `nk_atan2_f64x2_neon_`, etc.

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/*  NEON implementations using 2-wide f64 and 4-wide f32 SIMD.
 *  These require NEON trigonometric kernels from trigonometry/neon.h.
 */

NK_INTERNAL float64x2_t nk_haversine_f64x2_neon_(              //
    float64x2_t first_latitudes, float64x2_t first_longitudes, //
    float64x2_t second_latitudes, float64x2_t second_longitudes) {

    float64x2_t const earth_radius = vdupq_n_f64(NK_EARTH_MEDIATORIAL_RADIUS);
    float64x2_t const half = vdupq_n_f64(0.5);
    float64x2_t const one = vdupq_n_f64(1.0);
    float64x2_t const two = vdupq_n_f64(2.0);

    float64x2_t latitude_delta = vsubq_f64(second_latitudes, first_latitudes);
    float64x2_t longitude_delta = vsubq_f64(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    float64x2_t latitude_delta_half = vmulq_f64(latitude_delta, half);
    float64x2_t longitude_delta_half = vmulq_f64(longitude_delta, half);
    float64x2_t sin_latitude_delta_half = nk_sin_f64x2_neon_(latitude_delta_half);
    float64x2_t sin_longitude_delta_half = nk_sin_f64x2_neon_(longitude_delta_half);
    float64x2_t sin_squared_latitude_delta_half = vmulq_f64(sin_latitude_delta_half, sin_latitude_delta_half);
    float64x2_t sin_squared_longitude_delta_half = vmulq_f64(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    float64x2_t cos_first_latitude = nk_cos_f64x2_neon_(first_latitudes);
    float64x2_t cos_second_latitude = nk_cos_f64x2_neon_(second_latitudes);
    float64x2_t cos_latitude_product = vmulq_f64(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    float64x2_t haversine_term = vaddq_f64(sin_squared_latitude_delta_half,
                                           vmulq_f64(cos_latitude_product, sin_squared_longitude_delta_half));
    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    float64x2_t zero = vdupq_n_f64(0.0);
    haversine_term = vmaxq_f64(zero, vminq_f64(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    float64x2_t sqrt_haversine = vsqrtq_f64(haversine_term);
    float64x2_t sqrt_complement = vsqrtq_f64(vsubq_f64(one, haversine_term));
    float64x2_t central_angle = vmulq_f64(two, nk_atan2_f64x2_neon_(sqrt_haversine, sqrt_complement));

    return vmulq_f64(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f64_neon(               //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        float64x2_t first_latitudes = vld1q_f64(a_lats);
        float64x2_t first_longitudes = vld1q_f64(a_lons);
        float64x2_t second_latitudes = vld1q_f64(b_lats);
        float64x2_t second_longitudes = vld1q_f64(b_lons);

        float64x2_t distances = nk_haversine_f64x2_neon_(first_latitudes, first_longitudes, second_latitudes,
                                                         second_longitudes);
        vst1q_f64(results, distances);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle tail with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        float64x2_t distances = nk_haversine_f64x2_neon_(a_lat_vec.f64x2, a_lon_vec.f64x2, b_lat_vec.f64x2,
                                                         b_lon_vec.f64x2);
        result_vec.f64x2 = distances;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

NK_INTERNAL float32x4_t nk_haversine_f32x4_neon_(              //
    float32x4_t first_latitudes, float32x4_t first_longitudes, //
    float32x4_t second_latitudes, float32x4_t second_longitudes) {

    float32x4_t const earth_radius = vdupq_n_f32((float)NK_EARTH_MEDIATORIAL_RADIUS);
    float32x4_t const half = vdupq_n_f32(0.5f);
    float32x4_t const one = vdupq_n_f32(1.0f);
    float32x4_t const two = vdupq_n_f32(2.0f);

    float32x4_t latitude_delta = vsubq_f32(second_latitudes, first_latitudes);
    float32x4_t longitude_delta = vsubq_f32(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    float32x4_t latitude_delta_half = vmulq_f32(latitude_delta, half);
    float32x4_t longitude_delta_half = vmulq_f32(longitude_delta, half);
    float32x4_t sin_latitude_delta_half = nk_sin_f32x4_neon_(latitude_delta_half);
    float32x4_t sin_longitude_delta_half = nk_sin_f32x4_neon_(longitude_delta_half);
    float32x4_t sin_squared_latitude_delta_half = vmulq_f32(sin_latitude_delta_half, sin_latitude_delta_half);
    float32x4_t sin_squared_longitude_delta_half = vmulq_f32(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    float32x4_t cos_first_latitude = nk_cos_f32x4_neon_(first_latitudes);
    float32x4_t cos_second_latitude = nk_cos_f32x4_neon_(second_latitudes);
    float32x4_t cos_latitude_product = vmulq_f32(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    float32x4_t haversine_term = vaddq_f32(sin_squared_latitude_delta_half,
                                           vmulq_f32(cos_latitude_product, sin_squared_longitude_delta_half));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    float32x4_t zero = vdupq_n_f32(0.0f);
    haversine_term = vmaxq_f32(zero, vminq_f32(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    float32x4_t sqrt_haversine = vsqrtq_f32(haversine_term);
    float32x4_t sqrt_complement = vsqrtq_f32(vsubq_f32(one, haversine_term));
    float32x4_t central_angle = vmulq_f32(two, nk_atan2_f32x4_neon_(sqrt_haversine, sqrt_complement));

    return vmulq_f32(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f32_neon(               //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        float32x4_t first_latitudes = vld1q_f32(a_lats);
        float32x4_t first_longitudes = vld1q_f32(a_lons);
        float32x4_t second_latitudes = vld1q_f32(b_lats);
        float32x4_t second_longitudes = vld1q_f32(b_lons);

        float32x4_t distances = nk_haversine_f32x4_neon_(first_latitudes, first_longitudes, second_latitudes,
                                                         second_longitudes);
        vst1q_f32(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle tail with partial loads (n can be 0-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        float32x4_t distances = nk_haversine_f32x4_neon_(a_lat_vec.f32x4, a_lon_vec.f32x4, b_lat_vec.f32x4,
                                                         b_lon_vec.f32x4);
        result_vec.f32x4 = distances;
        nk_partial_store_b32x4_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  NEON helper for Vincenty's geodesic distance on 2 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL float64x2_t nk_vincenty_f64x2_neon_(               //
    float64x2_t first_latitudes, float64x2_t first_longitudes, //
    float64x2_t second_latitudes, float64x2_t second_longitudes) {

    float64x2_t const equatorial_radius = vdupq_n_f64(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    float64x2_t const polar_radius = vdupq_n_f64(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    float64x2_t const flattening = vdupq_n_f64(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    float64x2_t const convergence_threshold = vdupq_n_f64(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    float64x2_t const one = vdupq_n_f64(1.0);
    float64x2_t const two = vdupq_n_f64(2.0);
    float64x2_t const three = vdupq_n_f64(3.0);
    float64x2_t const four = vdupq_n_f64(4.0);
    float64x2_t const six = vdupq_n_f64(6.0);
    float64x2_t const sixteen = vdupq_n_f64(16.0);
    float64x2_t const epsilon = vdupq_n_f64(1e-15);

    // Longitude difference
    float64x2_t longitude_difference = vsubq_f64(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    float64x2_t one_minus_f = vsubq_f64(one, flattening);
    float64x2_t tan_first = vdivq_f64(nk_sin_f64x2_neon_(first_latitudes), nk_cos_f64x2_neon_(first_latitudes));
    float64x2_t tan_second = vdivq_f64(nk_sin_f64x2_neon_(second_latitudes), nk_cos_f64x2_neon_(second_latitudes));
    float64x2_t tan_reduced_first = vmulq_f64(one_minus_f, tan_first);
    float64x2_t tan_reduced_second = vmulq_f64(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    float64x2_t cos_reduced_first = vdivq_f64(one, vsqrtq_f64(vfmaq_f64(one, tan_reduced_first, tan_reduced_first)));
    float64x2_t sin_reduced_first = vmulq_f64(tan_reduced_first, cos_reduced_first);
    float64x2_t cos_reduced_second = vdivq_f64(one, vsqrtq_f64(vfmaq_f64(one, tan_reduced_second, tan_reduced_second)));
    float64x2_t sin_reduced_second = vmulq_f64(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    float64x2_t lambda = longitude_difference;
    float64x2_t sin_angular_distance, cos_angular_distance, angular_distance;
    float64x2_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    uint64x2_t converged_mask = vdupq_n_u64(0);
    uint64x2_t coincident_mask = vdupq_n_u64(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        uint64_t converged_bits = vgetq_lane_u64(converged_mask, 0) & vgetq_lane_u64(converged_mask, 1);
        if (converged_bits) break;

        float64x2_t sin_lambda = nk_sin_f64x2_neon_(lambda);
        float64x2_t cos_lambda = nk_cos_f64x2_neon_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        float64x2_t cross_term = vmulq_f64(cos_reduced_second, sin_lambda);
        float64x2_t mixed_term = vsubq_f64(vmulq_f64(cos_reduced_first, sin_reduced_second),
                                           vmulq_f64(vmulq_f64(sin_reduced_first, cos_reduced_second), cos_lambda));
        float64x2_t sin_angular_dist_sq = vfmaq_f64(vmulq_f64(mixed_term, mixed_term), cross_term, cross_term);
        sin_angular_distance = vsqrtq_f64(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = vcltq_f64(sin_angular_distance, epsilon);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = vfmaq_f64(vmulq_f64(sin_reduced_first, sin_reduced_second),
                                         vmulq_f64(cos_reduced_first, cos_reduced_second), cos_lambda);

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_atan2_f64x2_neon_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        // Avoid division by zero by using blending
        float64x2_t safe_sin_angular = vbslq_f64(coincident_mask, one, sin_angular_distance);
        sin_azimuth = vdivq_f64(vmulq_f64(vmulq_f64(cos_reduced_first, cos_reduced_second), sin_lambda),
                                safe_sin_angular);
        cos_squared_azimuth = vsubq_f64(one, vmulq_f64(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        uint64x2_t equatorial_mask = vcltq_f64(cos_squared_azimuth, epsilon);
        float64x2_t safe_cos_sq_azimuth = vbslq_f64(equatorial_mask, one, cos_squared_azimuth);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        float64x2_t sin_product = vmulq_f64(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = vsubq_f64(cos_angular_distance,
                                                vdivq_f64(vmulq_f64(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = vbslq_f64(equatorial_mask, vdupq_n_f64(0.0), cos_double_angular_midpoint);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        float64x2_t correction_factor = vmulq_f64(
            vdivq_f64(flattening, sixteen),
            vmulq_f64(cos_squared_azimuth, vfmaq_f64(four, flattening, vfmsq_f64(four, three, cos_squared_azimuth))));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        float64x2_t cos_2sm_sq = vmulq_f64(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        float64x2_t innermost = vfmaq_f64(vdupq_n_f64(-1.0), two, cos_2sm_sq);
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        float64x2_t middle = vfmaq_f64(cos_double_angular_midpoint, vmulq_f64(correction_factor, cos_angular_distance),
                                       innermost);
        // inner = C × sin(σ) × middle
        float64x2_t inner = vmulq_f64(vmulq_f64(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        float64x2_t lambda_new = vfmaq_f64(
            longitude_difference, vmulq_f64(vmulq_f64(vsubq_f64(one, correction_factor), flattening), sin_azimuth),
            vaddq_f64(angular_distance, inner));

        // Check convergence: |λ - λ'| < threshold
        float64x2_t lambda_diff = vsubq_f64(lambda_new, lambda);
        float64x2_t lambda_diff_abs = vabsq_f64(lambda_diff);
        uint64x2_t newly_converged = vcltq_f64(lambda_diff_abs, convergence_threshold);
        converged_mask = vorrq_u64(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        lambda = vbslq_f64(converged_mask, lambda, lambda_new);
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    float64x2_t a_sq = vmulq_f64(equatorial_radius, equatorial_radius);
    float64x2_t b_sq = vmulq_f64(polar_radius, polar_radius);
    float64x2_t u_squared = vdivq_f64(vmulq_f64(cos_squared_azimuth, vsubq_f64(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    float64x2_t series_a = vfmaq_f64(vdupq_n_f64(320.0), u_squared, vdupq_n_f64(-175.0));
    series_a = vfmaq_f64(vdupq_n_f64(-768.0), u_squared, series_a);
    series_a = vfmaq_f64(vdupq_n_f64(4096.0), u_squared, series_a);
    series_a = vfmaq_f64(one, vdivq_f64(u_squared, vdupq_n_f64(16384.0)), series_a);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    float64x2_t series_b = vfmaq_f64(vdupq_n_f64(74.0), u_squared, vdupq_n_f64(-47.0));
    series_b = vfmaq_f64(vdupq_n_f64(-128.0), u_squared, series_b);
    series_b = vfmaq_f64(vdupq_n_f64(256.0), u_squared, series_b);
    series_b = vmulq_f64(vdivq_f64(u_squared, vdupq_n_f64(1024.0)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) + B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 +
    // 4 × cos²(2σₘ))))
    float64x2_t cos_2sm_sq = vmulq_f64(cos_double_angular_midpoint, cos_double_angular_midpoint);
    float64x2_t sin_sq = vmulq_f64(sin_angular_distance, sin_angular_distance);
    float64x2_t term1 = vfmaq_f64(vdupq_n_f64(-1.0), two, cos_2sm_sq);
    term1 = vmulq_f64(cos_angular_distance, term1);
    float64x2_t term2 = vfmaq_f64(vdupq_n_f64(-3.0), four, sin_sq);
    float64x2_t term3 = vfmaq_f64(vdupq_n_f64(-3.0), four, cos_2sm_sq);
    term2 = vmulq_f64(vmulq_f64(vdivq_f64(series_b, six), cos_double_angular_midpoint), vmulq_f64(term2, term3));
    float64x2_t delta_sigma = vmulq_f64(
        series_b,
        vmulq_f64(sin_angular_distance, vaddq_f64(cos_double_angular_midpoint,
                                                  vmulq_f64(vdivq_f64(series_b, four), vsubq_f64(term1, term2)))));

    // s = b * A * (σ - Δσ)
    float64x2_t distances = vmulq_f64(vmulq_f64(polar_radius, series_a), vsubq_f64(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = vbslq_f64(coincident_mask, vdupq_n_f64(0.0), distances);

    return distances;
}

NK_PUBLIC void nk_vincenty_f64_neon(                //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        float64x2_t first_latitudes = vld1q_f64(a_lats);
        float64x2_t first_longitudes = vld1q_f64(a_lons);
        float64x2_t second_latitudes = vld1q_f64(b_lats);
        float64x2_t second_longitudes = vld1q_f64(b_lons);

        float64x2_t distances = nk_vincenty_f64x2_neon_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        vst1q_f64(results, distances);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle remaining elements with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        float64x2_t distances = nk_vincenty_f64x2_neon_(a_lat_vec.f64x2, a_lon_vec.f64x2, b_lat_vec.f64x2,
                                                        b_lon_vec.f64x2);
        result_vec.f64x2 = distances;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  NEON helper for Vincenty's geodesic distance on 4 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL float32x4_t nk_vincenty_f32x4_neon_(               //
    float32x4_t first_latitudes, float32x4_t first_longitudes, //
    float32x4_t second_latitudes, float32x4_t second_longitudes) {

    float32x4_t const equatorial_radius = vdupq_n_f32((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    float32x4_t const polar_radius = vdupq_n_f32((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    float32x4_t const flattening = vdupq_n_f32(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    float32x4_t const convergence_threshold = vdupq_n_f32(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32);
    float32x4_t const one = vdupq_n_f32(1.0f);
    float32x4_t const two = vdupq_n_f32(2.0f);
    float32x4_t const three = vdupq_n_f32(3.0f);
    float32x4_t const four = vdupq_n_f32(4.0f);
    float32x4_t const six = vdupq_n_f32(6.0f);
    float32x4_t const sixteen = vdupq_n_f32(16.0f);
    float32x4_t const epsilon = vdupq_n_f32(1e-7f);

    // Longitude difference
    float32x4_t longitude_difference = vsubq_f32(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    float32x4_t one_minus_f = vsubq_f32(one, flattening);
    float32x4_t tan_first = vdivq_f32(nk_sin_f32x4_neon_(first_latitudes), nk_cos_f32x4_neon_(first_latitudes));
    float32x4_t tan_second = vdivq_f32(nk_sin_f32x4_neon_(second_latitudes), nk_cos_f32x4_neon_(second_latitudes));
    float32x4_t tan_reduced_first = vmulq_f32(one_minus_f, tan_first);
    float32x4_t tan_reduced_second = vmulq_f32(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    float32x4_t cos_reduced_first = vdivq_f32(one, vsqrtq_f32(vfmaq_f32(one, tan_reduced_first, tan_reduced_first)));
    float32x4_t sin_reduced_first = vmulq_f32(tan_reduced_first, cos_reduced_first);
    float32x4_t cos_reduced_second = vdivq_f32(one, vsqrtq_f32(vfmaq_f32(one, tan_reduced_second, tan_reduced_second)));
    float32x4_t sin_reduced_second = vmulq_f32(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    float32x4_t lambda = longitude_difference;
    float32x4_t sin_angular_distance, cos_angular_distance, angular_distance;
    float32x4_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    uint32x4_t converged_mask = vdupq_n_u32(0);
    uint32x4_t coincident_mask = vdupq_n_u32(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged (all bits set = 0xFFFFFFFF per lane)
        uint32_t converged_bits = vminvq_u32(converged_mask);
        if (converged_bits == 0xFFFFFFFF) break;

        float32x4_t sin_lambda = nk_sin_f32x4_neon_(lambda);
        float32x4_t cos_lambda = nk_cos_f32x4_neon_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        float32x4_t cross_term = vmulq_f32(cos_reduced_second, sin_lambda);
        float32x4_t mixed_term = vsubq_f32(vmulq_f32(cos_reduced_first, sin_reduced_second),
                                           vmulq_f32(vmulq_f32(sin_reduced_first, cos_reduced_second), cos_lambda));
        float32x4_t sin_angular_dist_sq = vfmaq_f32(vmulq_f32(mixed_term, mixed_term), cross_term, cross_term);
        sin_angular_distance = vsqrtq_f32(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = vcltq_f32(sin_angular_distance, epsilon);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = vfmaq_f32(vmulq_f32(sin_reduced_first, sin_reduced_second),
                                         vmulq_f32(cos_reduced_first, cos_reduced_second), cos_lambda);

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_atan2_f32x4_neon_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        float32x4_t safe_sin_angular = vbslq_f32(coincident_mask, one, sin_angular_distance);
        sin_azimuth = vdivq_f32(vmulq_f32(vmulq_f32(cos_reduced_first, cos_reduced_second), sin_lambda),
                                safe_sin_angular);
        cos_squared_azimuth = vsubq_f32(one, vmulq_f32(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        uint32x4_t equatorial_mask = vcltq_f32(cos_squared_azimuth, epsilon);
        float32x4_t safe_cos_sq_azimuth = vbslq_f32(equatorial_mask, one, cos_squared_azimuth);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        float32x4_t sin_product = vmulq_f32(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = vsubq_f32(cos_angular_distance,
                                                vdivq_f32(vmulq_f32(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = vbslq_f32(equatorial_mask, vdupq_n_f32(0.0f), cos_double_angular_midpoint);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        float32x4_t correction_factor = vmulq_f32(
            vdivq_f32(flattening, sixteen),
            vmulq_f32(cos_squared_azimuth, vfmaq_f32(four, flattening, vfmsq_f32(four, three, cos_squared_azimuth))));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        float32x4_t cos_2sm_sq = vmulq_f32(cos_double_angular_midpoint, cos_double_angular_midpoint);
        float32x4_t innermost = vfmaq_f32(vdupq_n_f32(-1.0f), two, cos_2sm_sq);
        float32x4_t middle = vfmaq_f32(cos_double_angular_midpoint, vmulq_f32(correction_factor, cos_angular_distance),
                                       innermost);
        float32x4_t inner = vmulq_f32(vmulq_f32(correction_factor, sin_angular_distance), middle);

        float32x4_t lambda_new = vfmaq_f32(
            longitude_difference, vmulq_f32(vmulq_f32(vsubq_f32(one, correction_factor), flattening), sin_azimuth),
            vaddq_f32(angular_distance, inner));

        // Check convergence: |λ - λ'| < threshold
        float32x4_t lambda_diff = vsubq_f32(lambda_new, lambda);
        float32x4_t lambda_diff_abs = vabsq_f32(lambda_diff);
        uint32x4_t newly_converged = vcltq_f32(lambda_diff_abs, convergence_threshold);
        converged_mask = vorrq_u32(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        lambda = vbslq_f32(converged_mask, lambda, lambda_new);
    }

    // Final distance calculation
    float32x4_t a_sq = vmulq_f32(equatorial_radius, equatorial_radius);
    float32x4_t b_sq = vmulq_f32(polar_radius, polar_radius);
    float32x4_t u_squared = vdivq_f32(vmulq_f32(cos_squared_azimuth, vsubq_f32(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    float32x4_t series_a = vfmaq_f32(vdupq_n_f32(320.0f), u_squared, vdupq_n_f32(-175.0f));
    series_a = vfmaq_f32(vdupq_n_f32(-768.0f), u_squared, series_a);
    series_a = vfmaq_f32(vdupq_n_f32(4096.0f), u_squared, series_a);
    series_a = vfmaq_f32(one, vdivq_f32(u_squared, vdupq_n_f32(16384.0f)), series_a);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    float32x4_t series_b = vfmaq_f32(vdupq_n_f32(74.0f), u_squared, vdupq_n_f32(-47.0f));
    series_b = vfmaq_f32(vdupq_n_f32(-128.0f), u_squared, series_b);
    series_b = vfmaq_f32(vdupq_n_f32(256.0f), u_squared, series_b);
    series_b = vmulq_f32(vdivq_f32(u_squared, vdupq_n_f32(1024.0f)), series_b);

    // Δσ calculation
    float32x4_t cos_2sm_sq = vmulq_f32(cos_double_angular_midpoint, cos_double_angular_midpoint);
    float32x4_t sin_sq = vmulq_f32(sin_angular_distance, sin_angular_distance);
    float32x4_t term1 = vfmaq_f32(vdupq_n_f32(-1.0f), two, cos_2sm_sq);
    term1 = vmulq_f32(cos_angular_distance, term1);
    float32x4_t term2 = vfmaq_f32(vdupq_n_f32(-3.0f), four, sin_sq);
    float32x4_t term3 = vfmaq_f32(vdupq_n_f32(-3.0f), four, cos_2sm_sq);
    term2 = vmulq_f32(vmulq_f32(vdivq_f32(series_b, six), cos_double_angular_midpoint), vmulq_f32(term2, term3));
    float32x4_t delta_sigma = vmulq_f32(
        series_b,
        vmulq_f32(sin_angular_distance, vaddq_f32(cos_double_angular_midpoint,
                                                  vmulq_f32(vdivq_f32(series_b, four), vsubq_f32(term1, term2)))));

    // s = b * A * (σ - Δσ)
    float32x4_t distances = vmulq_f32(vmulq_f32(polar_radius, series_a), vsubq_f32(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = vbslq_f32(coincident_mask, vdupq_n_f32(0.0f), distances);

    return distances;
}

NK_PUBLIC void nk_vincenty_f32_neon(                //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        float32x4_t first_latitudes = vld1q_f32(a_lats);
        float32x4_t first_longitudes = vld1q_f32(a_lons);
        float32x4_t second_latitudes = vld1q_f32(b_lats);
        float32x4_t second_longitudes = vld1q_f32(b_lons);

        float32x4_t distances = nk_vincenty_f32x4_neon_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        vst1q_f32(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        float32x4_t distances = nk_vincenty_f32x4_neon_(a_lat_vec.f32x4, a_lon_vec.f32x4, b_lat_vec.f32x4,
                                                        b_lon_vec.f32x4);
        result_vec.f32x4 = distances;
        nk_partial_store_b32x4_serial_(&result_vec, results, n);
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

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_GEOSPATIAL_NEON_H
