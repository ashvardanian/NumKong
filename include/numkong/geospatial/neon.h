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
 *      Intrinsic   Instruction    M1 Firestorm  Graviton 3   Graviton 4
 *      vfmaq_f32   FMLA.S (vec)   4cy @ V0123   4cy @ V0123  4cy @ V0123
 *      vfmaq_f64   FMLA.D (vec)   4cy @ V0123   4cy @ V0123  4cy @ V0123
 *      vsqrtq_f32  FSQRT.S (vec)  10cy @ V02    10cy @ V02   9cy @ V02
 *      vsqrtq_f64  FSQRT.D (vec)  13cy @ V02    16cy @ V02   16cy @ V02
 */
#ifndef NK_GEOSPATIAL_NEON_H
#define NK_GEOSPATIAL_NEON_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEON

#include "numkong/types.h"
#include "numkong/trigonometry/neon.h" // `nk_sin_f64x2_neon_`, `nk_cos_f64x2_neon_`, `nk_atan2_f64x2_neon_`

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

NK_INTERNAL float64x2_t nk_haversine_f64x2_neon_(                          //
    float64x2_t first_latitudes_f64x2, float64x2_t first_longitudes_f64x2, //
    float64x2_t second_latitudes_f64x2, float64x2_t second_longitudes_f64x2) {

    float64x2_t const earth_radius_f64x2 = vdupq_n_f64(NK_EARTH_MEDIATORIAL_RADIUS);
    float64x2_t const half_f64x2 = vdupq_n_f64(0.5);
    float64x2_t const one_f64x2 = vdupq_n_f64(1.0);
    float64x2_t const two_f64x2 = vdupq_n_f64(2.0);

    float64x2_t latitude_delta_f64x2 = vsubq_f64(second_latitudes_f64x2, first_latitudes_f64x2);
    float64x2_t longitude_delta_f64x2 = vsubq_f64(second_longitudes_f64x2, first_longitudes_f64x2);

    // Haversine terms: sin²(Δ/2)
    float64x2_t latitude_delta_half_f64x2 = vmulq_f64(latitude_delta_f64x2, half_f64x2);
    float64x2_t longitude_delta_half_f64x2 = vmulq_f64(longitude_delta_f64x2, half_f64x2);
    float64x2_t sin_latitude_delta_half_f64x2 = nk_sin_f64x2_neon_(latitude_delta_half_f64x2);
    float64x2_t sin_longitude_delta_half_f64x2 = nk_sin_f64x2_neon_(longitude_delta_half_f64x2);
    float64x2_t sin_sq_latitude_delta_half_f64x2 = vmulq_f64(sin_latitude_delta_half_f64x2,
                                                             sin_latitude_delta_half_f64x2);
    float64x2_t sin_sq_longitude_delta_half_f64x2 = vmulq_f64(sin_longitude_delta_half_f64x2,
                                                              sin_longitude_delta_half_f64x2);

    // Latitude cosine product
    float64x2_t cos_first_latitude_f64x2 = nk_cos_f64x2_neon_(first_latitudes_f64x2);
    float64x2_t cos_second_latitude_f64x2 = nk_cos_f64x2_neon_(second_latitudes_f64x2);
    float64x2_t cos_latitude_product_f64x2 = vmulq_f64(cos_first_latitude_f64x2, cos_second_latitude_f64x2);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    float64x2_t haversine_term_f64x2 = vaddq_f64(
        sin_sq_latitude_delta_half_f64x2, vmulq_f64(cos_latitude_product_f64x2, sin_sq_longitude_delta_half_f64x2));
    // Clamp haversine_term_f64x2 to [0, 1] to prevent NaN from sqrt of negative values
    float64x2_t zero_f64x2 = vdupq_n_f64(0.0);
    haversine_term_f64x2 = vmaxq_f64(zero_f64x2, vminq_f64(one_f64x2, haversine_term_f64x2));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    float64x2_t sqrt_haversine_f64x2 = vsqrtq_f64(haversine_term_f64x2);
    float64x2_t sqrt_complement_f64x2 = vsqrtq_f64(vsubq_f64(one_f64x2, haversine_term_f64x2));
    float64x2_t central_angle_f64x2 = vmulq_f64(two_f64x2,
                                                nk_atan2_f64x2_neon_(sqrt_haversine_f64x2, sqrt_complement_f64x2));

    return vmulq_f64(earth_radius_f64x2, central_angle_f64x2);
}

NK_PUBLIC void nk_haversine_f64_neon(               //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        float64x2_t first_latitudes_f64x2 = vld1q_f64(a_lats);
        float64x2_t first_longitudes_f64x2 = vld1q_f64(a_lons);
        float64x2_t second_latitudes_f64x2 = vld1q_f64(b_lats);
        float64x2_t second_longitudes_f64x2 = vld1q_f64(b_lons);

        float64x2_t distances_f64x2 = nk_haversine_f64x2_neon_(first_latitudes_f64x2, first_longitudes_f64x2,
                                                               second_latitudes_f64x2, second_longitudes_f64x2);
        vst1q_f64(results, distances_f64x2);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle tail with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        float64x2_t distances_f64x2 = nk_haversine_f64x2_neon_(a_lat_vec.f64x2, a_lon_vec.f64x2, b_lat_vec.f64x2,
                                                               b_lon_vec.f64x2);
        result_vec.f64x2 = distances_f64x2;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

NK_INTERNAL float32x4_t nk_haversine_f32x4_neon_(                          //
    float32x4_t first_latitudes_f32x4, float32x4_t first_longitudes_f32x4, //
    float32x4_t second_latitudes_f32x4, float32x4_t second_longitudes_f32x4) {

    float32x4_t const earth_radius_f32x4 = vdupq_n_f32((float)NK_EARTH_MEDIATORIAL_RADIUS);
    float32x4_t const half_f32x4 = vdupq_n_f32(0.5f);
    float32x4_t const one_f32x4 = vdupq_n_f32(1.0f);
    float32x4_t const two_f32x4 = vdupq_n_f32(2.0f);

    float32x4_t latitude_delta_f32x4 = vsubq_f32(second_latitudes_f32x4, first_latitudes_f32x4);
    float32x4_t longitude_delta_f32x4 = vsubq_f32(second_longitudes_f32x4, first_longitudes_f32x4);

    // Haversine terms: sin²(Δ/2)
    float32x4_t latitude_delta_half_f32x4 = vmulq_f32(latitude_delta_f32x4, half_f32x4);
    float32x4_t longitude_delta_half_f32x4 = vmulq_f32(longitude_delta_f32x4, half_f32x4);
    float32x4_t sin_latitude_delta_half_f32x4 = nk_sin_f32x4_neon_(latitude_delta_half_f32x4);
    float32x4_t sin_longitude_delta_half_f32x4 = nk_sin_f32x4_neon_(longitude_delta_half_f32x4);
    float32x4_t sin_sq_latitude_delta_half_f32x4 = vmulq_f32(sin_latitude_delta_half_f32x4,
                                                             sin_latitude_delta_half_f32x4);
    float32x4_t sin_sq_longitude_delta_half_f32x4 = vmulq_f32(sin_longitude_delta_half_f32x4,
                                                              sin_longitude_delta_half_f32x4);

    // Latitude cosine product
    float32x4_t cos_first_latitude_f32x4 = nk_cos_f32x4_neon_(first_latitudes_f32x4);
    float32x4_t cos_second_latitude_f32x4 = nk_cos_f32x4_neon_(second_latitudes_f32x4);
    float32x4_t cos_latitude_product_f32x4 = vmulq_f32(cos_first_latitude_f32x4, cos_second_latitude_f32x4);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    float32x4_t haversine_term_f32x4 = vaddq_f32(
        sin_sq_latitude_delta_half_f32x4, vmulq_f32(cos_latitude_product_f32x4, sin_sq_longitude_delta_half_f32x4));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    float32x4_t zero_f32x4 = vdupq_n_f32(0.0f);
    haversine_term_f32x4 = vmaxq_f32(zero_f32x4, vminq_f32(one_f32x4, haversine_term_f32x4));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    float32x4_t sqrt_haversine_f32x4 = vsqrtq_f32(haversine_term_f32x4);
    float32x4_t sqrt_complement_f32x4 = vsqrtq_f32(vsubq_f32(one_f32x4, haversine_term_f32x4));
    float32x4_t central_angle_f32x4 = vmulq_f32(two_f32x4,
                                                nk_atan2_f32x4_neon_(sqrt_haversine_f32x4, sqrt_complement_f32x4));

    return vmulq_f32(earth_radius_f32x4, central_angle_f32x4);
}

NK_PUBLIC void nk_haversine_f32_neon(               //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        float32x4_t first_latitudes_f32x4 = vld1q_f32(a_lats);
        float32x4_t first_longitudes_f32x4 = vld1q_f32(a_lons);
        float32x4_t second_latitudes_f32x4 = vld1q_f32(b_lats);
        float32x4_t second_longitudes_f32x4 = vld1q_f32(b_lons);

        float32x4_t distances_f32x4 = nk_haversine_f32x4_neon_(first_latitudes_f32x4, first_longitudes_f32x4,
                                                               second_latitudes_f32x4, second_longitudes_f32x4);
        vst1q_f32(results, distances_f32x4);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle tail with partial loads (n can be 0-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        float32x4_t distances_f32x4 = nk_haversine_f32x4_neon_(a_lat_vec.f32x4, a_lon_vec.f32x4, b_lat_vec.f32x4,
                                                               b_lon_vec.f32x4);
        result_vec.f32x4 = distances_f32x4;
        nk_partial_store_b32x4_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  NEON helper for Vincenty's geodesic distance on 2 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL float64x2_t nk_vincenty_f64x2_neon_(                           //
    float64x2_t first_latitudes_f64x2, float64x2_t first_longitudes_f64x2, //
    float64x2_t second_latitudes_f64x2, float64x2_t second_longitudes_f64x2) {

    float64x2_t const equatorial_radius_f64x2 = vdupq_n_f64(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    float64x2_t const polar_radius_f64x2 = vdupq_n_f64(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    float64x2_t const flattening_f64x2 = vdupq_n_f64(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    float64x2_t const convergence_threshold_f64x2 = vdupq_n_f64(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    float64x2_t const one_f64x2 = vdupq_n_f64(1.0);
    float64x2_t const two_f64x2 = vdupq_n_f64(2.0);
    float64x2_t const three_f64x2 = vdupq_n_f64(3.0);
    float64x2_t const four_f64x2 = vdupq_n_f64(4.0);
    float64x2_t const six_f64x2 = vdupq_n_f64(6.0);
    float64x2_t const sixteen_f64x2 = vdupq_n_f64(16.0);
    float64x2_t const epsilon_f64x2 = vdupq_n_f64(1e-15);

    // Longitude difference
    float64x2_t longitude_diff_f64x2 = vsubq_f64(second_longitudes_f64x2, first_longitudes_f64x2);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    float64x2_t one_minus_f64x2 = vsubq_f64(one_f64x2, flattening_f64x2);
    float64x2_t tan_first_f64x2 = vdivq_f64(nk_sin_f64x2_neon_(first_latitudes_f64x2),
                                            nk_cos_f64x2_neon_(first_latitudes_f64x2));
    float64x2_t tan_second_f64x2 = vdivq_f64(nk_sin_f64x2_neon_(second_latitudes_f64x2),
                                             nk_cos_f64x2_neon_(second_latitudes_f64x2));
    float64x2_t tan_reduced_first_f64x2 = vmulq_f64(one_minus_f64x2, tan_first_f64x2);
    float64x2_t tan_reduced_second_f64x2 = vmulq_f64(one_minus_f64x2, tan_second_f64x2);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    float64x2_t cos_reduced_first_f64x2 = vdivq_f64(
        one_f64x2, vsqrtq_f64(vfmaq_f64(one_f64x2, tan_reduced_first_f64x2, tan_reduced_first_f64x2)));
    float64x2_t sin_reduced_first_f64x2 = vmulq_f64(tan_reduced_first_f64x2, cos_reduced_first_f64x2);
    float64x2_t cos_reduced_second_f64x2 = vdivq_f64(
        one_f64x2, vsqrtq_f64(vfmaq_f64(one_f64x2, tan_reduced_second_f64x2, tan_reduced_second_f64x2)));
    float64x2_t sin_reduced_second_f64x2 = vmulq_f64(tan_reduced_second_f64x2, cos_reduced_second_f64x2);

    // Initialize lambda_f64x2 and tracking variables
    float64x2_t lambda_f64x2 = longitude_diff_f64x2;
    float64x2_t sin_angular_distance_f64x2, cos_angular_distance_f64x2, angular_distance_f64x2;
    float64x2_t sin_azimuth_f64x2, cos_sq_azimuth_f64x2, cos_double_angular_midpoint_f64x2;

    // Track convergence and coincident points using masks
    uint64x2_t converged_mask_u64x2 = vdupq_n_u64(0);
    uint64x2_t coincident_mask_u64x2 = vdupq_n_u64(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        nk_u64_t converged_bits = vgetq_lane_u64(converged_mask_u64x2, 0) & vgetq_lane_u64(converged_mask_u64x2, 1);
        if (converged_bits) break;

        float64x2_t sin_lambda_f64x2 = nk_sin_f64x2_neon_(lambda_f64x2);
        float64x2_t cos_lambda_f64x2 = nk_cos_f64x2_neon_(lambda_f64x2);

        // sin²(angular_distance_f64x2) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        float64x2_t cross_term_f64x2 = vmulq_f64(cos_reduced_second_f64x2, sin_lambda_f64x2);
        float64x2_t mixed_term_f64x2 = vsubq_f64(
            vmulq_f64(cos_reduced_first_f64x2, sin_reduced_second_f64x2),
            vmulq_f64(vmulq_f64(sin_reduced_first_f64x2, cos_reduced_second_f64x2), cos_lambda_f64x2));
        float64x2_t sin_angular_dist_sq_f64x2 = vfmaq_f64(vmulq_f64(mixed_term_f64x2, mixed_term_f64x2),
                                                          cross_term_f64x2, cross_term_f64x2);
        sin_angular_distance_f64x2 = vsqrtq_f64(sin_angular_dist_sq_f64x2);

        // Check for coincident points (sin_angular_distance_f64x2 ≈ 0)
        coincident_mask_u64x2 = vcltq_f64(sin_angular_distance_f64x2, epsilon_f64x2);

        // cos(angular_distance_f64x2) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f64x2 = vfmaq_f64(vmulq_f64(sin_reduced_first_f64x2, sin_reduced_second_f64x2),
                                               vmulq_f64(cos_reduced_first_f64x2, cos_reduced_second_f64x2),
                                               cos_lambda_f64x2);

        // angular_distance_f64x2 = atan2(sin, cos)
        angular_distance_f64x2 = nk_atan2_f64x2_neon_(sin_angular_distance_f64x2, cos_angular_distance_f64x2);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance_f64x2)
        // Avoid division by zero by using blending
        float64x2_t safe_sin_angular_f64x2 = vbslq_f64(coincident_mask_u64x2, one_f64x2, sin_angular_distance_f64x2);
        sin_azimuth_f64x2 = vdivq_f64(
            vmulq_f64(vmulq_f64(cos_reduced_first_f64x2, cos_reduced_second_f64x2), sin_lambda_f64x2),
            safe_sin_angular_f64x2);
        cos_sq_azimuth_f64x2 = vsubq_f64(one_f64x2, vmulq_f64(sin_azimuth_f64x2, sin_azimuth_f64x2));

        // Handle equatorial case: cos²α ≈ 0
        uint64x2_t equatorial_mask_u64x2 = vcltq_f64(cos_sq_azimuth_f64x2, epsilon_f64x2);
        float64x2_t safe_cos_sq_azimuth_f64x2 = vbslq_f64(equatorial_mask_u64x2, one_f64x2, cos_sq_azimuth_f64x2);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        float64x2_t sin_product_f64x2 = vmulq_f64(sin_reduced_first_f64x2, sin_reduced_second_f64x2);
        cos_double_angular_midpoint_f64x2 = vsubq_f64(
            cos_angular_distance_f64x2, vdivq_f64(vmulq_f64(two_f64x2, sin_product_f64x2), safe_cos_sq_azimuth_f64x2));
        cos_double_angular_midpoint_f64x2 = vbslq_f64(equatorial_mask_u64x2, vdupq_n_f64(0.0),
                                                      cos_double_angular_midpoint_f64x2);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        float64x2_t correction_factor_f64x2 = vmulq_f64(
            vdivq_f64(flattening_f64x2, sixteen_f64x2),
            vmulq_f64(cos_sq_azimuth_f64x2, vfmaq_f64(four_f64x2, flattening_f64x2,
                                                      vfmsq_f64(four_f64x2, three_f64x2, cos_sq_azimuth_f64x2))));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        float64x2_t cos_2sm_sq_f64x2 = vmulq_f64(cos_double_angular_midpoint_f64x2, cos_double_angular_midpoint_f64x2);
        // innermost_f64x2 = -1 + 2 × cos²(2σₘ)
        float64x2_t innermost_f64x2 = vfmaq_f64(vdupq_n_f64(-1.0), two_f64x2, cos_2sm_sq_f64x2);
        // middle_f64x2 = cos(2σₘ) + C × cos(σ) × innermost_f64x2
        float64x2_t middle_f64x2 = vfmaq_f64(cos_double_angular_midpoint_f64x2,
                                             vmulq_f64(correction_factor_f64x2, cos_angular_distance_f64x2),
                                             innermost_f64x2);
        // inner_f64x2 = C × sin(σ) × middle_f64x2
        float64x2_t inner_f64x2 = vmulq_f64(vmulq_f64(correction_factor_f64x2, sin_angular_distance_f64x2),
                                            middle_f64x2);

        // λ' = L + (1-C) * f * sin_α * (σ + inner_f64x2)
        float64x2_t lambda_new_f64x2 = vfmaq_f64(
            longitude_diff_f64x2,
            vmulq_f64(vmulq_f64(vsubq_f64(one_f64x2, correction_factor_f64x2), flattening_f64x2), sin_azimuth_f64x2),
            vaddq_f64(angular_distance_f64x2, inner_f64x2));

        // Check convergence: |λ - λ'| < threshold
        float64x2_t lambda_diff_f64x2 = vsubq_f64(lambda_new_f64x2, lambda_f64x2);
        float64x2_t lambda_diff_abs_f64x2 = vabsq_f64(lambda_diff_f64x2);
        uint64x2_t newly_converged_u64x2 = vcltq_f64(lambda_diff_abs_f64x2, convergence_threshold_f64x2);
        converged_mask_u64x2 = vorrq_u64(converged_mask_u64x2, newly_converged_u64x2);

        // Only update lambda_f64x2 for non-converged lanes
        lambda_f64x2 = vbslq_f64(converged_mask_u64x2, lambda_f64x2, lambda_new_f64x2);
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    float64x2_t a_sq_f64x2 = vmulq_f64(equatorial_radius_f64x2, equatorial_radius_f64x2);
    float64x2_t b_sq_f64x2 = vmulq_f64(polar_radius_f64x2, polar_radius_f64x2);
    float64x2_t u_sq_f64x2 = vdivq_f64(vmulq_f64(cos_sq_azimuth_f64x2, vsubq_f64(a_sq_f64x2, b_sq_f64x2)), b_sq_f64x2);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    float64x2_t series_a_f64x2 = vfmaq_f64(vdupq_n_f64(320.0), u_sq_f64x2, vdupq_n_f64(-175.0));
    series_a_f64x2 = vfmaq_f64(vdupq_n_f64(-768.0), u_sq_f64x2, series_a_f64x2);
    series_a_f64x2 = vfmaq_f64(vdupq_n_f64(4096.0), u_sq_f64x2, series_a_f64x2);
    series_a_f64x2 = vfmaq_f64(one_f64x2, vdivq_f64(u_sq_f64x2, vdupq_n_f64(16384.0)), series_a_f64x2);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    float64x2_t series_b_f64x2 = vfmaq_f64(vdupq_n_f64(74.0), u_sq_f64x2, vdupq_n_f64(-47.0));
    series_b_f64x2 = vfmaq_f64(vdupq_n_f64(-128.0), u_sq_f64x2, series_b_f64x2);
    series_b_f64x2 = vfmaq_f64(vdupq_n_f64(256.0), u_sq_f64x2, series_b_f64x2);
    series_b_f64x2 = vmulq_f64(vdivq_f64(u_sq_f64x2, vdupq_n_f64(1024.0)), series_b_f64x2);

    // Δσ = B × sin(σ) × (cos(2σₘ) + B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 +
    // 4 × cos²(2σₘ))))
    float64x2_t cos_2sm_sq_f64x2 = vmulq_f64(cos_double_angular_midpoint_f64x2, cos_double_angular_midpoint_f64x2);
    float64x2_t sin_sq_f64x2 = vmulq_f64(sin_angular_distance_f64x2, sin_angular_distance_f64x2);
    float64x2_t term1_f64x2 = vfmaq_f64(vdupq_n_f64(-1.0), two_f64x2, cos_2sm_sq_f64x2);
    term1_f64x2 = vmulq_f64(cos_angular_distance_f64x2, term1_f64x2);
    float64x2_t term2_f64x2 = vfmaq_f64(vdupq_n_f64(-3.0), four_f64x2, sin_sq_f64x2);
    float64x2_t term3_f64x2 = vfmaq_f64(vdupq_n_f64(-3.0), four_f64x2, cos_2sm_sq_f64x2);
    term2_f64x2 = vmulq_f64(vmulq_f64(vdivq_f64(series_b_f64x2, six_f64x2), cos_double_angular_midpoint_f64x2),
                            vmulq_f64(term2_f64x2, term3_f64x2));
    float64x2_t delta_sigma_f64x2 = vmulq_f64(
        series_b_f64x2,
        vmulq_f64(sin_angular_distance_f64x2,
                  vaddq_f64(cos_double_angular_midpoint_f64x2,
                            vmulq_f64(vdivq_f64(series_b_f64x2, four_f64x2), vsubq_f64(term1_f64x2, term2_f64x2)))));

    // s = b * A * (σ - Δσ)
    float64x2_t distances_f64x2 = vmulq_f64(vmulq_f64(polar_radius_f64x2, series_a_f64x2),
                                            vsubq_f64(angular_distance_f64x2, delta_sigma_f64x2));

    // Set coincident points to zero
    distances_f64x2 = vbslq_f64(coincident_mask_u64x2, vdupq_n_f64(0.0), distances_f64x2);

    return distances_f64x2;
}

NK_PUBLIC void nk_vincenty_f64_neon(                //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        float64x2_t first_latitudes_f64x2 = vld1q_f64(a_lats);
        float64x2_t first_longitudes_f64x2 = vld1q_f64(a_lons);
        float64x2_t second_latitudes_f64x2 = vld1q_f64(b_lats);
        float64x2_t second_longitudes_f64x2 = vld1q_f64(b_lons);

        float64x2_t distances_f64x2 = nk_vincenty_f64x2_neon_(first_latitudes_f64x2, first_longitudes_f64x2,
                                                              second_latitudes_f64x2, second_longitudes_f64x2);
        vst1q_f64(results, distances_f64x2);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle remaining elements with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        float64x2_t distances_f64x2 = nk_vincenty_f64x2_neon_(a_lat_vec.f64x2, a_lon_vec.f64x2, b_lat_vec.f64x2,
                                                              b_lon_vec.f64x2);
        result_vec.f64x2 = distances_f64x2;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  NEON helper for Vincenty's geodesic distance on 4 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL float32x4_t nk_vincenty_f32x4_neon_(                           //
    float32x4_t first_latitudes_f32x4, float32x4_t first_longitudes_f32x4, //
    float32x4_t second_latitudes_f32x4, float32x4_t second_longitudes_f32x4) {

    float32x4_t const equatorial_radius_f32x4 = vdupq_n_f32((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    float32x4_t const polar_radius_f32x4 = vdupq_n_f32((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    float32x4_t const flattening_f32x4 = vdupq_n_f32(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    float32x4_t const convergence_threshold_f32x4 = vdupq_n_f32(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32);
    float32x4_t const one_f32x4 = vdupq_n_f32(1.0f);
    float32x4_t const two_f32x4 = vdupq_n_f32(2.0f);
    float32x4_t const three_f32x4 = vdupq_n_f32(3.0f);
    float32x4_t const four_f32x4 = vdupq_n_f32(4.0f);
    float32x4_t const six_f32x4 = vdupq_n_f32(6.0f);
    float32x4_t const sixteen_f32x4 = vdupq_n_f32(16.0f);
    float32x4_t const epsilon_f32x4 = vdupq_n_f32(1e-7f);

    // Longitude difference
    float32x4_t longitude_diff_f32x4 = vsubq_f32(second_longitudes_f32x4, first_longitudes_f32x4);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    float32x4_t one_minus_f32x4 = vsubq_f32(one_f32x4, flattening_f32x4);
    float32x4_t tan_first_f32x4 = vdivq_f32(nk_sin_f32x4_neon_(first_latitudes_f32x4),
                                            nk_cos_f32x4_neon_(first_latitudes_f32x4));
    float32x4_t tan_second_f32x4 = vdivq_f32(nk_sin_f32x4_neon_(second_latitudes_f32x4),
                                             nk_cos_f32x4_neon_(second_latitudes_f32x4));
    float32x4_t tan_reduced_first_f32x4 = vmulq_f32(one_minus_f32x4, tan_first_f32x4);
    float32x4_t tan_reduced_second_f32x4 = vmulq_f32(one_minus_f32x4, tan_second_f32x4);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    float32x4_t cos_reduced_first_f32x4 = vdivq_f32(
        one_f32x4, vsqrtq_f32(vfmaq_f32(one_f32x4, tan_reduced_first_f32x4, tan_reduced_first_f32x4)));
    float32x4_t sin_reduced_first_f32x4 = vmulq_f32(tan_reduced_first_f32x4, cos_reduced_first_f32x4);
    float32x4_t cos_reduced_second_f32x4 = vdivq_f32(
        one_f32x4, vsqrtq_f32(vfmaq_f32(one_f32x4, tan_reduced_second_f32x4, tan_reduced_second_f32x4)));
    float32x4_t sin_reduced_second_f32x4 = vmulq_f32(tan_reduced_second_f32x4, cos_reduced_second_f32x4);

    // Initialize lambda_f32x4 and tracking variables
    float32x4_t lambda_f32x4 = longitude_diff_f32x4;
    float32x4_t sin_angular_distance_f32x4, cos_angular_distance_f32x4, angular_distance_f32x4;
    float32x4_t sin_azimuth_f32x4, cos_sq_azimuth_f32x4, cos_double_angular_midpoint_f32x4;

    // Track convergence and coincident points using masks
    uint32x4_t converged_mask_u32x4 = vdupq_n_u32(0);
    uint32x4_t coincident_mask_u32x4 = vdupq_n_u32(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged (all bits set = 0xFFFFFFFF per lane)
        nk_u32_t converged_bits = vminvq_u32(converged_mask_u32x4);
        if (converged_bits == 0xFFFFFFFF) break;

        float32x4_t sin_lambda_f32x4 = nk_sin_f32x4_neon_(lambda_f32x4);
        float32x4_t cos_lambda_f32x4 = nk_cos_f32x4_neon_(lambda_f32x4);

        // sin²(angular_distance_f32x4) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        float32x4_t cross_term_f32x4 = vmulq_f32(cos_reduced_second_f32x4, sin_lambda_f32x4);
        float32x4_t mixed_term_f32x4 = vsubq_f32(
            vmulq_f32(cos_reduced_first_f32x4, sin_reduced_second_f32x4),
            vmulq_f32(vmulq_f32(sin_reduced_first_f32x4, cos_reduced_second_f32x4), cos_lambda_f32x4));
        float32x4_t sin_angular_dist_sq_f32x4 = vfmaq_f32(vmulq_f32(mixed_term_f32x4, mixed_term_f32x4),
                                                          cross_term_f32x4, cross_term_f32x4);
        sin_angular_distance_f32x4 = vsqrtq_f32(sin_angular_dist_sq_f32x4);

        // Check for coincident points (sin_angular_distance_f32x4 ≈ 0)
        coincident_mask_u32x4 = vcltq_f32(sin_angular_distance_f32x4, epsilon_f32x4);

        // cos(angular_distance_f32x4) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f32x4 = vfmaq_f32(vmulq_f32(sin_reduced_first_f32x4, sin_reduced_second_f32x4),
                                               vmulq_f32(cos_reduced_first_f32x4, cos_reduced_second_f32x4),
                                               cos_lambda_f32x4);

        // angular_distance_f32x4 = atan2(sin, cos)
        angular_distance_f32x4 = nk_atan2_f32x4_neon_(sin_angular_distance_f32x4, cos_angular_distance_f32x4);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance_f32x4)
        float32x4_t safe_sin_angular_f32x4 = vbslq_f32(coincident_mask_u32x4, one_f32x4, sin_angular_distance_f32x4);
        sin_azimuth_f32x4 = vdivq_f32(
            vmulq_f32(vmulq_f32(cos_reduced_first_f32x4, cos_reduced_second_f32x4), sin_lambda_f32x4),
            safe_sin_angular_f32x4);
        cos_sq_azimuth_f32x4 = vsubq_f32(one_f32x4, vmulq_f32(sin_azimuth_f32x4, sin_azimuth_f32x4));

        // Handle equatorial case: cos²α ≈ 0
        uint32x4_t equatorial_mask_u32x4 = vcltq_f32(cos_sq_azimuth_f32x4, epsilon_f32x4);
        float32x4_t safe_cos_sq_azimuth_f32x4 = vbslq_f32(equatorial_mask_u32x4, one_f32x4, cos_sq_azimuth_f32x4);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        float32x4_t sin_product_f32x4 = vmulq_f32(sin_reduced_first_f32x4, sin_reduced_second_f32x4);
        cos_double_angular_midpoint_f32x4 = vsubq_f32(
            cos_angular_distance_f32x4, vdivq_f32(vmulq_f32(two_f32x4, sin_product_f32x4), safe_cos_sq_azimuth_f32x4));
        cos_double_angular_midpoint_f32x4 = vbslq_f32(equatorial_mask_u32x4, vdupq_n_f32(0.0f),
                                                      cos_double_angular_midpoint_f32x4);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        float32x4_t correction_factor_f32x4 = vmulq_f32(
            vdivq_f32(flattening_f32x4, sixteen_f32x4),
            vmulq_f32(cos_sq_azimuth_f32x4, vfmaq_f32(four_f32x4, flattening_f32x4,
                                                      vfmsq_f32(four_f32x4, three_f32x4, cos_sq_azimuth_f32x4))));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        float32x4_t cos_2sm_sq_f32x4 = vmulq_f32(cos_double_angular_midpoint_f32x4, cos_double_angular_midpoint_f32x4);
        float32x4_t innermost_f32x4 = vfmaq_f32(vdupq_n_f32(-1.0f), two_f32x4, cos_2sm_sq_f32x4);
        float32x4_t middle_f32x4 = vfmaq_f32(cos_double_angular_midpoint_f32x4,
                                             vmulq_f32(correction_factor_f32x4, cos_angular_distance_f32x4),
                                             innermost_f32x4);
        float32x4_t inner_f32x4 = vmulq_f32(vmulq_f32(correction_factor_f32x4, sin_angular_distance_f32x4),
                                            middle_f32x4);

        float32x4_t lambda_new_f32x4 = vfmaq_f32(
            longitude_diff_f32x4,
            vmulq_f32(vmulq_f32(vsubq_f32(one_f32x4, correction_factor_f32x4), flattening_f32x4), sin_azimuth_f32x4),
            vaddq_f32(angular_distance_f32x4, inner_f32x4));

        // Check convergence: |λ - λ'| < threshold
        float32x4_t lambda_diff_f32x4 = vsubq_f32(lambda_new_f32x4, lambda_f32x4);
        float32x4_t lambda_diff_abs_f32x4 = vabsq_f32(lambda_diff_f32x4);
        uint32x4_t newly_converged_u32x4 = vcltq_f32(lambda_diff_abs_f32x4, convergence_threshold_f32x4);
        converged_mask_u32x4 = vorrq_u32(converged_mask_u32x4, newly_converged_u32x4);

        // Only update lambda_f32x4 for non-converged lanes
        lambda_f32x4 = vbslq_f32(converged_mask_u32x4, lambda_f32x4, lambda_new_f32x4);
    }

    // Final distance calculation
    float32x4_t a_sq_f32x4 = vmulq_f32(equatorial_radius_f32x4, equatorial_radius_f32x4);
    float32x4_t b_sq_f32x4 = vmulq_f32(polar_radius_f32x4, polar_radius_f32x4);
    float32x4_t u_sq_f32x4 = vdivq_f32(vmulq_f32(cos_sq_azimuth_f32x4, vsubq_f32(a_sq_f32x4, b_sq_f32x4)), b_sq_f32x4);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    float32x4_t series_a_f32x4 = vfmaq_f32(vdupq_n_f32(320.0f), u_sq_f32x4, vdupq_n_f32(-175.0f));
    series_a_f32x4 = vfmaq_f32(vdupq_n_f32(-768.0f), u_sq_f32x4, series_a_f32x4);
    series_a_f32x4 = vfmaq_f32(vdupq_n_f32(4096.0f), u_sq_f32x4, series_a_f32x4);
    series_a_f32x4 = vfmaq_f32(one_f32x4, vdivq_f32(u_sq_f32x4, vdupq_n_f32(16384.0f)), series_a_f32x4);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    float32x4_t series_b_f32x4 = vfmaq_f32(vdupq_n_f32(74.0f), u_sq_f32x4, vdupq_n_f32(-47.0f));
    series_b_f32x4 = vfmaq_f32(vdupq_n_f32(-128.0f), u_sq_f32x4, series_b_f32x4);
    series_b_f32x4 = vfmaq_f32(vdupq_n_f32(256.0f), u_sq_f32x4, series_b_f32x4);
    series_b_f32x4 = vmulq_f32(vdivq_f32(u_sq_f32x4, vdupq_n_f32(1024.0f)), series_b_f32x4);

    // Δσ calculation
    float32x4_t cos_2sm_sq_f32x4 = vmulq_f32(cos_double_angular_midpoint_f32x4, cos_double_angular_midpoint_f32x4);
    float32x4_t sin_sq_f32x4 = vmulq_f32(sin_angular_distance_f32x4, sin_angular_distance_f32x4);
    float32x4_t term1_f32x4 = vfmaq_f32(vdupq_n_f32(-1.0f), two_f32x4, cos_2sm_sq_f32x4);
    term1_f32x4 = vmulq_f32(cos_angular_distance_f32x4, term1_f32x4);
    float32x4_t term2_f32x4 = vfmaq_f32(vdupq_n_f32(-3.0f), four_f32x4, sin_sq_f32x4);
    float32x4_t term3_f32x4 = vfmaq_f32(vdupq_n_f32(-3.0f), four_f32x4, cos_2sm_sq_f32x4);
    term2_f32x4 = vmulq_f32(vmulq_f32(vdivq_f32(series_b_f32x4, six_f32x4), cos_double_angular_midpoint_f32x4),
                            vmulq_f32(term2_f32x4, term3_f32x4));
    float32x4_t delta_sigma_f32x4 = vmulq_f32(
        series_b_f32x4,
        vmulq_f32(sin_angular_distance_f32x4,
                  vaddq_f32(cos_double_angular_midpoint_f32x4,
                            vmulq_f32(vdivq_f32(series_b_f32x4, four_f32x4), vsubq_f32(term1_f32x4, term2_f32x4)))));

    // s = b * A * (σ - Δσ)
    float32x4_t distances_f32x4 = vmulq_f32(vmulq_f32(polar_radius_f32x4, series_a_f32x4),
                                            vsubq_f32(angular_distance_f32x4, delta_sigma_f32x4));

    // Set coincident points to zero
    distances_f32x4 = vbslq_f32(coincident_mask_u32x4, vdupq_n_f32(0.0f), distances_f32x4);

    return distances_f32x4;
}

NK_PUBLIC void nk_vincenty_f32_neon(                //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        float32x4_t first_latitudes_f32x4 = vld1q_f32(a_lats);
        float32x4_t first_longitudes_f32x4 = vld1q_f32(a_lons);
        float32x4_t second_latitudes_f32x4 = vld1q_f32(b_lats);
        float32x4_t second_longitudes_f32x4 = vld1q_f32(b_lons);

        float32x4_t distances_f32x4 = nk_vincenty_f32x4_neon_(first_latitudes_f32x4, first_longitudes_f32x4,
                                                              second_latitudes_f32x4, second_longitudes_f32x4);
        vst1q_f32(results, distances_f32x4);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        float32x4_t distances_f32x4 = nk_vincenty_f32x4_neon_(a_lat_vec.f32x4, a_lon_vec.f32x4, b_lat_vec.f32x4,
                                                              b_lon_vec.f32x4);
        result_vec.f32x4 = distances_f32x4;
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
#endif // NK_TARGET_ARM64_
#endif // NK_GEOSPATIAL_NEON_H
