/**
 *  @file include/numkong/geospatial/v128relaxed.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief SIMD-accelerated geospatial distances for WASM.
 *
 *  @sa include/numkong/geospatial.h
 *
 *  Every kernel broadcasts one row and sweeps the columns two or four at a time. The 3D squares
 *  load three registers of @b [x,y,z] rows and gather each axis with two @c i32x4.shuffle steps for
 *  @c i32 or one @c i64x2.shuffle for @c f64. The @c i32 differences square exactly into 64-bit
 *  lanes with @c i64x2.extmul_low_i32x4_s and its high twin, and the @c f64 ones sum in Dot2
 *  through Veltkamp splitting, as @c relaxed_madd may round twice. Haversine and Vincenty read the
 *  planar packs with plain loads, and Vincenty tests with @c i8x16_all_true whether all lanes
 *  converged at once.
 *
 *  @section geospatial_wasm_instructions Key WASM SIMD Instructions (beyond trig)
 *
 *  @verbatim
 *  Intrinsic                               Operation
 *  wasm_i32x4_shuffle(a, b, ...)           3D de-interleave (two sources per step)
 *  wasm_i64x2_extmul_low_i32x4(a, b)       Exact squares of 32-bit differences into 64-bit lanes
 *  wasm_f64x2_sqrt(a)                      Square root (2-way f64)
 *  wasm_f64x2_div(a, b)                    Division (2-way f64)
 *  wasm_i64x2_relaxed_laneselect(a, b, m)  Lane select for f64 masks (1 instr vs 3 on x86)
 *  wasm_i8x16_all_true(a)                  Vincenty convergence check (all lanes at once)
 *  @endverbatim
 */
#ifndef NUMKONG_GEOSPATIAL_V128RELAXED_H
#define NUMKONG_GEOSPATIAL_V128RELAXED_H

#if NUMKONG_ARCH_WASM_
#if NUMKONG_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/geospatial/serial.h"        // `nk_geospatial_packed_axes_f64_serial_`
#include "numkong/dot/v128.h"                 // `nk_dot2_f64x2_v128_`
#include "numkong/trigonometry/v128relaxed.h" // `nk_f64x2_sin_v128relaxed_`, `nk_f64x2_atan2_v128relaxed_`
#include "numkong/cast/serial.h"              // `nk_partial_load_b32x4_serial_`, `nk_partial_load_b64x2_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/** Exact squared distances from a broadcast 3D point to 4 interleaved @b [x,y,z] points in three
 *  registers, the first two sums in @p sums_low_u64x2 and the last two in @p sums_high_u64x2. */
NUMKONG_INLINE void nk_sqeuclidean3d_i32x4_v128relaxed_(               //
    v128_t a_x_i32x4, v128_t a_y_i32x4, v128_t a_z_i32x4,              //
    v128_t b_first_i32x4, v128_t b_second_i32x4, v128_t b_third_i32x4, //
    v128_t *sums_low_u64x2, v128_t *sums_high_u64x2) {
    // [x0 y0 z0 x1] and [y1 z1 x2 y2] hold three x, three y and two z; [z2 x3 y3 z3] the rest
    v128_t const b_x_i32x4 = wasm_i32x4_shuffle(wasm_i32x4_shuffle(b_first_i32x4, b_second_i32x4, 0, 3, 6, 6),
                                                b_third_i32x4, 0, 1, 2, 5);
    v128_t const b_y_i32x4 = wasm_i32x4_shuffle(wasm_i32x4_shuffle(b_first_i32x4, b_second_i32x4, 1, 4, 7, 7),
                                                b_third_i32x4, 0, 1, 2, 6);
    v128_t const b_z_i32x4 = wasm_i32x4_shuffle(wasm_i32x4_shuffle(b_first_i32x4, b_second_i32x4, 2, 5, 5, 5),
                                                b_third_i32x4, 0, 1, 4, 7);
    v128_t const difference_x_i32x4 = wasm_i32x4_sub(a_x_i32x4, b_x_i32x4);
    v128_t const difference_y_i32x4 = wasm_i32x4_sub(a_y_i32x4, b_y_i32x4);
    v128_t const difference_z_i32x4 = wasm_i32x4_sub(a_z_i32x4, b_z_i32x4);
    *sums_low_u64x2 = wasm_i64x2_add(
        wasm_i64x2_add(wasm_i64x2_extmul_low_i32x4(difference_x_i32x4, difference_x_i32x4),
                       wasm_i64x2_extmul_low_i32x4(difference_y_i32x4, difference_y_i32x4)),
        wasm_i64x2_extmul_low_i32x4(difference_z_i32x4, difference_z_i32x4));
    *sums_high_u64x2 = wasm_i64x2_add(
        wasm_i64x2_add(wasm_i64x2_extmul_high_i32x4(difference_x_i32x4, difference_x_i32x4),
                       wasm_i64x2_extmul_high_i32x4(difference_y_i32x4, difference_y_i32x4)),
        wasm_i64x2_extmul_high_i32x4(difference_z_i32x4, difference_z_i32x4));
}

/** Squared distances from a broadcast 3D point to 2 interleaved @b [x,y,z] points in three
 *  registers, the three squares summed in Dot2. */
NUMKONG_INLINE v128_t nk_sqeuclidean3d_f64x2_v128relaxed_( //
    v128_t a_x_f64x2, v128_t a_y_f64x2, v128_t a_z_f64x2,  //
    v128_t b_first_f64x2, v128_t b_second_f64x2, v128_t b_third_f64x2) {
    // From [x0 y0], [z0 x1] and [y1 z1]
    v128_t const difference_x_f64x2 = wasm_f64x2_sub(a_x_f64x2,
                                                     wasm_i64x2_shuffle(b_first_f64x2, b_second_f64x2, 0, 3));
    v128_t const difference_y_f64x2 = wasm_f64x2_sub(a_y_f64x2, wasm_i64x2_shuffle(b_first_f64x2, b_third_f64x2, 1, 2));
    v128_t const difference_z_f64x2 = wasm_f64x2_sub(a_z_f64x2,
                                                     wasm_i64x2_shuffle(b_second_f64x2, b_third_f64x2, 0, 3));
    v128_t sum_f64x2 = wasm_f64x2_splat(0.0), compensation_f64x2 = wasm_f64x2_splat(0.0);
    nk_dot2_f64x2_v128_(&sum_f64x2, &compensation_f64x2, difference_x_f64x2, difference_x_f64x2);
    nk_dot2_f64x2_v128_(&sum_f64x2, &compensation_f64x2, difference_y_f64x2, difference_y_f64x2);
    nk_dot2_f64x2_v128_(&sum_f64x2, &compensation_f64x2, difference_z_f64x2, difference_z_f64x2);
    return wasm_f64x2_add(sum_f64x2, compensation_f64x2);
}

/** Great-circle metres between lanes of unit vectors, 2R · atan2(‖a − b‖, ‖a + b‖),
 *  with both norms summed in Dot2. */
NUMKONG_INLINE v128_t nk_haversine_f64x2_v128relaxed_(                //
    v128_t first_x_f64x2, v128_t first_y_f64x2, v128_t first_z_f64x2, //
    v128_t second_x_f64x2, v128_t second_y_f64x2, v128_t second_z_f64x2) {
    v128_t const difference_x_f64x2 = wasm_f64x2_sub(first_x_f64x2, second_x_f64x2);
    v128_t const difference_y_f64x2 = wasm_f64x2_sub(first_y_f64x2, second_y_f64x2);
    v128_t const difference_z_f64x2 = wasm_f64x2_sub(first_z_f64x2, second_z_f64x2);
    v128_t const sum_x_f64x2 = wasm_f64x2_add(first_x_f64x2, second_x_f64x2);
    v128_t const sum_y_f64x2 = wasm_f64x2_add(first_y_f64x2, second_y_f64x2);
    v128_t const sum_z_f64x2 = wasm_f64x2_add(first_z_f64x2, second_z_f64x2);
    v128_t chord_f64x2 = wasm_f64x2_splat(0.0), chord_compensation_f64x2 = wasm_f64x2_splat(0.0);
    v128_t span_f64x2 = wasm_f64x2_splat(0.0), span_compensation_f64x2 = wasm_f64x2_splat(0.0);
    nk_dot2_f64x2_v128_(&chord_f64x2, &chord_compensation_f64x2, difference_x_f64x2, difference_x_f64x2);
    nk_dot2_f64x2_v128_(&chord_f64x2, &chord_compensation_f64x2, difference_y_f64x2, difference_y_f64x2);
    nk_dot2_f64x2_v128_(&chord_f64x2, &chord_compensation_f64x2, difference_z_f64x2, difference_z_f64x2);
    nk_dot2_f64x2_v128_(&span_f64x2, &span_compensation_f64x2, sum_x_f64x2, sum_x_f64x2);
    nk_dot2_f64x2_v128_(&span_f64x2, &span_compensation_f64x2, sum_y_f64x2, sum_y_f64x2);
    nk_dot2_f64x2_v128_(&span_f64x2, &span_compensation_f64x2, sum_z_f64x2, sum_z_f64x2);
    v128_t const angle_f64x2 = nk_f64x2_atan2_v128relaxed_(
        wasm_f64x2_sqrt(wasm_f64x2_add(chord_f64x2, chord_compensation_f64x2)),
        wasm_f64x2_sqrt(wasm_f64x2_add(span_f64x2, span_compensation_f64x2)));
    return wasm_f64x2_mul(wasm_f64x2_splat(2.0 * NUMKONG_EARTH_MEAN_RADIUS), angle_f64x2);
}

/**
 *  @brief WASM Relaxed SIMD helper for Vincenty's geodesic distance on 2 f64 point pairs, from
 *      their longitudes and the sines and cosines of their reduced latitudes.
 *  @note This is a true SIMD implementation using masked convergence tracking via blending.
 */
NUMKONG_INLINE v128_t nk_vincenty_f64x2_v128relaxed_(                                              //
    v128_t first_longitudes_f64x2, v128_t sin_reduced_first_f64x2, v128_t cos_reduced_first_f64x2, //
    v128_t second_longitudes_f64x2, v128_t sin_reduced_second_f64x2, v128_t cos_reduced_second_f64x2) {

    nk_f64_t const equatorial_radius = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f64_t const flattening = 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_f64_t const polar_radius = equatorial_radius * (1.0 - flattening);
    v128_t const polar_radius_f64x2 = wasm_f64x2_splat(polar_radius);
    v128_t const flattening_f64x2 = wasm_f64x2_splat(flattening);
    v128_t const second_eccentricity_squared_f64x2 = wasm_f64x2_splat(
        (equatorial_radius * equatorial_radius - polar_radius * polar_radius) / (polar_radius * polar_radius));
    v128_t const convergence_threshold_f64x2 = wasm_f64x2_splat(NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    v128_t const one_f64x2 = wasm_f64x2_splat(1.0);
    v128_t const two_f64x2 = wasm_f64x2_splat(2.0);
    v128_t const three_f64x2 = wasm_f64x2_splat(3.0);
    v128_t const four_f64x2 = wasm_f64x2_splat(4.0);
    v128_t const six_f64x2 = wasm_f64x2_splat(6.0);
    v128_t const sixteen_f64x2 = wasm_f64x2_splat(16.0);
    v128_t const zero_f64x2 = wasm_f64x2_splat(0.0);

    v128_t const longitude_difference_f64x2 = wasm_f64x2_sub(second_longitudes_f64x2, first_longitudes_f64x2);
    v128_t lambda_f64x2 = longitude_difference_f64x2;
    v128_t sin_angular_distance_f64x2, cos_angular_distance_f64x2, angular_distance_f64x2;
    v128_t sin_azimuth_f64x2, cos_squared_azimuth_f64x2, cos_double_angular_midpoint_f64x2;
    v128_t converged_mask_i64x2 = wasm_i64x2_splat(0);
    v128_t coincident_mask_i64x2 = wasm_i64x2_splat(0);

    for (nk_u32_t iteration = 0; iteration < NUMKONG_VINCENTY_MAX_ITERATIONS; ++iteration) {
        if (wasm_i8x16_all_true(converged_mask_i64x2)) break;

        v128_t sin_lambda_f64x2 = nk_f64x2_sin_v128relaxed_(lambda_f64x2);
        v128_t cos_lambda_f64x2 = nk_f64x2_cos_v128relaxed_(lambda_f64x2);

        // sin²(σ) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂)
        //   - sin(U₁) × cos(U₂) × cos(λ))²
        v128_t cross_term_f64x2 = wasm_f64x2_mul(cos_reduced_second_f64x2, sin_lambda_f64x2);
        v128_t mixed_term_f64x2 = wasm_f64x2_sub(
            wasm_f64x2_mul(cos_reduced_first_f64x2, sin_reduced_second_f64x2),
            wasm_f64x2_mul(wasm_f64x2_mul(sin_reduced_first_f64x2, cos_reduced_second_f64x2), cos_lambda_f64x2));
        v128_t sin_squared_angular_distance_f64x2 = wasm_f64x2_relaxed_madd(
            cross_term_f64x2, cross_term_f64x2, wasm_f64x2_mul(mixed_term_f64x2, mixed_term_f64x2));
        sin_angular_distance_f64x2 = wasm_f64x2_sqrt(sin_squared_angular_distance_f64x2);
        coincident_mask_i64x2 = wasm_f64x2_eq(sin_angular_distance_f64x2, zero_f64x2);

        // cos(σ) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f64x2 = wasm_f64x2_relaxed_madd(
            wasm_f64x2_mul(cos_reduced_first_f64x2, cos_reduced_second_f64x2), cos_lambda_f64x2,
            wasm_f64x2_mul(sin_reduced_first_f64x2, sin_reduced_second_f64x2));
        angular_distance_f64x2 = nk_f64x2_atan2_v128relaxed_(sin_angular_distance_f64x2, cos_angular_distance_f64x2);

        // sin(α) = cos(U₁) × cos(U₂) × sin(λ) / sin(σ), dividing coincident lanes by one
        v128_t safe_sin_angular_distance_f64x2 = wasm_i64x2_relaxed_laneselect(one_f64x2, sin_angular_distance_f64x2,
                                                                               coincident_mask_i64x2);
        sin_azimuth_f64x2 = wasm_f64x2_div(
            wasm_f64x2_mul(wasm_f64x2_mul(cos_reduced_first_f64x2, cos_reduced_second_f64x2), sin_lambda_f64x2),
            safe_sin_angular_distance_f64x2);
        cos_squared_azimuth_f64x2 = wasm_f64x2_relaxed_nmadd(sin_azimuth_f64x2, sin_azimuth_f64x2, one_f64x2);

        // An equatorial geodesic, cos²α = 0, has no midpoint term
        v128_t equatorial_mask_i64x2 = wasm_f64x2_eq(cos_squared_azimuth_f64x2, zero_f64x2);
        v128_t safe_cos_squared_azimuth_f64x2 = wasm_i64x2_relaxed_laneselect(one_f64x2, cos_squared_azimuth_f64x2,
                                                                              equatorial_mask_i64x2);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        v128_t sin_product_f64x2 = wasm_f64x2_mul(sin_reduced_first_f64x2, sin_reduced_second_f64x2);
        cos_double_angular_midpoint_f64x2 = wasm_f64x2_sub(
            cos_angular_distance_f64x2,
            wasm_f64x2_div(wasm_f64x2_mul(two_f64x2, sin_product_f64x2), safe_cos_squared_azimuth_f64x2));
        cos_double_angular_midpoint_f64x2 = wasm_i64x2_relaxed_laneselect(
            wasm_f64x2_splat(0.0), cos_double_angular_midpoint_f64x2, equatorial_mask_i64x2);

        // C = f/16 × cos²α × (4 + f × (4 - 3 × cos²α))
        v128_t correction_factor_f64x2 = wasm_f64x2_mul(
            wasm_f64x2_div(flattening_f64x2, sixteen_f64x2),
            wasm_f64x2_mul(
                cos_squared_azimuth_f64x2,
                wasm_f64x2_relaxed_madd(flattening_f64x2,
                                        wasm_f64x2_relaxed_nmadd(three_f64x2, cos_squared_azimuth_f64x2, four_f64x2),
                                        four_f64x2)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × middle), where
        // middle = cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))
        v128_t cos_double_angular_midpoint_squared_f64x2 = wasm_f64x2_mul(cos_double_angular_midpoint_f64x2,
                                                                          cos_double_angular_midpoint_f64x2);
        v128_t innermost_f64x2 = wasm_f64x2_relaxed_madd(two_f64x2, cos_double_angular_midpoint_squared_f64x2,
                                                         wasm_f64x2_splat(-1.0));
        v128_t middle_f64x2 = wasm_f64x2_relaxed_madd(
            wasm_f64x2_mul(correction_factor_f64x2, cos_angular_distance_f64x2), innermost_f64x2,
            cos_double_angular_midpoint_f64x2);
        v128_t inner_f64x2 = wasm_f64x2_mul(wasm_f64x2_mul(correction_factor_f64x2, sin_angular_distance_f64x2),
                                            middle_f64x2);
        v128_t lambda_next_f64x2 = wasm_f64x2_relaxed_madd(
            wasm_f64x2_mul(wasm_f64x2_mul(wasm_f64x2_sub(one_f64x2, correction_factor_f64x2), flattening_f64x2),
                           sin_azimuth_f64x2),
            wasm_f64x2_add(angular_distance_f64x2, inner_f64x2), longitude_difference_f64x2);

        v128_t lambda_change_f64x2 = wasm_f64x2_abs(wasm_f64x2_sub(lambda_next_f64x2, lambda_f64x2));
        // A lane stops where serial does: once its change is not above the threshold, NaN included
        v128_t moving_mask_i64x2 = wasm_f64x2_gt(lambda_change_f64x2, convergence_threshold_f64x2);
        converged_mask_i64x2 = wasm_v128_or(converged_mask_i64x2, wasm_v128_not(moving_mask_i64x2));

        // Converged lanes keep the λ their outputs came from, as the serial loop stops on it
        lambda_f64x2 = wasm_i64x2_relaxed_laneselect(lambda_f64x2, lambda_next_f64x2, converged_mask_i64x2);
    }

    // u² = cos²α × (a² - b²) / b²
    v128_t u_squared_f64x2 = wasm_f64x2_mul(cos_squared_azimuth_f64x2, second_eccentricity_squared_f64x2);

    // A = 1 + u²/16384 × (4096 + u² × (-768 + u² × (320 - 175 × u²)))
    v128_t series_a_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, wasm_f64x2_splat(-175.0), wasm_f64x2_splat(320.0));
    series_a_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, series_a_f64x2, wasm_f64x2_splat(-768.0));
    series_a_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, series_a_f64x2, wasm_f64x2_splat(4096.0));
    series_a_f64x2 = wasm_f64x2_relaxed_madd(wasm_f64x2_div(u_squared_f64x2, wasm_f64x2_splat(16384.0)), series_a_f64x2,
                                             one_f64x2);

    // B = u²/1024 × (256 + u² × (-128 + u² × (74 - 47 × u²)))
    v128_t series_b_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, wasm_f64x2_splat(-47.0), wasm_f64x2_splat(74.0));
    series_b_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, series_b_f64x2, wasm_f64x2_splat(-128.0));
    series_b_f64x2 = wasm_f64x2_relaxed_madd(u_squared_f64x2, series_b_f64x2, wasm_f64x2_splat(256.0));
    series_b_f64x2 = wasm_f64x2_mul(wasm_f64x2_div(u_squared_f64x2, wasm_f64x2_splat(1024.0)), series_b_f64x2);

    // Δσ = B × sin(σ) × (cos(2σₘ) + B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) -
    //      B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    v128_t cos_double_angular_midpoint_squared_f64x2 = wasm_f64x2_mul(cos_double_angular_midpoint_f64x2,
                                                                      cos_double_angular_midpoint_f64x2);
    v128_t sin_squared_angular_distance_f64x2 = wasm_f64x2_mul(sin_angular_distance_f64x2, sin_angular_distance_f64x2);
    v128_t cosine_term_f64x2 = wasm_f64x2_mul(
        cos_angular_distance_f64x2,
        wasm_f64x2_relaxed_madd(two_f64x2, cos_double_angular_midpoint_squared_f64x2, wasm_f64x2_splat(-1.0)));
    v128_t sine_factor_f64x2 = wasm_f64x2_relaxed_madd(four_f64x2, sin_squared_angular_distance_f64x2,
                                                       wasm_f64x2_splat(-3.0));
    v128_t midpoint_factor_f64x2 = wasm_f64x2_relaxed_madd(four_f64x2, cos_double_angular_midpoint_squared_f64x2,
                                                           wasm_f64x2_splat(-3.0));
    v128_t series_term_f64x2 = wasm_f64x2_mul(
        wasm_f64x2_mul(wasm_f64x2_div(series_b_f64x2, six_f64x2), cos_double_angular_midpoint_f64x2),
        wasm_f64x2_mul(sine_factor_f64x2, midpoint_factor_f64x2));
    v128_t angular_correction_f64x2 = wasm_f64x2_mul(
        series_b_f64x2,
        wasm_f64x2_mul(sin_angular_distance_f64x2,
                       wasm_f64x2_add(cos_double_angular_midpoint_f64x2,
                                      wasm_f64x2_mul(wasm_f64x2_div(series_b_f64x2, four_f64x2),
                                                     wasm_f64x2_sub(cosine_term_f64x2, series_term_f64x2)))));

    // s = b × A × (σ - Δσ), and zero for coincident points
    v128_t distances_f64x2 = wasm_f64x2_mul(wasm_f64x2_mul(polar_radius_f64x2, series_a_f64x2),
                                            wasm_f64x2_sub(angular_distance_f64x2, angular_correction_f64x2));
    return wasm_i64x2_relaxed_laneselect(wasm_f64x2_splat(0.0), distances_f64x2, coincident_mask_i64x2);
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_v128relaxed(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                          nk_size_t columns, nk_u64_t *results,
                                                          nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_u64_t *output = (nk_u64_t *)((char *)results + row * results_stride);
        v128_t const a_x_i32x4 = wasm_i32x4_splat(a[row * 3]);
        v128_t const a_y_i32x4 = wasm_i32x4_splat(a[row * 3 + 1]);
        v128_t const a_z_i32x4 = wasm_i32x4_splat(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 4 <= columns; column += 4) {
            nk_i32_t const *b_point = b + column * 3;
            v128_t sums_low_u64x2, sums_high_u64x2;
            nk_sqeuclidean3d_i32x4_v128relaxed_(a_x_i32x4, a_y_i32x4, a_z_i32x4, wasm_v128_load(b_point),
                                                wasm_v128_load(b_point + 4), wasm_v128_load(b_point + 8),
                                                &sums_low_u64x2, &sums_high_u64x2);
            wasm_v128_store(output + column, sums_low_u64x2);
            wasm_v128_store(output + column + 2, sums_high_u64x2);
        }
        if (column < columns) {
            nk_size_t const remaining = columns - column, values = remaining * 3;
            nk_b128_vec_t b_first_vec, b_second_vec, b_third_vec, sums_low_vec, sums_high_vec;
            nk_partial_load_b32x4_serial_(b + column * 3, &b_first_vec, values);
            nk_partial_load_b32x4_serial_(b + column * 3 + 4, &b_second_vec, values > 4 ? values - 4 : 0);
            nk_partial_load_b32x4_serial_(b + column * 3 + 8, &b_third_vec, values > 8 ? values - 8 : 0);
            nk_sqeuclidean3d_i32x4_v128relaxed_(a_x_i32x4, a_y_i32x4, a_z_i32x4, b_first_vec.v128, b_second_vec.v128,
                                                b_third_vec.v128, &sums_low_vec.v128, &sums_high_vec.v128);
            nk_partial_store_b64x2_serial_(&sums_low_vec, output + column, remaining);
            nk_partial_store_b64x2_serial_(&sums_high_vec, output + column + 2, remaining > 2 ? remaining - 2 : 0);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_v128relaxed(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                          nk_size_t columns, nk_f64_t *results,
                                                          nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        v128_t const a_x_f64x2 = wasm_f64x2_splat(a[row * 3]);
        v128_t const a_y_f64x2 = wasm_f64x2_splat(a[row * 3 + 1]);
        v128_t const a_z_f64x2 = wasm_f64x2_splat(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 2 <= columns; column += 2) {
            nk_f64_t const *b_point = b + column * 3;
            wasm_v128_store(output + column, nk_sqeuclidean3d_f64x2_v128relaxed_(
                                                 a_x_f64x2, a_y_f64x2, a_z_f64x2, wasm_v128_load(b_point),
                                                 wasm_v128_load(b_point + 2), wasm_v128_load(b_point + 4)));
        }
        if (column < columns) {
            nk_size_t const remaining = columns - column, values = remaining * 3;
            nk_b128_vec_t b_first_vec, b_second_vec, b_third_vec, distances_vec;
            nk_partial_load_b64x2_serial_(b + column * 3, &b_first_vec, values);
            nk_partial_load_b64x2_serial_(b + column * 3 + 2, &b_second_vec, values > 2 ? values - 2 : 0);
            nk_partial_load_b64x2_serial_(b + column * 3 + 4, &b_third_vec, values > 4 ? values - 4 : 0);
            distances_vec.v128 = nk_sqeuclidean3d_f64x2_v128relaxed_(a_x_f64x2, a_y_f64x2, a_z_f64x2, b_first_vec.v128,
                                                                     b_second_vec.v128, b_third_vec.v128);
            nk_partial_store_b64x2_serial_(&distances_vec, output + column, remaining);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_haversines_packed_f64_v128relaxed(void const *a_packed, void const *b_packed,
                                                             nk_f64_t *results, nk_size_t results_stride,
                                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_x = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_x = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_y = a_packed_x + a_stride, *a_packed_z = a_packed_y + a_stride;
    nk_f64_t const *b_packed_y = b_packed_x + b_stride, *b_packed_z = b_packed_y + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        v128_t const a_x_f64x2 = wasm_f64x2_splat(a_packed_x[row]);
        v128_t const a_y_f64x2 = wasm_f64x2_splat(a_packed_y[row]);
        v128_t const a_z_f64x2 = wasm_f64x2_splat(a_packed_z[row]);
        nk_size_t column = 0;
        for (; column + 2 <= columns; column += 2)
            wasm_v128_store(output + column, nk_haversine_f64x2_v128relaxed_(a_x_f64x2, a_y_f64x2, a_z_f64x2,
                                                                             wasm_v128_load(b_packed_x + column),
                                                                             wasm_v128_load(b_packed_y + column),
                                                                             wasm_v128_load(b_packed_z + column)));
        if (column < columns) {
            nk_b128_vec_t b_x_vec, b_y_vec, b_z_vec, distances_vec;
            nk_partial_load_b64x2_serial_(b_packed_x + column, &b_x_vec, columns - column);
            nk_partial_load_b64x2_serial_(b_packed_y + column, &b_y_vec, columns - column);
            nk_partial_load_b64x2_serial_(b_packed_z + column, &b_z_vec, columns - column);
            distances_vec.v128 = nk_haversine_f64x2_v128relaxed_(a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_vec.v128,
                                                                 b_y_vec.v128, b_z_vec.v128);
            nk_partial_store_b64x2_serial_(&distances_vec, output + column, columns - column);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vincentys_packed_f64_v128relaxed(void const *a_packed, void const *b_packed,
                                                            nk_f64_t *results, nk_size_t results_stride,
                                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_sines = a_packed_longitudes + a_stride, *a_packed_cosines = a_packed_sines + a_stride;
    nk_f64_t const *b_packed_sines = b_packed_longitudes + b_stride, *b_packed_cosines = b_packed_sines + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        v128_t const a_longitude_f64x2 = wasm_f64x2_splat(a_packed_longitudes[row]);
        v128_t const a_sine_f64x2 = wasm_f64x2_splat(a_packed_sines[row]);
        v128_t const a_cosine_f64x2 = wasm_f64x2_splat(a_packed_cosines[row]);
        nk_size_t column = 0;
        for (; column + 2 <= columns; column += 2)
            wasm_v128_store(output + column,
                            nk_vincenty_f64x2_v128relaxed_(a_longitude_f64x2, a_sine_f64x2, a_cosine_f64x2,
                                                           wasm_v128_load(b_packed_longitudes + column),
                                                           wasm_v128_load(b_packed_sines + column),
                                                           wasm_v128_load(b_packed_cosines + column)));
        if (column < columns) {
            nk_b128_vec_t b_longitude_vec, b_sine_vec, b_cosine_vec, distances_vec;
            nk_partial_load_b64x2_serial_(b_packed_longitudes + column, &b_longitude_vec, columns - column);
            nk_partial_load_b64x2_serial_(b_packed_sines + column, &b_sine_vec, columns - column);
            nk_partial_load_b64x2_serial_(b_packed_cosines + column, &b_cosine_vec, columns - column);
            distances_vec.v128 = nk_vincenty_f64x2_v128relaxed_(a_longitude_f64x2, a_sine_f64x2, a_cosine_f64x2,
                                                                b_longitude_vec.v128, b_sine_vec.v128,
                                                                b_cosine_vec.v128);
            nk_partial_store_b64x2_serial_(&distances_vec, output + column, columns - column);
        }
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_V128RELAXED
#endif // NUMKONG_ARCH_WASM_
#endif // NUMKONG_GEOSPATIAL_V128RELAXED_H
