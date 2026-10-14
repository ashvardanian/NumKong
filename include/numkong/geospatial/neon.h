/**
 *  @file include/numkong/geospatial/neon.h
 *  @author Ash Vardanian
 *  @date December 21, 2025
 *  @brief SIMD-accelerated geospatial distances for NEON.
 *
 *  @sa include/numkong/geospatial.h
 *
 *  Every kernel broadcasts one row and sweeps the columns two or four at a time. The 3D squares
 *  de-interleave @b [x,y,z] rows with @c LD3, and their tails with the single-structure form or
 *  from partial loads. The @c i32 differences square exactly into 64-bit lanes, the low halves with
 *  @c SMULL and @c SMLAL and the high ones with their @c SMULL2 and @c SMLAL2 twins. Haversine and
 *  Vincenty read the planar packs with plain vector loads.
 *
 *  @section geospatial_neon_instructions Key NEON Geospatial Instructions
 *
 *  @verbatim
 *  Intrinsic   Instruction    M1 Firestorm  Graviton 3   Graviton 4
 *  vmlal_s32   SMLAL (vec)    3cy @ V0123   4cy @ V02    4cy @ V02
 *  vfmaq_f64   FMLA.D (vec)   4cy @ V0123   4cy @ V0123  4cy @ V0123
 *  vsqrtq_f64  FSQRT.D (vec)  13cy @ V02    16cy @ V02   16cy @ V02
 *  @endverbatim
 */
#ifndef NUMKONG_GEOSPATIAL_NEON_H
#define NUMKONG_GEOSPATIAL_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/types.h"
#include "numkong/geospatial/serial.h" // `nk_geospatial_packed_axes_f64_serial_`
#include "numkong/dot/neon.h"          // `nk_dot2_f64x2_neon_`
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

/** Exact squared distances from a broadcast 3D point to 4 de-interleaved 3D points, the first two
 *  sums in @p sums_low_u64x2 and the last two in @p sums_high_u64x2. */
NUMKONG_INLINE void nk_sqeuclidean3d_i32x4_neon_(                  //
    int32x4_t a_x_i32x4, int32x4_t a_y_i32x4, int32x4_t a_z_i32x4, //
    int32x4_t b_x_i32x4, int32x4_t b_y_i32x4, int32x4_t b_z_i32x4, //
    uint64x2_t *sums_low_u64x2, uint64x2_t *sums_high_u64x2) {
    int32x4_t const difference_x_i32x4 = vsubq_s32(a_x_i32x4, b_x_i32x4);
    int32x4_t const difference_y_i32x4 = vsubq_s32(a_y_i32x4, b_y_i32x4);
    int32x4_t const difference_z_i32x4 = vsubq_s32(a_z_i32x4, b_z_i32x4);
    int64x2_t sums_low_i64x2 = vmull_s32(vget_low_s32(difference_x_i32x4), vget_low_s32(difference_x_i32x4));
    sums_low_i64x2 = vmlal_s32(sums_low_i64x2, vget_low_s32(difference_y_i32x4), vget_low_s32(difference_y_i32x4));
    sums_low_i64x2 = vmlal_s32(sums_low_i64x2, vget_low_s32(difference_z_i32x4), vget_low_s32(difference_z_i32x4));
    int64x2_t sums_high_i64x2 = vmull_high_s32(difference_x_i32x4, difference_x_i32x4);
    sums_high_i64x2 = vmlal_high_s32(sums_high_i64x2, difference_y_i32x4, difference_y_i32x4);
    sums_high_i64x2 = vmlal_high_s32(sums_high_i64x2, difference_z_i32x4, difference_z_i32x4);
    *sums_low_u64x2 = vreinterpretq_u64_s64(sums_low_i64x2);
    *sums_high_u64x2 = vreinterpretq_u64_s64(sums_high_i64x2);
}

/** Squared distances from a broadcast 3D point to 2 interleaved @b [x,y,z] points in three
 *  registers, the three squares summed in Dot2. */
NUMKONG_INLINE float64x2_t nk_sqeuclidean3d_f64x2_neon_(                             //
    float64x2_t first_x_f64x2, float64x2_t first_y_f64x2, float64x2_t first_z_f64x2, //
    float64x2_t second_x_f64x2, float64x2_t second_y_f64x2, float64x2_t second_z_f64x2) {
    float64x2_t const difference_x_f64x2 = vsubq_f64(first_x_f64x2, second_x_f64x2);
    float64x2_t const difference_y_f64x2 = vsubq_f64(first_y_f64x2, second_y_f64x2);
    float64x2_t const difference_z_f64x2 = vsubq_f64(first_z_f64x2, second_z_f64x2);
    float64x2_t sum_f64x2 = vdupq_n_f64(0), compensation_f64x2 = vdupq_n_f64(0);
    nk_dot2_f64x2_neon_(&sum_f64x2, &compensation_f64x2, difference_x_f64x2, difference_x_f64x2);
    nk_dot2_f64x2_neon_(&sum_f64x2, &compensation_f64x2, difference_y_f64x2, difference_y_f64x2);
    nk_dot2_f64x2_neon_(&sum_f64x2, &compensation_f64x2, difference_z_f64x2, difference_z_f64x2);
    return vaddq_f64(sum_f64x2, compensation_f64x2);
}

/** Great-circle metres between lanes of unit vectors, 2R · atan2(‖a − b‖, ‖a + b‖),
 *  with both norms summed in Dot2. */
NUMKONG_INLINE float64x2_t nk_haversine_f64x2_neon_(                                 //
    float64x2_t first_x_f64x2, float64x2_t first_y_f64x2, float64x2_t first_z_f64x2, //
    float64x2_t second_x_f64x2, float64x2_t second_y_f64x2, float64x2_t second_z_f64x2) {
    float64x2_t const difference_x_f64x2 = vsubq_f64(first_x_f64x2, second_x_f64x2);
    float64x2_t const difference_y_f64x2 = vsubq_f64(first_y_f64x2, second_y_f64x2);
    float64x2_t const difference_z_f64x2 = vsubq_f64(first_z_f64x2, second_z_f64x2);
    float64x2_t const sum_x_f64x2 = vaddq_f64(first_x_f64x2, second_x_f64x2);
    float64x2_t const sum_y_f64x2 = vaddq_f64(first_y_f64x2, second_y_f64x2);
    float64x2_t const sum_z_f64x2 = vaddq_f64(first_z_f64x2, second_z_f64x2);
    float64x2_t chord_f64x2 = vdupq_n_f64(0), chord_compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t span_f64x2 = vdupq_n_f64(0), span_compensation_f64x2 = vdupq_n_f64(0);
    nk_dot2_f64x2_neon_(&chord_f64x2, &chord_compensation_f64x2, difference_x_f64x2, difference_x_f64x2);
    nk_dot2_f64x2_neon_(&chord_f64x2, &chord_compensation_f64x2, difference_y_f64x2, difference_y_f64x2);
    nk_dot2_f64x2_neon_(&chord_f64x2, &chord_compensation_f64x2, difference_z_f64x2, difference_z_f64x2);
    nk_dot2_f64x2_neon_(&span_f64x2, &span_compensation_f64x2, sum_x_f64x2, sum_x_f64x2);
    nk_dot2_f64x2_neon_(&span_f64x2, &span_compensation_f64x2, sum_y_f64x2, sum_y_f64x2);
    nk_dot2_f64x2_neon_(&span_f64x2, &span_compensation_f64x2, sum_z_f64x2, sum_z_f64x2);
    float64x2_t const angle_f64x2 = nk_atan2_f64x2_neon_(vsqrtq_f64(vaddq_f64(chord_f64x2, chord_compensation_f64x2)),
                                                         vsqrtq_f64(vaddq_f64(span_f64x2, span_compensation_f64x2)));
    return vmulq_f64(vdupq_n_f64(2.0 * NUMKONG_EARTH_MEAN_RADIUS), angle_f64x2);
}

/**
 *  @brief NEON helper for Vincenty's geodesic distance on 2 f64 point pairs, from their longitudes
 *      and the sines and cosines of their reduced latitudes.
 *  @note This is a true SIMD implementation using masked convergence tracking via blending.
 */
NUMKONG_INLINE float64x2_t nk_vincenty_f64x2_neon_(                                                               //
    float64x2_t first_longitudes_f64x2, float64x2_t sin_reduced_first_f64x2, float64x2_t cos_reduced_first_f64x2, //
    float64x2_t second_longitudes_f64x2, float64x2_t sin_reduced_second_f64x2, float64x2_t cos_reduced_second_f64x2) {

    nk_f64_t const equatorial_radius = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f64_t const flattening = 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_f64_t const polar_radius = equatorial_radius * (1.0 - flattening);
    float64x2_t const polar_radius_f64x2 = vdupq_n_f64(polar_radius);
    float64x2_t const flattening_f64x2 = vdupq_n_f64(flattening);
    float64x2_t const second_eccentricity_squared_f64x2 = vdupq_n_f64(
        (equatorial_radius * equatorial_radius - polar_radius * polar_radius) / (polar_radius * polar_radius));
    float64x2_t const convergence_threshold_f64x2 = vdupq_n_f64(NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    float64x2_t const one_f64x2 = vdupq_n_f64(1.0);
    float64x2_t const two_f64x2 = vdupq_n_f64(2.0);
    float64x2_t const three_f64x2 = vdupq_n_f64(3.0);
    float64x2_t const four_f64x2 = vdupq_n_f64(4.0);
    float64x2_t const six_f64x2 = vdupq_n_f64(6.0);
    float64x2_t const sixteen_f64x2 = vdupq_n_f64(16.0);

    float64x2_t const longitude_difference_f64x2 = vsubq_f64(second_longitudes_f64x2, first_longitudes_f64x2);
    float64x2_t lambda_f64x2 = longitude_difference_f64x2;
    float64x2_t sin_angular_distance_f64x2, cos_angular_distance_f64x2, angular_distance_f64x2;
    float64x2_t sin_azimuth_f64x2, cos_squared_azimuth_f64x2, cos_double_angular_midpoint_f64x2;
    uint64x2_t converged_mask_u64x2 = vdupq_n_u64(0);
    uint64x2_t coincident_mask_u64x2 = vdupq_n_u64(0);

    for (nk_u32_t iteration = 0; iteration < NUMKONG_VINCENTY_MAX_ITERATIONS; ++iteration) {
        nk_u64_t converged_bits = vgetq_lane_u64(converged_mask_u64x2, 0) & vgetq_lane_u64(converged_mask_u64x2, 1);
        if (converged_bits) break;

        float64x2_t sin_lambda_f64x2 = nk_sin_f64x2_neon_(lambda_f64x2);
        float64x2_t cos_lambda_f64x2 = nk_cos_f64x2_neon_(lambda_f64x2);

        // sin²(σ) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂)
        //   - sin(U₁) × cos(U₂) × cos(λ))²
        float64x2_t cross_term_f64x2 = vmulq_f64(cos_reduced_second_f64x2, sin_lambda_f64x2);
        float64x2_t mixed_term_f64x2 = vsubq_f64(
            vmulq_f64(cos_reduced_first_f64x2, sin_reduced_second_f64x2),
            vmulq_f64(vmulq_f64(sin_reduced_first_f64x2, cos_reduced_second_f64x2), cos_lambda_f64x2));
        float64x2_t sin_squared_angular_distance_f64x2 = vfmaq_f64(vmulq_f64(mixed_term_f64x2, mixed_term_f64x2),
                                                                   cross_term_f64x2, cross_term_f64x2);
        sin_angular_distance_f64x2 = vsqrtq_f64(sin_squared_angular_distance_f64x2);
        coincident_mask_u64x2 = vceqzq_f64(sin_angular_distance_f64x2);

        // cos(σ) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f64x2 = vfmaq_f64(vmulq_f64(sin_reduced_first_f64x2, sin_reduced_second_f64x2),
                                               vmulq_f64(cos_reduced_first_f64x2, cos_reduced_second_f64x2),
                                               cos_lambda_f64x2);
        angular_distance_f64x2 = nk_atan2_f64x2_neon_(sin_angular_distance_f64x2, cos_angular_distance_f64x2);

        // sin(α) = cos(U₁) × cos(U₂) × sin(λ) / sin(σ), dividing coincident lanes by one
        float64x2_t safe_sin_angular_distance_f64x2 = vbslq_f64(coincident_mask_u64x2, one_f64x2,
                                                                sin_angular_distance_f64x2);
        sin_azimuth_f64x2 = vdivq_f64(
            vmulq_f64(vmulq_f64(cos_reduced_first_f64x2, cos_reduced_second_f64x2), sin_lambda_f64x2),
            safe_sin_angular_distance_f64x2);
        cos_squared_azimuth_f64x2 = vsubq_f64(one_f64x2, vmulq_f64(sin_azimuth_f64x2, sin_azimuth_f64x2));

        // An equatorial geodesic, cos²α = 0, has no midpoint term
        uint64x2_t equatorial_mask_u64x2 = vceqzq_f64(cos_squared_azimuth_f64x2);
        float64x2_t safe_cos_squared_azimuth_f64x2 = vbslq_f64(equatorial_mask_u64x2, one_f64x2,
                                                               cos_squared_azimuth_f64x2);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        float64x2_t sin_product_f64x2 = vmulq_f64(sin_reduced_first_f64x2, sin_reduced_second_f64x2);
        cos_double_angular_midpoint_f64x2 = vsubq_f64(
            cos_angular_distance_f64x2,
            vdivq_f64(vmulq_f64(two_f64x2, sin_product_f64x2), safe_cos_squared_azimuth_f64x2));
        cos_double_angular_midpoint_f64x2 = vbslq_f64(equatorial_mask_u64x2, vdupq_n_f64(0.0),
                                                      cos_double_angular_midpoint_f64x2);

        // C = f/16 × cos²α × (4 + f × (4 - 3 × cos²α))
        float64x2_t correction_factor_f64x2 = vmulq_f64(
            vdivq_f64(flattening_f64x2, sixteen_f64x2),
            vmulq_f64(cos_squared_azimuth_f64x2,
                      vfmaq_f64(four_f64x2, flattening_f64x2,
                                vfmsq_f64(four_f64x2, three_f64x2, cos_squared_azimuth_f64x2))));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × middle), where
        // middle = cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))
        float64x2_t cos_double_angular_midpoint_squared_f64x2 = vmulq_f64(cos_double_angular_midpoint_f64x2,
                                                                          cos_double_angular_midpoint_f64x2);
        float64x2_t innermost_f64x2 = vfmaq_f64(vdupq_n_f64(-1.0), two_f64x2,
                                                cos_double_angular_midpoint_squared_f64x2);
        float64x2_t middle_f64x2 = vfmaq_f64(cos_double_angular_midpoint_f64x2,
                                             vmulq_f64(correction_factor_f64x2, cos_angular_distance_f64x2),
                                             innermost_f64x2);
        float64x2_t inner_f64x2 = vmulq_f64(vmulq_f64(correction_factor_f64x2, sin_angular_distance_f64x2),
                                            middle_f64x2);
        float64x2_t lambda_next_f64x2 = vfmaq_f64(
            longitude_difference_f64x2,
            vmulq_f64(vmulq_f64(vsubq_f64(one_f64x2, correction_factor_f64x2), flattening_f64x2), sin_azimuth_f64x2),
            vaddq_f64(angular_distance_f64x2, inner_f64x2));

        float64x2_t lambda_change_f64x2 = vabsq_f64(vsubq_f64(lambda_next_f64x2, lambda_f64x2));
        // A lane stops where serial does: once its change is not above the threshold, NaN included
        uint64x2_t moving_mask_u64x2 = vcgtq_f64(lambda_change_f64x2, convergence_threshold_f64x2);
        converged_mask_u64x2 = vornq_u64(converged_mask_u64x2, moving_mask_u64x2);

        // Converged lanes keep the λ their outputs came from, as the serial loop stops on it
        lambda_f64x2 = vbslq_f64(converged_mask_u64x2, lambda_f64x2, lambda_next_f64x2);
    }

    // u² = cos²α × (a² - b²) / b²
    float64x2_t u_squared_f64x2 = vmulq_f64(cos_squared_azimuth_f64x2, second_eccentricity_squared_f64x2);

    // A = 1 + u²/16384 × (4096 + u² × (-768 + u² × (320 - 175 × u²)))
    float64x2_t series_a_f64x2 = vfmaq_f64(vdupq_n_f64(320.0), u_squared_f64x2, vdupq_n_f64(-175.0));
    series_a_f64x2 = vfmaq_f64(vdupq_n_f64(-768.0), u_squared_f64x2, series_a_f64x2);
    series_a_f64x2 = vfmaq_f64(vdupq_n_f64(4096.0), u_squared_f64x2, series_a_f64x2);
    series_a_f64x2 = vfmaq_f64(one_f64x2, vdivq_f64(u_squared_f64x2, vdupq_n_f64(16384.0)), series_a_f64x2);

    // B = u²/1024 × (256 + u² × (-128 + u² × (74 - 47 × u²)))
    float64x2_t series_b_f64x2 = vfmaq_f64(vdupq_n_f64(74.0), u_squared_f64x2, vdupq_n_f64(-47.0));
    series_b_f64x2 = vfmaq_f64(vdupq_n_f64(-128.0), u_squared_f64x2, series_b_f64x2);
    series_b_f64x2 = vfmaq_f64(vdupq_n_f64(256.0), u_squared_f64x2, series_b_f64x2);
    series_b_f64x2 = vmulq_f64(vdivq_f64(u_squared_f64x2, vdupq_n_f64(1024.0)), series_b_f64x2);

    // Δσ = B × sin(σ) × (cos(2σₘ) + B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) -
    //      B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    float64x2_t cos_double_angular_midpoint_squared_f64x2 = vmulq_f64(cos_double_angular_midpoint_f64x2,
                                                                      cos_double_angular_midpoint_f64x2);
    float64x2_t sin_squared_angular_distance_f64x2 = vmulq_f64(sin_angular_distance_f64x2, sin_angular_distance_f64x2);
    float64x2_t cosine_term_f64x2 = vmulq_f64(
        cos_angular_distance_f64x2, vfmaq_f64(vdupq_n_f64(-1.0), two_f64x2, cos_double_angular_midpoint_squared_f64x2));
    float64x2_t sine_factor_f64x2 = vfmaq_f64(vdupq_n_f64(-3.0), four_f64x2, sin_squared_angular_distance_f64x2);
    float64x2_t midpoint_factor_f64x2 = vfmaq_f64(vdupq_n_f64(-3.0), four_f64x2,
                                                  cos_double_angular_midpoint_squared_f64x2);
    float64x2_t series_term_f64x2 = vmulq_f64(
        vmulq_f64(vdivq_f64(series_b_f64x2, six_f64x2), cos_double_angular_midpoint_f64x2),
        vmulq_f64(sine_factor_f64x2, midpoint_factor_f64x2));
    float64x2_t angular_correction_f64x2 = vmulq_f64(
        series_b_f64x2,
        vmulq_f64(sin_angular_distance_f64x2, vaddq_f64(cos_double_angular_midpoint_f64x2,
                                                        vmulq_f64(vdivq_f64(series_b_f64x2, four_f64x2),
                                                                  vsubq_f64(cosine_term_f64x2, series_term_f64x2)))));

    // s = b × A × (σ - Δσ), and zero for coincident points
    float64x2_t distances_f64x2 = vmulq_f64(vmulq_f64(polar_radius_f64x2, series_a_f64x2),
                                            vsubq_f64(angular_distance_f64x2, angular_correction_f64x2));
    return vbslq_f64(coincident_mask_u64x2, vdupq_n_f64(0.0), distances_f64x2);
}

#if NUMKONG_TARGET_NEON

NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_neon(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                   nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_u64_t *output = (nk_u64_t *)((char *)results + row * results_stride);
        int32x4_t const a_x_i32x4 = vdupq_n_s32(a[row * 3]);
        int32x4_t const a_y_i32x4 = vdupq_n_s32(a[row * 3 + 1]);
        int32x4_t const a_z_i32x4 = vdupq_n_s32(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 4 <= columns; column += 4) {
            int32x4x3_t const b_i32x4x3 = vld3q_s32(b + column * 3);
            uint64x2_t sums_low_u64x2, sums_high_u64x2;
            nk_sqeuclidean3d_i32x4_neon_(a_x_i32x4, a_y_i32x4, a_z_i32x4, b_i32x4x3.val[0], b_i32x4x3.val[1],
                                         b_i32x4x3.val[2], &sums_low_u64x2, &sums_high_u64x2);
            vst1q_u64(output + column, sums_low_u64x2);
            vst1q_u64(output + column + 2, sums_high_u64x2);
        }
        if (column < columns) {
            // Partial loads fill three adjacent registers that `LD3` then reads like whole rows
            nk_size_t const remaining = columns - column, values = remaining * 3;
            nk_b128_vec_t b_tail_vec[3], sums_low_vec, sums_high_vec;
            nk_partial_load_b32x4_serial_(b + column * 3, &b_tail_vec[0], values);
            nk_partial_load_b32x4_serial_(b + column * 3 + 4, &b_tail_vec[1], values > 4 ? values - 4 : 0);
            nk_partial_load_b32x4_serial_(b + column * 3 + 8, &b_tail_vec[2], values > 8 ? values - 8 : 0);
            int32x4x3_t const b_i32x4x3 = vld3q_s32((nk_i32_t const *)b_tail_vec);
            nk_sqeuclidean3d_i32x4_neon_(a_x_i32x4, a_y_i32x4, a_z_i32x4, b_i32x4x3.val[0], b_i32x4x3.val[1],
                                         b_i32x4x3.val[2], &sums_low_vec.u64x2, &sums_high_vec.u64x2);
            nk_partial_store_b64x2_serial_(&sums_low_vec, output + column, remaining);
            nk_partial_store_b64x2_serial_(&sums_high_vec, output + column + 2, remaining > 2 ? remaining - 2 : 0);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_neon(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                   nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        float64x2_t const a_x_f64x2 = vdupq_n_f64(a[row * 3]);
        float64x2_t const a_y_f64x2 = vdupq_n_f64(a[row * 3 + 1]);
        float64x2_t const a_z_f64x2 = vdupq_n_f64(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 2 <= columns; column += 2) {
            float64x2x3_t const b_f64x2x3 = vld3q_f64(b + column * 3);
            vst1q_f64(output + column, nk_sqeuclidean3d_f64x2_neon_(a_x_f64x2, a_y_f64x2, a_z_f64x2, b_f64x2x3.val[0],
                                                                    b_f64x2x3.val[1], b_f64x2x3.val[2]));
        }
        if (column < columns) {
            float64x2x3_t b_f64x2x3 = {{vdupq_n_f64(0), vdupq_n_f64(0), vdupq_n_f64(0)}};
            b_f64x2x3 = vld3q_lane_f64(b + column * 3, b_f64x2x3, 0);
            nk_b128_vec_t distances_vec;
            distances_vec.f64x2 = nk_sqeuclidean3d_f64x2_neon_(a_x_f64x2, a_y_f64x2, a_z_f64x2, b_f64x2x3.val[0],
                                                               b_f64x2x3.val[1], b_f64x2x3.val[2]);
            nk_partial_store_b64x2_serial_(&distances_vec, output + column, columns - column);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_haversines_packed_f64_neon(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_x = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_x = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_y = a_packed_x + a_stride, *a_packed_z = a_packed_y + a_stride;
    nk_f64_t const *b_packed_y = b_packed_x + b_stride, *b_packed_z = b_packed_y + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        float64x2_t const a_x_f64x2 = vdupq_n_f64(a_packed_x[row]);
        float64x2_t const a_y_f64x2 = vdupq_n_f64(a_packed_y[row]);
        float64x2_t const a_z_f64x2 = vdupq_n_f64(a_packed_z[row]);
        nk_size_t column = 0;
        for (; column + 2 <= columns; column += 2)
            vst1q_f64(output + column,
                      nk_haversine_f64x2_neon_(a_x_f64x2, a_y_f64x2, a_z_f64x2, vld1q_f64(b_packed_x + column),
                                               vld1q_f64(b_packed_y + column), vld1q_f64(b_packed_z + column)));
        if (column < columns) {
            nk_b128_vec_t b_x_vec, b_y_vec, b_z_vec, distances_vec;
            nk_partial_load_b64x2_serial_(b_packed_x + column, &b_x_vec, columns - column);
            nk_partial_load_b64x2_serial_(b_packed_y + column, &b_y_vec, columns - column);
            nk_partial_load_b64x2_serial_(b_packed_z + column, &b_z_vec, columns - column);
            distances_vec.f64x2 = nk_haversine_f64x2_neon_(a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_vec.f64x2,
                                                           b_y_vec.f64x2, b_z_vec.f64x2);
            nk_partial_store_b64x2_serial_(&distances_vec, output + column, columns - column);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vincentys_packed_f64_neon(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_sines = a_packed_longitudes + a_stride, *a_packed_cosines = a_packed_sines + a_stride;
    nk_f64_t const *b_packed_sines = b_packed_longitudes + b_stride, *b_packed_cosines = b_packed_sines + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        float64x2_t const a_longitude_f64x2 = vdupq_n_f64(a_packed_longitudes[row]);
        float64x2_t const a_sine_f64x2 = vdupq_n_f64(a_packed_sines[row]);
        float64x2_t const a_cosine_f64x2 = vdupq_n_f64(a_packed_cosines[row]);
        nk_size_t column = 0;
        for (; column + 2 <= columns; column += 2)
            vst1q_f64(output + column, nk_vincenty_f64x2_neon_(a_longitude_f64x2, a_sine_f64x2, a_cosine_f64x2,
                                                               vld1q_f64(b_packed_longitudes + column),
                                                               vld1q_f64(b_packed_sines + column),
                                                               vld1q_f64(b_packed_cosines + column)));
        if (column < columns) {
            nk_b128_vec_t b_longitude_vec, b_sine_vec, b_cosine_vec, distances_vec;
            nk_partial_load_b64x2_serial_(b_packed_longitudes + column, &b_longitude_vec, columns - column);
            nk_partial_load_b64x2_serial_(b_packed_sines + column, &b_sine_vec, columns - column);
            nk_partial_load_b64x2_serial_(b_packed_cosines + column, &b_cosine_vec, columns - column);
            distances_vec.f64x2 = nk_vincenty_f64x2_neon_(a_longitude_f64x2, a_sine_f64x2, a_cosine_f64x2,
                                                          b_longitude_vec.f64x2, b_sine_vec.f64x2, b_cosine_vec.f64x2);
            nk_partial_store_b64x2_serial_(&distances_vec, output + column, columns - column);
        }
    }
    return nk_success_k;
}

#endif // NUMKONG_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_GEOSPATIAL_NEON_H
