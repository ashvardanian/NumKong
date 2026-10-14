/**
 *  @file include/numkong/geospatial/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief SIMD-accelerated geospatial distances for RISC-V.
 *
 *  @sa include/numkong/geospatial.h
 *
 *  Every kernel broadcasts one row and sweeps the columns as many at a time as @c vsetvl grants, so
 *  no tail needs a path of its own. The 3D squares de-interleave @b [x,y,z] rows with the segment
 *  loads @c vlseg3e32 and @c vlseg3e64, which allow at most LMUL=2 for three fields. The @c i32
 *  differences stay at LMUL=2, and @c vwmul and @c vwmacc square them exactly into 64-bit sums at
 *  LMUL=4, while the @c f64 ones run at LMUL=1 to match @c nk_dot2_f64m1_rvv_.
 *
 *  Haversine and Vincenty run at LMUL=4, the width of the trigonometric helpers of the RVV
 *  trigonometry header. Vincenty tracks convergence in @c vbool16_t mask registers, with @c vcpop
 *  checking whether all lanes have converged and @c vmerge applying per-lane updates.
 *
 *  @section rvv_geospatial_instructions Key RVV Geospatial Instructions
 *
 *  @verbatim
 *  Intrinsic                               Purpose
 *  __riscv_vlseg3e32_v_i32m2x3(p, vl)      De-interleave [x,y,z] rows of i32 into three registers
 *  __riscv_vwmacc_vv_i64m4(sum, a, b, vl)  Widening multiply-accumulate: sum += a × b in 64 bits
 *  __riscv_vfsqrt_v_f64m4(x, vl)           Square root (f64, LMUL=4)
 *  __riscv_vfdiv_vv_f64m4(a, b, vl)        Division (f64, LMUL=4)
 *  __riscv_vfmadd_vv_f64m4(a, b, c, vl)    Fused multiply-add: a*b+c (f64)
 *  __riscv_vcpop_m_b16(mask, vl)           Count set bits in mask (convergence check)
 *  __riscv_vmerge_vvm_f64m4(a, b, m, vl)   Conditional merge (per-lane select)
 *  @endverbatim
 */
#ifndef NUMKONG_GEOSPATIAL_RVV_H
#define NUMKONG_GEOSPATIAL_RVV_H

#if NUMKONG_ARCH_RISCV64_
#if NUMKONG_ARCH_RISCV64_RVV_

#include "numkong/types.h"
#include "numkong/geospatial/serial.h" // `nk_geospatial_packed_axes_f64_serial_`
#include "numkong/dot/rvv.h"           // `nk_dot2_f64m1_rvv_`, `nk_dot2_f64m4_rvv_`
#include "numkong/trigonometry/rvv.h"  // `nk_f64m4_sin_rvv_`, `nk_f64m4_cos_rvv_`, `nk_f64m4_atan2_rvv_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Haversine Distance

/** Great-circle metres from a unit vector to @p vector_length others,
 *  2R · atan2(‖a − b‖, ‖a + b‖), with both norms summed in Dot2. */
NUMKONG_INLINE void nk_haversine_f64_rvv_kernel_(                                                      //
    nk_f64_t first_x, nk_f64_t first_y, nk_f64_t first_z,                                              //
    nk_f64_t const *second_packed_x, nk_f64_t const *second_packed_y, nk_f64_t const *second_packed_z, //
    nk_size_t vector_length, nk_f64_t *results) {

    vfloat64m4_t const second_x_f64m4 = __riscv_vle64_v_f64m4(second_packed_x, vector_length);
    vfloat64m4_t const second_y_f64m4 = __riscv_vle64_v_f64m4(second_packed_y, vector_length);
    vfloat64m4_t const second_z_f64m4 = __riscv_vle64_v_f64m4(second_packed_z, vector_length);
    vfloat64m4_t const difference_x_f64m4 = __riscv_vfrsub_vf_f64m4(second_x_f64m4, first_x, vector_length);
    vfloat64m4_t const difference_y_f64m4 = __riscv_vfrsub_vf_f64m4(second_y_f64m4, first_y, vector_length);
    vfloat64m4_t const difference_z_f64m4 = __riscv_vfrsub_vf_f64m4(second_z_f64m4, first_z, vector_length);
    vfloat64m4_t const sum_x_f64m4 = __riscv_vfadd_vf_f64m4(second_x_f64m4, first_x, vector_length);
    vfloat64m4_t const sum_y_f64m4 = __riscv_vfadd_vf_f64m4(second_y_f64m4, first_y, vector_length);
    vfloat64m4_t const sum_z_f64m4 = __riscv_vfadd_vf_f64m4(second_z_f64m4, first_z, vector_length);

    vfloat64m4_t chord_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vector_length), chord_compensation_f64m4 = chord_f64m4;
    vfloat64m4_t span_f64m4 = chord_f64m4, span_compensation_f64m4 = chord_f64m4;
    nk_dot2_f64m4_rvv_(&chord_f64m4, &chord_compensation_f64m4, difference_x_f64m4, difference_x_f64m4, vector_length);
    nk_dot2_f64m4_rvv_(&chord_f64m4, &chord_compensation_f64m4, difference_y_f64m4, difference_y_f64m4, vector_length);
    nk_dot2_f64m4_rvv_(&chord_f64m4, &chord_compensation_f64m4, difference_z_f64m4, difference_z_f64m4, vector_length);
    nk_dot2_f64m4_rvv_(&span_f64m4, &span_compensation_f64m4, sum_x_f64m4, sum_x_f64m4, vector_length);
    nk_dot2_f64m4_rvv_(&span_f64m4, &span_compensation_f64m4, sum_y_f64m4, sum_y_f64m4, vector_length);
    nk_dot2_f64m4_rvv_(&span_f64m4, &span_compensation_f64m4, sum_z_f64m4, sum_z_f64m4, vector_length);

    vfloat64m4_t const angle_f64m4 = nk_f64m4_atan2_rvv_(
        __riscv_vfsqrt_v_f64m4(__riscv_vfadd_vv_f64m4(chord_f64m4, chord_compensation_f64m4, vector_length),
                               vector_length),
        __riscv_vfsqrt_v_f64m4(__riscv_vfadd_vv_f64m4(span_f64m4, span_compensation_f64m4, vector_length),
                               vector_length),
        vector_length);
    __riscv_vse64_v_f64m4(results, __riscv_vfmul_vf_f64m4(angle_f64m4, 2.0 * NUMKONG_EARTH_MEAN_RADIUS, vector_length),
                          vector_length);
}

#pragma endregion Haversine Distance

#pragma region Vincenty Distance

/**
 *  @brief RVV internal kernel for Vincenty's geodesic distance from one point to @p vector_length
 *      others, from their longitudes and the sines and cosines of their reduced latitudes.
 *  @note This is a true SIMD implementation using masked convergence tracking via vmerge.
 *
 *  Vincenty's formulae iterate to solve the geodesic on an oblate spheroid (WGS-84 ellipsoid). Each
 *  SIMD lane tracks its own convergence state via mask registers. The loop terminates when all
 *  lanes have converged (vcpop == vector_length) or after NUMKONG_VINCENTY_MAX_ITERATIONS.
 */
NUMKONG_INLINE void nk_vincenty_f64_rvv_kernel_(                                      //
    nk_f64_t first_longitude, nk_f64_t sin_reduced_first, nk_f64_t cos_reduced_first, //
    nk_f64_t const *second_packed_longitudes, nk_f64_t const *second_packed_sines,
    nk_f64_t const *second_packed_cosines, nk_size_t vector_length, nk_f64_t *results) {

    nk_f64_t const equatorial_radius = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f64_t const flattening = 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_f64_t const polar_radius = equatorial_radius * (1.0 - flattening);
    nk_f64_t const second_eccentricity_squared = (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                                 (polar_radius * polar_radius);
    vfloat64m4_t const flattening_f64m4 = __riscv_vfmv_v_f_f64m4(flattening, vector_length);
    vfloat64m4_t const convergence_threshold_f64m4 = __riscv_vfmv_v_f_f64m4(NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64,
                                                                            vector_length);
    vfloat64m4_t const one_f64m4 = __riscv_vfmv_v_f_f64m4(1.0, vector_length);
    vfloat64m4_t const two_f64m4 = __riscv_vfmv_v_f_f64m4(2.0, vector_length);
    vfloat64m4_t const three_f64m4 = __riscv_vfmv_v_f_f64m4(3.0, vector_length);
    vfloat64m4_t const four_f64m4 = __riscv_vfmv_v_f_f64m4(4.0, vector_length);
    vfloat64m4_t const six_f64m4 = __riscv_vfmv_v_f_f64m4(6.0, vector_length);
    vfloat64m4_t const zero_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vector_length);
    vfloat64m4_t const negative_one_f64m4 = __riscv_vfmv_v_f_f64m4(-1.0, vector_length);
    vfloat64m4_t const negative_three_f64m4 = __riscv_vfmv_v_f_f64m4(-3.0, vector_length);

    vfloat64m4_t const sin_reduced_first_f64m4 = __riscv_vfmv_v_f_f64m4(sin_reduced_first, vector_length);
    vfloat64m4_t const cos_reduced_first_f64m4 = __riscv_vfmv_v_f_f64m4(cos_reduced_first, vector_length);
    vfloat64m4_t const sin_reduced_second_f64m4 = __riscv_vle64_v_f64m4(second_packed_sines, vector_length);
    vfloat64m4_t const cos_reduced_second_f64m4 = __riscv_vle64_v_f64m4(second_packed_cosines, vector_length);
    vfloat64m4_t const longitude_difference_f64m4 = __riscv_vfsub_vf_f64m4(
        __riscv_vle64_v_f64m4(second_packed_longitudes, vector_length), first_longitude, vector_length);

    vfloat64m4_t lambda_f64m4 = longitude_difference_f64m4;
    vfloat64m4_t sin_angular_distance_f64m4 = zero_f64m4;
    vfloat64m4_t cos_angular_distance_f64m4 = zero_f64m4;
    vfloat64m4_t angular_distance_f64m4 = zero_f64m4;
    vfloat64m4_t cos_squared_azimuth_f64m4 = zero_f64m4;
    vfloat64m4_t cos_double_angular_midpoint_f64m4 = zero_f64m4;
    vbool16_t converged_mask_b16 = __riscv_vmclr_m_b16(vector_length);
    vbool16_t coincident_mask_b16 = converged_mask_b16;

    for (nk_u32_t iteration = 0; iteration < NUMKONG_VINCENTY_MAX_ITERATIONS; ++iteration) {
        if (__riscv_vcpop_m_b16(converged_mask_b16, vector_length) == vector_length) break;

        vfloat64m4_t sin_lambda_f64m4 = nk_f64m4_sin_rvv_(lambda_f64m4, vector_length);
        vfloat64m4_t cos_lambda_f64m4 = nk_f64m4_cos_rvv_(lambda_f64m4, vector_length);

        // sin²(σ) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂)
        //   - sin(U₁) × cos(U₂) × cos(λ))²
        vfloat64m4_t cross_term_f64m4 = __riscv_vfmul_vv_f64m4(cos_reduced_second_f64m4, sin_lambda_f64m4,
                                                               vector_length);
        vfloat64m4_t mixed_term_f64m4 = __riscv_vfsub_vv_f64m4(
            __riscv_vfmul_vv_f64m4(cos_reduced_first_f64m4, sin_reduced_second_f64m4, vector_length),
            __riscv_vfmul_vv_f64m4(
                __riscv_vfmul_vv_f64m4(sin_reduced_first_f64m4, cos_reduced_second_f64m4, vector_length),
                cos_lambda_f64m4, vector_length),
            vector_length);
        vfloat64m4_t sin_squared_angular_distance_f64m4 = __riscv_vfmadd_vv_f64m4(
            mixed_term_f64m4, mixed_term_f64m4,
            __riscv_vfmul_vv_f64m4(cross_term_f64m4, cross_term_f64m4, vector_length), vector_length);
        sin_angular_distance_f64m4 = __riscv_vfsqrt_v_f64m4(sin_squared_angular_distance_f64m4, vector_length);
        coincident_mask_b16 = __riscv_vmfeq_vf_f64m4_b16(sin_angular_distance_f64m4, 0.0, vector_length);

        // cos(σ) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        vfloat64m4_t cos_product_f64m4 = __riscv_vfmul_vv_f64m4(cos_reduced_first_f64m4, cos_reduced_second_f64m4,
                                                                vector_length);
        cos_angular_distance_f64m4 = __riscv_vfmadd_vv_f64m4(
            cos_product_f64m4, cos_lambda_f64m4,
            __riscv_vfmul_vv_f64m4(sin_reduced_first_f64m4, sin_reduced_second_f64m4, vector_length), vector_length);
        angular_distance_f64m4 = nk_f64m4_atan2_rvv_(sin_angular_distance_f64m4, cos_angular_distance_f64m4,
                                                     vector_length);

        // sin(α) = cos(U₁) × cos(U₂) × sin(λ) / sin(σ), dividing coincident lanes by one
        vfloat64m4_t safe_sin_angular_distance_f64m4 = __riscv_vfmerge_vfm_f64m4(sin_angular_distance_f64m4, 1.0,
                                                                                 coincident_mask_b16, vector_length);
        vfloat64m4_t sin_azimuth_f64m4 = __riscv_vfdiv_vv_f64m4(
            __riscv_vfmul_vv_f64m4(cos_product_f64m4, sin_lambda_f64m4, vector_length), safe_sin_angular_distance_f64m4,
            vector_length);
        cos_squared_azimuth_f64m4 = __riscv_vfnmsub_vv_f64m4(sin_azimuth_f64m4, sin_azimuth_f64m4, one_f64m4,
                                                             vector_length);

        // An equatorial geodesic, cos²α = 0, has no midpoint term
        vbool16_t equatorial_mask_b16 = __riscv_vmfeq_vf_f64m4_b16(cos_squared_azimuth_f64m4, 0.0, vector_length);
        vfloat64m4_t safe_cos_squared_azimuth_f64m4 = __riscv_vfmerge_vfm_f64m4(cos_squared_azimuth_f64m4, 1.0,
                                                                                equatorial_mask_b16, vector_length);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        vfloat64m4_t sin_product_f64m4 = __riscv_vfmul_vv_f64m4(sin_reduced_first_f64m4, sin_reduced_second_f64m4,
                                                                vector_length);
        cos_double_angular_midpoint_f64m4 = __riscv_vfsub_vv_f64m4(
            cos_angular_distance_f64m4,
            __riscv_vfdiv_vv_f64m4(__riscv_vfmul_vv_f64m4(two_f64m4, sin_product_f64m4, vector_length),
                                   safe_cos_squared_azimuth_f64m4, vector_length),
            vector_length);
        cos_double_angular_midpoint_f64m4 = __riscv_vfmerge_vfm_f64m4(cos_double_angular_midpoint_f64m4, 0.0,
                                                                      equatorial_mask_b16, vector_length);

        // C = f/16 × cos²α × (4 + f × (4 - 3 × cos²α))
        vfloat64m4_t correction_factor_f64m4 = __riscv_vfmul_vv_f64m4(
            __riscv_vfmul_vf_f64m4(cos_squared_azimuth_f64m4, flattening / 16.0, vector_length),
            __riscv_vfmadd_vv_f64m4(
                flattening_f64m4,
                __riscv_vfnmsub_vv_f64m4(three_f64m4, cos_squared_azimuth_f64m4, four_f64m4, vector_length), four_f64m4,
                vector_length),
            vector_length);

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × middle), where
        // middle = cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))
        vfloat64m4_t cos_double_angular_midpoint_squared_f64m4 = __riscv_vfmul_vv_f64m4(
            cos_double_angular_midpoint_f64m4, cos_double_angular_midpoint_f64m4, vector_length);
        vfloat64m4_t innermost_f64m4 = __riscv_vfmadd_vv_f64m4(two_f64m4, cos_double_angular_midpoint_squared_f64m4,
                                                               negative_one_f64m4, vector_length);
        vfloat64m4_t middle_f64m4 = __riscv_vfmadd_vv_f64m4(
            __riscv_vfmul_vv_f64m4(correction_factor_f64m4, cos_angular_distance_f64m4, vector_length), innermost_f64m4,
            cos_double_angular_midpoint_f64m4, vector_length);
        vfloat64m4_t inner_f64m4 = __riscv_vfmul_vv_f64m4(
            __riscv_vfmul_vv_f64m4(correction_factor_f64m4, sin_angular_distance_f64m4, vector_length), middle_f64m4,
            vector_length);
        vfloat64m4_t lambda_next_f64m4 = __riscv_vfmadd_vv_f64m4(
            __riscv_vfmul_vv_f64m4(
                __riscv_vfmul_vv_f64m4(__riscv_vfrsub_vf_f64m4(correction_factor_f64m4, 1.0, vector_length),
                                       flattening_f64m4, vector_length),
                sin_azimuth_f64m4, vector_length),
            __riscv_vfadd_vv_f64m4(angular_distance_f64m4, inner_f64m4, vector_length), longitude_difference_f64m4,
            vector_length);

        vfloat64m4_t lambda_change_f64m4 = __riscv_vfabs_v_f64m4(
            __riscv_vfsub_vv_f64m4(lambda_next_f64m4, lambda_f64m4, vector_length), vector_length);
        // A lane stops where serial does: once its change is not above the threshold, NaN included
        converged_mask_b16 = __riscv_vmorn_mm_b16(
            converged_mask_b16,
            __riscv_vmfgt_vv_f64m4_b16(lambda_change_f64m4, convergence_threshold_f64m4, vector_length), vector_length);

        // Converged lanes keep the λ their outputs came from, as the serial loop stops on it
        lambda_f64m4 = __riscv_vmerge_vvm_f64m4(lambda_next_f64m4, lambda_f64m4, converged_mask_b16, vector_length);
    }

    // u² = cos²α × (a² - b²) / b²
    vfloat64m4_t u_squared_f64m4 = __riscv_vfmul_vf_f64m4(cos_squared_azimuth_f64m4, second_eccentricity_squared,
                                                          vector_length);

    // A = 1 + u²/16384 × (4096 + u² × (-768 + u² × (320 - 175 × u²)))
    vfloat64m4_t series_a_f64m4 = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vf_f64m4(u_squared_f64m4, -175.0, vector_length),
                                                         320.0, vector_length);
    series_a_f64m4 = __riscv_vfmadd_vv_f64m4(u_squared_f64m4, series_a_f64m4,
                                             __riscv_vfmv_v_f_f64m4(-768.0, vector_length), vector_length);
    series_a_f64m4 = __riscv_vfmadd_vv_f64m4(u_squared_f64m4, series_a_f64m4,
                                             __riscv_vfmv_v_f_f64m4(4096.0, vector_length), vector_length);
    series_a_f64m4 = __riscv_vfmadd_vv_f64m4(__riscv_vfmul_vf_f64m4(u_squared_f64m4, 1.0 / 16384.0, vector_length),
                                             series_a_f64m4, one_f64m4, vector_length);

    // B = u²/1024 × (256 + u² × (-128 + u² × (74 - 47 × u²)))
    vfloat64m4_t series_b_f64m4 = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vf_f64m4(u_squared_f64m4, -47.0, vector_length),
                                                         74.0, vector_length);
    series_b_f64m4 = __riscv_vfmadd_vv_f64m4(u_squared_f64m4, series_b_f64m4,
                                             __riscv_vfmv_v_f_f64m4(-128.0, vector_length), vector_length);
    series_b_f64m4 = __riscv_vfmadd_vv_f64m4(u_squared_f64m4, series_b_f64m4,
                                             __riscv_vfmv_v_f_f64m4(256.0, vector_length), vector_length);
    series_b_f64m4 = __riscv_vfmul_vv_f64m4(__riscv_vfmul_vf_f64m4(u_squared_f64m4, 1.0 / 1024.0, vector_length),
                                            series_b_f64m4, vector_length);

    // Δσ = B × sin(σ) × (cos(2σₘ) + B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) -
    //      B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    vfloat64m4_t cos_double_angular_midpoint_squared_f64m4 = __riscv_vfmul_vv_f64m4(
        cos_double_angular_midpoint_f64m4, cos_double_angular_midpoint_f64m4, vector_length);
    vfloat64m4_t sin_squared_angular_distance_f64m4 = __riscv_vfmul_vv_f64m4(sin_angular_distance_f64m4,
                                                                             sin_angular_distance_f64m4, vector_length);
    vfloat64m4_t cosine_term_f64m4 = __riscv_vfmul_vv_f64m4(
        cos_angular_distance_f64m4,
        __riscv_vfmadd_vv_f64m4(two_f64m4, cos_double_angular_midpoint_squared_f64m4, negative_one_f64m4,
                                vector_length),
        vector_length);
    vfloat64m4_t sine_factor_f64m4 = __riscv_vfmadd_vv_f64m4(four_f64m4, sin_squared_angular_distance_f64m4,
                                                             negative_three_f64m4, vector_length);
    vfloat64m4_t midpoint_factor_f64m4 = __riscv_vfmadd_vv_f64m4(four_f64m4, cos_double_angular_midpoint_squared_f64m4,
                                                                 negative_three_f64m4, vector_length);
    vfloat64m4_t series_term_f64m4 = __riscv_vfmul_vv_f64m4(
        __riscv_vfmul_vv_f64m4(__riscv_vfdiv_vv_f64m4(series_b_f64m4, six_f64m4, vector_length),
                               cos_double_angular_midpoint_f64m4, vector_length),
        __riscv_vfmul_vv_f64m4(sine_factor_f64m4, midpoint_factor_f64m4, vector_length), vector_length);
    vfloat64m4_t angular_correction_f64m4 = __riscv_vfmul_vv_f64m4(
        __riscv_vfmul_vv_f64m4(series_b_f64m4, sin_angular_distance_f64m4, vector_length),
        __riscv_vfadd_vv_f64m4(
            cos_double_angular_midpoint_f64m4,
            __riscv_vfmul_vv_f64m4(__riscv_vfdiv_vv_f64m4(series_b_f64m4, four_f64m4, vector_length),
                                   __riscv_vfsub_vv_f64m4(cosine_term_f64m4, series_term_f64m4, vector_length),
                                   vector_length),
            vector_length),
        vector_length);

    // s = b × A × (σ - Δσ), and zero for coincident points
    vfloat64m4_t distances_f64m4 = __riscv_vfmul_vv_f64m4(
        __riscv_vfmul_vf_f64m4(series_a_f64m4, polar_radius, vector_length),
        __riscv_vfsub_vv_f64m4(angular_distance_f64m4, angular_correction_f64m4, vector_length), vector_length);
    distances_f64m4 = __riscv_vfmerge_vfm_f64m4(distances_f64m4, 0.0, coincident_mask_b16, vector_length);
    __riscv_vse64_v_f64m4(results, distances_f64m4, vector_length);
}

#pragma endregion Vincenty Distance

#if NUMKONG_TARGET_RVV

#pragma region 3D Squares

NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_rvv(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                  nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_u64_t *output = (nk_u64_t *)((char *)results + row * results_stride);
        nk_i32_t const a_x = a[row * 3], a_y = a[row * 3 + 1], a_z = a[row * 3 + 2];
        for (nk_size_t vector_length, column = 0; column < columns; column += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(columns - column);
            vint32m2x3_t const b_i32m2x3 = __riscv_vlseg3e32_v_i32m2x3(b + column * 3, vector_length);
            vint32m2_t const difference_x_i32m2 = __riscv_vrsub_vx_i32m2(__riscv_vget_v_i32m2x3_i32m2(b_i32m2x3, 0),
                                                                         a_x, vector_length);
            vint32m2_t const difference_y_i32m2 = __riscv_vrsub_vx_i32m2(__riscv_vget_v_i32m2x3_i32m2(b_i32m2x3, 1),
                                                                         a_y, vector_length);
            vint32m2_t const difference_z_i32m2 = __riscv_vrsub_vx_i32m2(__riscv_vget_v_i32m2x3_i32m2(b_i32m2x3, 2),
                                                                         a_z, vector_length);
            // Each square fits 62 bits, and their sum wraps identically as signed or unsigned
            vint64m4_t sums_i64m4 = __riscv_vwmul_vv_i64m4(difference_x_i32m2, difference_x_i32m2, vector_length);
            sums_i64m4 = __riscv_vwmacc_vv_i64m4(sums_i64m4, difference_y_i32m2, difference_y_i32m2, vector_length);
            sums_i64m4 = __riscv_vwmacc_vv_i64m4(sums_i64m4, difference_z_i32m2, difference_z_i32m2, vector_length);
            __riscv_vse64_v_u64m4(output + column, __riscv_vreinterpret_v_i64m4_u64m4(sums_i64m4), vector_length);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_rvv(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                  nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        nk_f64_t const a_x = a[row * 3], a_y = a[row * 3 + 1], a_z = a[row * 3 + 2];
        for (nk_size_t vector_length, column = 0; column < columns; column += vector_length) {
            vector_length = __riscv_vsetvl_e64m1(columns - column);
            vfloat64m1x3_t const b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b + column * 3, vector_length);
            vfloat64m1_t const difference_x_f64m1 = __riscv_vfrsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0),
                                                                            a_x, vector_length);
            vfloat64m1_t const difference_y_f64m1 = __riscv_vfrsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1),
                                                                            a_y, vector_length);
            vfloat64m1_t const difference_z_f64m1 = __riscv_vfrsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2),
                                                                            a_z, vector_length);
            vfloat64m1_t sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length), compensation_f64m1 = sum_f64m1;
            nk_dot2_f64m1_rvv_(&sum_f64m1, &compensation_f64m1, difference_x_f64m1, difference_x_f64m1, vector_length);
            nk_dot2_f64m1_rvv_(&sum_f64m1, &compensation_f64m1, difference_y_f64m1, difference_y_f64m1, vector_length);
            nk_dot2_f64m1_rvv_(&sum_f64m1, &compensation_f64m1, difference_z_f64m1, difference_z_f64m1, vector_length);
            __riscv_vse64_v_f64m1(output + column, __riscv_vfadd_vv_f64m1(sum_f64m1, compensation_f64m1, vector_length),
                                  vector_length);
        }
    }
    return nk_success_k;
}

#pragma endregion 3D Squares

#pragma region Haversine Distance

NUMKONG_API nk_status_t nk_haversines_packed_f64_rvv(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_x = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_x = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_y = a_packed_x + a_stride, *a_packed_z = a_packed_y + a_stride;
    nk_f64_t const *b_packed_y = b_packed_x + b_stride, *b_packed_z = b_packed_y + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t vector_length, column = 0; column < columns; column += vector_length) {
            vector_length = __riscv_vsetvl_e64m4(columns - column);
            nk_haversine_f64_rvv_kernel_(a_packed_x[row], a_packed_y[row], a_packed_z[row], b_packed_x + column,
                                         b_packed_y + column, b_packed_z + column, vector_length, output + column);
        }
    }
    return nk_success_k;
}

#pragma endregion Haversine Distance

#pragma region Vincenty Distance

NUMKONG_API nk_status_t nk_vincentys_packed_f64_rvv(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_sines = a_packed_longitudes + a_stride, *a_packed_cosines = a_packed_sines + a_stride;
    nk_f64_t const *b_packed_sines = b_packed_longitudes + b_stride, *b_packed_cosines = b_packed_sines + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t vector_length, column = 0; column < columns; column += vector_length) {
            vector_length = __riscv_vsetvl_e64m4(columns - column);
            nk_vincenty_f64_rvv_kernel_(a_packed_longitudes[row], a_packed_sines[row], a_packed_cosines[row],
                                        b_packed_longitudes + column, b_packed_sines + column,
                                        b_packed_cosines + column, vector_length, output + column);
        }
    }
    return nk_success_k;
}

#pragma endregion Vincenty Distance

#endif // NUMKONG_TARGET_RVV

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NUMKONG_ARCH_RISCV64_RVV_
#endif // NUMKONG_ARCH_RISCV64_
#endif // NUMKONG_GEOSPATIAL_RVV_H
