/**
 *  @brief SIMD-accelerated Geospatial Distances for RISC-V.
 *  @file include/numkong/geospatial/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/geospatial.h
 *
 *  Implements Haversine and Vincenty geodesic distance computations using RVV 1.0 intrinsics
 *  with LMUL=4 (m4) grouping for maximum throughput. The variable-length vector loop uses
 *  `__riscv_vsetvl_e64m4` / `__riscv_vsetvl_e32m4` so each iteration processes as many
 *  point-pairs as the hardware vector length allows, with no scalar tail handling needed.
 *
 *  Trigonometric helpers (sin, cos, atan2) come from trigonometry/rvv.h which provides
 *  polynomial approximations operating on `vfloat64m4_t` / `vfloat32m4_t` vectors.
 *
 *  Vincenty convergence tracking uses RVV mask registers (`vbool16_t` / `vbool8_t`) with
 *  `__riscv_vcpop_m` to check if all lanes have converged, and `__riscv_vmerge` for
 *  per-lane conditional updates.
 *
 *  @section rvv_geospatial_instructions Key RVV Geospatial Instructions
 *
 *      Intrinsic                               Purpose
 *      __riscv_vfsqrt_v_f64m4(x, vl)           Square root (f64, LMUL=4)
 *      __riscv_vfsqrt_v_f32m4(x, vl)           Square root (f32, LMUL=4)
 *      __riscv_vfdiv_vv_f64m4(a, b, vl)        Division (f64, LMUL=4)
 *      __riscv_vfdiv_vv_f32m4(a, b, vl)        Division (f32, LMUL=4)
 *      __riscv_vfmadd_vv_f64m4(a, b, c, vl)    Fused multiply-add: a*b+c (f64)
 *      __riscv_vfmadd_vv_f32m4(a, b, c, vl)    Fused multiply-add: a*b+c (f32)
 *      __riscv_vcpop_m_b16(mask, vl)           Count set bits in mask (convergence check)
 *      __riscv_vmerge_vvm_f64m4(a, b, m, vl)   Conditional merge (per-lane select)
 */
#ifndef NK_GEOSPATIAL_RVV_H
#define NK_GEOSPATIAL_RVV_H

#if NK_TARGET_RISCV64_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/trigonometry/rvv.h" // nk_f64m4_sin_rvv_, nk_f64m4_cos_rvv_, nk_f64m4_atan2_rvv_, etc.

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/*  RVV implementations using LMUL=4 vectors for f64 and f32 geospatial distances.
 *  These require RVV trigonometric kernels from trigonometry/rvv.h.
 */

#pragma region Haversine Distance

/**
 *  @brief  RVV internal kernel for Haversine distance on vector_length f64 point pairs.
 *
 *  Haversine formula:
 *      dlat = lat2 - lat1
 *      dlon = lon2 - lon1
 *      a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
 *      c = 2 * atan2(sqrt(a), sqrt(1 - a))
 *      distance = R * c
 *
 *  where R = NK_EARTH_MEDIATORIAL_RADIUS.
 */
NK_INTERNAL void nk_haversine_f64_rvv_kernel_(      //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t vector_length, nk_f64_t *results) {

    vfloat64m4_t lat1_f64m4 = __riscv_vle64_v_f64m4(a_lats, vector_length);
    vfloat64m4_t lon1_f64m4 = __riscv_vle64_v_f64m4(a_lons, vector_length);
    vfloat64m4_t lat2_f64m4 = __riscv_vle64_v_f64m4(b_lats, vector_length);
    vfloat64m4_t lon2_f64m4 = __riscv_vle64_v_f64m4(b_lons, vector_length);

    vfloat64m4_t dlat_f64m4 = __riscv_vfsub_vv_f64m4(lat2_f64m4, lat1_f64m4, vector_length);
    vfloat64m4_t dlon_f64m4 = __riscv_vfsub_vv_f64m4(lon2_f64m4, lon1_f64m4, vector_length);

    // sin(dlat/2) and sin(dlon/2)
    vfloat64m4_t half_dlat_f64m4 = __riscv_vfmul_vf_f64m4(dlat_f64m4, 0.5, vector_length);
    vfloat64m4_t half_dlon_f64m4 = __riscv_vfmul_vf_f64m4(dlon_f64m4, 0.5, vector_length);
    vfloat64m4_t sin_half_dlat_f64m4 = nk_f64m4_sin_rvv_(half_dlat_f64m4, vector_length);
    vfloat64m4_t sin_half_dlon_f64m4 = nk_f64m4_sin_rvv_(half_dlon_f64m4, vector_length);

    // sin^2(dlat/2) and sin^2(dlon/2)
    vfloat64m4_t sin_sq_half_dlat_f64m4 = __riscv_vfmul_vv_f64m4(sin_half_dlat_f64m4, sin_half_dlat_f64m4,
                                                                 vector_length);
    vfloat64m4_t sin_sq_half_dlon_f64m4 = __riscv_vfmul_vv_f64m4(sin_half_dlon_f64m4, sin_half_dlon_f64m4,
                                                                 vector_length);

    // cos(lat1) * cos(lat2)
    vfloat64m4_t cos_lat1_f64m4 = nk_f64m4_cos_rvv_(lat1_f64m4, vector_length);
    vfloat64m4_t cos_lat2_f64m4 = nk_f64m4_cos_rvv_(lat2_f64m4, vector_length);
    vfloat64m4_t cos_product_f64m4 = __riscv_vfmul_vv_f64m4(cos_lat1_f64m4, cos_lat2_f64m4, vector_length);

    // a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
    vfloat64m4_t haversine_term_f64m4 = __riscv_vfmadd_vv_f64m4(cos_product_f64m4, sin_sq_half_dlon_f64m4,
                                                                sin_sq_half_dlat_f64m4, vector_length);

    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    vfloat64m4_t zero_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vector_length);
    vfloat64m4_t one_f64m4 = __riscv_vfmv_v_f_f64m4(1.0, vector_length);
    haversine_term_f64m4 = __riscv_vfmax_vv_f64m4(zero_f64m4, haversine_term_f64m4, vector_length);
    haversine_term_f64m4 = __riscv_vfmin_vv_f64m4(one_f64m4, haversine_term_f64m4, vector_length);

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    vfloat64m4_t sqrt_haversine_f64m4 = __riscv_vfsqrt_v_f64m4(haversine_term_f64m4, vector_length);
    vfloat64m4_t complement_f64m4 = __riscv_vfsub_vv_f64m4(one_f64m4, haversine_term_f64m4, vector_length);
    vfloat64m4_t sqrt_complement_f64m4 = __riscv_vfsqrt_v_f64m4(complement_f64m4, vector_length);
    vfloat64m4_t central_angle_f64m4 = nk_f64m4_atan2_rvv_(sqrt_haversine_f64m4, sqrt_complement_f64m4, vector_length);
    central_angle_f64m4 = __riscv_vfmul_vf_f64m4(central_angle_f64m4, 2.0, vector_length);

    // distance = R * c
    vfloat64m4_t distances_f64m4 = __riscv_vfmul_vf_f64m4(central_angle_f64m4, NK_EARTH_MEDIATORIAL_RADIUS,
                                                          vector_length);
    __riscv_vse64_v_f64m4(results, distances_f64m4, vector_length);
}

NK_PUBLIC void nk_haversine_f64_rvv(                //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    for (nk_size_t vector_length; n > 0; n -= vector_length, a_lats += vector_length, a_lons += vector_length,
                                         b_lats += vector_length, b_lons += vector_length, results += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        nk_haversine_f64_rvv_kernel_(a_lats, a_lons, b_lats, b_lons, vector_length, results);
    }
}

/**
 *  @brief  RVV internal kernel for Haversine distance on vector_length f32 point pairs.
 */
NK_INTERNAL void nk_haversine_f32_rvv_kernel_(      //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t vector_length, nk_f32_t *results) {

    vfloat32m4_t lat1_f32m4 = __riscv_vle32_v_f32m4(a_lats, vector_length);
    vfloat32m4_t lon1_f32m4 = __riscv_vle32_v_f32m4(a_lons, vector_length);
    vfloat32m4_t lat2_f32m4 = __riscv_vle32_v_f32m4(b_lats, vector_length);
    vfloat32m4_t lon2_f32m4 = __riscv_vle32_v_f32m4(b_lons, vector_length);

    vfloat32m4_t dlat_f32m4 = __riscv_vfsub_vv_f32m4(lat2_f32m4, lat1_f32m4, vector_length);
    vfloat32m4_t dlon_f32m4 = __riscv_vfsub_vv_f32m4(lon2_f32m4, lon1_f32m4, vector_length);

    // sin(dlat/2) and sin(dlon/2)
    vfloat32m4_t half_dlat_f32m4 = __riscv_vfmul_vf_f32m4(dlat_f32m4, 0.5f, vector_length);
    vfloat32m4_t half_dlon_f32m4 = __riscv_vfmul_vf_f32m4(dlon_f32m4, 0.5f, vector_length);
    vfloat32m4_t sin_half_dlat_f32m4 = nk_f32m4_sin_rvv_(half_dlat_f32m4, vector_length);
    vfloat32m4_t sin_half_dlon_f32m4 = nk_f32m4_sin_rvv_(half_dlon_f32m4, vector_length);

    // sin^2(dlat/2) and sin^2(dlon/2)
    vfloat32m4_t sin_sq_half_dlat_f32m4 = __riscv_vfmul_vv_f32m4(sin_half_dlat_f32m4, sin_half_dlat_f32m4,
                                                                 vector_length);
    vfloat32m4_t sin_sq_half_dlon_f32m4 = __riscv_vfmul_vv_f32m4(sin_half_dlon_f32m4, sin_half_dlon_f32m4,
                                                                 vector_length);

    // cos(lat1) * cos(lat2)
    vfloat32m4_t cos_lat1_f32m4 = nk_f32m4_cos_rvv_(lat1_f32m4, vector_length);
    vfloat32m4_t cos_lat2_f32m4 = nk_f32m4_cos_rvv_(lat2_f32m4, vector_length);
    vfloat32m4_t cos_product_f32m4 = __riscv_vfmul_vv_f32m4(cos_lat1_f32m4, cos_lat2_f32m4, vector_length);

    // a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
    vfloat32m4_t haversine_term_f32m4 = __riscv_vfmadd_vv_f32m4(cos_product_f32m4, sin_sq_half_dlon_f32m4,
                                                                sin_sq_half_dlat_f32m4, vector_length);

    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    vfloat32m4_t zero_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vector_length);
    vfloat32m4_t one_f32m4 = __riscv_vfmv_v_f_f32m4(1.0f, vector_length);
    haversine_term_f32m4 = __riscv_vfmax_vv_f32m4(zero_f32m4, haversine_term_f32m4, vector_length);
    haversine_term_f32m4 = __riscv_vfmin_vv_f32m4(one_f32m4, haversine_term_f32m4, vector_length);

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    vfloat32m4_t sqrt_haversine_f32m4 = __riscv_vfsqrt_v_f32m4(haversine_term_f32m4, vector_length);
    vfloat32m4_t complement_f32m4 = __riscv_vfsub_vv_f32m4(one_f32m4, haversine_term_f32m4, vector_length);
    vfloat32m4_t sqrt_complement_f32m4 = __riscv_vfsqrt_v_f32m4(complement_f32m4, vector_length);
    vfloat32m4_t central_angle_f32m4 = nk_f32m4_atan2_rvv_(sqrt_haversine_f32m4, sqrt_complement_f32m4, vector_length);
    central_angle_f32m4 = __riscv_vfmul_vf_f32m4(central_angle_f32m4, 2.0f, vector_length);

    // distance = R * c
    vfloat32m4_t distances_f32m4 = __riscv_vfmul_vf_f32m4(central_angle_f32m4, (nk_f32_t)NK_EARTH_MEDIATORIAL_RADIUS,
                                                          vector_length);
    __riscv_vse32_v_f32m4(results, distances_f32m4, vector_length);
}

NK_PUBLIC void nk_haversine_f32_rvv(                //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    for (nk_size_t vector_length; n > 0; n -= vector_length, a_lats += vector_length, a_lons += vector_length,
                                         b_lats += vector_length, b_lons += vector_length, results += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        nk_haversine_f32_rvv_kernel_(a_lats, a_lons, b_lats, b_lons, vector_length, results);
    }
}

#pragma endregion Haversine Distance

#pragma region Vincenty Distance

/**
 *  @brief  RVV internal kernel for Vincenty's geodesic distance on vector_length f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via vmerge.
 *
 *  Vincenty's formulae iterate to solve the geodesic on an oblate spheroid (WGS-84 ellipsoid).
 *  Each SIMD lane tracks its own convergence state via mask registers. The loop terminates
 *  when all lanes have converged (vcpop == vector_length) or after NK_VINCENTY_MAX_ITERATIONS.
 */
NK_INTERNAL void nk_vincenty_f64_rvv_kernel_(       //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t vector_length, nk_f64_t *results) {

    vfloat64m4_t lat1_f64m4 = __riscv_vle64_v_f64m4(a_lats, vector_length);
    vfloat64m4_t lon1_f64m4 = __riscv_vle64_v_f64m4(a_lons, vector_length);
    vfloat64m4_t lat2_f64m4 = __riscv_vle64_v_f64m4(b_lats, vector_length);
    vfloat64m4_t lon2_f64m4 = __riscv_vle64_v_f64m4(b_lons, vector_length);

    vfloat64m4_t const v_equatorial_radius_f64m4 = __riscv_vfmv_v_f_f64m4(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS,
                                                                          vector_length);
    vfloat64m4_t const v_polar_radius_f64m4 = __riscv_vfmv_v_f_f64m4(NK_EARTH_ELLIPSOID_POLAR_RADIUS, vector_length);
    nk_f64_t const flattening_scalar = 1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    vfloat64m4_t const v_flattening_f64m4 = __riscv_vfmv_v_f_f64m4(flattening_scalar, vector_length);
    vfloat64m4_t const v_convergence_f64m4 = __riscv_vfmv_v_f_f64m4(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64,
                                                                    vector_length);
    vfloat64m4_t const v_one_f64m4 = __riscv_vfmv_v_f_f64m4(1.0, vector_length);
    vfloat64m4_t const v_two_f64m4 = __riscv_vfmv_v_f_f64m4(2.0, vector_length);
    vfloat64m4_t const v_three_f64m4 = __riscv_vfmv_v_f_f64m4(3.0, vector_length);
    vfloat64m4_t const v_four_f64m4 = __riscv_vfmv_v_f_f64m4(4.0, vector_length);
    vfloat64m4_t const v_six_f64m4 = __riscv_vfmv_v_f_f64m4(6.0, vector_length);
    vfloat64m4_t const v_sixteen_f64m4 = __riscv_vfmv_v_f_f64m4(16.0, vector_length);
    vfloat64m4_t const v_epsilon_f64m4 = __riscv_vfmv_v_f_f64m4(1e-15, vector_length);
    vfloat64m4_t const v_zero_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vector_length);
    vfloat64m4_t const v_neg_one_f64m4 = __riscv_vfmv_v_f_f64m4(-1.0, vector_length);

    // Longitude difference
    vfloat64m4_t longitude_diff_f64m4 = __riscv_vfsub_vv_f64m4(lon2_f64m4, lon1_f64m4, vector_length);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    vfloat64m4_t one_minus_f64m4 = __riscv_vfsub_vv_f64m4(v_one_f64m4, v_flattening_f64m4, vector_length);
    vfloat64m4_t sin_lat1_f64m4 = nk_f64m4_sin_rvv_(lat1_f64m4, vector_length);
    vfloat64m4_t cos_lat1_f64m4 = nk_f64m4_cos_rvv_(lat1_f64m4, vector_length);
    vfloat64m4_t sin_lat2_f64m4 = nk_f64m4_sin_rvv_(lat2_f64m4, vector_length);
    vfloat64m4_t cos_lat2_f64m4 = nk_f64m4_cos_rvv_(lat2_f64m4, vector_length);
    vfloat64m4_t tan_first_f64m4 = __riscv_vfdiv_vv_f64m4(sin_lat1_f64m4, cos_lat1_f64m4, vector_length);
    vfloat64m4_t tan_second_f64m4 = __riscv_vfdiv_vv_f64m4(sin_lat2_f64m4, cos_lat2_f64m4, vector_length);
    vfloat64m4_t tan_reduced_first_f64m4 = __riscv_vfmul_vv_f64m4(one_minus_f64m4, tan_first_f64m4, vector_length);
    vfloat64m4_t tan_reduced_second_f64m4 = __riscv_vfmul_vv_f64m4(one_minus_f64m4, tan_second_f64m4, vector_length);

    // cos(U) = 1/sqrt(1 + tan^2(U)), sin(U) = tan(U) * cos(U)
    vfloat64m4_t tan_sq_first_f64m4 = __riscv_vfmadd_vv_f64m4(tan_reduced_first_f64m4, tan_reduced_first_f64m4,
                                                              v_one_f64m4, vector_length);
    vfloat64m4_t cos_reduced_first_f64m4 = __riscv_vfdiv_vv_f64m4(
        v_one_f64m4, __riscv_vfsqrt_v_f64m4(tan_sq_first_f64m4, vector_length), vector_length);
    vfloat64m4_t sin_reduced_first_f64m4 = __riscv_vfmul_vv_f64m4(tan_reduced_first_f64m4, cos_reduced_first_f64m4,
                                                                  vector_length);

    vfloat64m4_t tan_sq_second_f64m4 = __riscv_vfmadd_vv_f64m4(tan_reduced_second_f64m4, tan_reduced_second_f64m4,
                                                               v_one_f64m4, vector_length);
    vfloat64m4_t cos_reduced_second_f64m4 = __riscv_vfdiv_vv_f64m4(
        v_one_f64m4, __riscv_vfsqrt_v_f64m4(tan_sq_second_f64m4, vector_length), vector_length);
    vfloat64m4_t sin_reduced_second_f64m4 = __riscv_vfmul_vv_f64m4(tan_reduced_second_f64m4, cos_reduced_second_f64m4,
                                                                   vector_length);

    // Initialize lambda and tracking variables
    vfloat64m4_t lambda_f64m4 = longitude_diff_f64m4;
    vfloat64m4_t sin_angular_distance_f64m4 = v_zero_f64m4;
    vfloat64m4_t cos_angular_distance_f64m4 = v_zero_f64m4;
    vfloat64m4_t angular_distance_f64m4 = v_zero_f64m4;
    vfloat64m4_t sin_azimuth_f64m4 = v_zero_f64m4;
    vfloat64m4_t cos_sq_azimuth_f64m4 = v_zero_f64m4;
    vfloat64m4_t cos_double_angular_midpoint_f64m4 = v_zero_f64m4;

    // Track convergence and coincident points using masks
    // vbool16_t is the mask type for LMUL=4 with 64-bit elements (64/4 = 16)
    vbool16_t converged_mask_b16 = __riscv_vmfeq_vv_f64m4_b16(v_zero_f64m4, v_one_f64m4, vector_length); // all false
    vbool16_t coincident_mask_b16 = converged_mask_b16;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        if (__riscv_vcpop_m_b16(converged_mask_b16, vector_length) == vector_length) break;

        vfloat64m4_t sin_lambda_f64m4 = nk_f64m4_sin_rvv_(lambda_f64m4, vector_length);
        vfloat64m4_t cos_lambda_f64m4 = nk_f64m4_cos_rvv_(lambda_f64m4, vector_length);

        // sin^2(angular_distance) = (cos(U2)*sin(l))^2 + (cos(U1)*sin(U2) - sin(U1)*cos(U2)*cos(l))^2
        vfloat64m4_t cross_term_f64m4 = __riscv_vfmul_vv_f64m4(cos_reduced_second_f64m4, sin_lambda_f64m4,
                                                               vector_length);
        vfloat64m4_t sin1_cos2_cosl_f64m4 = __riscv_vfmul_vv_f64m4(sin_reduced_first_f64m4, cos_reduced_second_f64m4,
                                                                   vector_length);
        sin1_cos2_cosl_f64m4 = __riscv_vfmul_vv_f64m4(sin1_cos2_cosl_f64m4, cos_lambda_f64m4, vector_length);
        vfloat64m4_t mixed_term_f64m4 = __riscv_vfmul_vv_f64m4(cos_reduced_first_f64m4, sin_reduced_second_f64m4,
                                                               vector_length);
        mixed_term_f64m4 = __riscv_vfsub_vv_f64m4(mixed_term_f64m4, sin1_cos2_cosl_f64m4, vector_length);

        vfloat64m4_t sin_angular_dist_sq_f64m4 = __riscv_vfmul_vv_f64m4(cross_term_f64m4, cross_term_f64m4,
                                                                        vector_length);
        sin_angular_dist_sq_f64m4 = __riscv_vfmadd_vv_f64m4(mixed_term_f64m4, mixed_term_f64m4,
                                                            sin_angular_dist_sq_f64m4, vector_length);
        sin_angular_distance_f64m4 = __riscv_vfsqrt_v_f64m4(sin_angular_dist_sq_f64m4, vector_length);

        // Check for coincident points (sin_angular_distance < epsilon)
        coincident_mask_b16 = __riscv_vmflt_vv_f64m4_b16(sin_angular_distance_f64m4, v_epsilon_f64m4, vector_length);

        // cos(angular_distance) = sin(U1)*sin(U2) + cos(U1)*cos(U2)*cos(l)
        vfloat64m4_t cos1_cos2_f64m4 = __riscv_vfmul_vv_f64m4(cos_reduced_first_f64m4, cos_reduced_second_f64m4,
                                                              vector_length);
        cos_angular_distance_f64m4 = __riscv_vfmul_vv_f64m4(sin_reduced_first_f64m4, sin_reduced_second_f64m4,
                                                            vector_length);
        cos_angular_distance_f64m4 = __riscv_vfmadd_vv_f64m4(cos1_cos2_f64m4, cos_lambda_f64m4,
                                                             cos_angular_distance_f64m4, vector_length);

        // angular_distance = atan2(sin, cos)
        angular_distance_f64m4 = nk_f64m4_atan2_rvv_(sin_angular_distance_f64m4, cos_angular_distance_f64m4,
                                                     vector_length);

        // sin(azimuth) = cos(U1)*cos(U2)*sin(l) / sin(angular_distance)
        // Avoid division by zero by substituting 1.0 for coincident lanes
        vfloat64m4_t safe_sin_angular_f64m4 = __riscv_vfmerge_vfm_f64m4(sin_angular_distance_f64m4, 1.0,
                                                                        coincident_mask_b16, vector_length);
        vfloat64m4_t numerator_f64m4 = __riscv_vfmul_vv_f64m4(cos1_cos2_f64m4, sin_lambda_f64m4, vector_length);
        sin_azimuth_f64m4 = __riscv_vfdiv_vv_f64m4(numerator_f64m4, safe_sin_angular_f64m4, vector_length);
        cos_sq_azimuth_f64m4 = __riscv_vfnmsub_vv_f64m4(sin_azimuth_f64m4, sin_azimuth_f64m4, v_one_f64m4,
                                                        vector_length);

        // Handle equatorial case: cos^2(a) < epsilon
        vbool16_t equatorial_mask_b16 = __riscv_vmflt_vv_f64m4_b16(cos_sq_azimuth_f64m4, v_epsilon_f64m4,
                                                                   vector_length);
        vfloat64m4_t safe_cos_sq_azimuth_f64m4 = __riscv_vfmerge_vfm_f64m4(cos_sq_azimuth_f64m4, 1.0,
                                                                           equatorial_mask_b16, vector_length);

        // cos(2sm) = cos(s) - 2*sin(U1)*sin(U2) / cos^2(a)
        vfloat64m4_t sin_product_f64m4 = __riscv_vfmul_vv_f64m4(sin_reduced_first_f64m4, sin_reduced_second_f64m4,
                                                                vector_length);
        vfloat64m4_t two_sin_product_f64m4 = __riscv_vfmul_vv_f64m4(v_two_f64m4, sin_product_f64m4, vector_length);
        cos_double_angular_midpoint_f64m4 = __riscv_vfdiv_vv_f64m4(two_sin_product_f64m4, safe_cos_sq_azimuth_f64m4,
                                                                   vector_length);
        cos_double_angular_midpoint_f64m4 = __riscv_vfsub_vv_f64m4(cos_angular_distance_f64m4,
                                                                   cos_double_angular_midpoint_f64m4, vector_length);
        // Set to zero for equatorial case
        cos_double_angular_midpoint_f64m4 = __riscv_vfmerge_vfm_f64m4(cos_double_angular_midpoint_f64m4, 0.0,
                                                                      equatorial_mask_b16, vector_length);

        // C = f/16 * cos^2(a) * (4 + f*(4 - 3*cos^2(a)))
        // inner = 4 - 3*cos^2(a)
        vfloat64m4_t inner_c_f64m4 = __riscv_vfnmsub_vv_f64m4(v_three_f64m4, cos_sq_azimuth_f64m4, v_four_f64m4,
                                                              vector_length);
        // 4 + f * inner_c
        vfloat64m4_t outer_c_f64m4 = __riscv_vfmadd_vv_f64m4(v_flattening_f64m4, inner_c_f64m4, v_four_f64m4,
                                                             vector_length);
        // f/16 * cos^2(a) * outer_c
        vfloat64m4_t correction_factor_f64m4 = __riscv_vfdiv_vv_f64m4(v_flattening_f64m4, v_sixteen_f64m4,
                                                                      vector_length);
        correction_factor_f64m4 = __riscv_vfmul_vv_f64m4(correction_factor_f64m4, cos_sq_azimuth_f64m4, vector_length);
        correction_factor_f64m4 = __riscv_vfmul_vv_f64m4(correction_factor_f64m4, outer_c_f64m4, vector_length);

        // lambda' = L + (1-C)*f*sin(a)*(s + C*sin(s)*(cos(2sm) + C*cos(s)*(-1 + 2*cos^2(2sm))))
        vfloat64m4_t cos_2sm_sq_f64m4 = __riscv_vfmul_vv_f64m4(cos_double_angular_midpoint_f64m4,
                                                               cos_double_angular_midpoint_f64m4, vector_length);
        // innermost = -1 + 2*cos^2(2sm)
        vfloat64m4_t innermost_f64m4 = __riscv_vfmadd_vv_f64m4(v_two_f64m4, cos_2sm_sq_f64m4, v_neg_one_f64m4,
                                                               vector_length);
        // middle = cos(2sm) + C*cos(s)*innermost
        vfloat64m4_t c_cos_s_f64m4 = __riscv_vfmul_vv_f64m4(correction_factor_f64m4, cos_angular_distance_f64m4,
                                                            vector_length);
        vfloat64m4_t middle_f64m4 = __riscv_vfmadd_vv_f64m4(c_cos_s_f64m4, innermost_f64m4,
                                                            cos_double_angular_midpoint_f64m4, vector_length);
        // inner = C*sin(s)*middle
        vfloat64m4_t c_sin_s_f64m4 = __riscv_vfmul_vv_f64m4(correction_factor_f64m4, sin_angular_distance_f64m4,
                                                            vector_length);
        vfloat64m4_t inner_value_f64m4 = __riscv_vfmul_vv_f64m4(c_sin_s_f64m4, middle_f64m4, vector_length);

        // (1-C)*f*sin_a*(s + inner)
        vfloat64m4_t one_minus_c_f64m4 = __riscv_vfsub_vv_f64m4(v_one_f64m4, correction_factor_f64m4, vector_length);
        vfloat64m4_t f_sin_a_f64m4 = __riscv_vfmul_vv_f64m4(v_flattening_f64m4, sin_azimuth_f64m4, vector_length);
        vfloat64m4_t s_plus_inner_f64m4 = __riscv_vfadd_vv_f64m4(angular_distance_f64m4, inner_value_f64m4,
                                                                 vector_length);
        vfloat64m4_t adjustment_f64m4 = __riscv_vfmul_vv_f64m4(one_minus_c_f64m4, f_sin_a_f64m4, vector_length);
        adjustment_f64m4 = __riscv_vfmul_vv_f64m4(adjustment_f64m4, s_plus_inner_f64m4, vector_length);
        vfloat64m4_t lambda_new_f64m4 = __riscv_vfadd_vv_f64m4(longitude_diff_f64m4, adjustment_f64m4, vector_length);

        // Check convergence: |lambda - lambda'| < threshold
        vfloat64m4_t lambda_diff_f64m4 = __riscv_vfsub_vv_f64m4(lambda_new_f64m4, lambda_f64m4, vector_length);
        // Absolute value via sign-bit clearing
        vfloat64m4_t lambda_diff_abs_f64m4 = __riscv_vfsgnjx_vv_f64m4(lambda_diff_f64m4, lambda_diff_f64m4,
                                                                      vector_length);
        vbool16_t newly_converged_b16 = __riscv_vmflt_vv_f64m4_b16(lambda_diff_abs_f64m4, v_convergence_f64m4,
                                                                   vector_length);
        converged_mask_b16 = __riscv_vmor_mm_b16(converged_mask_b16, newly_converged_b16, vector_length);

        // Only update lambda for non-converged lanes
        lambda_f64m4 = __riscv_vmerge_vvm_f64m4(lambda_new_f64m4, lambda_f64m4, converged_mask_b16, vector_length);
    }

    // Final distance calculation
    // u^2 = cos^2(a) * (a^2 - b^2) / b^2
    vfloat64m4_t a_sq_f64m4 = __riscv_vfmul_vv_f64m4(v_equatorial_radius_f64m4, v_equatorial_radius_f64m4,
                                                     vector_length);
    vfloat64m4_t b_sq_f64m4 = __riscv_vfmul_vv_f64m4(v_polar_radius_f64m4, v_polar_radius_f64m4, vector_length);
    vfloat64m4_t a_sq_minus_b_sq_f64m4 = __riscv_vfsub_vv_f64m4(a_sq_f64m4, b_sq_f64m4, vector_length);
    vfloat64m4_t u_sq_f64m4 = __riscv_vfmul_vv_f64m4(cos_sq_azimuth_f64m4, a_sq_minus_b_sq_f64m4, vector_length);
    u_sq_f64m4 = __riscv_vfdiv_vv_f64m4(u_sq_f64m4, b_sq_f64m4, vector_length);

    // A = 1 + u^2/16384 * (4096 + u^2*(-768 + u^2*(320 - 175*u^2)))
    vfloat64m4_t series_a_f64m4 = __riscv_vfmul_vf_f64m4(u_sq_f64m4, -175.0, vector_length);
    series_a_f64m4 = __riscv_vfadd_vf_f64m4(series_a_f64m4, 320.0, vector_length);
    series_a_f64m4 = __riscv_vfmadd_vv_f64m4(u_sq_f64m4, series_a_f64m4, __riscv_vfmv_v_f_f64m4(-768.0, vector_length),
                                             vector_length);
    series_a_f64m4 = __riscv_vfmadd_vv_f64m4(u_sq_f64m4, series_a_f64m4, __riscv_vfmv_v_f_f64m4(4096.0, vector_length),
                                             vector_length);
    vfloat64m4_t u_sq_over_16384_f64m4 = __riscv_vfmul_vf_f64m4(u_sq_f64m4, 1.0 / 16384.0, vector_length);
    series_a_f64m4 = __riscv_vfmadd_vv_f64m4(u_sq_over_16384_f64m4, series_a_f64m4, v_one_f64m4, vector_length);

    // B = u^2/1024 * (256 + u^2*(-128 + u^2*(74 - 47*u^2)))
    vfloat64m4_t series_b_f64m4 = __riscv_vfmul_vf_f64m4(u_sq_f64m4, -47.0, vector_length);
    series_b_f64m4 = __riscv_vfadd_vf_f64m4(series_b_f64m4, 74.0, vector_length);
    series_b_f64m4 = __riscv_vfmadd_vv_f64m4(u_sq_f64m4, series_b_f64m4, __riscv_vfmv_v_f_f64m4(-128.0, vector_length),
                                             vector_length);
    series_b_f64m4 = __riscv_vfmadd_vv_f64m4(u_sq_f64m4, series_b_f64m4, __riscv_vfmv_v_f_f64m4(256.0, vector_length),
                                             vector_length);
    vfloat64m4_t u_sq_over_1024_f64m4 = __riscv_vfmul_vf_f64m4(u_sq_f64m4, 1.0 / 1024.0, vector_length);
    series_b_f64m4 = __riscv_vfmul_vv_f64m4(u_sq_over_1024_f64m4, series_b_f64m4, vector_length);

    // Delta-sigma = B*sin(s)*(cos(2sm) + B/4*(cos(s)*(-1+2*cos^2(2sm)) -
    // B/6*cos(2sm)*(-3+4*sin^2(s))*(-3+4*cos^2(2sm))))
    vfloat64m4_t cos_2sm_sq_f64m4 = __riscv_vfmul_vv_f64m4(cos_double_angular_midpoint_f64m4,
                                                           cos_double_angular_midpoint_f64m4, vector_length);
    vfloat64m4_t sin_sq_f64m4 = __riscv_vfmul_vv_f64m4(sin_angular_distance_f64m4, sin_angular_distance_f64m4,
                                                       vector_length);

    // term1 = cos(s) * (-1 + 2*cos^2(2sm))
    vfloat64m4_t term1_f64m4 = __riscv_vfmadd_vv_f64m4(v_two_f64m4, cos_2sm_sq_f64m4, v_neg_one_f64m4, vector_length);
    term1_f64m4 = __riscv_vfmul_vv_f64m4(cos_angular_distance_f64m4, term1_f64m4, vector_length);

    // term2 = B/6 * cos(2sm) * (-3 + 4*sin^2(s)) * (-3 + 4*cos^2(2sm))
    vfloat64m4_t neg_three_f64m4 = __riscv_vfmv_v_f_f64m4(-3.0, vector_length);
    vfloat64m4_t factor_sin_f64m4 = __riscv_vfmadd_vv_f64m4(v_four_f64m4, sin_sq_f64m4, neg_three_f64m4, vector_length);
    vfloat64m4_t factor_cos_f64m4 = __riscv_vfmadd_vv_f64m4(v_four_f64m4, cos_2sm_sq_f64m4, neg_three_f64m4,
                                                            vector_length);
    vfloat64m4_t b_over_6_f64m4 = __riscv_vfdiv_vv_f64m4(series_b_f64m4, v_six_f64m4, vector_length);
    vfloat64m4_t term2_f64m4 = __riscv_vfmul_vv_f64m4(b_over_6_f64m4, cos_double_angular_midpoint_f64m4, vector_length);
    term2_f64m4 = __riscv_vfmul_vv_f64m4(term2_f64m4, factor_sin_f64m4, vector_length);
    term2_f64m4 = __riscv_vfmul_vv_f64m4(term2_f64m4, factor_cos_f64m4, vector_length);

    // B/4 * (term1 - term2)
    vfloat64m4_t b_over_4_f64m4 = __riscv_vfdiv_vv_f64m4(series_b_f64m4, v_four_f64m4, vector_length);
    vfloat64m4_t term1_minus_term2_f64m4 = __riscv_vfsub_vv_f64m4(term1_f64m4, term2_f64m4, vector_length);
    vfloat64m4_t b4_bracket_f64m4 = __riscv_vfmul_vv_f64m4(b_over_4_f64m4, term1_minus_term2_f64m4, vector_length);

    // cos(2sm) + B/4*(...)
    vfloat64m4_t bracket_f64m4 = __riscv_vfadd_vv_f64m4(cos_double_angular_midpoint_f64m4, b4_bracket_f64m4,
                                                        vector_length);

    // delta_sigma = B * sin(s) * bracket
    vfloat64m4_t delta_sigma_f64m4 = __riscv_vfmul_vv_f64m4(series_b_f64m4, sin_angular_distance_f64m4, vector_length);
    delta_sigma_f64m4 = __riscv_vfmul_vv_f64m4(delta_sigma_f64m4, bracket_f64m4, vector_length);

    // s = b * A * (sigma - delta_sigma)
    vfloat64m4_t sigma_minus_ds_f64m4 = __riscv_vfsub_vv_f64m4(angular_distance_f64m4, delta_sigma_f64m4,
                                                               vector_length);
    vfloat64m4_t distances_f64m4 = __riscv_vfmul_vv_f64m4(v_polar_radius_f64m4, series_a_f64m4, vector_length);
    distances_f64m4 = __riscv_vfmul_vv_f64m4(distances_f64m4, sigma_minus_ds_f64m4, vector_length);

    // Set coincident points to zero
    distances_f64m4 = __riscv_vfmerge_vfm_f64m4(distances_f64m4, 0.0, coincident_mask_b16, vector_length);

    __riscv_vse64_v_f64m4(results, distances_f64m4, vector_length);
}

NK_PUBLIC void nk_vincenty_f64_rvv(                 //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    for (nk_size_t vector_length; n > 0; n -= vector_length, a_lats += vector_length, a_lons += vector_length,
                                         b_lats += vector_length, b_lons += vector_length, results += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        nk_vincenty_f64_rvv_kernel_(a_lats, a_lons, b_lats, b_lons, vector_length, results);
    }
}

/**
 *  @brief  RVV internal kernel for Vincenty's geodesic distance on vector_length f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via vmerge.
 */
NK_INTERNAL void nk_vincenty_f32_rvv_kernel_(       //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t vector_length, nk_f32_t *results) {

    vfloat32m4_t lat1_f32m4 = __riscv_vle32_v_f32m4(a_lats, vector_length);
    vfloat32m4_t lon1_f32m4 = __riscv_vle32_v_f32m4(a_lons, vector_length);
    vfloat32m4_t lat2_f32m4 = __riscv_vle32_v_f32m4(b_lats, vector_length);
    vfloat32m4_t lon2_f32m4 = __riscv_vle32_v_f32m4(b_lons, vector_length);

    vfloat32m4_t const v_equatorial_radius_f32m4 = __riscv_vfmv_v_f_f32m4(
        (nk_f32_t)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS, vector_length);
    vfloat32m4_t const v_polar_radius_f32m4 = __riscv_vfmv_v_f_f32m4((nk_f32_t)NK_EARTH_ELLIPSOID_POLAR_RADIUS,
                                                                     vector_length);
    nk_f32_t const flattening_scalar = 1.0f / (nk_f32_t)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    vfloat32m4_t const v_flattening_f32m4 = __riscv_vfmv_v_f_f32m4(flattening_scalar, vector_length);
    vfloat32m4_t const v_convergence_f32m4 = __riscv_vfmv_v_f_f32m4(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32,
                                                                    vector_length);
    vfloat32m4_t const v_one_f32m4 = __riscv_vfmv_v_f_f32m4(1.0f, vector_length);
    vfloat32m4_t const v_two_f32m4 = __riscv_vfmv_v_f_f32m4(2.0f, vector_length);
    vfloat32m4_t const v_three_f32m4 = __riscv_vfmv_v_f_f32m4(3.0f, vector_length);
    vfloat32m4_t const v_four_f32m4 = __riscv_vfmv_v_f_f32m4(4.0f, vector_length);
    vfloat32m4_t const v_six_f32m4 = __riscv_vfmv_v_f_f32m4(6.0f, vector_length);
    vfloat32m4_t const v_sixteen_f32m4 = __riscv_vfmv_v_f_f32m4(16.0f, vector_length);
    vfloat32m4_t const v_epsilon_f32m4 = __riscv_vfmv_v_f_f32m4(1e-7f, vector_length);
    vfloat32m4_t const v_zero_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vector_length);
    vfloat32m4_t const v_neg_one_f32m4 = __riscv_vfmv_v_f_f32m4(-1.0f, vector_length);

    // Longitude difference
    vfloat32m4_t longitude_diff_f32m4 = __riscv_vfsub_vv_f32m4(lon2_f32m4, lon1_f32m4, vector_length);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    vfloat32m4_t one_minus_f32m4 = __riscv_vfsub_vv_f32m4(v_one_f32m4, v_flattening_f32m4, vector_length);
    vfloat32m4_t sin_lat1_f32m4 = nk_f32m4_sin_rvv_(lat1_f32m4, vector_length);
    vfloat32m4_t cos_lat1_f32m4 = nk_f32m4_cos_rvv_(lat1_f32m4, vector_length);
    vfloat32m4_t sin_lat2_f32m4 = nk_f32m4_sin_rvv_(lat2_f32m4, vector_length);
    vfloat32m4_t cos_lat2_f32m4 = nk_f32m4_cos_rvv_(lat2_f32m4, vector_length);
    vfloat32m4_t tan_first_f32m4 = __riscv_vfdiv_vv_f32m4(sin_lat1_f32m4, cos_lat1_f32m4, vector_length);
    vfloat32m4_t tan_second_f32m4 = __riscv_vfdiv_vv_f32m4(sin_lat2_f32m4, cos_lat2_f32m4, vector_length);
    vfloat32m4_t tan_reduced_first_f32m4 = __riscv_vfmul_vv_f32m4(one_minus_f32m4, tan_first_f32m4, vector_length);
    vfloat32m4_t tan_reduced_second_f32m4 = __riscv_vfmul_vv_f32m4(one_minus_f32m4, tan_second_f32m4, vector_length);

    // cos(U) = 1/sqrt(1 + tan^2(U)), sin(U) = tan(U) * cos(U)
    vfloat32m4_t tan_sq_first_f32m4 = __riscv_vfmadd_vv_f32m4(tan_reduced_first_f32m4, tan_reduced_first_f32m4,
                                                              v_one_f32m4, vector_length);
    vfloat32m4_t cos_reduced_first_f32m4 = __riscv_vfdiv_vv_f32m4(
        v_one_f32m4, __riscv_vfsqrt_v_f32m4(tan_sq_first_f32m4, vector_length), vector_length);
    vfloat32m4_t sin_reduced_first_f32m4 = __riscv_vfmul_vv_f32m4(tan_reduced_first_f32m4, cos_reduced_first_f32m4,
                                                                  vector_length);

    vfloat32m4_t tan_sq_second_f32m4 = __riscv_vfmadd_vv_f32m4(tan_reduced_second_f32m4, tan_reduced_second_f32m4,
                                                               v_one_f32m4, vector_length);
    vfloat32m4_t cos_reduced_second_f32m4 = __riscv_vfdiv_vv_f32m4(
        v_one_f32m4, __riscv_vfsqrt_v_f32m4(tan_sq_second_f32m4, vector_length), vector_length);
    vfloat32m4_t sin_reduced_second_f32m4 = __riscv_vfmul_vv_f32m4(tan_reduced_second_f32m4, cos_reduced_second_f32m4,
                                                                   vector_length);

    // Initialize lambda and tracking variables
    vfloat32m4_t lambda_f32m4 = longitude_diff_f32m4;
    vfloat32m4_t sin_angular_distance_f32m4 = v_zero_f32m4;
    vfloat32m4_t cos_angular_distance_f32m4 = v_zero_f32m4;
    vfloat32m4_t angular_distance_f32m4 = v_zero_f32m4;
    vfloat32m4_t sin_azimuth_f32m4 = v_zero_f32m4;
    vfloat32m4_t cos_sq_azimuth_f32m4 = v_zero_f32m4;
    vfloat32m4_t cos_double_angular_midpoint_f32m4 = v_zero_f32m4;

    // Track convergence and coincident points using masks
    // vbool8_t is the mask type for LMUL=4 with 32-bit elements (32/4 = 8)
    vbool8_t converged_mask_b8 = __riscv_vmfeq_vv_f32m4_b8(v_zero_f32m4, v_one_f32m4, vector_length); // all false
    vbool8_t coincident_mask_b8 = converged_mask_b8;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        if (__riscv_vcpop_m_b8(converged_mask_b8, vector_length) == vector_length) break;

        vfloat32m4_t sin_lambda_f32m4 = nk_f32m4_sin_rvv_(lambda_f32m4, vector_length);
        vfloat32m4_t cos_lambda_f32m4 = nk_f32m4_cos_rvv_(lambda_f32m4, vector_length);

        // sin^2(angular_distance) = (cos(U2)*sin(l))^2 + (cos(U1)*sin(U2) - sin(U1)*cos(U2)*cos(l))^2
        vfloat32m4_t cross_term_f32m4 = __riscv_vfmul_vv_f32m4(cos_reduced_second_f32m4, sin_lambda_f32m4,
                                                               vector_length);
        vfloat32m4_t sin1_cos2_cosl_f32m4 = __riscv_vfmul_vv_f32m4(sin_reduced_first_f32m4, cos_reduced_second_f32m4,
                                                                   vector_length);
        sin1_cos2_cosl_f32m4 = __riscv_vfmul_vv_f32m4(sin1_cos2_cosl_f32m4, cos_lambda_f32m4, vector_length);
        vfloat32m4_t mixed_term_f32m4 = __riscv_vfmul_vv_f32m4(cos_reduced_first_f32m4, sin_reduced_second_f32m4,
                                                               vector_length);
        mixed_term_f32m4 = __riscv_vfsub_vv_f32m4(mixed_term_f32m4, sin1_cos2_cosl_f32m4, vector_length);

        vfloat32m4_t sin_angular_dist_sq_f32m4 = __riscv_vfmul_vv_f32m4(cross_term_f32m4, cross_term_f32m4,
                                                                        vector_length);
        sin_angular_dist_sq_f32m4 = __riscv_vfmadd_vv_f32m4(mixed_term_f32m4, mixed_term_f32m4,
                                                            sin_angular_dist_sq_f32m4, vector_length);
        sin_angular_distance_f32m4 = __riscv_vfsqrt_v_f32m4(sin_angular_dist_sq_f32m4, vector_length);

        // Check for coincident points (sin_angular_distance < epsilon)
        coincident_mask_b8 = __riscv_vmflt_vv_f32m4_b8(sin_angular_distance_f32m4, v_epsilon_f32m4, vector_length);

        // cos(angular_distance) = sin(U1)*sin(U2) + cos(U1)*cos(U2)*cos(l)
        vfloat32m4_t cos1_cos2_f32m4 = __riscv_vfmul_vv_f32m4(cos_reduced_first_f32m4, cos_reduced_second_f32m4,
                                                              vector_length);
        cos_angular_distance_f32m4 = __riscv_vfmul_vv_f32m4(sin_reduced_first_f32m4, sin_reduced_second_f32m4,
                                                            vector_length);
        cos_angular_distance_f32m4 = __riscv_vfmadd_vv_f32m4(cos1_cos2_f32m4, cos_lambda_f32m4,
                                                             cos_angular_distance_f32m4, vector_length);

        // angular_distance = atan2(sin, cos)
        angular_distance_f32m4 = nk_f32m4_atan2_rvv_(sin_angular_distance_f32m4, cos_angular_distance_f32m4,
                                                     vector_length);

        // sin(azimuth) = cos(U1)*cos(U2)*sin(l) / sin(angular_distance)
        // Avoid division by zero by substituting 1.0 for coincident lanes
        vfloat32m4_t safe_sin_angular_f32m4 = __riscv_vfmerge_vfm_f32m4(sin_angular_distance_f32m4, 1.0f,
                                                                        coincident_mask_b8, vector_length);
        vfloat32m4_t numerator_f32m4 = __riscv_vfmul_vv_f32m4(cos1_cos2_f32m4, sin_lambda_f32m4, vector_length);
        sin_azimuth_f32m4 = __riscv_vfdiv_vv_f32m4(numerator_f32m4, safe_sin_angular_f32m4, vector_length);
        cos_sq_azimuth_f32m4 = __riscv_vfnmsub_vv_f32m4(sin_azimuth_f32m4, sin_azimuth_f32m4, v_one_f32m4,
                                                        vector_length);

        // Handle equatorial case: cos^2(a) < epsilon
        vbool8_t equatorial_mask_b8 = __riscv_vmflt_vv_f32m4_b8(cos_sq_azimuth_f32m4, v_epsilon_f32m4, vector_length);
        vfloat32m4_t safe_cos_sq_azimuth_f32m4 = __riscv_vfmerge_vfm_f32m4(cos_sq_azimuth_f32m4, 1.0f,
                                                                           equatorial_mask_b8, vector_length);

        // cos(2sm) = cos(s) - 2*sin(U1)*sin(U2) / cos^2(a)
        vfloat32m4_t sin_product_f32m4 = __riscv_vfmul_vv_f32m4(sin_reduced_first_f32m4, sin_reduced_second_f32m4,
                                                                vector_length);
        vfloat32m4_t two_sin_product_f32m4 = __riscv_vfmul_vv_f32m4(v_two_f32m4, sin_product_f32m4, vector_length);
        cos_double_angular_midpoint_f32m4 = __riscv_vfdiv_vv_f32m4(two_sin_product_f32m4, safe_cos_sq_azimuth_f32m4,
                                                                   vector_length);
        cos_double_angular_midpoint_f32m4 = __riscv_vfsub_vv_f32m4(cos_angular_distance_f32m4,
                                                                   cos_double_angular_midpoint_f32m4, vector_length);
        // Set to zero for equatorial case
        cos_double_angular_midpoint_f32m4 = __riscv_vfmerge_vfm_f32m4(cos_double_angular_midpoint_f32m4, 0.0f,
                                                                      equatorial_mask_b8, vector_length);

        // C = f/16 * cos^2(a) * (4 + f*(4 - 3*cos^2(a)))
        vfloat32m4_t inner_c_f32m4 = __riscv_vfnmsub_vv_f32m4(v_three_f32m4, cos_sq_azimuth_f32m4, v_four_f32m4,
                                                              vector_length);
        vfloat32m4_t outer_c_f32m4 = __riscv_vfmadd_vv_f32m4(v_flattening_f32m4, inner_c_f32m4, v_four_f32m4,
                                                             vector_length);
        vfloat32m4_t correction_factor_f32m4 = __riscv_vfdiv_vv_f32m4(v_flattening_f32m4, v_sixteen_f32m4,
                                                                      vector_length);
        correction_factor_f32m4 = __riscv_vfmul_vv_f32m4(correction_factor_f32m4, cos_sq_azimuth_f32m4, vector_length);
        correction_factor_f32m4 = __riscv_vfmul_vv_f32m4(correction_factor_f32m4, outer_c_f32m4, vector_length);

        // lambda' = L + (1-C)*f*sin(a)*(s + C*sin(s)*(cos(2sm) + C*cos(s)*(-1 + 2*cos^2(2sm))))
        vfloat32m4_t cos_2sm_sq_f32m4 = __riscv_vfmul_vv_f32m4(cos_double_angular_midpoint_f32m4,
                                                               cos_double_angular_midpoint_f32m4, vector_length);
        vfloat32m4_t innermost_f32m4 = __riscv_vfmadd_vv_f32m4(v_two_f32m4, cos_2sm_sq_f32m4, v_neg_one_f32m4,
                                                               vector_length);
        vfloat32m4_t c_cos_s_f32m4 = __riscv_vfmul_vv_f32m4(correction_factor_f32m4, cos_angular_distance_f32m4,
                                                            vector_length);
        vfloat32m4_t middle_f32m4 = __riscv_vfmadd_vv_f32m4(c_cos_s_f32m4, innermost_f32m4,
                                                            cos_double_angular_midpoint_f32m4, vector_length);
        vfloat32m4_t c_sin_s_f32m4 = __riscv_vfmul_vv_f32m4(correction_factor_f32m4, sin_angular_distance_f32m4,
                                                            vector_length);
        vfloat32m4_t inner_value_f32m4 = __riscv_vfmul_vv_f32m4(c_sin_s_f32m4, middle_f32m4, vector_length);

        vfloat32m4_t one_minus_c_f32m4 = __riscv_vfsub_vv_f32m4(v_one_f32m4, correction_factor_f32m4, vector_length);
        vfloat32m4_t f_sin_a_f32m4 = __riscv_vfmul_vv_f32m4(v_flattening_f32m4, sin_azimuth_f32m4, vector_length);
        vfloat32m4_t s_plus_inner_f32m4 = __riscv_vfadd_vv_f32m4(angular_distance_f32m4, inner_value_f32m4,
                                                                 vector_length);
        vfloat32m4_t adjustment_f32m4 = __riscv_vfmul_vv_f32m4(one_minus_c_f32m4, f_sin_a_f32m4, vector_length);
        adjustment_f32m4 = __riscv_vfmul_vv_f32m4(adjustment_f32m4, s_plus_inner_f32m4, vector_length);
        vfloat32m4_t lambda_new_f32m4 = __riscv_vfadd_vv_f32m4(longitude_diff_f32m4, adjustment_f32m4, vector_length);

        // Check convergence: |lambda - lambda'| < threshold
        vfloat32m4_t lambda_diff_f32m4 = __riscv_vfsub_vv_f32m4(lambda_new_f32m4, lambda_f32m4, vector_length);
        vfloat32m4_t lambda_diff_abs_f32m4 = __riscv_vfsgnjx_vv_f32m4(lambda_diff_f32m4, lambda_diff_f32m4,
                                                                      vector_length);
        vbool8_t newly_converged_b8 = __riscv_vmflt_vv_f32m4_b8(lambda_diff_abs_f32m4, v_convergence_f32m4,
                                                                vector_length);
        converged_mask_b8 = __riscv_vmor_mm_b8(converged_mask_b8, newly_converged_b8, vector_length);

        // Only update lambda for non-converged lanes
        lambda_f32m4 = __riscv_vmerge_vvm_f32m4(lambda_new_f32m4, lambda_f32m4, converged_mask_b8, vector_length);
    }

    // Final distance calculation
    // u^2 = cos^2(a) * (a^2 - b^2) / b^2
    vfloat32m4_t a_sq_f32m4 = __riscv_vfmul_vv_f32m4(v_equatorial_radius_f32m4, v_equatorial_radius_f32m4,
                                                     vector_length);
    vfloat32m4_t b_sq_f32m4 = __riscv_vfmul_vv_f32m4(v_polar_radius_f32m4, v_polar_radius_f32m4, vector_length);
    vfloat32m4_t a_sq_minus_b_sq_f32m4 = __riscv_vfsub_vv_f32m4(a_sq_f32m4, b_sq_f32m4, vector_length);
    vfloat32m4_t u_sq_f32m4 = __riscv_vfmul_vv_f32m4(cos_sq_azimuth_f32m4, a_sq_minus_b_sq_f32m4, vector_length);
    u_sq_f32m4 = __riscv_vfdiv_vv_f32m4(u_sq_f32m4, b_sq_f32m4, vector_length);

    // A = 1 + u^2/16384 * (4096 + u^2*(-768 + u^2*(320 - 175*u^2)))
    vfloat32m4_t series_a_f32m4 = __riscv_vfmul_vf_f32m4(u_sq_f32m4, -175.0f, vector_length);
    series_a_f32m4 = __riscv_vfadd_vf_f32m4(series_a_f32m4, 320.0f, vector_length);
    series_a_f32m4 = __riscv_vfmadd_vv_f32m4(u_sq_f32m4, series_a_f32m4, __riscv_vfmv_v_f_f32m4(-768.0f, vector_length),
                                             vector_length);
    series_a_f32m4 = __riscv_vfmadd_vv_f32m4(u_sq_f32m4, series_a_f32m4, __riscv_vfmv_v_f_f32m4(4096.0f, vector_length),
                                             vector_length);
    vfloat32m4_t u_sq_over_16384_f32m4 = __riscv_vfmul_vf_f32m4(u_sq_f32m4, 1.0f / 16384.0f, vector_length);
    series_a_f32m4 = __riscv_vfmadd_vv_f32m4(u_sq_over_16384_f32m4, series_a_f32m4, v_one_f32m4, vector_length);

    // B = u^2/1024 * (256 + u^2*(-128 + u^2*(74 - 47*u^2)))
    vfloat32m4_t series_b_f32m4 = __riscv_vfmul_vf_f32m4(u_sq_f32m4, -47.0f, vector_length);
    series_b_f32m4 = __riscv_vfadd_vf_f32m4(series_b_f32m4, 74.0f, vector_length);
    series_b_f32m4 = __riscv_vfmadd_vv_f32m4(u_sq_f32m4, series_b_f32m4, __riscv_vfmv_v_f_f32m4(-128.0f, vector_length),
                                             vector_length);
    series_b_f32m4 = __riscv_vfmadd_vv_f32m4(u_sq_f32m4, series_b_f32m4, __riscv_vfmv_v_f_f32m4(256.0f, vector_length),
                                             vector_length);
    vfloat32m4_t u_sq_over_1024_f32m4 = __riscv_vfmul_vf_f32m4(u_sq_f32m4, 1.0f / 1024.0f, vector_length);
    series_b_f32m4 = __riscv_vfmul_vv_f32m4(u_sq_over_1024_f32m4, series_b_f32m4, vector_length);

    // Delta-sigma calculation
    vfloat32m4_t cos_2sm_sq_f32m4 = __riscv_vfmul_vv_f32m4(cos_double_angular_midpoint_f32m4,
                                                           cos_double_angular_midpoint_f32m4, vector_length);
    vfloat32m4_t sin_sq_f32m4 = __riscv_vfmul_vv_f32m4(sin_angular_distance_f32m4, sin_angular_distance_f32m4,
                                                       vector_length);

    // term1 = cos(s) * (-1 + 2*cos^2(2sm))
    vfloat32m4_t term1_f32m4 = __riscv_vfmadd_vv_f32m4(v_two_f32m4, cos_2sm_sq_f32m4, v_neg_one_f32m4, vector_length);
    term1_f32m4 = __riscv_vfmul_vv_f32m4(cos_angular_distance_f32m4, term1_f32m4, vector_length);

    // term2 = B/6 * cos(2sm) * (-3 + 4*sin^2(s)) * (-3 + 4*cos^2(2sm))
    vfloat32m4_t neg_three_f32m4 = __riscv_vfmv_v_f_f32m4(-3.0f, vector_length);
    vfloat32m4_t factor_sin_f32m4 = __riscv_vfmadd_vv_f32m4(v_four_f32m4, sin_sq_f32m4, neg_three_f32m4, vector_length);
    vfloat32m4_t factor_cos_f32m4 = __riscv_vfmadd_vv_f32m4(v_four_f32m4, cos_2sm_sq_f32m4, neg_three_f32m4,
                                                            vector_length);
    vfloat32m4_t b_over_6_f32m4 = __riscv_vfdiv_vv_f32m4(series_b_f32m4, v_six_f32m4, vector_length);
    vfloat32m4_t term2_f32m4 = __riscv_vfmul_vv_f32m4(b_over_6_f32m4, cos_double_angular_midpoint_f32m4, vector_length);
    term2_f32m4 = __riscv_vfmul_vv_f32m4(term2_f32m4, factor_sin_f32m4, vector_length);
    term2_f32m4 = __riscv_vfmul_vv_f32m4(term2_f32m4, factor_cos_f32m4, vector_length);

    // B/4 * (term1 - term2)
    vfloat32m4_t b_over_4_f32m4 = __riscv_vfdiv_vv_f32m4(series_b_f32m4, v_four_f32m4, vector_length);
    vfloat32m4_t term1_minus_term2_f32m4 = __riscv_vfsub_vv_f32m4(term1_f32m4, term2_f32m4, vector_length);
    vfloat32m4_t b4_bracket_f32m4 = __riscv_vfmul_vv_f32m4(b_over_4_f32m4, term1_minus_term2_f32m4, vector_length);

    // cos(2sm) + B/4*(...)
    vfloat32m4_t bracket_f32m4 = __riscv_vfadd_vv_f32m4(cos_double_angular_midpoint_f32m4, b4_bracket_f32m4,
                                                        vector_length);

    // delta_sigma = B * sin(s) * bracket
    vfloat32m4_t delta_sigma_f32m4 = __riscv_vfmul_vv_f32m4(series_b_f32m4, sin_angular_distance_f32m4, vector_length);
    delta_sigma_f32m4 = __riscv_vfmul_vv_f32m4(delta_sigma_f32m4, bracket_f32m4, vector_length);

    // s = b * A * (sigma - delta_sigma)
    vfloat32m4_t sigma_minus_ds_f32m4 = __riscv_vfsub_vv_f32m4(angular_distance_f32m4, delta_sigma_f32m4,
                                                               vector_length);
    vfloat32m4_t distances_f32m4 = __riscv_vfmul_vv_f32m4(v_polar_radius_f32m4, series_a_f32m4, vector_length);
    distances_f32m4 = __riscv_vfmul_vv_f32m4(distances_f32m4, sigma_minus_ds_f32m4, vector_length);

    // Set coincident points to zero
    distances_f32m4 = __riscv_vfmerge_vfm_f32m4(distances_f32m4, 0.0f, coincident_mask_b8, vector_length);

    __riscv_vse32_v_f32m4(results, distances_f32m4, vector_length);
}

NK_PUBLIC void nk_vincenty_f32_rvv(                 //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    for (nk_size_t vector_length; n > 0; n -= vector_length, a_lats += vector_length, a_lons += vector_length,
                                         b_lats += vector_length, b_lons += vector_length, results += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        nk_vincenty_f32_rvv_kernel_(a_lats, a_lons, b_lats, b_lons, vector_length, results);
    }
}

#pragma endregion Vincenty Distance

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV64_
#endif // NK_GEOSPATIAL_RVV_H
