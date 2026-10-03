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

#if NK_TARGET_RISCV_
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

#pragma region - Haversine Distance

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

    vfloat64m4_t lat1 = __riscv_vle64_v_f64m4(a_lats, vector_length);
    vfloat64m4_t lon1 = __riscv_vle64_v_f64m4(a_lons, vector_length);
    vfloat64m4_t lat2 = __riscv_vle64_v_f64m4(b_lats, vector_length);
    vfloat64m4_t lon2 = __riscv_vle64_v_f64m4(b_lons, vector_length);

    vfloat64m4_t dlat = __riscv_vfsub_vv_f64m4(lat2, lat1, vector_length);
    vfloat64m4_t dlon = __riscv_vfsub_vv_f64m4(lon2, lon1, vector_length);

    // sin(dlat/2) and sin(dlon/2)
    vfloat64m4_t half_dlat = __riscv_vfmul_vf_f64m4(dlat, 0.5, vector_length);
    vfloat64m4_t half_dlon = __riscv_vfmul_vf_f64m4(dlon, 0.5, vector_length);
    vfloat64m4_t sin_half_dlat = nk_f64m4_sin_rvv_(half_dlat, vector_length);
    vfloat64m4_t sin_half_dlon = nk_f64m4_sin_rvv_(half_dlon, vector_length);

    // sin^2(dlat/2) and sin^2(dlon/2)
    vfloat64m4_t sin_sq_half_dlat = __riscv_vfmul_vv_f64m4(sin_half_dlat, sin_half_dlat, vector_length);
    vfloat64m4_t sin_sq_half_dlon = __riscv_vfmul_vv_f64m4(sin_half_dlon, sin_half_dlon, vector_length);

    // cos(lat1) * cos(lat2)
    vfloat64m4_t cos_lat1 = nk_f64m4_cos_rvv_(lat1, vector_length);
    vfloat64m4_t cos_lat2 = nk_f64m4_cos_rvv_(lat2, vector_length);
    vfloat64m4_t cos_product = __riscv_vfmul_vv_f64m4(cos_lat1, cos_lat2, vector_length);

    // a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
    vfloat64m4_t haversine_term = __riscv_vfmadd_vv_f64m4(cos_product, sin_sq_half_dlon, sin_sq_half_dlat,
                                                          vector_length);

    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    vfloat64m4_t zero = __riscv_vfmv_v_f_f64m4(0.0, vector_length);
    vfloat64m4_t one = __riscv_vfmv_v_f_f64m4(1.0, vector_length);
    haversine_term = __riscv_vfmax_vv_f64m4(zero, haversine_term, vector_length);
    haversine_term = __riscv_vfmin_vv_f64m4(one, haversine_term, vector_length);

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    vfloat64m4_t sqrt_haversine = __riscv_vfsqrt_v_f64m4(haversine_term, vector_length);
    vfloat64m4_t complement = __riscv_vfsub_vv_f64m4(one, haversine_term, vector_length);
    vfloat64m4_t sqrt_complement = __riscv_vfsqrt_v_f64m4(complement, vector_length);
    vfloat64m4_t central_angle = nk_f64m4_atan2_rvv_(sqrt_haversine, sqrt_complement, vector_length);
    central_angle = __riscv_vfmul_vf_f64m4(central_angle, 2.0, vector_length);

    // distance = R * c
    vfloat64m4_t distances = __riscv_vfmul_vf_f64m4(central_angle, NK_EARTH_MEDIATORIAL_RADIUS, vector_length);
    __riscv_vse64_v_f64m4(results, distances, vector_length);
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

    vfloat32m4_t lat1 = __riscv_vle32_v_f32m4(a_lats, vector_length);
    vfloat32m4_t lon1 = __riscv_vle32_v_f32m4(a_lons, vector_length);
    vfloat32m4_t lat2 = __riscv_vle32_v_f32m4(b_lats, vector_length);
    vfloat32m4_t lon2 = __riscv_vle32_v_f32m4(b_lons, vector_length);

    vfloat32m4_t dlat = __riscv_vfsub_vv_f32m4(lat2, lat1, vector_length);
    vfloat32m4_t dlon = __riscv_vfsub_vv_f32m4(lon2, lon1, vector_length);

    // sin(dlat/2) and sin(dlon/2)
    vfloat32m4_t half_dlat = __riscv_vfmul_vf_f32m4(dlat, 0.5f, vector_length);
    vfloat32m4_t half_dlon = __riscv_vfmul_vf_f32m4(dlon, 0.5f, vector_length);
    vfloat32m4_t sin_half_dlat = nk_f32m4_sin_rvv_(half_dlat, vector_length);
    vfloat32m4_t sin_half_dlon = nk_f32m4_sin_rvv_(half_dlon, vector_length);

    // sin^2(dlat/2) and sin^2(dlon/2)
    vfloat32m4_t sin_sq_half_dlat = __riscv_vfmul_vv_f32m4(sin_half_dlat, sin_half_dlat, vector_length);
    vfloat32m4_t sin_sq_half_dlon = __riscv_vfmul_vv_f32m4(sin_half_dlon, sin_half_dlon, vector_length);

    // cos(lat1) * cos(lat2)
    vfloat32m4_t cos_lat1 = nk_f32m4_cos_rvv_(lat1, vector_length);
    vfloat32m4_t cos_lat2 = nk_f32m4_cos_rvv_(lat2, vector_length);
    vfloat32m4_t cos_product = __riscv_vfmul_vv_f32m4(cos_lat1, cos_lat2, vector_length);

    // a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
    vfloat32m4_t haversine_term = __riscv_vfmadd_vv_f32m4(cos_product, sin_sq_half_dlon, sin_sq_half_dlat,
                                                          vector_length);

    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    vfloat32m4_t zero = __riscv_vfmv_v_f_f32m4(0.0f, vector_length);
    vfloat32m4_t one = __riscv_vfmv_v_f_f32m4(1.0f, vector_length);
    haversine_term = __riscv_vfmax_vv_f32m4(zero, haversine_term, vector_length);
    haversine_term = __riscv_vfmin_vv_f32m4(one, haversine_term, vector_length);

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    vfloat32m4_t sqrt_haversine = __riscv_vfsqrt_v_f32m4(haversine_term, vector_length);
    vfloat32m4_t complement = __riscv_vfsub_vv_f32m4(one, haversine_term, vector_length);
    vfloat32m4_t sqrt_complement = __riscv_vfsqrt_v_f32m4(complement, vector_length);
    vfloat32m4_t central_angle = nk_f32m4_atan2_rvv_(sqrt_haversine, sqrt_complement, vector_length);
    central_angle = __riscv_vfmul_vf_f32m4(central_angle, 2.0f, vector_length);

    // distance = R * c
    vfloat32m4_t distances = __riscv_vfmul_vf_f32m4(central_angle, (nk_f32_t)NK_EARTH_MEDIATORIAL_RADIUS,
                                                    vector_length);
    __riscv_vse32_v_f32m4(results, distances, vector_length);
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

#pragma endregion - Haversine Distance

#pragma region - Vincenty Distance

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

    vfloat64m4_t lat1 = __riscv_vle64_v_f64m4(a_lats, vector_length);
    vfloat64m4_t lon1 = __riscv_vle64_v_f64m4(a_lons, vector_length);
    vfloat64m4_t lat2 = __riscv_vle64_v_f64m4(b_lats, vector_length);
    vfloat64m4_t lon2 = __riscv_vle64_v_f64m4(b_lons, vector_length);

    vfloat64m4_t const v_equatorial_radius = __riscv_vfmv_v_f_f64m4(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS,
                                                                    vector_length);
    vfloat64m4_t const v_polar_radius = __riscv_vfmv_v_f_f64m4(NK_EARTH_ELLIPSOID_POLAR_RADIUS, vector_length);
    nk_f64_t const flattening_scalar = 1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    vfloat64m4_t const v_flattening = __riscv_vfmv_v_f_f64m4(flattening_scalar, vector_length);
    vfloat64m4_t const v_convergence = __riscv_vfmv_v_f_f64m4(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64, vector_length);
    vfloat64m4_t const v_one = __riscv_vfmv_v_f_f64m4(1.0, vector_length);
    vfloat64m4_t const v_two = __riscv_vfmv_v_f_f64m4(2.0, vector_length);
    vfloat64m4_t const v_three = __riscv_vfmv_v_f_f64m4(3.0, vector_length);
    vfloat64m4_t const v_four = __riscv_vfmv_v_f_f64m4(4.0, vector_length);
    vfloat64m4_t const v_six = __riscv_vfmv_v_f_f64m4(6.0, vector_length);
    vfloat64m4_t const v_sixteen = __riscv_vfmv_v_f_f64m4(16.0, vector_length);
    vfloat64m4_t const v_epsilon = __riscv_vfmv_v_f_f64m4(1e-15, vector_length);
    vfloat64m4_t const v_zero = __riscv_vfmv_v_f_f64m4(0.0, vector_length);
    vfloat64m4_t const v_neg_one = __riscv_vfmv_v_f_f64m4(-1.0, vector_length);

    // Longitude difference
    vfloat64m4_t longitude_difference = __riscv_vfsub_vv_f64m4(lon2, lon1, vector_length);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    vfloat64m4_t one_minus_f = __riscv_vfsub_vv_f64m4(v_one, v_flattening, vector_length);
    vfloat64m4_t sin_lat1 = nk_f64m4_sin_rvv_(lat1, vector_length);
    vfloat64m4_t cos_lat1 = nk_f64m4_cos_rvv_(lat1, vector_length);
    vfloat64m4_t sin_lat2 = nk_f64m4_sin_rvv_(lat2, vector_length);
    vfloat64m4_t cos_lat2 = nk_f64m4_cos_rvv_(lat2, vector_length);
    vfloat64m4_t tan_first = __riscv_vfdiv_vv_f64m4(sin_lat1, cos_lat1, vector_length);
    vfloat64m4_t tan_second = __riscv_vfdiv_vv_f64m4(sin_lat2, cos_lat2, vector_length);
    vfloat64m4_t tan_reduced_first = __riscv_vfmul_vv_f64m4(one_minus_f, tan_first, vector_length);
    vfloat64m4_t tan_reduced_second = __riscv_vfmul_vv_f64m4(one_minus_f, tan_second, vector_length);

    // cos(U) = 1/sqrt(1 + tan^2(U)), sin(U) = tan(U) * cos(U)
    vfloat64m4_t tan_sq_first = __riscv_vfmadd_vv_f64m4(tan_reduced_first, tan_reduced_first, v_one, vector_length);
    vfloat64m4_t cos_reduced_first = __riscv_vfdiv_vv_f64m4(v_one, __riscv_vfsqrt_v_f64m4(tan_sq_first, vector_length),
                                                            vector_length);
    vfloat64m4_t sin_reduced_first = __riscv_vfmul_vv_f64m4(tan_reduced_first, cos_reduced_first, vector_length);

    vfloat64m4_t tan_sq_second = __riscv_vfmadd_vv_f64m4(tan_reduced_second, tan_reduced_second, v_one, vector_length);
    vfloat64m4_t cos_reduced_second = __riscv_vfdiv_vv_f64m4(
        v_one, __riscv_vfsqrt_v_f64m4(tan_sq_second, vector_length), vector_length);
    vfloat64m4_t sin_reduced_second = __riscv_vfmul_vv_f64m4(tan_reduced_second, cos_reduced_second, vector_length);

    // Initialize lambda and tracking variables
    vfloat64m4_t lambda = longitude_difference;
    vfloat64m4_t sin_angular_distance = v_zero;
    vfloat64m4_t cos_angular_distance = v_zero;
    vfloat64m4_t angular_distance = v_zero;
    vfloat64m4_t sin_azimuth = v_zero;
    vfloat64m4_t cos_squared_azimuth = v_zero;
    vfloat64m4_t cos_double_angular_midpoint = v_zero;

    // Track convergence and coincident points using masks
    // vbool16_t is the mask type for LMUL=4 with 64-bit elements (64/4 = 16)
    vbool16_t converged_mask_b16 = __riscv_vmfeq_vv_f64m4_b16(v_zero, v_one, vector_length); // all false
    vbool16_t coincident_mask_b16 = converged_mask_b16;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        if (__riscv_vcpop_m_b16(converged_mask_b16, vector_length) == vector_length) break;

        vfloat64m4_t sin_lambda = nk_f64m4_sin_rvv_(lambda, vector_length);
        vfloat64m4_t cos_lambda = nk_f64m4_cos_rvv_(lambda, vector_length);

        // sin^2(angular_distance) = (cos(U2)*sin(l))^2 + (cos(U1)*sin(U2) - sin(U1)*cos(U2)*cos(l))^2
        vfloat64m4_t cross_term = __riscv_vfmul_vv_f64m4(cos_reduced_second, sin_lambda, vector_length);
        vfloat64m4_t sin1_cos2_cosl = __riscv_vfmul_vv_f64m4(sin_reduced_first, cos_reduced_second, vector_length);
        sin1_cos2_cosl = __riscv_vfmul_vv_f64m4(sin1_cos2_cosl, cos_lambda, vector_length);
        vfloat64m4_t mixed_term = __riscv_vfmul_vv_f64m4(cos_reduced_first, sin_reduced_second, vector_length);
        mixed_term = __riscv_vfsub_vv_f64m4(mixed_term, sin1_cos2_cosl, vector_length);

        vfloat64m4_t sin_angular_dist_sq = __riscv_vfmul_vv_f64m4(cross_term, cross_term, vector_length);
        sin_angular_dist_sq = __riscv_vfmadd_vv_f64m4(mixed_term, mixed_term, sin_angular_dist_sq, vector_length);
        sin_angular_distance = __riscv_vfsqrt_v_f64m4(sin_angular_dist_sq, vector_length);

        // Check for coincident points (sin_angular_distance < epsilon)
        coincident_mask_b16 = __riscv_vmflt_vv_f64m4_b16(sin_angular_distance, v_epsilon, vector_length);

        // cos(angular_distance) = sin(U1)*sin(U2) + cos(U1)*cos(U2)*cos(l)
        vfloat64m4_t cos1_cos2 = __riscv_vfmul_vv_f64m4(cos_reduced_first, cos_reduced_second, vector_length);
        cos_angular_distance = __riscv_vfmul_vv_f64m4(sin_reduced_first, sin_reduced_second, vector_length);
        cos_angular_distance = __riscv_vfmadd_vv_f64m4(cos1_cos2, cos_lambda, cos_angular_distance, vector_length);

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f64m4_atan2_rvv_(sin_angular_distance, cos_angular_distance, vector_length);

        // sin(azimuth) = cos(U1)*cos(U2)*sin(l) / sin(angular_distance)
        // Avoid division by zero by substituting 1.0 for coincident lanes
        vfloat64m4_t safe_sin_angular = __riscv_vfmerge_vfm_f64m4(sin_angular_distance, 1.0, coincident_mask_b16,
                                                                  vector_length);
        vfloat64m4_t numerator = __riscv_vfmul_vv_f64m4(cos1_cos2, sin_lambda, vector_length);
        sin_azimuth = __riscv_vfdiv_vv_f64m4(numerator, safe_sin_angular, vector_length);
        cos_squared_azimuth = __riscv_vfnmsub_vv_f64m4(sin_azimuth, sin_azimuth, v_one, vector_length);

        // Handle equatorial case: cos^2(a) < epsilon
        vbool16_t equatorial_mask_b16 = __riscv_vmflt_vv_f64m4_b16(cos_squared_azimuth, v_epsilon, vector_length);
        vfloat64m4_t safe_cos_sq_azimuth = __riscv_vfmerge_vfm_f64m4(cos_squared_azimuth, 1.0, equatorial_mask_b16,
                                                                     vector_length);

        // cos(2sm) = cos(s) - 2*sin(U1)*sin(U2) / cos^2(a)
        vfloat64m4_t sin_product = __riscv_vfmul_vv_f64m4(sin_reduced_first, sin_reduced_second, vector_length);
        vfloat64m4_t two_sin_product = __riscv_vfmul_vv_f64m4(v_two, sin_product, vector_length);
        cos_double_angular_midpoint = __riscv_vfdiv_vv_f64m4(two_sin_product, safe_cos_sq_azimuth, vector_length);
        cos_double_angular_midpoint = __riscv_vfsub_vv_f64m4(cos_angular_distance, cos_double_angular_midpoint,
                                                             vector_length);
        // Set to zero for equatorial case
        cos_double_angular_midpoint = __riscv_vfmerge_vfm_f64m4(cos_double_angular_midpoint, 0.0, equatorial_mask_b16,
                                                                vector_length);

        // C = f/16 * cos^2(a) * (4 + f*(4 - 3*cos^2(a)))
        // inner = 4 - 3*cos^2(a)
        vfloat64m4_t inner_c = __riscv_vfnmsub_vv_f64m4(v_three, cos_squared_azimuth, v_four, vector_length);
        // 4 + f * inner_c
        vfloat64m4_t outer_c = __riscv_vfmadd_vv_f64m4(v_flattening, inner_c, v_four, vector_length);
        // f/16 * cos^2(a) * outer_c
        vfloat64m4_t correction_factor = __riscv_vfdiv_vv_f64m4(v_flattening, v_sixteen, vector_length);
        correction_factor = __riscv_vfmul_vv_f64m4(correction_factor, cos_squared_azimuth, vector_length);
        correction_factor = __riscv_vfmul_vv_f64m4(correction_factor, outer_c, vector_length);

        // lambda' = L + (1-C)*f*sin(a)*(s + C*sin(s)*(cos(2sm) + C*cos(s)*(-1 + 2*cos^2(2sm))))
        vfloat64m4_t cos_2sm_sq = __riscv_vfmul_vv_f64m4(cos_double_angular_midpoint, cos_double_angular_midpoint,
                                                         vector_length);
        // innermost = -1 + 2*cos^2(2sm)
        vfloat64m4_t innermost = __riscv_vfmadd_vv_f64m4(v_two, cos_2sm_sq, v_neg_one, vector_length);
        // middle = cos(2sm) + C*cos(s)*innermost
        vfloat64m4_t c_cos_s = __riscv_vfmul_vv_f64m4(correction_factor, cos_angular_distance, vector_length);
        vfloat64m4_t middle = __riscv_vfmadd_vv_f64m4(c_cos_s, innermost, cos_double_angular_midpoint, vector_length);
        // inner = C*sin(s)*middle
        vfloat64m4_t c_sin_s = __riscv_vfmul_vv_f64m4(correction_factor, sin_angular_distance, vector_length);
        vfloat64m4_t inner_val = __riscv_vfmul_vv_f64m4(c_sin_s, middle, vector_length);

        // (1-C)*f*sin_a*(s + inner)
        vfloat64m4_t one_minus_c = __riscv_vfsub_vv_f64m4(v_one, correction_factor, vector_length);
        vfloat64m4_t f_sin_a = __riscv_vfmul_vv_f64m4(v_flattening, sin_azimuth, vector_length);
        vfloat64m4_t s_plus_inner = __riscv_vfadd_vv_f64m4(angular_distance, inner_val, vector_length);
        vfloat64m4_t adjustment = __riscv_vfmul_vv_f64m4(one_minus_c, f_sin_a, vector_length);
        adjustment = __riscv_vfmul_vv_f64m4(adjustment, s_plus_inner, vector_length);
        vfloat64m4_t lambda_new = __riscv_vfadd_vv_f64m4(longitude_difference, adjustment, vector_length);

        // Check convergence: |lambda - lambda'| < threshold
        vfloat64m4_t lambda_diff = __riscv_vfsub_vv_f64m4(lambda_new, lambda, vector_length);
        // Absolute value via sign-bit clearing
        vfloat64m4_t lambda_diff_abs = __riscv_vfsgnjx_vv_f64m4(lambda_diff, lambda_diff, vector_length);
        vbool16_t newly_converged_b16 = __riscv_vmflt_vv_f64m4_b16(lambda_diff_abs, v_convergence, vector_length);
        converged_mask_b16 = __riscv_vmor_mm_b16(converged_mask_b16, newly_converged_b16, vector_length);

        // Only update lambda for non-converged lanes
        lambda = __riscv_vmerge_vvm_f64m4(lambda_new, lambda, converged_mask_b16, vector_length);
    }

    // Final distance calculation
    // u^2 = cos^2(a) * (a^2 - b^2) / b^2
    vfloat64m4_t a_sq = __riscv_vfmul_vv_f64m4(v_equatorial_radius, v_equatorial_radius, vector_length);
    vfloat64m4_t b_sq = __riscv_vfmul_vv_f64m4(v_polar_radius, v_polar_radius, vector_length);
    vfloat64m4_t a_sq_minus_b_sq = __riscv_vfsub_vv_f64m4(a_sq, b_sq, vector_length);
    vfloat64m4_t u_squared = __riscv_vfmul_vv_f64m4(cos_squared_azimuth, a_sq_minus_b_sq, vector_length);
    u_squared = __riscv_vfdiv_vv_f64m4(u_squared, b_sq, vector_length);

    // A = 1 + u^2/16384 * (4096 + u^2*(-768 + u^2*(320 - 175*u^2)))
    vfloat64m4_t series_a = __riscv_vfmul_vf_f64m4(u_squared, -175.0, vector_length);
    series_a = __riscv_vfadd_vf_f64m4(series_a, 320.0, vector_length);
    series_a = __riscv_vfmadd_vv_f64m4(u_squared, series_a, __riscv_vfmv_v_f_f64m4(-768.0, vector_length),
                                       vector_length);
    series_a = __riscv_vfmadd_vv_f64m4(u_squared, series_a, __riscv_vfmv_v_f_f64m4(4096.0, vector_length),
                                       vector_length);
    vfloat64m4_t u_sq_over_16384 = __riscv_vfmul_vf_f64m4(u_squared, 1.0 / 16384.0, vector_length);
    series_a = __riscv_vfmadd_vv_f64m4(u_sq_over_16384, series_a, v_one, vector_length);

    // B = u^2/1024 * (256 + u^2*(-128 + u^2*(74 - 47*u^2)))
    vfloat64m4_t series_b = __riscv_vfmul_vf_f64m4(u_squared, -47.0, vector_length);
    series_b = __riscv_vfadd_vf_f64m4(series_b, 74.0, vector_length);
    series_b = __riscv_vfmadd_vv_f64m4(u_squared, series_b, __riscv_vfmv_v_f_f64m4(-128.0, vector_length),
                                       vector_length);
    series_b = __riscv_vfmadd_vv_f64m4(u_squared, series_b, __riscv_vfmv_v_f_f64m4(256.0, vector_length),
                                       vector_length);
    vfloat64m4_t u_sq_over_1024 = __riscv_vfmul_vf_f64m4(u_squared, 1.0 / 1024.0, vector_length);
    series_b = __riscv_vfmul_vv_f64m4(u_sq_over_1024, series_b, vector_length);

    // Delta-sigma = B*sin(s)*(cos(2sm) + B/4*(cos(s)*(-1+2*cos^2(2sm)) -
    // B/6*cos(2sm)*(-3+4*sin^2(s))*(-3+4*cos^2(2sm))))
    vfloat64m4_t cos_2sm_sq = __riscv_vfmul_vv_f64m4(cos_double_angular_midpoint, cos_double_angular_midpoint,
                                                     vector_length);
    vfloat64m4_t sin_sq = __riscv_vfmul_vv_f64m4(sin_angular_distance, sin_angular_distance, vector_length);

    // term1 = cos(s) * (-1 + 2*cos^2(2sm))
    vfloat64m4_t term1 = __riscv_vfmadd_vv_f64m4(v_two, cos_2sm_sq, v_neg_one, vector_length);
    term1 = __riscv_vfmul_vv_f64m4(cos_angular_distance, term1, vector_length);

    // term2 = B/6 * cos(2sm) * (-3 + 4*sin^2(s)) * (-3 + 4*cos^2(2sm))
    vfloat64m4_t neg_three = __riscv_vfmv_v_f_f64m4(-3.0, vector_length);
    vfloat64m4_t factor_sin = __riscv_vfmadd_vv_f64m4(v_four, sin_sq, neg_three, vector_length);
    vfloat64m4_t factor_cos = __riscv_vfmadd_vv_f64m4(v_four, cos_2sm_sq, neg_three, vector_length);
    vfloat64m4_t b_over_6 = __riscv_vfdiv_vv_f64m4(series_b, v_six, vector_length);
    vfloat64m4_t term2 = __riscv_vfmul_vv_f64m4(b_over_6, cos_double_angular_midpoint, vector_length);
    term2 = __riscv_vfmul_vv_f64m4(term2, factor_sin, vector_length);
    term2 = __riscv_vfmul_vv_f64m4(term2, factor_cos, vector_length);

    // B/4 * (term1 - term2)
    vfloat64m4_t b_over_4 = __riscv_vfdiv_vv_f64m4(series_b, v_four, vector_length);
    vfloat64m4_t term1_minus_term2 = __riscv_vfsub_vv_f64m4(term1, term2, vector_length);
    vfloat64m4_t b4_bracket = __riscv_vfmul_vv_f64m4(b_over_4, term1_minus_term2, vector_length);

    // cos(2sm) + B/4*(...)
    vfloat64m4_t bracket = __riscv_vfadd_vv_f64m4(cos_double_angular_midpoint, b4_bracket, vector_length);

    // delta_sigma = B * sin(s) * bracket
    vfloat64m4_t delta_sigma = __riscv_vfmul_vv_f64m4(series_b, sin_angular_distance, vector_length);
    delta_sigma = __riscv_vfmul_vv_f64m4(delta_sigma, bracket, vector_length);

    // s = b * A * (sigma - delta_sigma)
    vfloat64m4_t sigma_minus_ds = __riscv_vfsub_vv_f64m4(angular_distance, delta_sigma, vector_length);
    vfloat64m4_t distances = __riscv_vfmul_vv_f64m4(v_polar_radius, series_a, vector_length);
    distances = __riscv_vfmul_vv_f64m4(distances, sigma_minus_ds, vector_length);

    // Set coincident points to zero
    distances = __riscv_vfmerge_vfm_f64m4(distances, 0.0, coincident_mask_b16, vector_length);

    __riscv_vse64_v_f64m4(results, distances, vector_length);
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

    vfloat32m4_t lat1 = __riscv_vle32_v_f32m4(a_lats, vector_length);
    vfloat32m4_t lon1 = __riscv_vle32_v_f32m4(a_lons, vector_length);
    vfloat32m4_t lat2 = __riscv_vle32_v_f32m4(b_lats, vector_length);
    vfloat32m4_t lon2 = __riscv_vle32_v_f32m4(b_lons, vector_length);

    vfloat32m4_t const v_equatorial_radius = __riscv_vfmv_v_f_f32m4((nk_f32_t)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS,
                                                                    vector_length);
    vfloat32m4_t const v_polar_radius = __riscv_vfmv_v_f_f32m4((nk_f32_t)NK_EARTH_ELLIPSOID_POLAR_RADIUS,
                                                               vector_length);
    nk_f32_t const flattening_scalar = 1.0f / (nk_f32_t)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    vfloat32m4_t const v_flattening = __riscv_vfmv_v_f_f32m4(flattening_scalar, vector_length);
    vfloat32m4_t const v_convergence = __riscv_vfmv_v_f_f32m4(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32, vector_length);
    vfloat32m4_t const v_one = __riscv_vfmv_v_f_f32m4(1.0f, vector_length);
    vfloat32m4_t const v_two = __riscv_vfmv_v_f_f32m4(2.0f, vector_length);
    vfloat32m4_t const v_three = __riscv_vfmv_v_f_f32m4(3.0f, vector_length);
    vfloat32m4_t const v_four = __riscv_vfmv_v_f_f32m4(4.0f, vector_length);
    vfloat32m4_t const v_six = __riscv_vfmv_v_f_f32m4(6.0f, vector_length);
    vfloat32m4_t const v_sixteen = __riscv_vfmv_v_f_f32m4(16.0f, vector_length);
    vfloat32m4_t const v_epsilon = __riscv_vfmv_v_f_f32m4(1e-7f, vector_length);
    vfloat32m4_t const v_zero = __riscv_vfmv_v_f_f32m4(0.0f, vector_length);
    vfloat32m4_t const v_neg_one = __riscv_vfmv_v_f_f32m4(-1.0f, vector_length);

    // Longitude difference
    vfloat32m4_t longitude_difference = __riscv_vfsub_vv_f32m4(lon2, lon1, vector_length);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    vfloat32m4_t one_minus_f = __riscv_vfsub_vv_f32m4(v_one, v_flattening, vector_length);
    vfloat32m4_t sin_lat1 = nk_f32m4_sin_rvv_(lat1, vector_length);
    vfloat32m4_t cos_lat1 = nk_f32m4_cos_rvv_(lat1, vector_length);
    vfloat32m4_t sin_lat2 = nk_f32m4_sin_rvv_(lat2, vector_length);
    vfloat32m4_t cos_lat2 = nk_f32m4_cos_rvv_(lat2, vector_length);
    vfloat32m4_t tan_first = __riscv_vfdiv_vv_f32m4(sin_lat1, cos_lat1, vector_length);
    vfloat32m4_t tan_second = __riscv_vfdiv_vv_f32m4(sin_lat2, cos_lat2, vector_length);
    vfloat32m4_t tan_reduced_first = __riscv_vfmul_vv_f32m4(one_minus_f, tan_first, vector_length);
    vfloat32m4_t tan_reduced_second = __riscv_vfmul_vv_f32m4(one_minus_f, tan_second, vector_length);

    // cos(U) = 1/sqrt(1 + tan^2(U)), sin(U) = tan(U) * cos(U)
    vfloat32m4_t tan_sq_first = __riscv_vfmadd_vv_f32m4(tan_reduced_first, tan_reduced_first, v_one, vector_length);
    vfloat32m4_t cos_reduced_first = __riscv_vfdiv_vv_f32m4(v_one, __riscv_vfsqrt_v_f32m4(tan_sq_first, vector_length),
                                                            vector_length);
    vfloat32m4_t sin_reduced_first = __riscv_vfmul_vv_f32m4(tan_reduced_first, cos_reduced_first, vector_length);

    vfloat32m4_t tan_sq_second = __riscv_vfmadd_vv_f32m4(tan_reduced_second, tan_reduced_second, v_one, vector_length);
    vfloat32m4_t cos_reduced_second = __riscv_vfdiv_vv_f32m4(
        v_one, __riscv_vfsqrt_v_f32m4(tan_sq_second, vector_length), vector_length);
    vfloat32m4_t sin_reduced_second = __riscv_vfmul_vv_f32m4(tan_reduced_second, cos_reduced_second, vector_length);

    // Initialize lambda and tracking variables
    vfloat32m4_t lambda = longitude_difference;
    vfloat32m4_t sin_angular_distance = v_zero;
    vfloat32m4_t cos_angular_distance = v_zero;
    vfloat32m4_t angular_distance = v_zero;
    vfloat32m4_t sin_azimuth = v_zero;
    vfloat32m4_t cos_squared_azimuth = v_zero;
    vfloat32m4_t cos_double_angular_midpoint = v_zero;

    // Track convergence and coincident points using masks
    // vbool8_t is the mask type for LMUL=4 with 32-bit elements (32/4 = 8)
    vbool8_t converged_mask_b8 = __riscv_vmfeq_vv_f32m4_b8(v_zero, v_one, vector_length); // all false
    vbool8_t coincident_mask_b8 = converged_mask_b8;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        if (__riscv_vcpop_m_b8(converged_mask_b8, vector_length) == vector_length) break;

        vfloat32m4_t sin_lambda = nk_f32m4_sin_rvv_(lambda, vector_length);
        vfloat32m4_t cos_lambda = nk_f32m4_cos_rvv_(lambda, vector_length);

        // sin^2(angular_distance) = (cos(U2)*sin(l))^2 + (cos(U1)*sin(U2) - sin(U1)*cos(U2)*cos(l))^2
        vfloat32m4_t cross_term = __riscv_vfmul_vv_f32m4(cos_reduced_second, sin_lambda, vector_length);
        vfloat32m4_t sin1_cos2_cosl = __riscv_vfmul_vv_f32m4(sin_reduced_first, cos_reduced_second, vector_length);
        sin1_cos2_cosl = __riscv_vfmul_vv_f32m4(sin1_cos2_cosl, cos_lambda, vector_length);
        vfloat32m4_t mixed_term = __riscv_vfmul_vv_f32m4(cos_reduced_first, sin_reduced_second, vector_length);
        mixed_term = __riscv_vfsub_vv_f32m4(mixed_term, sin1_cos2_cosl, vector_length);

        vfloat32m4_t sin_angular_dist_sq = __riscv_vfmul_vv_f32m4(cross_term, cross_term, vector_length);
        sin_angular_dist_sq = __riscv_vfmadd_vv_f32m4(mixed_term, mixed_term, sin_angular_dist_sq, vector_length);
        sin_angular_distance = __riscv_vfsqrt_v_f32m4(sin_angular_dist_sq, vector_length);

        // Check for coincident points (sin_angular_distance < epsilon)
        coincident_mask_b8 = __riscv_vmflt_vv_f32m4_b8(sin_angular_distance, v_epsilon, vector_length);

        // cos(angular_distance) = sin(U1)*sin(U2) + cos(U1)*cos(U2)*cos(l)
        vfloat32m4_t cos1_cos2 = __riscv_vfmul_vv_f32m4(cos_reduced_first, cos_reduced_second, vector_length);
        cos_angular_distance = __riscv_vfmul_vv_f32m4(sin_reduced_first, sin_reduced_second, vector_length);
        cos_angular_distance = __riscv_vfmadd_vv_f32m4(cos1_cos2, cos_lambda, cos_angular_distance, vector_length);

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f32m4_atan2_rvv_(sin_angular_distance, cos_angular_distance, vector_length);

        // sin(azimuth) = cos(U1)*cos(U2)*sin(l) / sin(angular_distance)
        // Avoid division by zero by substituting 1.0 for coincident lanes
        vfloat32m4_t safe_sin_angular = __riscv_vfmerge_vfm_f32m4(sin_angular_distance, 1.0f, coincident_mask_b8,
                                                                  vector_length);
        vfloat32m4_t numerator = __riscv_vfmul_vv_f32m4(cos1_cos2, sin_lambda, vector_length);
        sin_azimuth = __riscv_vfdiv_vv_f32m4(numerator, safe_sin_angular, vector_length);
        cos_squared_azimuth = __riscv_vfnmsub_vv_f32m4(sin_azimuth, sin_azimuth, v_one, vector_length);

        // Handle equatorial case: cos^2(a) < epsilon
        vbool8_t equatorial_mask_b8 = __riscv_vmflt_vv_f32m4_b8(cos_squared_azimuth, v_epsilon, vector_length);
        vfloat32m4_t safe_cos_sq_azimuth = __riscv_vfmerge_vfm_f32m4(cos_squared_azimuth, 1.0f, equatorial_mask_b8,
                                                                     vector_length);

        // cos(2sm) = cos(s) - 2*sin(U1)*sin(U2) / cos^2(a)
        vfloat32m4_t sin_product = __riscv_vfmul_vv_f32m4(sin_reduced_first, sin_reduced_second, vector_length);
        vfloat32m4_t two_sin_product = __riscv_vfmul_vv_f32m4(v_two, sin_product, vector_length);
        cos_double_angular_midpoint = __riscv_vfdiv_vv_f32m4(two_sin_product, safe_cos_sq_azimuth, vector_length);
        cos_double_angular_midpoint = __riscv_vfsub_vv_f32m4(cos_angular_distance, cos_double_angular_midpoint,
                                                             vector_length);
        // Set to zero for equatorial case
        cos_double_angular_midpoint = __riscv_vfmerge_vfm_f32m4(cos_double_angular_midpoint, 0.0f, equatorial_mask_b8,
                                                                vector_length);

        // C = f/16 * cos^2(a) * (4 + f*(4 - 3*cos^2(a)))
        vfloat32m4_t inner_c = __riscv_vfnmsub_vv_f32m4(v_three, cos_squared_azimuth, v_four, vector_length);
        vfloat32m4_t outer_c = __riscv_vfmadd_vv_f32m4(v_flattening, inner_c, v_four, vector_length);
        vfloat32m4_t correction_factor = __riscv_vfdiv_vv_f32m4(v_flattening, v_sixteen, vector_length);
        correction_factor = __riscv_vfmul_vv_f32m4(correction_factor, cos_squared_azimuth, vector_length);
        correction_factor = __riscv_vfmul_vv_f32m4(correction_factor, outer_c, vector_length);

        // lambda' = L + (1-C)*f*sin(a)*(s + C*sin(s)*(cos(2sm) + C*cos(s)*(-1 + 2*cos^2(2sm))))
        vfloat32m4_t cos_2sm_sq = __riscv_vfmul_vv_f32m4(cos_double_angular_midpoint, cos_double_angular_midpoint,
                                                         vector_length);
        vfloat32m4_t innermost = __riscv_vfmadd_vv_f32m4(v_two, cos_2sm_sq, v_neg_one, vector_length);
        vfloat32m4_t c_cos_s = __riscv_vfmul_vv_f32m4(correction_factor, cos_angular_distance, vector_length);
        vfloat32m4_t middle = __riscv_vfmadd_vv_f32m4(c_cos_s, innermost, cos_double_angular_midpoint, vector_length);
        vfloat32m4_t c_sin_s = __riscv_vfmul_vv_f32m4(correction_factor, sin_angular_distance, vector_length);
        vfloat32m4_t inner_val = __riscv_vfmul_vv_f32m4(c_sin_s, middle, vector_length);

        vfloat32m4_t one_minus_c = __riscv_vfsub_vv_f32m4(v_one, correction_factor, vector_length);
        vfloat32m4_t f_sin_a = __riscv_vfmul_vv_f32m4(v_flattening, sin_azimuth, vector_length);
        vfloat32m4_t s_plus_inner = __riscv_vfadd_vv_f32m4(angular_distance, inner_val, vector_length);
        vfloat32m4_t adjustment = __riscv_vfmul_vv_f32m4(one_minus_c, f_sin_a, vector_length);
        adjustment = __riscv_vfmul_vv_f32m4(adjustment, s_plus_inner, vector_length);
        vfloat32m4_t lambda_new = __riscv_vfadd_vv_f32m4(longitude_difference, adjustment, vector_length);

        // Check convergence: |lambda - lambda'| < threshold
        vfloat32m4_t lambda_diff = __riscv_vfsub_vv_f32m4(lambda_new, lambda, vector_length);
        vfloat32m4_t lambda_diff_abs = __riscv_vfsgnjx_vv_f32m4(lambda_diff, lambda_diff, vector_length);
        vbool8_t newly_converged_b8 = __riscv_vmflt_vv_f32m4_b8(lambda_diff_abs, v_convergence, vector_length);
        converged_mask_b8 = __riscv_vmor_mm_b8(converged_mask_b8, newly_converged_b8, vector_length);

        // Only update lambda for non-converged lanes
        lambda = __riscv_vmerge_vvm_f32m4(lambda_new, lambda, converged_mask_b8, vector_length);
    }

    // Final distance calculation
    // u^2 = cos^2(a) * (a^2 - b^2) / b^2
    vfloat32m4_t a_sq = __riscv_vfmul_vv_f32m4(v_equatorial_radius, v_equatorial_radius, vector_length);
    vfloat32m4_t b_sq = __riscv_vfmul_vv_f32m4(v_polar_radius, v_polar_radius, vector_length);
    vfloat32m4_t a_sq_minus_b_sq = __riscv_vfsub_vv_f32m4(a_sq, b_sq, vector_length);
    vfloat32m4_t u_squared = __riscv_vfmul_vv_f32m4(cos_squared_azimuth, a_sq_minus_b_sq, vector_length);
    u_squared = __riscv_vfdiv_vv_f32m4(u_squared, b_sq, vector_length);

    // A = 1 + u^2/16384 * (4096 + u^2*(-768 + u^2*(320 - 175*u^2)))
    vfloat32m4_t series_a = __riscv_vfmul_vf_f32m4(u_squared, -175.0f, vector_length);
    series_a = __riscv_vfadd_vf_f32m4(series_a, 320.0f, vector_length);
    series_a = __riscv_vfmadd_vv_f32m4(u_squared, series_a, __riscv_vfmv_v_f_f32m4(-768.0f, vector_length),
                                       vector_length);
    series_a = __riscv_vfmadd_vv_f32m4(u_squared, series_a, __riscv_vfmv_v_f_f32m4(4096.0f, vector_length),
                                       vector_length);
    vfloat32m4_t u_sq_over_16384 = __riscv_vfmul_vf_f32m4(u_squared, 1.0f / 16384.0f, vector_length);
    series_a = __riscv_vfmadd_vv_f32m4(u_sq_over_16384, series_a, v_one, vector_length);

    // B = u^2/1024 * (256 + u^2*(-128 + u^2*(74 - 47*u^2)))
    vfloat32m4_t series_b = __riscv_vfmul_vf_f32m4(u_squared, -47.0f, vector_length);
    series_b = __riscv_vfadd_vf_f32m4(series_b, 74.0f, vector_length);
    series_b = __riscv_vfmadd_vv_f32m4(u_squared, series_b, __riscv_vfmv_v_f_f32m4(-128.0f, vector_length),
                                       vector_length);
    series_b = __riscv_vfmadd_vv_f32m4(u_squared, series_b, __riscv_vfmv_v_f_f32m4(256.0f, vector_length),
                                       vector_length);
    vfloat32m4_t u_sq_over_1024 = __riscv_vfmul_vf_f32m4(u_squared, 1.0f / 1024.0f, vector_length);
    series_b = __riscv_vfmul_vv_f32m4(u_sq_over_1024, series_b, vector_length);

    // Delta-sigma calculation
    vfloat32m4_t cos_2sm_sq = __riscv_vfmul_vv_f32m4(cos_double_angular_midpoint, cos_double_angular_midpoint,
                                                     vector_length);
    vfloat32m4_t sin_sq = __riscv_vfmul_vv_f32m4(sin_angular_distance, sin_angular_distance, vector_length);

    // term1 = cos(s) * (-1 + 2*cos^2(2sm))
    vfloat32m4_t term1 = __riscv_vfmadd_vv_f32m4(v_two, cos_2sm_sq, v_neg_one, vector_length);
    term1 = __riscv_vfmul_vv_f32m4(cos_angular_distance, term1, vector_length);

    // term2 = B/6 * cos(2sm) * (-3 + 4*sin^2(s)) * (-3 + 4*cos^2(2sm))
    vfloat32m4_t neg_three = __riscv_vfmv_v_f_f32m4(-3.0f, vector_length);
    vfloat32m4_t factor_sin = __riscv_vfmadd_vv_f32m4(v_four, sin_sq, neg_three, vector_length);
    vfloat32m4_t factor_cos = __riscv_vfmadd_vv_f32m4(v_four, cos_2sm_sq, neg_three, vector_length);
    vfloat32m4_t b_over_6 = __riscv_vfdiv_vv_f32m4(series_b, v_six, vector_length);
    vfloat32m4_t term2 = __riscv_vfmul_vv_f32m4(b_over_6, cos_double_angular_midpoint, vector_length);
    term2 = __riscv_vfmul_vv_f32m4(term2, factor_sin, vector_length);
    term2 = __riscv_vfmul_vv_f32m4(term2, factor_cos, vector_length);

    // B/4 * (term1 - term2)
    vfloat32m4_t b_over_4 = __riscv_vfdiv_vv_f32m4(series_b, v_four, vector_length);
    vfloat32m4_t term1_minus_term2 = __riscv_vfsub_vv_f32m4(term1, term2, vector_length);
    vfloat32m4_t b4_bracket = __riscv_vfmul_vv_f32m4(b_over_4, term1_minus_term2, vector_length);

    // cos(2sm) + B/4*(...)
    vfloat32m4_t bracket = __riscv_vfadd_vv_f32m4(cos_double_angular_midpoint, b4_bracket, vector_length);

    // delta_sigma = B * sin(s) * bracket
    vfloat32m4_t delta_sigma = __riscv_vfmul_vv_f32m4(series_b, sin_angular_distance, vector_length);
    delta_sigma = __riscv_vfmul_vv_f32m4(delta_sigma, bracket, vector_length);

    // s = b * A * (sigma - delta_sigma)
    vfloat32m4_t sigma_minus_ds = __riscv_vfsub_vv_f32m4(angular_distance, delta_sigma, vector_length);
    vfloat32m4_t distances = __riscv_vfmul_vv_f32m4(v_polar_radius, series_a, vector_length);
    distances = __riscv_vfmul_vv_f32m4(distances, sigma_minus_ds, vector_length);

    // Set coincident points to zero
    distances = __riscv_vfmerge_vfm_f32m4(distances, 0.0f, coincident_mask_b8, vector_length);

    __riscv_vse32_v_f32m4(results, distances, vector_length);
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

#pragma endregion - Vincenty Distance

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_GEOSPATIAL_RVV_H
