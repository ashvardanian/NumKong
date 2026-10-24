/**
 *  @brief SIMD-accelerated Point Cloud Alignment for Haswell.
 *  @file include/numkong/mesh/haswell.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section haswell_mesh_instructions Key AVX2 Mesh Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm256_fmadd_ps             VFMADD (YMM, YMM, YMM)          5cy         0.5/cy      p01
 *      _mm256_hadd_ps              VHADDPS (YMM, YMM, YMM)         7cy         0.5/cy      p01+p5
 *      _mm256_permute2f128_ps      VPERM2F128 (YMM, YMM, YMM, I8)  3cy         1/cy        p5
 *      _mm256_extractf128_ps       VEXTRACTF128 (XMM, YMM, I8)     3cy         1/cy        p5
 *      _mm256_i32gather_ps         VGATHERDPS (YMM, M, YMM, YMM)   12cy        5/cy        p0+p23
 *
 *  Point cloud operations (centroid, covariance, Kabsch alignment) use gather instructions for
 *  stride-3 xyz deinterleaving. Multiple FMA accumulators hide the 5-cycle FMA latency. VHADDPS
 *  interleaves results across lanes, requiring additional shuffles for final scalar reduction.
 */
#ifndef NK_MESH_HASWELL_H
#define NK_MESH_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/dot/haswell.h"
#include "numkong/mesh/serial.h"
#include "numkong/reduce/haswell.h"  // `nk_reduce_add_f32x8_haswell_`
#include "numkong/spatial/haswell.h" // `nk_f32_sqrt_haswell`, `nk_f64_sqrt_haswell`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

/*  Deinterleave 24 floats (8 xyz triplets) into separate x, y, z vectors.
 *  Uses AVX2 gather instructions for clean stride-3 access.
 *
 *  Input: 24 contiguous floats [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
 *  Output: x[8], y[8], z[8] vectors
 */
NK_INTERNAL void nk_deinterleave_f32x8_haswell_(nk_f32_t const *ptr, __m256 *x_out, __m256 *y_out, __m256 *z_out) {
    // Gather indices: 0, 3, 6, 9, 12, 15, 18, 21 (stride 3)
    __m256i idx = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    *x_out = _mm256_i32gather_ps(ptr + 0, idx, 4);
    *y_out = _mm256_i32gather_ps(ptr + 1, idx, 4);
    *z_out = _mm256_i32gather_ps(ptr + 2, idx, 4);
}

/*  Deinterleave 12 f64 values (4 xyz triplets) into separate x, y, z vectors.
 *  Uses scalar extraction for simplicity as AVX2 lacks efficient stride-3 gather for f64.
 *
 *  Input: 12 contiguous f64 [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors
 */
NK_INTERNAL void nk_deinterleave_f64x4_haswell_(nk_f64_t const *ptr, __m256d *x_out, __m256d *y_out, __m256d *z_out) {
    nk_f64_t x0 = ptr[0], x1 = ptr[3], x2 = ptr[6], x3 = ptr[9];
    nk_f64_t y0 = ptr[1], y1 = ptr[4], y2 = ptr[7], y3 = ptr[10];
    nk_f64_t z0 = ptr[2], z1 = ptr[5], z2 = ptr[8], z3 = ptr[11];

    *x_out = _mm256_setr_pd(x0, x1, x2, x3);
    *y_out = _mm256_setr_pd(y0, y1, y2, y3);
    *z_out = _mm256_setr_pd(z0, z1, z2, z3);
}

/* Horizontal reduction helpers moved to reduce.h:
 * - nk_reduce_add_f32x8_haswell_
 * - nk_reduce_add_f64x4_haswell_
 */

NK_INTERNAL nk_f64_t nk_reduce_stable_f64x4_haswell_(__m256d values_f64x4) {
    nk_b256_vec_t values;
    values.ymm_pd = values_f64x4;
    nk_f64_t sum = 0.0, compensation = 0.0;
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[0]);
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[1]);
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[2]);
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[3]);
    return sum + compensation;
}

NK_INTERNAL void nk_rotation_from_svd_f64_haswell_(nk_f64_t const *svd_u, nk_f64_t const *svd_v, nk_f64_t *rotation) {
    nk_rotation_from_svd_f64_serial_(svd_u, svd_v, rotation);
}

NK_INTERNAL void nk_accumulate_square_f64x4_haswell_(__m256d *sum_f64x4, __m256d *compensation_f64x4,
                                                     __m256d values_f64x4) {
    __m256d product_f64x4 = _mm256_mul_pd(values_f64x4, values_f64x4);
    __m256d product_error_f64x4 = _mm256_fmsub_pd(values_f64x4, values_f64x4, product_f64x4);
    __m256d tentative_sum_f64x4 = _mm256_add_pd(*sum_f64x4, product_f64x4);
    __m256d virtual_addend_f64x4 = _mm256_sub_pd(tentative_sum_f64x4, *sum_f64x4);
    __m256d sum_error_f64x4 = _mm256_add_pd(
        _mm256_sub_pd(*sum_f64x4, _mm256_sub_pd(tentative_sum_f64x4, virtual_addend_f64x4)),
        _mm256_sub_pd(product_f64x4, virtual_addend_f64x4));
    *sum_f64x4 = tentative_sum_f64x4;
    *compensation_f64x4 = _mm256_add_pd(*compensation_f64x4, _mm256_add_pd(sum_error_f64x4, product_error_f64x4));
}

/*  Compute sum of squared distances after applying rotation (and optional scale).
 *  Used by kabsch (scale=1.0) and umeyama (scale=computed_scale).
 *  Returns sum_squared, caller computes sqrt(sum_squared / n).
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f32_haswell_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                     nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                     nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                     nk_f64_t centroid_b_z) {
    __m256d scaled_rotation_x_x_f64x4 = _mm256_set1_pd(scale * r[0]);
    __m256d scaled_rotation_x_y_f64x4 = _mm256_set1_pd(scale * r[1]);
    __m256d scaled_rotation_x_z_f64x4 = _mm256_set1_pd(scale * r[2]);
    __m256d scaled_rotation_y_x_f64x4 = _mm256_set1_pd(scale * r[3]);
    __m256d scaled_rotation_y_y_f64x4 = _mm256_set1_pd(scale * r[4]);
    __m256d scaled_rotation_y_z_f64x4 = _mm256_set1_pd(scale * r[5]);
    __m256d scaled_rotation_z_x_f64x4 = _mm256_set1_pd(scale * r[6]);
    __m256d scaled_rotation_z_y_f64x4 = _mm256_set1_pd(scale * r[7]);
    __m256d scaled_rotation_z_z_f64x4 = _mm256_set1_pd(scale * r[8]);
    __m256d centroid_a_x_f64x4 = _mm256_set1_pd(centroid_a_x), centroid_a_y_f64x4 = _mm256_set1_pd(centroid_a_y);
    __m256d centroid_a_z_f64x4 = _mm256_set1_pd(centroid_a_z), centroid_b_x_f64x4 = _mm256_set1_pd(centroid_b_x);
    __m256d centroid_b_y_f64x4 = _mm256_set1_pd(centroid_b_y), centroid_b_z_f64x4 = _mm256_set1_pd(centroid_b_z);
    __m256d sum_squared_f64x4 = _mm256_setzero_pd();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t index = 0;

    for (; index + 8 <= n; index += 8) {
        nk_deinterleave_f32x8_haswell_(a + index * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8),
            nk_deinterleave_f32x8_haswell_(b + index * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        __m256d a_x_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_f32x8));
        __m256d a_x_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_f32x8, 1));
        __m256d a_y_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_f32x8));
        __m256d a_y_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_f32x8, 1));
        __m256d a_z_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_f32x8));
        __m256d a_z_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_f32x8, 1));
        __m256d b_x_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_f32x8));
        __m256d b_x_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_f32x8, 1));
        __m256d b_y_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_f32x8));
        __m256d b_y_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_f32x8, 1));
        __m256d b_z_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_f32x8));
        __m256d b_z_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_f32x8, 1));

        __m256d centered_a_x_lower_f64x4 = _mm256_sub_pd(a_x_lower_f64x4, centroid_a_x_f64x4);
        __m256d centered_a_x_upper_f64x4 = _mm256_sub_pd(a_x_upper_f64x4, centroid_a_x_f64x4);
        __m256d centered_a_y_lower_f64x4 = _mm256_sub_pd(a_y_lower_f64x4, centroid_a_y_f64x4);
        __m256d centered_a_y_upper_f64x4 = _mm256_sub_pd(a_y_upper_f64x4, centroid_a_y_f64x4);
        __m256d centered_a_z_lower_f64x4 = _mm256_sub_pd(a_z_lower_f64x4, centroid_a_z_f64x4);
        __m256d centered_a_z_upper_f64x4 = _mm256_sub_pd(a_z_upper_f64x4, centroid_a_z_f64x4);
        __m256d centered_b_x_lower_f64x4 = _mm256_sub_pd(b_x_lower_f64x4, centroid_b_x_f64x4);
        __m256d centered_b_x_upper_f64x4 = _mm256_sub_pd(b_x_upper_f64x4, centroid_b_x_f64x4);
        __m256d centered_b_y_lower_f64x4 = _mm256_sub_pd(b_y_lower_f64x4, centroid_b_y_f64x4);
        __m256d centered_b_y_upper_f64x4 = _mm256_sub_pd(b_y_upper_f64x4, centroid_b_y_f64x4);
        __m256d centered_b_z_lower_f64x4 = _mm256_sub_pd(b_z_lower_f64x4, centroid_b_z_f64x4);
        __m256d centered_b_z_upper_f64x4 = _mm256_sub_pd(b_z_upper_f64x4, centroid_b_z_f64x4);

        __m256d rotated_a_x_lower_f64x4 = _mm256_fmadd_pd(
            scaled_rotation_x_z_f64x4, centered_a_z_lower_f64x4,
            _mm256_fmadd_pd(scaled_rotation_x_y_f64x4, centered_a_y_lower_f64x4,
                            _mm256_mul_pd(scaled_rotation_x_x_f64x4, centered_a_x_lower_f64x4)));
        __m256d rotated_a_x_upper_f64x4 = _mm256_fmadd_pd(
            scaled_rotation_x_z_f64x4, centered_a_z_upper_f64x4,
            _mm256_fmadd_pd(scaled_rotation_x_y_f64x4, centered_a_y_upper_f64x4,
                            _mm256_mul_pd(scaled_rotation_x_x_f64x4, centered_a_x_upper_f64x4)));
        __m256d rotated_a_y_lower_f64x4 = _mm256_fmadd_pd(
            scaled_rotation_y_z_f64x4, centered_a_z_lower_f64x4,
            _mm256_fmadd_pd(scaled_rotation_y_y_f64x4, centered_a_y_lower_f64x4,
                            _mm256_mul_pd(scaled_rotation_y_x_f64x4, centered_a_x_lower_f64x4)));
        __m256d rotated_a_y_upper_f64x4 = _mm256_fmadd_pd(
            scaled_rotation_y_z_f64x4, centered_a_z_upper_f64x4,
            _mm256_fmadd_pd(scaled_rotation_y_y_f64x4, centered_a_y_upper_f64x4,
                            _mm256_mul_pd(scaled_rotation_y_x_f64x4, centered_a_x_upper_f64x4)));
        __m256d rotated_a_z_lower_f64x4 = _mm256_fmadd_pd(
            scaled_rotation_z_z_f64x4, centered_a_z_lower_f64x4,
            _mm256_fmadd_pd(scaled_rotation_z_y_f64x4, centered_a_y_lower_f64x4,
                            _mm256_mul_pd(scaled_rotation_z_x_f64x4, centered_a_x_lower_f64x4)));
        __m256d rotated_a_z_upper_f64x4 = _mm256_fmadd_pd(
            scaled_rotation_z_z_f64x4, centered_a_z_upper_f64x4,
            _mm256_fmadd_pd(scaled_rotation_z_y_f64x4, centered_a_y_upper_f64x4,
                            _mm256_mul_pd(scaled_rotation_z_x_f64x4, centered_a_x_upper_f64x4)));

        __m256d delta_x_lower_f64x4 = _mm256_sub_pd(rotated_a_x_lower_f64x4, centered_b_x_lower_f64x4);
        __m256d delta_x_upper_f64x4 = _mm256_sub_pd(rotated_a_x_upper_f64x4, centered_b_x_upper_f64x4);
        __m256d delta_y_lower_f64x4 = _mm256_sub_pd(rotated_a_y_lower_f64x4, centered_b_y_lower_f64x4);
        __m256d delta_y_upper_f64x4 = _mm256_sub_pd(rotated_a_y_upper_f64x4, centered_b_y_upper_f64x4);
        __m256d delta_z_lower_f64x4 = _mm256_sub_pd(rotated_a_z_lower_f64x4, centered_b_z_lower_f64x4);
        __m256d delta_z_upper_f64x4 = _mm256_sub_pd(rotated_a_z_upper_f64x4, centered_b_z_upper_f64x4);

        __m256d batch_sum_squared_f64x4 = _mm256_add_pd(_mm256_mul_pd(delta_x_lower_f64x4, delta_x_lower_f64x4),
                                                        _mm256_mul_pd(delta_x_upper_f64x4, delta_x_upper_f64x4));
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_y_lower_f64x4, delta_y_lower_f64x4, batch_sum_squared_f64x4);
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_y_upper_f64x4, delta_y_upper_f64x4, batch_sum_squared_f64x4);
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_z_lower_f64x4, delta_z_lower_f64x4, batch_sum_squared_f64x4);
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_z_upper_f64x4, delta_z_upper_f64x4, batch_sum_squared_f64x4);
        sum_squared_f64x4 = _mm256_add_pd(sum_squared_f64x4, batch_sum_squared_f64x4);
    }

    nk_f64_t sum_squared = nk_reduce_add_f64x4_haswell_(sum_squared_f64x4);
    for (; index < n; ++index) {
        nk_f64_t centered_a_x = (nk_f64_t)a[index * 3 + 0] - centroid_a_x;
        nk_f64_t centered_a_y = (nk_f64_t)a[index * 3 + 1] - centroid_a_y;
        nk_f64_t centered_a_z = (nk_f64_t)a[index * 3 + 2] - centroid_a_z;
        nk_f64_t centered_b_x = (nk_f64_t)b[index * 3 + 0] - centroid_b_x;
        nk_f64_t centered_b_y = (nk_f64_t)b[index * 3 + 1] - centroid_b_y;
        nk_f64_t centered_b_z = (nk_f64_t)b[index * 3 + 2] - centroid_b_z;
        nk_f64_t rotated_a_x = scale * (r[0] * centered_a_x + r[1] * centered_a_y + r[2] * centered_a_z);
        nk_f64_t rotated_a_y = scale * (r[3] * centered_a_x + r[4] * centered_a_y + r[5] * centered_a_z);
        nk_f64_t rotated_a_z = scale * (r[6] * centered_a_x + r[7] * centered_a_y + r[8] * centered_a_z);
        nk_f64_t delta_x = rotated_a_x - centered_b_x, delta_y = rotated_a_y - centered_b_y,
                 delta_z = rotated_a_z - centered_b_z;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    return sum_squared;
}

/*  Compute sum of squared distances for f64 after applying rotation (and optional scale).
 *  Rotation matrix, scale and data are all f64 for full precision.
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_haswell_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                     nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                     nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                     nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m256d scaled_rotation_x_x_f64x4 = _mm256_set1_pd(scale * r[0]);
    __m256d scaled_rotation_x_y_f64x4 = _mm256_set1_pd(scale * r[1]);
    __m256d scaled_rotation_x_z_f64x4 = _mm256_set1_pd(scale * r[2]);
    __m256d scaled_rotation_y_x_f64x4 = _mm256_set1_pd(scale * r[3]);
    __m256d scaled_rotation_y_y_f64x4 = _mm256_set1_pd(scale * r[4]);
    __m256d scaled_rotation_y_z_f64x4 = _mm256_set1_pd(scale * r[5]);
    __m256d scaled_rotation_z_x_f64x4 = _mm256_set1_pd(scale * r[6]);
    __m256d scaled_rotation_z_y_f64x4 = _mm256_set1_pd(scale * r[7]);
    __m256d scaled_rotation_z_z_f64x4 = _mm256_set1_pd(scale * r[8]);

    // Broadcast centroids
    __m256d centroid_a_x_f64x4 = _mm256_set1_pd(centroid_a_x);
    __m256d centroid_a_y_f64x4 = _mm256_set1_pd(centroid_a_y);
    __m256d centroid_a_z_f64x4 = _mm256_set1_pd(centroid_a_z);
    __m256d centroid_b_x_f64x4 = _mm256_set1_pd(centroid_b_x);
    __m256d centroid_b_y_f64x4 = _mm256_set1_pd(centroid_b_y);
    __m256d centroid_b_z_f64x4 = _mm256_set1_pd(centroid_b_z);

    __m256d sum_squared_f64x4 = _mm256_setzero_pd();
    __m256d sum_squared_compensation_f64x4 = _mm256_setzero_pd();
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;
    nk_size_t j = 0;

    for (; j + 4 <= n; j += 4) {
        nk_deinterleave_f64x4_haswell_(a + j * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + j * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        // Center points
        __m256d pa_x_f64x4 = _mm256_sub_pd(a_x_f64x4, centroid_a_x_f64x4);
        __m256d pa_y_f64x4 = _mm256_sub_pd(a_y_f64x4, centroid_a_y_f64x4);
        __m256d pa_z_f64x4 = _mm256_sub_pd(a_z_f64x4, centroid_a_z_f64x4);
        __m256d pb_x_f64x4 = _mm256_sub_pd(b_x_f64x4, centroid_b_x_f64x4);
        __m256d pb_y_f64x4 = _mm256_sub_pd(b_y_f64x4, centroid_b_y_f64x4);
        __m256d pb_z_f64x4 = _mm256_sub_pd(b_z_f64x4, centroid_b_z_f64x4);

        // Rotate and scale: ra = scale * R * pa
        __m256d ra_x_f64x4 = _mm256_fmadd_pd(scaled_rotation_x_z_f64x4, pa_z_f64x4,
                                             _mm256_fmadd_pd(scaled_rotation_x_y_f64x4, pa_y_f64x4,
                                                             _mm256_mul_pd(scaled_rotation_x_x_f64x4, pa_x_f64x4)));
        __m256d ra_y_f64x4 = _mm256_fmadd_pd(scaled_rotation_y_z_f64x4, pa_z_f64x4,
                                             _mm256_fmadd_pd(scaled_rotation_y_y_f64x4, pa_y_f64x4,
                                                             _mm256_mul_pd(scaled_rotation_y_x_f64x4, pa_x_f64x4)));
        __m256d ra_z_f64x4 = _mm256_fmadd_pd(scaled_rotation_z_z_f64x4, pa_z_f64x4,
                                             _mm256_fmadd_pd(scaled_rotation_z_y_f64x4, pa_y_f64x4,
                                                             _mm256_mul_pd(scaled_rotation_z_x_f64x4, pa_x_f64x4)));

        // Delta and accumulate
        __m256d delta_x_f64x4 = _mm256_sub_pd(ra_x_f64x4, pb_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(ra_y_f64x4, pb_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(ra_z_f64x4, pb_z_f64x4);

        nk_accumulate_square_f64x4_haswell_(&sum_squared_f64x4, &sum_squared_compensation_f64x4, delta_x_f64x4);
        nk_accumulate_square_f64x4_haswell_(&sum_squared_f64x4, &sum_squared_compensation_f64x4, delta_y_f64x4);
        nk_accumulate_square_f64x4_haswell_(&sum_squared_f64x4, &sum_squared_compensation_f64x4, delta_z_f64x4);
    }

    nk_f64_t sum_squared = nk_dot_stable_sum_f64x4_haswell_(sum_squared_f64x4, sum_squared_compensation_f64x4);
    nk_f64_t sum_squared_compensation = 0.0;

    // Scalar tail
    for (; j < n; ++j) {
        nk_f64_t pa_x = a[j * 3 + 0] - centroid_a_x;
        nk_f64_t pa_y = a[j * 3 + 1] - centroid_a_y;
        nk_f64_t pa_z = a[j * 3 + 2] - centroid_a_z;
        nk_f64_t pb_x = b[j * 3 + 0] - centroid_b_x;
        nk_f64_t pb_y = b[j * 3 + 1] - centroid_b_y;
        nk_f64_t pb_z = b[j * 3 + 2] - centroid_b_z;

        nk_f64_t ra_x = scale * (r[0] * pa_x + r[1] * pa_y + r[2] * pa_z);
        nk_f64_t ra_y = scale * (r[3] * pa_x + r[4] * pa_y + r[5] * pa_z);
        nk_f64_t ra_z = scale * (r[6] * pa_x + r[7] * pa_y + r[8] * pa_z);

        nk_f64_t delta_x = ra_x - pb_x;
        nk_f64_t delta_y = ra_y - pb_y;
        nk_f64_t delta_z = ra_z - pb_z;
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_x);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_y);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_z);
    }

    return sum_squared + sum_squared_compensation;
}

NK_PUBLIC void nk_rmsd_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;

    __m256d sum_a_x_f64x4 = _mm256_setzero_pd(), sum_a_y_f64x4 = _mm256_setzero_pd();
    __m256d sum_a_z_f64x4 = _mm256_setzero_pd(), sum_b_x_f64x4 = _mm256_setzero_pd();
    __m256d sum_b_y_f64x4 = _mm256_setzero_pd(), sum_b_z_f64x4 = _mm256_setzero_pd();
    __m256d sum_squared_f64x4 = _mm256_setzero_pd();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t index = 0;

    for (; index + 8 <= n; index += 8) {
        nk_deinterleave_f32x8_haswell_(a + index * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8),
            nk_deinterleave_f32x8_haswell_(b + index * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        __m256d a_x_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_f32x8));
        __m256d a_x_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_f32x8, 1));
        __m256d a_y_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_f32x8));
        __m256d a_y_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_f32x8, 1));
        __m256d a_z_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_f32x8));
        __m256d a_z_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_f32x8, 1));
        __m256d b_x_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_f32x8));
        __m256d b_x_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_f32x8, 1));
        __m256d b_y_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_f32x8));
        __m256d b_y_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_f32x8, 1));
        __m256d b_z_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_f32x8));
        __m256d b_z_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_f32x8, 1));

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, _mm256_add_pd(a_x_lower_f64x4, a_x_upper_f64x4));
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, _mm256_add_pd(a_y_lower_f64x4, a_y_upper_f64x4));
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, _mm256_add_pd(a_z_lower_f64x4, a_z_upper_f64x4));
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, _mm256_add_pd(b_x_lower_f64x4, b_x_upper_f64x4));
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, _mm256_add_pd(b_y_lower_f64x4, b_y_upper_f64x4));
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, _mm256_add_pd(b_z_lower_f64x4, b_z_upper_f64x4));

        __m256d delta_x_lower_f64x4 = _mm256_sub_pd(a_x_lower_f64x4, b_x_lower_f64x4);
        __m256d delta_x_upper_f64x4 = _mm256_sub_pd(a_x_upper_f64x4, b_x_upper_f64x4);
        __m256d delta_y_lower_f64x4 = _mm256_sub_pd(a_y_lower_f64x4, b_y_lower_f64x4);
        __m256d delta_y_upper_f64x4 = _mm256_sub_pd(a_y_upper_f64x4, b_y_upper_f64x4);
        __m256d delta_z_lower_f64x4 = _mm256_sub_pd(a_z_lower_f64x4, b_z_lower_f64x4);
        __m256d delta_z_upper_f64x4 = _mm256_sub_pd(a_z_upper_f64x4, b_z_upper_f64x4);
        __m256d batch_sum_squared_f64x4 = _mm256_add_pd(_mm256_mul_pd(delta_x_lower_f64x4, delta_x_lower_f64x4),
                                                        _mm256_mul_pd(delta_x_upper_f64x4, delta_x_upper_f64x4));
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_y_lower_f64x4, delta_y_lower_f64x4, batch_sum_squared_f64x4);
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_y_upper_f64x4, delta_y_upper_f64x4, batch_sum_squared_f64x4);
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_z_lower_f64x4, delta_z_lower_f64x4, batch_sum_squared_f64x4);
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_z_upper_f64x4, delta_z_upper_f64x4, batch_sum_squared_f64x4);
        sum_squared_f64x4 = _mm256_add_pd(sum_squared_f64x4, batch_sum_squared_f64x4);
    }

    nk_f64_t total_a_x = nk_reduce_add_f64x4_haswell_(sum_a_x_f64x4);
    nk_f64_t total_a_y = nk_reduce_add_f64x4_haswell_(sum_a_y_f64x4);
    nk_f64_t total_a_z = nk_reduce_add_f64x4_haswell_(sum_a_z_f64x4);
    nk_f64_t total_b_x = nk_reduce_add_f64x4_haswell_(sum_b_x_f64x4);
    nk_f64_t total_b_y = nk_reduce_add_f64x4_haswell_(sum_b_y_f64x4);
    nk_f64_t total_b_z = nk_reduce_add_f64x4_haswell_(sum_b_z_f64x4);
    nk_f64_t sum_squared = nk_reduce_add_f64x4_haswell_(sum_squared_f64x4);

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        total_a_x += a_x, total_a_y += a_y, total_a_z += a_z;
        total_b_x += b_x, total_b_y += b_y, total_b_z += b_z;
        nk_f64_t delta_x = a_x - b_x, delta_y = a_y - b_y, delta_z = a_z - b_z;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = total_a_x * inv_n, centroid_a_y = total_a_y * inv_n, centroid_a_z = total_a_z * inv_n;
    nk_f64_t centroid_b_x = total_b_x * inv_n, centroid_b_y = total_b_y * inv_n, centroid_b_z = total_b_z * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    nk_f64_t mean_delta_x = centroid_a_x - centroid_b_x, mean_delta_y = centroid_a_y - centroid_b_y,
             mean_delta_z = centroid_a_z - centroid_b_z;
    nk_f64_t mean_delta_squared = mean_delta_x * mean_delta_x + mean_delta_y * mean_delta_y +
                                  mean_delta_z * mean_delta_z;
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n - mean_delta_squared);
}

NK_PUBLIC void nk_rmsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    // Accumulators for centroids and squared differences
    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;
    __m256d sum_squared_x_f64x4 = zeros_f64x4, sum_squared_y_f64x4 = zeros_f64x4, sum_squared_z_f64x4 = zeros_f64x4;

    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;
    nk_size_t i = 0;

    // Main loop with 2x unrolling
    for (; i + 8 <= n; i += 8) {
        // Iteration 0
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_f64x4);

        __m256d delta_x_f64x4 = _mm256_sub_pd(a_x_f64x4, b_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(a_y_f64x4, b_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(a_z_f64x4, b_z_f64x4);

        sum_squared_x_f64x4 = _mm256_fmadd_pd(delta_x_f64x4, delta_x_f64x4, sum_squared_x_f64x4);
        sum_squared_y_f64x4 = _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, sum_squared_y_f64x4);
        sum_squared_z_f64x4 = _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4, sum_squared_z_f64x4);

        // Iteration 1
        __m256d a_x1_f64x4, a_y1_f64x4, a_z1_f64x4, b_x1_f64x4, b_y1_f64x4, b_z1_f64x4;
        nk_deinterleave_f64x4_haswell_(a + (i + 4) * 3, &a_x1_f64x4, &a_y1_f64x4, &a_z1_f64x4);
        nk_deinterleave_f64x4_haswell_(b + (i + 4) * 3, &b_x1_f64x4, &b_y1_f64x4, &b_z1_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x1_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y1_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z1_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x1_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y1_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z1_f64x4);

        __m256d delta_x1_f64x4 = _mm256_sub_pd(a_x1_f64x4, b_x1_f64x4);
        __m256d delta_y1_f64x4 = _mm256_sub_pd(a_y1_f64x4, b_y1_f64x4);
        __m256d delta_z1_f64x4 = _mm256_sub_pd(a_z1_f64x4, b_z1_f64x4);

        sum_squared_x_f64x4 = _mm256_fmadd_pd(delta_x1_f64x4, delta_x1_f64x4, sum_squared_x_f64x4);
        sum_squared_y_f64x4 = _mm256_fmadd_pd(delta_y1_f64x4, delta_y1_f64x4, sum_squared_y_f64x4);
        sum_squared_z_f64x4 = _mm256_fmadd_pd(delta_z1_f64x4, delta_z1_f64x4, sum_squared_z_f64x4);
    }

    // Handle 4-point remainder
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_f64x4);

        __m256d delta_x_f64x4 = _mm256_sub_pd(a_x_f64x4, b_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(a_y_f64x4, b_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(a_z_f64x4, b_z_f64x4);

        sum_squared_x_f64x4 = _mm256_fmadd_pd(delta_x_f64x4, delta_x_f64x4, sum_squared_x_f64x4);
        sum_squared_y_f64x4 = _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, sum_squared_y_f64x4);
        sum_squared_z_f64x4 = _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4, sum_squared_z_f64x4);
    }

    // Reduce vectors to scalars
    nk_f64_t total_ax = nk_reduce_stable_f64x4_haswell_(sum_a_x_f64x4), total_ax_compensation = 0.0;
    nk_f64_t total_ay = nk_reduce_stable_f64x4_haswell_(sum_a_y_f64x4), total_ay_compensation = 0.0;
    nk_f64_t total_az = nk_reduce_stable_f64x4_haswell_(sum_a_z_f64x4), total_az_compensation = 0.0;
    nk_f64_t total_bx = nk_reduce_stable_f64x4_haswell_(sum_b_x_f64x4), total_bx_compensation = 0.0;
    nk_f64_t total_by = nk_reduce_stable_f64x4_haswell_(sum_b_y_f64x4), total_by_compensation = 0.0;
    nk_f64_t total_bz = nk_reduce_stable_f64x4_haswell_(sum_b_z_f64x4), total_bz_compensation = 0.0;
    nk_f64_t total_sq_x = nk_reduce_stable_f64x4_haswell_(sum_squared_x_f64x4), total_sq_x_compensation = 0.0;
    nk_f64_t total_sq_y = nk_reduce_stable_f64x4_haswell_(sum_squared_y_f64x4), total_sq_y_compensation = 0.0;
    nk_f64_t total_sq_z = nk_reduce_stable_f64x4_haswell_(sum_squared_z_f64x4), total_sq_z_compensation = 0.0;

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&total_ax, &total_ax_compensation, ax);
        nk_accumulate_sum_f64_(&total_ay, &total_ay_compensation, ay);
        nk_accumulate_sum_f64_(&total_az, &total_az_compensation, az);
        nk_accumulate_sum_f64_(&total_bx, &total_bx_compensation, bx);
        nk_accumulate_sum_f64_(&total_by, &total_by_compensation, by);
        nk_accumulate_sum_f64_(&total_bz, &total_bz_compensation, bz);
        nk_f64_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        nk_accumulate_square_f64_(&total_sq_x, &total_sq_x_compensation, delta_x);
        nk_accumulate_square_f64_(&total_sq_y, &total_sq_y_compensation, delta_y);
        nk_accumulate_square_f64_(&total_sq_z, &total_sq_z_compensation, delta_z);
    }

    total_ax += total_ax_compensation, total_ay += total_ay_compensation, total_az += total_az_compensation;
    total_bx += total_bx_compensation, total_by += total_by_compensation, total_bz += total_bz_compensation;
    total_sq_x += total_sq_x_compensation, total_sq_y += total_sq_y_compensation, total_sq_z += total_sq_z_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = total_ax * inv_n;
    nk_f64_t centroid_a_y = total_ay * inv_n;
    nk_f64_t centroid_a_z = total_az * inv_n;
    nk_f64_t centroid_b_x = total_bx * inv_n;
    nk_f64_t centroid_b_y = total_by * inv_n;
    nk_f64_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute RMSD
    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f64_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f64_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f64_sqrt_haswell(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (scale) *scale = 1.0f;
    __m256d sum_a_x_f64x4 = _mm256_setzero_pd(), sum_a_y_f64x4 = _mm256_setzero_pd();
    __m256d sum_a_z_f64x4 = _mm256_setzero_pd(), sum_b_x_f64x4 = _mm256_setzero_pd();
    __m256d sum_b_y_f64x4 = _mm256_setzero_pd(), sum_b_z_f64x4 = _mm256_setzero_pd();
    __m256d covariance_00_f64x4 = _mm256_setzero_pd(), covariance_01_f64x4 = _mm256_setzero_pd();
    __m256d covariance_02_f64x4 = _mm256_setzero_pd(), covariance_10_f64x4 = _mm256_setzero_pd();
    __m256d covariance_11_f64x4 = _mm256_setzero_pd(), covariance_12_f64x4 = _mm256_setzero_pd();
    __m256d covariance_20_f64x4 = _mm256_setzero_pd(), covariance_21_f64x4 = _mm256_setzero_pd();
    __m256d covariance_22_f64x4 = _mm256_setzero_pd();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t index = 0;

    for (; index + 8 <= n; index += 8) {
        nk_deinterleave_f32x8_haswell_(a + index * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8),
            nk_deinterleave_f32x8_haswell_(b + index * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);
        __m256d a_x_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_f32x8));
        __m256d a_x_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_f32x8, 1));
        __m256d a_y_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_f32x8));
        __m256d a_y_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_f32x8, 1));
        __m256d a_z_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_f32x8));
        __m256d a_z_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_f32x8, 1));
        __m256d b_x_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_f32x8));
        __m256d b_x_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_f32x8, 1));
        __m256d b_y_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_f32x8));
        __m256d b_y_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_f32x8, 1));
        __m256d b_z_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_f32x8));
        __m256d b_z_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_f32x8, 1));

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, _mm256_add_pd(a_x_lower_f64x4, a_x_upper_f64x4));
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, _mm256_add_pd(a_y_lower_f64x4, a_y_upper_f64x4));
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, _mm256_add_pd(a_z_lower_f64x4, a_z_upper_f64x4));
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, _mm256_add_pd(b_x_lower_f64x4, b_x_upper_f64x4));
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, _mm256_add_pd(b_y_lower_f64x4, b_y_upper_f64x4));
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, _mm256_add_pd(b_z_lower_f64x4, b_z_upper_f64x4));

        covariance_00_f64x4 = _mm256_add_pd(covariance_00_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_x_lower_f64x4, b_x_lower_f64x4),
                                                          _mm256_mul_pd(a_x_upper_f64x4, b_x_upper_f64x4)));
        covariance_01_f64x4 = _mm256_add_pd(covariance_01_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_x_lower_f64x4, b_y_lower_f64x4),
                                                          _mm256_mul_pd(a_x_upper_f64x4, b_y_upper_f64x4)));
        covariance_02_f64x4 = _mm256_add_pd(covariance_02_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_x_lower_f64x4, b_z_lower_f64x4),
                                                          _mm256_mul_pd(a_x_upper_f64x4, b_z_upper_f64x4)));
        covariance_10_f64x4 = _mm256_add_pd(covariance_10_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_y_lower_f64x4, b_x_lower_f64x4),
                                                          _mm256_mul_pd(a_y_upper_f64x4, b_x_upper_f64x4)));
        covariance_11_f64x4 = _mm256_add_pd(covariance_11_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_y_lower_f64x4, b_y_lower_f64x4),
                                                          _mm256_mul_pd(a_y_upper_f64x4, b_y_upper_f64x4)));
        covariance_12_f64x4 = _mm256_add_pd(covariance_12_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_y_lower_f64x4, b_z_lower_f64x4),
                                                          _mm256_mul_pd(a_y_upper_f64x4, b_z_upper_f64x4)));
        covariance_20_f64x4 = _mm256_add_pd(covariance_20_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_z_lower_f64x4, b_x_lower_f64x4),
                                                          _mm256_mul_pd(a_z_upper_f64x4, b_x_upper_f64x4)));
        covariance_21_f64x4 = _mm256_add_pd(covariance_21_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_z_lower_f64x4, b_y_lower_f64x4),
                                                          _mm256_mul_pd(a_z_upper_f64x4, b_y_upper_f64x4)));
        covariance_22_f64x4 = _mm256_add_pd(covariance_22_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_z_lower_f64x4, b_z_lower_f64x4),
                                                          _mm256_mul_pd(a_z_upper_f64x4, b_z_upper_f64x4)));
    }

    nk_f64_t sum_a_x = nk_reduce_add_f64x4_haswell_(sum_a_x_f64x4);
    nk_f64_t sum_a_y = nk_reduce_add_f64x4_haswell_(sum_a_y_f64x4);
    nk_f64_t sum_a_z = nk_reduce_add_f64x4_haswell_(sum_a_z_f64x4);
    nk_f64_t sum_b_x = nk_reduce_add_f64x4_haswell_(sum_b_x_f64x4);
    nk_f64_t sum_b_y = nk_reduce_add_f64x4_haswell_(sum_b_y_f64x4);
    nk_f64_t sum_b_z = nk_reduce_add_f64x4_haswell_(sum_b_z_f64x4);
    nk_f64_t h[9] = {
        nk_reduce_add_f64x4_haswell_(covariance_00_f64x4), nk_reduce_add_f64x4_haswell_(covariance_01_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_02_f64x4), nk_reduce_add_f64x4_haswell_(covariance_10_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_11_f64x4), nk_reduce_add_f64x4_haswell_(covariance_12_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_20_f64x4), nk_reduce_add_f64x4_haswell_(covariance_21_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_22_f64x4)};

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        h[0] += a_x * b_x, h[1] += a_x * b_y, h[2] += a_x * b_z;
        h[3] += a_y * b_x, h[4] += a_y * b_y, h[5] += a_y * b_z;
        h[6] += a_z * b_x, h[7] += a_z * b_y, h[8] += a_z * b_z;
    }

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    h[0] -= (nk_f64_t)n * centroid_a_x * centroid_b_x, h[1] -= (nk_f64_t)n * centroid_a_x * centroid_b_y,
        h[2] -= (nk_f64_t)n * centroid_a_x * centroid_b_z, h[3] -= (nk_f64_t)n * centroid_a_y * centroid_b_x,
        h[4] -= (nk_f64_t)n * centroid_a_y * centroid_b_y, h[5] -= (nk_f64_t)n * centroid_a_y * centroid_b_z,
        h[6] -= (nk_f64_t)n * centroid_a_z * centroid_b_x, h[7] -= (nk_f64_t)n * centroid_a_z * centroid_b_y,
        h[8] -= (nk_f64_t)n * centroid_a_z * centroid_b_z;

    nk_f64_t cross_covariance[9] = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9], r[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation)
        for (int j = 0; j != 9; ++j) rotation[j] = (nk_f32_t)r[j];
    nk_f64_t sum_squared = nk_transformed_ssd_f32_haswell_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_haswell(sum_squared / (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    // Accumulators for centroids
    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;

    // Accumulators for covariance matrix (sum of outer products)
    __m256d cov_xx_f64x4 = zeros_f64x4, cov_xy_f64x4 = zeros_f64x4, cov_xz_f64x4 = zeros_f64x4;
    __m256d cov_yx_f64x4 = zeros_f64x4, cov_yy_f64x4 = zeros_f64x4, cov_yz_f64x4 = zeros_f64x4;
    __m256d cov_zx_f64x4 = zeros_f64x4, cov_zy_f64x4 = zeros_f64x4, cov_zz_f64x4 = zeros_f64x4;

    nk_size_t i = 0;
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;

    // Fused single-pass
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_f64x4);

        cov_xx_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_x_f64x4, cov_xx_f64x4);
        cov_xy_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_y_f64x4, cov_xy_f64x4);
        cov_xz_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_z_f64x4, cov_xz_f64x4);
        cov_yx_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_x_f64x4, cov_yx_f64x4);
        cov_yy_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_y_f64x4, cov_yy_f64x4);
        cov_yz_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_z_f64x4, cov_yz_f64x4);
        cov_zx_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_x_f64x4, cov_zx_f64x4);
        cov_zy_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_y_f64x4, cov_zy_f64x4);
        cov_zz_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_z_f64x4, cov_zz_f64x4);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = nk_reduce_stable_f64x4_haswell_(sum_a_x_f64x4), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x4_haswell_(sum_a_y_f64x4), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x4_haswell_(sum_a_z_f64x4), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x4_haswell_(sum_b_x_f64x4), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x4_haswell_(sum_b_y_f64x4), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x4_haswell_(sum_b_z_f64x4), sum_b_z_compensation = 0.0;

    nk_f64_t covariance_x_x = nk_reduce_stable_f64x4_haswell_(cov_xx_f64x4), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x4_haswell_(cov_xy_f64x4), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x4_haswell_(cov_xz_f64x4), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x4_haswell_(cov_yx_f64x4), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x4_haswell_(cov_yy_f64x4), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x4_haswell_(cov_yz_f64x4), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x4_haswell_(cov_zx_f64x4), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x4_haswell_(cov_zy_f64x4), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x4_haswell_(cov_zz_f64x4), covariance_z_z_compensation = 0.0;

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax);
        nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay);
        nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx);
        nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by);
        nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx);
        nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by);
        nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx);
        nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by);
        nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx);
        nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by);
        nk_accumulate_product_f64_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n;
    nk_f64_t centroid_a_y = sum_a_y * inv_n;
    nk_f64_t centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n;
    nk_f64_t centroid_b_y = sum_b_y * inv_n;
    nk_f64_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation using f64 precision (svd_s is 9-element diagonal matrix)
    nk_f64_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_haswell_(svd_u, svd_v, r);

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_haswell_(svd_u, svd_v, r);
    }

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_haswell_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    __m256d sum_a_x_f64x4 = _mm256_setzero_pd(), sum_a_y_f64x4 = _mm256_setzero_pd();
    __m256d sum_a_z_f64x4 = _mm256_setzero_pd(), sum_b_x_f64x4 = _mm256_setzero_pd();
    __m256d sum_b_y_f64x4 = _mm256_setzero_pd(), sum_b_z_f64x4 = _mm256_setzero_pd();
    __m256d covariance_00_f64x4 = _mm256_setzero_pd(), covariance_01_f64x4 = _mm256_setzero_pd();
    __m256d covariance_02_f64x4 = _mm256_setzero_pd(), covariance_10_f64x4 = _mm256_setzero_pd();
    __m256d covariance_11_f64x4 = _mm256_setzero_pd(), covariance_12_f64x4 = _mm256_setzero_pd();
    __m256d covariance_20_f64x4 = _mm256_setzero_pd(), covariance_21_f64x4 = _mm256_setzero_pd();
    __m256d covariance_22_f64x4 = _mm256_setzero_pd(), variance_a_f64x4 = _mm256_setzero_pd();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t index = 0;

    for (; index + 8 <= n; index += 8) {
        nk_deinterleave_f32x8_haswell_(a + index * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8),
            nk_deinterleave_f32x8_haswell_(b + index * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);
        __m256d a_x_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_f32x8));
        __m256d a_x_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_f32x8, 1));
        __m256d a_y_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_f32x8));
        __m256d a_y_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_f32x8, 1));
        __m256d a_z_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_f32x8));
        __m256d a_z_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_f32x8, 1));
        __m256d b_x_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_f32x8));
        __m256d b_x_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_f32x8, 1));
        __m256d b_y_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_f32x8));
        __m256d b_y_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_f32x8, 1));
        __m256d b_z_lower_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_f32x8));
        __m256d b_z_upper_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_f32x8, 1));

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, _mm256_add_pd(a_x_lower_f64x4, a_x_upper_f64x4));
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, _mm256_add_pd(a_y_lower_f64x4, a_y_upper_f64x4));
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, _mm256_add_pd(a_z_lower_f64x4, a_z_upper_f64x4));
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, _mm256_add_pd(b_x_lower_f64x4, b_x_upper_f64x4));
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, _mm256_add_pd(b_y_lower_f64x4, b_y_upper_f64x4));
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, _mm256_add_pd(b_z_lower_f64x4, b_z_upper_f64x4));
        covariance_00_f64x4 = _mm256_add_pd(covariance_00_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_x_lower_f64x4, b_x_lower_f64x4),
                                                          _mm256_mul_pd(a_x_upper_f64x4, b_x_upper_f64x4)));
        covariance_01_f64x4 = _mm256_add_pd(covariance_01_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_x_lower_f64x4, b_y_lower_f64x4),
                                                          _mm256_mul_pd(a_x_upper_f64x4, b_y_upper_f64x4)));
        covariance_02_f64x4 = _mm256_add_pd(covariance_02_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_x_lower_f64x4, b_z_lower_f64x4),
                                                          _mm256_mul_pd(a_x_upper_f64x4, b_z_upper_f64x4)));
        covariance_10_f64x4 = _mm256_add_pd(covariance_10_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_y_lower_f64x4, b_x_lower_f64x4),
                                                          _mm256_mul_pd(a_y_upper_f64x4, b_x_upper_f64x4)));
        covariance_11_f64x4 = _mm256_add_pd(covariance_11_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_y_lower_f64x4, b_y_lower_f64x4),
                                                          _mm256_mul_pd(a_y_upper_f64x4, b_y_upper_f64x4)));
        covariance_12_f64x4 = _mm256_add_pd(covariance_12_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_y_lower_f64x4, b_z_lower_f64x4),
                                                          _mm256_mul_pd(a_y_upper_f64x4, b_z_upper_f64x4)));
        covariance_20_f64x4 = _mm256_add_pd(covariance_20_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_z_lower_f64x4, b_x_lower_f64x4),
                                                          _mm256_mul_pd(a_z_upper_f64x4, b_x_upper_f64x4)));
        covariance_21_f64x4 = _mm256_add_pd(covariance_21_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_z_lower_f64x4, b_y_lower_f64x4),
                                                          _mm256_mul_pd(a_z_upper_f64x4, b_y_upper_f64x4)));
        covariance_22_f64x4 = _mm256_add_pd(covariance_22_f64x4,
                                            _mm256_add_pd(_mm256_mul_pd(a_z_lower_f64x4, b_z_lower_f64x4),
                                                          _mm256_mul_pd(a_z_upper_f64x4, b_z_upper_f64x4)));
        variance_a_f64x4 = _mm256_add_pd(
            variance_a_f64x4,
            _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a_x_lower_f64x4, a_x_lower_f64x4),
                                        _mm256_mul_pd(a_x_upper_f64x4, a_x_upper_f64x4)),
                          _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a_y_lower_f64x4, a_y_lower_f64x4),
                                                      _mm256_mul_pd(a_y_upper_f64x4, a_y_upper_f64x4)),
                                        _mm256_add_pd(_mm256_mul_pd(a_z_lower_f64x4, a_z_lower_f64x4),
                                                      _mm256_mul_pd(a_z_upper_f64x4, a_z_upper_f64x4)))));
    }

    nk_f64_t sum_a_x = nk_reduce_add_f64x4_haswell_(sum_a_x_f64x4);
    nk_f64_t sum_a_y = nk_reduce_add_f64x4_haswell_(sum_a_y_f64x4);
    nk_f64_t sum_a_z = nk_reduce_add_f64x4_haswell_(sum_a_z_f64x4);
    nk_f64_t sum_b_x = nk_reduce_add_f64x4_haswell_(sum_b_x_f64x4);
    nk_f64_t sum_b_y = nk_reduce_add_f64x4_haswell_(sum_b_y_f64x4);
    nk_f64_t sum_b_z = nk_reduce_add_f64x4_haswell_(sum_b_z_f64x4);
    nk_f64_t h[9] = {
        nk_reduce_add_f64x4_haswell_(covariance_00_f64x4), nk_reduce_add_f64x4_haswell_(covariance_01_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_02_f64x4), nk_reduce_add_f64x4_haswell_(covariance_10_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_11_f64x4), nk_reduce_add_f64x4_haswell_(covariance_12_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_20_f64x4), nk_reduce_add_f64x4_haswell_(covariance_21_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_22_f64x4)};
    nk_f64_t variance_a = nk_reduce_add_f64x4_haswell_(variance_a_f64x4);

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        h[0] += a_x * b_x, h[1] += a_x * b_y, h[2] += a_x * b_z;
        h[3] += a_y * b_x, h[4] += a_y * b_y, h[5] += a_y * b_z;
        h[6] += a_z * b_x, h[7] += a_z * b_y, h[8] += a_z * b_z;
        variance_a += a_x * a_x + a_y * a_y + a_z * a_z;
    }

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    variance_a = variance_a * inv_n -
                 (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);
    h[0] -= (nk_f64_t)n * centroid_a_x * centroid_b_x, h[1] -= (nk_f64_t)n * centroid_a_x * centroid_b_y,
        h[2] -= (nk_f64_t)n * centroid_a_x * centroid_b_z, h[3] -= (nk_f64_t)n * centroid_a_y * centroid_b_x,
        h[4] -= (nk_f64_t)n * centroid_a_y * centroid_b_y, h[5] -= (nk_f64_t)n * centroid_a_y * centroid_b_z,
        h[6] -= (nk_f64_t)n * centroid_a_z * centroid_b_x, h[7] -= (nk_f64_t)n * centroid_a_z * centroid_b_y,
        h[8] -= (nk_f64_t)n * centroid_a_z * centroid_b_z;

    nk_f64_t cross_covariance[9] = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9], r[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    nk_f64_t det = nk_det3x3_f64_(r), sign_correction = det < 0 ? -1.0 : 1.0;
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    nk_f64_t applied_scale = (svd_s[0] + svd_s[4] + sign_correction * svd_s[8]) / ((nk_f64_t)n * variance_a);
    if (rotation)
        for (int j = 0; j != 9; ++j) rotation[j] = (nk_f32_t)r[j];
    if (scale) *scale = (nk_f32_t)applied_scale;
    *result = nk_f64_sqrt_haswell(nk_transformed_ssd_f32_haswell_(a, b, n, r, applied_scale, centroid_a_x, centroid_a_y,
                                                                  centroid_a_z, centroid_b_x, centroid_b_y,
                                                                  centroid_b_z) /
                                  (nk_f64_t)n);
}

NK_PUBLIC void nk_umeyama_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;
    __m256d cov_xx_f64x4 = zeros_f64x4, cov_xy_f64x4 = zeros_f64x4, cov_xz_f64x4 = zeros_f64x4;
    __m256d cov_yx_f64x4 = zeros_f64x4, cov_yy_f64x4 = zeros_f64x4, cov_yz_f64x4 = zeros_f64x4;
    __m256d cov_zx_f64x4 = zeros_f64x4, cov_zy_f64x4 = zeros_f64x4, cov_zz_f64x4 = zeros_f64x4;
    __m256d variance_a_f64x4 = zeros_f64x4;

    nk_size_t i = 0;
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;

    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_f64x4),
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_f64x4),
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_f64x4);

        cov_xx_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_x_f64x4, cov_xx_f64x4),
        cov_xy_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_y_f64x4, cov_xy_f64x4);
        cov_xz_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_z_f64x4, cov_xz_f64x4);
        cov_yx_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_x_f64x4, cov_yx_f64x4),
        cov_yy_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_y_f64x4, cov_yy_f64x4);
        cov_yz_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_z_f64x4, cov_yz_f64x4);
        cov_zx_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_x_f64x4, cov_zx_f64x4),
        cov_zy_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_y_f64x4, cov_zy_f64x4);
        cov_zz_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_z_f64x4, cov_zz_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_x_f64x4, a_x_f64x4, variance_a_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_y_f64x4, a_y_f64x4, variance_a_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_z_f64x4, a_z_f64x4, variance_a_f64x4);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = nk_reduce_stable_f64x4_haswell_(sum_a_x_f64x4), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x4_haswell_(sum_a_y_f64x4), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x4_haswell_(sum_a_z_f64x4), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x4_haswell_(sum_b_x_f64x4), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x4_haswell_(sum_b_y_f64x4), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x4_haswell_(sum_b_z_f64x4), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x4_haswell_(cov_xx_f64x4), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x4_haswell_(cov_xy_f64x4), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x4_haswell_(cov_xz_f64x4), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x4_haswell_(cov_yx_f64x4), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x4_haswell_(cov_yy_f64x4), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x4_haswell_(cov_yz_f64x4), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x4_haswell_(cov_zx_f64x4), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x4_haswell_(cov_zy_f64x4), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x4_haswell_(cov_zz_f64x4), covariance_z_z_compensation = 0.0;
    nk_f64_t variance_a_sum = nk_reduce_stable_f64x4_haswell_(variance_a_f64x4), variance_a_compensation = 0.0;

    // Scalar tail loop for remaining points
    for (; i < n; i++) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax);
        nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay);
        nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx);
        nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by);
        nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx);
        nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by);
        nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx);
        nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by);
        nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx);
        nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by);
        nk_accumulate_product_f64_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
        nk_accumulate_square_f64_(&variance_a_sum, &variance_a_compensation, ax);
        nk_accumulate_square_f64_(&variance_a_sum, &variance_a_compensation, ay);
        nk_accumulate_square_f64_(&variance_a_sum, &variance_a_compensation, az);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    variance_a_sum += variance_a_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    nk_f64_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    nk_f64_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = covariance_x_y - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = covariance_x_z - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = covariance_y_x - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = covariance_y_y - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = covariance_y_z - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = covariance_z_x - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = covariance_z_y - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = covariance_z_z - sum_a_z * sum_b_z * inv_n;

    // SVD using f64 for full precision (svd_s is 9-element diagonal matrix)
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_haswell_(svd_u, svd_v, r);

    // Scale factor: c = trace(D × S) / (n × variance(a))
    // svd_s diagonal: [0], [4], [8]
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
    nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_s[0], 1.0, svd_s[4], 1.0, svd_s[8], d3);
    nk_f64_t c = trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_haswell_(svd_u, svd_v, r);
    }

    /* Output rotation matrix */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    // Compute RMSD with scaling
    nk_f64_t sum_squared = nk_transformed_ssd_f64_haswell_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n);
}

/*  Deinterleave 8 f16 xyz triplets (24 f16 values) and convert to 3 x __m256 f32.
 *  Uses scalar extraction for clean stride-3 access, then F16C conversion.
 *
 *  Input: 24 contiguous f16 [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
 *  Output: x[8], y[8], z[8] vectors in f32
 */
NK_INTERNAL void nk_deinterleave_f16x8_to_f32x8_haswell_(nk_f16_t const *ptr, __m256 *x_out, __m256 *y_out,
                                                         __m256 *z_out) {
    // Extract x, y, z components with stride-3 access
    nk_b256_vec_t x_vec, y_vec, z_vec;
    x_vec.f16s[0] = ptr[0], x_vec.f16s[1] = ptr[3], x_vec.f16s[2] = ptr[6], x_vec.f16s[3] = ptr[9];
    x_vec.f16s[4] = ptr[12], x_vec.f16s[5] = ptr[15], x_vec.f16s[6] = ptr[18], x_vec.f16s[7] = ptr[21];
    y_vec.f16s[0] = ptr[1], y_vec.f16s[1] = ptr[4], y_vec.f16s[2] = ptr[7], y_vec.f16s[3] = ptr[10];
    y_vec.f16s[4] = ptr[13], y_vec.f16s[5] = ptr[16], y_vec.f16s[6] = ptr[19], y_vec.f16s[7] = ptr[22];
    z_vec.f16s[0] = ptr[2], z_vec.f16s[1] = ptr[5], z_vec.f16s[2] = ptr[8], z_vec.f16s[3] = ptr[11];
    z_vec.f16s[4] = ptr[14], z_vec.f16s[5] = ptr[17], z_vec.f16s[6] = ptr[20], z_vec.f16s[7] = ptr[23];
    // Convert f16 to f32 using F16C
    *x_out = _mm256_cvtph_ps(x_vec.xmms[0]);
    *y_out = _mm256_cvtph_ps(y_vec.xmms[0]);
    *z_out = _mm256_cvtph_ps(z_vec.xmms[0]);
}

/*  Deinterleave 8 bf16 xyz triplets (24 bf16 values) and convert to 3 x __m256 f32.
 *  Uses scalar extraction for clean stride-3 access, then bit-shift conversion.
 *
 *  Input: 24 contiguous bf16 [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
 *  Output: x[8], y[8], z[8] vectors in f32
 */
NK_INTERNAL void nk_deinterleave_bf16x8_to_f32x8_haswell_(nk_bf16_t const *ptr, __m256 *x_out, __m256 *y_out,
                                                          __m256 *z_out) {
    // Extract x, y, z components with stride-3 access
    nk_b256_vec_t x_vec, y_vec, z_vec;
    x_vec.bf16s[0] = ptr[0], x_vec.bf16s[1] = ptr[3], x_vec.bf16s[2] = ptr[6], x_vec.bf16s[3] = ptr[9];
    x_vec.bf16s[4] = ptr[12], x_vec.bf16s[5] = ptr[15], x_vec.bf16s[6] = ptr[18], x_vec.bf16s[7] = ptr[21];
    y_vec.bf16s[0] = ptr[1], y_vec.bf16s[1] = ptr[4], y_vec.bf16s[2] = ptr[7], y_vec.bf16s[3] = ptr[10];
    y_vec.bf16s[4] = ptr[13], y_vec.bf16s[5] = ptr[16], y_vec.bf16s[6] = ptr[19], y_vec.bf16s[7] = ptr[22];
    z_vec.bf16s[0] = ptr[2], z_vec.bf16s[1] = ptr[5], z_vec.bf16s[2] = ptr[8], z_vec.bf16s[3] = ptr[11];
    z_vec.bf16s[4] = ptr[14], z_vec.bf16s[5] = ptr[17], z_vec.bf16s[6] = ptr[20], z_vec.bf16s[7] = ptr[23];
    // Convert bf16 to f32 by left-shifting 16 bits
    *x_out = nk_bf16x8_to_f32x8_haswell_(x_vec.xmms[0]);
    *y_out = nk_bf16x8_to_f32x8_haswell_(y_vec.xmms[0]);
    *z_out = nk_bf16x8_to_f32x8_haswell_(z_vec.xmms[0]);
}

/*  Compute sum of squared distances for f16 data after applying rotation (and optional scale).
 *  Loads f16 data, converts to f32 during processing.
 *  Note: rotation matrix r is f32 (from SVD), scale and computation done in f32.
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_f16_haswell_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                     nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                     nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                     nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m256 scaled_rotation_x_x_f32x8 = _mm256_set1_ps(scale * r[0]);
    __m256 scaled_rotation_x_y_f32x8 = _mm256_set1_ps(scale * r[1]);
    __m256 scaled_rotation_x_z_f32x8 = _mm256_set1_ps(scale * r[2]);
    __m256 scaled_rotation_y_x_f32x8 = _mm256_set1_ps(scale * r[3]);
    __m256 scaled_rotation_y_y_f32x8 = _mm256_set1_ps(scale * r[4]);
    __m256 scaled_rotation_y_z_f32x8 = _mm256_set1_ps(scale * r[5]);
    __m256 scaled_rotation_z_x_f32x8 = _mm256_set1_ps(scale * r[6]);
    __m256 scaled_rotation_z_y_f32x8 = _mm256_set1_ps(scale * r[7]);
    __m256 scaled_rotation_z_z_f32x8 = _mm256_set1_ps(scale * r[8]);

    // Broadcast centroids
    __m256 centroid_a_x_f32x8 = _mm256_set1_ps(centroid_a_x);
    __m256 centroid_a_y_f32x8 = _mm256_set1_ps(centroid_a_y);
    __m256 centroid_a_z_f32x8 = _mm256_set1_ps(centroid_a_z);
    __m256 centroid_b_x_f32x8 = _mm256_set1_ps(centroid_b_x);
    __m256 centroid_b_y_f32x8 = _mm256_set1_ps(centroid_b_y);
    __m256 centroid_b_z_f32x8 = _mm256_set1_ps(centroid_b_z);

    __m256 sum_squared_f32x8 = _mm256_setzero_ps();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t j = 0;

    for (; j + 8 <= n; j += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + j * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + j * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Center points
        __m256 pa_x_f32x8 = _mm256_sub_ps(a_x_f32x8, centroid_a_x_f32x8);
        __m256 pa_y_f32x8 = _mm256_sub_ps(a_y_f32x8, centroid_a_y_f32x8);
        __m256 pa_z_f32x8 = _mm256_sub_ps(a_z_f32x8, centroid_a_z_f32x8);
        __m256 pb_x_f32x8 = _mm256_sub_ps(b_x_f32x8, centroid_b_x_f32x8);
        __m256 pb_y_f32x8 = _mm256_sub_ps(b_y_f32x8, centroid_b_y_f32x8);
        __m256 pb_z_f32x8 = _mm256_sub_ps(b_z_f32x8, centroid_b_z_f32x8);

        // Rotate and scale: ra = scale * R * pa
        __m256 ra_x_f32x8 = _mm256_fmadd_ps(scaled_rotation_x_z_f32x8, pa_z_f32x8,
                                            _mm256_fmadd_ps(scaled_rotation_x_y_f32x8, pa_y_f32x8,
                                                            _mm256_mul_ps(scaled_rotation_x_x_f32x8, pa_x_f32x8)));
        __m256 ra_y_f32x8 = _mm256_fmadd_ps(scaled_rotation_y_z_f32x8, pa_z_f32x8,
                                            _mm256_fmadd_ps(scaled_rotation_y_y_f32x8, pa_y_f32x8,
                                                            _mm256_mul_ps(scaled_rotation_y_x_f32x8, pa_x_f32x8)));
        __m256 ra_z_f32x8 = _mm256_fmadd_ps(scaled_rotation_z_z_f32x8, pa_z_f32x8,
                                            _mm256_fmadd_ps(scaled_rotation_z_y_f32x8, pa_y_f32x8,
                                                            _mm256_mul_ps(scaled_rotation_z_x_f32x8, pa_x_f32x8)));

        // Delta and accumulate
        __m256 delta_x_f32x8 = _mm256_sub_ps(ra_x_f32x8, pb_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(ra_y_f32x8, pb_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(ra_z_f32x8, pb_z_f32x8);

        sum_squared_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_f32x8);
    }

    nk_f32_t sum_squared = nk_reduce_add_f32x8_haswell_(sum_squared_f32x8);

    // Scalar tail
    for (; j < n; ++j) {
        nk_f32_t a_x_f32, a_y_f32, a_z_f32, b_x_f32, b_y_f32, b_z_f32;
        nk_f16_to_f32_haswell(&a[j * 3 + 0], &a_x_f32);
        nk_f16_to_f32_haswell(&a[j * 3 + 1], &a_y_f32);
        nk_f16_to_f32_haswell(&a[j * 3 + 2], &a_z_f32);
        nk_f16_to_f32_haswell(&b[j * 3 + 0], &b_x_f32);
        nk_f16_to_f32_haswell(&b[j * 3 + 1], &b_y_f32);
        nk_f16_to_f32_haswell(&b[j * 3 + 2], &b_z_f32);

        nk_f32_t pa_x = a_x_f32 - centroid_a_x;
        nk_f32_t pa_y = a_y_f32 - centroid_a_y;
        nk_f32_t pa_z = a_z_f32 - centroid_a_z;
        nk_f32_t pb_x = b_x_f32 - centroid_b_x;
        nk_f32_t pb_y = b_y_f32 - centroid_b_y;
        nk_f32_t pb_z = b_z_f32 - centroid_b_z;

        nk_f32_t ra_x = scale * (r[0] * pa_x + r[1] * pa_y + r[2] * pa_z);
        nk_f32_t ra_y = scale * (r[3] * pa_x + r[4] * pa_y + r[5] * pa_z);
        nk_f32_t ra_z = scale * (r[6] * pa_x + r[7] * pa_y + r[8] * pa_z);

        nk_f32_t delta_x = ra_x - pb_x;
        nk_f32_t delta_y = ra_y - pb_y;
        nk_f32_t delta_z = ra_z - pb_z;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    return sum_squared;
}

/*  Compute sum of squared distances for bf16 data after applying rotation (and optional scale).
 *  Loads bf16 data, converts to f32 during processing.
 *  Note: rotation matrix r is f32 (from SVD), scale and computation done in f32.
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_bf16_haswell_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                      nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                      nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                      nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m256 scaled_rotation_x_x_f32x8 = _mm256_set1_ps(scale * r[0]);
    __m256 scaled_rotation_x_y_f32x8 = _mm256_set1_ps(scale * r[1]);
    __m256 scaled_rotation_x_z_f32x8 = _mm256_set1_ps(scale * r[2]);
    __m256 scaled_rotation_y_x_f32x8 = _mm256_set1_ps(scale * r[3]);
    __m256 scaled_rotation_y_y_f32x8 = _mm256_set1_ps(scale * r[4]);
    __m256 scaled_rotation_y_z_f32x8 = _mm256_set1_ps(scale * r[5]);
    __m256 scaled_rotation_z_x_f32x8 = _mm256_set1_ps(scale * r[6]);
    __m256 scaled_rotation_z_y_f32x8 = _mm256_set1_ps(scale * r[7]);
    __m256 scaled_rotation_z_z_f32x8 = _mm256_set1_ps(scale * r[8]);

    // Broadcast centroids
    __m256 centroid_a_x_f32x8 = _mm256_set1_ps(centroid_a_x);
    __m256 centroid_a_y_f32x8 = _mm256_set1_ps(centroid_a_y);
    __m256 centroid_a_z_f32x8 = _mm256_set1_ps(centroid_a_z);
    __m256 centroid_b_x_f32x8 = _mm256_set1_ps(centroid_b_x);
    __m256 centroid_b_y_f32x8 = _mm256_set1_ps(centroid_b_y);
    __m256 centroid_b_z_f32x8 = _mm256_set1_ps(centroid_b_z);

    __m256 sum_squared_f32x8 = _mm256_setzero_ps();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t j = 0;

    for (; j + 8 <= n; j += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + j * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + j * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Center points
        __m256 pa_x_f32x8 = _mm256_sub_ps(a_x_f32x8, centroid_a_x_f32x8);
        __m256 pa_y_f32x8 = _mm256_sub_ps(a_y_f32x8, centroid_a_y_f32x8);
        __m256 pa_z_f32x8 = _mm256_sub_ps(a_z_f32x8, centroid_a_z_f32x8);
        __m256 pb_x_f32x8 = _mm256_sub_ps(b_x_f32x8, centroid_b_x_f32x8);
        __m256 pb_y_f32x8 = _mm256_sub_ps(b_y_f32x8, centroid_b_y_f32x8);
        __m256 pb_z_f32x8 = _mm256_sub_ps(b_z_f32x8, centroid_b_z_f32x8);

        // Rotate and scale: ra = scale * R * pa
        __m256 ra_x_f32x8 = _mm256_fmadd_ps(scaled_rotation_x_z_f32x8, pa_z_f32x8,
                                            _mm256_fmadd_ps(scaled_rotation_x_y_f32x8, pa_y_f32x8,
                                                            _mm256_mul_ps(scaled_rotation_x_x_f32x8, pa_x_f32x8)));
        __m256 ra_y_f32x8 = _mm256_fmadd_ps(scaled_rotation_y_z_f32x8, pa_z_f32x8,
                                            _mm256_fmadd_ps(scaled_rotation_y_y_f32x8, pa_y_f32x8,
                                                            _mm256_mul_ps(scaled_rotation_y_x_f32x8, pa_x_f32x8)));
        __m256 ra_z_f32x8 = _mm256_fmadd_ps(scaled_rotation_z_z_f32x8, pa_z_f32x8,
                                            _mm256_fmadd_ps(scaled_rotation_z_y_f32x8, pa_y_f32x8,
                                                            _mm256_mul_ps(scaled_rotation_z_x_f32x8, pa_x_f32x8)));

        // Delta and accumulate
        __m256 delta_x_f32x8 = _mm256_sub_ps(ra_x_f32x8, pb_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(ra_y_f32x8, pb_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(ra_z_f32x8, pb_z_f32x8);

        sum_squared_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_f32x8);
    }

    nk_f32_t sum_squared = nk_reduce_add_f32x8_haswell_(sum_squared_f32x8);

    // Scalar tail
    for (; j < n; ++j) {
        nk_f32_t a_x_f32, a_y_f32, a_z_f32, b_x_f32, b_y_f32, b_z_f32;
        nk_bf16_to_f32_serial(&a[j * 3 + 0], &a_x_f32);
        nk_bf16_to_f32_serial(&a[j * 3 + 1], &a_y_f32);
        nk_bf16_to_f32_serial(&a[j * 3 + 2], &a_z_f32);
        nk_bf16_to_f32_serial(&b[j * 3 + 0], &b_x_f32);
        nk_bf16_to_f32_serial(&b[j * 3 + 1], &b_y_f32);
        nk_bf16_to_f32_serial(&b[j * 3 + 2], &b_z_f32);

        nk_f32_t pa_x = a_x_f32 - centroid_a_x;
        nk_f32_t pa_y = a_y_f32 - centroid_a_y;
        nk_f32_t pa_z = a_z_f32 - centroid_a_z;
        nk_f32_t pb_x = b_x_f32 - centroid_b_x;
        nk_f32_t pb_y = b_y_f32 - centroid_b_y;
        nk_f32_t pb_z = b_z_f32 - centroid_b_z;

        nk_f32_t ra_x = scale * (r[0] * pa_x + r[1] * pa_y + r[2] * pa_z);
        nk_f32_t ra_y = scale * (r[3] * pa_x + r[4] * pa_y + r[5] * pa_z);
        nk_f32_t ra_z = scale * (r[6] * pa_x + r[7] * pa_y + r[8] * pa_z);

        nk_f32_t delta_x = ra_x - pb_x;
        nk_f32_t delta_y = ra_y - pb_y;
        nk_f32_t delta_z = ra_z - pb_z;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    return sum_squared;
}

NK_PUBLIC void nk_rmsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids and squared differences (all in f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 sum_squared_x_f32x8 = zeros_f32x8, sum_squared_y_f32x8 = zeros_f32x8, sum_squared_z_f32x8 = zeros_f32x8;

    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t i = 0;

    // Main loop processing 8 points at a time
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        __m256 delta_x_f32x8 = _mm256_sub_ps(a_x_f32x8, b_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(a_y_f32x8, b_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(a_z_f32x8, b_z_f32x8);

        sum_squared_x_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_x_f32x8);
        sum_squared_y_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_y_f32x8);
        sum_squared_z_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_z_f32x8);
    }

    // Reduce vectors to scalars
    nk_f32_t total_ax = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t total_ay = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t total_az = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t total_bx = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t total_by = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t total_bz = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t total_sq_x = nk_reduce_add_f32x8_haswell_(sum_squared_x_f32x8);
    nk_f32_t total_sq_y = nk_reduce_add_f32x8_haswell_(sum_squared_y_f32x8);
    nk_f32_t total_sq_z = nk_reduce_add_f32x8_haswell_(sum_squared_z_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_haswell(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_haswell(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_haswell(&a[i * 3 + 2], &az);
        nk_f16_to_f32_haswell(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_haswell(&b[i * 3 + 1], &by);
        nk_f16_to_f32_haswell(&b[i * 3 + 2], &bz);
        total_ax += ax;
        total_ay += ay;
        total_az += az;
        total_bx += bx;
        total_by += by;
        total_bz += bz;
        nk_f32_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        total_sq_x += delta_x * delta_x;
        total_sq_y += delta_y * delta_y;
        total_sq_z += delta_z * delta_z;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = total_ax * inv_n;
    nk_f32_t centroid_a_y = total_ay * inv_n;
    nk_f32_t centroid_a_z = total_az * inv_n;
    nk_f32_t centroid_b_x = total_bx * inv_n;
    nk_f32_t centroid_b_y = total_by * inv_n;
    nk_f32_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute RMSD
    nk_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f32_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f32_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    nk_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f32_sqrt_haswell(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_rmsd_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids and squared differences (all in f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 sum_squared_x_f32x8 = zeros_f32x8, sum_squared_y_f32x8 = zeros_f32x8, sum_squared_z_f32x8 = zeros_f32x8;

    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t i = 0;

    // Main loop processing 8 points at a time
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        __m256 delta_x_f32x8 = _mm256_sub_ps(a_x_f32x8, b_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(a_y_f32x8, b_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(a_z_f32x8, b_z_f32x8);

        sum_squared_x_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_x_f32x8);
        sum_squared_y_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_y_f32x8);
        sum_squared_z_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_z_f32x8);
    }

    // Reduce vectors to scalars
    nk_f32_t total_ax = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t total_ay = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t total_az = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t total_bx = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t total_by = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t total_bz = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t total_sq_x = nk_reduce_add_f32x8_haswell_(sum_squared_x_f32x8);
    nk_f32_t total_sq_y = nk_reduce_add_f32x8_haswell_(sum_squared_y_f32x8);
    nk_f32_t total_sq_z = nk_reduce_add_f32x8_haswell_(sum_squared_z_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_serial(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_serial(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_serial(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_serial(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_serial(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_serial(&b[i * 3 + 2], &bz);
        total_ax += ax;
        total_ay += ay;
        total_az += az;
        total_bx += bx;
        total_by += by;
        total_bz += bz;
        nk_f32_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        total_sq_x += delta_x * delta_x;
        total_sq_y += delta_y * delta_y;
        total_sq_z += delta_z * delta_z;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = total_ax * inv_n;
    nk_f32_t centroid_a_y = total_ay * inv_n;
    nk_f32_t centroid_a_z = total_az * inv_n;
    nk_f32_t centroid_b_x = total_bx * inv_n;
    nk_f32_t centroid_b_y = total_by * inv_n;
    nk_f32_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute RMSD
    nk_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f32_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f32_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    nk_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f32_sqrt_haswell(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load f16, convert to f32, compute centroids and covariance
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids (f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;

    // Accumulators for covariance matrix (sum of outer products)
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);

    nk_f32_t covariance_x_x = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t covariance_x_y = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t covariance_x_z = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t covariance_y_x = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t covariance_y_y = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t covariance_y_z = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t covariance_z_x = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t covariance_z_y = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t covariance_z_z = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_haswell(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_haswell(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_haswell(&a[i * 3 + 2], &az);
        nk_f16_to_f32_haswell(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_haswell(&b[i * 3 + 1], &by);
        nk_f16_to_f32_haswell(&b[i * 3 + 2], &bz);
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        covariance_x_x += ax * bx;
        covariance_x_y += ax * by;
        covariance_x_z += ax * bz;
        covariance_y_x += ay * bx;
        covariance_y_y += ay * by;
        covariance_y_z += ay * bz;
        covariance_z_x += az * bx;
        covariance_z_y += az * by;
        covariance_z_z += az * bz;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n;
    nk_f32_t centroid_a_y = sum_a_y * inv_n;
    nk_f32_t centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n;
    nk_f32_t centroid_b_y = sum_b_y * inv_n;
    nk_f32_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation
    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (nk_det3x3_f32_(r) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0f;

    // Compute RMSD after optimal rotation
    nk_f32_t sum_squared = nk_transformed_ssd_f16_haswell_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_kabsch_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load bf16, convert to f32, compute centroids and covariance
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids (f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;

    // Accumulators for covariance matrix (sum of outer products)
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);

    nk_f32_t covariance_x_x = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t covariance_x_y = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t covariance_x_z = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t covariance_y_x = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t covariance_y_y = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t covariance_y_z = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t covariance_z_x = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t covariance_z_y = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t covariance_z_z = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_serial(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_serial(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_serial(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_serial(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_serial(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_serial(&b[i * 3 + 2], &bz);
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        covariance_x_x += ax * bx;
        covariance_x_y += ax * by;
        covariance_x_z += ax * bz;
        covariance_y_x += ay * bx;
        covariance_y_y += ay * by;
        covariance_y_z += ay * bz;
        covariance_z_x += az * bx;
        covariance_z_y += az * by;
        covariance_z_z += az * bz;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n;
    nk_f32_t centroid_a_y = sum_a_y * inv_n;
    nk_f32_t centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n;
    nk_f32_t centroid_b_y = sum_b_y * inv_n;
    nk_f32_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation
    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (nk_det3x3_f32_(r) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0f;

    // Compute RMSD after optimal rotation
    nk_f32_t sum_squared = nk_transformed_ssd_bf16_haswell_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                            centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load f16, convert to f32, compute centroids, covariance, and variance
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;
    __m256 variance_a_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);

        // Accumulate variance of A
        variance_a_f32x8 = _mm256_fmadd_ps(a_x_f32x8, a_x_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_y_f32x8, a_y_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_z_f32x8, a_z_f32x8, variance_a_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t covariance_x_x = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t covariance_x_y = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t covariance_x_z = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t covariance_y_x = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t covariance_y_y = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t covariance_y_z = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t covariance_z_x = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t covariance_z_y = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t covariance_z_z = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);
    nk_f32_t variance_a_sum = nk_reduce_add_f32x8_haswell_(variance_a_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_haswell(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_haswell(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_haswell(&a[i * 3 + 2], &az);
        nk_f16_to_f32_haswell(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_haswell(&b[i * 3 + 1], &by);
        nk_f16_to_f32_haswell(&b[i * 3 + 2], &bz);
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        covariance_x_x += ax * bx;
        covariance_x_y += ax * by;
        covariance_x_z += ax * bz;
        covariance_y_x += ay * bx;
        covariance_y_y += ay * by;
        covariance_y_z += ay * bz;
        covariance_z_x += az * bx;
        covariance_z_y += az * by;
        covariance_z_z += az * bz;
        variance_a_sum += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    // Apply centering correction to covariance matrix
    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Scale factor: c = trace(D × S) / (n × variance(a))
    nk_f32_t det = nk_det3x3_f32_(r);
    nk_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    nk_f32_t c = trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    // Compute RMSD with scaling
    nk_f32_t sum_squared = nk_transformed_ssd_f16_haswell_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load bf16, convert to f32, compute centroids, covariance, and variance
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;
    __m256 variance_a_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);

        // Accumulate variance of A
        variance_a_f32x8 = _mm256_fmadd_ps(a_x_f32x8, a_x_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_y_f32x8, a_y_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_z_f32x8, a_z_f32x8, variance_a_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t covariance_x_x = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t covariance_x_y = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t covariance_x_z = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t covariance_y_x = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t covariance_y_y = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t covariance_y_z = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t covariance_z_x = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t covariance_z_y = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t covariance_z_z = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);
    nk_f32_t variance_a_sum = nk_reduce_add_f32x8_haswell_(variance_a_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_serial(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_serial(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_serial(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_serial(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_serial(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_serial(&b[i * 3 + 2], &bz);
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        covariance_x_x += ax * bx;
        covariance_x_y += ax * by;
        covariance_x_z += ax * bz;
        covariance_y_x += ay * bx;
        covariance_y_y += ay * by;
        covariance_y_z += ay * bz;
        covariance_z_x += az * bx;
        covariance_z_y += az * by;
        covariance_z_z += az * bz;
        variance_a_sum += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    // Apply centering correction to covariance matrix
    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Scale factor: c = trace(D × S) / (n × variance(a))
    nk_f32_t det = nk_det3x3_f32_(r);
    nk_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    nk_f32_t c = trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    // Compute RMSD with scaling
    nk_f32_t sum_squared = nk_transformed_ssd_bf16_haswell_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                            centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
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
#endif // NK_TARGET_X86_
#endif // NK_MESH_HASWELL_H
