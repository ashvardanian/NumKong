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
 *      Intrinsic               Instruction                     Haswell         Genoa
 *      _mm256_fmadd_ps         VFMADD (YMM, YMM, YMM)          5cy @ p01       4cy @ p01
 *      _mm256_hadd_ps          VHADDPS (YMM, YMM, YMM)         7cy @ p1+p5     4cy @ p123+p23+p23
 *      _mm256_permute2f128_ps  VPERM2F128 (YMM, YMM, YMM, I8)  3cy @ p5        2cy @ p12
 *      _mm256_extractf128_ps   VEXTRACTF128 (XMM, YMM, I8)     3cy @ p5        1cy @ p0123
 *      _mm256_i32gather_ps     VGATHERDPS (YMM, M, YMM, YMM)   22cy (34 uops)  19cy (17 uops)
 *
 *  Point cloud operations (centroid, covariance, Kabsch alignment) use gather instructions for
 *  stride-3 xyz deinterleaving. Multiple FMA accumulators hide the 5-cycle FMA latency. VHADDPS
 *  interleaves results across lanes, requiring additional shuffles for final scalar reduction.
 */
#ifndef NK_MESH_HASWELL_H
#define NK_MESH_HASWELL_H

#if NK_TARGET_X8664_
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
    __m256i idx_i32x8 = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    *x_out = _mm256_i32gather_ps(ptr + 0, idx_i32x8, 4);
    *y_out = _mm256_i32gather_ps(ptr + 1, idx_i32x8, 4);
    *z_out = _mm256_i32gather_ps(ptr + 2, idx_i32x8, 4);
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

NK_PUBLIC void nk_rmsd_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;

    __m256d sum_squared_f64x4 = _mm256_setzero_pd();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t index = 0;

    for (; index + 8 <= n; index += 8) {
        nk_deinterleave_f32x8_haswell_(a + index * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8),
            nk_deinterleave_f32x8_haswell_(b + index * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        __m256d delta_x_low_f64x4 = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(a_x_f32x8)),
                                                  _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_f32x8)));
        __m256d delta_x_high_f64x4 = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(a_x_f32x8, 1)),
                                                   _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_f32x8, 1)));
        __m256d delta_y_low_f64x4 = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(a_y_f32x8)),
                                                  _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_f32x8)));
        __m256d delta_y_high_f64x4 = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(a_y_f32x8, 1)),
                                                   _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_f32x8, 1)));
        __m256d delta_z_low_f64x4 = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(a_z_f32x8)),
                                                  _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_f32x8)));
        __m256d delta_z_high_f64x4 = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(a_z_f32x8, 1)),
                                                   _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_f32x8, 1)));

        __m256d batch_sum_squared_f64x4 = _mm256_add_pd(_mm256_mul_pd(delta_x_low_f64x4, delta_x_low_f64x4),
                                                        _mm256_mul_pd(delta_x_high_f64x4, delta_x_high_f64x4));
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_y_low_f64x4, delta_y_low_f64x4, batch_sum_squared_f64x4);
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_y_high_f64x4, delta_y_high_f64x4, batch_sum_squared_f64x4);
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_z_low_f64x4, delta_z_low_f64x4, batch_sum_squared_f64x4);
        batch_sum_squared_f64x4 = _mm256_fmadd_pd(delta_z_high_f64x4, delta_z_high_f64x4, batch_sum_squared_f64x4);
        sum_squared_f64x4 = _mm256_add_pd(sum_squared_f64x4, batch_sum_squared_f64x4);
    }

    nk_f64_t sum_squared = nk_reduce_add_f64x4_haswell_(sum_squared_f64x4);

    for (; index < n; ++index) {
        nk_f64_t delta_x = (nk_f64_t)a[index * 3 + 0] - (nk_f64_t)b[index * 3 + 0];
        nk_f64_t delta_y = (nk_f64_t)a[index * 3 + 1] - (nk_f64_t)b[index * 3 + 1];
        nk_f64_t delta_z = (nk_f64_t)a[index * 3 + 2] - (nk_f64_t)b[index * 3 + 2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = nk_f64_sqrt_haswell(sum_squared / (nk_f64_t)n);
}

NK_PUBLIC void nk_rmsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0;
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    __m256d sum_squared_x_f64x4 = zeros_f64x4, sum_squared_y_f64x4 = zeros_f64x4, sum_squared_z_f64x4 = zeros_f64x4;

    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;
    nk_size_t i = 0;

    // Main loop with 2× unrolling
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        __m256d delta_x_f64x4 = _mm256_sub_pd(a_x_f64x4, b_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(a_y_f64x4, b_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(a_z_f64x4, b_z_f64x4);

        sum_squared_x_f64x4 = _mm256_fmadd_pd(delta_x_f64x4, delta_x_f64x4, sum_squared_x_f64x4);
        sum_squared_y_f64x4 = _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, sum_squared_y_f64x4);
        sum_squared_z_f64x4 = _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4, sum_squared_z_f64x4);

        __m256d a_x1_f64x4, a_y1_f64x4, a_z1_f64x4, b_x1_f64x4, b_y1_f64x4, b_z1_f64x4;
        nk_deinterleave_f64x4_haswell_(a + (i + 4) * 3, &a_x1_f64x4, &a_y1_f64x4, &a_z1_f64x4);
        nk_deinterleave_f64x4_haswell_(b + (i + 4) * 3, &b_x1_f64x4, &b_y1_f64x4, &b_z1_f64x4);

        __m256d delta_x1_f64x4 = _mm256_sub_pd(a_x1_f64x4, b_x1_f64x4);
        __m256d delta_y1_f64x4 = _mm256_sub_pd(a_y1_f64x4, b_y1_f64x4);
        __m256d delta_z1_f64x4 = _mm256_sub_pd(a_z1_f64x4, b_z1_f64x4);

        sum_squared_x_f64x4 = _mm256_fmadd_pd(delta_x1_f64x4, delta_x1_f64x4, sum_squared_x_f64x4);
        sum_squared_y_f64x4 = _mm256_fmadd_pd(delta_y1_f64x4, delta_y1_f64x4, sum_squared_y_f64x4);
        sum_squared_z_f64x4 = _mm256_fmadd_pd(delta_z1_f64x4, delta_z1_f64x4, sum_squared_z_f64x4);
    }

    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        __m256d delta_x_f64x4 = _mm256_sub_pd(a_x_f64x4, b_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(a_y_f64x4, b_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(a_z_f64x4, b_z_f64x4);

        sum_squared_x_f64x4 = _mm256_fmadd_pd(delta_x_f64x4, delta_x_f64x4, sum_squared_x_f64x4);
        sum_squared_y_f64x4 = _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, sum_squared_y_f64x4);
        sum_squared_z_f64x4 = _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4, sum_squared_z_f64x4);
    }

    nk_f64_t total_sq_x = nk_reduce_stable_f64x4_haswell_(sum_squared_x_f64x4), total_sq_x_compensation = 0.0;
    nk_f64_t total_sq_y = nk_reduce_stable_f64x4_haswell_(sum_squared_y_f64x4), total_sq_y_compensation = 0.0;
    nk_f64_t total_sq_z = nk_reduce_stable_f64x4_haswell_(sum_squared_z_f64x4), total_sq_z_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t delta_x = a[i * 3 + 0] - b[i * 3 + 0];
        nk_f64_t delta_y = a[i * 3 + 1] - b[i * 3 + 1];
        nk_f64_t delta_z = a[i * 3 + 2] - b[i * 3 + 2];
        nk_accumulate_square_f64_(&total_sq_x, &total_sq_x_compensation, delta_x);
        nk_accumulate_square_f64_(&total_sq_y, &total_sq_y_compensation, delta_y);
        nk_accumulate_square_f64_(&total_sq_z, &total_sq_z_compensation, delta_z);
    }

    total_sq_x += total_sq_x_compensation, total_sq_y += total_sq_y_compensation, total_sq_z += total_sq_z_compensation;

    *result = nk_f64_sqrt_haswell((total_sq_x + total_sq_y + total_sq_z) / (nk_f64_t)n);
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
    __m256d norm_squared_a_f64x4 = _mm256_setzero_pd(), norm_squared_b_f64x4 = _mm256_setzero_pd();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t index = 0;

    for (; index + 8 <= n; index += 8) {
        nk_deinterleave_f32x8_haswell_(a + index * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8),
            nk_deinterleave_f32x8_haswell_(b + index * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);
        __m256d a_x_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_f32x8));
        __m256d a_x_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_f32x8, 1));
        __m256d a_y_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_f32x8));
        __m256d a_y_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_f32x8, 1));
        __m256d a_z_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_f32x8));
        __m256d a_z_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_f32x8, 1));
        __m256d b_x_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_f32x8));
        __m256d b_x_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_f32x8, 1));
        __m256d b_y_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_f32x8));
        __m256d b_y_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_f32x8, 1));
        __m256d b_z_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_f32x8));
        __m256d b_z_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_f32x8, 1));

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, _mm256_add_pd(a_x_low_f64x4, a_x_high_f64x4));
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, _mm256_add_pd(a_y_low_f64x4, a_y_high_f64x4));
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, _mm256_add_pd(a_z_low_f64x4, a_z_high_f64x4));
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, _mm256_add_pd(b_x_low_f64x4, b_x_high_f64x4));
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, _mm256_add_pd(b_y_low_f64x4, b_y_high_f64x4));
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, _mm256_add_pd(b_z_low_f64x4, b_z_high_f64x4));

        covariance_00_f64x4 = _mm256_add_pd(
            covariance_00_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_x_low_f64x4, b_x_low_f64x4), _mm256_mul_pd(a_x_high_f64x4, b_x_high_f64x4)));
        covariance_01_f64x4 = _mm256_add_pd(
            covariance_01_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_x_low_f64x4, b_y_low_f64x4), _mm256_mul_pd(a_x_high_f64x4, b_y_high_f64x4)));
        covariance_02_f64x4 = _mm256_add_pd(
            covariance_02_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_x_low_f64x4, b_z_low_f64x4), _mm256_mul_pd(a_x_high_f64x4, b_z_high_f64x4)));
        covariance_10_f64x4 = _mm256_add_pd(
            covariance_10_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_y_low_f64x4, b_x_low_f64x4), _mm256_mul_pd(a_y_high_f64x4, b_x_high_f64x4)));
        covariance_11_f64x4 = _mm256_add_pd(
            covariance_11_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_y_low_f64x4, b_y_low_f64x4), _mm256_mul_pd(a_y_high_f64x4, b_y_high_f64x4)));
        covariance_12_f64x4 = _mm256_add_pd(
            covariance_12_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_y_low_f64x4, b_z_low_f64x4), _mm256_mul_pd(a_y_high_f64x4, b_z_high_f64x4)));
        covariance_20_f64x4 = _mm256_add_pd(
            covariance_20_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_z_low_f64x4, b_x_low_f64x4), _mm256_mul_pd(a_z_high_f64x4, b_x_high_f64x4)));
        covariance_21_f64x4 = _mm256_add_pd(
            covariance_21_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_z_low_f64x4, b_y_low_f64x4), _mm256_mul_pd(a_z_high_f64x4, b_y_high_f64x4)));
        covariance_22_f64x4 = _mm256_add_pd(
            covariance_22_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_z_low_f64x4, b_z_low_f64x4), _mm256_mul_pd(a_z_high_f64x4, b_z_high_f64x4)));
        norm_squared_a_f64x4 = _mm256_add_pd(
            norm_squared_a_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_x_low_f64x4, a_x_low_f64x4), _mm256_mul_pd(a_x_high_f64x4, a_x_high_f64x4)));
        norm_squared_a_f64x4 = _mm256_add_pd(
            norm_squared_a_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_y_low_f64x4, a_y_low_f64x4), _mm256_mul_pd(a_y_high_f64x4, a_y_high_f64x4)));
        norm_squared_a_f64x4 = _mm256_add_pd(
            norm_squared_a_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_z_low_f64x4, a_z_low_f64x4), _mm256_mul_pd(a_z_high_f64x4, a_z_high_f64x4)));
        norm_squared_b_f64x4 = _mm256_add_pd(
            norm_squared_b_f64x4,
            _mm256_add_pd(_mm256_mul_pd(b_x_low_f64x4, b_x_low_f64x4), _mm256_mul_pd(b_x_high_f64x4, b_x_high_f64x4)));
        norm_squared_b_f64x4 = _mm256_add_pd(
            norm_squared_b_f64x4,
            _mm256_add_pd(_mm256_mul_pd(b_y_low_f64x4, b_y_low_f64x4), _mm256_mul_pd(b_y_high_f64x4, b_y_high_f64x4)));
        norm_squared_b_f64x4 = _mm256_add_pd(
            norm_squared_b_f64x4,
            _mm256_add_pd(_mm256_mul_pd(b_z_low_f64x4, b_z_low_f64x4), _mm256_mul_pd(b_z_high_f64x4, b_z_high_f64x4)));
    }

    nk_f64_t sum_a_x = nk_reduce_add_f64x4_haswell_(sum_a_x_f64x4);
    nk_f64_t sum_a_y = nk_reduce_add_f64x4_haswell_(sum_a_y_f64x4);
    nk_f64_t sum_a_z = nk_reduce_add_f64x4_haswell_(sum_a_z_f64x4);
    nk_f64_t sum_b_x = nk_reduce_add_f64x4_haswell_(sum_b_x_f64x4);
    nk_f64_t sum_b_y = nk_reduce_add_f64x4_haswell_(sum_b_y_f64x4);
    nk_f64_t sum_b_z = nk_reduce_add_f64x4_haswell_(sum_b_z_f64x4);
    nk_f64_t cross_covariance[9] = {
        nk_reduce_add_f64x4_haswell_(covariance_00_f64x4), nk_reduce_add_f64x4_haswell_(covariance_01_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_02_f64x4), nk_reduce_add_f64x4_haswell_(covariance_10_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_11_f64x4), nk_reduce_add_f64x4_haswell_(covariance_12_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_20_f64x4), nk_reduce_add_f64x4_haswell_(covariance_21_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_22_f64x4)};
    nk_f64_t norm_squared_a = nk_reduce_add_f64x4_haswell_(norm_squared_a_f64x4);
    nk_f64_t norm_squared_b = nk_reduce_add_f64x4_haswell_(norm_squared_b_f64x4);

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        cross_covariance[0] += a_x * b_x, cross_covariance[1] += a_x * b_y, cross_covariance[2] += a_x * b_z;
        cross_covariance[3] += a_y * b_x, cross_covariance[4] += a_y * b_y, cross_covariance[5] += a_y * b_z;
        cross_covariance[6] += a_z * b_x, cross_covariance[7] += a_z * b_y, cross_covariance[8] += a_z * b_z;
        norm_squared_a += a_x * a_x + a_y * a_y + a_z * a_z;
        norm_squared_b += b_x * b_x + b_y * b_y + b_z * b_z;
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

    cross_covariance[0] -= (nk_f64_t)n * centroid_a_x * centroid_b_x,
        cross_covariance[1] -= (nk_f64_t)n * centroid_a_x * centroid_b_y,
        cross_covariance[2] -= (nk_f64_t)n * centroid_a_x * centroid_b_z,
        cross_covariance[3] -= (nk_f64_t)n * centroid_a_y * centroid_b_x,
        cross_covariance[4] -= (nk_f64_t)n * centroid_a_y * centroid_b_y,
        cross_covariance[5] -= (nk_f64_t)n * centroid_a_y * centroid_b_z,
        cross_covariance[6] -= (nk_f64_t)n * centroid_a_z * centroid_b_x,
        cross_covariance[7] -= (nk_f64_t)n * centroid_a_z * centroid_b_y,
        cross_covariance[8] -= (nk_f64_t)n * centroid_a_z * centroid_b_z;

    nk_f64_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f64_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f64_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;

    // Identity-dominant short-circuit: R = I, trace(R * H) = H[0]+H[4]+H[8]. Skips SVD + two
    // rotation_from_svd reconstructions when the inputs are already aligned.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        if (nk_det3x3_f64_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
            optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
            optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
            optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
            optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
            optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
            optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
            optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
            optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        }
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    if (rotation)
        for (int j = 0; j != 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f64_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0 * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_haswell(sum_squared / (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    // Accumulators for centroids
    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;

    // Accumulators for covariance matrix (sum of outer products)
    __m256d covariance_xx_f64x4 = zeros_f64x4, covariance_xy_f64x4 = zeros_f64x4, covariance_xz_f64x4 = zeros_f64x4;
    __m256d covariance_yx_f64x4 = zeros_f64x4, covariance_yy_f64x4 = zeros_f64x4, covariance_yz_f64x4 = zeros_f64x4;
    __m256d covariance_zx_f64x4 = zeros_f64x4, covariance_zy_f64x4 = zeros_f64x4, covariance_zz_f64x4 = zeros_f64x4;
    __m256d norm_squared_a_f64x4 = zeros_f64x4, norm_squared_b_f64x4 = zeros_f64x4;

    nk_size_t i = 0;
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;

    // Fused single-pass (centroids + covariance + norm-squared for folded SSD)
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_f64x4);

        covariance_xx_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_x_f64x4, covariance_xx_f64x4);
        covariance_xy_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_y_f64x4, covariance_xy_f64x4);
        covariance_xz_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_z_f64x4, covariance_xz_f64x4);
        covariance_yx_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_x_f64x4, covariance_yx_f64x4);
        covariance_yy_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_y_f64x4, covariance_yy_f64x4);
        covariance_yz_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_z_f64x4, covariance_yz_f64x4);
        covariance_zx_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_x_f64x4, covariance_zx_f64x4);
        covariance_zy_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_y_f64x4, covariance_zy_f64x4);
        covariance_zz_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_z_f64x4, covariance_zz_f64x4);
        norm_squared_a_f64x4 = _mm256_fmadd_pd(a_x_f64x4, a_x_f64x4, norm_squared_a_f64x4);
        norm_squared_a_f64x4 = _mm256_fmadd_pd(a_y_f64x4, a_y_f64x4, norm_squared_a_f64x4);
        norm_squared_a_f64x4 = _mm256_fmadd_pd(a_z_f64x4, a_z_f64x4, norm_squared_a_f64x4);
        norm_squared_b_f64x4 = _mm256_fmadd_pd(b_x_f64x4, b_x_f64x4, norm_squared_b_f64x4);
        norm_squared_b_f64x4 = _mm256_fmadd_pd(b_y_f64x4, b_y_f64x4, norm_squared_b_f64x4);
        norm_squared_b_f64x4 = _mm256_fmadd_pd(b_z_f64x4, b_z_f64x4, norm_squared_b_f64x4);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = nk_reduce_stable_f64x4_haswell_(sum_a_x_f64x4), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x4_haswell_(sum_a_y_f64x4), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x4_haswell_(sum_a_z_f64x4), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x4_haswell_(sum_b_x_f64x4), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x4_haswell_(sum_b_y_f64x4), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x4_haswell_(sum_b_z_f64x4), sum_b_z_compensation = 0.0;

    nk_f64_t covariance_x_x = nk_reduce_stable_f64x4_haswell_(covariance_xx_f64x4), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x4_haswell_(covariance_xy_f64x4), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x4_haswell_(covariance_xz_f64x4), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x4_haswell_(covariance_yx_f64x4), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x4_haswell_(covariance_yy_f64x4), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x4_haswell_(covariance_yz_f64x4), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x4_haswell_(covariance_zx_f64x4), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x4_haswell_(covariance_zy_f64x4), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x4_haswell_(covariance_zz_f64x4), covariance_z_z_compensation = 0.0;
    nk_f64_t norm_squared_a_sum = nk_reduce_stable_f64x4_haswell_(norm_squared_a_f64x4),
             norm_squared_a_compensation = 0.0;
    nk_f64_t norm_squared_b_sum = nk_reduce_stable_f64x4_haswell_(norm_squared_b_f64x4),
             norm_squared_b_compensation = 0.0;

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
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ax);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ay);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, az);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bx);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, by);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bz);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    norm_squared_a_sum += norm_squared_a_compensation;
    norm_squared_b_sum += norm_squared_b_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n;
    nk_f64_t centroid_a_y = sum_a_y * inv_n;
    nk_f64_t centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n;
    nk_f64_t centroid_b_y = sum_b_y * inv_n;
    nk_f64_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    covariance_x_x -= (nk_f64_t)n * centroid_a_x * centroid_b_x;
    covariance_x_y -= (nk_f64_t)n * centroid_a_x * centroid_b_y;
    covariance_x_z -= (nk_f64_t)n * centroid_a_x * centroid_b_z;
    covariance_y_x -= (nk_f64_t)n * centroid_a_y * centroid_b_x;
    covariance_y_y -= (nk_f64_t)n * centroid_a_y * centroid_b_y;
    covariance_y_z -= (nk_f64_t)n * centroid_a_y * centroid_b_z;
    covariance_z_x -= (nk_f64_t)n * centroid_a_z * centroid_b_x;
    covariance_z_y -= (nk_f64_t)n * centroid_a_z * centroid_b_y;
    covariance_z_z -= (nk_f64_t)n * centroid_a_z * centroid_b_z;

    nk_f64_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f64_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f64_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f64_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;

    // Identity-dominant short-circuit: if H is essentially diagonal with positive diagonals, R = I.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1.0, optimal_rotation[1] = 0.0, optimal_rotation[2] = 0.0;
        optimal_rotation[3] = 0.0, optimal_rotation[4] = 1.0, optimal_rotation[5] = 0.0;
        optimal_rotation[6] = 0.0, optimal_rotation[7] = 0.0, optimal_rotation[8] = 1.0;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        if (nk_det3x3_f64_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0;

    nk_f64_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0 * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
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
    __m256d covariance_22_f64x4 = _mm256_setzero_pd();
    __m256d norm_squared_a_f64x4 = _mm256_setzero_pd(), norm_squared_b_f64x4 = _mm256_setzero_pd();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t index = 0;

    for (; index + 8 <= n; index += 8) {
        nk_deinterleave_f32x8_haswell_(a + index * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8),
            nk_deinterleave_f32x8_haswell_(b + index * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);
        __m256d a_x_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_f32x8));
        __m256d a_x_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_f32x8, 1));
        __m256d a_y_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_f32x8));
        __m256d a_y_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_f32x8, 1));
        __m256d a_z_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_f32x8));
        __m256d a_z_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_f32x8, 1));
        __m256d b_x_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_f32x8));
        __m256d b_x_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_f32x8, 1));
        __m256d b_y_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_f32x8));
        __m256d b_y_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_f32x8, 1));
        __m256d b_z_low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_f32x8));
        __m256d b_z_high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_f32x8, 1));

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, _mm256_add_pd(a_x_low_f64x4, a_x_high_f64x4));
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, _mm256_add_pd(a_y_low_f64x4, a_y_high_f64x4));
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, _mm256_add_pd(a_z_low_f64x4, a_z_high_f64x4));
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, _mm256_add_pd(b_x_low_f64x4, b_x_high_f64x4));
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, _mm256_add_pd(b_y_low_f64x4, b_y_high_f64x4));
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, _mm256_add_pd(b_z_low_f64x4, b_z_high_f64x4));
        covariance_00_f64x4 = _mm256_add_pd(
            covariance_00_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_x_low_f64x4, b_x_low_f64x4), _mm256_mul_pd(a_x_high_f64x4, b_x_high_f64x4)));
        covariance_01_f64x4 = _mm256_add_pd(
            covariance_01_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_x_low_f64x4, b_y_low_f64x4), _mm256_mul_pd(a_x_high_f64x4, b_y_high_f64x4)));
        covariance_02_f64x4 = _mm256_add_pd(
            covariance_02_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_x_low_f64x4, b_z_low_f64x4), _mm256_mul_pd(a_x_high_f64x4, b_z_high_f64x4)));
        covariance_10_f64x4 = _mm256_add_pd(
            covariance_10_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_y_low_f64x4, b_x_low_f64x4), _mm256_mul_pd(a_y_high_f64x4, b_x_high_f64x4)));
        covariance_11_f64x4 = _mm256_add_pd(
            covariance_11_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_y_low_f64x4, b_y_low_f64x4), _mm256_mul_pd(a_y_high_f64x4, b_y_high_f64x4)));
        covariance_12_f64x4 = _mm256_add_pd(
            covariance_12_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_y_low_f64x4, b_z_low_f64x4), _mm256_mul_pd(a_y_high_f64x4, b_z_high_f64x4)));
        covariance_20_f64x4 = _mm256_add_pd(
            covariance_20_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_z_low_f64x4, b_x_low_f64x4), _mm256_mul_pd(a_z_high_f64x4, b_x_high_f64x4)));
        covariance_21_f64x4 = _mm256_add_pd(
            covariance_21_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_z_low_f64x4, b_y_low_f64x4), _mm256_mul_pd(a_z_high_f64x4, b_y_high_f64x4)));
        covariance_22_f64x4 = _mm256_add_pd(
            covariance_22_f64x4,
            _mm256_add_pd(_mm256_mul_pd(a_z_low_f64x4, b_z_low_f64x4), _mm256_mul_pd(a_z_high_f64x4, b_z_high_f64x4)));
        norm_squared_a_f64x4 = _mm256_add_pd(
            norm_squared_a_f64x4,
            _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a_x_low_f64x4, a_x_low_f64x4),
                                        _mm256_mul_pd(a_x_high_f64x4, a_x_high_f64x4)),
                          _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(a_y_low_f64x4, a_y_low_f64x4),
                                                      _mm256_mul_pd(a_y_high_f64x4, a_y_high_f64x4)),
                                        _mm256_add_pd(_mm256_mul_pd(a_z_low_f64x4, a_z_low_f64x4),
                                                      _mm256_mul_pd(a_z_high_f64x4, a_z_high_f64x4)))));
        norm_squared_b_f64x4 = _mm256_add_pd(
            norm_squared_b_f64x4,
            _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(b_x_low_f64x4, b_x_low_f64x4),
                                        _mm256_mul_pd(b_x_high_f64x4, b_x_high_f64x4)),
                          _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(b_y_low_f64x4, b_y_low_f64x4),
                                                      _mm256_mul_pd(b_y_high_f64x4, b_y_high_f64x4)),
                                        _mm256_add_pd(_mm256_mul_pd(b_z_low_f64x4, b_z_low_f64x4),
                                                      _mm256_mul_pd(b_z_high_f64x4, b_z_high_f64x4)))));
    }

    nk_f64_t sum_a_x = nk_reduce_add_f64x4_haswell_(sum_a_x_f64x4);
    nk_f64_t sum_a_y = nk_reduce_add_f64x4_haswell_(sum_a_y_f64x4);
    nk_f64_t sum_a_z = nk_reduce_add_f64x4_haswell_(sum_a_z_f64x4);
    nk_f64_t sum_b_x = nk_reduce_add_f64x4_haswell_(sum_b_x_f64x4);
    nk_f64_t sum_b_y = nk_reduce_add_f64x4_haswell_(sum_b_y_f64x4);
    nk_f64_t sum_b_z = nk_reduce_add_f64x4_haswell_(sum_b_z_f64x4);
    nk_f64_t cross_covariance[9] = {
        nk_reduce_add_f64x4_haswell_(covariance_00_f64x4), nk_reduce_add_f64x4_haswell_(covariance_01_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_02_f64x4), nk_reduce_add_f64x4_haswell_(covariance_10_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_11_f64x4), nk_reduce_add_f64x4_haswell_(covariance_12_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_20_f64x4), nk_reduce_add_f64x4_haswell_(covariance_21_f64x4),
        nk_reduce_add_f64x4_haswell_(covariance_22_f64x4)};
    nk_f64_t norm_squared_a_sum = nk_reduce_add_f64x4_haswell_(norm_squared_a_f64x4);
    nk_f64_t norm_squared_b_sum = nk_reduce_add_f64x4_haswell_(norm_squared_b_f64x4);

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        cross_covariance[0] += a_x * b_x, cross_covariance[1] += a_x * b_y, cross_covariance[2] += a_x * b_z;
        cross_covariance[3] += a_y * b_x, cross_covariance[4] += a_y * b_y, cross_covariance[5] += a_y * b_z;
        cross_covariance[6] += a_z * b_x, cross_covariance[7] += a_z * b_y, cross_covariance[8] += a_z * b_z;
        norm_squared_a_sum += a_x * a_x + a_y * a_y + a_z * a_z;
        norm_squared_b_sum += b_x * b_x + b_y * b_y + b_z * b_z;
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

    nk_f64_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f64_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f64_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;
    cross_covariance[0] -= (nk_f64_t)n * centroid_a_x * centroid_b_x,
        cross_covariance[1] -= (nk_f64_t)n * centroid_a_x * centroid_b_y,
        cross_covariance[2] -= (nk_f64_t)n * centroid_a_x * centroid_b_z,
        cross_covariance[3] -= (nk_f64_t)n * centroid_a_y * centroid_b_x,
        cross_covariance[4] -= (nk_f64_t)n * centroid_a_y * centroid_b_y,
        cross_covariance[5] -= (nk_f64_t)n * centroid_a_y * centroid_b_z,
        cross_covariance[6] -= (nk_f64_t)n * centroid_a_z * centroid_b_x,
        cross_covariance[7] -= (nk_f64_t)n * centroid_a_z * centroid_b_y,
        cross_covariance[8] -= (nk_f64_t)n * centroid_a_z * centroid_b_z;

    // Identity-dominant short-circuit: if H is essentially diagonal with positive diagonals,
    // R = I and trace(DS) reduces to trace(H) directly.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t applied_scale;
    nk_f64_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1.0, optimal_rotation[1] = 0.0, optimal_rotation[2] = 0.0;
        optimal_rotation[3] = 0.0, optimal_rotation[4] = 1.0, optimal_rotation[5] = 0.0;
        optimal_rotation[6] = 0.0, optimal_rotation[7] = 0.0, optimal_rotation[8] = 1.0;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        applied_scale = centered_norm_squared_a > 0.0 ? trace_rotation_covariance / centered_norm_squared_a : 0.0;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];

        nk_f64_t det = nk_det3x3_f64_(optimal_rotation), sign_correction = det < 0 ? -1.0 : 1.0;
        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
            optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
            optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
            optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
            optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
            optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
            optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
            optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
            optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        }
        nk_f64_t trace_ds = svd_diagonal[0] + svd_diagonal[4] + sign_correction * svd_diagonal[8];
        applied_scale = centered_norm_squared_a > 0.0 ? trace_ds / centered_norm_squared_a : 0.0;
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    if (rotation)
        for (int j = 0; j != 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];
    if (scale) *scale = (nk_f32_t)applied_scale;

    // Folded SSD with scale: c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f64_t sum_squared = applied_scale * applied_scale * centered_norm_squared_a + centered_norm_squared_b -
                           2.0 * applied_scale * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_haswell(sum_squared / (nk_f64_t)n);
}

NK_PUBLIC void nk_umeyama_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;
    __m256d covariance_xx_f64x4 = zeros_f64x4, covariance_xy_f64x4 = zeros_f64x4, covariance_xz_f64x4 = zeros_f64x4;
    __m256d covariance_yx_f64x4 = zeros_f64x4, covariance_yy_f64x4 = zeros_f64x4, covariance_yz_f64x4 = zeros_f64x4;
    __m256d covariance_zx_f64x4 = zeros_f64x4, covariance_zy_f64x4 = zeros_f64x4, covariance_zz_f64x4 = zeros_f64x4;
    __m256d norm_squared_a_f64x4 = zeros_f64x4, norm_squared_b_f64x4 = zeros_f64x4;

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

        covariance_xx_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_x_f64x4, covariance_xx_f64x4),
        covariance_xy_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_y_f64x4, covariance_xy_f64x4);
        covariance_xz_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_z_f64x4, covariance_xz_f64x4);
        covariance_yx_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_x_f64x4, covariance_yx_f64x4),
        covariance_yy_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_y_f64x4, covariance_yy_f64x4);
        covariance_yz_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_z_f64x4, covariance_yz_f64x4);
        covariance_zx_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_x_f64x4, covariance_zx_f64x4),
        covariance_zy_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_y_f64x4, covariance_zy_f64x4);
        covariance_zz_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_z_f64x4, covariance_zz_f64x4);
        norm_squared_a_f64x4 = _mm256_fmadd_pd(a_x_f64x4, a_x_f64x4, norm_squared_a_f64x4);
        norm_squared_a_f64x4 = _mm256_fmadd_pd(a_y_f64x4, a_y_f64x4, norm_squared_a_f64x4);
        norm_squared_a_f64x4 = _mm256_fmadd_pd(a_z_f64x4, a_z_f64x4, norm_squared_a_f64x4);
        norm_squared_b_f64x4 = _mm256_fmadd_pd(b_x_f64x4, b_x_f64x4, norm_squared_b_f64x4);
        norm_squared_b_f64x4 = _mm256_fmadd_pd(b_y_f64x4, b_y_f64x4, norm_squared_b_f64x4);
        norm_squared_b_f64x4 = _mm256_fmadd_pd(b_z_f64x4, b_z_f64x4, norm_squared_b_f64x4);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = nk_reduce_stable_f64x4_haswell_(sum_a_x_f64x4), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x4_haswell_(sum_a_y_f64x4), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x4_haswell_(sum_a_z_f64x4), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x4_haswell_(sum_b_x_f64x4), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x4_haswell_(sum_b_y_f64x4), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x4_haswell_(sum_b_z_f64x4), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x4_haswell_(covariance_xx_f64x4), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x4_haswell_(covariance_xy_f64x4), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x4_haswell_(covariance_xz_f64x4), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x4_haswell_(covariance_yx_f64x4), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x4_haswell_(covariance_yy_f64x4), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x4_haswell_(covariance_yz_f64x4), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x4_haswell_(covariance_zx_f64x4), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x4_haswell_(covariance_zy_f64x4), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x4_haswell_(covariance_zz_f64x4), covariance_z_z_compensation = 0.0;
    nk_f64_t norm_squared_a_sum = nk_reduce_stable_f64x4_haswell_(norm_squared_a_f64x4),
             norm_squared_a_compensation = 0.0;
    nk_f64_t norm_squared_b_sum = nk_reduce_stable_f64x4_haswell_(norm_squared_b_f64x4),
             norm_squared_b_compensation = 0.0;

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
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ax);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ay);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, az);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bx);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, by);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bz);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    norm_squared_a_sum += norm_squared_a_compensation;
    norm_squared_b_sum += norm_squared_b_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Centered norm-squared via parallel-axis identity; clamped at zero for numeric safety.
    nk_f64_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f64_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f64_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;

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

    // Identity-dominant short-circuit: if H is essentially diagonal with positive diagonals,
    // R = I and trace(DS) reduces to trace(H) directly.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t c;
    nk_f64_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1.0, optimal_rotation[1] = 0.0, optimal_rotation[2] = 0.0;
        optimal_rotation[3] = 0.0, optimal_rotation[4] = 1.0, optimal_rotation[5] = 0.0;
        optimal_rotation[6] = 0.0, optimal_rotation[7] = 0.0, optimal_rotation[8] = 1.0;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        c = centered_norm_squared_a > 0.0 ? trace_rotation_covariance / centered_norm_squared_a : 0.0;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);

        nk_f64_t det = nk_det3x3_f64_(optimal_rotation);
        nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
        nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_diagonal[0], 1.0, svd_diagonal[4], 1.0, svd_diagonal[8], d3);
        c = centered_norm_squared_a > 0.0 ? trace_ds / centered_norm_squared_a : 0.0;

        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    if (scale) *scale = c;
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    // Folded SSD with scale: c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f64_t sum_squared = c * c * centered_norm_squared_a + centered_norm_squared_b -
                           2.0 * c * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
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

NK_PUBLIC void nk_rmsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    __m256 const zeros_f32x8 = _mm256_setzero_ps();
    __m256 sum_squared_x_f32x8 = zeros_f32x8, sum_squared_y_f32x8 = zeros_f32x8, sum_squared_z_f32x8 = zeros_f32x8;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);
        __m256 delta_x_f32x8 = _mm256_sub_ps(a_x_f32x8, b_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(a_y_f32x8, b_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(a_z_f32x8, b_z_f32x8);
        sum_squared_x_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_x_f32x8);
        sum_squared_y_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_y_f32x8);
        sum_squared_z_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_z_f32x8);
    }

    nk_f32_t sum_squared = nk_reduce_add_f32x8_haswell_(sum_squared_x_f32x8) +
                           nk_reduce_add_f32x8_haswell_(sum_squared_y_f32x8) +
                           nk_reduce_add_f32x8_haswell_(sum_squared_z_f32x8);
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_haswell(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_haswell(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_haswell(&a[i * 3 + 2], &az);
        nk_f16_to_f32_haswell(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_haswell(&b[i * 3 + 1], &by);
        nk_f16_to_f32_haswell(&b[i * 3 + 2], &bz);
        nk_f32_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = nk_f32_sqrt_haswell(sum_squared / (nk_f32_t)n);
}

NK_PUBLIC void nk_rmsd_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    __m256 const zeros_f32x8 = _mm256_setzero_ps();
    __m256 sum_squared_x_f32x8 = zeros_f32x8, sum_squared_y_f32x8 = zeros_f32x8, sum_squared_z_f32x8 = zeros_f32x8;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);
        __m256 delta_x_f32x8 = _mm256_sub_ps(a_x_f32x8, b_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(a_y_f32x8, b_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(a_z_f32x8, b_z_f32x8);
        sum_squared_x_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_x_f32x8);
        sum_squared_y_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_y_f32x8);
        sum_squared_z_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_z_f32x8);
    }

    nk_f32_t sum_squared = nk_reduce_add_f32x8_haswell_(sum_squared_x_f32x8) +
                           nk_reduce_add_f32x8_haswell_(sum_squared_y_f32x8) +
                           nk_reduce_add_f32x8_haswell_(sum_squared_z_f32x8);
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_serial(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_serial(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_serial(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_serial(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_serial(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_serial(&b[i * 3 + 2], &bz);
        nk_f32_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = nk_f32_sqrt_haswell(sum_squared / (nk_f32_t)n);
}

NK_PUBLIC void nk_kabsch_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load f16, convert to f32, compute centroids, covariance, and ‖a‖²/‖b‖²
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids (f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;

    // Accumulators for covariance matrix (sum of outer products)
    __m256 covariance_xx_f32x8 = zeros_f32x8, covariance_xy_f32x8 = zeros_f32x8, covariance_xz_f32x8 = zeros_f32x8;
    __m256 covariance_yx_f32x8 = zeros_f32x8, covariance_yy_f32x8 = zeros_f32x8, covariance_yz_f32x8 = zeros_f32x8;
    __m256 covariance_zx_f32x8 = zeros_f32x8, covariance_zy_f32x8 = zeros_f32x8, covariance_zz_f32x8 = zeros_f32x8;
    __m256 norm_squared_a_f32x8 = zeros_f32x8, norm_squared_b_f32x8 = zeros_f32x8;

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
        covariance_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, covariance_xx_f32x8);
        covariance_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, covariance_xy_f32x8);
        covariance_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, covariance_xz_f32x8);
        covariance_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, covariance_yx_f32x8);
        covariance_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, covariance_yy_f32x8);
        covariance_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, covariance_yz_f32x8);
        covariance_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, covariance_zx_f32x8);
        covariance_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, covariance_zy_f32x8);
        covariance_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, covariance_zz_f32x8);

        // Accumulate ‖a‖² and ‖b‖² for folded SSD
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_x_f32x8, a_x_f32x8, norm_squared_a_f32x8);
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_y_f32x8, a_y_f32x8, norm_squared_a_f32x8);
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_z_f32x8, a_z_f32x8, norm_squared_a_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_x_f32x8, b_x_f32x8, norm_squared_b_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_y_f32x8, b_y_f32x8, norm_squared_b_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_z_f32x8, b_z_f32x8, norm_squared_b_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);

    nk_f32_t covariance_x_x = nk_reduce_add_f32x8_haswell_(covariance_xx_f32x8);
    nk_f32_t covariance_x_y = nk_reduce_add_f32x8_haswell_(covariance_xy_f32x8);
    nk_f32_t covariance_x_z = nk_reduce_add_f32x8_haswell_(covariance_xz_f32x8);
    nk_f32_t covariance_y_x = nk_reduce_add_f32x8_haswell_(covariance_yx_f32x8);
    nk_f32_t covariance_y_y = nk_reduce_add_f32x8_haswell_(covariance_yy_f32x8);
    nk_f32_t covariance_y_z = nk_reduce_add_f32x8_haswell_(covariance_yz_f32x8);
    nk_f32_t covariance_z_x = nk_reduce_add_f32x8_haswell_(covariance_zx_f32x8);
    nk_f32_t covariance_z_y = nk_reduce_add_f32x8_haswell_(covariance_zy_f32x8);
    nk_f32_t covariance_z_z = nk_reduce_add_f32x8_haswell_(covariance_zz_f32x8);
    nk_f32_t norm_squared_a_sum = nk_reduce_add_f32x8_haswell_(norm_squared_a_f32x8);
    nk_f32_t norm_squared_b_sum = nk_reduce_add_f32x8_haswell_(norm_squared_b_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_haswell(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_haswell(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_haswell(&a[i * 3 + 2], &az);
        nk_f16_to_f32_haswell(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_haswell(&b[i * 3 + 1], &by);
        nk_f16_to_f32_haswell(&b[i * 3 + 2], &bz);
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        covariance_x_x += ax * bx, covariance_x_y += ax * by, covariance_x_z += ax * bz;
        covariance_y_x += ay * bx, covariance_y_y += ay * by, covariance_y_z += ay * bz;
        covariance_z_x += az * bx, covariance_z_y += az * by, covariance_z_z += az * bz;
        norm_squared_a_sum += ax * ax + ay * ay + az * az;
        norm_squared_b_sum += bx * bx + by * by + bz * bz;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n;
    nk_f32_t centroid_a_y = sum_a_y * inv_n;
    nk_f32_t centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n;
    nk_f32_t centroid_b_y = sum_b_y * inv_n;
    nk_f32_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    covariance_x_x -= (nk_f32_t)n * centroid_a_x * centroid_b_x;
    covariance_x_y -= (nk_f32_t)n * centroid_a_x * centroid_b_y;
    covariance_x_z -= (nk_f32_t)n * centroid_a_x * centroid_b_z;
    covariance_y_x -= (nk_f32_t)n * centroid_a_y * centroid_b_x;
    covariance_y_y -= (nk_f32_t)n * centroid_a_y * centroid_b_y;
    covariance_y_z -= (nk_f32_t)n * centroid_a_y * centroid_b_z;
    covariance_z_x -= (nk_f32_t)n * centroid_a_z * centroid_b_x;
    covariance_z_y -= (nk_f32_t)n * centroid_a_z * centroid_b_y;
    covariance_z_z -= (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // Centered ‖a-ā‖², ‖b-b̄‖² via the parallel-axis identity
    nk_f32_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

    // Identity-dominant short-circuit: R = I, trace(R · H) = H[0]+H[4]+H[8]. Skips SVD + two
    // rotation reconstructions when the inputs are already aligned.
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1.0f, optimal_rotation[1] = 0.0f, optimal_rotation[2] = 0.0f;
        optimal_rotation[3] = 0.0f, optimal_rotation[4] = 1.0f, optimal_rotation[5] = 0.0f;
        optimal_rotation[6] = 0.0f, optimal_rotation[7] = 0.0f, optimal_rotation[8] = 1.0f;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
    }
    else {
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
        // R = V * Uᵀ
        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        if (nk_det3x3_f32_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
            optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
            optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
            optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
            optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
            optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
            optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
            optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
            optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        }
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f32_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0f * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_kabsch_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load bf16, convert to f32, compute centroids, covariance, and ‖a‖²/‖b‖²
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids (f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;

    // Accumulators for covariance matrix (sum of outer products)
    __m256 covariance_xx_f32x8 = zeros_f32x8, covariance_xy_f32x8 = zeros_f32x8, covariance_xz_f32x8 = zeros_f32x8;
    __m256 covariance_yx_f32x8 = zeros_f32x8, covariance_yy_f32x8 = zeros_f32x8, covariance_yz_f32x8 = zeros_f32x8;
    __m256 covariance_zx_f32x8 = zeros_f32x8, covariance_zy_f32x8 = zeros_f32x8, covariance_zz_f32x8 = zeros_f32x8;
    __m256 norm_squared_a_f32x8 = zeros_f32x8, norm_squared_b_f32x8 = zeros_f32x8;

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
        covariance_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, covariance_xx_f32x8);
        covariance_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, covariance_xy_f32x8);
        covariance_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, covariance_xz_f32x8);
        covariance_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, covariance_yx_f32x8);
        covariance_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, covariance_yy_f32x8);
        covariance_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, covariance_yz_f32x8);
        covariance_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, covariance_zx_f32x8);
        covariance_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, covariance_zy_f32x8);
        covariance_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, covariance_zz_f32x8);

        // Accumulate ‖a‖² and ‖b‖² for folded SSD
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_x_f32x8, a_x_f32x8, norm_squared_a_f32x8);
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_y_f32x8, a_y_f32x8, norm_squared_a_f32x8);
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_z_f32x8, a_z_f32x8, norm_squared_a_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_x_f32x8, b_x_f32x8, norm_squared_b_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_y_f32x8, b_y_f32x8, norm_squared_b_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_z_f32x8, b_z_f32x8, norm_squared_b_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);

    nk_f32_t covariance_x_x = nk_reduce_add_f32x8_haswell_(covariance_xx_f32x8);
    nk_f32_t covariance_x_y = nk_reduce_add_f32x8_haswell_(covariance_xy_f32x8);
    nk_f32_t covariance_x_z = nk_reduce_add_f32x8_haswell_(covariance_xz_f32x8);
    nk_f32_t covariance_y_x = nk_reduce_add_f32x8_haswell_(covariance_yx_f32x8);
    nk_f32_t covariance_y_y = nk_reduce_add_f32x8_haswell_(covariance_yy_f32x8);
    nk_f32_t covariance_y_z = nk_reduce_add_f32x8_haswell_(covariance_yz_f32x8);
    nk_f32_t covariance_z_x = nk_reduce_add_f32x8_haswell_(covariance_zx_f32x8);
    nk_f32_t covariance_z_y = nk_reduce_add_f32x8_haswell_(covariance_zy_f32x8);
    nk_f32_t covariance_z_z = nk_reduce_add_f32x8_haswell_(covariance_zz_f32x8);
    nk_f32_t norm_squared_a_sum = nk_reduce_add_f32x8_haswell_(norm_squared_a_f32x8);
    nk_f32_t norm_squared_b_sum = nk_reduce_add_f32x8_haswell_(norm_squared_b_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_serial(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_serial(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_serial(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_serial(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_serial(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_serial(&b[i * 3 + 2], &bz);
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        covariance_x_x += ax * bx, covariance_x_y += ax * by, covariance_x_z += ax * bz;
        covariance_y_x += ay * bx, covariance_y_y += ay * by, covariance_y_z += ay * bz;
        covariance_z_x += az * bx, covariance_z_y += az * by, covariance_z_z += az * bz;
        norm_squared_a_sum += ax * ax + ay * ay + az * az;
        norm_squared_b_sum += bx * bx + by * by + bz * bz;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n;
    nk_f32_t centroid_a_y = sum_a_y * inv_n;
    nk_f32_t centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n;
    nk_f32_t centroid_b_y = sum_b_y * inv_n;
    nk_f32_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    covariance_x_x -= (nk_f32_t)n * centroid_a_x * centroid_b_x;
    covariance_x_y -= (nk_f32_t)n * centroid_a_x * centroid_b_y;
    covariance_x_z -= (nk_f32_t)n * centroid_a_x * centroid_b_z;
    covariance_y_x -= (nk_f32_t)n * centroid_a_y * centroid_b_x;
    covariance_y_y -= (nk_f32_t)n * centroid_a_y * centroid_b_y;
    covariance_y_z -= (nk_f32_t)n * centroid_a_y * centroid_b_z;
    covariance_z_x -= (nk_f32_t)n * centroid_a_z * centroid_b_x;
    covariance_z_y -= (nk_f32_t)n * centroid_a_z * centroid_b_y;
    covariance_z_z -= (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // Centered ‖a-ā‖², ‖b-b̄‖² via the parallel-axis identity
    nk_f32_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

    // Identity-dominant short-circuit: R = I, trace(R · H) = H[0]+H[4]+H[8]. Skips SVD + two
    // rotation reconstructions when the inputs are already aligned.
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1.0f, optimal_rotation[1] = 0.0f, optimal_rotation[2] = 0.0f;
        optimal_rotation[3] = 0.0f, optimal_rotation[4] = 1.0f, optimal_rotation[5] = 0.0f;
        optimal_rotation[6] = 0.0f, optimal_rotation[7] = 0.0f, optimal_rotation[8] = 1.0f;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
    }
    else {
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
        // R = V * Uᵀ
        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        if (nk_det3x3_f32_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
            optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
            optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
            optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
            optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
            optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
            optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
            optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
            optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        }
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f32_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0f * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load f16, convert to f32, compute centroids, covariance, and ‖a‖²/‖b‖²
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 covariance_xx_f32x8 = zeros_f32x8, covariance_xy_f32x8 = zeros_f32x8, covariance_xz_f32x8 = zeros_f32x8;
    __m256 covariance_yx_f32x8 = zeros_f32x8, covariance_yy_f32x8 = zeros_f32x8, covariance_yz_f32x8 = zeros_f32x8;
    __m256 covariance_zx_f32x8 = zeros_f32x8, covariance_zy_f32x8 = zeros_f32x8, covariance_zz_f32x8 = zeros_f32x8;
    __m256 norm_squared_a_f32x8 = zeros_f32x8, norm_squared_b_f32x8 = zeros_f32x8;

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
        covariance_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, covariance_xx_f32x8);
        covariance_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, covariance_xy_f32x8);
        covariance_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, covariance_xz_f32x8);
        covariance_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, covariance_yx_f32x8);
        covariance_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, covariance_yy_f32x8);
        covariance_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, covariance_yz_f32x8);
        covariance_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, covariance_zx_f32x8);
        covariance_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, covariance_zy_f32x8);
        covariance_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, covariance_zz_f32x8);

        // Accumulate ‖a‖² and ‖b‖² for folded SSD
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_x_f32x8, a_x_f32x8, norm_squared_a_f32x8);
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_y_f32x8, a_y_f32x8, norm_squared_a_f32x8);
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_z_f32x8, a_z_f32x8, norm_squared_a_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_x_f32x8, b_x_f32x8, norm_squared_b_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_y_f32x8, b_y_f32x8, norm_squared_b_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_z_f32x8, b_z_f32x8, norm_squared_b_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t covariance_x_x = nk_reduce_add_f32x8_haswell_(covariance_xx_f32x8);
    nk_f32_t covariance_x_y = nk_reduce_add_f32x8_haswell_(covariance_xy_f32x8);
    nk_f32_t covariance_x_z = nk_reduce_add_f32x8_haswell_(covariance_xz_f32x8);
    nk_f32_t covariance_y_x = nk_reduce_add_f32x8_haswell_(covariance_yx_f32x8);
    nk_f32_t covariance_y_y = nk_reduce_add_f32x8_haswell_(covariance_yy_f32x8);
    nk_f32_t covariance_y_z = nk_reduce_add_f32x8_haswell_(covariance_yz_f32x8);
    nk_f32_t covariance_z_x = nk_reduce_add_f32x8_haswell_(covariance_zx_f32x8);
    nk_f32_t covariance_z_y = nk_reduce_add_f32x8_haswell_(covariance_zy_f32x8);
    nk_f32_t covariance_z_z = nk_reduce_add_f32x8_haswell_(covariance_zz_f32x8);
    nk_f32_t norm_squared_a_sum = nk_reduce_add_f32x8_haswell_(norm_squared_a_f32x8);
    nk_f32_t norm_squared_b_sum = nk_reduce_add_f32x8_haswell_(norm_squared_b_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_haswell(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_haswell(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_haswell(&a[i * 3 + 2], &az);
        nk_f16_to_f32_haswell(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_haswell(&b[i * 3 + 1], &by);
        nk_f16_to_f32_haswell(&b[i * 3 + 2], &bz);
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        covariance_x_x += ax * bx, covariance_x_y += ax * by, covariance_x_z += ax * bz;
        covariance_y_x += ay * bx, covariance_y_y += ay * by, covariance_y_z += ay * bz;
        covariance_z_x += az * bx, covariance_z_y += az * by, covariance_z_z += az * bz;
        norm_squared_a_sum += ax * ax + ay * ay + az * az;
        norm_squared_b_sum += bx * bx + by * by + bz * bz;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply centering correction to covariance matrix
    covariance_x_x -= (nk_f32_t)n * centroid_a_x * centroid_b_x;
    covariance_x_y -= (nk_f32_t)n * centroid_a_x * centroid_b_y;
    covariance_x_z -= (nk_f32_t)n * centroid_a_x * centroid_b_z;
    covariance_y_x -= (nk_f32_t)n * centroid_a_y * centroid_b_x;
    covariance_y_y -= (nk_f32_t)n * centroid_a_y * centroid_b_y;
    covariance_y_z -= (nk_f32_t)n * centroid_a_y * centroid_b_z;
    covariance_z_x -= (nk_f32_t)n * centroid_a_z * centroid_b_x;
    covariance_z_y -= (nk_f32_t)n * centroid_a_z * centroid_b_y;
    covariance_z_z -= (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // Centered ‖a-ā‖², ‖b-b̄‖² via the parallel-axis identity
    nk_f32_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

    // Identity-dominant short-circuit: if H is essentially diagonal with positive diagonals,
    // R = I and trace(DS) = trace(H).
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t applied_scale;
    nk_f32_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1.0f, optimal_rotation[1] = 0.0f, optimal_rotation[2] = 0.0f;
        optimal_rotation[3] = 0.0f, optimal_rotation[4] = 1.0f, optimal_rotation[5] = 0.0f;
        optimal_rotation[6] = 0.0f, optimal_rotation[7] = 0.0f, optimal_rotation[8] = 1.0f;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        applied_scale = centered_norm_squared_a > 0.0f ? trace_rotation_covariance / centered_norm_squared_a : 0.0f;
    }
    else {
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
        // R = V * Uᵀ
        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];

        nk_f32_t det = nk_det3x3_f32_(optimal_rotation);
        nk_f32_t sign_correction = det < 0 ? -1.0f : 1.0f;
        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
            optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
            optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
            optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
            optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
            optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
            optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
            optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
            optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        }
        nk_f32_t trace_ds = svd_diagonal[0] + svd_diagonal[4] + sign_correction * svd_diagonal[8];
        applied_scale = centered_norm_squared_a > 0.0f ? trace_ds / centered_norm_squared_a : 0.0f;
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    // Output rotation matrix and scale
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = applied_scale;

    // Folded SSD with scale: c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f32_t sum_squared = applied_scale * applied_scale * centered_norm_squared_a + centered_norm_squared_b -
                           2.0f * applied_scale * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load bf16, convert to f32, compute centroids, covariance, and ‖a‖²/‖b‖²
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 covariance_xx_f32x8 = zeros_f32x8, covariance_xy_f32x8 = zeros_f32x8, covariance_xz_f32x8 = zeros_f32x8;
    __m256 covariance_yx_f32x8 = zeros_f32x8, covariance_yy_f32x8 = zeros_f32x8, covariance_yz_f32x8 = zeros_f32x8;
    __m256 covariance_zx_f32x8 = zeros_f32x8, covariance_zy_f32x8 = zeros_f32x8, covariance_zz_f32x8 = zeros_f32x8;
    __m256 norm_squared_a_f32x8 = zeros_f32x8, norm_squared_b_f32x8 = zeros_f32x8;

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
        covariance_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, covariance_xx_f32x8);
        covariance_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, covariance_xy_f32x8);
        covariance_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, covariance_xz_f32x8);
        covariance_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, covariance_yx_f32x8);
        covariance_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, covariance_yy_f32x8);
        covariance_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, covariance_yz_f32x8);
        covariance_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, covariance_zx_f32x8);
        covariance_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, covariance_zy_f32x8);
        covariance_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, covariance_zz_f32x8);

        // Accumulate ‖a‖² and ‖b‖² for folded SSD
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_x_f32x8, a_x_f32x8, norm_squared_a_f32x8);
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_y_f32x8, a_y_f32x8, norm_squared_a_f32x8);
        norm_squared_a_f32x8 = _mm256_fmadd_ps(a_z_f32x8, a_z_f32x8, norm_squared_a_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_x_f32x8, b_x_f32x8, norm_squared_b_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_y_f32x8, b_y_f32x8, norm_squared_b_f32x8);
        norm_squared_b_f32x8 = _mm256_fmadd_ps(b_z_f32x8, b_z_f32x8, norm_squared_b_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t covariance_x_x = nk_reduce_add_f32x8_haswell_(covariance_xx_f32x8);
    nk_f32_t covariance_x_y = nk_reduce_add_f32x8_haswell_(covariance_xy_f32x8);
    nk_f32_t covariance_x_z = nk_reduce_add_f32x8_haswell_(covariance_xz_f32x8);
    nk_f32_t covariance_y_x = nk_reduce_add_f32x8_haswell_(covariance_yx_f32x8);
    nk_f32_t covariance_y_y = nk_reduce_add_f32x8_haswell_(covariance_yy_f32x8);
    nk_f32_t covariance_y_z = nk_reduce_add_f32x8_haswell_(covariance_yz_f32x8);
    nk_f32_t covariance_z_x = nk_reduce_add_f32x8_haswell_(covariance_zx_f32x8);
    nk_f32_t covariance_z_y = nk_reduce_add_f32x8_haswell_(covariance_zy_f32x8);
    nk_f32_t covariance_z_z = nk_reduce_add_f32x8_haswell_(covariance_zz_f32x8);
    nk_f32_t norm_squared_a_sum = nk_reduce_add_f32x8_haswell_(norm_squared_a_f32x8);
    nk_f32_t norm_squared_b_sum = nk_reduce_add_f32x8_haswell_(norm_squared_b_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_serial(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_serial(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_serial(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_serial(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_serial(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_serial(&b[i * 3 + 2], &bz);
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        covariance_x_x += ax * bx, covariance_x_y += ax * by, covariance_x_z += ax * bz;
        covariance_y_x += ay * bx, covariance_y_y += ay * by, covariance_y_z += ay * bz;
        covariance_z_x += az * bx, covariance_z_y += az * by, covariance_z_z += az * bz;
        norm_squared_a_sum += ax * ax + ay * ay + az * az;
        norm_squared_b_sum += bx * bx + by * by + bz * bz;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply centering correction to covariance matrix
    covariance_x_x -= (nk_f32_t)n * centroid_a_x * centroid_b_x;
    covariance_x_y -= (nk_f32_t)n * centroid_a_x * centroid_b_y;
    covariance_x_z -= (nk_f32_t)n * centroid_a_x * centroid_b_z;
    covariance_y_x -= (nk_f32_t)n * centroid_a_y * centroid_b_x;
    covariance_y_y -= (nk_f32_t)n * centroid_a_y * centroid_b_y;
    covariance_y_z -= (nk_f32_t)n * centroid_a_y * centroid_b_z;
    covariance_z_x -= (nk_f32_t)n * centroid_a_z * centroid_b_x;
    covariance_z_y -= (nk_f32_t)n * centroid_a_z * centroid_b_y;
    covariance_z_z -= (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // Centered ‖a-ā‖², ‖b-b̄‖² via the parallel-axis identity
    nk_f32_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

    // Identity-dominant short-circuit: if H is essentially diagonal with positive diagonals,
    // R = I and trace(DS) = trace(H).
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t applied_scale;
    nk_f32_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1.0f, optimal_rotation[1] = 0.0f, optimal_rotation[2] = 0.0f;
        optimal_rotation[3] = 0.0f, optimal_rotation[4] = 1.0f, optimal_rotation[5] = 0.0f;
        optimal_rotation[6] = 0.0f, optimal_rotation[7] = 0.0f, optimal_rotation[8] = 1.0f;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        applied_scale = centered_norm_squared_a > 0.0f ? trace_rotation_covariance / centered_norm_squared_a : 0.0f;
    }
    else {
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
        // R = V * Uᵀ
        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];

        nk_f32_t det = nk_det3x3_f32_(optimal_rotation);
        nk_f32_t sign_correction = det < 0 ? -1.0f : 1.0f;
        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
            optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
            optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
            optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
            optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
            optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
            optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
            optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
            optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        }
        nk_f32_t trace_ds = svd_diagonal[0] + svd_diagonal[4] + sign_correction * svd_diagonal[8];
        applied_scale = centered_norm_squared_a > 0.0f ? trace_ds / centered_norm_squared_a : 0.0f;
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    // Output rotation matrix and scale
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = applied_scale;

    // Folded SSD with scale: c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f32_t sum_squared = applied_scale * applied_scale * centered_norm_squared_a + centered_norm_squared_b -
                           2.0f * applied_scale * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
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
#endif // NK_TARGET_X8664_
#endif // NK_MESH_HASWELL_H
