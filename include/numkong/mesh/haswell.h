/**
 *  @file include/numkong/mesh/haswell.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief SIMD-accelerated point cloud alignment for Haswell.
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section haswell_mesh_instructions Key AVX2 Mesh Instructions
 *
 *  @verbatim
 *  Intrinsic               Instruction                     Haswell         Genoa
 *  _mm256_fmadd_ps         VFMADD (YMM, YMM, YMM)          5cy @ p01       4cy @ p01
 *  _mm256_hadd_ps          VHADDPS (YMM, YMM, YMM)         7cy @ p1+p5     4cy @ p123+p23+p23
 *  _mm256_permute2f128_ps  VPERM2F128 (YMM, YMM, YMM, I8)  3cy @ p5        2cy @ p12
 *  _mm256_extractf128_ps   VEXTRACTF128 (XMM, YMM, I8)     3cy @ p5        1cy @ p0123
 *  _mm256_i32gather_ps     VGATHERDPS (YMM, M, YMM, YMM)   22cy (34 uops)  19cy (17 uops)
 *  @endverbatim
 *
 *  Point cloud operations (centroid, covariance, Kabsch alignment) use gather instructions for
 *  stride-3 xyz deinterleaving. Multiple FMA accumulators hide the 5-cycle FMA latency. VHADDPS
 *  interleaves results across lanes, requiring additional shuffles for final scalar reduction.
 */
#ifndef NUMKONG_MESH_HASWELL_H
#define NUMKONG_MESH_HASWELL_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_HASWELL_

#include "numkong/types.h"
#include "numkong/dot/haswell.h"
#include "numkong/mesh/serial.h"
#include "numkong/reduce/haswell.h" // `nk_reduce_add_f32x8_haswell_`

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
 *  Output: x[8], y[8], z[8] vectors */
NUMKONG_INLINE void nk_deinterleave_f32x8_haswell_(nk_f32_t const *ptr, __m256 *x_out, __m256 *y_out, __m256 *z_out) {
    // Gather indices: 0, 3, 6, 9, 12, 15, 18, 21 (stride 3)
    __m256i index_i32x8 = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    *x_out = _mm256_i32gather_ps(ptr + 0, index_i32x8, 4);
    *y_out = _mm256_i32gather_ps(ptr + 1, index_i32x8, 4);
    *z_out = _mm256_i32gather_ps(ptr + 2, index_i32x8, 4);
}

/*  Deinterleave 12 f64 values (4 xyz triplets) into separate x, y, z vectors.
 *  Uses scalar extraction for simplicity as AVX2 lacks efficient stride-3 gather for f64.
 *
 *  Input: 12 contiguous f64 [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors */
NUMKONG_INLINE void nk_deinterleave_f64x4_haswell_(nk_f64_t const *ptr, __m256d *x_out, __m256d *y_out,
                                                   __m256d *z_out) {
    nk_f64_t x0 = ptr[0], x1 = ptr[3], x2 = ptr[6], x3 = ptr[9];
    nk_f64_t y0 = ptr[1], y1 = ptr[4], y2 = ptr[7], y3 = ptr[10];
    nk_f64_t z0 = ptr[2], z1 = ptr[5], z2 = ptr[8], z3 = ptr[11];

    *x_out = _mm256_setr_pd(x0, x1, x2, x3);
    *y_out = _mm256_setr_pd(y0, y1, y2, y3);
    *z_out = _mm256_setr_pd(z0, z1, z2, z3);
}

NUMKONG_INLINE nk_f64_t nk_reduce_stable_f64x4_haswell_(__m256d values_f64x4) {
    nk_b256_vec_t values;
    values.ymm_pd = values_f64x4;
    nk_f64_t sum = 0.0, compensation = 0.0;
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[0]);
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[1]);
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[2]);
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[3]);
    return sum + compensation;
}

/** Adds @p value² to the lanes of @p sum, with TwoProd and TwoSum errors into @p compensation. */
NUMKONG_INLINE void nk_accumulate_square_f64x4_haswell_(__m256d *sum_f64x4, __m256d *compensation_f64x4,
                                                        __m256d value_f64x4) {
    __m256d const product_f64x4 = _mm256_mul_pd(value_f64x4, value_f64x4);
    __m256d const product_error_f64x4 = _mm256_fmsub_pd(value_f64x4, value_f64x4, product_f64x4);
    __m256d const tentative_sum_f64x4 = _mm256_add_pd(*sum_f64x4, product_f64x4);
    __m256d const virtual_addend_f64x4 = _mm256_sub_pd(tentative_sum_f64x4, *sum_f64x4);
    __m256d const sum_error_f64x4 = _mm256_add_pd(
        _mm256_sub_pd(*sum_f64x4, _mm256_sub_pd(tentative_sum_f64x4, virtual_addend_f64x4)),
        _mm256_sub_pd(product_f64x4, virtual_addend_f64x4));
    *sum_f64x4 = tentative_sum_f64x4;
    *compensation_f64x4 = _mm256_add_pd(*compensation_f64x4, _mm256_add_pd(sum_error_f64x4, product_error_f64x4));
}

/** Σ‖s · R · (aᵢ − ā) − (bᵢ − b̄)‖² summed from the residuals, since folding it via trace(R · H)
 *  cancels to √ε once the clouds align. */
NUMKONG_INLINE nk_f64_t nk_transformed_ssd_f64_haswell_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                        nk_f64_t const *centroid_a, nk_f64_t const *centroid_b,
                                                        nk_f64_t const *rotation, nk_f64_t scale) {
    nk_f64_t r[9];
    for (int j = 0; j < 9; ++j) r[j] = scale * rotation[j];
    __m256d const centroid_a_x_f64x4 = _mm256_set1_pd(centroid_a[0]),
                  centroid_a_y_f64x4 = _mm256_set1_pd(centroid_a[1]),
                  centroid_a_z_f64x4 = _mm256_set1_pd(centroid_a[2]);
    __m256d const centroid_b_x_f64x4 = _mm256_set1_pd(centroid_b[0]),
                  centroid_b_y_f64x4 = _mm256_set1_pd(centroid_b[1]),
                  centroid_b_z_f64x4 = _mm256_set1_pd(centroid_b[2]);
    __m256d const r0_f64x4 = _mm256_set1_pd(r[0]), r1_f64x4 = _mm256_set1_pd(r[1]), r2_f64x4 = _mm256_set1_pd(r[2]);
    __m256d const r3_f64x4 = _mm256_set1_pd(r[3]), r4_f64x4 = _mm256_set1_pd(r[4]), r5_f64x4 = _mm256_set1_pd(r[5]);
    __m256d const r6_f64x4 = _mm256_set1_pd(r[6]), r7_f64x4 = _mm256_set1_pd(r[7]), r8_f64x4 = _mm256_set1_pd(r[8]);
    __m256d sum_squared_f64x4 = _mm256_setzero_pd(), compensation_f64x4 = _mm256_setzero_pd();
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);
        a_x_f64x4 = _mm256_sub_pd(a_x_f64x4, centroid_a_x_f64x4),
        a_y_f64x4 = _mm256_sub_pd(a_y_f64x4, centroid_a_y_f64x4),
        a_z_f64x4 = _mm256_sub_pd(a_z_f64x4, centroid_a_z_f64x4);
        b_x_f64x4 = _mm256_sub_pd(b_x_f64x4, centroid_b_x_f64x4),
        b_y_f64x4 = _mm256_sub_pd(b_y_f64x4, centroid_b_y_f64x4),
        b_z_f64x4 = _mm256_sub_pd(b_z_f64x4, centroid_b_z_f64x4);
        __m256d delta_x_f64x4 = _mm256_fmadd_pd(
            r2_f64x4, a_z_f64x4, _mm256_fmadd_pd(r1_f64x4, a_y_f64x4, _mm256_fmsub_pd(r0_f64x4, a_x_f64x4, b_x_f64x4)));
        __m256d delta_y_f64x4 = _mm256_fmadd_pd(
            r5_f64x4, a_z_f64x4, _mm256_fmadd_pd(r4_f64x4, a_y_f64x4, _mm256_fmsub_pd(r3_f64x4, a_x_f64x4, b_y_f64x4)));
        __m256d delta_z_f64x4 = _mm256_fmadd_pd(
            r8_f64x4, a_z_f64x4, _mm256_fmadd_pd(r7_f64x4, a_y_f64x4, _mm256_fmsub_pd(r6_f64x4, a_x_f64x4, b_z_f64x4)));
        nk_accumulate_square_f64x4_haswell_(&sum_squared_f64x4, &compensation_f64x4, delta_x_f64x4);
        nk_accumulate_square_f64x4_haswell_(&sum_squared_f64x4, &compensation_f64x4, delta_y_f64x4);
        nk_accumulate_square_f64x4_haswell_(&sum_squared_f64x4, &compensation_f64x4, delta_z_f64x4);
    }
    nk_f64_t sum_squared = nk_dot_stable_sum_f64x4_haswell_(sum_squared_f64x4, compensation_f64x4);
    nk_f64_t sum_squared_compensation = 0.0;
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0] - centroid_a[0], ay = a[i * 3 + 1] - centroid_a[1],
                 az = a[i * 3 + 2] - centroid_a[2];
        nk_f64_t bx = b[i * 3 + 0] - centroid_b[0], by = b[i * 3 + 1] - centroid_b[1],
                 bz = b[i * 3 + 2] - centroid_b[2];
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, r[0] * ax + r[1] * ay + r[2] * az - bx);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, r[3] * ax + r[4] * ay + r[5] * az - by);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, r[6] * ax + r[7] * ay + r[8] * az - bz);
    }
    return sum_squared + sum_squared_compensation;
}

/** Centroids, centered cross-covariance and ‖a − ā‖² of f32 clouds in a pivot-shifted f64 pass. */
NUMKONG_INLINE void nk_centered_moments_f32_haswell_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f64_t *centroid_a, nk_f64_t *centroid_b,
                                                     nk_f64_t *cross_covariance, nk_f64_t *centered_norm_squared_a) {
    nk_f64_t const pivot_a[3] = {a[0], a[1], a[2]}, pivot_b[3] = {b[0], b[1], b[2]};
    __m256d pivot_a_f64x4[3], pivot_b_f64x4[3], sum_a_f64x4[3], sum_b_f64x4[3], covariance_f64x4[9];
    __m256d norm_squared_a_f64x4 = _mm256_setzero_pd();
    for (int j = 0; j != 3; ++j)
        pivot_a_f64x4[j] = _mm256_set1_pd(pivot_a[j]), pivot_b_f64x4[j] = _mm256_set1_pd(pivot_b[j]),
        sum_a_f64x4[j] = _mm256_setzero_pd(), sum_b_f64x4[j] = _mm256_setzero_pd();
    for (int j = 0; j != 9; ++j) covariance_f64x4[j] = _mm256_setzero_pd();
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8[3], b_f32x8[3];
        nk_deinterleave_f32x8_haswell_(a + i * 3, &a_f32x8[0], &a_f32x8[1], &a_f32x8[2]);
        nk_deinterleave_f32x8_haswell_(b + i * 3, &b_f32x8[0], &b_f32x8[1], &b_f32x8[2]);
        for (int half = 0; half != 2; ++half) {
            __m256d a_f64x4[3], b_f64x4[3];
            for (int j = 0; j != 3; ++j) {
                a_f64x4[j] = _mm256_sub_pd(
                    _mm256_cvtps_pd(half ? _mm256_extractf128_ps(a_f32x8[j], 1) : _mm256_castps256_ps128(a_f32x8[j])),
                    pivot_a_f64x4[j]);
                b_f64x4[j] = _mm256_sub_pd(
                    _mm256_cvtps_pd(half ? _mm256_extractf128_ps(b_f32x8[j], 1) : _mm256_castps256_ps128(b_f32x8[j])),
                    pivot_b_f64x4[j]);
                sum_a_f64x4[j] = _mm256_add_pd(sum_a_f64x4[j], a_f64x4[j]);
                sum_b_f64x4[j] = _mm256_add_pd(sum_b_f64x4[j], b_f64x4[j]);
                norm_squared_a_f64x4 = _mm256_fmadd_pd(a_f64x4[j], a_f64x4[j], norm_squared_a_f64x4);
            }
            for (int j = 0; j != 9; ++j)
                covariance_f64x4[j] = _mm256_fmadd_pd(a_f64x4[j / 3], b_f64x4[j % 3], covariance_f64x4[j]);
        }
    }
    nk_f64_t sum_a[3], sum_b[3], covariance[9];
    nk_f64_t norm_squared_a = nk_reduce_stable_f64x4_haswell_(norm_squared_a_f64x4);
    for (int j = 0; j != 3; ++j)
        sum_a[j] = nk_reduce_stable_f64x4_haswell_(sum_a_f64x4[j]),
        sum_b[j] = nk_reduce_stable_f64x4_haswell_(sum_b_f64x4[j]);
    for (int j = 0; j != 9; ++j) covariance[j] = nk_reduce_stable_f64x4_haswell_(covariance_f64x4[j]);
    for (; i < n; ++i) {
        nk_f64_t a_point[3], b_point[3];
        for (int j = 0; j != 3; ++j) a_point[j] = a[i * 3 + j] - pivot_a[j], b_point[j] = b[i * 3 + j] - pivot_b[j];
        nk_centered_moments_update_f64_(a_point, b_point, sum_a, sum_b, covariance, &norm_squared_a);
    }
    nk_centered_moments_finalize_f64_(n, pivot_a, pivot_b, sum_a, sum_b, covariance, norm_squared_a, centroid_a,
                                      centroid_b, cross_covariance, centered_norm_squared_a);
}

/** Σ‖s · R · (aᵢ − ā) − (bᵢ − b̄)‖² of f32 clouds in f64, from residuals like the f64 variant. */
NUMKONG_INLINE nk_f64_t nk_transformed_ssd_f32_haswell_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                        nk_f64_t const *centroid_a, nk_f64_t const *centroid_b,
                                                        nk_f64_t const *rotation, nk_f64_t scale) {
    nk_f64_t r[9];
    __m256d r_f64x4[9], centroid_a_f64x4[3], centroid_b_f64x4[3];
    __m256d sum_squared_f64x4 = _mm256_setzero_pd(), compensation_f64x4 = _mm256_setzero_pd();
    for (int j = 0; j != 9; ++j) r[j] = scale * rotation[j], r_f64x4[j] = _mm256_set1_pd(r[j]);
    for (int j = 0; j != 3; ++j)
        centroid_a_f64x4[j] = _mm256_set1_pd(centroid_a[j]), centroid_b_f64x4[j] = _mm256_set1_pd(centroid_b[j]);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8[3], b_f32x8[3];
        nk_deinterleave_f32x8_haswell_(a + i * 3, &a_f32x8[0], &a_f32x8[1], &a_f32x8[2]);
        nk_deinterleave_f32x8_haswell_(b + i * 3, &b_f32x8[0], &b_f32x8[1], &b_f32x8[2]);
        for (int half = 0; half != 2; ++half) {
            __m256d a_f64x4[3], b_f64x4[3];
            for (int j = 0; j != 3; ++j) {
                a_f64x4[j] = _mm256_sub_pd(
                    _mm256_cvtps_pd(half ? _mm256_extractf128_ps(a_f32x8[j], 1) : _mm256_castps256_ps128(a_f32x8[j])),
                    centroid_a_f64x4[j]);
                b_f64x4[j] = _mm256_sub_pd(
                    _mm256_cvtps_pd(half ? _mm256_extractf128_ps(b_f32x8[j], 1) : _mm256_castps256_ps128(b_f32x8[j])),
                    centroid_b_f64x4[j]);
            }
            for (int j = 0; j != 3; ++j) {
                __m256d delta_f64x4 = _mm256_fmsub_pd(r_f64x4[j * 3], a_f64x4[0], b_f64x4[j]);
                delta_f64x4 = _mm256_fmadd_pd(r_f64x4[j * 3 + 1], a_f64x4[1], delta_f64x4);
                delta_f64x4 = _mm256_fmadd_pd(r_f64x4[j * 3 + 2], a_f64x4[2], delta_f64x4);
                nk_accumulate_square_f64x4_haswell_(&sum_squared_f64x4, &compensation_f64x4, delta_f64x4);
            }
        }
    }
    nk_f64_t sum_squared = nk_dot_stable_sum_f64x4_haswell_(sum_squared_f64x4, compensation_f64x4);
    nk_f64_t sum_squared_compensation = 0.0;
    for (; i < n; ++i) {
        nk_f64_t a_point[3], b_point[3];
        for (int j = 0; j != 3; ++j)
            a_point[j] = a[i * 3 + j] - centroid_a[j], b_point[j] = b[i * 3 + j] - centroid_b[j];
        nk_accumulate_residual_f64_(&sum_squared, &sum_squared_compensation, r, a_point, b_point);
    }
    return sum_squared + sum_squared_compensation;
}

#if NUMKONG_TARGET_HASWELL
NUMKONG_API nk_status_t nk_rmsd_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;

    if (n == 0) {
        *result = 0;
        return nk_success_k;
    }

    __m256d sum_sq_f64x4 = _mm256_setzero_pd();
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

        __m256d batch_sum_sq_f64x4 = _mm256_add_pd(_mm256_mul_pd(delta_x_low_f64x4, delta_x_low_f64x4),
                                                   _mm256_mul_pd(delta_x_high_f64x4, delta_x_high_f64x4));
        batch_sum_sq_f64x4 = _mm256_fmadd_pd(delta_y_low_f64x4, delta_y_low_f64x4, batch_sum_sq_f64x4);
        batch_sum_sq_f64x4 = _mm256_fmadd_pd(delta_y_high_f64x4, delta_y_high_f64x4, batch_sum_sq_f64x4);
        batch_sum_sq_f64x4 = _mm256_fmadd_pd(delta_z_low_f64x4, delta_z_low_f64x4, batch_sum_sq_f64x4);
        batch_sum_sq_f64x4 = _mm256_fmadd_pd(delta_z_high_f64x4, delta_z_high_f64x4, batch_sum_sq_f64x4);
        sum_sq_f64x4 = _mm256_add_pd(sum_sq_f64x4, batch_sum_sq_f64x4);
    }

    nk_f64_t sum_sq = nk_reduce_add_f64x4_haswell_(sum_sq_f64x4);

    for (; index < n; ++index) {
        nk_f64_t delta_x = (nk_f64_t)a[index * 3 + 0] - (nk_f64_t)b[index * 3 + 0];
        nk_f64_t delta_y = (nk_f64_t)a[index * 3 + 1] - (nk_f64_t)b[index * 3 + 1];
        nk_f64_t delta_z = (nk_f64_t)a[index * 3 + 2] - (nk_f64_t)b[index * 3 + 2];
        sum_sq += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(sum_sq / (nk_f64_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0;

    if (n == 0) {
        *result = 0;
        return nk_success_k;
    }
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    __m256d sum_sq_x_f64x4 = zeros_f64x4, sum_sq_y_f64x4 = zeros_f64x4, sum_sq_z_f64x4 = zeros_f64x4;

    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;
    nk_size_t i = 0;

    // Main loop with 2× unrolling
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        __m256d delta_x_f64x4 = _mm256_sub_pd(a_x_f64x4, b_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(a_y_f64x4, b_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(a_z_f64x4, b_z_f64x4);

        sum_sq_x_f64x4 = _mm256_fmadd_pd(delta_x_f64x4, delta_x_f64x4, sum_sq_x_f64x4);
        sum_sq_y_f64x4 = _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, sum_sq_y_f64x4);
        sum_sq_z_f64x4 = _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4, sum_sq_z_f64x4);

        __m256d a_x1_f64x4, a_y1_f64x4, a_z1_f64x4, b_x1_f64x4, b_y1_f64x4, b_z1_f64x4;
        nk_deinterleave_f64x4_haswell_(a + (i + 4) * 3, &a_x1_f64x4, &a_y1_f64x4, &a_z1_f64x4);
        nk_deinterleave_f64x4_haswell_(b + (i + 4) * 3, &b_x1_f64x4, &b_y1_f64x4, &b_z1_f64x4);

        __m256d delta_x1_f64x4 = _mm256_sub_pd(a_x1_f64x4, b_x1_f64x4);
        __m256d delta_y1_f64x4 = _mm256_sub_pd(a_y1_f64x4, b_y1_f64x4);
        __m256d delta_z1_f64x4 = _mm256_sub_pd(a_z1_f64x4, b_z1_f64x4);

        sum_sq_x_f64x4 = _mm256_fmadd_pd(delta_x1_f64x4, delta_x1_f64x4, sum_sq_x_f64x4);
        sum_sq_y_f64x4 = _mm256_fmadd_pd(delta_y1_f64x4, delta_y1_f64x4, sum_sq_y_f64x4);
        sum_sq_z_f64x4 = _mm256_fmadd_pd(delta_z1_f64x4, delta_z1_f64x4, sum_sq_z_f64x4);
    }

    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        __m256d delta_x_f64x4 = _mm256_sub_pd(a_x_f64x4, b_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(a_y_f64x4, b_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(a_z_f64x4, b_z_f64x4);

        sum_sq_x_f64x4 = _mm256_fmadd_pd(delta_x_f64x4, delta_x_f64x4, sum_sq_x_f64x4);
        sum_sq_y_f64x4 = _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, sum_sq_y_f64x4);
        sum_sq_z_f64x4 = _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4, sum_sq_z_f64x4);
    }

    nk_f64_t total_sq_x = nk_reduce_stable_f64x4_haswell_(sum_sq_x_f64x4), total_sq_x_compensation = 0.0;
    nk_f64_t total_sq_y = nk_reduce_stable_f64x4_haswell_(sum_sq_y_f64x4), total_sq_y_compensation = 0.0;
    nk_f64_t total_sq_z = nk_reduce_stable_f64x4_haswell_(sum_sq_z_f64x4), total_sq_z_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t delta_x = a[i * 3 + 0] - b[i * 3 + 0];
        nk_f64_t delta_y = a[i * 3 + 1] - b[i * 3 + 1];
        nk_f64_t delta_z = a[i * 3 + 2] - b[i * 3 + 2];
        nk_accumulate_square_f64_(&total_sq_x, &total_sq_x_compensation, delta_x);
        nk_accumulate_square_f64_(&total_sq_y, &total_sq_y_compensation, delta_y);
        nk_accumulate_square_f64_(&total_sq_z, &total_sq_z_compensation, delta_z);
    }

    total_sq_x += total_sq_x_compensation, total_sq_y += total_sq_y_compensation, total_sq_z += total_sq_z_compensation;

    *result = _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd((total_sq_x + total_sq_y + total_sq_z) / (nk_f64_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    nk_f64_t centroid_a[3], centroid_b[3], cross_covariance[9];
    nk_centered_moments_f32_haswell_(a, b, n, centroid_a, centroid_b, cross_covariance, NUMKONG_NULL);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a[0], a_centroid[1] = (nk_f32_t)centroid_a[1],
        a_centroid[2] = (nk_f32_t)centroid_a[2];
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b[0], b_centroid[1] = (nk_f32_t)centroid_b[1],
        b_centroid[2] = (nk_f32_t)centroid_b[2];

    // Identity-dominant short-circuit, skipping the SVD and rotation rebuilds for aligned inputs.
    nk_f64_t covariance_diagonal_norm_sq = cross_covariance[0] * cross_covariance[0] +
                                           cross_covariance[4] * cross_covariance[4] +
                                           cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_sq =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_sq < 1e-20 * covariance_diagonal_norm_sq && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
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
    }

    if (rotation)
        for (int j = 0; j != 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];
    if (scale) *scale = 1.0f;

    nk_f64_t sum_sq = nk_transformed_ssd_f32_haswell_(a, b, n, centroid_a, centroid_b, optimal_rotation, 1.0);
    *result = _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(sum_sq / (nk_f64_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                              nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                              nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0;
        *result = 0;
        return nk_success_k;
    }

    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    // Accumulators for centroids
    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;

    // Accumulators for covariance matrix (sum of outer products)
    __m256d covariance_xx_f64x4 = zeros_f64x4, covariance_xy_f64x4 = zeros_f64x4, covariance_xz_f64x4 = zeros_f64x4;
    __m256d covariance_yx_f64x4 = zeros_f64x4, covariance_yy_f64x4 = zeros_f64x4, covariance_yz_f64x4 = zeros_f64x4;
    __m256d covariance_zx_f64x4 = zeros_f64x4, covariance_zy_f64x4 = zeros_f64x4, covariance_zz_f64x4 = zeros_f64x4;
    // The first-point shift keeps the centering correction from cancelling far from the origin.
    __m256d const pivot_a_x_f64x4 = _mm256_set1_pd(a[0]), pivot_a_y_f64x4 = _mm256_set1_pd(a[1]),
                  pivot_a_z_f64x4 = _mm256_set1_pd(a[2]);
    __m256d const pivot_b_x_f64x4 = _mm256_set1_pd(b[0]), pivot_b_y_f64x4 = _mm256_set1_pd(b[1]),
                  pivot_b_z_f64x4 = _mm256_set1_pd(b[2]);

    nk_size_t i = 0;
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;

    // Fused single pass over centroids and covariance
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);
        a_x_f64x4 = _mm256_sub_pd(a_x_f64x4, pivot_a_x_f64x4), a_y_f64x4 = _mm256_sub_pd(a_y_f64x4, pivot_a_y_f64x4),
        a_z_f64x4 = _mm256_sub_pd(a_z_f64x4, pivot_a_z_f64x4);
        b_x_f64x4 = _mm256_sub_pd(b_x_f64x4, pivot_b_x_f64x4), b_y_f64x4 = _mm256_sub_pd(b_y_f64x4, pivot_b_y_f64x4),
        b_z_f64x4 = _mm256_sub_pd(b_z_f64x4, pivot_b_z_f64x4);

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

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0] - a[0], ay = a[i * 3 + 1] - a[1], az = a[i * 3 + 2] - a[2];
        nk_f64_t bx = b[i * 3 + 0] - b[0], by = b[i * 3 + 1] - b[1], bz = b[i * 3 + 2] - b[2];
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
    nk_f64_t mean_a_x = sum_a_x * inv_n, mean_a_y = sum_a_y * inv_n, mean_a_z = sum_a_z * inv_n;
    nk_f64_t mean_b_x = sum_b_x * inv_n, mean_b_y = sum_b_y * inv_n, mean_b_z = sum_b_z * inv_n;
    nk_f64_t centroid_a_x = a[0] + mean_a_x, centroid_a_y = a[1] + mean_a_y, centroid_a_z = a[2] + mean_a_z;
    nk_f64_t centroid_b_x = b[0] + mean_b_x, centroid_b_y = b[1] + mean_b_y, centroid_b_z = b[2] + mean_b_z;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply the centering correction to the sums shifted by the first point:
    // H_centered = H - n * mean_a * mean_bᵀ
    covariance_x_x -= (nk_f64_t)n * mean_a_x * mean_b_x;
    covariance_x_y -= (nk_f64_t)n * mean_a_x * mean_b_y;
    covariance_x_z -= (nk_f64_t)n * mean_a_x * mean_b_z;
    covariance_y_x -= (nk_f64_t)n * mean_a_y * mean_b_x;
    covariance_y_y -= (nk_f64_t)n * mean_a_y * mean_b_y;
    covariance_y_z -= (nk_f64_t)n * mean_a_y * mean_b_z;
    covariance_z_x -= (nk_f64_t)n * mean_a_z * mean_b_x;
    covariance_z_y -= (nk_f64_t)n * mean_a_z * mean_b_y;
    covariance_z_z -= (nk_f64_t)n * mean_a_z * mean_b_z;

    nk_f64_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // Identity-dominant short-circuit: if H is essentially diagonal with positive diagonals, R = I.
    nk_f64_t covariance_diagonal_norm_sq = cross_covariance[0] * cross_covariance[0] +
                                           cross_covariance[4] * cross_covariance[4] +
                                           cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_sq =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_sq < 1e-20 * covariance_diagonal_norm_sq && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1.0, optimal_rotation[1] = 0.0, optimal_rotation[2] = 0.0;
        optimal_rotation[3] = 0.0, optimal_rotation[4] = 1.0, optimal_rotation[5] = 0.0;
        optimal_rotation[6] = 0.0, optimal_rotation[7] = 0.0, optimal_rotation[8] = 1.0;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        if (nk_det3x3_f64_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
    }

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0;

    nk_f64_t const centroid_a[3] = {centroid_a_x, centroid_a_y, centroid_a_z};
    nk_f64_t const centroid_b[3] = {centroid_b_x, centroid_b_y, centroid_b_z};
    nk_f64_t sum_sq = nk_transformed_ssd_f64_haswell_(a, b, n, centroid_a, centroid_b, optimal_rotation, 1.0);
    *result = _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(sum_sq * inv_n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                               nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    nk_f64_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_sq_a;
    nk_centered_moments_f32_haswell_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_sq_a);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a[0], a_centroid[1] = (nk_f32_t)centroid_a[1],
        a_centroid[2] = (nk_f32_t)centroid_a[2];
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b[0], b_centroid[1] = (nk_f32_t)centroid_b[1],
        b_centroid[2] = (nk_f32_t)centroid_b[2];

    // Identity-dominant short-circuit: if H is essentially diagonal with positive diagonals,
    // R = I and trace(DS) reduces to trace(H) directly.
    nk_f64_t covariance_diagonal_norm_sq = cross_covariance[0] * cross_covariance[0] +
                                           cross_covariance[4] * cross_covariance[4] +
                                           cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_sq =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t applied_scale;
    if (covariance_offdiagonal_norm_sq < 1e-20 * covariance_diagonal_norm_sq && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1.0, optimal_rotation[1] = 0.0, optimal_rotation[2] = 0.0;
        optimal_rotation[3] = 0.0, optimal_rotation[4] = 1.0, optimal_rotation[5] = 0.0;
        optimal_rotation[6] = 0.0, optimal_rotation[7] = 0.0, optimal_rotation[8] = 1.0;
        applied_scale = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_sq_a;
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
        applied_scale = trace_ds / centered_norm_sq_a;
    }

    if (rotation)
        for (int j = 0; j != 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];
    if (scale) *scale = (nk_f32_t)applied_scale;

    nk_f64_t sum_sq = nk_transformed_ssd_f32_haswell_(a, b, n, centroid_a, centroid_b, optimal_rotation, applied_scale);
    *result = _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(sum_sq / (nk_f64_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                               nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                               nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0;
        *result = 0;
        return nk_success_k;
    }

    // Fused single-pass: centroids, covariance, and variance of A
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;
    __m256d covariance_xx_f64x4 = zeros_f64x4, covariance_xy_f64x4 = zeros_f64x4, covariance_xz_f64x4 = zeros_f64x4;
    __m256d covariance_yx_f64x4 = zeros_f64x4, covariance_yy_f64x4 = zeros_f64x4, covariance_yz_f64x4 = zeros_f64x4;
    __m256d covariance_zx_f64x4 = zeros_f64x4, covariance_zy_f64x4 = zeros_f64x4, covariance_zz_f64x4 = zeros_f64x4;
    __m256d norm_sq_a_f64x4 = zeros_f64x4;
    // The first-point shift keeps the centering correction from cancelling far from the origin.
    __m256d const pivot_a_x_f64x4 = _mm256_set1_pd(a[0]), pivot_a_y_f64x4 = _mm256_set1_pd(a[1]),
                  pivot_a_z_f64x4 = _mm256_set1_pd(a[2]);
    __m256d const pivot_b_x_f64x4 = _mm256_set1_pd(b[0]), pivot_b_y_f64x4 = _mm256_set1_pd(b[1]),
                  pivot_b_z_f64x4 = _mm256_set1_pd(b[2]);

    nk_size_t i = 0;
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;

    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);
        a_x_f64x4 = _mm256_sub_pd(a_x_f64x4, pivot_a_x_f64x4), a_y_f64x4 = _mm256_sub_pd(a_y_f64x4, pivot_a_y_f64x4),
        a_z_f64x4 = _mm256_sub_pd(a_z_f64x4, pivot_a_z_f64x4);
        b_x_f64x4 = _mm256_sub_pd(b_x_f64x4, pivot_b_x_f64x4), b_y_f64x4 = _mm256_sub_pd(b_y_f64x4, pivot_b_y_f64x4),
        b_z_f64x4 = _mm256_sub_pd(b_z_f64x4, pivot_b_z_f64x4);

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
        norm_sq_a_f64x4 = _mm256_fmadd_pd(a_x_f64x4, a_x_f64x4, norm_sq_a_f64x4);
        norm_sq_a_f64x4 = _mm256_fmadd_pd(a_y_f64x4, a_y_f64x4, norm_sq_a_f64x4);
        norm_sq_a_f64x4 = _mm256_fmadd_pd(a_z_f64x4, a_z_f64x4, norm_sq_a_f64x4);
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
    nk_f64_t norm_sq_a_sum = nk_reduce_stable_f64x4_haswell_(norm_sq_a_f64x4), norm_sq_a_compensation = 0.0;

    // Scalar tail loop for remaining points
    for (; i < n; i++) {
        nk_f64_t ax = a[i * 3 + 0] - a[0], ay = a[i * 3 + 1] - a[1], az = a[i * 3 + 2] - a[2];
        nk_f64_t bx = b[i * 3 + 0] - b[0], by = b[i * 3 + 1] - b[1], bz = b[i * 3 + 2] - b[2];
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
        nk_accumulate_square_f64_(&norm_sq_a_sum, &norm_sq_a_compensation, ax);
        nk_accumulate_square_f64_(&norm_sq_a_sum, &norm_sq_a_compensation, ay);
        nk_accumulate_square_f64_(&norm_sq_a_sum, &norm_sq_a_compensation, az);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    norm_sq_a_sum += norm_sq_a_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;

    nk_f64_t mean_a_x = sum_a_x * inv_n, mean_a_y = sum_a_y * inv_n, mean_a_z = sum_a_z * inv_n;
    nk_f64_t mean_b_x = sum_b_x * inv_n, mean_b_y = sum_b_y * inv_n, mean_b_z = sum_b_z * inv_n;
    nk_f64_t centroid_a_x = a[0] + mean_a_x, centroid_a_y = a[1] + mean_a_y, centroid_a_z = a[2] + mean_a_z;
    nk_f64_t centroid_b_x = b[0] + mean_b_x, centroid_b_y = b[1] + mean_b_y, centroid_b_z = b[2] + mean_b_z;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Centered norm-squared via parallel-axis identity; clamped at zero for numeric safety.
    nk_f64_t centered_norm_sq_a = norm_sq_a_sum -
                                  (nk_f64_t)n * (mean_a_x * mean_a_x + mean_a_y * mean_a_y + mean_a_z * mean_a_z);
    if (centered_norm_sq_a < 0.0) centered_norm_sq_a = 0.0;

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
    nk_f64_t covariance_diagonal_norm_sq = cross_covariance[0] * cross_covariance[0] +
                                           cross_covariance[4] * cross_covariance[4] +
                                           cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_sq =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t c;
    if (covariance_offdiagonal_norm_sq < 1e-20 * covariance_diagonal_norm_sq && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1.0, optimal_rotation[1] = 0.0, optimal_rotation[2] = 0.0;
        optimal_rotation[3] = 0.0, optimal_rotation[4] = 1.0, optimal_rotation[5] = 0.0;
        optimal_rotation[6] = 0.0, optimal_rotation[7] = 0.0, optimal_rotation[8] = 1.0;
        c = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_sq_a;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);

        nk_f64_t det = nk_det3x3_f64_(optimal_rotation);
        nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
        nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_diagonal[0], 1.0, svd_diagonal[4], 1.0, svd_diagonal[8], d3);
        c = trace_ds / centered_norm_sq_a;

        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
    }

    if (scale) *scale = c;
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    nk_f64_t const centroid_a[3] = {centroid_a_x, centroid_a_y, centroid_a_z};
    nk_f64_t const centroid_b[3] = {centroid_b_x, centroid_b_y, centroid_b_z};
    nk_f64_t sum_sq = nk_transformed_ssd_f64_haswell_(a, b, n, centroid_a, centroid_b, optimal_rotation, c);
    *result = _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(sum_sq * inv_n)));
    return nk_success_k;
}
#endif // NUMKONG_TARGET_HASWELL

/*  Deinterleave 8 f16 xyz triplets (24 f16 values) and convert to 3 x __m256 f32.
 *  Uses scalar extraction for clean stride-3 access, then F16C conversion.
 *
 *  Input: 24 contiguous f16 [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
 *  Output: x[8], y[8], z[8] vectors in f32 */
NUMKONG_INLINE void nk_deinterleave_f16x8_to_f32x8_haswell_(nk_f16_t const *ptr, __m256 *x_out, __m256 *y_out,
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
 *  Output: x[8], y[8], z[8] vectors in f32 */
NUMKONG_INLINE void nk_deinterleave_bf16x8_to_f32x8_haswell_(nk_bf16_t const *ptr, __m256 *x_out, __m256 *y_out,
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

/** Folds 8 widened, pivot-shifted points into sums for @ref nk_centered_moments_finalize_f32_. */
NUMKONG_INLINE void nk_centered_moments_update_f32x8_haswell_(__m256 const *a_f32x8, __m256 const *b_f32x8,
                                                              __m256 *sum_a_f32x8, __m256 *sum_b_f32x8,
                                                              __m256 *covariance_f32x8, __m256 *norm_squared_a_f32x8,
                                                              __m256 *norm_squared_b_f32x8) {
    for (int j = 0; j != 3; ++j) {
        sum_a_f32x8[j] = _mm256_add_ps(sum_a_f32x8[j], a_f32x8[j]);
        sum_b_f32x8[j] = _mm256_add_ps(sum_b_f32x8[j], b_f32x8[j]);
        *norm_squared_a_f32x8 = _mm256_fmadd_ps(a_f32x8[j], a_f32x8[j], *norm_squared_a_f32x8);
        *norm_squared_b_f32x8 = _mm256_fmadd_ps(b_f32x8[j], b_f32x8[j], *norm_squared_b_f32x8);
    }
    for (int j = 0; j != 9; ++j)
        covariance_f32x8[j] = _mm256_fmadd_ps(a_f32x8[j / 3], b_f32x8[j % 3], covariance_f32x8[j]);
}

/** Centroids, centered cross-covariance, ‖a − ā‖² and ‖b − b̄‖² of f16 clouds,
 *  in one pass shifted by the pivots in f32. */
NUMKONG_INLINE void nk_centered_moments_f16_haswell_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f32_t *centroid_a, nk_f32_t *centroid_b,
                                                     nk_f32_t *cross_covariance, nk_f32_t *centered_norm_squared_a,
                                                     nk_f32_t *centered_norm_squared_b) {
    nk_f32_t pivot_a[3], pivot_b[3];
    for (int j = 0; j != 3; ++j) nk_f16_to_f32_(a + j, pivot_a + j), nk_f16_to_f32_(b + j, pivot_b + j);
    __m256 pivot_a_f32x8[3], pivot_b_f32x8[3], sum_a_f32x8[3], sum_b_f32x8[3], covariance_f32x8[9];
    __m256 norm_squared_a_f32x8 = _mm256_setzero_ps(), norm_squared_b_f32x8 = _mm256_setzero_ps();
    for (int j = 0; j != 3; ++j)
        pivot_a_f32x8[j] = _mm256_set1_ps(pivot_a[j]), pivot_b_f32x8[j] = _mm256_set1_ps(pivot_b[j]),
        sum_a_f32x8[j] = _mm256_setzero_ps(), sum_b_f32x8[j] = _mm256_setzero_ps();
    for (int j = 0; j != 9; ++j) covariance_f32x8[j] = _mm256_setzero_ps();
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8[3], b_f32x8[3];
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + i * 3, &a_f32x8[0], &a_f32x8[1], &a_f32x8[2]);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + i * 3, &b_f32x8[0], &b_f32x8[1], &b_f32x8[2]);
        for (int j = 0; j != 3; ++j)
            a_f32x8[j] = _mm256_sub_ps(a_f32x8[j], pivot_a_f32x8[j]),
            b_f32x8[j] = _mm256_sub_ps(b_f32x8[j], pivot_b_f32x8[j]);
        nk_centered_moments_update_f32x8_haswell_(a_f32x8, b_f32x8, sum_a_f32x8, sum_b_f32x8, covariance_f32x8,
                                                  &norm_squared_a_f32x8, &norm_squared_b_f32x8);
    }
    nk_f32_t sum_a[3], sum_b[3], covariance[9], norm_squared_a = nk_reduce_add_f32x8_haswell_(norm_squared_a_f32x8),
                                                norm_squared_b = nk_reduce_add_f32x8_haswell_(norm_squared_b_f32x8);
    for (int j = 0; j != 3; ++j)
        sum_a[j] = nk_reduce_add_f32x8_haswell_(sum_a_f32x8[j]),
        sum_b[j] = nk_reduce_add_f32x8_haswell_(sum_b_f32x8[j]);
    for (int j = 0; j != 9; ++j) covariance[j] = nk_reduce_add_f32x8_haswell_(covariance_f32x8[j]);
    for (; i < n; ++i) {
        nk_f32_t a_point[3], b_point[3];
        for (int j = 0; j != 3; ++j) {
            nk_f16_to_f32_(a + i * 3 + j, a_point + j), nk_f16_to_f32_(b + i * 3 + j, b_point + j);
            a_point[j] -= pivot_a[j], b_point[j] -= pivot_b[j];
        }
        nk_centered_moments_update_f32_(a_point, b_point, sum_a, sum_b, covariance, &norm_squared_a, &norm_squared_b);
    }
    nk_centered_moments_finalize_f32_(n, pivot_a, pivot_b, sum_a, sum_b, covariance, norm_squared_a, norm_squared_b,
                                      centroid_a, centroid_b, cross_covariance, centered_norm_squared_a,
                                      centered_norm_squared_b);
}

/** Centroids, centered cross-covariance, ‖a − ā‖² and ‖b − b̄‖² of bf16 clouds,
 *  in one pass shifted by the pivots in f32. */
NUMKONG_INLINE void nk_centered_moments_bf16_haswell_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_f32_t *centroid_a, nk_f32_t *centroid_b,
                                                      nk_f32_t *cross_covariance, nk_f32_t *centered_norm_squared_a,
                                                      nk_f32_t *centered_norm_squared_b) {
    nk_f32_t pivot_a[3], pivot_b[3];
    for (int j = 0; j != 3; ++j) nk_bf16_to_f32_(a + j, pivot_a + j), nk_bf16_to_f32_(b + j, pivot_b + j);
    __m256 pivot_a_f32x8[3], pivot_b_f32x8[3], sum_a_f32x8[3], sum_b_f32x8[3], covariance_f32x8[9];
    __m256 norm_squared_a_f32x8 = _mm256_setzero_ps(), norm_squared_b_f32x8 = _mm256_setzero_ps();
    for (int j = 0; j != 3; ++j)
        pivot_a_f32x8[j] = _mm256_set1_ps(pivot_a[j]), pivot_b_f32x8[j] = _mm256_set1_ps(pivot_b[j]),
        sum_a_f32x8[j] = _mm256_setzero_ps(), sum_b_f32x8[j] = _mm256_setzero_ps();
    for (int j = 0; j != 9; ++j) covariance_f32x8[j] = _mm256_setzero_ps();
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8[3], b_f32x8[3];
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + i * 3, &a_f32x8[0], &a_f32x8[1], &a_f32x8[2]);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + i * 3, &b_f32x8[0], &b_f32x8[1], &b_f32x8[2]);
        for (int j = 0; j != 3; ++j)
            a_f32x8[j] = _mm256_sub_ps(a_f32x8[j], pivot_a_f32x8[j]),
            b_f32x8[j] = _mm256_sub_ps(b_f32x8[j], pivot_b_f32x8[j]);
        nk_centered_moments_update_f32x8_haswell_(a_f32x8, b_f32x8, sum_a_f32x8, sum_b_f32x8, covariance_f32x8,
                                                  &norm_squared_a_f32x8, &norm_squared_b_f32x8);
    }
    nk_f32_t sum_a[3], sum_b[3], covariance[9], norm_squared_a = nk_reduce_add_f32x8_haswell_(norm_squared_a_f32x8),
                                                norm_squared_b = nk_reduce_add_f32x8_haswell_(norm_squared_b_f32x8);
    for (int j = 0; j != 3; ++j)
        sum_a[j] = nk_reduce_add_f32x8_haswell_(sum_a_f32x8[j]),
        sum_b[j] = nk_reduce_add_f32x8_haswell_(sum_b_f32x8[j]);
    for (int j = 0; j != 9; ++j) covariance[j] = nk_reduce_add_f32x8_haswell_(covariance_f32x8[j]);
    for (; i < n; ++i) {
        nk_f32_t a_point[3], b_point[3];
        for (int j = 0; j != 3; ++j) {
            nk_bf16_to_f32_(a + i * 3 + j, a_point + j), nk_bf16_to_f32_(b + i * 3 + j, b_point + j);
            a_point[j] -= pivot_a[j], b_point[j] -= pivot_b[j];
        }
        nk_centered_moments_update_f32_(a_point, b_point, sum_a, sum_b, covariance, &norm_squared_a, &norm_squared_b);
    }
    nk_centered_moments_finalize_f32_(n, pivot_a, pivot_b, sum_a, sum_b, covariance, norm_squared_a, norm_squared_b,
                                      centroid_a, centroid_b, cross_covariance, centered_norm_squared_a,
                                      centered_norm_squared_b);
}

#if NUMKONG_TARGET_HASWELL
NUMKONG_API nk_status_t nk_rmsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;

    if (n == 0) {
        *result = 0;
        return nk_success_k;
    }
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    __m256 const zeros_f32x8 = _mm256_setzero_ps();
    __m256 sum_sq_x_f32x8 = zeros_f32x8, sum_sq_y_f32x8 = zeros_f32x8, sum_sq_z_f32x8 = zeros_f32x8;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);
        __m256 delta_x_f32x8 = _mm256_sub_ps(a_x_f32x8, b_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(a_y_f32x8, b_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(a_z_f32x8, b_z_f32x8);
        sum_sq_x_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_sq_x_f32x8);
        sum_sq_y_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_sq_y_f32x8);
        sum_sq_z_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_sq_z_f32x8);
    }

    nk_f32_t sum_sq = nk_reduce_add_f32x8_haswell_(sum_sq_x_f32x8) + nk_reduce_add_f32x8_haswell_(sum_sq_y_f32x8) +
                      nk_reduce_add_f32x8_haswell_(sum_sq_z_f32x8);
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_(&a[i * 3 + 2], &az);
        nk_f16_to_f32_(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_(&b[i * 3 + 1], &by);
        nk_f16_to_f32_(&b[i * 3 + 2], &bz);
        nk_f32_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        sum_sq += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(sum_sq / (nk_f32_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsd_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                             nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;

    if (n == 0) {
        *result = 0;
        return nk_success_k;
    }
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    __m256 const zeros_f32x8 = _mm256_setzero_ps();
    __m256 sum_sq_x_f32x8 = zeros_f32x8, sum_sq_y_f32x8 = zeros_f32x8, sum_sq_z_f32x8 = zeros_f32x8;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);
        __m256 delta_x_f32x8 = _mm256_sub_ps(a_x_f32x8, b_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(a_y_f32x8, b_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(a_z_f32x8, b_z_f32x8);
        sum_sq_x_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_sq_x_f32x8);
        sum_sq_y_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_sq_y_f32x8);
        sum_sq_z_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_sq_z_f32x8);
    }

    nk_f32_t sum_sq = nk_reduce_add_f32x8_haswell_(sum_sq_x_f32x8) + nk_reduce_add_f32x8_haswell_(sum_sq_y_f32x8) +
                      nk_reduce_add_f32x8_haswell_(sum_sq_z_f32x8);
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_(&b[i * 3 + 2], &bz);
        nk_f32_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        sum_sq += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(sum_sq / (nk_f32_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_sq_a, centered_norm_sq_b;
    nk_centered_moments_f16_haswell_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_sq_a,
                                     &centered_norm_sq_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    // Identity-dominant short-circuit, skipping the SVD and rotation rebuilds for aligned inputs.
    nk_f32_t covariance_diagonal_norm_sq = cross_covariance[0] * cross_covariance[0] +
                                           cross_covariance[4] * cross_covariance[4] +
                                           cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_sq =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_sq < 1e-12f * covariance_diagonal_norm_sq && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1.0f, optimal_rotation[1] = 0.0f, optimal_rotation[2] = 0.0f;
        optimal_rotation[3] = 0.0f, optimal_rotation[4] = 1.0f, optimal_rotation[5] = 0.0f;
        optimal_rotation[6] = 0.0f, optimal_rotation[7] = 0.0f, optimal_rotation[8] = 1.0f;
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
    }

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    nk_f32_t sum_sq = nk_folded_ssd_f32_(optimal_rotation, 1.0f, cross_covariance, centered_norm_sq_a,
                                         centered_norm_sq_b);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(sum_sq / (nk_f32_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                               nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                               nk_f32_t *scale, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_sq_a, centered_norm_sq_b;
    nk_centered_moments_bf16_haswell_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_sq_a,
                                      &centered_norm_sq_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    // Identity-dominant short-circuit, skipping the SVD and rotation rebuilds for aligned inputs.
    nk_f32_t covariance_diagonal_norm_sq = cross_covariance[0] * cross_covariance[0] +
                                           cross_covariance[4] * cross_covariance[4] +
                                           cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_sq =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_sq < 1e-12f * covariance_diagonal_norm_sq && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1.0f, optimal_rotation[1] = 0.0f, optimal_rotation[2] = 0.0f;
        optimal_rotation[3] = 0.0f, optimal_rotation[4] = 1.0f, optimal_rotation[5] = 0.0f;
        optimal_rotation[6] = 0.0f, optimal_rotation[7] = 0.0f, optimal_rotation[8] = 1.0f;
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
    }

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    nk_f32_t sum_sq = nk_folded_ssd_f32_(optimal_rotation, 1.0f, cross_covariance, centered_norm_sq_a,
                                         centered_norm_sq_b);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(sum_sq / (nk_f32_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                               nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_sq_a, centered_norm_sq_b;
    nk_centered_moments_f16_haswell_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_sq_a,
                                     &centered_norm_sq_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    // Identity-dominant short-circuit: if H is essentially diagonal with positive diagonals,
    // R = I and trace(DS) = trace(H).
    nk_f32_t covariance_diagonal_norm_sq = cross_covariance[0] * cross_covariance[0] +
                                           cross_covariance[4] * cross_covariance[4] +
                                           cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_sq =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t applied_scale;
    if (covariance_offdiagonal_norm_sq < 1e-12f * covariance_diagonal_norm_sq && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1.0f, optimal_rotation[1] = 0.0f, optimal_rotation[2] = 0.0f;
        optimal_rotation[3] = 0.0f, optimal_rotation[4] = 1.0f, optimal_rotation[5] = 0.0f;
        optimal_rotation[6] = 0.0f, optimal_rotation[7] = 0.0f, optimal_rotation[8] = 1.0f;
        applied_scale = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_sq_a;
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
        applied_scale = trace_ds / centered_norm_sq_a;
    }

    // Output rotation matrix and scale
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = applied_scale;

    nk_f32_t sum_sq = nk_folded_ssd_f32_(optimal_rotation, applied_scale, cross_covariance, centered_norm_sq_a,
                                         centered_norm_sq_b);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(sum_sq / (nk_f32_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                                nk_f32_t *scale, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_sq_a, centered_norm_sq_b;
    nk_centered_moments_bf16_haswell_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_sq_a,
                                      &centered_norm_sq_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    // Identity-dominant short-circuit: if H is essentially diagonal with positive diagonals,
    // R = I and trace(DS) = trace(H).
    nk_f32_t covariance_diagonal_norm_sq = cross_covariance[0] * cross_covariance[0] +
                                           cross_covariance[4] * cross_covariance[4] +
                                           cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_sq =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t applied_scale;
    if (covariance_offdiagonal_norm_sq < 1e-12f * covariance_diagonal_norm_sq && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1.0f, optimal_rotation[1] = 0.0f, optimal_rotation[2] = 0.0f;
        optimal_rotation[3] = 0.0f, optimal_rotation[4] = 1.0f, optimal_rotation[5] = 0.0f;
        optimal_rotation[6] = 0.0f, optimal_rotation[7] = 0.0f, optimal_rotation[8] = 1.0f;
        applied_scale = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_sq_a;
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
        applied_scale = trace_ds / centered_norm_sq_a;
    }

    // Output rotation matrix and scale
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = applied_scale;

    nk_f32_t sum_sq = nk_folded_ssd_f32_(optimal_rotation, applied_scale, cross_covariance, centered_norm_sq_a,
                                         centered_norm_sq_b);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(sum_sq / (nk_f32_t)n)));
    return nk_success_k;
}
#endif // NUMKONG_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_HASWELL_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_MESH_HASWELL_H
