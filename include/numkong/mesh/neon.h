/**
 *  @file include/numkong/mesh/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief SIMD-accelerated point cloud alignment for NEON.
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section neon_mesh_instructions Key NEON Mesh Instructions
 *
 *  Point cloud operations use these ARM NEON instructions:
 *
 *  @verbatim
 *  Intrinsic    Instruction                    A76       M5
 *  vfmaq_f32    FMLA (V.4S, V.4S, V.4S)        4cy @ 2p  3cy @ 4p
 *  vmulq_n_f32  FMUL (V.4S, V.4S, V.S[0])      3cy @ 2p  3cy @ 4p
 *  vsubq_f32    FSUB (V.4S, V.4S, V.4S)        2cy @ 2p  2cy @ 4p
 *  vaddvq_f32   FADDP+FADDP (reduce)           5cy @ 1p  8cy @ 1p
 *  vld3q_f32    LD3 ({Vt.4S, Vt2.4S, Vt3.4S})  4cy @ 1p  4cy @ 1p
 *  @endverbatim
 *
 *  LD3 provides hardware stride-3 deinterleaving for XYZ point data. The 6cy latency and 1/cy
 *  throughput make it the memory bottleneck regardless of core microarchitecture.
 *
 *  FMA throughput doubles on 4-pipe cores — Apple M4+, Graviton3+, Oryon. Using 2x loop unrolling
 *  with independent accumulators hides FMA latency and saturates 2 FP pipes on A76-class cores; 4x
 *  unrolling may further benefit 4-pipe cores.
 */
#ifndef NUMKONG_MESH_NEON_H
#define NUMKONG_MESH_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/types.h"
#include "numkong/dot/neon.h"
#include "numkong/mesh/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NUMKONG_INLINE void nk_deinterleave_f32x4_neon_(nk_f32_t const *ptr, float32x4_t *x_out, float32x4_t *y_out,
                                                float32x4_t *z_out) {
    // Deinterleave 12 floats (4 xyz triplets) into separate x, y, z vectors.
    // Uses NEON vld3q for efficient stride-3 deinterleaving.
    //
    // Input: 12 contiguous floats [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
    // Output: x[4], y[4], z[4] vectors
    float32x4x3_t xyz_f32x4x3 = vld3q_f32(ptr);
    *x_out = xyz_f32x4x3.val[0];
    *y_out = xyz_f32x4x3.val[1];
    *z_out = xyz_f32x4x3.val[2];
}

NUMKONG_INLINE void nk_deinterleave_f64x2_neon_(nk_f64_t const *ptr, float64x2_t *x_out, float64x2_t *y_out,
                                                float64x2_t *z_out) {
    // Deinterleave 6 f64 values (2 xyz triplets) into separate x, y, z vectors.
    //
    // Input: 6 contiguous f64 [x0,y0,z0, x1,y1,z1]
    // Output: x[2], y[2], z[2] vectors
    // NEON doesn't have vld3q_f64, so we use vcombine to avoid stack round-trips
    // Load 2 xyz triplets: [x0,y0,z0, x1,y1,z1]
    *x_out = vcombine_f64(vld1_f64(&ptr[0]), vld1_f64(&ptr[3]));
    *y_out = vcombine_f64(vld1_f64(&ptr[1]), vld1_f64(&ptr[4]));
    *z_out = vcombine_f64(vld1_f64(&ptr[2]), vld1_f64(&ptr[5]));
}

NUMKONG_INLINE nk_f64_t nk_reduce_stable_f64x2_neon_(float64x2_t values_f64x2) {
    nk_b128_vec_t values;
    values.f64x2 = values_f64x2;
    nk_f64_t sum = 0.0, compensation = 0.0;
    nk_accumulate_sum_f64_serial_(&sum, &compensation, values.f64s[0]);
    nk_accumulate_sum_f64_serial_(&sum, &compensation, values.f64s[1]);
    return sum + compensation;
}

/** Adds @p value² to the lanes of @p sum, with TwoProd and TwoSum errors into @p compensation. */
NUMKONG_INLINE void nk_accumulate_square_f64x2_neon_(float64x2_t *sum_f64x2, float64x2_t *compensation_f64x2,
                                                     float64x2_t value_f64x2) {
    float64x2_t const product_f64x2 = vmulq_f64(value_f64x2, value_f64x2);
    float64x2_t const product_error_f64x2 = vnegq_f64(vfmsq_f64(product_f64x2, value_f64x2, value_f64x2));
    float64x2_t const tentative_sum_f64x2 = vaddq_f64(*sum_f64x2, product_f64x2);
    float64x2_t const virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, *sum_f64x2);
    float64x2_t const sum_error_f64x2 = vaddq_f64(
        vsubq_f64(*sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
        vsubq_f64(product_f64x2, virtual_addend_f64x2));
    *sum_f64x2 = tentative_sum_f64x2;
    *compensation_f64x2 = vaddq_f64(*compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));
}

/** Σ‖s · R · (aᵢ − ā) − (bᵢ − b̄)‖² summed from the residuals, since folding it via trace(R · H)
 *  cancels to √ε once the clouds align. */
NUMKONG_INLINE nk_f64_t nk_transformed_ssd_f64_neon_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t const *centroid_a, nk_f64_t const *centroid_b,
                                                     nk_f64_t const *rotation, nk_f64_t scale) {
    nk_f64_t r[9];
    for (int j = 0; j < 9; ++j) r[j] = scale * rotation[j];
    float64x2_t const centroid_a_x_f64x2 = vdupq_n_f64(centroid_a[0]), centroid_a_y_f64x2 = vdupq_n_f64(centroid_a[1]),
                      centroid_a_z_f64x2 = vdupq_n_f64(centroid_a[2]);
    float64x2_t const centroid_b_x_f64x2 = vdupq_n_f64(centroid_b[0]), centroid_b_y_f64x2 = vdupq_n_f64(centroid_b[1]),
                      centroid_b_z_f64x2 = vdupq_n_f64(centroid_b[2]);
    float64x2_t const r0_f64x2 = vdupq_n_f64(r[0]), r1_f64x2 = vdupq_n_f64(r[1]), r2_f64x2 = vdupq_n_f64(r[2]);
    float64x2_t const r3_f64x2 = vdupq_n_f64(r[3]), r4_f64x2 = vdupq_n_f64(r[4]), r5_f64x2 = vdupq_n_f64(r[5]);
    float64x2_t const r6_f64x2 = vdupq_n_f64(r[6]), r7_f64x2 = vdupq_n_f64(r[7]), r8_f64x2 = vdupq_n_f64(r[8]);
    float64x2_t sum_squared_f64x2 = vdupq_n_f64(0), compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);
        a_x_f64x2 = vsubq_f64(a_x_f64x2, centroid_a_x_f64x2), a_y_f64x2 = vsubq_f64(a_y_f64x2, centroid_a_y_f64x2),
        a_z_f64x2 = vsubq_f64(a_z_f64x2, centroid_a_z_f64x2);
        b_x_f64x2 = vsubq_f64(b_x_f64x2, centroid_b_x_f64x2), b_y_f64x2 = vsubq_f64(b_y_f64x2, centroid_b_y_f64x2),
        b_z_f64x2 = vsubq_f64(b_z_f64x2, centroid_b_z_f64x2);
        float64x2_t delta_x_f64x2 = vfmaq_f64(
            vfmaq_f64(vfmaq_f64(vnegq_f64(b_x_f64x2), r0_f64x2, a_x_f64x2), r1_f64x2, a_y_f64x2), r2_f64x2, a_z_f64x2);
        float64x2_t delta_y_f64x2 = vfmaq_f64(
            vfmaq_f64(vfmaq_f64(vnegq_f64(b_y_f64x2), r3_f64x2, a_x_f64x2), r4_f64x2, a_y_f64x2), r5_f64x2, a_z_f64x2);
        float64x2_t delta_z_f64x2 = vfmaq_f64(
            vfmaq_f64(vfmaq_f64(vnegq_f64(b_z_f64x2), r6_f64x2, a_x_f64x2), r7_f64x2, a_y_f64x2), r8_f64x2, a_z_f64x2);
        nk_accumulate_square_f64x2_neon_(&sum_squared_f64x2, &compensation_f64x2, delta_x_f64x2);
        nk_accumulate_square_f64x2_neon_(&sum_squared_f64x2, &compensation_f64x2, delta_y_f64x2);
        nk_accumulate_square_f64x2_neon_(&sum_squared_f64x2, &compensation_f64x2, delta_z_f64x2);
    }
    nk_f64_t sum_squared = nk_dot_stable_sum_f64x2_neon_(sum_squared_f64x2, compensation_f64x2);
    nk_f64_t sum_squared_compensation = 0.0;
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0] - centroid_a[0], ay = a[i * 3 + 1] - centroid_a[1],
                 az = a[i * 3 + 2] - centroid_a[2];
        nk_f64_t bx = b[i * 3 + 0] - centroid_b[0], by = b[i * 3 + 1] - centroid_b[1],
                 bz = b[i * 3 + 2] - centroid_b[2];
        nk_accumulate_square_f64_serial_(&sum_squared, &sum_squared_compensation,
                                         r[0] * ax + r[1] * ay + r[2] * az - bx);
        nk_accumulate_square_f64_serial_(&sum_squared, &sum_squared_compensation,
                                         r[3] * ax + r[4] * ay + r[5] * az - by);
        nk_accumulate_square_f64_serial_(&sum_squared, &sum_squared_compensation,
                                         r[6] * ax + r[7] * ay + r[8] * az - bz);
    }
    return sum_squared + sum_squared_compensation;
}

/** Centroids, centered cross-covariance and ‖a − ā‖² of f32 clouds in a pivot-shifted f64 pass. */
NUMKONG_INLINE void nk_centered_moments_f32_neon_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                  nk_f64_t *centroid_a, nk_f64_t *centroid_b,
                                                  nk_f64_t *cross_covariance, nk_f64_t *centered_norm_squared_a) {
    nk_f64_t const pivot_a[3] = {a[0], a[1], a[2]}, pivot_b[3] = {b[0], b[1], b[2]};
    float64x2_t pivot_a_f64x2[3], pivot_b_f64x2[3], sum_a_f64x2[3], sum_b_f64x2[3], covariance_f64x2[9];
    float64x2_t norm_squared_a_f64x2 = vdupq_n_f64(0);
    for (int j = 0; j != 3; ++j)
        pivot_a_f64x2[j] = vdupq_n_f64(pivot_a[j]), pivot_b_f64x2[j] = vdupq_n_f64(pivot_b[j]),
        sum_a_f64x2[j] = vdupq_n_f64(0), sum_b_f64x2[j] = vdupq_n_f64(0);
    for (int j = 0; j != 9; ++j) covariance_f64x2[j] = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4[3], b_f32x4[3];
        nk_deinterleave_f32x4_neon_(a + i * 3, &a_f32x4[0], &a_f32x4[1], &a_f32x4[2]);
        nk_deinterleave_f32x4_neon_(b + i * 3, &b_f32x4[0], &b_f32x4[1], &b_f32x4[2]);
        for (int half = 0; half != 2; ++half) {
            float64x2_t a_f64x2[3], b_f64x2[3];
            for (int j = 0; j != 3; ++j) {
                a_f64x2[j] = vsubq_f64(half ? vcvt_high_f64_f32(a_f32x4[j]) : vcvt_f64_f32(vget_low_f32(a_f32x4[j])),
                                       pivot_a_f64x2[j]);
                b_f64x2[j] = vsubq_f64(half ? vcvt_high_f64_f32(b_f32x4[j]) : vcvt_f64_f32(vget_low_f32(b_f32x4[j])),
                                       pivot_b_f64x2[j]);
                sum_a_f64x2[j] = vaddq_f64(sum_a_f64x2[j], a_f64x2[j]);
                sum_b_f64x2[j] = vaddq_f64(sum_b_f64x2[j], b_f64x2[j]);
                norm_squared_a_f64x2 = vfmaq_f64(norm_squared_a_f64x2, a_f64x2[j], a_f64x2[j]);
            }
            for (int j = 0; j != 9; ++j)
                covariance_f64x2[j] = vfmaq_f64(covariance_f64x2[j], a_f64x2[j / 3], b_f64x2[j % 3]);
        }
    }
    nk_f64_t sum_a[3], sum_b[3], covariance[9], norm_squared_a = vaddvq_f64(norm_squared_a_f64x2);
    for (int j = 0; j != 3; ++j) sum_a[j] = vaddvq_f64(sum_a_f64x2[j]), sum_b[j] = vaddvq_f64(sum_b_f64x2[j]);
    for (int j = 0; j != 9; ++j) covariance[j] = vaddvq_f64(covariance_f64x2[j]);
    for (; i < n; ++i) {
        nk_f64_t a_point[3], b_point[3];
        for (int j = 0; j != 3; ++j) a_point[j] = a[i * 3 + j] - pivot_a[j], b_point[j] = b[i * 3 + j] - pivot_b[j];
        nk_centered_moments_update_f64_serial_(a_point, b_point, sum_a, sum_b, covariance, &norm_squared_a);
    }
    nk_centered_moments_finalize_f64_serial_(n, pivot_a, pivot_b, sum_a, sum_b, covariance, norm_squared_a, centroid_a,
                                             centroid_b, cross_covariance, centered_norm_squared_a);
}

/** Σ‖s · R · (aᵢ − ā) − (bᵢ − b̄)‖² of f32 clouds in f64, from residuals like the f64 variant. */
NUMKONG_INLINE nk_f64_t nk_transformed_ssd_f32_neon_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f64_t const *centroid_a, nk_f64_t const *centroid_b,
                                                     nk_f64_t const *rotation, nk_f64_t scale) {
    nk_f64_t r[9];
    float64x2_t r_f64x2[9], centroid_a_f64x2[3], centroid_b_f64x2[3];
    float64x2_t sum_squared_f64x2 = vdupq_n_f64(0), compensation_f64x2 = vdupq_n_f64(0);
    for (int j = 0; j != 9; ++j) r[j] = scale * rotation[j], r_f64x2[j] = vdupq_n_f64(r[j]);
    for (int j = 0; j != 3; ++j)
        centroid_a_f64x2[j] = vdupq_n_f64(centroid_a[j]), centroid_b_f64x2[j] = vdupq_n_f64(centroid_b[j]);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4[3], b_f32x4[3];
        nk_deinterleave_f32x4_neon_(a + i * 3, &a_f32x4[0], &a_f32x4[1], &a_f32x4[2]);
        nk_deinterleave_f32x4_neon_(b + i * 3, &b_f32x4[0], &b_f32x4[1], &b_f32x4[2]);
        for (int half = 0; half != 2; ++half) {
            float64x2_t a_f64x2[3], b_f64x2[3];
            for (int j = 0; j != 3; ++j) {
                a_f64x2[j] = vsubq_f64(half ? vcvt_high_f64_f32(a_f32x4[j]) : vcvt_f64_f32(vget_low_f32(a_f32x4[j])),
                                       centroid_a_f64x2[j]);
                b_f64x2[j] = vsubq_f64(half ? vcvt_high_f64_f32(b_f32x4[j]) : vcvt_f64_f32(vget_low_f32(b_f32x4[j])),
                                       centroid_b_f64x2[j]);
            }
            for (int j = 0; j != 3; ++j) {
                float64x2_t delta_f64x2 = vnegq_f64(b_f64x2[j]);
                for (int k = 0; k != 3; ++k) delta_f64x2 = vfmaq_f64(delta_f64x2, r_f64x2[j * 3 + k], a_f64x2[k]);
                nk_accumulate_square_f64x2_neon_(&sum_squared_f64x2, &compensation_f64x2, delta_f64x2);
            }
        }
    }
    nk_f64_t sum_squared = nk_dot_stable_sum_f64x2_neon_(sum_squared_f64x2, compensation_f64x2);
    nk_f64_t sum_squared_compensation = 0.0;
    for (; i < n; ++i) {
        nk_f64_t a_point[3], b_point[3];
        for (int j = 0; j != 3; ++j)
            a_point[j] = a[i * 3 + j] - centroid_a[j], b_point[j] = b[i * 3 + j] - centroid_b[j];
        nk_accumulate_residual_f64_serial_(&sum_squared, &sum_squared_compensation, r, a_point, b_point);
    }
    return sum_squared + sum_squared_compensation;
}

NUMKONG_INLINE void nk_deinterleave_f16x8_to_f32x4x2_neon_(nk_f16_t const *ptr,                             //
                                                           float32x4_t *x_low_out, float32x4_t *x_high_out, //
                                                           float32x4_t *y_low_out, float32x4_t *y_high_out, //
                                                           float32x4_t *z_low_out, float32x4_t *z_high_out) {
    // Deinterleave 24 f16 values (8 xyz triplets) into separate x, y, z vectors.
    // Uses NEON vld3q_u16 for efficient stride-3 deinterleaving, then converts to f32.
    // Avoids vld3q_f16 which is unavailable on MSVC for ARM.
    //
    // Input: 24 contiguous f16 values [x0,y0,z0, ..., x7,y7,z7]
    // Output: x_low[4]+x_high[4], y_low[4]+y_high[4], z_low[4]+z_high[4] vectors in f32
    uint16x8x3_t xyz_u16x8x3 = vld3q_u16((nk_u16_t const *)ptr);
    float16x8_t x_f16x8 = vreinterpretq_f16_u16(xyz_u16x8x3.val[0]);
    float16x8_t y_f16x8 = vreinterpretq_f16_u16(xyz_u16x8x3.val[1]);
    float16x8_t z_f16x8 = vreinterpretq_f16_u16(xyz_u16x8x3.val[2]);
    *x_low_out = vcvt_f32_f16(vget_low_f16(x_f16x8));
    *x_high_out = vcvt_high_f32_f16(x_f16x8);
    *y_low_out = vcvt_f32_f16(vget_low_f16(y_f16x8));
    *y_high_out = vcvt_high_f32_f16(y_f16x8);
    *z_low_out = vcvt_f32_f16(vget_low_f16(z_f16x8));
    *z_high_out = vcvt_high_f32_f16(z_f16x8);
}

NUMKONG_INLINE void nk_partial_deinterleave_f16_to_f32x4x2_neon_(nk_f16_t const *ptr, nk_size_t n_points, //
                                                                 float32x4_t *x_low_out,
                                                                 float32x4_t *x_high_out, //
                                                                 float32x4_t *y_low_out,
                                                                 float32x4_t *y_high_out, //
                                                                 float32x4_t *z_low_out, float32x4_t *z_high_out) {
    nk_u16_t buf[24] = {0};
    nk_u16_t const *src = (nk_u16_t const *)ptr;
    for (nk_size_t k = 0; k < n_points * 3; ++k) buf[k] = src[k];
    nk_deinterleave_f16x8_to_f32x4x2_neon_((nk_f16_t const *)buf, x_low_out, x_high_out, y_low_out, y_high_out,
                                           z_low_out, z_high_out);
}

/** Centroids, centered cross-covariance, ‖a − ā‖² and ‖b − b̄‖² of f16 clouds,
 *  in one pass shifted by the pivots in f32. */
NUMKONG_INLINE void nk_centered_moments_f16_neon_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                  nk_f32_t *centroid_a, nk_f32_t *centroid_b,
                                                  nk_f32_t *cross_covariance, nk_f32_t *centered_norm_squared_a,
                                                  nk_f32_t *centered_norm_squared_b) {
    nk_f32_t pivot_a[3], pivot_b[3];
    for (int j = 0; j != 3; ++j) nk_f16_to_f32_serial_(a + j, pivot_a + j), nk_f16_to_f32_serial_(b + j, pivot_b + j);
    float32x4_t pivot_a_f32x4[3], pivot_b_f32x4[3], sum_a_f32x4[3], sum_b_f32x4[3], covariance_f32x4[9];
    float32x4_t norm_squared_a_f32x4 = vdupq_n_f32(0), norm_squared_b_f32x4 = vdupq_n_f32(0);
    for (int j = 0; j != 3; ++j)
        pivot_a_f32x4[j] = vdupq_n_f32(pivot_a[j]), pivot_b_f32x4[j] = vdupq_n_f32(pivot_b[j]),
        sum_a_f32x4[j] = vdupq_n_f32(0), sum_b_f32x4[j] = vdupq_n_f32(0);
    for (int j = 0; j != 9; ++j) covariance_f32x4[j] = vdupq_n_f32(0);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t a_f32x4[6], b_f32x4[6];
        nk_deinterleave_f16x8_to_f32x4x2_neon_(a + i * 3, &a_f32x4[0], &a_f32x4[3], &a_f32x4[1], &a_f32x4[4],
                                               &a_f32x4[2], &a_f32x4[5]);
        nk_deinterleave_f16x8_to_f32x4x2_neon_(b + i * 3, &b_f32x4[0], &b_f32x4[3], &b_f32x4[1], &b_f32x4[4],
                                               &b_f32x4[2], &b_f32x4[5]);
        for (int half = 0; half != 6; half += 3) {
            for (int j = 0; j != 3; ++j) {
                a_f32x4[half + j] = vsubq_f32(a_f32x4[half + j], pivot_a_f32x4[j]);
                b_f32x4[half + j] = vsubq_f32(b_f32x4[half + j], pivot_b_f32x4[j]);
                sum_a_f32x4[j] = vaddq_f32(sum_a_f32x4[j], a_f32x4[half + j]);
                sum_b_f32x4[j] = vaddq_f32(sum_b_f32x4[j], b_f32x4[half + j]);
                norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_f32x4[half + j], a_f32x4[half + j]);
                norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_f32x4[half + j], b_f32x4[half + j]);
            }
            for (int j = 0; j != 9; ++j)
                covariance_f32x4[j] = vfmaq_f32(covariance_f32x4[j], a_f32x4[half + j / 3], b_f32x4[half + j % 3]);
        }
    }
    nk_f32_t sum_a[3], sum_b[3], covariance[9], norm_squared_a = vaddvq_f32(norm_squared_a_f32x4),
                                                norm_squared_b = vaddvq_f32(norm_squared_b_f32x4);
    for (int j = 0; j != 3; ++j) sum_a[j] = vaddvq_f32(sum_a_f32x4[j]), sum_b[j] = vaddvq_f32(sum_b_f32x4[j]);
    for (int j = 0; j != 9; ++j) covariance[j] = vaddvq_f32(covariance_f32x4[j]);
    for (; i < n; ++i) {
        nk_f32_t a_point[3], b_point[3];
        for (int j = 0; j != 3; ++j) {
            nk_f16_to_f32_serial_(a + i * 3 + j, a_point + j), nk_f16_to_f32_serial_(b + i * 3 + j, b_point + j);
            a_point[j] -= pivot_a[j], b_point[j] -= pivot_b[j];
        }
        nk_centered_moments_update_f32_serial_(a_point, b_point, sum_a, sum_b, covariance, &norm_squared_a,
                                               &norm_squared_b);
    }
    nk_centered_moments_finalize_f32_serial_(n, pivot_a, pivot_b, sum_a, sum_b, covariance, norm_squared_a,
                                             norm_squared_b, centroid_a, centroid_b, cross_covariance,
                                             centered_norm_squared_a, centered_norm_squared_b);
}

#if NUMKONG_TARGET_NEON

NUMKONG_API nk_status_t nk_rmsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                         nk_stream_t stream) {
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

    float64x2_t zero_f64x2 = vdupq_n_f64(0.0);
    float64x2_t sum_squared_x_low_f64x2 = zero_f64x2, sum_squared_x_high_f64x2 = zero_f64x2;
    float64x2_t sum_squared_y_low_f64x2 = zero_f64x2, sum_squared_y_high_f64x2 = zero_f64x2;
    float64x2_t sum_squared_z_low_f64x2 = zero_f64x2, sum_squared_z_high_f64x2 = zero_f64x2;
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_neon_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4),
            nk_deinterleave_f32x4_neon_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        float64x2_t delta_x_low_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(a_x_f32x4)),
                                                  vcvt_f64_f32(vget_low_f32(b_x_f32x4)));
        float64x2_t delta_x_high_f64x2 = vsubq_f64(vcvt_high_f64_f32(a_x_f32x4), vcvt_high_f64_f32(b_x_f32x4));
        float64x2_t delta_y_low_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(a_y_f32x4)),
                                                  vcvt_f64_f32(vget_low_f32(b_y_f32x4)));
        float64x2_t delta_y_high_f64x2 = vsubq_f64(vcvt_high_f64_f32(a_y_f32x4), vcvt_high_f64_f32(b_y_f32x4));
        float64x2_t delta_z_low_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(a_z_f32x4)),
                                                  vcvt_f64_f32(vget_low_f32(b_z_f32x4)));
        float64x2_t delta_z_high_f64x2 = vsubq_f64(vcvt_high_f64_f32(a_z_f32x4), vcvt_high_f64_f32(b_z_f32x4));

        sum_squared_x_low_f64x2 = vfmaq_f64(sum_squared_x_low_f64x2, delta_x_low_f64x2, delta_x_low_f64x2),
        sum_squared_x_high_f64x2 = vfmaq_f64(sum_squared_x_high_f64x2, delta_x_high_f64x2, delta_x_high_f64x2);
        sum_squared_y_low_f64x2 = vfmaq_f64(sum_squared_y_low_f64x2, delta_y_low_f64x2, delta_y_low_f64x2),
        sum_squared_y_high_f64x2 = vfmaq_f64(sum_squared_y_high_f64x2, delta_y_high_f64x2, delta_y_high_f64x2);
        sum_squared_z_low_f64x2 = vfmaq_f64(sum_squared_z_low_f64x2, delta_z_low_f64x2, delta_z_low_f64x2),
        sum_squared_z_high_f64x2 = vfmaq_f64(sum_squared_z_high_f64x2, delta_z_high_f64x2, delta_z_high_f64x2);
    }

    nk_f64_t sum_squared_x = vaddvq_f64(vaddq_f64(sum_squared_x_low_f64x2, sum_squared_x_high_f64x2));
    nk_f64_t sum_squared_y = vaddvq_f64(vaddq_f64(sum_squared_y_low_f64x2, sum_squared_y_high_f64x2));
    nk_f64_t sum_squared_z = vaddvq_f64(vaddq_f64(sum_squared_z_low_f64x2, sum_squared_z_high_f64x2));

    for (; index < n; ++index) {
        nk_f64_t delta_x = (nk_f64_t)a[index * 3 + 0] - (nk_f64_t)b[index * 3 + 0];
        nk_f64_t delta_y = (nk_f64_t)a[index * 3 + 1] - (nk_f64_t)b[index * 3 + 1];
        nk_f64_t delta_z = (nk_f64_t)a[index * 3 + 2] - (nk_f64_t)b[index * 3 + 2];
        sum_squared_x += delta_x * delta_x, sum_squared_y += delta_y * delta_y, sum_squared_z += delta_z * delta_z;
    }

    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64((sum_squared_x + sum_squared_y + sum_squared_z) / (nk_f64_t)n)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsd_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                         nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                         nk_stream_t stream) {
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

    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_squared_x_f64x2 = zeros_f64x2, sum_squared_y_f64x2 = zeros_f64x2, sum_squared_z_f64x2 = zeros_f64x2;

    float64x2_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;
    nk_size_t i = 0;

    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        float64x2_t delta_x_f64x2 = vsubq_f64(a_x_f64x2, b_x_f64x2);
        float64x2_t delta_y_f64x2 = vsubq_f64(a_y_f64x2, b_y_f64x2);
        float64x2_t delta_z_f64x2 = vsubq_f64(a_z_f64x2, b_z_f64x2);

        sum_squared_x_f64x2 = vfmaq_f64(sum_squared_x_f64x2, delta_x_f64x2, delta_x_f64x2);
        sum_squared_y_f64x2 = vfmaq_f64(sum_squared_y_f64x2, delta_y_f64x2, delta_y_f64x2);
        sum_squared_z_f64x2 = vfmaq_f64(sum_squared_z_f64x2, delta_z_f64x2, delta_z_f64x2);
    }

    nk_f64_t total_squared_x = nk_reduce_stable_f64x2_neon_(sum_squared_x_f64x2), total_squared_x_compensation = 0.0;
    nk_f64_t total_squared_y = nk_reduce_stable_f64x2_neon_(sum_squared_y_f64x2), total_squared_y_compensation = 0.0;
    nk_f64_t total_squared_z = nk_reduce_stable_f64x2_neon_(sum_squared_z_f64x2), total_squared_z_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t delta_x = a[i * 3 + 0] - b[i * 3 + 0];
        nk_f64_t delta_y = a[i * 3 + 1] - b[i * 3 + 1];
        nk_f64_t delta_z = a[i * 3 + 2] - b[i * 3 + 2];
        nk_accumulate_square_f64_serial_(&total_squared_x, &total_squared_x_compensation, delta_x);
        nk_accumulate_square_f64_serial_(&total_squared_y, &total_squared_y_compensation, delta_y);
        nk_accumulate_square_f64_serial_(&total_squared_z, &total_squared_z_compensation, delta_z);
    }

    total_squared_x += total_squared_x_compensation, total_squared_y += total_squared_y_compensation,
        total_squared_z += total_squared_z_compensation;

    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64((total_squared_x + total_squared_y + total_squared_z) / (nk_f64_t)n)),
                            0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                           nk_stream_t stream) {
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
    nk_centered_moments_f32_neon_(a, b, n, centroid_a, centroid_b, cross_covariance, NUMKONG_NULL);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a[0], a_centroid[1] = (nk_f32_t)centroid_a[1],
        a_centroid[2] = (nk_f32_t)centroid_a[2];
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b[0], b_centroid[1] = (nk_f32_t)centroid_b[1],
        b_centroid[2] = (nk_f32_t)centroid_b[2];

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);

        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];

        if (nk_det3x3_f64_serial_(optimal_rotation) < 0) {
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

    nk_f64_t sum_squared = nk_transformed_ssd_f32_neon_(a, b, n, centroid_a, centroid_b, optimal_rotation, 1.0);
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(sum_squared / (nk_f64_t)n)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                           nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                           nk_stream_t stream) {
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

    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    // 2x unrolling with dual accumulators to hide FMA latency.
    float64x2_t sum_a_x_a_f64x2 = zeros_f64x2, sum_a_y_a_f64x2 = zeros_f64x2, sum_a_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_a_f64x2 = zeros_f64x2, sum_b_y_a_f64x2 = zeros_f64x2, sum_b_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_a_x_b_f64x2 = zeros_f64x2, sum_a_y_b_f64x2 = zeros_f64x2, sum_a_z_b_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_b_f64x2 = zeros_f64x2, sum_b_y_b_f64x2 = zeros_f64x2, sum_b_z_b_f64x2 = zeros_f64x2;

    float64x2_t covariance_xx_a_f64x2 = zeros_f64x2, covariance_xy_a_f64x2 = zeros_f64x2,
                covariance_xz_a_f64x2 = zeros_f64x2;
    float64x2_t covariance_yx_a_f64x2 = zeros_f64x2, covariance_yy_a_f64x2 = zeros_f64x2,
                covariance_yz_a_f64x2 = zeros_f64x2;
    float64x2_t covariance_zx_a_f64x2 = zeros_f64x2, covariance_zy_a_f64x2 = zeros_f64x2,
                covariance_zz_a_f64x2 = zeros_f64x2;
    float64x2_t covariance_xx_b_f64x2 = zeros_f64x2, covariance_xy_b_f64x2 = zeros_f64x2,
                covariance_xz_b_f64x2 = zeros_f64x2;
    float64x2_t covariance_yx_b_f64x2 = zeros_f64x2, covariance_yy_b_f64x2 = zeros_f64x2,
                covariance_yz_b_f64x2 = zeros_f64x2;
    float64x2_t covariance_zx_b_f64x2 = zeros_f64x2, covariance_zy_b_f64x2 = zeros_f64x2,
                covariance_zz_b_f64x2 = zeros_f64x2;
    // The first-point shift keeps the centering correction from cancelling far from the origin.
    float64x2_t const pivot_a_x_f64x2 = vdupq_n_f64(a[0]), pivot_a_y_f64x2 = vdupq_n_f64(a[1]),
                      pivot_a_z_f64x2 = vdupq_n_f64(a[2]);
    float64x2_t const pivot_b_x_f64x2 = vdupq_n_f64(b[0]), pivot_b_y_f64x2 = vdupq_n_f64(b[1]),
                      pivot_b_z_f64x2 = vdupq_n_f64(b[2]);

    nk_size_t i = 0;
    float64x2_t a1_x_f64x2, a1_y_f64x2, a1_z_f64x2, b1_x_f64x2, b1_y_f64x2, b1_z_f64x2;
    float64x2_t a2_x_f64x2, a2_y_f64x2, a2_z_f64x2, b2_x_f64x2, b2_y_f64x2, b2_z_f64x2;

    // Main loop: 4 points per iteration (2x unrolled)
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        nk_deinterleave_f64x2_neon_(a + (i + 2) * 3, &a2_x_f64x2, &a2_y_f64x2, &a2_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + (i + 2) * 3, &b2_x_f64x2, &b2_y_f64x2, &b2_z_f64x2);
        a1_x_f64x2 = vsubq_f64(a1_x_f64x2, pivot_a_x_f64x2), a1_y_f64x2 = vsubq_f64(a1_y_f64x2, pivot_a_y_f64x2),
        a1_z_f64x2 = vsubq_f64(a1_z_f64x2, pivot_a_z_f64x2);
        b1_x_f64x2 = vsubq_f64(b1_x_f64x2, pivot_b_x_f64x2), b1_y_f64x2 = vsubq_f64(b1_y_f64x2, pivot_b_y_f64x2),
        b1_z_f64x2 = vsubq_f64(b1_z_f64x2, pivot_b_z_f64x2);
        a2_x_f64x2 = vsubq_f64(a2_x_f64x2, pivot_a_x_f64x2), a2_y_f64x2 = vsubq_f64(a2_y_f64x2, pivot_a_y_f64x2),
        a2_z_f64x2 = vsubq_f64(a2_z_f64x2, pivot_a_z_f64x2);
        b2_x_f64x2 = vsubq_f64(b2_x_f64x2, pivot_b_x_f64x2), b2_y_f64x2 = vsubq_f64(b2_y_f64x2, pivot_b_y_f64x2),
        b2_z_f64x2 = vsubq_f64(b2_z_f64x2, pivot_b_z_f64x2);

        // Interleaved accumulation
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_x_b_f64x2 = vaddq_f64(sum_a_x_b_f64x2, a2_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_y_b_f64x2 = vaddq_f64(sum_a_y_b_f64x2, a2_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_a_z_b_f64x2 = vaddq_f64(sum_a_z_b_f64x2, a2_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_x_b_f64x2 = vaddq_f64(sum_b_x_b_f64x2, b2_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_y_b_f64x2 = vaddq_f64(sum_b_y_b_f64x2, b2_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        sum_b_z_b_f64x2 = vaddq_f64(sum_b_z_b_f64x2, b2_z_f64x2);

        covariance_xx_a_f64x2 = vfmaq_f64(covariance_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        covariance_xx_b_f64x2 = vfmaq_f64(covariance_xx_b_f64x2, a2_x_f64x2, b2_x_f64x2);
        covariance_xy_a_f64x2 = vfmaq_f64(covariance_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        covariance_xy_b_f64x2 = vfmaq_f64(covariance_xy_b_f64x2, a2_x_f64x2, b2_y_f64x2);
        covariance_xz_a_f64x2 = vfmaq_f64(covariance_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        covariance_xz_b_f64x2 = vfmaq_f64(covariance_xz_b_f64x2, a2_x_f64x2, b2_z_f64x2);
        covariance_yx_a_f64x2 = vfmaq_f64(covariance_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        covariance_yx_b_f64x2 = vfmaq_f64(covariance_yx_b_f64x2, a2_y_f64x2, b2_x_f64x2);
        covariance_yy_a_f64x2 = vfmaq_f64(covariance_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        covariance_yy_b_f64x2 = vfmaq_f64(covariance_yy_b_f64x2, a2_y_f64x2, b2_y_f64x2);
        covariance_yz_a_f64x2 = vfmaq_f64(covariance_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        covariance_yz_b_f64x2 = vfmaq_f64(covariance_yz_b_f64x2, a2_y_f64x2, b2_z_f64x2);
        covariance_zx_a_f64x2 = vfmaq_f64(covariance_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        covariance_zx_b_f64x2 = vfmaq_f64(covariance_zx_b_f64x2, a2_z_f64x2, b2_x_f64x2);
        covariance_zy_a_f64x2 = vfmaq_f64(covariance_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        covariance_zy_b_f64x2 = vfmaq_f64(covariance_zy_b_f64x2, a2_z_f64x2, b2_y_f64x2);
        covariance_zz_a_f64x2 = vfmaq_f64(covariance_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
        covariance_zz_b_f64x2 = vfmaq_f64(covariance_zz_b_f64x2, a2_z_f64x2, b2_z_f64x2);
    }

    // 2-point tail
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        a1_x_f64x2 = vsubq_f64(a1_x_f64x2, pivot_a_x_f64x2), a1_y_f64x2 = vsubq_f64(a1_y_f64x2, pivot_a_y_f64x2),
        a1_z_f64x2 = vsubq_f64(a1_z_f64x2, pivot_a_z_f64x2);
        b1_x_f64x2 = vsubq_f64(b1_x_f64x2, pivot_b_x_f64x2), b1_y_f64x2 = vsubq_f64(b1_y_f64x2, pivot_b_y_f64x2),
        b1_z_f64x2 = vsubq_f64(b1_z_f64x2, pivot_b_z_f64x2);
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        covariance_xx_a_f64x2 = vfmaq_f64(covariance_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        covariance_xy_a_f64x2 = vfmaq_f64(covariance_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        covariance_xz_a_f64x2 = vfmaq_f64(covariance_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        covariance_yx_a_f64x2 = vfmaq_f64(covariance_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        covariance_yy_a_f64x2 = vfmaq_f64(covariance_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        covariance_yz_a_f64x2 = vfmaq_f64(covariance_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        covariance_zx_a_f64x2 = vfmaq_f64(covariance_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        covariance_zy_a_f64x2 = vfmaq_f64(covariance_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        covariance_zz_a_f64x2 = vfmaq_f64(covariance_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
    }

    // Combine dual accumulators
    float64x2_t sum_a_x_f64x2 = vaddq_f64(sum_a_x_a_f64x2, sum_a_x_b_f64x2);
    float64x2_t sum_a_y_f64x2 = vaddq_f64(sum_a_y_a_f64x2, sum_a_y_b_f64x2);
    float64x2_t sum_a_z_f64x2 = vaddq_f64(sum_a_z_a_f64x2, sum_a_z_b_f64x2);
    float64x2_t sum_b_x_f64x2 = vaddq_f64(sum_b_x_a_f64x2, sum_b_x_b_f64x2);
    float64x2_t sum_b_y_f64x2 = vaddq_f64(sum_b_y_a_f64x2, sum_b_y_b_f64x2);
    float64x2_t sum_b_z_f64x2 = vaddq_f64(sum_b_z_a_f64x2, sum_b_z_b_f64x2);
    float64x2_t covariance_xx_f64x2 = vaddq_f64(covariance_xx_a_f64x2, covariance_xx_b_f64x2);
    float64x2_t covariance_xy_f64x2 = vaddq_f64(covariance_xy_a_f64x2, covariance_xy_b_f64x2);
    float64x2_t covariance_xz_f64x2 = vaddq_f64(covariance_xz_a_f64x2, covariance_xz_b_f64x2);
    float64x2_t covariance_yx_f64x2 = vaddq_f64(covariance_yx_a_f64x2, covariance_yx_b_f64x2);
    float64x2_t covariance_yy_f64x2 = vaddq_f64(covariance_yy_a_f64x2, covariance_yy_b_f64x2);
    float64x2_t covariance_yz_f64x2 = vaddq_f64(covariance_yz_a_f64x2, covariance_yz_b_f64x2);
    float64x2_t covariance_zx_f64x2 = vaddq_f64(covariance_zx_a_f64x2, covariance_zx_b_f64x2);
    float64x2_t covariance_zy_f64x2 = vaddq_f64(covariance_zy_a_f64x2, covariance_zy_b_f64x2);
    float64x2_t covariance_zz_f64x2 = vaddq_f64(covariance_zz_a_f64x2, covariance_zz_b_f64x2);

    // Reduce vector accumulators.
    nk_f64_t sum_a_x = nk_reduce_stable_f64x2_neon_(sum_a_x_f64x2), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x2_neon_(sum_a_y_f64x2), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x2_neon_(sum_a_z_f64x2), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x2_neon_(sum_b_x_f64x2), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x2_neon_(sum_b_y_f64x2), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x2_neon_(sum_b_z_f64x2), sum_b_z_compensation = 0.0;

    nk_f64_t covariance_x_x = nk_reduce_stable_f64x2_neon_(covariance_xx_f64x2), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x2_neon_(covariance_xy_f64x2), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x2_neon_(covariance_xz_f64x2), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x2_neon_(covariance_yx_f64x2), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x2_neon_(covariance_yy_f64x2), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x2_neon_(covariance_yz_f64x2), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x2_neon_(covariance_zx_f64x2), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x2_neon_(covariance_zy_f64x2), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x2_neon_(covariance_zz_f64x2), covariance_z_z_compensation = 0.0;

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0] - a[0], ay = a[i * 3 + 1] - a[1], az = a[i * 3 + 2] - a[2];
        nk_f64_t bx = b[i * 3 + 0] - b[0], by = b[i * 3 + 1] - b[1], bz = b[i * 3 + 2] - b[2];
        nk_accumulate_sum_f64_serial_(&sum_a_x, &sum_a_x_compensation, ax),
            nk_accumulate_sum_f64_serial_(&sum_a_y, &sum_a_y_compensation, ay),
            nk_accumulate_sum_f64_serial_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_serial_(&sum_b_x, &sum_b_x_compensation, bx),
            nk_accumulate_sum_f64_serial_(&sum_b_y, &sum_b_y_compensation, by),
            nk_accumulate_sum_f64_serial_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_serial_(&covariance_x_x, &covariance_x_x_compensation, ax, bx),
            nk_accumulate_product_f64_serial_(&covariance_x_y, &covariance_x_y_compensation, ax, by),
            nk_accumulate_product_f64_serial_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_serial_(&covariance_y_x, &covariance_y_x_compensation, ay, bx),
            nk_accumulate_product_f64_serial_(&covariance_y_y, &covariance_y_y_compensation, ay, by),
            nk_accumulate_product_f64_serial_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_serial_(&covariance_z_x, &covariance_z_x_compensation, az, bx),
            nk_accumulate_product_f64_serial_(&covariance_z_y, &covariance_z_y_compensation, az, by),
            nk_accumulate_product_f64_serial_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
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

    // Compute SVD and optimal rotation
    nk_f64_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);

        // Handle reflection: if det(R) < 0, negate third column of V and recompute R
        if (nk_det3x3_f64_serial_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
    }

    // Output rotation matrix and scale=1.0.
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    if (scale) *scale = 1.0;

    nk_f64_t const centroid_a[3] = {centroid_a_x, centroid_a_y, centroid_a_z};
    nk_f64_t const centroid_b[3] = {centroid_b_x, centroid_b_y, centroid_b_z};
    nk_f64_t sum_squared = nk_transformed_ssd_f64_neon_(a, b, n, centroid_a, centroid_b, optimal_rotation, 1.0);
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(sum_squared * inv_n)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                            nk_stream_t stream) {
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

    nk_f64_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_squared_a;
    nk_centered_moments_f32_neon_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_squared_a);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a[0], a_centroid[1] = (nk_f32_t)centroid_a[1],
        a_centroid[2] = (nk_f32_t)centroid_a[2];
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b[0], b_centroid[1] = (nk_f32_t)centroid_b[1],
        b_centroid[2] = (nk_f32_t)centroid_b[2];

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I and trace(R · H) = trace(H).
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t applied_scale;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        applied_scale = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_squared_a;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);

        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];

        nk_f64_t det = nk_det3x3_f64_serial_(optimal_rotation), sign_correction = det < 0 ? -1.0 : 1.0;
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
        applied_scale = trace_ds / centered_norm_squared_a;
    }
    if (rotation)
        for (int j = 0; j != 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];
    if (scale) *scale = (nk_f32_t)applied_scale;

    nk_f64_t sum_squared = nk_transformed_ssd_f32_neon_(a, b, n, centroid_a, centroid_b, optimal_rotation,
                                                        applied_scale);
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(sum_squared / (nk_f64_t)n)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                            nk_stream_t stream) {
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

    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    // 2x unrolling with dual accumulators to hide FMA latency.
    float64x2_t sum_a_x_a_f64x2 = zeros_f64x2, sum_a_y_a_f64x2 = zeros_f64x2, sum_a_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_a_f64x2 = zeros_f64x2, sum_b_y_a_f64x2 = zeros_f64x2, sum_b_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_a_x_b_f64x2 = zeros_f64x2, sum_a_y_b_f64x2 = zeros_f64x2, sum_a_z_b_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_b_f64x2 = zeros_f64x2, sum_b_y_b_f64x2 = zeros_f64x2, sum_b_z_b_f64x2 = zeros_f64x2;

    float64x2_t covariance_xx_a_f64x2 = zeros_f64x2, covariance_xy_a_f64x2 = zeros_f64x2,
                covariance_xz_a_f64x2 = zeros_f64x2;
    float64x2_t covariance_yx_a_f64x2 = zeros_f64x2, covariance_yy_a_f64x2 = zeros_f64x2,
                covariance_yz_a_f64x2 = zeros_f64x2;
    float64x2_t covariance_zx_a_f64x2 = zeros_f64x2, covariance_zy_a_f64x2 = zeros_f64x2,
                covariance_zz_a_f64x2 = zeros_f64x2;
    float64x2_t covariance_xx_b_f64x2 = zeros_f64x2, covariance_xy_b_f64x2 = zeros_f64x2,
                covariance_xz_b_f64x2 = zeros_f64x2;
    float64x2_t covariance_yx_b_f64x2 = zeros_f64x2, covariance_yy_b_f64x2 = zeros_f64x2,
                covariance_yz_b_f64x2 = zeros_f64x2;
    float64x2_t covariance_zx_b_f64x2 = zeros_f64x2, covariance_zy_b_f64x2 = zeros_f64x2,
                covariance_zz_b_f64x2 = zeros_f64x2;
    float64x2_t norm_squared_a_a_f64x2 = zeros_f64x2, norm_squared_a_b_f64x2 = zeros_f64x2;
    // The first-point shift keeps the centering correction from cancelling far from the origin.
    float64x2_t const pivot_a_x_f64x2 = vdupq_n_f64(a[0]), pivot_a_y_f64x2 = vdupq_n_f64(a[1]),
                      pivot_a_z_f64x2 = vdupq_n_f64(a[2]);
    float64x2_t const pivot_b_x_f64x2 = vdupq_n_f64(b[0]), pivot_b_y_f64x2 = vdupq_n_f64(b[1]),
                      pivot_b_z_f64x2 = vdupq_n_f64(b[2]);

    nk_size_t i = 0;
    float64x2_t a1_x_f64x2, a1_y_f64x2, a1_z_f64x2, b1_x_f64x2, b1_y_f64x2, b1_z_f64x2;
    float64x2_t a2_x_f64x2, a2_y_f64x2, a2_z_f64x2, b2_x_f64x2, b2_y_f64x2, b2_z_f64x2;

    // Main loop: 4 points per iteration (2x unrolled)
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        nk_deinterleave_f64x2_neon_(a + (i + 2) * 3, &a2_x_f64x2, &a2_y_f64x2, &a2_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + (i + 2) * 3, &b2_x_f64x2, &b2_y_f64x2, &b2_z_f64x2);
        a1_x_f64x2 = vsubq_f64(a1_x_f64x2, pivot_a_x_f64x2), a1_y_f64x2 = vsubq_f64(a1_y_f64x2, pivot_a_y_f64x2),
        a1_z_f64x2 = vsubq_f64(a1_z_f64x2, pivot_a_z_f64x2);
        b1_x_f64x2 = vsubq_f64(b1_x_f64x2, pivot_b_x_f64x2), b1_y_f64x2 = vsubq_f64(b1_y_f64x2, pivot_b_y_f64x2),
        b1_z_f64x2 = vsubq_f64(b1_z_f64x2, pivot_b_z_f64x2);
        a2_x_f64x2 = vsubq_f64(a2_x_f64x2, pivot_a_x_f64x2), a2_y_f64x2 = vsubq_f64(a2_y_f64x2, pivot_a_y_f64x2),
        a2_z_f64x2 = vsubq_f64(a2_z_f64x2, pivot_a_z_f64x2);
        b2_x_f64x2 = vsubq_f64(b2_x_f64x2, pivot_b_x_f64x2), b2_y_f64x2 = vsubq_f64(b2_y_f64x2, pivot_b_y_f64x2),
        b2_z_f64x2 = vsubq_f64(b2_z_f64x2, pivot_b_z_f64x2);

        // Interleaved accumulation
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_x_b_f64x2 = vaddq_f64(sum_a_x_b_f64x2, a2_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_y_b_f64x2 = vaddq_f64(sum_a_y_b_f64x2, a2_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_a_z_b_f64x2 = vaddq_f64(sum_a_z_b_f64x2, a2_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_x_b_f64x2 = vaddq_f64(sum_b_x_b_f64x2, b2_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_y_b_f64x2 = vaddq_f64(sum_b_y_b_f64x2, b2_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        sum_b_z_b_f64x2 = vaddq_f64(sum_b_z_b_f64x2, b2_z_f64x2);

        covariance_xx_a_f64x2 = vfmaq_f64(covariance_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        covariance_xx_b_f64x2 = vfmaq_f64(covariance_xx_b_f64x2, a2_x_f64x2, b2_x_f64x2);
        covariance_xy_a_f64x2 = vfmaq_f64(covariance_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        covariance_xy_b_f64x2 = vfmaq_f64(covariance_xy_b_f64x2, a2_x_f64x2, b2_y_f64x2);
        covariance_xz_a_f64x2 = vfmaq_f64(covariance_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        covariance_xz_b_f64x2 = vfmaq_f64(covariance_xz_b_f64x2, a2_x_f64x2, b2_z_f64x2);
        covariance_yx_a_f64x2 = vfmaq_f64(covariance_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        covariance_yx_b_f64x2 = vfmaq_f64(covariance_yx_b_f64x2, a2_y_f64x2, b2_x_f64x2);
        covariance_yy_a_f64x2 = vfmaq_f64(covariance_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        covariance_yy_b_f64x2 = vfmaq_f64(covariance_yy_b_f64x2, a2_y_f64x2, b2_y_f64x2);
        covariance_yz_a_f64x2 = vfmaq_f64(covariance_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        covariance_yz_b_f64x2 = vfmaq_f64(covariance_yz_b_f64x2, a2_y_f64x2, b2_z_f64x2);
        covariance_zx_a_f64x2 = vfmaq_f64(covariance_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        covariance_zx_b_f64x2 = vfmaq_f64(covariance_zx_b_f64x2, a2_z_f64x2, b2_x_f64x2);
        covariance_zy_a_f64x2 = vfmaq_f64(covariance_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        covariance_zy_b_f64x2 = vfmaq_f64(covariance_zy_b_f64x2, a2_z_f64x2, b2_y_f64x2);
        covariance_zz_a_f64x2 = vfmaq_f64(covariance_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
        covariance_zz_b_f64x2 = vfmaq_f64(covariance_zz_b_f64x2, a2_z_f64x2, b2_z_f64x2);

        norm_squared_a_a_f64x2 = vfmaq_f64(norm_squared_a_a_f64x2, a1_x_f64x2, a1_x_f64x2);
        norm_squared_a_b_f64x2 = vfmaq_f64(norm_squared_a_b_f64x2, a2_x_f64x2, a2_x_f64x2);
        norm_squared_a_a_f64x2 = vfmaq_f64(norm_squared_a_a_f64x2, a1_y_f64x2, a1_y_f64x2);
        norm_squared_a_b_f64x2 = vfmaq_f64(norm_squared_a_b_f64x2, a2_y_f64x2, a2_y_f64x2);
        norm_squared_a_a_f64x2 = vfmaq_f64(norm_squared_a_a_f64x2, a1_z_f64x2, a1_z_f64x2);
        norm_squared_a_b_f64x2 = vfmaq_f64(norm_squared_a_b_f64x2, a2_z_f64x2, a2_z_f64x2);
    }

    // 2-point tail
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        a1_x_f64x2 = vsubq_f64(a1_x_f64x2, pivot_a_x_f64x2), a1_y_f64x2 = vsubq_f64(a1_y_f64x2, pivot_a_y_f64x2),
        a1_z_f64x2 = vsubq_f64(a1_z_f64x2, pivot_a_z_f64x2);
        b1_x_f64x2 = vsubq_f64(b1_x_f64x2, pivot_b_x_f64x2), b1_y_f64x2 = vsubq_f64(b1_y_f64x2, pivot_b_y_f64x2),
        b1_z_f64x2 = vsubq_f64(b1_z_f64x2, pivot_b_z_f64x2);
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        covariance_xx_a_f64x2 = vfmaq_f64(covariance_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        covariance_xy_a_f64x2 = vfmaq_f64(covariance_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        covariance_xz_a_f64x2 = vfmaq_f64(covariance_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        covariance_yx_a_f64x2 = vfmaq_f64(covariance_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        covariance_yy_a_f64x2 = vfmaq_f64(covariance_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        covariance_yz_a_f64x2 = vfmaq_f64(covariance_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        covariance_zx_a_f64x2 = vfmaq_f64(covariance_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        covariance_zy_a_f64x2 = vfmaq_f64(covariance_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        covariance_zz_a_f64x2 = vfmaq_f64(covariance_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
        norm_squared_a_a_f64x2 = vfmaq_f64(norm_squared_a_a_f64x2, a1_x_f64x2, a1_x_f64x2);
        norm_squared_a_a_f64x2 = vfmaq_f64(norm_squared_a_a_f64x2, a1_y_f64x2, a1_y_f64x2);
        norm_squared_a_a_f64x2 = vfmaq_f64(norm_squared_a_a_f64x2, a1_z_f64x2, a1_z_f64x2);
    }

    // Combine dual accumulators
    float64x2_t sum_a_x_f64x2 = vaddq_f64(sum_a_x_a_f64x2, sum_a_x_b_f64x2);
    float64x2_t sum_a_y_f64x2 = vaddq_f64(sum_a_y_a_f64x2, sum_a_y_b_f64x2);
    float64x2_t sum_a_z_f64x2 = vaddq_f64(sum_a_z_a_f64x2, sum_a_z_b_f64x2);
    float64x2_t sum_b_x_f64x2 = vaddq_f64(sum_b_x_a_f64x2, sum_b_x_b_f64x2);
    float64x2_t sum_b_y_f64x2 = vaddq_f64(sum_b_y_a_f64x2, sum_b_y_b_f64x2);
    float64x2_t sum_b_z_f64x2 = vaddq_f64(sum_b_z_a_f64x2, sum_b_z_b_f64x2);
    float64x2_t covariance_xx_f64x2 = vaddq_f64(covariance_xx_a_f64x2, covariance_xx_b_f64x2);
    float64x2_t covariance_xy_f64x2 = vaddq_f64(covariance_xy_a_f64x2, covariance_xy_b_f64x2);
    float64x2_t covariance_xz_f64x2 = vaddq_f64(covariance_xz_a_f64x2, covariance_xz_b_f64x2);
    float64x2_t covariance_yx_f64x2 = vaddq_f64(covariance_yx_a_f64x2, covariance_yx_b_f64x2);
    float64x2_t covariance_yy_f64x2 = vaddq_f64(covariance_yy_a_f64x2, covariance_yy_b_f64x2);
    float64x2_t covariance_yz_f64x2 = vaddq_f64(covariance_yz_a_f64x2, covariance_yz_b_f64x2);
    float64x2_t covariance_zx_f64x2 = vaddq_f64(covariance_zx_a_f64x2, covariance_zx_b_f64x2);
    float64x2_t covariance_zy_f64x2 = vaddq_f64(covariance_zy_a_f64x2, covariance_zy_b_f64x2);
    float64x2_t covariance_zz_f64x2 = vaddq_f64(covariance_zz_a_f64x2, covariance_zz_b_f64x2);
    float64x2_t norm_squared_a_f64x2 = vaddq_f64(norm_squared_a_a_f64x2, norm_squared_a_b_f64x2);

    // Reduce vector accumulators.
    nk_f64_t sum_a_x = nk_reduce_stable_f64x2_neon_(sum_a_x_f64x2), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x2_neon_(sum_a_y_f64x2), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x2_neon_(sum_a_z_f64x2), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x2_neon_(sum_b_x_f64x2), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x2_neon_(sum_b_y_f64x2), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x2_neon_(sum_b_z_f64x2), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x2_neon_(covariance_xx_f64x2), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x2_neon_(covariance_xy_f64x2), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x2_neon_(covariance_xz_f64x2), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x2_neon_(covariance_yx_f64x2), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x2_neon_(covariance_yy_f64x2), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x2_neon_(covariance_yz_f64x2), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x2_neon_(covariance_zx_f64x2), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x2_neon_(covariance_zy_f64x2), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x2_neon_(covariance_zz_f64x2), covariance_z_z_compensation = 0.0;
    nk_f64_t norm_squared_a_sum = nk_reduce_stable_f64x2_neon_(norm_squared_a_f64x2), norm_squared_a_compensation = 0.0;

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0] - a[0], ay = a[i * 3 + 1] - a[1], az = a[i * 3 + 2] - a[2];
        nk_f64_t bx = b[i * 3 + 0] - b[0], by = b[i * 3 + 1] - b[1], bz = b[i * 3 + 2] - b[2];
        nk_accumulate_sum_f64_serial_(&sum_a_x, &sum_a_x_compensation, ax),
            nk_accumulate_sum_f64_serial_(&sum_a_y, &sum_a_y_compensation, ay),
            nk_accumulate_sum_f64_serial_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_serial_(&sum_b_x, &sum_b_x_compensation, bx),
            nk_accumulate_sum_f64_serial_(&sum_b_y, &sum_b_y_compensation, by),
            nk_accumulate_sum_f64_serial_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_serial_(&covariance_x_x, &covariance_x_x_compensation, ax, bx),
            nk_accumulate_product_f64_serial_(&covariance_x_y, &covariance_x_y_compensation, ax, by),
            nk_accumulate_product_f64_serial_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_serial_(&covariance_y_x, &covariance_y_x_compensation, ay, bx),
            nk_accumulate_product_f64_serial_(&covariance_y_y, &covariance_y_y_compensation, ay, by),
            nk_accumulate_product_f64_serial_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_serial_(&covariance_z_x, &covariance_z_x_compensation, az, bx),
            nk_accumulate_product_f64_serial_(&covariance_z_y, &covariance_z_y_compensation, az, by),
            nk_accumulate_product_f64_serial_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
        nk_accumulate_square_f64_serial_(&norm_squared_a_sum, &norm_squared_a_compensation, ax),
            nk_accumulate_square_f64_serial_(&norm_squared_a_sum, &norm_squared_a_compensation, ay),
            nk_accumulate_square_f64_serial_(&norm_squared_a_sum, &norm_squared_a_compensation, az);
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

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t mean_a_x = sum_a_x * inv_n, mean_a_y = sum_a_y * inv_n, mean_a_z = sum_a_z * inv_n;
    nk_f64_t mean_b_x = sum_b_x * inv_n, mean_b_y = sum_b_y * inv_n, mean_b_z = sum_b_z * inv_n;
    nk_f64_t centroid_a_x = a[0] + mean_a_x, centroid_a_y = a[1] + mean_a_y, centroid_a_z = a[2] + mean_a_z;
    nk_f64_t centroid_b_x = b[0] + mean_b_x, centroid_b_y = b[1] + mean_b_y, centroid_b_z = b[2] + mean_b_z;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Centered norm-squared via parallel-axis identity; clamp at zero for numeric safety.
    nk_f64_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f64_t)n * (mean_a_x * mean_a_x + mean_a_y * mean_a_y + mean_a_z * mean_a_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;

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

    // Compute SVD
    nk_f64_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I and trace(R · H) = trace(H).
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t computed_scale;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        computed_scale = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_squared_a;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);

        // Handle reflection and compute scale
        nk_f64_t det = nk_det3x3_f64_serial_(optimal_rotation);
        nk_f64_t trace_d_s = svd_diagonal[0] + svd_diagonal[4] + (det < 0 ? -svd_diagonal[8] : svd_diagonal[8]);
        computed_scale = trace_d_s / centered_norm_squared_a;

        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
    }

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = computed_scale;

    nk_f64_t const centroid_a[3] = {centroid_a_x, centroid_a_y, centroid_a_z};
    nk_f64_t const centroid_b[3] = {centroid_b_x, centroid_b_y, centroid_b_z};
    nk_f64_t sum_squared = nk_transformed_ssd_f64_neon_(a, b, n, centroid_a, centroid_b, optimal_rotation,
                                                        computed_scale);
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(sum_squared * inv_n)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsd_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                         nk_stream_t stream) {
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

    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_squared_x_f32x4 = zeros_f32x4, sum_squared_y_f32x4 = zeros_f32x4, sum_squared_z_f32x4 = zeros_f32x4;
    float32x4_t a_x_low_f32x4, a_x_high_f32x4, a_y_low_f32x4, a_y_high_f32x4, a_z_low_f32x4, a_z_high_f32x4;
    float32x4_t b_x_low_f32x4, b_x_high_f32x4, b_y_low_f32x4, b_y_high_f32x4, b_z_low_f32x4, b_z_high_f32x4;
    nk_size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f32x4x2_neon_(a + i * 3, &a_x_low_f32x4, &a_x_high_f32x4, &a_y_low_f32x4,
                                               &a_y_high_f32x4, &a_z_low_f32x4, &a_z_high_f32x4);
        nk_deinterleave_f16x8_to_f32x4x2_neon_(b + i * 3, &b_x_low_f32x4, &b_x_high_f32x4, &b_y_low_f32x4,
                                               &b_y_high_f32x4, &b_z_low_f32x4, &b_z_high_f32x4);

        float32x4_t delta_x_f32x4 = vsubq_f32(a_x_low_f32x4, b_x_low_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(a_y_low_f32x4, b_y_low_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(a_z_low_f32x4, b_z_low_f32x4);
        sum_squared_x_f32x4 = vfmaq_f32(sum_squared_x_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_y_f32x4 = vfmaq_f32(sum_squared_y_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_z_f32x4 = vfmaq_f32(sum_squared_z_f32x4, delta_z_f32x4, delta_z_f32x4);

        delta_x_f32x4 = vsubq_f32(a_x_high_f32x4, b_x_high_f32x4);
        delta_y_f32x4 = vsubq_f32(a_y_high_f32x4, b_y_high_f32x4);
        delta_z_f32x4 = vsubq_f32(a_z_high_f32x4, b_z_high_f32x4);
        sum_squared_x_f32x4 = vfmaq_f32(sum_squared_x_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_y_f32x4 = vfmaq_f32(sum_squared_y_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_z_f32x4 = vfmaq_f32(sum_squared_z_f32x4, delta_z_f32x4, delta_z_f32x4);
    }

    if (i < n) {
        nk_partial_deinterleave_f16_to_f32x4x2_neon_(a + i * 3, n - i, &a_x_low_f32x4, &a_x_high_f32x4, &a_y_low_f32x4,
                                                     &a_y_high_f32x4, &a_z_low_f32x4, &a_z_high_f32x4);
        nk_partial_deinterleave_f16_to_f32x4x2_neon_(b + i * 3, n - i, &b_x_low_f32x4, &b_x_high_f32x4, &b_y_low_f32x4,
                                                     &b_y_high_f32x4, &b_z_low_f32x4, &b_z_high_f32x4);

        float32x4_t delta_x_f32x4 = vsubq_f32(a_x_low_f32x4, b_x_low_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(a_y_low_f32x4, b_y_low_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(a_z_low_f32x4, b_z_low_f32x4);
        sum_squared_x_f32x4 = vfmaq_f32(sum_squared_x_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_y_f32x4 = vfmaq_f32(sum_squared_y_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_z_f32x4 = vfmaq_f32(sum_squared_z_f32x4, delta_z_f32x4, delta_z_f32x4);

        delta_x_f32x4 = vsubq_f32(a_x_high_f32x4, b_x_high_f32x4);
        delta_y_f32x4 = vsubq_f32(a_y_high_f32x4, b_y_high_f32x4);
        delta_z_f32x4 = vsubq_f32(a_z_high_f32x4, b_z_high_f32x4);
        sum_squared_x_f32x4 = vfmaq_f32(sum_squared_x_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_y_f32x4 = vfmaq_f32(sum_squared_y_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_z_f32x4 = vfmaq_f32(sum_squared_z_f32x4, delta_z_f32x4, delta_z_f32x4);
    }

    nk_f32_t sum_squared = vaddvq_f32(sum_squared_x_f32x4) + vaddvq_f32(sum_squared_y_f32x4) +
                           vaddvq_f32(sum_squared_z_f32x4);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(sum_squared / (nk_f32_t)n)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           nk_stream_t stream) {
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

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_squared_a, centered_norm_squared_b;
    nk_centered_moments_f16_neon_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_squared_a,
                                  &centered_norm_squared_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I.
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
    }
    else {
        // SVD of H = U * S * Vᵀ
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);

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

        // Handle reflection: if det(R) < 0, negate third column of V and recompute
        nk_f32_t rotation_determinant = nk_det3x3_f32_serial_(optimal_rotation);
        if (rotation_determinant < 0) {
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
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    nk_f32_t sum_squared = nk_folded_ssd_f32_serial_(optimal_rotation, 1.0f, cross_covariance, centered_norm_squared_a,
                                                     centered_norm_squared_b);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(sum_squared / (nk_f32_t)n)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            nk_stream_t stream) {
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

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_squared_a, centered_norm_squared_b;
    nk_centered_moments_f16_neon_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_squared_a,
                                  &centered_norm_squared_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I and trace(R · H) = trace(H).
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t scale_factor;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        scale_factor = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_squared_a;
    }
    else {
        // SVD of H = U * S * Vᵀ
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);

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

        // Handle reflection and compute scale: c = trace(D · S) / ‖a-ā‖²
        nk_f32_t rotation_determinant = nk_det3x3_f32_serial_(optimal_rotation);
        nk_f32_t sign_det = rotation_determinant < 0 ? -1.0f : 1.0f;
        nk_f32_t trace_scaled_s = svd_diagonal[0] + svd_diagonal[4] + sign_det * svd_diagonal[8];
        scale_factor = trace_scaled_s / centered_norm_squared_a;

        if (rotation_determinant < 0) {
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
    if (scale) *scale = scale_factor;

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    nk_f32_t sum_squared = nk_folded_ssd_f32_serial_(optimal_rotation, scale_factor, cross_covariance,
                                                     centered_norm_squared_a, centered_norm_squared_b);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(sum_squared / (nk_f32_t)n)), 0);
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
#endif // NUMKONG_MESH_NEON_H
