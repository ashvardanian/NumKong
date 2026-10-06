/**
 *  @brief SIMD-accelerated Point Cloud Alignment for WASM Relaxed SIMD.
 *  @file include/numkong/mesh/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 10, 2026
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section wasm_mesh_instructions Key WASM Relaxed SIMD Mesh Instructions
 *
 *  Point cloud operations use these WASM Relaxed SIMD instructions:
 *
 *      Intrinsic                       Operation
 *      wasm_f32x4_relaxed_madd         Fused multiply-add (4-way f32)
 *      wasm_f64x2_relaxed_madd         Fused multiply-add (2-way f64)
 *      wasm_f32x4_mul                  Multiply (4-way f32)
 *      wasm_f64x2_mul                  Multiply (2-way f64)
 *      wasm_f32x4_add/sub              Add/subtract (4-way f32)
 *      wasm_f64x2_add/sub              Add/subtract (2-way f64)
 *      wasm_f32x4_splat                Broadcast scalar to all lanes
 *      wasm_f64x2_splat                Broadcast scalar to all lanes
 *      wasm_i32x4_shuffle              Cross-vector lane permutation (f32)
 *      wasm_i64x2_shuffle              Cross-vector lane permutation (f64)
 *
 *  WASM lacks hardware stride-3 deinterleaving (no LD3 equivalent), so XYZ
 *  deinterleaving is done via shuffle chains. No dual-accumulator unrolling is
 *  used since WASM engines already handle instruction scheduling.
 */
#ifndef NK_MESH_V128RELAXED_H
#define NK_MESH_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/mesh/serial.h"
#include "numkong/dot/v128relaxed.h"
#include "numkong/reduce/v128relaxed.h"
#include "numkong/scalar/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  Deinterleave 12 contiguous f32 values (4 XYZ triplets) into separate x, y, z vectors.
 *
 *  Input:  [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors
 */
NK_INTERNAL void nk_deinterleave_f32x4_v128relaxed_(nk_f32_t const *ptr, v128_t *xs_f32x4, v128_t *ys_f32x4,
                                                    v128_t *zs_f32x4) {
    v128_t v0_f32x4 = wasm_v128_load(ptr);     // x0 y0 z0 x1
    v128_t v1_f32x4 = wasm_v128_load(ptr + 4); // y1 z1 x2 y2
    v128_t v2_f32x4 = wasm_v128_load(ptr + 8); // z2 x3 y3 z3
    // x0 x1 x2 x3
    v128_t tmp01 = wasm_i32x4_shuffle(v0_f32x4, v1_f32x4, 0, 3, 6, 0); // x0 x1 x2 _
    *xs_f32x4 = wasm_i32x4_shuffle(tmp01, v2_f32x4, 0, 1, 2, 5);       // x0 x1 x2 x3
    // y0 y1 y2 y3
    v128_t tmp23 = wasm_i32x4_shuffle(v0_f32x4, v1_f32x4, 1, 4, 7, 0); // y0 y1 y2 _
    *ys_f32x4 = wasm_i32x4_shuffle(tmp23, v2_f32x4, 0, 1, 2, 6);       // y0 y1 y2 y3
    // z0 z1 z2 z3
    v128_t tmp45 = wasm_i32x4_shuffle(v0_f32x4, v1_f32x4, 2, 5, 0, 0); // z0 z1 _ _
    *zs_f32x4 = wasm_i32x4_shuffle(tmp45, v2_f32x4, 0, 1, 4, 7);       // z0 z1 z2 z3
}

/*  Deinterleave 6 contiguous f64 values (2 XYZ triplets) into separate x, y, z vectors.
 *
 *  Input:  [x0,y0,z0, x1,y1,z1]
 *  Output: x[2], y[2], z[2] vectors
 */
NK_INTERNAL void nk_deinterleave_f64x2_v128relaxed_(nk_f64_t const *ptr, v128_t *xs_f64x2, v128_t *ys_f64x2,
                                                    v128_t *zs_f64x2) {
    v128_t v0_f64x2 = wasm_v128_load(ptr);                    // x0 y0
    v128_t v1_f64x2 = wasm_v128_load(ptr + 2);                // z0 x1
    v128_t v2_f64x2 = wasm_v128_load(ptr + 4);                // y1 z1
    *xs_f64x2 = wasm_i64x2_shuffle(v0_f64x2, v1_f64x2, 0, 3); // x0 x1
    *ys_f64x2 = wasm_i64x2_shuffle(v0_f64x2, v2_f64x2, 1, 2); // y0 y1
    *zs_f64x2 = wasm_i64x2_shuffle(v1_f64x2, v2_f64x2, 0, 3); // z0 z1
}

/* Horizontal sum of all 4 f32 lanes. */
NK_INTERNAL nk_f32_t nk_hsum_f32x4_v128relaxed_(v128_t v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) +
           wasm_f32x4_extract_lane(v, 3);
}

/* Horizontal sum of both f64 lanes. */
NK_INTERNAL nk_f64_t nk_hsum_f64x2_v128relaxed_(v128_t v) {
    return wasm_f64x2_extract_lane(v, 0) + wasm_f64x2_extract_lane(v, 1);
}

NK_INTERNAL nk_f64_t nk_reduce_stable_f64x2_v128relaxed_(v128_t values_f64x2) {
    nk_b128_vec_t values;
    values.v128 = values_f64x2;
    nk_f64_t sum = 0.0, compensation = 0.0;
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[0]);
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[1]);
    return sum + compensation;
}

NK_INTERNAL void nk_rotation_from_svd_f64_v128relaxed_(nk_f64_t const *svd_u, nk_f64_t const *svd_v,
                                                       nk_f64_t *rotation) {
    nk_rotation_from_svd_f64_serial_(svd_u, svd_v, rotation);
}

NK_INTERNAL void nk_accumulate_square_f64x2_v128relaxed_(v128_t *sum_f64x2, v128_t *compensation_f64x2,
                                                         v128_t values_f64x2) {
    v128_t product_f64x2 = wasm_f64x2_mul(values_f64x2, values_f64x2);
    v128_t product_error_f64x2 = wasm_f64x2_sub(
        wasm_f64x2_relaxed_madd(values_f64x2, values_f64x2, wasm_f64x2_splat(0.0)), product_f64x2);
    v128_t tentative_sum_f64x2 = wasm_f64x2_add(*sum_f64x2, product_f64x2);
    v128_t virtual_addend_f64x2 = wasm_f64x2_sub(tentative_sum_f64x2, *sum_f64x2);
    v128_t sum_error_f64x2 = wasm_f64x2_add(
        wasm_f64x2_sub(*sum_f64x2, wasm_f64x2_sub(tentative_sum_f64x2, virtual_addend_f64x2)),
        wasm_f64x2_sub(product_f64x2, virtual_addend_f64x2));
    *sum_f64x2 = tentative_sum_f64x2;
    *compensation_f64x2 = wasm_f64x2_add(*compensation_f64x2, wasm_f64x2_add(sum_error_f64x2, product_error_f64x2));
}

NK_INTERNAL void nk_centroid_and_cross_covariance_f32_v128relaxed_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,              //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,                 //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,                 //
    nk_f64_t h[9]) {
    v128_t zero_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_a_x_lower_f64x2 = zero_f64x2, sum_a_x_upper_f64x2 = zero_f64x2;
    v128_t sum_a_y_lower_f64x2 = zero_f64x2, sum_a_y_upper_f64x2 = zero_f64x2;
    v128_t sum_a_z_lower_f64x2 = zero_f64x2, sum_a_z_upper_f64x2 = zero_f64x2;
    v128_t sum_b_x_lower_f64x2 = zero_f64x2, sum_b_x_upper_f64x2 = zero_f64x2;
    v128_t sum_b_y_lower_f64x2 = zero_f64x2, sum_b_y_upper_f64x2 = zero_f64x2;
    v128_t sum_b_z_lower_f64x2 = zero_f64x2, sum_b_z_upper_f64x2 = zero_f64x2;
    v128_t cross_00_lower_f64x2 = zero_f64x2, cross_00_upper_f64x2 = zero_f64x2;
    v128_t cross_01_lower_f64x2 = zero_f64x2, cross_01_upper_f64x2 = zero_f64x2;
    v128_t cross_02_lower_f64x2 = zero_f64x2, cross_02_upper_f64x2 = zero_f64x2;
    v128_t cross_10_lower_f64x2 = zero_f64x2, cross_10_upper_f64x2 = zero_f64x2;
    v128_t cross_11_lower_f64x2 = zero_f64x2, cross_11_upper_f64x2 = zero_f64x2;
    v128_t cross_12_lower_f64x2 = zero_f64x2, cross_12_upper_f64x2 = zero_f64x2;
    v128_t cross_20_lower_f64x2 = zero_f64x2, cross_20_upper_f64x2 = zero_f64x2;
    v128_t cross_21_lower_f64x2 = zero_f64x2, cross_21_upper_f64x2 = zero_f64x2;
    v128_t cross_22_lower_f64x2 = zero_f64x2, cross_22_upper_f64x2 = zero_f64x2;
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_v128relaxed_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        v128_t a_x_lower_f64x2 = wasm_f64x2_promote_low_f32x4(a_x_f32x4);
        v128_t a_x_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_x_f32x4, a_x_f32x4, 2, 3, 0, 1));
        v128_t a_y_lower_f64x2 = wasm_f64x2_promote_low_f32x4(a_y_f32x4);
        v128_t a_y_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_y_f32x4, a_y_f32x4, 2, 3, 0, 1));
        v128_t a_z_lower_f64x2 = wasm_f64x2_promote_low_f32x4(a_z_f32x4);
        v128_t a_z_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_z_f32x4, a_z_f32x4, 2, 3, 0, 1));
        v128_t b_x_lower_f64x2 = wasm_f64x2_promote_low_f32x4(b_x_f32x4);
        v128_t b_x_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_x_f32x4, b_x_f32x4, 2, 3, 0, 1));
        v128_t b_y_lower_f64x2 = wasm_f64x2_promote_low_f32x4(b_y_f32x4);
        v128_t b_y_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_y_f32x4, b_y_f32x4, 2, 3, 0, 1));
        v128_t b_z_lower_f64x2 = wasm_f64x2_promote_low_f32x4(b_z_f32x4);
        v128_t b_z_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_z_f32x4, b_z_f32x4, 2, 3, 0, 1));

        sum_a_x_lower_f64x2 = wasm_f64x2_add(sum_a_x_lower_f64x2, a_x_lower_f64x2),
        sum_a_x_upper_f64x2 = wasm_f64x2_add(sum_a_x_upper_f64x2, a_x_upper_f64x2);
        sum_a_y_lower_f64x2 = wasm_f64x2_add(sum_a_y_lower_f64x2, a_y_lower_f64x2),
        sum_a_y_upper_f64x2 = wasm_f64x2_add(sum_a_y_upper_f64x2, a_y_upper_f64x2);
        sum_a_z_lower_f64x2 = wasm_f64x2_add(sum_a_z_lower_f64x2, a_z_lower_f64x2),
        sum_a_z_upper_f64x2 = wasm_f64x2_add(sum_a_z_upper_f64x2, a_z_upper_f64x2);
        sum_b_x_lower_f64x2 = wasm_f64x2_add(sum_b_x_lower_f64x2, b_x_lower_f64x2),
        sum_b_x_upper_f64x2 = wasm_f64x2_add(sum_b_x_upper_f64x2, b_x_upper_f64x2);
        sum_b_y_lower_f64x2 = wasm_f64x2_add(sum_b_y_lower_f64x2, b_y_lower_f64x2),
        sum_b_y_upper_f64x2 = wasm_f64x2_add(sum_b_y_upper_f64x2, b_y_upper_f64x2);
        sum_b_z_lower_f64x2 = wasm_f64x2_add(sum_b_z_lower_f64x2, b_z_lower_f64x2),
        sum_b_z_upper_f64x2 = wasm_f64x2_add(sum_b_z_upper_f64x2, b_z_upper_f64x2);

        cross_00_lower_f64x2 = wasm_f64x2_relaxed_madd(a_x_lower_f64x2, b_x_lower_f64x2, cross_00_lower_f64x2),
        cross_00_upper_f64x2 = wasm_f64x2_relaxed_madd(a_x_upper_f64x2, b_x_upper_f64x2, cross_00_upper_f64x2);
        cross_01_lower_f64x2 = wasm_f64x2_relaxed_madd(a_x_lower_f64x2, b_y_lower_f64x2, cross_01_lower_f64x2),
        cross_01_upper_f64x2 = wasm_f64x2_relaxed_madd(a_x_upper_f64x2, b_y_upper_f64x2, cross_01_upper_f64x2);
        cross_02_lower_f64x2 = wasm_f64x2_relaxed_madd(a_x_lower_f64x2, b_z_lower_f64x2, cross_02_lower_f64x2),
        cross_02_upper_f64x2 = wasm_f64x2_relaxed_madd(a_x_upper_f64x2, b_z_upper_f64x2, cross_02_upper_f64x2);
        cross_10_lower_f64x2 = wasm_f64x2_relaxed_madd(a_y_lower_f64x2, b_x_lower_f64x2, cross_10_lower_f64x2),
        cross_10_upper_f64x2 = wasm_f64x2_relaxed_madd(a_y_upper_f64x2, b_x_upper_f64x2, cross_10_upper_f64x2);
        cross_11_lower_f64x2 = wasm_f64x2_relaxed_madd(a_y_lower_f64x2, b_y_lower_f64x2, cross_11_lower_f64x2),
        cross_11_upper_f64x2 = wasm_f64x2_relaxed_madd(a_y_upper_f64x2, b_y_upper_f64x2, cross_11_upper_f64x2);
        cross_12_lower_f64x2 = wasm_f64x2_relaxed_madd(a_y_lower_f64x2, b_z_lower_f64x2, cross_12_lower_f64x2),
        cross_12_upper_f64x2 = wasm_f64x2_relaxed_madd(a_y_upper_f64x2, b_z_upper_f64x2, cross_12_upper_f64x2);
        cross_20_lower_f64x2 = wasm_f64x2_relaxed_madd(a_z_lower_f64x2, b_x_lower_f64x2, cross_20_lower_f64x2),
        cross_20_upper_f64x2 = wasm_f64x2_relaxed_madd(a_z_upper_f64x2, b_x_upper_f64x2, cross_20_upper_f64x2);
        cross_21_lower_f64x2 = wasm_f64x2_relaxed_madd(a_z_lower_f64x2, b_y_lower_f64x2, cross_21_lower_f64x2),
        cross_21_upper_f64x2 = wasm_f64x2_relaxed_madd(a_z_upper_f64x2, b_y_upper_f64x2, cross_21_upper_f64x2);
        cross_22_lower_f64x2 = wasm_f64x2_relaxed_madd(a_z_lower_f64x2, b_z_lower_f64x2, cross_22_lower_f64x2),
        cross_22_upper_f64x2 = wasm_f64x2_relaxed_madd(a_z_upper_f64x2, b_z_upper_f64x2, cross_22_upper_f64x2);
    }

    nk_f64_t sum_a_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_x_lower_f64x2, sum_a_x_upper_f64x2));
    nk_f64_t sum_a_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_y_lower_f64x2, sum_a_y_upper_f64x2));
    nk_f64_t sum_a_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_z_lower_f64x2, sum_a_z_upper_f64x2));
    nk_f64_t sum_b_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_x_lower_f64x2, sum_b_x_upper_f64x2));
    nk_f64_t sum_b_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_y_lower_f64x2, sum_b_y_upper_f64x2));
    nk_f64_t sum_b_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_z_lower_f64x2, sum_b_z_upper_f64x2));
    nk_f64_t cross_00 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_00_lower_f64x2, cross_00_upper_f64x2));
    nk_f64_t cross_01 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_01_lower_f64x2, cross_01_upper_f64x2));
    nk_f64_t cross_02 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_02_lower_f64x2, cross_02_upper_f64x2));
    nk_f64_t cross_10 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_10_lower_f64x2, cross_10_upper_f64x2));
    nk_f64_t cross_11 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_11_lower_f64x2, cross_11_upper_f64x2));
    nk_f64_t cross_12 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_12_lower_f64x2, cross_12_upper_f64x2));
    nk_f64_t cross_20 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_20_lower_f64x2, cross_20_upper_f64x2));
    nk_f64_t cross_21 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_21_lower_f64x2, cross_21_upper_f64x2));
    nk_f64_t cross_22 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_22_lower_f64x2, cross_22_upper_f64x2));

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        cross_00 += a_x * b_x, cross_01 += a_x * b_y, cross_02 += a_x * b_z;
        cross_10 += a_y * b_x, cross_11 += a_y * b_y, cross_12 += a_y * b_z;
        cross_20 += a_z * b_x, cross_21 += a_z * b_y, cross_22 += a_z * b_z;
    }

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    *ca_x = sum_a_x * inv_n, *ca_y = sum_a_y * inv_n, *ca_z = sum_a_z * inv_n;
    *cb_x = sum_b_x * inv_n, *cb_y = sum_b_y * inv_n, *cb_z = sum_b_z * inv_n;

    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = cross_00 - n_f64 * (*ca_x) * (*cb_x), h[1] = cross_01 - n_f64 * (*ca_x) * (*cb_y),
    h[2] = cross_02 - n_f64 * (*ca_x) * (*cb_z);
    h[3] = cross_10 - n_f64 * (*ca_y) * (*cb_x), h[4] = cross_11 - n_f64 * (*ca_y) * (*cb_y),
    h[5] = cross_12 - n_f64 * (*ca_y) * (*cb_z);
    h[6] = cross_20 - n_f64 * (*ca_z) * (*cb_x), h[7] = cross_21 - n_f64 * (*ca_z) * (*cb_y),
    h[8] = cross_22 - n_f64 * (*ca_z) * (*cb_z);
}

NK_INTERNAL void nk_centroid_and_cross_covariance_and_variance_f32_v128relaxed_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,                           //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,                              //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,                              //
    nk_f64_t h[9], nk_f64_t *variance_a) {
    v128_t zero_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_a_x_lower_f64x2 = zero_f64x2, sum_a_x_upper_f64x2 = zero_f64x2;
    v128_t sum_a_y_lower_f64x2 = zero_f64x2, sum_a_y_upper_f64x2 = zero_f64x2;
    v128_t sum_a_z_lower_f64x2 = zero_f64x2, sum_a_z_upper_f64x2 = zero_f64x2;
    v128_t sum_b_x_lower_f64x2 = zero_f64x2, sum_b_x_upper_f64x2 = zero_f64x2;
    v128_t sum_b_y_lower_f64x2 = zero_f64x2, sum_b_y_upper_f64x2 = zero_f64x2;
    v128_t sum_b_z_lower_f64x2 = zero_f64x2, sum_b_z_upper_f64x2 = zero_f64x2;
    v128_t cross_00_lower_f64x2 = zero_f64x2, cross_00_upper_f64x2 = zero_f64x2;
    v128_t cross_01_lower_f64x2 = zero_f64x2, cross_01_upper_f64x2 = zero_f64x2;
    v128_t cross_02_lower_f64x2 = zero_f64x2, cross_02_upper_f64x2 = zero_f64x2;
    v128_t cross_10_lower_f64x2 = zero_f64x2, cross_10_upper_f64x2 = zero_f64x2;
    v128_t cross_11_lower_f64x2 = zero_f64x2, cross_11_upper_f64x2 = zero_f64x2;
    v128_t cross_12_lower_f64x2 = zero_f64x2, cross_12_upper_f64x2 = zero_f64x2;
    v128_t cross_20_lower_f64x2 = zero_f64x2, cross_20_upper_f64x2 = zero_f64x2;
    v128_t cross_21_lower_f64x2 = zero_f64x2, cross_21_upper_f64x2 = zero_f64x2;
    v128_t cross_22_lower_f64x2 = zero_f64x2, cross_22_upper_f64x2 = zero_f64x2;
    v128_t sum_norm_squared_lower_f64x2 = zero_f64x2, sum_norm_squared_upper_f64x2 = zero_f64x2;
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_v128relaxed_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        v128_t a_x_lower_f64x2 = wasm_f64x2_promote_low_f32x4(a_x_f32x4);
        v128_t a_x_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_x_f32x4, a_x_f32x4, 2, 3, 0, 1));
        v128_t a_y_lower_f64x2 = wasm_f64x2_promote_low_f32x4(a_y_f32x4);
        v128_t a_y_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_y_f32x4, a_y_f32x4, 2, 3, 0, 1));
        v128_t a_z_lower_f64x2 = wasm_f64x2_promote_low_f32x4(a_z_f32x4);
        v128_t a_z_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_z_f32x4, a_z_f32x4, 2, 3, 0, 1));
        v128_t b_x_lower_f64x2 = wasm_f64x2_promote_low_f32x4(b_x_f32x4);
        v128_t b_x_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_x_f32x4, b_x_f32x4, 2, 3, 0, 1));
        v128_t b_y_lower_f64x2 = wasm_f64x2_promote_low_f32x4(b_y_f32x4);
        v128_t b_y_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_y_f32x4, b_y_f32x4, 2, 3, 0, 1));
        v128_t b_z_lower_f64x2 = wasm_f64x2_promote_low_f32x4(b_z_f32x4);
        v128_t b_z_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_z_f32x4, b_z_f32x4, 2, 3, 0, 1));

        sum_a_x_lower_f64x2 = wasm_f64x2_add(sum_a_x_lower_f64x2, a_x_lower_f64x2),
        sum_a_x_upper_f64x2 = wasm_f64x2_add(sum_a_x_upper_f64x2, a_x_upper_f64x2);
        sum_a_y_lower_f64x2 = wasm_f64x2_add(sum_a_y_lower_f64x2, a_y_lower_f64x2),
        sum_a_y_upper_f64x2 = wasm_f64x2_add(sum_a_y_upper_f64x2, a_y_upper_f64x2);
        sum_a_z_lower_f64x2 = wasm_f64x2_add(sum_a_z_lower_f64x2, a_z_lower_f64x2),
        sum_a_z_upper_f64x2 = wasm_f64x2_add(sum_a_z_upper_f64x2, a_z_upper_f64x2);
        sum_b_x_lower_f64x2 = wasm_f64x2_add(sum_b_x_lower_f64x2, b_x_lower_f64x2),
        sum_b_x_upper_f64x2 = wasm_f64x2_add(sum_b_x_upper_f64x2, b_x_upper_f64x2);
        sum_b_y_lower_f64x2 = wasm_f64x2_add(sum_b_y_lower_f64x2, b_y_lower_f64x2),
        sum_b_y_upper_f64x2 = wasm_f64x2_add(sum_b_y_upper_f64x2, b_y_upper_f64x2);
        sum_b_z_lower_f64x2 = wasm_f64x2_add(sum_b_z_lower_f64x2, b_z_lower_f64x2),
        sum_b_z_upper_f64x2 = wasm_f64x2_add(sum_b_z_upper_f64x2, b_z_upper_f64x2);

        cross_00_lower_f64x2 = wasm_f64x2_relaxed_madd(a_x_lower_f64x2, b_x_lower_f64x2, cross_00_lower_f64x2),
        cross_00_upper_f64x2 = wasm_f64x2_relaxed_madd(a_x_upper_f64x2, b_x_upper_f64x2, cross_00_upper_f64x2);
        cross_01_lower_f64x2 = wasm_f64x2_relaxed_madd(a_x_lower_f64x2, b_y_lower_f64x2, cross_01_lower_f64x2),
        cross_01_upper_f64x2 = wasm_f64x2_relaxed_madd(a_x_upper_f64x2, b_y_upper_f64x2, cross_01_upper_f64x2);
        cross_02_lower_f64x2 = wasm_f64x2_relaxed_madd(a_x_lower_f64x2, b_z_lower_f64x2, cross_02_lower_f64x2),
        cross_02_upper_f64x2 = wasm_f64x2_relaxed_madd(a_x_upper_f64x2, b_z_upper_f64x2, cross_02_upper_f64x2);
        cross_10_lower_f64x2 = wasm_f64x2_relaxed_madd(a_y_lower_f64x2, b_x_lower_f64x2, cross_10_lower_f64x2),
        cross_10_upper_f64x2 = wasm_f64x2_relaxed_madd(a_y_upper_f64x2, b_x_upper_f64x2, cross_10_upper_f64x2);
        cross_11_lower_f64x2 = wasm_f64x2_relaxed_madd(a_y_lower_f64x2, b_y_lower_f64x2, cross_11_lower_f64x2),
        cross_11_upper_f64x2 = wasm_f64x2_relaxed_madd(a_y_upper_f64x2, b_y_upper_f64x2, cross_11_upper_f64x2);
        cross_12_lower_f64x2 = wasm_f64x2_relaxed_madd(a_y_lower_f64x2, b_z_lower_f64x2, cross_12_lower_f64x2),
        cross_12_upper_f64x2 = wasm_f64x2_relaxed_madd(a_y_upper_f64x2, b_z_upper_f64x2, cross_12_upper_f64x2);
        cross_20_lower_f64x2 = wasm_f64x2_relaxed_madd(a_z_lower_f64x2, b_x_lower_f64x2, cross_20_lower_f64x2),
        cross_20_upper_f64x2 = wasm_f64x2_relaxed_madd(a_z_upper_f64x2, b_x_upper_f64x2, cross_20_upper_f64x2);
        cross_21_lower_f64x2 = wasm_f64x2_relaxed_madd(a_z_lower_f64x2, b_y_lower_f64x2, cross_21_lower_f64x2),
        cross_21_upper_f64x2 = wasm_f64x2_relaxed_madd(a_z_upper_f64x2, b_y_upper_f64x2, cross_21_upper_f64x2);
        cross_22_lower_f64x2 = wasm_f64x2_relaxed_madd(a_z_lower_f64x2, b_z_lower_f64x2, cross_22_lower_f64x2),
        cross_22_upper_f64x2 = wasm_f64x2_relaxed_madd(a_z_upper_f64x2, b_z_upper_f64x2, cross_22_upper_f64x2);

        // Variance: accumulate ||a||^2.
        v128_t norm_squared_lower_f64x2 = wasm_f64x2_relaxed_madd(a_y_lower_f64x2, a_y_lower_f64x2,
                                                                  wasm_f64x2_mul(a_x_lower_f64x2, a_x_lower_f64x2));
        v128_t norm_squared_upper_f64x2 = wasm_f64x2_relaxed_madd(a_y_upper_f64x2, a_y_upper_f64x2,
                                                                  wasm_f64x2_mul(a_x_upper_f64x2, a_x_upper_f64x2));
        norm_squared_lower_f64x2 = wasm_f64x2_relaxed_madd(a_z_lower_f64x2, a_z_lower_f64x2, norm_squared_lower_f64x2);
        norm_squared_upper_f64x2 = wasm_f64x2_relaxed_madd(a_z_upper_f64x2, a_z_upper_f64x2, norm_squared_upper_f64x2);
        sum_norm_squared_lower_f64x2 = wasm_f64x2_add(sum_norm_squared_lower_f64x2, norm_squared_lower_f64x2);
        sum_norm_squared_upper_f64x2 = wasm_f64x2_add(sum_norm_squared_upper_f64x2, norm_squared_upper_f64x2);
    }

    nk_f64_t sum_a_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_x_lower_f64x2, sum_a_x_upper_f64x2));
    nk_f64_t sum_a_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_y_lower_f64x2, sum_a_y_upper_f64x2));
    nk_f64_t sum_a_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_z_lower_f64x2, sum_a_z_upper_f64x2));
    nk_f64_t sum_b_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_x_lower_f64x2, sum_b_x_upper_f64x2));
    nk_f64_t sum_b_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_y_lower_f64x2, sum_b_y_upper_f64x2));
    nk_f64_t sum_b_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_z_lower_f64x2, sum_b_z_upper_f64x2));
    nk_f64_t cross_00 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_00_lower_f64x2, cross_00_upper_f64x2));
    nk_f64_t cross_01 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_01_lower_f64x2, cross_01_upper_f64x2));
    nk_f64_t cross_02 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_02_lower_f64x2, cross_02_upper_f64x2));
    nk_f64_t cross_10 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_10_lower_f64x2, cross_10_upper_f64x2));
    nk_f64_t cross_11 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_11_lower_f64x2, cross_11_upper_f64x2));
    nk_f64_t cross_12 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_12_lower_f64x2, cross_12_upper_f64x2));
    nk_f64_t cross_20 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_20_lower_f64x2, cross_20_upper_f64x2));
    nk_f64_t cross_21 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_21_lower_f64x2, cross_21_upper_f64x2));
    nk_f64_t cross_22 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_22_lower_f64x2, cross_22_upper_f64x2));
    nk_f64_t sum_norm_squared = nk_hsum_f64x2_v128relaxed_(
        wasm_f64x2_add(sum_norm_squared_lower_f64x2, sum_norm_squared_upper_f64x2));

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        cross_00 += a_x * b_x, cross_01 += a_x * b_y, cross_02 += a_x * b_z;
        cross_10 += a_y * b_x, cross_11 += a_y * b_y, cross_12 += a_y * b_z;
        cross_20 += a_z * b_x, cross_21 += a_z * b_y, cross_22 += a_z * b_z;
        sum_norm_squared += a_x * a_x + a_y * a_y + a_z * a_z;
    }

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    *ca_x = sum_a_x * inv_n, *ca_y = sum_a_y * inv_n, *ca_z = sum_a_z * inv_n;
    *cb_x = sum_b_x * inv_n, *cb_y = sum_b_y * inv_n, *cb_z = sum_b_z * inv_n;

    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = cross_00 - n_f64 * (*ca_x) * (*cb_x), h[1] = cross_01 - n_f64 * (*ca_x) * (*cb_y),
    h[2] = cross_02 - n_f64 * (*ca_x) * (*cb_z);
    h[3] = cross_10 - n_f64 * (*ca_y) * (*cb_x), h[4] = cross_11 - n_f64 * (*ca_y) * (*cb_y),
    h[5] = cross_12 - n_f64 * (*ca_y) * (*cb_z);
    h[6] = cross_20 - n_f64 * (*ca_z) * (*cb_x), h[7] = cross_21 - n_f64 * (*ca_z) * (*cb_y),
    h[8] = cross_22 - n_f64 * (*ca_z) * (*cb_z);
    *variance_a = sum_norm_squared * inv_n - ((*ca_x) * (*ca_x) + (*ca_y) * (*ca_y) + (*ca_z) * (*ca_z));
}

NK_INTERNAL nk_f64_t nk_transformed_ssd_f32_v128relaxed_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
    nk_f64_t centroid_a_y, nk_f64_t centroid_a_z, nk_f64_t centroid_b_x, nk_f64_t centroid_b_y, nk_f64_t centroid_b_z) {
    v128_t scaled_rotation_x_x_f64x2 = wasm_f64x2_splat(scale * r[0]);
    v128_t scaled_rotation_x_y_f64x2 = wasm_f64x2_splat(scale * r[1]);
    v128_t scaled_rotation_x_z_f64x2 = wasm_f64x2_splat(scale * r[2]);
    v128_t scaled_rotation_y_x_f64x2 = wasm_f64x2_splat(scale * r[3]);
    v128_t scaled_rotation_y_y_f64x2 = wasm_f64x2_splat(scale * r[4]);
    v128_t scaled_rotation_y_z_f64x2 = wasm_f64x2_splat(scale * r[5]);
    v128_t scaled_rotation_z_x_f64x2 = wasm_f64x2_splat(scale * r[6]);
    v128_t scaled_rotation_z_y_f64x2 = wasm_f64x2_splat(scale * r[7]);
    v128_t scaled_rotation_z_z_f64x2 = wasm_f64x2_splat(scale * r[8]);
    v128_t centroid_a_x_f64x2 = wasm_f64x2_splat(centroid_a_x), centroid_a_y_f64x2 = wasm_f64x2_splat(centroid_a_y);
    v128_t centroid_a_z_f64x2 = wasm_f64x2_splat(centroid_a_z), centroid_b_x_f64x2 = wasm_f64x2_splat(centroid_b_x);
    v128_t centroid_b_y_f64x2 = wasm_f64x2_splat(centroid_b_y), centroid_b_z_f64x2 = wasm_f64x2_splat(centroid_b_z);
    v128_t sum_squared_lower_f64x2 = wasm_f64x2_splat(0.0), sum_squared_upper_f64x2 = wasm_f64x2_splat(0.0);
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_v128relaxed_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        v128_t centered_a_x_lower_f64x2 = wasm_f64x2_sub(wasm_f64x2_promote_low_f32x4(a_x_f32x4), centroid_a_x_f64x2);
        v128_t centered_a_x_upper_f64x2 = wasm_f64x2_sub(
            wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_x_f32x4, a_x_f32x4, 2, 3, 0, 1)), centroid_a_x_f64x2);
        v128_t centered_a_y_lower_f64x2 = wasm_f64x2_sub(wasm_f64x2_promote_low_f32x4(a_y_f32x4), centroid_a_y_f64x2);
        v128_t centered_a_y_upper_f64x2 = wasm_f64x2_sub(
            wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_y_f32x4, a_y_f32x4, 2, 3, 0, 1)), centroid_a_y_f64x2);
        v128_t centered_a_z_lower_f64x2 = wasm_f64x2_sub(wasm_f64x2_promote_low_f32x4(a_z_f32x4), centroid_a_z_f64x2);
        v128_t centered_a_z_upper_f64x2 = wasm_f64x2_sub(
            wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_z_f32x4, a_z_f32x4, 2, 3, 0, 1)), centroid_a_z_f64x2);
        v128_t centered_b_x_lower_f64x2 = wasm_f64x2_sub(wasm_f64x2_promote_low_f32x4(b_x_f32x4), centroid_b_x_f64x2);
        v128_t centered_b_x_upper_f64x2 = wasm_f64x2_sub(
            wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_x_f32x4, b_x_f32x4, 2, 3, 0, 1)), centroid_b_x_f64x2);
        v128_t centered_b_y_lower_f64x2 = wasm_f64x2_sub(wasm_f64x2_promote_low_f32x4(b_y_f32x4), centroid_b_y_f64x2);
        v128_t centered_b_y_upper_f64x2 = wasm_f64x2_sub(
            wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_y_f32x4, b_y_f32x4, 2, 3, 0, 1)), centroid_b_y_f64x2);
        v128_t centered_b_z_lower_f64x2 = wasm_f64x2_sub(wasm_f64x2_promote_low_f32x4(b_z_f32x4), centroid_b_z_f64x2);
        v128_t centered_b_z_upper_f64x2 = wasm_f64x2_sub(
            wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_z_f32x4, b_z_f32x4, 2, 3, 0, 1)), centroid_b_z_f64x2);

        v128_t rotated_a_x_lower_f64x2 = wasm_f64x2_relaxed_madd(
            scaled_rotation_x_z_f64x2, centered_a_z_lower_f64x2,
            wasm_f64x2_relaxed_madd(scaled_rotation_x_y_f64x2, centered_a_y_lower_f64x2,
                                    wasm_f64x2_mul(scaled_rotation_x_x_f64x2, centered_a_x_lower_f64x2)));
        v128_t rotated_a_x_upper_f64x2 = wasm_f64x2_relaxed_madd(
            scaled_rotation_x_z_f64x2, centered_a_z_upper_f64x2,
            wasm_f64x2_relaxed_madd(scaled_rotation_x_y_f64x2, centered_a_y_upper_f64x2,
                                    wasm_f64x2_mul(scaled_rotation_x_x_f64x2, centered_a_x_upper_f64x2)));
        v128_t rotated_a_y_lower_f64x2 = wasm_f64x2_relaxed_madd(
            scaled_rotation_y_z_f64x2, centered_a_z_lower_f64x2,
            wasm_f64x2_relaxed_madd(scaled_rotation_y_y_f64x2, centered_a_y_lower_f64x2,
                                    wasm_f64x2_mul(scaled_rotation_y_x_f64x2, centered_a_x_lower_f64x2)));
        v128_t rotated_a_y_upper_f64x2 = wasm_f64x2_relaxed_madd(
            scaled_rotation_y_z_f64x2, centered_a_z_upper_f64x2,
            wasm_f64x2_relaxed_madd(scaled_rotation_y_y_f64x2, centered_a_y_upper_f64x2,
                                    wasm_f64x2_mul(scaled_rotation_y_x_f64x2, centered_a_x_upper_f64x2)));
        v128_t rotated_a_z_lower_f64x2 = wasm_f64x2_relaxed_madd(
            scaled_rotation_z_z_f64x2, centered_a_z_lower_f64x2,
            wasm_f64x2_relaxed_madd(scaled_rotation_z_y_f64x2, centered_a_y_lower_f64x2,
                                    wasm_f64x2_mul(scaled_rotation_z_x_f64x2, centered_a_x_lower_f64x2)));
        v128_t rotated_a_z_upper_f64x2 = wasm_f64x2_relaxed_madd(
            scaled_rotation_z_z_f64x2, centered_a_z_upper_f64x2,
            wasm_f64x2_relaxed_madd(scaled_rotation_z_y_f64x2, centered_a_y_upper_f64x2,
                                    wasm_f64x2_mul(scaled_rotation_z_x_f64x2, centered_a_x_upper_f64x2)));

        v128_t delta_x_lower_f64x2 = wasm_f64x2_sub(rotated_a_x_lower_f64x2, centered_b_x_lower_f64x2);
        v128_t delta_x_upper_f64x2 = wasm_f64x2_sub(rotated_a_x_upper_f64x2, centered_b_x_upper_f64x2);
        v128_t delta_y_lower_f64x2 = wasm_f64x2_sub(rotated_a_y_lower_f64x2, centered_b_y_lower_f64x2);
        v128_t delta_y_upper_f64x2 = wasm_f64x2_sub(rotated_a_y_upper_f64x2, centered_b_y_upper_f64x2);
        v128_t delta_z_lower_f64x2 = wasm_f64x2_sub(rotated_a_z_lower_f64x2, centered_b_z_lower_f64x2);
        v128_t delta_z_upper_f64x2 = wasm_f64x2_sub(rotated_a_z_upper_f64x2, centered_b_z_upper_f64x2);

        sum_squared_lower_f64x2 = wasm_f64x2_relaxed_madd(delta_x_lower_f64x2, delta_x_lower_f64x2,
                                                          sum_squared_lower_f64x2);
        sum_squared_upper_f64x2 = wasm_f64x2_relaxed_madd(delta_x_upper_f64x2, delta_x_upper_f64x2,
                                                          sum_squared_upper_f64x2);
        sum_squared_lower_f64x2 = wasm_f64x2_relaxed_madd(delta_y_lower_f64x2, delta_y_lower_f64x2,
                                                          sum_squared_lower_f64x2);
        sum_squared_upper_f64x2 = wasm_f64x2_relaxed_madd(delta_y_upper_f64x2, delta_y_upper_f64x2,
                                                          sum_squared_upper_f64x2);
        sum_squared_lower_f64x2 = wasm_f64x2_relaxed_madd(delta_z_lower_f64x2, delta_z_lower_f64x2,
                                                          sum_squared_lower_f64x2);
        sum_squared_upper_f64x2 = wasm_f64x2_relaxed_madd(delta_z_upper_f64x2, delta_z_upper_f64x2,
                                                          sum_squared_upper_f64x2);
    }

    nk_f64_t sum_squared = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_squared_lower_f64x2, sum_squared_upper_f64x2));
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

/*  Compute sum of squared distances for f64 after applying rotation (and optional scale). */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_v128relaxed_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                         nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                         nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                         nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                         nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    v128_t scaled_rotation_x_x_f64x2 = wasm_f64x2_splat(scale * r[0]);
    v128_t scaled_rotation_x_y_f64x2 = wasm_f64x2_splat(scale * r[1]);
    v128_t scaled_rotation_x_z_f64x2 = wasm_f64x2_splat(scale * r[2]);
    v128_t scaled_rotation_y_x_f64x2 = wasm_f64x2_splat(scale * r[3]);
    v128_t scaled_rotation_y_y_f64x2 = wasm_f64x2_splat(scale * r[4]);
    v128_t scaled_rotation_y_z_f64x2 = wasm_f64x2_splat(scale * r[5]);
    v128_t scaled_rotation_z_x_f64x2 = wasm_f64x2_splat(scale * r[6]);
    v128_t scaled_rotation_z_y_f64x2 = wasm_f64x2_splat(scale * r[7]);
    v128_t scaled_rotation_z_z_f64x2 = wasm_f64x2_splat(scale * r[8]);

    // Broadcast centroids
    v128_t centroid_a_x_f64x2 = wasm_f64x2_splat(centroid_a_x);
    v128_t centroid_a_y_f64x2 = wasm_f64x2_splat(centroid_a_y);
    v128_t centroid_a_z_f64x2 = wasm_f64x2_splat(centroid_a_z);
    v128_t centroid_b_x_f64x2 = wasm_f64x2_splat(centroid_b_x);
    v128_t centroid_b_y_f64x2 = wasm_f64x2_splat(centroid_b_y);
    v128_t centroid_b_z_f64x2 = wasm_f64x2_splat(centroid_b_z);

    v128_t sum_squared_f64x2 = wasm_f64x2_splat(0), sum_squared_compensation_f64x2 = wasm_f64x2_splat(0);
    nk_size_t j = 0;

    // Main loop: process 2 points per iteration
    for (; j + 2 <= n; j += 2) {
        v128_t a_x, a_y, a_z, b_x, b_y, b_z;
        nk_deinterleave_f64x2_v128relaxed_(a + j * 3, &a_x, &a_y, &a_z);
        nk_deinterleave_f64x2_v128relaxed_(b + j * 3, &b_x, &b_y, &b_z);

        v128_t pa_x = wasm_f64x2_sub(a_x, centroid_a_x_f64x2);
        v128_t pa_y = wasm_f64x2_sub(a_y, centroid_a_y_f64x2);
        v128_t pa_z = wasm_f64x2_sub(a_z, centroid_a_z_f64x2);
        v128_t pb_x = wasm_f64x2_sub(b_x, centroid_b_x_f64x2);
        v128_t pb_y = wasm_f64x2_sub(b_y, centroid_b_y_f64x2);
        v128_t pb_z = wasm_f64x2_sub(b_z, centroid_b_z_f64x2);

        // Rotate and scale: ra = scale * R * pa
        v128_t ra_x = wasm_f64x2_relaxed_madd(
            scaled_rotation_x_z_f64x2, pa_z,
            wasm_f64x2_relaxed_madd(scaled_rotation_x_y_f64x2, pa_y, wasm_f64x2_mul(scaled_rotation_x_x_f64x2, pa_x)));
        v128_t ra_y = wasm_f64x2_relaxed_madd(
            scaled_rotation_y_z_f64x2, pa_z,
            wasm_f64x2_relaxed_madd(scaled_rotation_y_y_f64x2, pa_y, wasm_f64x2_mul(scaled_rotation_y_x_f64x2, pa_x)));
        v128_t ra_z = wasm_f64x2_relaxed_madd(
            scaled_rotation_z_z_f64x2, pa_z,
            wasm_f64x2_relaxed_madd(scaled_rotation_z_y_f64x2, pa_y, wasm_f64x2_mul(scaled_rotation_z_x_f64x2, pa_x)));

        v128_t delta_x = wasm_f64x2_sub(ra_x, pb_x);
        v128_t delta_y = wasm_f64x2_sub(ra_y, pb_y);
        v128_t delta_z = wasm_f64x2_sub(ra_z, pb_z);

        nk_accumulate_square_f64x2_v128relaxed_(&sum_squared_f64x2, &sum_squared_compensation_f64x2, delta_x);
        nk_accumulate_square_f64x2_v128relaxed_(&sum_squared_f64x2, &sum_squared_compensation_f64x2, delta_y);
        nk_accumulate_square_f64x2_v128relaxed_(&sum_squared_f64x2, &sum_squared_compensation_f64x2, delta_z);
    }

    nk_f64_t sum_squared = nk_dot_stable_sum_f64x2_v128relaxed_(sum_squared_f64x2, sum_squared_compensation_f64x2);
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

NK_PUBLIC void nk_rmsd_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    // Fused single-pass: accumulate centroids and squared differences simultaneously.
    // RMSD = √(E[(a−b)²] − (ā − b̄)²)
    v128_t zero_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_a_x_lower_f64x2 = zero_f64x2, sum_a_x_upper_f64x2 = zero_f64x2;
    v128_t sum_a_y_lower_f64x2 = zero_f64x2, sum_a_y_upper_f64x2 = zero_f64x2;
    v128_t sum_a_z_lower_f64x2 = zero_f64x2, sum_a_z_upper_f64x2 = zero_f64x2;
    v128_t sum_b_x_lower_f64x2 = zero_f64x2, sum_b_x_upper_f64x2 = zero_f64x2;
    v128_t sum_b_y_lower_f64x2 = zero_f64x2, sum_b_y_upper_f64x2 = zero_f64x2;
    v128_t sum_b_z_lower_f64x2 = zero_f64x2, sum_b_z_upper_f64x2 = zero_f64x2;
    v128_t sum_sq_x_lower_f64x2 = zero_f64x2, sum_sq_x_upper_f64x2 = zero_f64x2;
    v128_t sum_sq_y_lower_f64x2 = zero_f64x2, sum_sq_y_upper_f64x2 = zero_f64x2;
    v128_t sum_sq_z_lower_f64x2 = zero_f64x2, sum_sq_z_upper_f64x2 = zero_f64x2;
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_v128relaxed_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        // Promote lower and upper halves to f64. Deltas computed in f64 to avoid
        // f32 cancellation in the single-pass formula RMSD = √(E[(a−b)²] − (ā − b̄)²).
        v128_t a_x_lower_f64x2 = wasm_f64x2_promote_low_f32x4(a_x_f32x4);
        v128_t a_x_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_x_f32x4, a_x_f32x4, 2, 3, 0, 1));
        v128_t a_y_lower_f64x2 = wasm_f64x2_promote_low_f32x4(a_y_f32x4);
        v128_t a_y_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_y_f32x4, a_y_f32x4, 2, 3, 0, 1));
        v128_t a_z_lower_f64x2 = wasm_f64x2_promote_low_f32x4(a_z_f32x4);
        v128_t a_z_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_z_f32x4, a_z_f32x4, 2, 3, 0, 1));
        v128_t b_x_lower_f64x2 = wasm_f64x2_promote_low_f32x4(b_x_f32x4);
        v128_t b_x_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_x_f32x4, b_x_f32x4, 2, 3, 0, 1));
        v128_t b_y_lower_f64x2 = wasm_f64x2_promote_low_f32x4(b_y_f32x4);
        v128_t b_y_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_y_f32x4, b_y_f32x4, 2, 3, 0, 1));
        v128_t b_z_lower_f64x2 = wasm_f64x2_promote_low_f32x4(b_z_f32x4);
        v128_t b_z_upper_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_z_f32x4, b_z_f32x4, 2, 3, 0, 1));

        // Accumulate centroids.
        sum_a_x_lower_f64x2 = wasm_f64x2_add(sum_a_x_lower_f64x2, a_x_lower_f64x2);
        sum_a_x_upper_f64x2 = wasm_f64x2_add(sum_a_x_upper_f64x2, a_x_upper_f64x2);
        sum_a_y_lower_f64x2 = wasm_f64x2_add(sum_a_y_lower_f64x2, a_y_lower_f64x2);
        sum_a_y_upper_f64x2 = wasm_f64x2_add(sum_a_y_upper_f64x2, a_y_upper_f64x2);
        sum_a_z_lower_f64x2 = wasm_f64x2_add(sum_a_z_lower_f64x2, a_z_lower_f64x2);
        sum_a_z_upper_f64x2 = wasm_f64x2_add(sum_a_z_upper_f64x2, a_z_upper_f64x2);
        sum_b_x_lower_f64x2 = wasm_f64x2_add(sum_b_x_lower_f64x2, b_x_lower_f64x2);
        sum_b_x_upper_f64x2 = wasm_f64x2_add(sum_b_x_upper_f64x2, b_x_upper_f64x2);
        sum_b_y_lower_f64x2 = wasm_f64x2_add(sum_b_y_lower_f64x2, b_y_lower_f64x2);
        sum_b_y_upper_f64x2 = wasm_f64x2_add(sum_b_y_upper_f64x2, b_y_upper_f64x2);
        sum_b_z_lower_f64x2 = wasm_f64x2_add(sum_b_z_lower_f64x2, b_z_lower_f64x2);
        sum_b_z_upper_f64x2 = wasm_f64x2_add(sum_b_z_upper_f64x2, b_z_upper_f64x2);

        // Accumulate squared differences in f64 — deltas computed in f64 for precision.
        v128_t dx_lower_f64x2 = wasm_f64x2_sub(a_x_lower_f64x2, b_x_lower_f64x2);
        v128_t dx_upper_f64x2 = wasm_f64x2_sub(a_x_upper_f64x2, b_x_upper_f64x2);
        v128_t dy_lower_f64x2 = wasm_f64x2_sub(a_y_lower_f64x2, b_y_lower_f64x2);
        v128_t dy_upper_f64x2 = wasm_f64x2_sub(a_y_upper_f64x2, b_y_upper_f64x2);
        v128_t dz_lower_f64x2 = wasm_f64x2_sub(a_z_lower_f64x2, b_z_lower_f64x2);
        v128_t dz_upper_f64x2 = wasm_f64x2_sub(a_z_upper_f64x2, b_z_upper_f64x2);

        sum_sq_x_lower_f64x2 = wasm_f64x2_relaxed_madd(dx_lower_f64x2, dx_lower_f64x2, sum_sq_x_lower_f64x2);
        sum_sq_x_upper_f64x2 = wasm_f64x2_relaxed_madd(dx_upper_f64x2, dx_upper_f64x2, sum_sq_x_upper_f64x2);
        sum_sq_y_lower_f64x2 = wasm_f64x2_relaxed_madd(dy_lower_f64x2, dy_lower_f64x2, sum_sq_y_lower_f64x2);
        sum_sq_y_upper_f64x2 = wasm_f64x2_relaxed_madd(dy_upper_f64x2, dy_upper_f64x2, sum_sq_y_upper_f64x2);
        sum_sq_z_lower_f64x2 = wasm_f64x2_relaxed_madd(dz_lower_f64x2, dz_lower_f64x2, sum_sq_z_lower_f64x2);
        sum_sq_z_upper_f64x2 = wasm_f64x2_relaxed_madd(dz_upper_f64x2, dz_upper_f64x2, sum_sq_z_upper_f64x2);
    }

    nk_f64_t sum_a_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_x_lower_f64x2, sum_a_x_upper_f64x2));
    nk_f64_t sum_a_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_y_lower_f64x2, sum_a_y_upper_f64x2));
    nk_f64_t sum_a_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_z_lower_f64x2, sum_a_z_upper_f64x2));
    nk_f64_t sum_b_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_x_lower_f64x2, sum_b_x_upper_f64x2));
    nk_f64_t sum_b_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_y_lower_f64x2, sum_b_y_upper_f64x2));
    nk_f64_t sum_b_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_z_lower_f64x2, sum_b_z_upper_f64x2));
    nk_f64_t sum_sq_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_sq_x_lower_f64x2, sum_sq_x_upper_f64x2));
    nk_f64_t sum_sq_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_sq_y_lower_f64x2, sum_sq_y_upper_f64x2));
    nk_f64_t sum_sq_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_sq_z_lower_f64x2, sum_sq_z_upper_f64x2));

    // Scalar tail.
    for (; index < n; ++index) {
        nk_f64_t ax = a[index * 3 + 0], ay = a[index * 3 + 1], az = a[index * 3 + 2];
        nk_f64_t bx = b[index * 3 + 0], by = b[index * 3 + 1], bz = b[index * 3 + 2];
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        nk_f64_t dx = ax - bx, dy = ay - by, dz = az - bz;
        sum_sq_x += dx * dx, sum_sq_y += dy * dy, sum_sq_z += dz * dz;
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

    nk_f64_t sum_squared = sum_sq_x + sum_sq_y + sum_sq_z;
    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f64_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f64_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;
    *result = nk_f64_sqrt_v128relaxed(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_rmsd_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                       nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // RMSD uses identity rotation and scale=1.0
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0;

    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Accumulators for centroids and squared differences
    v128_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    v128_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;
    v128_t sum_squared_x_f64x2 = zeros_f64x2, sum_squared_y_f64x2 = zeros_f64x2, sum_squared_z_f64x2 = zeros_f64x2;

    v128_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;
    nk_size_t i = 0;

    // Main loop processing 2 points at a time
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_v128relaxed_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_v128relaxed_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        sum_a_x_f64x2 = wasm_f64x2_add(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = wasm_f64x2_add(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = wasm_f64x2_add(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = wasm_f64x2_add(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = wasm_f64x2_add(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = wasm_f64x2_add(sum_b_z_f64x2, b_z_f64x2);

        v128_t delta_x_f64x2 = wasm_f64x2_sub(a_x_f64x2, b_x_f64x2);
        v128_t delta_y_f64x2 = wasm_f64x2_sub(a_y_f64x2, b_y_f64x2);
        v128_t delta_z_f64x2 = wasm_f64x2_sub(a_z_f64x2, b_z_f64x2);

        sum_squared_x_f64x2 = wasm_f64x2_relaxed_madd(delta_x_f64x2, delta_x_f64x2, sum_squared_x_f64x2);
        sum_squared_y_f64x2 = wasm_f64x2_relaxed_madd(delta_y_f64x2, delta_y_f64x2, sum_squared_y_f64x2);
        sum_squared_z_f64x2 = wasm_f64x2_relaxed_madd(delta_z_f64x2, delta_z_f64x2, sum_squared_z_f64x2);
    }

    // Reduce vectors to scalars.
    nk_f64_t total_ax = nk_reduce_stable_f64x2_v128relaxed_(sum_a_x_f64x2), total_ax_compensation = 0.0;
    nk_f64_t total_ay = nk_reduce_stable_f64x2_v128relaxed_(sum_a_y_f64x2), total_ay_compensation = 0.0;
    nk_f64_t total_az = nk_reduce_stable_f64x2_v128relaxed_(sum_a_z_f64x2), total_az_compensation = 0.0;
    nk_f64_t total_bx = nk_reduce_stable_f64x2_v128relaxed_(sum_b_x_f64x2), total_bx_compensation = 0.0;
    nk_f64_t total_by = nk_reduce_stable_f64x2_v128relaxed_(sum_b_y_f64x2), total_by_compensation = 0.0;
    nk_f64_t total_bz = nk_reduce_stable_f64x2_v128relaxed_(sum_b_z_f64x2), total_bz_compensation = 0.0;
    nk_f64_t total_squared_x = nk_reduce_stable_f64x2_v128relaxed_(sum_squared_x_f64x2),
             total_squared_x_compensation = 0.0;
    nk_f64_t total_squared_y = nk_reduce_stable_f64x2_v128relaxed_(sum_squared_y_f64x2),
             total_squared_y_compensation = 0.0;
    nk_f64_t total_squared_z = nk_reduce_stable_f64x2_v128relaxed_(sum_squared_z_f64x2),
             total_squared_z_compensation = 0.0;

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
        nk_accumulate_square_f64_(&total_squared_x, &total_squared_x_compensation, delta_x);
        nk_accumulate_square_f64_(&total_squared_y, &total_squared_y_compensation, delta_y);
        nk_accumulate_square_f64_(&total_squared_z, &total_squared_z_compensation, delta_z);
    }

    total_ax += total_ax_compensation, total_ay += total_ay_compensation, total_az += total_az_compensation;
    total_bx += total_bx_compensation, total_by += total_by_compensation, total_bz += total_bz_compensation;
    total_squared_x += total_squared_x_compensation, total_squared_y += total_squared_y_compensation,
        total_squared_z += total_squared_z_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = total_ax * inv_n, centroid_a_y = total_ay * inv_n, centroid_a_z = total_az * inv_n;
    nk_f64_t centroid_b_x = total_bx * inv_n, centroid_b_y = total_by * inv_n, centroid_b_z = total_bz * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute RMSD
    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f64_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f64_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t sum_squared = total_squared_x + total_squared_y + total_squared_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f64_sqrt_v128relaxed(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z;
    nk_f64_t h[9];
    nk_centroid_and_cross_covariance_f32_v128relaxed_(a, b, n, &centroid_a_x, &centroid_a_y, &centroid_a_z,
                                                      &centroid_b_x, &centroid_b_y, &centroid_b_z, h);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R.
    if (nk_det3x3_f64_(r) < 0) {
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

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)r[j];
    }
    if (scale) *scale = 1.0f;

    *result = nk_f64_sqrt_v128relaxed(nk_transformed_ssd_f32_v128relaxed_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y,
                                                                          centroid_a_z, centroid_b_x, centroid_b_y,
                                                                          centroid_b_z) /
                                      (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                         nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Single set of accumulators (1x unrolling)
    v128_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    v128_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;

    v128_t cov_xx_f64x2 = zeros_f64x2, cov_xy_f64x2 = zeros_f64x2, cov_xz_f64x2 = zeros_f64x2;
    v128_t cov_yx_f64x2 = zeros_f64x2, cov_yy_f64x2 = zeros_f64x2, cov_yz_f64x2 = zeros_f64x2;
    v128_t cov_zx_f64x2 = zeros_f64x2, cov_zy_f64x2 = zeros_f64x2, cov_zz_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    v128_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;

    // Main loop: 2 points per iteration
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_v128relaxed_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_v128relaxed_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        sum_a_x_f64x2 = wasm_f64x2_add(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = wasm_f64x2_add(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = wasm_f64x2_add(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = wasm_f64x2_add(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = wasm_f64x2_add(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = wasm_f64x2_add(sum_b_z_f64x2, b_z_f64x2);

        cov_xx_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_x_f64x2, cov_xx_f64x2);
        cov_xy_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_y_f64x2, cov_xy_f64x2);
        cov_xz_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_z_f64x2, cov_xz_f64x2);
        cov_yx_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_x_f64x2, cov_yx_f64x2);
        cov_yy_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_y_f64x2, cov_yy_f64x2);
        cov_yz_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_z_f64x2, cov_yz_f64x2);
        cov_zx_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_x_f64x2, cov_zx_f64x2);
        cov_zy_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_y_f64x2, cov_zy_f64x2);
        cov_zz_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_z_f64x2, cov_zz_f64x2);
    }

    // Reduce vector accumulators.
    nk_f64_t sum_a_x = nk_reduce_stable_f64x2_v128relaxed_(sum_a_x_f64x2), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x2_v128relaxed_(sum_a_y_f64x2), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x2_v128relaxed_(sum_a_z_f64x2), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x2_v128relaxed_(sum_b_x_f64x2), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x2_v128relaxed_(sum_b_y_f64x2), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x2_v128relaxed_(sum_b_z_f64x2), sum_b_z_compensation = 0.0;

    nk_f64_t covariance_x_x = nk_reduce_stable_f64x2_v128relaxed_(cov_xx_f64x2), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x2_v128relaxed_(cov_xy_f64x2), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x2_v128relaxed_(cov_xz_f64x2), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x2_v128relaxed_(cov_yx_f64x2), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x2_v128relaxed_(cov_yy_f64x2), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x2_v128relaxed_(cov_yz_f64x2), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x2_v128relaxed_(cov_zx_f64x2), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x2_v128relaxed_(cov_zy_f64x2), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x2_v128relaxed_(cov_zz_f64x2), covariance_z_z_compensation = 0.0;

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax),
            nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay),
            nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx),
            nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by),
            nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx),
            nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by),
            nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx),
            nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by),
            nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx),
            nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by),
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
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bT
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
    nk_f64_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_v128relaxed_(svd_u, svd_v, r);

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_v128relaxed_(svd_u, svd_v, r);
    }

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];

    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_v128relaxed_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y,
                                                               centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_v128relaxed(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                          nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z, variance_a;
    nk_f64_t h[9];
    nk_centroid_and_cross_covariance_and_variance_f32_v128relaxed_( //
        a, b, n, &centroid_a_x, &centroid_a_y, &centroid_a_z,       //
        &centroid_b_x, &centroid_b_y, &centroid_b_z, h, &variance_a);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    nk_f64_t det = nk_det3x3_f64_(r);
    if (det < 0) {
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

    nk_f64_t trace_signed_singular_values = svd_s[0] + svd_s[4] + (det < 0 ? -svd_s[8] : svd_s[8]);
    nk_f64_t computed_scale = trace_signed_singular_values / ((nk_f64_t)n * variance_a);
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)r[j];
    if (scale) *scale = (nk_f32_t)computed_scale;

    *result = nk_f64_sqrt_v128relaxed(nk_transformed_ssd_f32_v128relaxed_(a, b, n, r, computed_scale, centroid_a_x,
                                                                          centroid_a_y, centroid_a_z, centroid_b_x,
                                                                          centroid_b_y, centroid_b_z) /
                                      (nk_f64_t)n);
}

NK_PUBLIC void nk_umeyama_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                          nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Single set of accumulators (1x unrolling)
    v128_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    v128_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;

    v128_t cov_xx_f64x2 = zeros_f64x2, cov_xy_f64x2 = zeros_f64x2, cov_xz_f64x2 = zeros_f64x2;
    v128_t cov_yx_f64x2 = zeros_f64x2, cov_yy_f64x2 = zeros_f64x2, cov_yz_f64x2 = zeros_f64x2;
    v128_t cov_zx_f64x2 = zeros_f64x2, cov_zy_f64x2 = zeros_f64x2, cov_zz_f64x2 = zeros_f64x2;
    v128_t variance_a_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    v128_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;

    // Main loop: 2 points per iteration
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_v128relaxed_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_v128relaxed_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        sum_a_x_f64x2 = wasm_f64x2_add(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = wasm_f64x2_add(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = wasm_f64x2_add(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = wasm_f64x2_add(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = wasm_f64x2_add(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = wasm_f64x2_add(sum_b_z_f64x2, b_z_f64x2);

        cov_xx_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_x_f64x2, cov_xx_f64x2);
        cov_xy_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_y_f64x2, cov_xy_f64x2);
        cov_xz_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_z_f64x2, cov_xz_f64x2);
        cov_yx_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_x_f64x2, cov_yx_f64x2);
        cov_yy_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_y_f64x2, cov_yy_f64x2);
        cov_yz_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_z_f64x2, cov_yz_f64x2);
        cov_zx_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_x_f64x2, cov_zx_f64x2);
        cov_zy_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_y_f64x2, cov_zy_f64x2);
        cov_zz_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_z_f64x2, cov_zz_f64x2);

        variance_a_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, a_x_f64x2, variance_a_f64x2);
        variance_a_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, a_y_f64x2, variance_a_f64x2);
        variance_a_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, a_z_f64x2, variance_a_f64x2);
    }

    // Reduce vector accumulators.
    nk_f64_t sum_a_x = nk_reduce_stable_f64x2_v128relaxed_(sum_a_x_f64x2), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x2_v128relaxed_(sum_a_y_f64x2), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x2_v128relaxed_(sum_a_z_f64x2), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x2_v128relaxed_(sum_b_x_f64x2), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x2_v128relaxed_(sum_b_y_f64x2), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x2_v128relaxed_(sum_b_z_f64x2), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x2_v128relaxed_(cov_xx_f64x2), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x2_v128relaxed_(cov_xy_f64x2), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x2_v128relaxed_(cov_xz_f64x2), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x2_v128relaxed_(cov_yx_f64x2), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x2_v128relaxed_(cov_yy_f64x2), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x2_v128relaxed_(cov_yz_f64x2), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x2_v128relaxed_(cov_zx_f64x2), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x2_v128relaxed_(cov_zy_f64x2), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x2_v128relaxed_(cov_zz_f64x2), covariance_z_z_compensation = 0.0;
    nk_f64_t sum_sq_a = nk_reduce_stable_f64x2_v128relaxed_(variance_a_f64x2), sum_sq_a_compensation = 0.0;

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax),
            nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay),
            nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx),
            nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by),
            nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx),
            nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by),
            nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx),
            nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by),
            nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx),
            nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by),
            nk_accumulate_product_f64_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
        nk_accumulate_square_f64_(&sum_sq_a, &sum_sq_a_compensation, ax),
            nk_accumulate_square_f64_(&sum_sq_a, &sum_sq_a_compensation, ay),
            nk_accumulate_square_f64_(&sum_sq_a, &sum_sq_a_compensation, az);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    sum_sq_a += sum_sq_a_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute variance of A (centered)
    nk_f64_t centroid_sq = centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z;
    nk_f64_t var_a = sum_sq_a * inv_n - centroid_sq;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bT
    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    // Compute SVD
    nk_f64_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_v128relaxed_(svd_u, svd_v, r);

    // Handle reflection and compute scale
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t trace_d_s = svd_s[0] + svd_s[4] + (det < 0 ? -svd_s[8] : svd_s[8]);
    nk_f64_t computed_scale = trace_d_s / (n * var_a);

    if (det < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_v128relaxed_(svd_u, svd_v, r);
    }

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    if (scale) *scale = computed_scale;

    // Compute RMSD after transformation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_v128relaxed_(a, b, n, r, computed_scale, centroid_a_x, centroid_a_y,
                                                               centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_v128relaxed(sum_squared * inv_n);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_MESH_V128RELAXED_H
