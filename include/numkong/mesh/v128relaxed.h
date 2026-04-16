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
    v128_t x_partial_f32x4 = wasm_i32x4_shuffle(v0_f32x4, v1_f32x4, 0, 3, 6, 0); // x0 x1 x2 _
    *xs_f32x4 = wasm_i32x4_shuffle(x_partial_f32x4, v2_f32x4, 0, 1, 2, 5);       // x0 x1 x2 x3
    // y0 y1 y2 y3
    v128_t y_partial_f32x4 = wasm_i32x4_shuffle(v0_f32x4, v1_f32x4, 1, 4, 7, 0); // y0 y1 y2 _
    *ys_f32x4 = wasm_i32x4_shuffle(y_partial_f32x4, v2_f32x4, 0, 1, 2, 6);       // y0 y1 y2 y3
    // z0 z1 z2 z3
    v128_t z_partial_f32x4 = wasm_i32x4_shuffle(v0_f32x4, v1_f32x4, 2, 5, 0, 0); // z0 z1 _ _
    *zs_f32x4 = wasm_i32x4_shuffle(z_partial_f32x4, v2_f32x4, 0, 1, 4, 7);       // z0 z1 z2 z3
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

NK_INTERNAL void nk_centroid_and_cross_covariance_f32_v128relaxed_(         //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,                      //
    nk_f64_t *centroid_a_x, nk_f64_t *centroid_a_y, nk_f64_t *centroid_a_z, //
    nk_f64_t *centroid_b_x, nk_f64_t *centroid_b_y, nk_f64_t *centroid_b_z, //
    nk_f64_t cross_covariance[9], nk_f64_t *centered_norm_squared_a, nk_f64_t *centered_norm_squared_b) {
    v128_t zero_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_a_x_low_f64x2 = zero_f64x2, sum_a_x_high_f64x2 = zero_f64x2;
    v128_t sum_a_y_low_f64x2 = zero_f64x2, sum_a_y_high_f64x2 = zero_f64x2;
    v128_t sum_a_z_low_f64x2 = zero_f64x2, sum_a_z_high_f64x2 = zero_f64x2;
    v128_t sum_b_x_low_f64x2 = zero_f64x2, sum_b_x_high_f64x2 = zero_f64x2;
    v128_t sum_b_y_low_f64x2 = zero_f64x2, sum_b_y_high_f64x2 = zero_f64x2;
    v128_t sum_b_z_low_f64x2 = zero_f64x2, sum_b_z_high_f64x2 = zero_f64x2;
    v128_t cross_00_low_f64x2 = zero_f64x2, cross_00_high_f64x2 = zero_f64x2;
    v128_t cross_01_low_f64x2 = zero_f64x2, cross_01_high_f64x2 = zero_f64x2;
    v128_t cross_02_low_f64x2 = zero_f64x2, cross_02_high_f64x2 = zero_f64x2;
    v128_t cross_10_low_f64x2 = zero_f64x2, cross_10_high_f64x2 = zero_f64x2;
    v128_t cross_11_low_f64x2 = zero_f64x2, cross_11_high_f64x2 = zero_f64x2;
    v128_t cross_12_low_f64x2 = zero_f64x2, cross_12_high_f64x2 = zero_f64x2;
    v128_t cross_20_low_f64x2 = zero_f64x2, cross_20_high_f64x2 = zero_f64x2;
    v128_t cross_21_low_f64x2 = zero_f64x2, cross_21_high_f64x2 = zero_f64x2;
    v128_t cross_22_low_f64x2 = zero_f64x2, cross_22_high_f64x2 = zero_f64x2;
    v128_t norm_squared_a_low_f64x2 = zero_f64x2, norm_squared_a_high_f64x2 = zero_f64x2;
    v128_t norm_squared_b_low_f64x2 = zero_f64x2, norm_squared_b_high_f64x2 = zero_f64x2;
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_v128relaxed_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        v128_t a_x_low_f64x2 = wasm_f64x2_promote_low_f32x4(a_x_f32x4);
        v128_t a_x_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_x_f32x4, a_x_f32x4, 2, 3, 0, 1));
        v128_t a_y_low_f64x2 = wasm_f64x2_promote_low_f32x4(a_y_f32x4);
        v128_t a_y_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_y_f32x4, a_y_f32x4, 2, 3, 0, 1));
        v128_t a_z_low_f64x2 = wasm_f64x2_promote_low_f32x4(a_z_f32x4);
        v128_t a_z_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_z_f32x4, a_z_f32x4, 2, 3, 0, 1));
        v128_t b_x_low_f64x2 = wasm_f64x2_promote_low_f32x4(b_x_f32x4);
        v128_t b_x_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_x_f32x4, b_x_f32x4, 2, 3, 0, 1));
        v128_t b_y_low_f64x2 = wasm_f64x2_promote_low_f32x4(b_y_f32x4);
        v128_t b_y_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_y_f32x4, b_y_f32x4, 2, 3, 0, 1));
        v128_t b_z_low_f64x2 = wasm_f64x2_promote_low_f32x4(b_z_f32x4);
        v128_t b_z_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_z_f32x4, b_z_f32x4, 2, 3, 0, 1));

        sum_a_x_low_f64x2 = wasm_f64x2_add(sum_a_x_low_f64x2, a_x_low_f64x2),
        sum_a_x_high_f64x2 = wasm_f64x2_add(sum_a_x_high_f64x2, a_x_high_f64x2);
        sum_a_y_low_f64x2 = wasm_f64x2_add(sum_a_y_low_f64x2, a_y_low_f64x2),
        sum_a_y_high_f64x2 = wasm_f64x2_add(sum_a_y_high_f64x2, a_y_high_f64x2);
        sum_a_z_low_f64x2 = wasm_f64x2_add(sum_a_z_low_f64x2, a_z_low_f64x2),
        sum_a_z_high_f64x2 = wasm_f64x2_add(sum_a_z_high_f64x2, a_z_high_f64x2);
        sum_b_x_low_f64x2 = wasm_f64x2_add(sum_b_x_low_f64x2, b_x_low_f64x2),
        sum_b_x_high_f64x2 = wasm_f64x2_add(sum_b_x_high_f64x2, b_x_high_f64x2);
        sum_b_y_low_f64x2 = wasm_f64x2_add(sum_b_y_low_f64x2, b_y_low_f64x2),
        sum_b_y_high_f64x2 = wasm_f64x2_add(sum_b_y_high_f64x2, b_y_high_f64x2);
        sum_b_z_low_f64x2 = wasm_f64x2_add(sum_b_z_low_f64x2, b_z_low_f64x2),
        sum_b_z_high_f64x2 = wasm_f64x2_add(sum_b_z_high_f64x2, b_z_high_f64x2);

        cross_00_low_f64x2 = wasm_f64x2_relaxed_madd(a_x_low_f64x2, b_x_low_f64x2, cross_00_low_f64x2),
        cross_00_high_f64x2 = wasm_f64x2_relaxed_madd(a_x_high_f64x2, b_x_high_f64x2, cross_00_high_f64x2);
        cross_01_low_f64x2 = wasm_f64x2_relaxed_madd(a_x_low_f64x2, b_y_low_f64x2, cross_01_low_f64x2),
        cross_01_high_f64x2 = wasm_f64x2_relaxed_madd(a_x_high_f64x2, b_y_high_f64x2, cross_01_high_f64x2);
        cross_02_low_f64x2 = wasm_f64x2_relaxed_madd(a_x_low_f64x2, b_z_low_f64x2, cross_02_low_f64x2),
        cross_02_high_f64x2 = wasm_f64x2_relaxed_madd(a_x_high_f64x2, b_z_high_f64x2, cross_02_high_f64x2);
        cross_10_low_f64x2 = wasm_f64x2_relaxed_madd(a_y_low_f64x2, b_x_low_f64x2, cross_10_low_f64x2),
        cross_10_high_f64x2 = wasm_f64x2_relaxed_madd(a_y_high_f64x2, b_x_high_f64x2, cross_10_high_f64x2);
        cross_11_low_f64x2 = wasm_f64x2_relaxed_madd(a_y_low_f64x2, b_y_low_f64x2, cross_11_low_f64x2),
        cross_11_high_f64x2 = wasm_f64x2_relaxed_madd(a_y_high_f64x2, b_y_high_f64x2, cross_11_high_f64x2);
        cross_12_low_f64x2 = wasm_f64x2_relaxed_madd(a_y_low_f64x2, b_z_low_f64x2, cross_12_low_f64x2),
        cross_12_high_f64x2 = wasm_f64x2_relaxed_madd(a_y_high_f64x2, b_z_high_f64x2, cross_12_high_f64x2);
        cross_20_low_f64x2 = wasm_f64x2_relaxed_madd(a_z_low_f64x2, b_x_low_f64x2, cross_20_low_f64x2),
        cross_20_high_f64x2 = wasm_f64x2_relaxed_madd(a_z_high_f64x2, b_x_high_f64x2, cross_20_high_f64x2);
        cross_21_low_f64x2 = wasm_f64x2_relaxed_madd(a_z_low_f64x2, b_y_low_f64x2, cross_21_low_f64x2),
        cross_21_high_f64x2 = wasm_f64x2_relaxed_madd(a_z_high_f64x2, b_y_high_f64x2, cross_21_high_f64x2);
        cross_22_low_f64x2 = wasm_f64x2_relaxed_madd(a_z_low_f64x2, b_z_low_f64x2, cross_22_low_f64x2),
        cross_22_high_f64x2 = wasm_f64x2_relaxed_madd(a_z_high_f64x2, b_z_high_f64x2, cross_22_high_f64x2);

        norm_squared_a_low_f64x2 = wasm_f64x2_relaxed_madd(a_x_low_f64x2, a_x_low_f64x2, norm_squared_a_low_f64x2);
        norm_squared_a_high_f64x2 = wasm_f64x2_relaxed_madd(a_x_high_f64x2, a_x_high_f64x2, norm_squared_a_high_f64x2);
        norm_squared_a_low_f64x2 = wasm_f64x2_relaxed_madd(a_y_low_f64x2, a_y_low_f64x2, norm_squared_a_low_f64x2);
        norm_squared_a_high_f64x2 = wasm_f64x2_relaxed_madd(a_y_high_f64x2, a_y_high_f64x2, norm_squared_a_high_f64x2);
        norm_squared_a_low_f64x2 = wasm_f64x2_relaxed_madd(a_z_low_f64x2, a_z_low_f64x2, norm_squared_a_low_f64x2);
        norm_squared_a_high_f64x2 = wasm_f64x2_relaxed_madd(a_z_high_f64x2, a_z_high_f64x2, norm_squared_a_high_f64x2);
        norm_squared_b_low_f64x2 = wasm_f64x2_relaxed_madd(b_x_low_f64x2, b_x_low_f64x2, norm_squared_b_low_f64x2);
        norm_squared_b_high_f64x2 = wasm_f64x2_relaxed_madd(b_x_high_f64x2, b_x_high_f64x2, norm_squared_b_high_f64x2);
        norm_squared_b_low_f64x2 = wasm_f64x2_relaxed_madd(b_y_low_f64x2, b_y_low_f64x2, norm_squared_b_low_f64x2);
        norm_squared_b_high_f64x2 = wasm_f64x2_relaxed_madd(b_y_high_f64x2, b_y_high_f64x2, norm_squared_b_high_f64x2);
        norm_squared_b_low_f64x2 = wasm_f64x2_relaxed_madd(b_z_low_f64x2, b_z_low_f64x2, norm_squared_b_low_f64x2);
        norm_squared_b_high_f64x2 = wasm_f64x2_relaxed_madd(b_z_high_f64x2, b_z_high_f64x2, norm_squared_b_high_f64x2);
    }

    nk_f64_t sum_a_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_x_low_f64x2, sum_a_x_high_f64x2));
    nk_f64_t sum_a_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_y_low_f64x2, sum_a_y_high_f64x2));
    nk_f64_t sum_a_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_z_low_f64x2, sum_a_z_high_f64x2));
    nk_f64_t sum_b_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_x_low_f64x2, sum_b_x_high_f64x2));
    nk_f64_t sum_b_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_y_low_f64x2, sum_b_y_high_f64x2));
    nk_f64_t sum_b_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_z_low_f64x2, sum_b_z_high_f64x2));
    nk_f64_t cross_00 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_00_low_f64x2, cross_00_high_f64x2));
    nk_f64_t cross_01 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_01_low_f64x2, cross_01_high_f64x2));
    nk_f64_t cross_02 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_02_low_f64x2, cross_02_high_f64x2));
    nk_f64_t cross_10 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_10_low_f64x2, cross_10_high_f64x2));
    nk_f64_t cross_11 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_11_low_f64x2, cross_11_high_f64x2));
    nk_f64_t cross_12 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_12_low_f64x2, cross_12_high_f64x2));
    nk_f64_t cross_20 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_20_low_f64x2, cross_20_high_f64x2));
    nk_f64_t cross_21 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_21_low_f64x2, cross_21_high_f64x2));
    nk_f64_t cross_22 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_22_low_f64x2, cross_22_high_f64x2));
    nk_f64_t norm_squared_a_sum = nk_hsum_f64x2_v128relaxed_(
        wasm_f64x2_add(norm_squared_a_low_f64x2, norm_squared_a_high_f64x2));
    nk_f64_t norm_squared_b_sum = nk_hsum_f64x2_v128relaxed_(
        wasm_f64x2_add(norm_squared_b_low_f64x2, norm_squared_b_high_f64x2));

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        cross_00 += a_x * b_x, cross_01 += a_x * b_y, cross_02 += a_x * b_z;
        cross_10 += a_y * b_x, cross_11 += a_y * b_y, cross_12 += a_y * b_z;
        cross_20 += a_z * b_x, cross_21 += a_z * b_y, cross_22 += a_z * b_z;
        norm_squared_a_sum += a_x * a_x + a_y * a_y + a_z * a_z;
        norm_squared_b_sum += b_x * b_x + b_y * b_y + b_z * b_z;
    }

    nk_f64_t inv_points_count = 1.0 / (nk_f64_t)n;
    *centroid_a_x = sum_a_x * inv_points_count, *centroid_a_y = sum_a_y * inv_points_count,
    *centroid_a_z = sum_a_z * inv_points_count;
    *centroid_b_x = sum_b_x * inv_points_count, *centroid_b_y = sum_b_y * inv_points_count,
    *centroid_b_z = sum_b_z * inv_points_count;

    nk_f64_t n_f64 = (nk_f64_t)n;
    cross_covariance[0] = cross_00 - n_f64 * (*centroid_a_x) * (*centroid_b_x),
    cross_covariance[1] = cross_01 - n_f64 * (*centroid_a_x) * (*centroid_b_y),
    cross_covariance[2] = cross_02 - n_f64 * (*centroid_a_x) * (*centroid_b_z);
    cross_covariance[3] = cross_10 - n_f64 * (*centroid_a_y) * (*centroid_b_x),
    cross_covariance[4] = cross_11 - n_f64 * (*centroid_a_y) * (*centroid_b_y),
    cross_covariance[5] = cross_12 - n_f64 * (*centroid_a_y) * (*centroid_b_z);
    cross_covariance[6] = cross_20 - n_f64 * (*centroid_a_z) * (*centroid_b_x),
    cross_covariance[7] = cross_21 - n_f64 * (*centroid_a_z) * (*centroid_b_y),
    cross_covariance[8] = cross_22 - n_f64 * (*centroid_a_z) * (*centroid_b_z);

    *centered_norm_squared_a = norm_squared_a_sum -
                               n_f64 * ((*centroid_a_x) * (*centroid_a_x) + (*centroid_a_y) * (*centroid_a_y) +
                                        (*centroid_a_z) * (*centroid_a_z));
    *centered_norm_squared_b = norm_squared_b_sum -
                               n_f64 * ((*centroid_b_x) * (*centroid_b_x) + (*centroid_b_y) * (*centroid_b_y) +
                                        (*centroid_b_z) * (*centroid_b_z));
    if (*centered_norm_squared_a < 0.0) *centered_norm_squared_a = 0.0;
    if (*centered_norm_squared_b < 0.0) *centered_norm_squared_b = 0.0;
}

NK_INTERNAL void nk_centroid_and_cross_covariance_and_variance_f32_v128relaxed_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,                           //
    nk_f64_t *centroid_a_x, nk_f64_t *centroid_a_y, nk_f64_t *centroid_a_z,      //
    nk_f64_t *centroid_b_x, nk_f64_t *centroid_b_y, nk_f64_t *centroid_b_z,      //
    nk_f64_t cross_covariance[9], nk_f64_t *centered_norm_squared_a, nk_f64_t *centered_norm_squared_b) {
    v128_t zero_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_a_x_low_f64x2 = zero_f64x2, sum_a_x_high_f64x2 = zero_f64x2;
    v128_t sum_a_y_low_f64x2 = zero_f64x2, sum_a_y_high_f64x2 = zero_f64x2;
    v128_t sum_a_z_low_f64x2 = zero_f64x2, sum_a_z_high_f64x2 = zero_f64x2;
    v128_t sum_b_x_low_f64x2 = zero_f64x2, sum_b_x_high_f64x2 = zero_f64x2;
    v128_t sum_b_y_low_f64x2 = zero_f64x2, sum_b_y_high_f64x2 = zero_f64x2;
    v128_t sum_b_z_low_f64x2 = zero_f64x2, sum_b_z_high_f64x2 = zero_f64x2;
    v128_t norm_squared_b_low_f64x2 = zero_f64x2, norm_squared_b_high_f64x2 = zero_f64x2;
    v128_t cross_00_low_f64x2 = zero_f64x2, cross_00_high_f64x2 = zero_f64x2;
    v128_t cross_01_low_f64x2 = zero_f64x2, cross_01_high_f64x2 = zero_f64x2;
    v128_t cross_02_low_f64x2 = zero_f64x2, cross_02_high_f64x2 = zero_f64x2;
    v128_t cross_10_low_f64x2 = zero_f64x2, cross_10_high_f64x2 = zero_f64x2;
    v128_t cross_11_low_f64x2 = zero_f64x2, cross_11_high_f64x2 = zero_f64x2;
    v128_t cross_12_low_f64x2 = zero_f64x2, cross_12_high_f64x2 = zero_f64x2;
    v128_t cross_20_low_f64x2 = zero_f64x2, cross_20_high_f64x2 = zero_f64x2;
    v128_t cross_21_low_f64x2 = zero_f64x2, cross_21_high_f64x2 = zero_f64x2;
    v128_t cross_22_low_f64x2 = zero_f64x2, cross_22_high_f64x2 = zero_f64x2;
    v128_t sum_norm_squared_low_f64x2 = zero_f64x2, sum_norm_squared_high_f64x2 = zero_f64x2;
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_v128relaxed_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        v128_t a_x_low_f64x2 = wasm_f64x2_promote_low_f32x4(a_x_f32x4);
        v128_t a_x_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_x_f32x4, a_x_f32x4, 2, 3, 0, 1));
        v128_t a_y_low_f64x2 = wasm_f64x2_promote_low_f32x4(a_y_f32x4);
        v128_t a_y_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_y_f32x4, a_y_f32x4, 2, 3, 0, 1));
        v128_t a_z_low_f64x2 = wasm_f64x2_promote_low_f32x4(a_z_f32x4);
        v128_t a_z_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_z_f32x4, a_z_f32x4, 2, 3, 0, 1));
        v128_t b_x_low_f64x2 = wasm_f64x2_promote_low_f32x4(b_x_f32x4);
        v128_t b_x_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_x_f32x4, b_x_f32x4, 2, 3, 0, 1));
        v128_t b_y_low_f64x2 = wasm_f64x2_promote_low_f32x4(b_y_f32x4);
        v128_t b_y_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_y_f32x4, b_y_f32x4, 2, 3, 0, 1));
        v128_t b_z_low_f64x2 = wasm_f64x2_promote_low_f32x4(b_z_f32x4);
        v128_t b_z_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_z_f32x4, b_z_f32x4, 2, 3, 0, 1));

        sum_a_x_low_f64x2 = wasm_f64x2_add(sum_a_x_low_f64x2, a_x_low_f64x2),
        sum_a_x_high_f64x2 = wasm_f64x2_add(sum_a_x_high_f64x2, a_x_high_f64x2);
        sum_a_y_low_f64x2 = wasm_f64x2_add(sum_a_y_low_f64x2, a_y_low_f64x2),
        sum_a_y_high_f64x2 = wasm_f64x2_add(sum_a_y_high_f64x2, a_y_high_f64x2);
        sum_a_z_low_f64x2 = wasm_f64x2_add(sum_a_z_low_f64x2, a_z_low_f64x2),
        sum_a_z_high_f64x2 = wasm_f64x2_add(sum_a_z_high_f64x2, a_z_high_f64x2);
        sum_b_x_low_f64x2 = wasm_f64x2_add(sum_b_x_low_f64x2, b_x_low_f64x2),
        sum_b_x_high_f64x2 = wasm_f64x2_add(sum_b_x_high_f64x2, b_x_high_f64x2);
        sum_b_y_low_f64x2 = wasm_f64x2_add(sum_b_y_low_f64x2, b_y_low_f64x2),
        sum_b_y_high_f64x2 = wasm_f64x2_add(sum_b_y_high_f64x2, b_y_high_f64x2);
        sum_b_z_low_f64x2 = wasm_f64x2_add(sum_b_z_low_f64x2, b_z_low_f64x2),
        sum_b_z_high_f64x2 = wasm_f64x2_add(sum_b_z_high_f64x2, b_z_high_f64x2);

        cross_00_low_f64x2 = wasm_f64x2_relaxed_madd(a_x_low_f64x2, b_x_low_f64x2, cross_00_low_f64x2),
        cross_00_high_f64x2 = wasm_f64x2_relaxed_madd(a_x_high_f64x2, b_x_high_f64x2, cross_00_high_f64x2);
        cross_01_low_f64x2 = wasm_f64x2_relaxed_madd(a_x_low_f64x2, b_y_low_f64x2, cross_01_low_f64x2),
        cross_01_high_f64x2 = wasm_f64x2_relaxed_madd(a_x_high_f64x2, b_y_high_f64x2, cross_01_high_f64x2);
        cross_02_low_f64x2 = wasm_f64x2_relaxed_madd(a_x_low_f64x2, b_z_low_f64x2, cross_02_low_f64x2),
        cross_02_high_f64x2 = wasm_f64x2_relaxed_madd(a_x_high_f64x2, b_z_high_f64x2, cross_02_high_f64x2);
        cross_10_low_f64x2 = wasm_f64x2_relaxed_madd(a_y_low_f64x2, b_x_low_f64x2, cross_10_low_f64x2),
        cross_10_high_f64x2 = wasm_f64x2_relaxed_madd(a_y_high_f64x2, b_x_high_f64x2, cross_10_high_f64x2);
        cross_11_low_f64x2 = wasm_f64x2_relaxed_madd(a_y_low_f64x2, b_y_low_f64x2, cross_11_low_f64x2),
        cross_11_high_f64x2 = wasm_f64x2_relaxed_madd(a_y_high_f64x2, b_y_high_f64x2, cross_11_high_f64x2);
        cross_12_low_f64x2 = wasm_f64x2_relaxed_madd(a_y_low_f64x2, b_z_low_f64x2, cross_12_low_f64x2),
        cross_12_high_f64x2 = wasm_f64x2_relaxed_madd(a_y_high_f64x2, b_z_high_f64x2, cross_12_high_f64x2);
        cross_20_low_f64x2 = wasm_f64x2_relaxed_madd(a_z_low_f64x2, b_x_low_f64x2, cross_20_low_f64x2),
        cross_20_high_f64x2 = wasm_f64x2_relaxed_madd(a_z_high_f64x2, b_x_high_f64x2, cross_20_high_f64x2);
        cross_21_low_f64x2 = wasm_f64x2_relaxed_madd(a_z_low_f64x2, b_y_low_f64x2, cross_21_low_f64x2),
        cross_21_high_f64x2 = wasm_f64x2_relaxed_madd(a_z_high_f64x2, b_y_high_f64x2, cross_21_high_f64x2);
        cross_22_low_f64x2 = wasm_f64x2_relaxed_madd(a_z_low_f64x2, b_z_low_f64x2, cross_22_low_f64x2),
        cross_22_high_f64x2 = wasm_f64x2_relaxed_madd(a_z_high_f64x2, b_z_high_f64x2, cross_22_high_f64x2);

        // Norm-squared accumulators for both point sets (used for folded SSD).
        v128_t norm_squared_low_f64x2 = wasm_f64x2_relaxed_madd(a_y_low_f64x2, a_y_low_f64x2,
                                                                wasm_f64x2_mul(a_x_low_f64x2, a_x_low_f64x2));
        v128_t norm_squared_high_f64x2 = wasm_f64x2_relaxed_madd(a_y_high_f64x2, a_y_high_f64x2,
                                                                 wasm_f64x2_mul(a_x_high_f64x2, a_x_high_f64x2));
        norm_squared_low_f64x2 = wasm_f64x2_relaxed_madd(a_z_low_f64x2, a_z_low_f64x2, norm_squared_low_f64x2);
        norm_squared_high_f64x2 = wasm_f64x2_relaxed_madd(a_z_high_f64x2, a_z_high_f64x2, norm_squared_high_f64x2);
        sum_norm_squared_low_f64x2 = wasm_f64x2_add(sum_norm_squared_low_f64x2, norm_squared_low_f64x2);
        sum_norm_squared_high_f64x2 = wasm_f64x2_add(sum_norm_squared_high_f64x2, norm_squared_high_f64x2);

        norm_squared_b_low_f64x2 = wasm_f64x2_relaxed_madd(b_x_low_f64x2, b_x_low_f64x2, norm_squared_b_low_f64x2);
        norm_squared_b_high_f64x2 = wasm_f64x2_relaxed_madd(b_x_high_f64x2, b_x_high_f64x2, norm_squared_b_high_f64x2);
        norm_squared_b_low_f64x2 = wasm_f64x2_relaxed_madd(b_y_low_f64x2, b_y_low_f64x2, norm_squared_b_low_f64x2);
        norm_squared_b_high_f64x2 = wasm_f64x2_relaxed_madd(b_y_high_f64x2, b_y_high_f64x2, norm_squared_b_high_f64x2);
        norm_squared_b_low_f64x2 = wasm_f64x2_relaxed_madd(b_z_low_f64x2, b_z_low_f64x2, norm_squared_b_low_f64x2);
        norm_squared_b_high_f64x2 = wasm_f64x2_relaxed_madd(b_z_high_f64x2, b_z_high_f64x2, norm_squared_b_high_f64x2);
    }

    nk_f64_t sum_a_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_x_low_f64x2, sum_a_x_high_f64x2));
    nk_f64_t sum_a_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_y_low_f64x2, sum_a_y_high_f64x2));
    nk_f64_t sum_a_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_a_z_low_f64x2, sum_a_z_high_f64x2));
    nk_f64_t sum_b_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_x_low_f64x2, sum_b_x_high_f64x2));
    nk_f64_t sum_b_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_y_low_f64x2, sum_b_y_high_f64x2));
    nk_f64_t sum_b_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_b_z_low_f64x2, sum_b_z_high_f64x2));
    nk_f64_t cross_00 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_00_low_f64x2, cross_00_high_f64x2));
    nk_f64_t cross_01 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_01_low_f64x2, cross_01_high_f64x2));
    nk_f64_t cross_02 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_02_low_f64x2, cross_02_high_f64x2));
    nk_f64_t cross_10 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_10_low_f64x2, cross_10_high_f64x2));
    nk_f64_t cross_11 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_11_low_f64x2, cross_11_high_f64x2));
    nk_f64_t cross_12 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_12_low_f64x2, cross_12_high_f64x2));
    nk_f64_t cross_20 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_20_low_f64x2, cross_20_high_f64x2));
    nk_f64_t cross_21 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_21_low_f64x2, cross_21_high_f64x2));
    nk_f64_t cross_22 = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(cross_22_low_f64x2, cross_22_high_f64x2));
    nk_f64_t norm_squared_a_sum = nk_hsum_f64x2_v128relaxed_(
        wasm_f64x2_add(sum_norm_squared_low_f64x2, sum_norm_squared_high_f64x2));
    nk_f64_t norm_squared_b_sum = nk_hsum_f64x2_v128relaxed_(
        wasm_f64x2_add(norm_squared_b_low_f64x2, norm_squared_b_high_f64x2));

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        cross_00 += a_x * b_x, cross_01 += a_x * b_y, cross_02 += a_x * b_z;
        cross_10 += a_y * b_x, cross_11 += a_y * b_y, cross_12 += a_y * b_z;
        cross_20 += a_z * b_x, cross_21 += a_z * b_y, cross_22 += a_z * b_z;
        norm_squared_a_sum += a_x * a_x + a_y * a_y + a_z * a_z;
        norm_squared_b_sum += b_x * b_x + b_y * b_y + b_z * b_z;
    }

    nk_f64_t inv_points_count = 1.0 / (nk_f64_t)n;
    *centroid_a_x = sum_a_x * inv_points_count, *centroid_a_y = sum_a_y * inv_points_count,
    *centroid_a_z = sum_a_z * inv_points_count;
    *centroid_b_x = sum_b_x * inv_points_count, *centroid_b_y = sum_b_y * inv_points_count,
    *centroid_b_z = sum_b_z * inv_points_count;

    nk_f64_t n_f64 = (nk_f64_t)n;
    cross_covariance[0] = cross_00 - n_f64 * (*centroid_a_x) * (*centroid_b_x),
    cross_covariance[1] = cross_01 - n_f64 * (*centroid_a_x) * (*centroid_b_y),
    cross_covariance[2] = cross_02 - n_f64 * (*centroid_a_x) * (*centroid_b_z);
    cross_covariance[3] = cross_10 - n_f64 * (*centroid_a_y) * (*centroid_b_x),
    cross_covariance[4] = cross_11 - n_f64 * (*centroid_a_y) * (*centroid_b_y),
    cross_covariance[5] = cross_12 - n_f64 * (*centroid_a_y) * (*centroid_b_z);
    cross_covariance[6] = cross_20 - n_f64 * (*centroid_a_z) * (*centroid_b_x),
    cross_covariance[7] = cross_21 - n_f64 * (*centroid_a_z) * (*centroid_b_y),
    cross_covariance[8] = cross_22 - n_f64 * (*centroid_a_z) * (*centroid_b_z);

    *centered_norm_squared_a = norm_squared_a_sum -
                               n_f64 * ((*centroid_a_x) * (*centroid_a_x) + (*centroid_a_y) * (*centroid_a_y) +
                                        (*centroid_a_z) * (*centroid_a_z));
    *centered_norm_squared_b = norm_squared_b_sum -
                               n_f64 * ((*centroid_b_x) * (*centroid_b_x) + (*centroid_b_y) * (*centroid_b_y) +
                                        (*centroid_b_z) * (*centroid_b_z));
    if (*centered_norm_squared_a < 0.0) *centered_norm_squared_a = 0.0;
    if (*centered_norm_squared_b < 0.0) *centered_norm_squared_b = 0.0;
}

NK_PUBLIC void nk_rmsd_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    v128_t zero_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_sq_x_low_f64x2 = zero_f64x2, sum_sq_x_high_f64x2 = zero_f64x2;
    v128_t sum_sq_y_low_f64x2 = zero_f64x2, sum_sq_y_high_f64x2 = zero_f64x2;
    v128_t sum_sq_z_low_f64x2 = zero_f64x2, sum_sq_z_high_f64x2 = zero_f64x2;
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_v128relaxed_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        // Promote lower and upper halves to f64 for precision.
        v128_t a_x_low_f64x2 = wasm_f64x2_promote_low_f32x4(a_x_f32x4);
        v128_t a_x_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_x_f32x4, a_x_f32x4, 2, 3, 0, 1));
        v128_t a_y_low_f64x2 = wasm_f64x2_promote_low_f32x4(a_y_f32x4);
        v128_t a_y_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_y_f32x4, a_y_f32x4, 2, 3, 0, 1));
        v128_t a_z_low_f64x2 = wasm_f64x2_promote_low_f32x4(a_z_f32x4);
        v128_t a_z_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(a_z_f32x4, a_z_f32x4, 2, 3, 0, 1));
        v128_t b_x_low_f64x2 = wasm_f64x2_promote_low_f32x4(b_x_f32x4);
        v128_t b_x_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_x_f32x4, b_x_f32x4, 2, 3, 0, 1));
        v128_t b_y_low_f64x2 = wasm_f64x2_promote_low_f32x4(b_y_f32x4);
        v128_t b_y_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_y_f32x4, b_y_f32x4, 2, 3, 0, 1));
        v128_t b_z_low_f64x2 = wasm_f64x2_promote_low_f32x4(b_z_f32x4);
        v128_t b_z_high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(b_z_f32x4, b_z_f32x4, 2, 3, 0, 1));

        // Accumulate squared differences in f64.
        v128_t dx_low_f64x2 = wasm_f64x2_sub(a_x_low_f64x2, b_x_low_f64x2);
        v128_t dx_high_f64x2 = wasm_f64x2_sub(a_x_high_f64x2, b_x_high_f64x2);
        v128_t dy_low_f64x2 = wasm_f64x2_sub(a_y_low_f64x2, b_y_low_f64x2);
        v128_t dy_high_f64x2 = wasm_f64x2_sub(a_y_high_f64x2, b_y_high_f64x2);
        v128_t dz_low_f64x2 = wasm_f64x2_sub(a_z_low_f64x2, b_z_low_f64x2);
        v128_t dz_high_f64x2 = wasm_f64x2_sub(a_z_high_f64x2, b_z_high_f64x2);

        sum_sq_x_low_f64x2 = wasm_f64x2_relaxed_madd(dx_low_f64x2, dx_low_f64x2, sum_sq_x_low_f64x2);
        sum_sq_x_high_f64x2 = wasm_f64x2_relaxed_madd(dx_high_f64x2, dx_high_f64x2, sum_sq_x_high_f64x2);
        sum_sq_y_low_f64x2 = wasm_f64x2_relaxed_madd(dy_low_f64x2, dy_low_f64x2, sum_sq_y_low_f64x2);
        sum_sq_y_high_f64x2 = wasm_f64x2_relaxed_madd(dy_high_f64x2, dy_high_f64x2, sum_sq_y_high_f64x2);
        sum_sq_z_low_f64x2 = wasm_f64x2_relaxed_madd(dz_low_f64x2, dz_low_f64x2, sum_sq_z_low_f64x2);
        sum_sq_z_high_f64x2 = wasm_f64x2_relaxed_madd(dz_high_f64x2, dz_high_f64x2, sum_sq_z_high_f64x2);
    }

    nk_f64_t sum_sq_x = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_sq_x_low_f64x2, sum_sq_x_high_f64x2));
    nk_f64_t sum_sq_y = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_sq_y_low_f64x2, sum_sq_y_high_f64x2));
    nk_f64_t sum_sq_z = nk_hsum_f64x2_v128relaxed_(wasm_f64x2_add(sum_sq_z_low_f64x2, sum_sq_z_high_f64x2));

    // Scalar tail.
    for (; index < n; ++index) {
        nk_f64_t ax = a[index * 3 + 0], ay = a[index * 3 + 1], az = a[index * 3 + 2];
        nk_f64_t bx = b[index * 3 + 0], by = b[index * 3 + 1], bz = b[index * 3 + 2];
        nk_f64_t dx = ax - bx, dy = ay - by, dz = az - bz;
        sum_sq_x += dx * dx, sum_sq_y += dy * dy, sum_sq_z += dz * dz;
    }

    *result = nk_f64_sqrt_v128relaxed((sum_sq_x + sum_sq_y + sum_sq_z) / (nk_f64_t)n);
}

NK_PUBLIC void nk_rmsd_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                       nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Accumulators for squared differences
    v128_t sum_squared_x_f64x2 = zeros_f64x2, sum_squared_y_f64x2 = zeros_f64x2, sum_squared_z_f64x2 = zeros_f64x2;

    v128_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;
    nk_size_t i = 0;

    // Main loop processing 2 points at a time
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_v128relaxed_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_v128relaxed_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        v128_t delta_x_f64x2 = wasm_f64x2_sub(a_x_f64x2, b_x_f64x2);
        v128_t delta_y_f64x2 = wasm_f64x2_sub(a_y_f64x2, b_y_f64x2);
        v128_t delta_z_f64x2 = wasm_f64x2_sub(a_z_f64x2, b_z_f64x2);

        sum_squared_x_f64x2 = wasm_f64x2_relaxed_madd(delta_x_f64x2, delta_x_f64x2, sum_squared_x_f64x2);
        sum_squared_y_f64x2 = wasm_f64x2_relaxed_madd(delta_y_f64x2, delta_y_f64x2, sum_squared_y_f64x2);
        sum_squared_z_f64x2 = wasm_f64x2_relaxed_madd(delta_z_f64x2, delta_z_f64x2, sum_squared_z_f64x2);
    }

    // Reduce vectors to scalars.
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
        nk_f64_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        nk_accumulate_square_f64_(&total_squared_x, &total_squared_x_compensation, delta_x);
        nk_accumulate_square_f64_(&total_squared_y, &total_squared_y_compensation, delta_y);
        nk_accumulate_square_f64_(&total_squared_z, &total_squared_z_compensation, delta_z);
    }

    total_squared_x += total_squared_x_compensation, total_squared_y += total_squared_y_compensation,
        total_squared_z += total_squared_z_compensation;

    *result = nk_f64_sqrt_v128relaxed((total_squared_x + total_squared_y + total_squared_z) / (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z;
    nk_f64_t centered_norm_squared_a, centered_norm_squared_b;
    nk_f64_t cross_covariance[9];
    nk_centroid_and_cross_covariance_f32_v128relaxed_(a, b, n, &centroid_a_x, &centroid_a_y, &centroid_a_z,
                                                      &centroid_b_x, &centroid_b_y, &centroid_b_z, cross_covariance,
                                                      &centered_norm_squared_a, &centered_norm_squared_b);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    // Identity-dominant short-circuit: if H ≈ diag(positive entries), R = I and trace(R·H) = trace(H).
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

        // Handle reflection: if det(R) < 0, negate third column of V and recompute R.
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
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];
    if (scale) *scale = 1.0f;

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f64_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0 * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_v128relaxed(sum_squared / (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                         nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Single set of accumulators (1x unrolling)
    v128_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    v128_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;

    v128_t covariance_xx_f64x2 = zeros_f64x2, covariance_xy_f64x2 = zeros_f64x2, covariance_xz_f64x2 = zeros_f64x2;
    v128_t covariance_yx_f64x2 = zeros_f64x2, covariance_yy_f64x2 = zeros_f64x2, covariance_yz_f64x2 = zeros_f64x2;
    v128_t covariance_zx_f64x2 = zeros_f64x2, covariance_zy_f64x2 = zeros_f64x2, covariance_zz_f64x2 = zeros_f64x2;
    v128_t norm_squared_a_f64x2 = zeros_f64x2, norm_squared_b_f64x2 = zeros_f64x2;

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

        covariance_xx_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_x_f64x2, covariance_xx_f64x2);
        covariance_xy_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_y_f64x2, covariance_xy_f64x2);
        covariance_xz_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_z_f64x2, covariance_xz_f64x2);
        covariance_yx_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_x_f64x2, covariance_yx_f64x2);
        covariance_yy_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_y_f64x2, covariance_yy_f64x2);
        covariance_yz_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_z_f64x2, covariance_yz_f64x2);
        covariance_zx_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_x_f64x2, covariance_zx_f64x2);
        covariance_zy_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_y_f64x2, covariance_zy_f64x2);
        covariance_zz_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_z_f64x2, covariance_zz_f64x2);
        norm_squared_a_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, a_x_f64x2, norm_squared_a_f64x2);
        norm_squared_a_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, a_y_f64x2, norm_squared_a_f64x2);
        norm_squared_a_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, a_z_f64x2, norm_squared_a_f64x2);
        norm_squared_b_f64x2 = wasm_f64x2_relaxed_madd(b_x_f64x2, b_x_f64x2, norm_squared_b_f64x2);
        norm_squared_b_f64x2 = wasm_f64x2_relaxed_madd(b_y_f64x2, b_y_f64x2, norm_squared_b_f64x2);
        norm_squared_b_f64x2 = wasm_f64x2_relaxed_madd(b_z_f64x2, b_z_f64x2, norm_squared_b_f64x2);
    }

    // Reduce vector accumulators.
    nk_f64_t sum_a_x = nk_reduce_stable_f64x2_v128relaxed_(sum_a_x_f64x2), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x2_v128relaxed_(sum_a_y_f64x2), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x2_v128relaxed_(sum_a_z_f64x2), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x2_v128relaxed_(sum_b_x_f64x2), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x2_v128relaxed_(sum_b_y_f64x2), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x2_v128relaxed_(sum_b_z_f64x2), sum_b_z_compensation = 0.0;

    nk_f64_t covariance_x_x = nk_reduce_stable_f64x2_v128relaxed_(covariance_xx_f64x2),
             covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x2_v128relaxed_(covariance_xy_f64x2),
             covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x2_v128relaxed_(covariance_xz_f64x2),
             covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x2_v128relaxed_(covariance_yx_f64x2),
             covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x2_v128relaxed_(covariance_yy_f64x2),
             covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x2_v128relaxed_(covariance_yz_f64x2),
             covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x2_v128relaxed_(covariance_zx_f64x2),
             covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x2_v128relaxed_(covariance_zy_f64x2),
             covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x2_v128relaxed_(covariance_zz_f64x2),
             covariance_z_z_compensation = 0.0;
    nk_f64_t norm_squared_a_sum = nk_reduce_stable_f64x2_v128relaxed_(norm_squared_a_f64x2),
             norm_squared_a_compensation = 0.0;
    nk_f64_t norm_squared_b_sum = nk_reduce_stable_f64x2_v128relaxed_(norm_squared_b_f64x2),
             norm_squared_b_compensation = 0.0;

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
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ax),
            nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ay),
            nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, az);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bx),
            nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, by),
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
    nk_f64_t inv_points_count = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_points_count, centroid_a_y = sum_a_y * inv_points_count,
             centroid_a_z = sum_a_z * inv_points_count;
    nk_f64_t centroid_b_x = sum_b_x * inv_points_count, centroid_b_y = sum_b_y * inv_points_count,
             centroid_b_z = sum_b_z * inv_points_count;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Centered norm-squared via parallel-axis identity; clamp at zero for numeric safety.
    nk_f64_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f64_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f64_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;

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

    // Identity-dominant short-circuit: if H ≈ diag(positive entries), R = I and trace(R·H) = trace(H).
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
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);

        // Handle reflection: if det(R) < 0, negate third column of V and recompute R
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

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f64_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0 * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_v128relaxed(sum_squared * inv_points_count);
}

NK_PUBLIC void nk_umeyama_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                          nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z;
    nk_f64_t centered_norm_squared_a, centered_norm_squared_b;
    nk_f64_t cross_covariance[9];
    nk_centroid_and_cross_covariance_and_variance_f32_v128relaxed_( //
        a, b, n, &centroid_a_x, &centroid_a_y, &centroid_a_z,       //
        &centroid_b_x, &centroid_b_y, &centroid_b_z, cross_covariance, &centered_norm_squared_a,
        &centered_norm_squared_b);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    // Identity-dominant short-circuit: if H ≈ diag(positive entries), R = I and trace(R·H) = trace(H).
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t trace_rotation_covariance;
    nk_f64_t computed_scale;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        computed_scale = centered_norm_squared_a > 0.0 ? trace_rotation_covariance / centered_norm_squared_a : 0.0;
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

        nk_f64_t det = nk_det3x3_f64_(optimal_rotation);
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

        nk_f64_t trace_signed_singular_values = svd_diagonal[0] + svd_diagonal[4] +
                                                (det < 0 ? -svd_diagonal[8] : svd_diagonal[8]);
        computed_scale = centered_norm_squared_a > 0.0 ? trace_signed_singular_values / centered_norm_squared_a : 0.0;

        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];
    if (scale) *scale = (nk_f32_t)computed_scale;

    // Folded SSD with scale: c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f64_t sum_squared = computed_scale * computed_scale * centered_norm_squared_a + centered_norm_squared_b -
                           2.0 * computed_scale * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_v128relaxed(sum_squared / (nk_f64_t)n);
}

NK_PUBLIC void nk_umeyama_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                          nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Single set of accumulators (1x unrolling)
    v128_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    v128_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;

    v128_t covariance_xx_f64x2 = zeros_f64x2, covariance_xy_f64x2 = zeros_f64x2, covariance_xz_f64x2 = zeros_f64x2;
    v128_t covariance_yx_f64x2 = zeros_f64x2, covariance_yy_f64x2 = zeros_f64x2, covariance_yz_f64x2 = zeros_f64x2;
    v128_t covariance_zx_f64x2 = zeros_f64x2, covariance_zy_f64x2 = zeros_f64x2, covariance_zz_f64x2 = zeros_f64x2;
    v128_t norm_squared_a_f64x2 = zeros_f64x2, norm_squared_b_f64x2 = zeros_f64x2;

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

        covariance_xx_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_x_f64x2, covariance_xx_f64x2);
        covariance_xy_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_y_f64x2, covariance_xy_f64x2);
        covariance_xz_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_z_f64x2, covariance_xz_f64x2);
        covariance_yx_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_x_f64x2, covariance_yx_f64x2);
        covariance_yy_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_y_f64x2, covariance_yy_f64x2);
        covariance_yz_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_z_f64x2, covariance_yz_f64x2);
        covariance_zx_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_x_f64x2, covariance_zx_f64x2);
        covariance_zy_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_y_f64x2, covariance_zy_f64x2);
        covariance_zz_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_z_f64x2, covariance_zz_f64x2);

        norm_squared_a_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, a_x_f64x2, norm_squared_a_f64x2);
        norm_squared_a_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, a_y_f64x2, norm_squared_a_f64x2);
        norm_squared_a_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, a_z_f64x2, norm_squared_a_f64x2);
        norm_squared_b_f64x2 = wasm_f64x2_relaxed_madd(b_x_f64x2, b_x_f64x2, norm_squared_b_f64x2);
        norm_squared_b_f64x2 = wasm_f64x2_relaxed_madd(b_y_f64x2, b_y_f64x2, norm_squared_b_f64x2);
        norm_squared_b_f64x2 = wasm_f64x2_relaxed_madd(b_z_f64x2, b_z_f64x2, norm_squared_b_f64x2);
    }

    // Reduce vector accumulators.
    nk_f64_t sum_a_x = nk_reduce_stable_f64x2_v128relaxed_(sum_a_x_f64x2), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x2_v128relaxed_(sum_a_y_f64x2), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x2_v128relaxed_(sum_a_z_f64x2), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x2_v128relaxed_(sum_b_x_f64x2), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x2_v128relaxed_(sum_b_y_f64x2), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x2_v128relaxed_(sum_b_z_f64x2), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x2_v128relaxed_(covariance_xx_f64x2),
             covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x2_v128relaxed_(covariance_xy_f64x2),
             covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x2_v128relaxed_(covariance_xz_f64x2),
             covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x2_v128relaxed_(covariance_yx_f64x2),
             covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x2_v128relaxed_(covariance_yy_f64x2),
             covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x2_v128relaxed_(covariance_yz_f64x2),
             covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x2_v128relaxed_(covariance_zx_f64x2),
             covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x2_v128relaxed_(covariance_zy_f64x2),
             covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x2_v128relaxed_(covariance_zz_f64x2),
             covariance_z_z_compensation = 0.0;
    nk_f64_t norm_squared_a_sum = nk_reduce_stable_f64x2_v128relaxed_(norm_squared_a_f64x2),
             norm_squared_a_compensation = 0.0;
    nk_f64_t norm_squared_b_sum = nk_reduce_stable_f64x2_v128relaxed_(norm_squared_b_f64x2),
             norm_squared_b_compensation = 0.0;

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
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ax),
            nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ay),
            nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, az);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bx),
            nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, by),
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
    nk_f64_t inv_points_count = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_points_count, centroid_a_y = sum_a_y * inv_points_count,
             centroid_a_z = sum_a_z * inv_points_count;
    nk_f64_t centroid_b_x = sum_b_x * inv_points_count, centroid_b_y = sum_b_y * inv_points_count,
             centroid_b_z = sum_b_z * inv_points_count;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Centered norm-squared via parallel-axis identity; clamp at zero for numeric safety.
    nk_f64_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f64_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f64_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;

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

    // Identity-dominant short-circuit: if H ≈ diag(positive entries), R = I and trace(R·H) = trace(H).
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t trace_rotation_covariance;
    nk_f64_t computed_scale;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        computed_scale = centered_norm_squared_a > 0.0 ? trace_rotation_covariance / centered_norm_squared_a : 0.0;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);

        // Handle reflection and compute scale
        nk_f64_t det = nk_det3x3_f64_(optimal_rotation);
        nk_f64_t trace_d_s = svd_diagonal[0] + svd_diagonal[4] + (det < 0 ? -svd_diagonal[8] : svd_diagonal[8]);
        computed_scale = centered_norm_squared_a > 0.0 ? trace_d_s / centered_norm_squared_a : 0.0;

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

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = computed_scale;

    // Folded SSD with scale: c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f64_t sum_squared = computed_scale * computed_scale * centered_norm_squared_a + centered_norm_squared_b -
                           2.0 * computed_scale * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_v128relaxed(sum_squared * inv_points_count);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_MESH_V128RELAXED_H
