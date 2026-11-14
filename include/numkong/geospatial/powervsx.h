/**
 *  @file include/numkong/geospatial/powervsx.h
 *  @author Ash Vardanian
 *  @date November 13, 2026
 *  @brief SIMD-accelerated geospatial distances for Power VSX.
 *
 *  @sa c/numkong/geospatial.h
 *
 *  Every kernel broadcasts one row and sweeps the columns two or four at a time. The 3D squares
 *  load three registers of @b [x,y,z] rows and gather each axis with @c vec_perm, two steps for
 *  @c i32 and one for @c f64. The @c i32 differences square exactly into 64-bit lanes with
 *  @c vec_mule and @c vec_mulo, and the @c f64 ones sum in Dot2. Tails load and store with
 *  @c vec_xl_len and @c vec_xst_len.
 *
 *  @section geospatial_powervsx_instructions Key Power VSX Geospatial Instructions
 *
 *  @verbatim
 *  Intrinsic     Instruction   Operation
 *  vec_perm      VPERM         3D de-interleave, two sources per step
 *  vec_mule      VMULESW       Exact squares of even 32-bit lanes into 64-bit lanes
 *  vec_mulo      VMULOSW       Exact squares of odd 32-bit lanes into 64-bit lanes
 *  vec_msub      XVMSUBADP     Product error of the Dot2 step
 *  vec_xl_len    LXVL          Tail loads, zero-filling the rest
 *  @endverbatim
 */
#ifndef NUMKONG_GEOSPATIAL_POWERVSX_H
#define NUMKONG_GEOSPATIAL_POWERVSX_H

#if NUMKONG_ARCH_PPC64_
#if NUMKONG_ARCH_PPC64_POWERVSX_

#include "numkong/types.h"
#include "numkong/cast/powervsx.h" // `nk_partial_load_b32x4_powervsx_`, `nk_partial_store_b64x2_powervsx_`
#include "numkong/dot/powervsx.h"  // `nk_dot2_f64x2_powervsx_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/** Exact squared distances from a broadcast 3D point to 4 interleaved @b [x,y,z] points in three
 *  registers, the first two sums in @p sums_low_u64x2 and the last two in @p sums_high_u64x2. */
NUMKONG_INLINE void nk_sqeuclidean3d_i32x4_powervsx_(                                 //
    nk_vi32x4_t a_x_i32x4, nk_vi32x4_t a_y_i32x4, nk_vi32x4_t a_z_i32x4,              //
    nk_vi32x4_t b_first_i32x4, nk_vi32x4_t b_second_i32x4, nk_vi32x4_t b_third_i32x4, //
    nk_vu64x2_t *sums_low_u64x2, nk_vu64x2_t *sums_high_u64x2) {
    // From [x0 y0 z0 x1], [y1 z1 x2 y2] and [z2 x3 y3 z3], each axis in lanes of points 0, 2, 1, 3,
    // so the even products are the first two sums and the odd ones the last two
    nk_vu8x16_t const x_front_u8x16 = {0, 1, 2, 3, 24, 25, 26, 27, 12, 13, 14, 15, 12, 13, 14, 15};
    nk_vu8x16_t const y_front_u8x16 = {4, 5, 6, 7, 28, 29, 30, 31, 16, 17, 18, 19, 16, 17, 18, 19};
    nk_vu8x16_t const z_front_u8x16 = {8, 9, 10, 11, 20, 21, 22, 23, 20, 21, 22, 23, 20, 21, 22, 23};
    nk_vu8x16_t const x_back_u8x16 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23};
    nk_vu8x16_t const y_back_u8x16 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 24, 25, 26, 27};
    nk_vu8x16_t const z_back_u8x16 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 28, 29, 30, 31};
    nk_vi32x4_t const b_x_i32x4 = vec_perm(vec_perm(b_first_i32x4, b_second_i32x4, x_front_u8x16), b_third_i32x4,
                                           x_back_u8x16);
    nk_vi32x4_t const b_y_i32x4 = vec_perm(vec_perm(b_first_i32x4, b_second_i32x4, y_front_u8x16), b_third_i32x4,
                                           y_back_u8x16);
    nk_vi32x4_t const b_z_i32x4 = vec_perm(vec_perm(b_first_i32x4, b_second_i32x4, z_front_u8x16), b_third_i32x4,
                                           z_back_u8x16);
    nk_vi32x4_t const difference_x_i32x4 = vec_sub(a_x_i32x4, b_x_i32x4);
    nk_vi32x4_t const difference_y_i32x4 = vec_sub(a_y_i32x4, b_y_i32x4);
    nk_vi32x4_t const difference_z_i32x4 = vec_sub(a_z_i32x4, b_z_i32x4);
    *sums_low_u64x2 = vec_add(vec_add((nk_vu64x2_t)vec_mule(difference_x_i32x4, difference_x_i32x4),
                                      (nk_vu64x2_t)vec_mule(difference_y_i32x4, difference_y_i32x4)),
                              (nk_vu64x2_t)vec_mule(difference_z_i32x4, difference_z_i32x4));
    *sums_high_u64x2 = vec_add(vec_add((nk_vu64x2_t)vec_mulo(difference_x_i32x4, difference_x_i32x4),
                                       (nk_vu64x2_t)vec_mulo(difference_y_i32x4, difference_y_i32x4)),
                               (nk_vu64x2_t)vec_mulo(difference_z_i32x4, difference_z_i32x4));
}

/** Squared distances from a broadcast 3D point to 2 interleaved @b [x,y,z] points in three
 *  registers, the three squares summed in Dot2. */
NUMKONG_INLINE nk_vf64x2_t nk_sqeuclidean3d_f64x2_powervsx_(             //
    nk_vf64x2_t a_x_f64x2, nk_vf64x2_t a_y_f64x2, nk_vf64x2_t a_z_f64x2, //
    nk_vf64x2_t b_first_f64x2, nk_vf64x2_t b_second_f64x2, nk_vf64x2_t b_third_f64x2) {
    // From [x0 y0], [z0 x1] and [y1 z1]
    nk_vu8x16_t const first_and_last_u8x16 = {0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31};
    nk_vu8x16_t const last_and_first_u8x16 = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    nk_vf64x2_t const difference_x_f64x2 = vec_sub(a_x_f64x2,
                                                   vec_perm(b_first_f64x2, b_second_f64x2, first_and_last_u8x16));
    nk_vf64x2_t const difference_y_f64x2 = vec_sub(a_y_f64x2,
                                                   vec_perm(b_first_f64x2, b_third_f64x2, last_and_first_u8x16));
    nk_vf64x2_t const difference_z_f64x2 = vec_sub(a_z_f64x2,
                                                   vec_perm(b_second_f64x2, b_third_f64x2, first_and_last_u8x16));
    nk_vf64x2_t sum_f64x2 = vec_splats(0.0), compensation_f64x2 = vec_splats(0.0);
    nk_dot2_f64x2_powervsx_(&sum_f64x2, &compensation_f64x2, difference_x_f64x2, difference_x_f64x2);
    nk_dot2_f64x2_powervsx_(&sum_f64x2, &compensation_f64x2, difference_y_f64x2, difference_y_f64x2);
    nk_dot2_f64x2_powervsx_(&sum_f64x2, &compensation_f64x2, difference_z_f64x2, difference_z_f64x2);
    return vec_add(sum_f64x2, compensation_f64x2);
}

#if NUMKONG_TARGET_POWERVSX

NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_powervsx(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                       nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_u64_t *output = (nk_u64_t *)((char *)results + row * results_stride);
        nk_vi32x4_t const a_x_i32x4 = vec_splats(a[row * 3]);
        nk_vi32x4_t const a_y_i32x4 = vec_splats(a[row * 3 + 1]);
        nk_vi32x4_t const a_z_i32x4 = vec_splats(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 4 <= columns; column += 4) {
            nk_i32_t const *b_point = b + column * 3;
            nk_vu64x2_t sums_low_u64x2, sums_high_u64x2;
            nk_sqeuclidean3d_i32x4_powervsx_(a_x_i32x4, a_y_i32x4, a_z_i32x4, vec_xl(0, b_point), vec_xl(16, b_point),
                                             vec_xl(32, b_point), &sums_low_u64x2, &sums_high_u64x2);
            vec_xst((nk_vu8x16_t)sums_low_u64x2, 0, (nk_u8_t *)(output + column));
            vec_xst((nk_vu8x16_t)sums_high_u64x2, 0, (nk_u8_t *)(output + column + 2));
        }
        if (column < columns) {
            // `lxvl` and `stxvl` move at most 16 bytes, however many the count asks for
            nk_size_t const remaining = columns - column, values = remaining * 3;
            nk_b128_vec_t b_first_vec, b_second_vec, b_third_vec, sums_low_vec, sums_high_vec;
            nk_partial_load_b32x4_powervsx_(b + column * 3, &b_first_vec, values);
            nk_partial_load_b32x4_powervsx_(b + column * 3 + 4, &b_second_vec, values > 4 ? values - 4 : 0);
            nk_partial_load_b32x4_powervsx_(b + column * 3 + 8, &b_third_vec, values > 8 ? values - 8 : 0);
            nk_sqeuclidean3d_i32x4_powervsx_(a_x_i32x4, a_y_i32x4, a_z_i32x4, b_first_vec.vi32x4, b_second_vec.vi32x4,
                                             b_third_vec.vi32x4, &sums_low_vec.vu64x2, &sums_high_vec.vu64x2);
            nk_partial_store_b64x2_powervsx_(&sums_low_vec, output + column, remaining);
            nk_partial_store_b64x2_powervsx_(&sums_high_vec, output + column + 2, remaining > 2 ? remaining - 2 : 0);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_powervsx(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                       nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        nk_vf64x2_t const a_x_f64x2 = vec_splats(a[row * 3]);
        nk_vf64x2_t const a_y_f64x2 = vec_splats(a[row * 3 + 1]);
        nk_vf64x2_t const a_z_f64x2 = vec_splats(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 2 <= columns; column += 2) {
            nk_f64_t const *b_point = b + column * 3;
            vec_xst(nk_sqeuclidean3d_f64x2_powervsx_(a_x_f64x2, a_y_f64x2, a_z_f64x2, vec_xl(0, b_point),
                                                     vec_xl(16, b_point), vec_xl(32, b_point)),
                    0, output + column);
        }
        if (column < columns) {
            nk_b128_vec_t b_first_vec, b_second_vec, distances_vec;
            nk_load_b128_powervsx_(b + column * 3, &b_first_vec);
            nk_partial_load_b64x2_powervsx_(b + column * 3 + 2, &b_second_vec, 1);
            distances_vec.vf64x2 = nk_sqeuclidean3d_f64x2_powervsx_(a_x_f64x2, a_y_f64x2, a_z_f64x2, b_first_vec.vf64x2,
                                                                    b_second_vec.vf64x2, vec_splats(0.0));
            nk_partial_store_b64x2_powervsx_(&distances_vec, output + column, 1);
        }
    }
    return nk_success_k;
}

#endif // NUMKONG_TARGET_POWERVSX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_PPC64_POWERVSX_
#endif // NUMKONG_ARCH_PPC64_
#endif // NUMKONG_GEOSPATIAL_POWERVSX_H
