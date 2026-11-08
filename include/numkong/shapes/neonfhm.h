/**
 *  @file include/numkong/shapes/neonfhm.h
 *  @author Ash Vardanian
 *  @date April 15, 2026
 *  @brief Many-to-many shape distances between 3D point clouds for NEON FHM, with widening FMA.
 *
 *  @sa c/numkong/shapes.h
 *
 *  @section shapes_neonfhm_instructions ARM NEON FP16 FHM Shape Instructions (ARMv8.4-FHM)
 *
 *  @verbatim
 *  Intrinsic         Instruction                 A76        M5
 *  vfmlslq_low_f16   FMLSL (V.4S, V.8H, V.8H)    4cy @ 2p   4cy @ 4p
 *  vfmlslq_high_f16  FMLSL2 (V.4S, V.8H, V.8H)   4cy @ 2p   4cy @ 4p
 *  vfmlalq_low_f16   FMLAL (V.4S, V.8H, V.8H)    4cy @ 2p   4cy @ 4p
 *  vfmlalq_high_f16  FMLAL2 (V.4S, V.8H, V.8H)   4cy @ 2p   4cy @ 4p
 *  vmulq_f32         FMUL (V.4S, V.4S, V.4S)     3cy @ 2p   3cy @ 4p
 *  vfmaq_f32         FMLA (V.4S, V.4S, V.4S)     4cy @ 2p   3cy @ 4p
 *  vminq_f32         FMIN (V.4S, V.4S, V.4S)     2cy @ 2p   2cy @ 4p
 *  vpminq_f32        FMINP (V.4S, V.4S, V.4S)    2cy @ 2p   2cy @ 4p
 *  vbslq_u16         BSL (V.16B, V.16B, V.16B)   1cy @ 2p   2cy @ 4p
 *  @endverbatim
 *
 *  Chamfer and Hausdorff of @c f16 packs broadcast one outer point into f32 lanes and sweep the
 *  inner cloud eight points at a time. FMLSL and FMLAL subtract and add each inner coordinate times
 *  1, an exact product, onto the outer one with a single rounding into f32, as FCVTL and FSUB round
 *  a − b in the NEON kernel, but without the FCVTL. The squares and the flips then run in the same
 *  f32 lanes as the NEON kernel, on twice the points per load.
 *
 *  The differences and sums are not taken in f16, where they would round by up to 2⁻¹¹ of
 *  themselves and overflow past 65504, missing the f32 error bounds. @c nk_sqeuclidean_f16_neon and
 *  @c nk_mahalanobis_f16_neon widen before subtracting for the same reason. Padded lanes take an
 *  infinite x, so they never win a minimum, and a last octuple of four points or fewer skips its
 *  high half, wasting no more lanes than the NEON kernel's quadruples.
 *
 *  RMSD sums each of its 9 products over eight points per step, FMLAL and FMLAL2 taking one half of
 *  a register each, through the zero padding. The fold and the SVD stay scalar.
 */
#ifndef NUMKONG_SHAPES_NEONFHM_H
#define NUMKONG_SHAPES_NEONFHM_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/shapes/serial.h" // `nk_shapes_best_flip_f32_serial_`, `nk_shapes_mutual_f16_t`
#include "numkong/shapes/neon.h"   // `nk_shapes_lower_nearest_differences_f32_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

/** Lowers the running minima of the 8 flips by one outer point's squared distances to the first
 *  four of eight inner points, each axis' difference and sum formed by FMLSL and FMLAL by 1. */
NUMKONG_INLINE void nk_shapes_lower_nearest_low_f16_neonfhm_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                             float32x4_t outer_z_f32x4, float16x8_t inner_x_f16x8,
                                                             float16x8_t inner_y_f16x8, float16x8_t inner_z_f16x8,
                                                             float32x4_t *nearest_f32x4) {
    float16x8_t const ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    float32x4_t const minus_x_low_f32x4 = vfmlslq_low_f16(outer_x_f32x4, inner_x_f16x8, ones_f16x8);
    float32x4_t const plus_x_low_f32x4 = vfmlalq_low_f16(outer_x_f32x4, inner_x_f16x8, ones_f16x8);
    float32x4_t const minus_y_low_f32x4 = vfmlslq_low_f16(outer_y_f32x4, inner_y_f16x8, ones_f16x8);
    float32x4_t const plus_y_low_f32x4 = vfmlalq_low_f16(outer_y_f32x4, inner_y_f16x8, ones_f16x8);
    float32x4_t const minus_z_low_f32x4 = vfmlslq_low_f16(outer_z_f32x4, inner_z_f16x8, ones_f16x8);
    float32x4_t const plus_z_low_f32x4 = vfmlalq_low_f16(outer_z_f32x4, inner_z_f16x8, ones_f16x8);
    nk_shapes_lower_nearest_differences_f32_neon_(minus_x_low_f32x4, plus_x_low_f32x4, minus_y_low_f32x4,
                                                  plus_y_low_f32x4, minus_z_low_f32x4, plus_z_low_f32x4, nearest_f32x4);
}

/** Lowers the running minima of the 8 flips as @ref nk_shapes_lower_nearest_low_f16_neonfhm_ does,
 *  by the last four of the eight inner points. */
NUMKONG_INLINE void nk_shapes_lower_nearest_high_f16_neonfhm_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                              float32x4_t outer_z_f32x4, float16x8_t inner_x_f16x8,
                                                              float16x8_t inner_y_f16x8, float16x8_t inner_z_f16x8,
                                                              float32x4_t *nearest_f32x4) {
    float16x8_t const ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    float32x4_t const minus_x_high_f32x4 = vfmlslq_high_f16(outer_x_f32x4, inner_x_f16x8, ones_f16x8);
    float32x4_t const plus_x_high_f32x4 = vfmlalq_high_f16(outer_x_f32x4, inner_x_f16x8, ones_f16x8);
    float32x4_t const minus_y_high_f32x4 = vfmlslq_high_f16(outer_y_f32x4, inner_y_f16x8, ones_f16x8);
    float32x4_t const plus_y_high_f32x4 = vfmlalq_high_f16(outer_y_f32x4, inner_y_f16x8, ones_f16x8);
    float32x4_t const minus_z_high_f32x4 = vfmlslq_high_f16(outer_z_f32x4, inner_z_f16x8, ones_f16x8);
    float32x4_t const plus_z_high_f32x4 = vfmlalq_high_f16(outer_z_f32x4, inner_z_f16x8, ones_f16x8);
    nk_shapes_lower_nearest_differences_f32_neon_(minus_x_high_f32x4, plus_x_high_f32x4, minus_y_high_f32x4,
                                                  plus_y_high_f32x4, minus_z_high_f32x4, plus_z_high_f32x4,
                                                  nearest_f32x4);
}

/** Lowers the 8 flips' nearest squares and the inner points' @p reverse_minima by the first four of
 *  eight inner points, each axis' difference and sum formed by FMLSL and FMLAL by 1. */
NUMKONG_INLINE void nk_shapes_lower_mutual_low_f16_neonfhm_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                            float32x4_t outer_z_f32x4, float16x8_t inner_x_f16x8,
                                                            float16x8_t inner_y_f16x8, float16x8_t inner_z_f16x8,
                                                            float32x4_t *nearest_f32x4, nk_f32_t *reverse_minima,
                                                            nk_size_t reverse_stride) {
    float16x8_t const ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    nk_shapes_lower_mutual_differences_f32_neon_(vfmlslq_low_f16(outer_x_f32x4, inner_x_f16x8, ones_f16x8),
                                                 vfmlalq_low_f16(outer_x_f32x4, inner_x_f16x8, ones_f16x8),
                                                 vfmlslq_low_f16(outer_y_f32x4, inner_y_f16x8, ones_f16x8),
                                                 vfmlalq_low_f16(outer_y_f32x4, inner_y_f16x8, ones_f16x8),
                                                 vfmlslq_low_f16(outer_z_f32x4, inner_z_f16x8, ones_f16x8),
                                                 vfmlalq_low_f16(outer_z_f32x4, inner_z_f16x8, ones_f16x8),
                                                 nearest_f32x4, reverse_minima, reverse_stride);
}

/** @ref nk_shapes_lower_mutual_low_f16_neonfhm_ by the last four of the eight inner points. */
NUMKONG_INLINE void nk_shapes_lower_mutual_high_f16_neonfhm_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                             float32x4_t outer_z_f32x4, float16x8_t inner_x_f16x8,
                                                             float16x8_t inner_y_f16x8, float16x8_t inner_z_f16x8,
                                                             float32x4_t *nearest_f32x4, nk_f32_t *reverse_minima,
                                                             nk_size_t reverse_stride) {
    float16x8_t const ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    nk_shapes_lower_mutual_differences_f32_neon_(vfmlslq_high_f16(outer_x_f32x4, inner_x_f16x8, ones_f16x8),
                                                 vfmlalq_high_f16(outer_x_f32x4, inner_x_f16x8, ones_f16x8),
                                                 vfmlslq_high_f16(outer_y_f32x4, inner_y_f16x8, ones_f16x8),
                                                 vfmlalq_high_f16(outer_y_f32x4, inner_y_f16x8, ones_f16x8),
                                                 vfmlslq_high_f16(outer_z_f32x4, inner_z_f16x8, ones_f16x8),
                                                 vfmlalq_high_f16(outer_z_f32x4, inner_z_f16x8, ones_f16x8),
                                                 nearest_f32x4, reverse_minima + 4, reverse_stride);
}

/** One outer point's sweep of @c f16 inner points in f32 lanes, eight per load. A last octuple off
 *  the multiple takes an infinite x in its padded lanes and skips a high half with no points. */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f16_neonfhm_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                         float32x4_t outer_z_f32x4, nk_f16_t const *inner_axes,
                                                         nk_size_t inner_stride, nk_size_t inner_count,
                                                         nk_b128_vec_t const *tail_vec, float32x4_t *nearest_f32x4) {
    nk_u16_t const *inner_bits = (nk_u16_t const *)inner_axes;
    nk_size_t const octuples_end = inner_count & ~(nk_size_t)7;
    for (nk_size_t inner_index = 0; inner_index < inner_count; inner_index += 8) {
        uint16x8_t const x_bits_u16x8 = vld1q_u16(inner_bits + inner_index);
        float16x8_t const inner_x_f16x8 = vreinterpretq_f16_u16(
            inner_index == octuples_end ? vbslq_u16(tail_vec->u16x8, x_bits_u16x8, vdupq_n_u16(0x7C00)) : x_bits_u16x8);
        float16x8_t const inner_y_f16x8 = vreinterpretq_f16_u16(vld1q_u16(inner_bits + inner_stride + inner_index));
        float16x8_t const inner_z_f16x8 = vreinterpretq_f16_u16(vld1q_u16(inner_bits + 2 * inner_stride + inner_index));
        nk_shapes_lower_nearest_low_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f16x8,
                                                 inner_y_f16x8, inner_z_f16x8, nearest_f32x4);
        if (inner_index + 4 < inner_count)
            nk_shapes_lower_nearest_high_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f16x8,
                                                      inner_y_f16x8, inner_z_f16x8, nearest_f32x4);
    }
}

/** @ref nk_shapes_sweep_nearest_f16_neonfhm_ that also lowers the views' reverse minima. */
NUMKONG_INLINE void nk_shapes_sweep_mutual_f16_neonfhm_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                        float32x4_t outer_z_f32x4, nk_shapes_mutual_f16_t const *views,
                                                        nk_b128_vec_t const *tail_vec, float32x4_t *nearest_f32x4) {
    nk_u16_t const *inner_bits = (nk_u16_t const *)views->inner_axes;
    nk_size_t const inner_stride = views->inner_stride, inner_count = views->inner_count;
    nk_size_t const octuples_end = inner_count & ~(nk_size_t)7;
    for (nk_size_t inner_index = 0; inner_index < inner_count; inner_index += 8) {
        uint16x8_t const x_bits_u16x8 = vld1q_u16(inner_bits + inner_index);
        float16x8_t const inner_x_f16x8 = vreinterpretq_f16_u16(
            inner_index == octuples_end ? vbslq_u16(tail_vec->u16x8, x_bits_u16x8, vdupq_n_u16(0x7C00)) : x_bits_u16x8);
        float16x8_t const inner_y_f16x8 = vreinterpretq_f16_u16(vld1q_u16(inner_bits + inner_stride + inner_index));
        float16x8_t const inner_z_f16x8 = vreinterpretq_f16_u16(vld1q_u16(inner_bits + 2 * inner_stride + inner_index));
        nk_shapes_lower_mutual_low_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f16x8,
                                                inner_y_f16x8, inner_z_f16x8, nearest_f32x4,
                                                views->reverse_minima + inner_index, views->reverse_stride);
        if (inner_index + 4 < inner_count)
            nk_shapes_lower_mutual_high_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f16x8,
                                                     inner_y_f16x8, inner_z_f16x8, nearest_f32x4,
                                                     views->reverse_minima + inner_index, views->reverse_stride);
    }
}

/** Fused Chamfer flips of @c f16 packs, eight inner points per load, reductions in lanes. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f16_neonfhm_(void const *row_packed, void const *column_packed,
                                                                nk_f32_t limit, void *workspace,
                                                                nk_size_t workspace_bytes, nk_f32_t *chamfers) {
    nk_shapes_mutual_f16_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f16_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    float32x4_t const infinity_f32x4 = vdupq_n_f32(NUMKONG_F32_INF);
    for (nk_size_t index = 0; index < 8 * views.reverse_stride; index += 4)
        vst1q_f32(views.reverse_minima + index, infinity_f32x4);
    nk_b128_vec_t const tail_vec = nk_shapes_tail_neon_(views.inner_count, 8);
    nk_u16_t const *outer_bits = (nk_u16_t const *)views.outer_axes;
    nk_f32_t const outer_points = (nk_f32_t)views.outer_count;
    float32x4_t const threshold_f32x4 = vdupq_n_f32(2 * limit * outer_points);
    float32x4_t sums_f32x4[2] = {vdupq_n_f32(0), vdupq_n_f32(0)}, nearest_f32x4[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        float32x4_t const outer_x_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_index)));
        float32x4_t const outer_y_f32x4 = vcvt_f32_f16(
            vreinterpret_f16_u16(vld1_dup_u16(outer_bits + views.outer_stride + outer_index)));
        float32x4_t const outer_z_f32x4 = vcvt_f32_f16(
            vreinterpret_f16_u16(vld1_dup_u16(outer_bits + 2 * views.outer_stride + outer_index)));
        for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
        nk_shapes_sweep_mutual_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, &views, &tail_vec,
                                            nearest_f32x4);
        if (nk_shapes_chamfer_row_f32_neon_(nearest_f32x4, sums_f32x4, threshold_f32x4, outer_points, limit, chamfers))
            return nk_success_k;
    }
    nk_shapes_chamfer_finish_f16_neon_(sums_f32x4, &views, chamfers);
    return nk_success_k;
}

/** Hausdorff flips of @c f16 packs, eight inner points per load, stopping once every flip's
 *  farthest square passes @p limit . */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f16_neonfhm_(void const *row_packed, void const *column_packed,
                                                           nk_f32_t limit, nk_f32_t *hausdorffs) {
    nk_size_t const rows = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    nk_size_t const columns = ((nk_shapes_packed_header_f32_t const *)column_packed)->points;
    if (!rows || !columns) {
        for (int flip = 0; flip != 8; ++flip) hausdorffs[flip] = rows == columns ? 0 : NUMKONG_F32_INF;
        return;
    }
    nk_f32_t const gap = nk_shapes_radius_gap_f32_serial_(row_packed, column_packed);
    if (gap > limit) {
        for (int flip = 0; flip != 8; ++flip) hausdorffs[flip] = gap;
        return;
    }
    nk_size_t row_stride, column_stride;
    nk_f16_t const *row_axes = nk_shapes_packed_axes_f16_serial_(row_packed, &row_stride);
    nk_f16_t const *column_axes = nk_shapes_packed_axes_f16_serial_(column_packed, &column_stride);
    nk_b128_vec_t const row_tail_vec = nk_shapes_tail_neon_(rows, 8);
    nk_b128_vec_t const column_tail_vec = nk_shapes_tail_neon_(columns, 8);
    nk_f32_t const limit_squared = limit * limit;
    float32x4_t const infinity_f32x4 = vdupq_n_f32(NUMKONG_F32_INF);
    float32x4_t farthest_f32x4[2] = {vdupq_n_f32(0), vdupq_n_f32(0)}, nearest_f32x4[8];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_u16_t const *outer_bits = (nk_u16_t const *)(direction ? column_axes : row_axes);
        nk_f16_t const *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        nk_b128_vec_t const *tail_vec = direction ? &row_tail_vec : &column_tail_vec;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            float32x4_t const outer_x_f32x4 = vcvt_f32_f16(
                vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_index)));
            float32x4_t const outer_y_f32x4 = vcvt_f32_f16(
                vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_stride + outer_index)));
            float32x4_t const outer_z_f32x4 = vcvt_f32_f16(
                vreinterpret_f16_u16(vld1_dup_u16(outer_bits + 2 * outer_stride + outer_index)));
            for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
            nk_shapes_sweep_nearest_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_axes, inner_stride,
                                                 inner_count, tail_vec, nearest_f32x4);
            rejected = nk_shapes_hausdorff_row_f32_neon_(nearest_f32x4, farthest_f32x4, limit, limit_squared);
        }
    }
    vst1q_f32(hausdorffs, vsqrtq_f32(farthest_f32x4[0])), vst1q_f32(hausdorffs + 4, vsqrtq_f32(farthest_f32x4[1]));
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_neonfhm(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f32_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_f32_t const *limit, void *workspace,
                                                       nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const pair_limit = *limit;
    nk_f32_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_status_t const status = nk_shapes_chamfer_flips_f16_neonfhm_(a_packed[row], b_packed[column], pair_limit,
                                                                            workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f32_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_neonfhm(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f32_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_f32_t const *limit,
                                                         nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const pair_limit = *limit;
    nk_f32_t hausdorffs[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_hausdorff_flips_f16_neonfhm_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f32_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

/** Σ‖xᵢ′‖² of a pack of @c f16 points, 8 points per step through the zero padding, FMLAL widening
 *  each half of a register. */
NUMKONG_INLINE nk_f32_t nk_rmsds_norm_f16_neonfhm_(void const *packed) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t axis_stride;
    nk_u16_t const *axes = (nk_u16_t const *)nk_shapes_packed_axes_f16_serial_(packed, &axis_stride);
    float32x4_t norm_f32x4 = vdupq_n_f32(0);
    for (int axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < count; index += 8) {
            float16x8_t const values_f16x8 = vreinterpretq_f16_u16(vld1q_u16(axes + axis * axis_stride + index));
            norm_f32x4 = vfmlalq_low_f16(norm_f32x4, values_f16x8, values_f16x8);
            norm_f32x4 = vfmlalq_high_f16(norm_f32x4, values_f16x8, values_f16x8);
        }
    return vaddvq_f32(norm_f32x4);
}

/** Superposed RMSD of one pair of @c f16 packs with Ga + Gb in @p norms_squared, M in @c f32. */
NUMKONG_INLINE nk_f32_t nk_rmsds_pair_f16_neonfhm_(void const *row_packed, void const *column_packed,
                                                   nk_f32_t norms_squared) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_u16_t const *a_axes = (nk_u16_t const *)nk_shapes_packed_axes_f16_serial_(row_packed, &axis_stride);
    nk_u16_t const *b_axes = (nk_u16_t const *)nk_shapes_packed_axes_f16_serial_(column_packed, &axis_stride);
    float32x4_t sums_f32x4[9];
    float16x8_t a_f16x8[3], b_f16x8[3];
    for (int cell = 0; cell != 9; ++cell) sums_f32x4[cell] = vdupq_n_f32(0);
    for (nk_size_t index = 0; index < count; index += 8) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f16x8[axis] = vreinterpretq_f16_u16(vld1q_u16(a_axes + axis * axis_stride + index));
            b_f16x8[axis] = vreinterpretq_f16_u16(vld1q_u16(b_axes + axis * axis_stride + index));
        }
        for (int row = 0; row != 3; ++row)
            for (int column = 0; column != 3; ++column) {
                sums_f32x4[row * 3 + column] = vfmlalq_low_f16(sums_f32x4[row * 3 + column], a_f16x8[row],
                                                               b_f16x8[column]);
                sums_f32x4[row * 3 + column] = vfmlalq_high_f16(sums_f32x4[row * 3 + column], a_f16x8[row],
                                                                b_f16x8[column]);
            }
    }
    nk_f32_t cross_covariance[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = vaddvq_f32(sums_f32x4[cell]);
    nk_f32_t const residual = nk_rmsds_folded_f32_serial_(cross_covariance, norms_squared);
    return nk_sqrt_f32_neon_(residual / (nk_f32_t)count);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_neonfhm(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                    nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = ((nk_shapes_packed_header_f32_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f16_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f16_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    nk_f32_t row_norms[nk_rmsds_tile_frames_k], column_norms[nk_rmsds_tile_frames_k];
    for (nk_size_t row_begin = 0; row_begin < rows; row_begin += nk_rmsds_tile_frames_k) {
        nk_size_t const row_end = row_begin + nk_rmsds_tile_frames_k < rows ? row_begin + nk_rmsds_tile_frames_k : rows;
        for (nk_size_t row = row_begin; row != row_end; ++row)
            row_norms[row - row_begin] = nk_rmsds_norm_f16_neonfhm_(a_packed[row]);
        for (nk_size_t column_begin = 0; column_begin < columns; column_begin += nk_rmsds_tile_frames_k) {
            nk_size_t const column_end = column_begin + nk_rmsds_tile_frames_k < columns
                                             ? column_begin + nk_rmsds_tile_frames_k
                                             : columns;
            for (nk_size_t column = column_begin; column != column_end; ++column)
                column_norms[column - column_begin] = nk_rmsds_norm_f16_neonfhm_(b_packed[column]);
            for (nk_size_t row = row_begin; row != row_end; ++row) {
                nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
                for (nk_size_t column = column_begin; column != column_end; ++column)
                    results_row[column] = nk_rmsds_pair_f16_neonfhm_(
                        a_packed[row], b_packed[column],
                        row_norms[row - row_begin] + column_norms[column - column_begin]);
            }
        }
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONFHM
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SHAPES_NEONFHM_H
