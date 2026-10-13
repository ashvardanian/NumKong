/**
 *  @file include/numkong/meshes/neonfhm.h
 *  @author Ash Vardanian
 *  @date April 15, 2026
 *  @brief Many-to-many shape distances between 3D point clouds for NEON FHM, with widening FMA.
 *
 *  @sa include/numkong/meshes.h
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
 *  a − b in the NEON kernel, but without the FCVTL. The squares, the flips and the fold run the
 *  NEON kernel's f32 lanes, so both kernels return the same bits.
 *
 *  The differences and sums are not taken in f16, where they would round by up to 2⁻¹¹ of
 *  themselves and overflow past 65504, missing the f32 error bounds. @c nk_sqeuclidean_f16_neon and
 *  @c nk_mahalanobis_f16_neon widen before subtracting for the same reason. Padded lanes take an
 *  infinite x, so they never win a minimum, and a last octuple of four points or fewer skips its
 *  high half, wasting no more lanes than the NEON kernel's quadruples.
 */
#ifndef NUMKONG_MESHES_NEONFHM_H
#define NUMKONG_MESHES_NEONFHM_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/meshes/serial.h" // `nk_define_shapes_flip_kernels_`
#include "numkong/meshes/neon.h"   // `nk_shapes_lower_nearest_differences_f32_neon_`

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

/** One directed Chamfer and Hausdorff pass over f16 packs in f32 lanes, eight inner points at a
 *  time. The last octuple of a count off the multiple takes an infinite x in its padded lanes and
 *  skips its high half when the low one holds all its points. */
NUMKONG_INLINE void nk_shapes_nearest_f16_neonfhm_(nk_f16_t const *outer_axes, nk_size_t outer_stride,
                                                   nk_size_t outer_count, nk_f16_t const *inner_axes,
                                                   nk_size_t inner_stride, nk_size_t inner_count, nk_f32_t *sums,
                                                   nk_f32_t *maxima) {
    nk_u16_t const *outer_bits = (nk_u16_t const *)outer_axes, *inner_bits = (nk_u16_t const *)inner_axes;
    uint16x8_t const infinity_u16x8 = vdupq_n_u16(0x7C00);
    float32x4_t const infinity_f32x4 = vdupq_n_f32(NUMKONG_F32_INF);
    nk_size_t const octuples_end = inner_count & ~(nk_size_t)7;
    nk_b128_vec_t tail_vec;
    for (nk_size_t lane = 0; lane != 8; ++lane) tail_vec.u16s[lane] = octuples_end + lane < inner_count ? 0xFFFF : 0;
    float32x4_t sums_f32x4[2] = {vld1q_f32(sums), vld1q_f32(sums + 4)};
    float32x4_t maxima_f32x4[2] = {vld1q_f32(maxima), vld1q_f32(maxima + 4)};
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        float32x4_t const outer_x_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_index)));
        float32x4_t const outer_y_f32x4 = vcvt_f32_f16(
            vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_stride + outer_index)));
        float32x4_t const outer_z_f32x4 = vcvt_f32_f16(
            vreinterpret_f16_u16(vld1_dup_u16(outer_bits + 2 * outer_stride + outer_index)));
        float32x4_t nearest_f32x4[8];
        for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
        for (nk_size_t inner_index = 0; inner_index != octuples_end; inner_index += 8) {
            float16x8_t const inner_x_f16x8 = vreinterpretq_f16_u16(vld1q_u16(inner_bits + inner_index));
            float16x8_t const inner_y_f16x8 = vreinterpretq_f16_u16(vld1q_u16(inner_bits + inner_stride + inner_index));
            float16x8_t const inner_z_f16x8 = vreinterpretq_f16_u16(
                vld1q_u16(inner_bits + 2 * inner_stride + inner_index));
            nk_shapes_lower_nearest_low_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f16x8,
                                                     inner_y_f16x8, inner_z_f16x8, nearest_f32x4);
            nk_shapes_lower_nearest_high_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f16x8,
                                                      inner_y_f16x8, inner_z_f16x8, nearest_f32x4);
        }
        if (octuples_end != inner_count) {
            float16x8_t const inner_x_f16x8 = vreinterpretq_f16_u16(
                vbslq_u16(tail_vec.u16x8, vld1q_u16(inner_bits + octuples_end), infinity_u16x8));
            float16x8_t const inner_y_f16x8 = vreinterpretq_f16_u16(
                vld1q_u16(inner_bits + inner_stride + octuples_end));
            float16x8_t const inner_z_f16x8 = vreinterpretq_f16_u16(
                vld1q_u16(inner_bits + 2 * inner_stride + octuples_end));
            nk_shapes_lower_nearest_low_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f16x8,
                                                     inner_y_f16x8, inner_z_f16x8, nearest_f32x4);
            if (octuples_end + 4 < inner_count)
                nk_shapes_lower_nearest_high_f16_neonfhm_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f16x8,
                                                          inner_y_f16x8, inner_z_f16x8, nearest_f32x4);
        }
        nk_shapes_fold_nearest_f32_neon_(nearest_f32x4, sums_f32x4, maxima_f32x4);
    }
    vst1q_f32(sums, sums_f32x4[0]), vst1q_f32(sums + 4, sums_f32x4[1]);
    vst1q_f32(maxima, maxima_f32x4[0]), vst1q_f32(maxima + 4, maxima_f32x4[1]);
}

nk_define_shapes_flip_kernels_(f16, f32, neonfhm, nk_shapes_nearest_f16_neonfhm_, NUMKONG_F32_INF)

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
#endif // NUMKONG_MESHES_NEONFHM_H
