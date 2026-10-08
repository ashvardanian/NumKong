/**
 *  @file include/numkong/mesh/neonfhm.h
 *  @author Ash Vardanian
 *  @date April 15, 2026
 *  @brief SIMD-accelerated point cloud alignment for NEON FP16 FHM, with widening FMA.
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section mesh_neonfhm_instructions ARM NEON FP16 Matrix Instructions (ARMv8.4-FHM)
 *
 *  @verbatim
 *  Intrinsic         Instruction                A76       M5
 *  vld3q_u16         LD3 (V.8H x 3)             6cy @ 1p  6cy @ 1p
 *  vsubq_f16         FSUB (V.8H, V.8H, V.8H)    2cy @ 2p  2cy @ 4p
 *  vfmlalq_low_f16   FMLAL (V.4S, V.8H, V.8H)   4cy @ 2p  4cy @ 4p
 *  vfmlalq_high_f16  FMLAL2 (V.4S, V.8H, V.8H)  4cy @ 2p  4cy @ 4p
 *  vcvt_f32_f16      FCVTL (V.4S, V.4H)         4cy @ 2p  3cy @ 4p
 *  vcvt_high_f32_f16 FCVTL2 (V.4S, V.8H)        4cy @ 2p  3cy @ 4p
 *  vfmaq_f32         FMLA (V.4S, V.4S, V.4S)    4cy @ 2p  3cy @ 4p
 *  vaddq_f32         FADD (V.4S, V.4S, V.4S)    2cy @ 2p  2cy @ 4p
 *  vaddvq_f32        FADDP+FADDP (V.4S)         5cy @ 1p  8cy @ 1p
 *  @endverbatim
 *
 *  The ARMv8.4-FHM extension (FEAT_FHM) provides FMLAL/FMLSL instructions that fuse FP16 to FP32
 *  widening with multiply-accumulate into a single operation. @c vfmlalq_low_f16 operates on
 *  elements 0-3 of the FP16 inputs; @c vfmlalq_high_f16 operates on elements 4-7 — together they
 *  process a full @c float16x8_t of data into two @c float32x4_t accumulators with full FP32
 *  accumulator precision.
 *
 *  Kabsch and Umeyama feed every moment, centroid sums included, through FMLAL in a single pass.
 *  They first shift each point by the first one with @c vsubq_f16, so the centering correction
 *  doesn't cancel far from the origin. That FP16 difference rounds by at most half a ULP of the
 *  shifted value, below the quantization of the inputs themselves. A point over 65504 away from its
 *  pivot overflows that shift, so such clouds take the widening NEON pass. RMSD widens before
 *  subtracting, since its FP16 difference would round where the serial FP32 one does not.
 */
#ifndef NUMKONG_MESH_NEONFHM_H
#define NUMKONG_MESH_NEONFHM_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/mesh/neon.h" // `nk_centered_moments_f16_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

/*  Load 8 fp16 xyz triplets (24 fp16 values) → 3x float16x8_t.
 *  Uses vld3q_u16 to de-interleave, then reinterprets as f16 (avoids vld3q_f16 which is
 *  unavailable on MSVC for ARM64).
 *
 *  Input: 24 contiguous fp16 [x0,y0,z0, ..., x7,y7,z7]
 *  Output: x_f16x8, y_f16x8, z_f16x8 channel vectors (8 lanes each) */
NUMKONG_INLINE void nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(nk_f16_t const *ptr, //
                                                              float16x8_t *x_out, float16x8_t *y_out,
                                                              float16x8_t *z_out) {
    uint16x8x3_t xyz_u16x8x3 = vld3q_u16((nk_u16_t const *)ptr);
    *x_out = vreinterpretq_f16_u16(xyz_u16x8x3.val[0]);
    *y_out = vreinterpretq_f16_u16(xyz_u16x8x3.val[1]);
    *z_out = vreinterpretq_f16_u16(xyz_u16x8x3.val[2]);
}

NUMKONG_INLINE void nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(nk_f16_t const *ptr, nk_size_t n_points, //
                                                                    float16x8_t *x_out, float16x8_t *y_out,
                                                                    float16x8_t *z_out) {
    nk_u16_t buf[24] = {0};
    nk_u16_t const *src = (nk_u16_t const *)ptr;
    for (nk_size_t k = 0; k < n_points * 3; ++k) buf[k] = src[k];
    nk_deinterleave_f16x8_to_f16x8x3_neonfhm_((nk_f16_t const *)buf, x_out, y_out, z_out);
}

/*  Widens before subtracting, like serial, as an F16 difference rounds and overflows past 65504. */
NUMKONG_INLINE void nk_accumulate_squared_delta_f16x8_neonfhm_(float16x8_t a_f16x8, float16x8_t b_f16x8,
                                                               float32x4_t *low_f32x4, float32x4_t *high_f32x4) {
    float32x4_t delta_low_f32x4 = vsubq_f32(vcvt_f32_f16(vget_low_f16(a_f16x8)), vcvt_f32_f16(vget_low_f16(b_f16x8)));
    float32x4_t delta_high_f32x4 = vsubq_f32(vcvt_high_f32_f16(a_f16x8), vcvt_high_f32_f16(b_f16x8));
    *low_f32x4 = vfmaq_f32(*low_f32x4, delta_low_f32x4, delta_low_f32x4);
    *high_f32x4 = vfmaq_f32(*high_f32x4, delta_high_f32x4, delta_high_f32x4);
}

NUMKONG_API nk_status_t nk_rmsd_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
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
    // Squared-delta accumulators split into low (elements 0-3) and high (4-7) halves
    float32x4_t sum_squared_x_low_f32x4 = zeros_f32x4, sum_squared_x_high_f32x4 = zeros_f32x4;
    float32x4_t sum_squared_y_low_f32x4 = zeros_f32x4, sum_squared_y_high_f32x4 = zeros_f32x4;
    float32x4_t sum_squared_z_low_f32x4 = zeros_f32x4, sum_squared_z_high_f32x4 = zeros_f32x4;

    float16x8_t a_x_f16x8, a_y_f16x8, a_z_f16x8;
    float16x8_t b_x_f16x8, b_y_f16x8, b_z_f16x8;
    nk_size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(a + i * 3, &a_x_f16x8, &a_y_f16x8, &a_z_f16x8);
        nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(b + i * 3, &b_x_f16x8, &b_y_f16x8, &b_z_f16x8);

        nk_accumulate_squared_delta_f16x8_neonfhm_(a_x_f16x8, b_x_f16x8, &sum_squared_x_low_f32x4,
                                                   &sum_squared_x_high_f32x4);
        nk_accumulate_squared_delta_f16x8_neonfhm_(a_y_f16x8, b_y_f16x8, &sum_squared_y_low_f32x4,
                                                   &sum_squared_y_high_f32x4);
        nk_accumulate_squared_delta_f16x8_neonfhm_(a_z_f16x8, b_z_f16x8, &sum_squared_z_low_f32x4,
                                                   &sum_squared_z_high_f32x4);
    }

    if (i < n) {
        nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(a + i * 3, n - i, &a_x_f16x8, &a_y_f16x8, &a_z_f16x8);
        nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(b + i * 3, n - i, &b_x_f16x8, &b_y_f16x8, &b_z_f16x8);

        nk_accumulate_squared_delta_f16x8_neonfhm_(a_x_f16x8, b_x_f16x8, &sum_squared_x_low_f32x4,
                                                   &sum_squared_x_high_f32x4);
        nk_accumulate_squared_delta_f16x8_neonfhm_(a_y_f16x8, b_y_f16x8, &sum_squared_y_low_f32x4,
                                                   &sum_squared_y_high_f32x4);
        nk_accumulate_squared_delta_f16x8_neonfhm_(a_z_f16x8, b_z_f16x8, &sum_squared_z_low_f32x4,
                                                   &sum_squared_z_high_f32x4);
    }

    nk_f32_t sum_squared = vaddvq_f32(vaddq_f32(sum_squared_x_low_f32x4, sum_squared_x_high_f32x4)) +
                           vaddvq_f32(vaddq_f32(sum_squared_y_low_f32x4, sum_squared_y_high_f32x4)) +
                           vaddvq_f32(vaddq_f32(sum_squared_z_low_f32x4, sum_squared_z_high_f32x4));
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(sum_squared / (nk_f32_t)n)), 0);
    return nk_success_k;
}

/** Centroids, centered cross-covariance, ‖a − ā‖² and ‖b − b̄‖² of f16 clouds,
 *  in one FMLAL pass shifted by the pivots in f16. */
NUMKONG_INLINE void nk_centered_moments_f16_neonfhm_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f32_t *centroid_a, nk_f32_t *centroid_b,
                                                     nk_f32_t *cross_covariance, nk_f32_t *centered_norm_squared_a,
                                                     nk_f32_t *centered_norm_squared_b) {
    nk_u16_t const *a_u16 = (nk_u16_t const *)a, *b_u16 = (nk_u16_t const *)b;
    float16x8_t const ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    float16x8_t pivot_a_f16x8[3], pivot_b_f16x8[3];
    float32x4_t sum_a_f32x4[3], sum_b_f32x4[3], covariance_f32x4[9];
    float32x4_t norm_squared_a_f32x4 = vdupq_n_f32(0), norm_squared_b_f32x4 = vdupq_n_f32(0);
    for (int j = 0; j != 3; ++j)
        pivot_a_f16x8[j] = vreinterpretq_f16_u16(vdupq_n_u16(a_u16[j])),
        pivot_b_f16x8[j] = vreinterpretq_f16_u16(vdupq_n_u16(b_u16[j])), sum_a_f32x4[j] = vdupq_n_f32(0),
        sum_b_f32x4[j] = vdupq_n_f32(0);
    for (int j = 0; j != 9; ++j) covariance_f32x4[j] = vdupq_n_f32(0);
    // Pads the tail with pivots, which shift to zeros and add nothing to the moments.
    nk_u16_t tail_a_u16[24], tail_b_u16[24];
    for (int k = 0; k != 24; ++k) tail_a_u16[k] = a_u16[k % 3], tail_b_u16[k] = b_u16[k % 3];
    for (nk_size_t i = 0; i < n; i += 8) {
        nk_u16_t const *a_chunk_u16 = a_u16 + i * 3, *b_chunk_u16 = b_u16 + i * 3;
        if (i + 8 > n) {
            for (nk_size_t k = 0; k != (n - i) * 3; ++k) tail_a_u16[k] = a_chunk_u16[k], tail_b_u16[k] = b_chunk_u16[k];
            a_chunk_u16 = tail_a_u16, b_chunk_u16 = tail_b_u16;
        }
        uint16x8x3_t a_u16x8x3 = vld3q_u16(a_chunk_u16), b_u16x8x3 = vld3q_u16(b_chunk_u16);
        float16x8_t a_f16x8[3], b_f16x8[3];
        for (int j = 0; j != 3; ++j) {
            a_f16x8[j] = vsubq_f16(vreinterpretq_f16_u16(a_u16x8x3.val[j]), pivot_a_f16x8[j]);
            b_f16x8[j] = vsubq_f16(vreinterpretq_f16_u16(b_u16x8x3.val[j]), pivot_b_f16x8[j]);
            sum_a_f32x4[j] = vfmlalq_high_f16(vfmlalq_low_f16(sum_a_f32x4[j], a_f16x8[j], ones_f16x8), a_f16x8[j],
                                              ones_f16x8);
            sum_b_f32x4[j] = vfmlalq_high_f16(vfmlalq_low_f16(sum_b_f32x4[j], b_f16x8[j], ones_f16x8), b_f16x8[j],
                                              ones_f16x8);
            norm_squared_a_f32x4 = vfmlalq_high_f16(vfmlalq_low_f16(norm_squared_a_f32x4, a_f16x8[j], a_f16x8[j]),
                                                    a_f16x8[j], a_f16x8[j]);
            norm_squared_b_f32x4 = vfmlalq_high_f16(vfmlalq_low_f16(norm_squared_b_f32x4, b_f16x8[j], b_f16x8[j]),
                                                    b_f16x8[j], b_f16x8[j]);
        }
        for (int j = 0; j != 9; ++j)
            covariance_f32x4[j] = vfmlalq_high_f16(vfmlalq_low_f16(covariance_f32x4[j], a_f16x8[j / 3], b_f16x8[j % 3]),
                                                   a_f16x8[j / 3], b_f16x8[j % 3]);
    }
    nk_f32_t norm_squared_a = vaddvq_f32(norm_squared_a_f32x4), norm_squared_b = vaddvq_f32(norm_squared_b_f32x4);
    // An FP16 shift past 65504 leaves an infinite norm, so such clouds widen before shifting.
    if ((norm_squared_a + norm_squared_b) - (norm_squared_a + norm_squared_b) != 0) {
        nk_centered_moments_f16_neon_(a, b, n, centroid_a, centroid_b, cross_covariance, centered_norm_squared_a,
                                      centered_norm_squared_b);
        return;
    }
    nk_f32_t pivot_a[3], pivot_b[3], sum_a[3], sum_b[3], covariance[9];
    for (int j = 0; j != 3; ++j)
        nk_f16_to_f32_serial_(a + j, pivot_a + j), nk_f16_to_f32_serial_(b + j, pivot_b + j),
            sum_a[j] = vaddvq_f32(sum_a_f32x4[j]), sum_b[j] = vaddvq_f32(sum_b_f32x4[j]);
    for (int j = 0; j != 9; ++j) covariance[j] = vaddvq_f32(covariance_f32x4[j]);
    nk_centered_moments_finalize_f32_serial_(n, pivot_a, pivot_b, sum_a, sum_b, covariance, norm_squared_a,
                                             norm_squared_b, centroid_a, centroid_b, cross_covariance,
                                             centered_norm_squared_a, centered_norm_squared_b);
}

NUMKONG_API nk_status_t nk_kabsch_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f32_t *result, nk_stream_t stream) {
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
    nk_centered_moments_f16_neonfhm_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_squared_a,
                                     &centered_norm_squared_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9], optimal_rotation[9];
    nk_svd3x3_f32_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);
    nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    if (nk_det3x3_f32_serial_(optimal_rotation) < 0) {
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    nk_f32_t sum_squared = nk_folded_ssd_f32_serial_(optimal_rotation, 1.0f, cross_covariance, centered_norm_squared_a,
                                                     centered_norm_squared_b);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(sum_squared / (nk_f32_t)n)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                               nk_f32_t *result, nk_stream_t stream) {
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
    nk_centered_moments_f16_neonfhm_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_squared_a,
                                     &centered_norm_squared_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9], optimal_rotation[9];
    nk_svd3x3_f32_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);
    nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);

    // Scale factor: c = trace(D · S) / ‖a − ā‖², the last singular value signed by the reflection.
    nk_f32_t determinant = nk_det3x3_f32_serial_(optimal_rotation);
    nk_f32_t trace_ds = svd_diagonal[0] + svd_diagonal[4] + (determinant < 0 ? -svd_diagonal[8] : svd_diagonal[8]);
    nk_f32_t c = trace_ds / centered_norm_squared_a;
    if (scale) *scale = c;
    if (determinant < 0) {
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    nk_f32_t sum_squared = nk_folded_ssd_f32_serial_(optimal_rotation, c, cross_covariance, centered_norm_squared_a,
                                                     centered_norm_squared_b);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(sum_squared / (nk_f32_t)n)), 0);
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
#endif // NUMKONG_MESH_NEONFHM_H
