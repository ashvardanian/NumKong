/**
 *  @file include/numkong/mesh/genoa.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief SIMD-accelerated point cloud alignment for Genoa, AVX-512-BF16.
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section genoa_mesh_instructions Key AVX-512 BF16 Mesh Instructions
 *
 *  @verbatim
 *  Intrinsic                 Instruction                  Genoa      Sapphire
 *  _mm512_dpbf16_ps          VDPBF16PS (ZMM, ZMM, ZMM)    6cy @ p01  6cy @ p05
 *  _mm512_permutexvar_epi16  VPERMW (ZMM, ZMM, ZMM)       3cy @ p5   6cy @ p5
 *  _mm512_maskz_loadu_epi16  VMOVDQU16 (ZMM{k}, M)        9cy @ L1   9cy @ L1
 *  @endverbatim
 *
 *  Kabsch and Umeyama use a 15-lane channel-grouped layout: 10 xyz triplets per register (30 values
 *  laid out as [x0..x9, y0..y9, z0..z9, _, _] after a single VPERMW). That maps cleanly onto
 *  VDPBF16PS, which pairs adjacent bf16 values per fp32 lane; 5 channel-consecutive pairs give a
 *  single H-cell per lane-range. Three product accumulators (a*b, a*rot1(b), a*rot2(b)) cover the 9
 *  cross-covariance cells, matching the Skylake structure. Each point is first shifted by the first
 *  one, so the centering correction doesn't cancel far from the origin. With no BF16 subtraction,
 *  the shift widens, subtracts in FP32 and rounds back with VCVTNE2PS2BF16, by at most half a ULP
 *  of the shifted value, below the quantization of the inputs themselves.
 *
 *  RMSD reuses the Skylake kernel, as Σa² + Σb² − 2Σab over raw pairs cancels for close clouds.
 */
#ifndef NUMKONG_MESH_GENOA_H
#define NUMKONG_MESH_GENOA_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_GENOA

#include "numkong/types.h"
#include "numkong/mesh/skylake.h" // `nk_rmsd_bf16_through_f32_skylake_`, `nk_bf16x16_to_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512bf16", "f16c", "fma", "bmi", "bmi2")
#endif

NUMKONG_API nk_status_t nk_rmsd_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_rmsd_bf16_through_f32_skylake_(a, b, n, a_centroid, b_centroid, rotation, scale, result);
    return nk_success_k;
}

/** Loads up to 10 bf16 triplets padded with the pivot, shifts them by it in f32, and rounds them
 *  back to bf16. */
NUMKONG_INLINE __m512i nk_load_shifted_bf16x32_genoa_(nk_bf16_t const *points, __mmask32 lanes_m32,
                                                      __m512i pivot_bf16x32, __m512 const *pivot_f32x16) {
    __m512i values_bf16x32 = _mm512_mask_loadu_epi16(pivot_bf16x32, lanes_m32, points);
    __m512 low_f32x16 = _mm512_sub_ps(nk_bf16x16_to_f32x16_skylake_(_mm512_castsi512_si256(values_bf16x32)),
                                      pivot_f32x16[0]);
    __m512 high_f32x16 = _mm512_sub_ps(nk_bf16x16_to_f32x16_skylake_(_mm512_extracti64x4_epi64(values_bf16x32, 1)),
                                       pivot_f32x16[1]);
    return (__m512i)_mm512_cvtne2ps_pbh(high_f32x16, low_f32x16);
}

/** Centroids, centered cross-covariance, ‖a − ā‖² and ‖b − b̄‖² of bf16 clouds,
 *  in one VDPBF16PS pass shifted by the pivots. */
NUMKONG_INLINE void nk_centered_moments_bf16_genoa_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                    nk_f32_t *centroid_a, nk_f32_t *centroid_b,
                                                    nk_f32_t *cross_covariance, nk_f32_t *centered_norm_squared_a,
                                                    nk_f32_t *centered_norm_squared_b) {
    // Groups 10 triplets as [x0..x9, y0..y9, z0..z9, _, _], so fp32 lanes 0-4, 5-9, 10-14 hold xyz.
    __m512i const channel_group_indices_i16x32 = _mm512_set_epi16( //
        31, 30, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 27, 24, 21, 18, 15, 12, 9, 6,
        3, 0);
    // Groups b with each channel slot holding the next channel, pairing the cells xy, yz, zx.
    __m512i const rotation_1_indices_i16x32 = _mm512_set_epi16( //
        31, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 28, 25, 22, 19, 16, 13, 10, 7,
        4, 1);
    // Groups b with each channel slot holding the channel after next, pairing the cells xz, yx, zy.
    __m512i const rotation_2_indices_i16x32 = _mm512_set_epi16( //
        31, 30, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, 29, 26, 23, 20, 17, 14, 11, 8,
        5, 2);
    __m512i const triplet_indices_i16x32 = _mm512_set_epi16( //
        1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0);
    __m512i const ones_bf16x32 = _mm512_set1_epi16(0x3F80);
    __m512i const pivot_a_bf16x32 = _mm512_permutexvar_epi16(triplet_indices_i16x32, _mm512_maskz_loadu_epi16(0x7, a));
    __m512i const pivot_b_bf16x32 = _mm512_permutexvar_epi16(triplet_indices_i16x32, _mm512_maskz_loadu_epi16(0x7, b));
    __m512 const pivot_a_f32x16[2] = {nk_bf16x16_to_f32x16_skylake_(_mm512_castsi512_si256(pivot_a_bf16x32)),
                                      nk_bf16x16_to_f32x16_skylake_(_mm512_extracti64x4_epi64(pivot_a_bf16x32, 1))};
    __m512 const pivot_b_f32x16[2] = {nk_bf16x16_to_f32x16_skylake_(_mm512_castsi512_si256(pivot_b_bf16x32)),
                                      nk_bf16x16_to_f32x16_skylake_(_mm512_extracti64x4_epi64(pivot_b_bf16x32, 1))};

    // Σa, Σb, Σ‖a‖², Σ‖b‖², and Σ a · b against b grouped with rotations 0, 1, 2.
    __m512 moments_f32x16[7];
    for (int j = 0; j != 7; ++j) moments_f32x16[j] = _mm512_setzero_ps();
    for (nk_size_t index = 0; index < n; index += 10) {
        nk_size_t points = n - index < 10 ? n - index : 10;
        __mmask32 lanes_m32 = (__mmask32)_bzhi_u32(0x3FFFFFFF, (nk_u32_t)(points * 3));
        __m512i a_bf16x32 = nk_load_shifted_bf16x32_genoa_(a + index * 3, lanes_m32, pivot_a_bf16x32, pivot_a_f32x16);
        __m512i b_bf16x32 = nk_load_shifted_bf16x32_genoa_(b + index * 3, lanes_m32, pivot_b_bf16x32, pivot_b_f32x16);
        __m512bh a_grouped_bf16x32 = nk_m512bh_from_m512i_(
            _mm512_permutexvar_epi16(channel_group_indices_i16x32, a_bf16x32));
        __m512bh b_grouped_bf16x32 = nk_m512bh_from_m512i_(
            _mm512_permutexvar_epi16(channel_group_indices_i16x32, b_bf16x32));
        __m512bh b_rotation_1_bf16x32 = nk_m512bh_from_m512i_(
            _mm512_permutexvar_epi16(rotation_1_indices_i16x32, b_bf16x32));
        __m512bh b_rotation_2_bf16x32 = nk_m512bh_from_m512i_(
            _mm512_permutexvar_epi16(rotation_2_indices_i16x32, b_bf16x32));
        moments_f32x16[0] = _mm512_dpbf16_ps(moments_f32x16[0], a_grouped_bf16x32, nk_m512bh_from_m512i_(ones_bf16x32));
        moments_f32x16[1] = _mm512_dpbf16_ps(moments_f32x16[1], b_grouped_bf16x32, nk_m512bh_from_m512i_(ones_bf16x32));
        moments_f32x16[2] = _mm512_dpbf16_ps(moments_f32x16[2], a_grouped_bf16x32, a_grouped_bf16x32);
        moments_f32x16[3] = _mm512_dpbf16_ps(moments_f32x16[3], b_grouped_bf16x32, b_grouped_bf16x32);
        moments_f32x16[4] = _mm512_dpbf16_ps(moments_f32x16[4], a_grouped_bf16x32, b_grouped_bf16x32);
        moments_f32x16[5] = _mm512_dpbf16_ps(moments_f32x16[5], a_grouped_bf16x32, b_rotation_1_bf16x32);
        moments_f32x16[6] = _mm512_dpbf16_ps(moments_f32x16[6], a_grouped_bf16x32, b_rotation_2_bf16x32);
    }

    __mmask16 const channel_m16[3] = {0x001F, 0x03E0, 0x7C00};
    nk_f32_t pivot_a[3], pivot_b[3], sum_a[3], sum_b[3], covariance[9];
    for (int j = 0; j != 3; ++j) {
        nk_bf16_to_f32_(a + j, pivot_a + j), nk_bf16_to_f32_(b + j, pivot_b + j);
        sum_a[j] = _mm512_mask_reduce_add_ps(channel_m16[j], moments_f32x16[0]);
        sum_b[j] = _mm512_mask_reduce_add_ps(channel_m16[j], moments_f32x16[1]);
        covariance[j * 3 + j] = _mm512_mask_reduce_add_ps(channel_m16[j], moments_f32x16[4]);
        covariance[j * 3 + (j + 1) % 3] = _mm512_mask_reduce_add_ps(channel_m16[j], moments_f32x16[5]);
        covariance[j * 3 + (j + 2) % 3] = _mm512_mask_reduce_add_ps(channel_m16[j], moments_f32x16[6]);
    }
    nk_centered_moments_finalize_f32_(n, pivot_a, pivot_b, sum_a, sum_b, covariance,
                                      _mm512_reduce_add_ps(moments_f32x16[2]), _mm512_reduce_add_ps(moments_f32x16[3]),
                                      centroid_a, centroid_b, cross_covariance, centered_norm_squared_a,
                                      centered_norm_squared_b);
}

NUMKONG_API nk_status_t nk_kabsch_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
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

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_squared_a, centered_norm_squared_b;
    nk_centered_moments_bf16_genoa_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_squared_a,
                                    &centered_norm_squared_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9], optimal_rotation[9];
    nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
    nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    if (nk_det3x3_f32_(optimal_rotation) < 0) {
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    nk_f32_t sum_squared = nk_folded_ssd_f32_(optimal_rotation, 1.0f, cross_covariance, centered_norm_squared_a,
                                              centered_norm_squared_b);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(sum_squared / (nk_f32_t)n)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
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

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_squared_a, centered_norm_squared_b;
    nk_centered_moments_bf16_genoa_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_squared_a,
                                    &centered_norm_squared_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9], optimal_rotation[9];
    nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
    nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);

    // Scale factor: c = trace(D · S) / ‖a − ā‖², the last singular value signed by the reflection.
    nk_f32_t determinant = nk_det3x3_f32_(optimal_rotation);
    nk_f32_t trace_ds = svd_diagonal[0] + svd_diagonal[4] + (determinant < 0 ? -svd_diagonal[8] : svd_diagonal[8]);
    nk_f32_t c = trace_ds / centered_norm_squared_a;
    if (scale) *scale = c;
    if (determinant < 0) {
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    nk_f32_t sum_squared = nk_folded_ssd_f32_(optimal_rotation, c, cross_covariance, centered_norm_squared_a,
                                              centered_norm_squared_b);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(sum_squared / (nk_f32_t)n)));
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

#endif // NUMKONG_TARGET_GENOA
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_MESH_GENOA_H
