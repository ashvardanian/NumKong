/**
 *  @brief SIMD-accelerated Point Cloud Alignment for Genoa (AVX-512-BF16).
 *  @file include/numkong/mesh/genoa.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section genoa_mesh_instructions Key AVX-512 BF16 Mesh Instructions
 *
 *      Intrinsic                 Instruction                  Genoa      Sapphire
 *      _mm512_dpbf16_ps          VDPBF16PS (ZMM, ZMM, ZMM)    6cy @ p01  6cy @ p05
 *      _mm512_permutexvar_epi16  VPERMW (ZMM, ZMM, ZMM)       3cy @ p5   6cy @ p5
 *      _mm512_maskz_loadu_epi16  VMOVDQU16 (ZMM{k}, M)        9cy @ L1   9cy @ L1
 *
 *  The bf16 mesh kernels use a 15-lane channel-grouped layout: 10 xyz triplets per ZMM (30 bf16
 *  values laid out as [x0..x9, y0..y9, z0..z9, _, _] after a single VPERMW). That maps cleanly
 *  onto VDPBF16PS, which pairs adjacent bf16 values per fp32 lane; 5 channel-consecutive pairs
 *  give a single H-cell per lane-range. Three product accumulators (a*b, a*rot1(b), a*rot2(b))
 *  cover the 9 cross-covariance cells, matching the Skylake structure.
 */
#ifndef NK_MESH_GENOA_H
#define NK_MESH_GENOA_H

#if NK_TARGET_X8664_
#if NK_TARGET_GENOA

#include "numkong/types.h"
#include "numkong/mesh/serial.h"
#include "numkong/spatial/haswell.h" // `nk_f32_sqrt_haswell`

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

NK_PUBLIC void nk_rmsd_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    // 32-lane bf16 chunks = 10 triplets + 2 padding bf16 per register.
    // VDPBF16PS pairs adjacent bf16 per fp32 lane: lane[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1].
    // For RMSD we need Σ(a-b)², computed via Σ a² + Σ b² - 2 Σ a·b.
    __m512 const zeros_f32x16 = _mm512_setzero_ps();
    __m512 norm_squared_a_f32x16 = zeros_f32x16;
    __m512 norm_squared_b_f32x16 = zeros_f32x16;
    __m512 cross_product_f32x16 = zeros_f32x16;
    nk_size_t index = 0;

    __mmask32 const full_mask_bf16 = (__mmask32)0x3FFFFFFF; // 30 bf16 valid, 2 bf16 padding zeros

    for (; index + 10 <= n; index += 10) {
        __m512i a_bf16x32 = _mm512_maskz_loadu_epi16(full_mask_bf16, (__m512i const *)(a + index * 3));
        __m512i b_bf16x32 = _mm512_maskz_loadu_epi16(full_mask_bf16, (__m512i const *)(b + index * 3));
        norm_squared_a_f32x16 = _mm512_dpbf16_ps(norm_squared_a_f32x16, nk_m512bh_from_m512i_(a_bf16x32),
                                                 nk_m512bh_from_m512i_(a_bf16x32));
        norm_squared_b_f32x16 = _mm512_dpbf16_ps(norm_squared_b_f32x16, nk_m512bh_from_m512i_(b_bf16x32),
                                                 nk_m512bh_from_m512i_(b_bf16x32));
        cross_product_f32x16 = _mm512_dpbf16_ps(cross_product_f32x16, nk_m512bh_from_m512i_(a_bf16x32),
                                                nk_m512bh_from_m512i_(b_bf16x32));
    }

    if (index < n) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0x3FFFFFFF, (nk_u32_t)((n - index) * 3));
        __m512i a_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (__m512i const *)(a + index * 3));
        __m512i b_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (__m512i const *)(b + index * 3));
        norm_squared_a_f32x16 = _mm512_dpbf16_ps(norm_squared_a_f32x16, nk_m512bh_from_m512i_(a_bf16x32),
                                                 nk_m512bh_from_m512i_(a_bf16x32));
        norm_squared_b_f32x16 = _mm512_dpbf16_ps(norm_squared_b_f32x16, nk_m512bh_from_m512i_(b_bf16x32),
                                                 nk_m512bh_from_m512i_(b_bf16x32));
        cross_product_f32x16 = _mm512_dpbf16_ps(cross_product_f32x16, nk_m512bh_from_m512i_(a_bf16x32),
                                                nk_m512bh_from_m512i_(b_bf16x32));
    }

    nk_f32_t norm_squared_a = _mm512_reduce_add_ps(norm_squared_a_f32x16);
    nk_f32_t norm_squared_b = _mm512_reduce_add_ps(norm_squared_b_f32x16);
    nk_f32_t cross_product = _mm512_reduce_add_ps(cross_product_f32x16);
    nk_f32_t sum_squared = norm_squared_a + norm_squared_b - 2.0f * cross_product;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_haswell(sum_squared / (nk_f32_t)n);
}

// Channel-grouping permute: 10 xyz triplets + 2 padding bf16 → [x0..x9, y0..y9, z0..z9, _, _].
// After VPERMW lanes 0..4 carry the x-channel (2 bf16 per fp32 lane), 5..9 carry y, 10..14 carry z.
#define NK_MESH_GENOA_CHANNEL_GROUP_INDICES_                                                                           \
    _mm512_set_epi16(31, 30, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 27, 24, 21, 18, \
                     15, 12, 9, 6, 3, 0)

// Rotation-1 applied during channel-grouping: each channel slot carries the *next* channel of b.
//    x-slot gets b.y, y-slot gets b.z, z-slot gets b.x.  Pairs covariance cells (xy, yz, zx).
#define NK_MESH_GENOA_ROTATION_1_INDICES_                                                                             \
    _mm512_set_epi16(31, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 28, 25, 22, 19, \
                     16, 13, 10, 7, 4, 1)

// Rotation-2: x-slot gets b.z, y-slot gets b.x, z-slot gets b.y.  Pairs covariance cells (xz, yx, zy).
#define NK_MESH_GENOA_ROTATION_2_INDICES_                                                                             \
    _mm512_set_epi16(31, 30, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, 29, 26, 23, 20, \
                     17, 14, 11, 8, 5, 2)

NK_PUBLIC void nk_kabsch_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    __m512i const idx_channel_group_i16x32 = NK_MESH_GENOA_CHANNEL_GROUP_INDICES_;
    __m512i const idx_rotation_1_i16x32 = NK_MESH_GENOA_ROTATION_1_INDICES_;
    __m512i const idx_rotation_2_i16x32 = NK_MESH_GENOA_ROTATION_2_INDICES_;
    __m512i const ones_bf16x32 = _mm512_set1_epi16(0x3F80); // bf16 representation of 1.0

    __m512 const zeros_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_f32x16 = zeros_f32x16, sum_b_f32x16 = zeros_f32x16;
    __m512 norm_squared_a_f32x16 = zeros_f32x16, norm_squared_b_f32x16 = zeros_f32x16;
    __m512 product_diagonal_f32x16 = zeros_f32x16;
    __m512 product_rotation_1_f32x16 = zeros_f32x16;
    __m512 product_rotation_2_f32x16 = zeros_f32x16;

    __mmask32 const full_mask_bf16 = (__mmask32)0x3FFFFFFF;

    nk_size_t index = 0;
    for (; index + 10 <= n; index += 10) {
        __m512i a_raw_bf16x32 = _mm512_maskz_loadu_epi16(full_mask_bf16, (__m512i const *)(a + index * 3));
        __m512i b_raw_bf16x32 = _mm512_maskz_loadu_epi16(full_mask_bf16, (__m512i const *)(b + index * 3));
        __m512i a_grouped_bf16x32 = _mm512_permutexvar_epi16(idx_channel_group_i16x32, a_raw_bf16x32);
        __m512i b_grouped_bf16x32 = _mm512_permutexvar_epi16(idx_channel_group_i16x32, b_raw_bf16x32);
        __m512i b_rotation_1_bf16x32 = _mm512_permutexvar_epi16(idx_rotation_1_i16x32, b_raw_bf16x32);
        __m512i b_rotation_2_bf16x32 = _mm512_permutexvar_epi16(idx_rotation_2_i16x32, b_raw_bf16x32);

        sum_a_f32x16 = _mm512_dpbf16_ps(sum_a_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                        nk_m512bh_from_m512i_(ones_bf16x32));
        sum_b_f32x16 = _mm512_dpbf16_ps(sum_b_f32x16, nk_m512bh_from_m512i_(b_grouped_bf16x32),
                                        nk_m512bh_from_m512i_(ones_bf16x32));
        norm_squared_a_f32x16 = _mm512_dpbf16_ps(norm_squared_a_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                 nk_m512bh_from_m512i_(a_grouped_bf16x32));
        norm_squared_b_f32x16 = _mm512_dpbf16_ps(norm_squared_b_f32x16, nk_m512bh_from_m512i_(b_grouped_bf16x32),
                                                 nk_m512bh_from_m512i_(b_grouped_bf16x32));
        product_diagonal_f32x16 = _mm512_dpbf16_ps(product_diagonal_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                   nk_m512bh_from_m512i_(b_grouped_bf16x32));
        product_rotation_1_f32x16 = _mm512_dpbf16_ps(product_rotation_1_f32x16,
                                                     nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                     nk_m512bh_from_m512i_(b_rotation_1_bf16x32));
        product_rotation_2_f32x16 = _mm512_dpbf16_ps(product_rotation_2_f32x16,
                                                     nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                     nk_m512bh_from_m512i_(b_rotation_2_bf16x32));
    }

    if (index < n) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0x3FFFFFFF, (nk_u32_t)((n - index) * 3));
        __m512i a_raw_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (__m512i const *)(a + index * 3));
        __m512i b_raw_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (__m512i const *)(b + index * 3));
        __m512i a_grouped_bf16x32 = _mm512_permutexvar_epi16(idx_channel_group_i16x32, a_raw_bf16x32);
        __m512i b_grouped_bf16x32 = _mm512_permutexvar_epi16(idx_channel_group_i16x32, b_raw_bf16x32);
        __m512i b_rotation_1_bf16x32 = _mm512_permutexvar_epi16(idx_rotation_1_i16x32, b_raw_bf16x32);
        __m512i b_rotation_2_bf16x32 = _mm512_permutexvar_epi16(idx_rotation_2_i16x32, b_raw_bf16x32);

        sum_a_f32x16 = _mm512_dpbf16_ps(sum_a_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                        nk_m512bh_from_m512i_(ones_bf16x32));
        sum_b_f32x16 = _mm512_dpbf16_ps(sum_b_f32x16, nk_m512bh_from_m512i_(b_grouped_bf16x32),
                                        nk_m512bh_from_m512i_(ones_bf16x32));
        norm_squared_a_f32x16 = _mm512_dpbf16_ps(norm_squared_a_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                 nk_m512bh_from_m512i_(a_grouped_bf16x32));
        norm_squared_b_f32x16 = _mm512_dpbf16_ps(norm_squared_b_f32x16, nk_m512bh_from_m512i_(b_grouped_bf16x32),
                                                 nk_m512bh_from_m512i_(b_grouped_bf16x32));
        product_diagonal_f32x16 = _mm512_dpbf16_ps(product_diagonal_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                   nk_m512bh_from_m512i_(b_grouped_bf16x32));
        product_rotation_1_f32x16 = _mm512_dpbf16_ps(product_rotation_1_f32x16,
                                                     nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                     nk_m512bh_from_m512i_(b_rotation_1_bf16x32));
        product_rotation_2_f32x16 = _mm512_dpbf16_ps(product_rotation_2_f32x16,
                                                     nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                     nk_m512bh_from_m512i_(b_rotation_2_bf16x32));
    }

    // Channel demux by lane range (x=0..4, y=5..9, z=10..14, lane 15 padding).
    __mmask16 const mask_channel_x_f32 = 0x001F;
    __mmask16 const mask_channel_y_f32 = 0x03E0;
    __mmask16 const mask_channel_z_f32 = 0x7C00;

    nk_f32_t sum_a_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_a_f32x16);
    nk_f32_t sum_a_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_a_f32x16);
    nk_f32_t sum_a_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_a_f32x16);
    nk_f32_t sum_b_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_b_f32x16);
    nk_f32_t sum_b_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_b_f32x16);
    nk_f32_t sum_b_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_b_f32x16);
    nk_f32_t norm_squared_a = _mm512_reduce_add_ps(norm_squared_a_f32x16);
    nk_f32_t norm_squared_b = _mm512_reduce_add_ps(norm_squared_b_f32x16);

    nk_f32_t covariance_x_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_diagonal_f32x16);
    nk_f32_t covariance_y_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_diagonal_f32x16);
    nk_f32_t covariance_z_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_diagonal_f32x16);
    nk_f32_t covariance_x_y = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_y_z = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_x = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_x_z = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_x = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_z_y = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_2_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Parallel-axis correction.
    nk_f32_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - (nk_f32_t)n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - (nk_f32_t)n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - (nk_f32_t)n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - (nk_f32_t)n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - (nk_f32_t)n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - (nk_f32_t)n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - (nk_f32_t)n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - (nk_f32_t)n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

    // Identity-dominant short-circuit.
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
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
        if (nk_det3x3_f32_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
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
    if (scale) *scale = 1.0f;

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f32_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0f * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    __m512i const idx_channel_group_i16x32 = NK_MESH_GENOA_CHANNEL_GROUP_INDICES_;
    __m512i const idx_rotation_1_i16x32 = NK_MESH_GENOA_ROTATION_1_INDICES_;
    __m512i const idx_rotation_2_i16x32 = NK_MESH_GENOA_ROTATION_2_INDICES_;
    __m512i const ones_bf16x32 = _mm512_set1_epi16(0x3F80);

    __m512 const zeros_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_f32x16 = zeros_f32x16, sum_b_f32x16 = zeros_f32x16;
    __m512 norm_squared_a_f32x16 = zeros_f32x16, norm_squared_b_f32x16 = zeros_f32x16;
    __m512 product_diagonal_f32x16 = zeros_f32x16;
    __m512 product_rotation_1_f32x16 = zeros_f32x16;
    __m512 product_rotation_2_f32x16 = zeros_f32x16;

    __mmask32 const full_mask_bf16 = (__mmask32)0x3FFFFFFF;

    nk_size_t index = 0;
    for (; index + 10 <= n; index += 10) {
        __m512i a_raw_bf16x32 = _mm512_maskz_loadu_epi16(full_mask_bf16, (__m512i const *)(a + index * 3));
        __m512i b_raw_bf16x32 = _mm512_maskz_loadu_epi16(full_mask_bf16, (__m512i const *)(b + index * 3));
        __m512i a_grouped_bf16x32 = _mm512_permutexvar_epi16(idx_channel_group_i16x32, a_raw_bf16x32);
        __m512i b_grouped_bf16x32 = _mm512_permutexvar_epi16(idx_channel_group_i16x32, b_raw_bf16x32);
        __m512i b_rotation_1_bf16x32 = _mm512_permutexvar_epi16(idx_rotation_1_i16x32, b_raw_bf16x32);
        __m512i b_rotation_2_bf16x32 = _mm512_permutexvar_epi16(idx_rotation_2_i16x32, b_raw_bf16x32);

        sum_a_f32x16 = _mm512_dpbf16_ps(sum_a_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                        nk_m512bh_from_m512i_(ones_bf16x32));
        sum_b_f32x16 = _mm512_dpbf16_ps(sum_b_f32x16, nk_m512bh_from_m512i_(b_grouped_bf16x32),
                                        nk_m512bh_from_m512i_(ones_bf16x32));
        norm_squared_a_f32x16 = _mm512_dpbf16_ps(norm_squared_a_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                 nk_m512bh_from_m512i_(a_grouped_bf16x32));
        norm_squared_b_f32x16 = _mm512_dpbf16_ps(norm_squared_b_f32x16, nk_m512bh_from_m512i_(b_grouped_bf16x32),
                                                 nk_m512bh_from_m512i_(b_grouped_bf16x32));
        product_diagonal_f32x16 = _mm512_dpbf16_ps(product_diagonal_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                   nk_m512bh_from_m512i_(b_grouped_bf16x32));
        product_rotation_1_f32x16 = _mm512_dpbf16_ps(product_rotation_1_f32x16,
                                                     nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                     nk_m512bh_from_m512i_(b_rotation_1_bf16x32));
        product_rotation_2_f32x16 = _mm512_dpbf16_ps(product_rotation_2_f32x16,
                                                     nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                     nk_m512bh_from_m512i_(b_rotation_2_bf16x32));
    }

    if (index < n) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0x3FFFFFFF, (nk_u32_t)((n - index) * 3));
        __m512i a_raw_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (__m512i const *)(a + index * 3));
        __m512i b_raw_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (__m512i const *)(b + index * 3));
        __m512i a_grouped_bf16x32 = _mm512_permutexvar_epi16(idx_channel_group_i16x32, a_raw_bf16x32);
        __m512i b_grouped_bf16x32 = _mm512_permutexvar_epi16(idx_channel_group_i16x32, b_raw_bf16x32);
        __m512i b_rotation_1_bf16x32 = _mm512_permutexvar_epi16(idx_rotation_1_i16x32, b_raw_bf16x32);
        __m512i b_rotation_2_bf16x32 = _mm512_permutexvar_epi16(idx_rotation_2_i16x32, b_raw_bf16x32);

        sum_a_f32x16 = _mm512_dpbf16_ps(sum_a_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                        nk_m512bh_from_m512i_(ones_bf16x32));
        sum_b_f32x16 = _mm512_dpbf16_ps(sum_b_f32x16, nk_m512bh_from_m512i_(b_grouped_bf16x32),
                                        nk_m512bh_from_m512i_(ones_bf16x32));
        norm_squared_a_f32x16 = _mm512_dpbf16_ps(norm_squared_a_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                 nk_m512bh_from_m512i_(a_grouped_bf16x32));
        norm_squared_b_f32x16 = _mm512_dpbf16_ps(norm_squared_b_f32x16, nk_m512bh_from_m512i_(b_grouped_bf16x32),
                                                 nk_m512bh_from_m512i_(b_grouped_bf16x32));
        product_diagonal_f32x16 = _mm512_dpbf16_ps(product_diagonal_f32x16, nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                   nk_m512bh_from_m512i_(b_grouped_bf16x32));
        product_rotation_1_f32x16 = _mm512_dpbf16_ps(product_rotation_1_f32x16,
                                                     nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                     nk_m512bh_from_m512i_(b_rotation_1_bf16x32));
        product_rotation_2_f32x16 = _mm512_dpbf16_ps(product_rotation_2_f32x16,
                                                     nk_m512bh_from_m512i_(a_grouped_bf16x32),
                                                     nk_m512bh_from_m512i_(b_rotation_2_bf16x32));
    }

    __mmask16 const mask_channel_x_f32 = 0x001F;
    __mmask16 const mask_channel_y_f32 = 0x03E0;
    __mmask16 const mask_channel_z_f32 = 0x7C00;

    nk_f32_t sum_a_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_a_f32x16);
    nk_f32_t sum_a_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_a_f32x16);
    nk_f32_t sum_a_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_a_f32x16);
    nk_f32_t sum_b_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_b_f32x16);
    nk_f32_t sum_b_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_b_f32x16);
    nk_f32_t sum_b_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_b_f32x16);
    nk_f32_t norm_squared_a = _mm512_reduce_add_ps(norm_squared_a_f32x16);
    nk_f32_t norm_squared_b = _mm512_reduce_add_ps(norm_squared_b_f32x16);

    nk_f32_t covariance_x_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_diagonal_f32x16);
    nk_f32_t covariance_y_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_diagonal_f32x16);
    nk_f32_t covariance_z_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_diagonal_f32x16);
    nk_f32_t covariance_x_y = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_y_z = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_x = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_x_z = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_x = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_z_y = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_2_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    nk_f32_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - (nk_f32_t)n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - (nk_f32_t)n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - (nk_f32_t)n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - (nk_f32_t)n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - (nk_f32_t)n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - (nk_f32_t)n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - (nk_f32_t)n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - (nk_f32_t)n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

    // Identity-dominant short-circuit.
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t c;
    nk_f32_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1.0f, optimal_rotation[1] = 0.0f, optimal_rotation[2] = 0.0f;
        optimal_rotation[3] = 0.0f, optimal_rotation[4] = 1.0f, optimal_rotation[5] = 0.0f;
        optimal_rotation[6] = 0.0f, optimal_rotation[7] = 0.0f, optimal_rotation[8] = 1.0f;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        c = centered_norm_squared_a > 0.0f ? trace_rotation_covariance / centered_norm_squared_a : 0.0f;
    }
    else {
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);

        // Scale factor: c = trace(D · S) / ‖a-ā‖², with reflection sign via d3.
        nk_f32_t det = nk_det3x3_f32_(optimal_rotation);
        nk_f32_t d3 = det < 0.0f ? -1.0f : 1.0f;
        nk_f32_t trace_ds = nk_sum_three_products_f32_(svd_diagonal[0], 1.0f, svd_diagonal[4], 1.0f, svd_diagonal[8],
                                                       d3);
        c = centered_norm_squared_a > 0.0f ? trace_ds / centered_norm_squared_a : 0.0f;

        if (det < 0.0f) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
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
    nk_f32_t sum_squared = c * c * centered_norm_squared_a + centered_norm_squared_b -
                           2.0f * c * trace_rotation_covariance;
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

#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X8664_
#endif // NK_MESH_GENOA_H
