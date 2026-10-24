/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for RISC-V BF16.
 *  @file include/numkong/spatial/rvvbf16.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  Zvfbfwma provides widening bf16 fused multiply-accumulate to f32:
 *    vfwmaccbf16: f32 ← bf16 × bf16
 *
 *  For L2 distance, we use the identity: (a−b)² = a² + b² − 2 × a × b
 *  This allows us to use vfwmaccbf16 for all computations.
 *
 *  Requires: RVV 1.0 + Zvfbfwma extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_SPATIAL_RVVBF16_H
#define NK_SPATIAL_RVVBF16_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVVBF16

#include "numkong/types.h"
#include "numkong/spatial/rvv.h" // `nk_f32_sqrt_rvv`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvfbfwma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvfbfwma")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_sqeuclidean_bf16_rvvbf16(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars,
                                           nk_size_t count_scalars, nk_f32_t *result) {
    // Per-lane accumulators — deferred horizontal reduction
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sq_sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax); // a² + b²
    vfloat32m2_t ab_sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax); // a × b

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)b_scalars, vector_length);
        vbfloat16m1_t a_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(a_u16m1);
        vbfloat16m1_t b_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(b_u16m1);

        // Accumulate a², b², and a×b per-lane (no per-iteration reduction)
        sq_sum_f32m2 = __riscv_vfwmaccbf16_vv_f32m2_tu(sq_sum_f32m2, a_bf16m1, a_bf16m1, vector_length);
        sq_sum_f32m2 = __riscv_vfwmaccbf16_vv_f32m2_tu(sq_sum_f32m2, b_bf16m1, b_bf16m1, vector_length);
        ab_sum_f32m2 = __riscv_vfwmaccbf16_vv_f32m2_tu(ab_sum_f32m2, a_bf16m1, b_bf16m1, vector_length);
    }

    // Single horizontal reduction after the loop
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    nk_f32_t sq_sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sq_sum_f32m2, zero_f32m1, vlmax));
    nk_f32_t ab_sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(ab_sum_f32m2, zero_f32m1, vlmax));
    *result = sq_sum - 2.0f * ab_sum;
}

NK_PUBLIC void nk_euclidean_bf16_rvvbf16(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    nk_sqeuclidean_bf16_rvvbf16(a_scalars, b_scalars, count_scalars, result);
    // Handle potential negative values from floating point errors
    *result = *result > 0.0f ? nk_f32_sqrt_rvv(*result) : 0.0f;
}

NK_PUBLIC void nk_angular_bf16_rvvbf16(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    // Per-lane accumulators — deferred horizontal reduction
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t dot_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t a_sq_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t b_sq_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)b_scalars, vector_length);
        vbfloat16m1_t a_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(a_u16m1);
        vbfloat16m1_t b_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(b_u16m1);

        // dot += a × b
        dot_f32m2 = __riscv_vfwmaccbf16_vv_f32m2_tu(dot_f32m2, a_bf16m1, b_bf16m1, vector_length);
        // a_sq += a × a
        a_sq_f32m2 = __riscv_vfwmaccbf16_vv_f32m2_tu(a_sq_f32m2, a_bf16m1, a_bf16m1, vector_length);
        // b_sq += b × b
        b_sq_f32m2 = __riscv_vfwmaccbf16_vv_f32m2_tu(b_sq_f32m2, b_bf16m1, b_bf16m1, vector_length);
    }

    // Single horizontal reduction after the loop
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    nk_f32_t dot = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(dot_f32m2, zero_f32m1, vlmax));
    nk_f32_t a_sq = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(a_sq_f32m2, zero_f32m1, vlmax));
    nk_f32_t b_sq = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(b_sq_f32m2, zero_f32m1, vlmax));

    // Normalize: 1 − dot / sqrt(‖a‖² × ‖b‖²)
    if (a_sq == 0.0f && b_sq == 0.0f) { *result = 0.0f; }
    else if (dot == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot * nk_f32_rsqrt_rvv(a_sq) * nk_f32_rsqrt_rvv(b_sq);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVVBF16
#endif // NK_TARGET_RISCV_
#endif // NK_SPATIAL_RVVBF16_H
