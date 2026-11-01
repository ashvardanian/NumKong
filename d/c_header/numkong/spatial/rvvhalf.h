/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for RISC-V FP16.
 *  @file include/numkong/spatial/rvvhalf.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  Zvfh provides native half-precision (f16) vector operations.
 *  Uses widening operations (f16 → f32) for precision accumulation.
 *
 *  Requires: RVV 1.0 + Zvfh extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_SPATIAL_RVVHALF_H
#define NK_SPATIAL_RVVHALF_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVVHALF

#include "numkong/types.h"
#include "numkong/spatial/rvv.h" // `nk_f32_sqrt_rvv`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvfh"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvfh")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_sqeuclidean_f16_rvvhalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                          nk_f32_t *result) {
    // Per-lane accumulator — deferred horizontal reduction
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)b_scalars, vector_length);
        vfloat16m1_t a_f16m1 = __riscv_vreinterpret_v_u16m1_f16m1(a_u16m1);
        vfloat16m1_t b_f16m1 = __riscv_vreinterpret_v_u16m1_f16m1(b_u16m1);
        // Upcast to f32 before subtraction to avoid catastrophic cancellation in f16
        vfloat32m2_t a_f32m2 = __riscv_vfwcvt_f_f_v_f32m2(a_f16m1, vector_length);
        vfloat32m2_t b_f32m2 = __riscv_vfwcvt_f_f_v_f32m2(b_f16m1, vector_length);
        vfloat32m2_t diff_f32m2 = __riscv_vfsub_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        // Accumulate diff² in f32
        sum_f32m2 = __riscv_vfmacc_vv_f32m2_tu(sum_f32m2, diff_f32m2, diff_f32m2, vector_length);
    }

    // Single horizontal reduction after the loop
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_euclidean_f16_rvvhalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f32_t *result) {
    nk_sqeuclidean_f16_rvvhalf(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_rvv(*result);
}

NK_PUBLIC void nk_angular_f16_rvvhalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
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
        vfloat16m1_t a_f16m1 = __riscv_vreinterpret_v_u16m1_f16m1(a_u16m1);
        vfloat16m1_t b_f16m1 = __riscv_vreinterpret_v_u16m1_f16m1(b_u16m1);

        // dot += a × b (widened to f32)
        dot_f32m2 = __riscv_vfwmacc_vv_f32m2_tu(dot_f32m2, a_f16m1, b_f16m1, vector_length);
        // a_sq += a × a
        a_sq_f32m2 = __riscv_vfwmacc_vv_f32m2_tu(a_sq_f32m2, a_f16m1, a_f16m1, vector_length);
        // b_sq += b × b
        b_sq_f32m2 = __riscv_vfwmacc_vv_f32m2_tu(b_sq_f32m2, b_f16m1, b_f16m1, vector_length);
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

#endif // NK_TARGET_RVVHALF
#endif // NK_TARGET_RISCV_
#endif // NK_SPATIAL_RVVHALF_H
