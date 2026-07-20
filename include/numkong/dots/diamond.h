/**
 *  @brief SIMD-accelerated Batched Dot Products for Diamond Rapids.
 *  @file include/numkong/dots/diamond.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses VCVTHF82PH/VCVTBF82PH for native FP8→FP16 conversion, then VDPPHPS for
 *  FP16-pair dot products accumulating into FP32. Processes 32 FP8 elements per iteration.
 */
#ifndef NK_DOTS_DIAMOND_H
#define NK_DOTS_DIAMOND_H

#if NK_TARGET_X8664_
#if NK_TARGET_DIAMOND

#include "numkong/dot/diamond.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                    \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2-512,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx10.2-512", "f16c", "fma", \
                   "bmi", "bmi2")
#endif

/* E4M3 GEMM: depth_simd_dimensions=32 (32 e4m3s = 32 bytes), FP16 intermediate, F32 accumulator */
nk_define_cross_pack_size_(dots, e4m3, diamond, e4m3, e4m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, e4m3, diamond)
    nk_define_cross_pack_(dots, e4m3, diamond, e4m3, e4m3, nk_b512_vec_t, nk_load_b512_skylake_,
                          nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                          /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                          /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e4m3, diamond, e4m3, f32, nk_b512_vec_t, nk_dot_through_f16_state_diamond_t_,
                           nk_b128_vec_t, nk_dot_through_f16_init_diamond_, nk_load_e4m3x32_to_f16x32_diamond_,
                           nk_partial_load_e4m3x32_to_f16x32_diamond_, nk_dot_through_f16_update_diamond_,
                           nk_dot_through_f16_finalize_diamond_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, diamond, e4m3, e4m3, f32, nk_b512_vec_t, nk_dot_through_f16_state_diamond_t_,
                        nk_b128_vec_t, nk_dot_through_f16_init_diamond_, nk_load_e4m3x32_to_f16x32_diamond_,
                        nk_partial_load_e4m3x32_to_f16x32_diamond_, nk_load_e4m3x32_to_f16x32_diamond_,
                        nk_partial_load_e4m3x32_to_f16x32_diamond_, nk_dot_through_f16_update_diamond_,
                        nk_dot_through_f16_finalize_diamond_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=32 (32 e5m2s = 32 bytes), FP16 intermediate, F32 accumulator */
nk_define_cross_pack_size_(dots, e5m2, diamond, e5m2, e5m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, e5m2, diamond)
    nk_define_cross_pack_(dots, e5m2, diamond, e5m2, e5m2, nk_b512_vec_t, nk_load_b512_skylake_,
                          nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                          /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                          /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e5m2, diamond, e5m2, f32, nk_b512_vec_t, nk_dot_through_f16_state_diamond_t_,
                           nk_b128_vec_t, nk_dot_through_f16_init_diamond_, nk_load_e5m2x32_to_f16x32_diamond_,
                           nk_partial_load_e5m2x32_to_f16x32_diamond_, nk_dot_through_f16_update_diamond_,
                           nk_dot_through_f16_finalize_diamond_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, diamond, e5m2, e5m2, f32, nk_b512_vec_t, nk_dot_through_f16_state_diamond_t_,
                        nk_b128_vec_t, nk_dot_through_f16_init_diamond_, nk_load_e5m2x32_to_f16x32_diamond_,
                        nk_partial_load_e5m2x32_to_f16x32_diamond_, nk_load_e5m2x32_to_f16x32_diamond_,
                        nk_partial_load_e5m2x32_to_f16x32_diamond_, nk_dot_through_f16_update_diamond_,
                        nk_dot_through_f16_finalize_diamond_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_DIAMOND
#endif // NK_TARGET_X8664_
#endif // NK_DOTS_DIAMOND_H
