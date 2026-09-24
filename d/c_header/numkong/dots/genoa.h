/**
 *  @brief SIMD-accelerated Batched Dot Products for Genoa.
 *  @file include/numkong/dots/genoa.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h
 */
#ifndef NK_DOTS_GENOA_H
#define NK_DOTS_GENOA_H

#if NK_TARGET_X86_
#if NK_TARGET_GENOA

#include "numkong/dot/genoa.h"
#include "numkong/cast/icelake.h" // `nk_load_e4m3x32_to_bf16x32_icelake_`

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

/* BF16 GEMM: depth_simd_dimensions=32 (32 bf16s = 64 bytes = 1 cache line) */
nk_define_cross_pack_size_(dots, bf16, genoa, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, bf16, genoa, bf16, bf16, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_bf16_, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, genoa, bf16, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_,
                           nk_b128_vec_t, nk_dot_through_bf16_init_genoa_, nk_load_b512_skylake_,
                           nk_partial_load_b16x32_skylake_, nk_dot_through_bf16_update_genoa_,
                           nk_dot_through_bf16_finalize_genoa_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, genoa, bf16, bf16, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_,
                        nk_b128_vec_t, nk_dot_through_bf16_init_genoa_, nk_load_b512_skylake_,
                        nk_partial_load_b16x32_skylake_, nk_load_b512_skylake_, nk_partial_load_b16x32_skylake_,
                        nk_dot_through_bf16_update_genoa_, nk_dot_through_bf16_finalize_genoa_, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* E4M3 GEMM: depth_simd_dimensions=32 (32 e4m3s = 32 bytes = half cache line), F32 accumulator */
nk_define_cross_pack_size_(dots, e4m3, genoa, e4m3, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e4m3, genoa, e4m3, bf16, nk_e4m3_to_bf16, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e4m3_, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e4m3, genoa, e4m3, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_,
                           nk_b128_vec_t, nk_dot_through_bf16_init_genoa_, nk_load_e4m3x32_to_bf16x32_icelake_,
                           nk_partial_load_e4m3x32_to_bf16x32_icelake_, nk_dot_through_bf16_update_genoa_,
                           nk_dot_through_bf16_finalize_genoa_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, genoa, e4m3, bf16, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_,
                        nk_b128_vec_t, nk_dot_through_bf16_init_genoa_, nk_load_e4m3x32_to_bf16x32_icelake_,
                        nk_partial_load_e4m3x32_to_bf16x32_icelake_, nk_load_b512_skylake_,
                        nk_partial_load_b16x32_skylake_, nk_dot_through_bf16_update_genoa_,
                        nk_dot_through_bf16_finalize_genoa_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=32 (32 e5m2s = 32 bytes = half cache line), F32 accumulator */
nk_define_cross_pack_size_(dots, e5m2, genoa, e5m2, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e5m2, genoa, e5m2, bf16, nk_e5m2_to_bf16, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e5m2_, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e5m2, genoa, e5m2, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_,
                           nk_b128_vec_t, nk_dot_through_bf16_init_genoa_, nk_load_e5m2x32_to_bf16x32_icelake_,
                           nk_partial_load_e5m2x32_to_bf16x32_icelake_, nk_dot_through_bf16_update_genoa_,
                           nk_dot_through_bf16_finalize_genoa_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, genoa, e5m2, bf16, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_,
                        nk_b128_vec_t, nk_dot_through_bf16_init_genoa_, nk_load_e5m2x32_to_bf16x32_icelake_,
                        nk_partial_load_e5m2x32_to_bf16x32_icelake_, nk_load_b512_skylake_,
                        nk_partial_load_b16x32_skylake_, nk_dot_through_bf16_update_genoa_,
                        nk_dot_through_bf16_finalize_genoa_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_
#endif // NK_DOTS_GENOA_H
