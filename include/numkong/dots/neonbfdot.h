/**
 *  @brief SIMD-accelerated Batched Dot Products for NEON BF16.
 *  @file include/numkong/dots/neonbfdot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h
 */
#ifndef NK_DOTS_NEONBFDOT_H
#define NK_DOTS_NEONBFDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONBFDOT

#include "numkong/dot/neonbfdot.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16+fp16")
#endif

/* BF16 GEMM: depth_simd_dimensions=8 (8 bf16s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, bf16, neonbfdot, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, bf16, neonbfdot, bf16, bf16, nk_b128_vec_t, nk_load_b128_neon_,
                      nk_partial_load_b16x8_serial_, nk_store_b128_neon_, nk_partial_store_b16x8_serial_,
                      /*simd_width=*/8, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_bf16_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, neonbfdot, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_neonbfdot_t,
                           nk_b128_vec_t, nk_dot_bf16x8_init_neonbfdot, nk_load_b128_neon_,
                           nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_neonbfdot,
                           nk_dot_bf16x8_finalize_neonbfdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, neonbfdot, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_neonbfdot_t,
                        nk_b128_vec_t, nk_dot_bf16x8_init_neonbfdot, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                        nk_load_b128_neon_, nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_neonbfdot,
                        nk_dot_bf16x8_finalize_neonbfdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8,
                        /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM64_
#endif // NK_DOTS_NEONBFDOT_H
