/**
 *  @brief SIMD-accelerated Batched Dot Products for NEON FP16.
 *  @file include/numkong/dots/neonhalf.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h
 */
#ifndef NK_DOTS_NEONHALF_H
#define NK_DOTS_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/dot/neonhalf.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

/* F16 GEMM: depth_simd_dimensions=4 (4 f16s = 8 bytes = 64-bit input for direct f32 conversion) */
nk_define_cross_pack_size_(dots, f16, neonhalf, f16, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/4,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, neonhalf, f16, f16, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_f16_, /*depth_simd_dimensions=*/4,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, neonhalf, f16, f32, nk_b64_vec_t, nk_dot_f16x4_state_neonhalf_t, nk_b128_vec_t,
                           nk_dot_f16x4_init_neonhalf, nk_load_b64_neon_, nk_partial_load_b16x4_serial_,
                           nk_dot_f16x4_update_neonhalf, nk_dot_f16x4_finalize_neonhalf, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, neonhalf, f16, f16, f32, nk_b64_vec_t, nk_dot_f16x4_state_neonhalf_t, nk_b128_vec_t,
                        nk_dot_f16x4_init_neonhalf, nk_load_b64_neon_, nk_partial_load_b16x4_serial_, nk_load_b64_neon_,
                        nk_partial_load_b16x4_serial_, nk_dot_f16x4_update_neonhalf, nk_dot_f16x4_finalize_neonhalf,
                        nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_
#endif // NK_DOTS_NEONHALF_H
