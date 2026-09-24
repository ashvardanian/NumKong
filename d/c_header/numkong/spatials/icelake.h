/**
 *  @brief Batched Spatial Distances for Ice Lake (AVX-512 VNNI/VBMI).
 *  @file include/numkong/spatials/icelake.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_ICELAKE_H
#define NK_SPATIALS_ICELAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICELAKE

#include "numkong/spatial/haswell.h"
#include "numkong/dots/icelake.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                        \
    __attribute__((                                                                                                  \
        target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512vbmi,avx512vpopcntdq,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512vbmi", \
                   "avx512vpopcntdq", "f16c", "fma", "bmi", "bmi2")
#endif

nk_define_cross_normalized_packed_(angular, i8, icelake, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_icelake, nk_angular_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, i8, icelake, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_icelake, nk_euclidean_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, icelake, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_icelake, nk_angular_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, icelake, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_icelake, nk_euclidean_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, u8, icelake, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_icelake, nk_angular_through_u32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, u8, icelake, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_icelake, nk_euclidean_through_u32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, icelake, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_icelake, nk_angular_through_u32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, icelake, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_icelake, nk_euclidean_through_u32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, i4, icelake, i4x2, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i4_icelake, nk_angular_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i4_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 2)
nk_define_cross_normalized_packed_(euclidean, i4, icelake, i4x2, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i4_icelake, nk_euclidean_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i4_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 2)
nk_define_cross_normalized_symmetric_(angular, i4, icelake, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i4_icelake, nk_angular_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i4_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 2)
nk_define_cross_normalized_symmetric_(euclidean, i4, icelake, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i4_icelake, nk_euclidean_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i4_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 2)

nk_define_cross_normalized_packed_(angular, u4, icelake, u4x2, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u4_icelake, nk_angular_through_u32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_u4_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 2)
nk_define_cross_normalized_packed_(euclidean, u4, icelake, u4x2, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u4_icelake, nk_euclidean_through_u32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_u4_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 2)
nk_define_cross_normalized_symmetric_(angular, u4, icelake, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u4_icelake, nk_angular_through_u32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_u4_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 2)
nk_define_cross_normalized_symmetric_(euclidean, u4, icelake, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u4_icelake, nk_euclidean_through_u32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_u4_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 2)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X86_
#endif // NK_SPATIALS_ICELAKE_H
