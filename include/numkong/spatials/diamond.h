/**
 *  @file include/numkong/spatials/diamond.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *  @brief SIMD-accelerated batched spatial distances for Diamond Rapids.
 *
 *  @sa include/numkong/spatials.h
 *
 *  Uses VCVTHF82PH/VCVTBF82PH + VDPPHPS via the batched dots/ infrastructure, then derives angular
 *  and Euclidean distances.
 */
#ifndef NUMKONG_SPATIALS_DIAMOND_H
#define NUMKONG_SPATIALS_DIAMOND_H

#if NUMKONG_ARCH_X86_64_
#if NUMKONG_TARGET_DIAMOND

#include "numkong/dots/diamond.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__) && __clang_major__ >= 22
#pragma clang attribute push(                                                                                \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                    \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2-512,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx10.2", "f16c", "fma", \
                   "bmi", "bmi2")
#endif

nk_define_cross_normalized_packed_(angular, e4m3, diamond, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_diamond, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, e4m3, diamond, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e4m3_diamond,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e4m3_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, e4m3, diamond, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_diamond, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e4m3, diamond, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_diamond, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, e5m2, diamond, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_diamond, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, e5m2, diamond, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e5m2_diamond,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e5m2_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, e5m2, diamond, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_diamond, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e5m2, diamond, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_diamond, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_DIAMOND
#endif // NUMKONG_ARCH_X86_64_
#endif // NUMKONG_SPATIALS_DIAMOND_H
