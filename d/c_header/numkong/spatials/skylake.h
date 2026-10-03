/**
 *  @brief Batched Spatial Distances for Skylake (AVX-512).
 *  @file include/numkong/spatials/skylake.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_SKYLAKE_H
#define NK_SPATIALS_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/spatial/skylake.h"
#include "numkong/spatial/haswell.h"
#include "numkong/dots/skylake.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

nk_define_cross_normalized_packed_(angular, f64, skylake, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_skylake, nk_angular_f64x4_from_dot_skylake_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_haswell_, nk_partial_load_b64x4_skylake_,
                                   nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, f64, skylake, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_skylake, nk_euclidean_f64x4_from_dot_skylake_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_haswell_, nk_partial_load_b64x4_skylake_,
                                   nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, f64, skylake, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_skylake, nk_angular_f64x4_from_dot_skylake_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_haswell_, nk_partial_load_b64x4_skylake_,
                                      nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f64, skylake, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_skylake, nk_euclidean_f64x4_from_dot_skylake_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_haswell_, nk_partial_load_b64x4_skylake_,
                                      nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, f32, skylake, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_skylake, nk_angular_f64x4_from_dot_skylake_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_haswell_, nk_partial_load_b64x4_skylake_,
                                   nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, f32, skylake, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_skylake, nk_euclidean_f64x4_from_dot_skylake_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_haswell_, nk_partial_load_b64x4_skylake_,
                                   nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, f32, skylake, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_skylake, nk_angular_f64x4_from_dot_skylake_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_haswell_, nk_partial_load_b64x4_skylake_,
                                      nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f32, skylake, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_skylake, nk_euclidean_f64x4_from_dot_skylake_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_haswell_, nk_partial_load_b64x4_skylake_,
                                      nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, f16, skylake, f16, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_skylake, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, skylake, f16, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_skylake, nk_euclidean_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, skylake, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_skylake, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, skylake, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_skylake, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, bf16, skylake, bf16, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_skylake, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, skylake, bf16, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_skylake,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_bf16_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, skylake, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_skylake, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, skylake, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_skylake, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, e4m3, skylake, e4m3, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_skylake, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, e4m3, skylake, e4m3, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e4m3_skylake,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e4m3_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, e4m3, skylake, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_skylake, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e4m3, skylake, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_skylake, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, e5m2, skylake, e5m2, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_skylake, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, e5m2, skylake, e5m2, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e5m2_skylake,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e5m2_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, e5m2, skylake, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_skylake, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e5m2, skylake, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_skylake, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, e2m3, skylake, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m3_skylake, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, e2m3, skylake, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m3_skylake,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e2m3_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, e2m3, skylake, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_skylake, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e2m3, skylake, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_skylake, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, e3m2, skylake, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e3m2_skylake, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e3m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, e3m2, skylake, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e3m2_skylake,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e3m2_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, e3m2, skylake, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_skylake, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e3m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e3m2, skylake, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_skylake, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e3m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_SPATIALS_SKYLAKE_H
