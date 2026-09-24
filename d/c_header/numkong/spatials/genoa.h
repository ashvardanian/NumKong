/**
 *  @brief Batched Spatial Distances for Genoa (AVX-512 BF16).
 *  @file include/numkong/spatials/genoa.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_GENOA_H
#define NK_SPATIALS_GENOA_H

#if NK_TARGET_X86_
#if NK_TARGET_GENOA

#include "numkong/spatial/haswell.h"
#include "numkong/dots/genoa.h"

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

nk_define_cross_normalized_packed_(angular, bf16, genoa, bf16, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_genoa, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, genoa, bf16, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_genoa, nk_euclidean_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, genoa, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_genoa, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, genoa, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_genoa, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, e4m3, genoa, e4m3, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_genoa, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, e4m3, genoa, e4m3, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_genoa, nk_euclidean_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, e4m3, genoa, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_genoa, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e4m3, genoa, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_genoa, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)

nk_define_cross_normalized_packed_(angular, e5m2, genoa, e5m2, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_genoa, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_packed_(euclidean, e5m2, genoa, e5m2, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_genoa, nk_euclidean_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(angular, e5m2, genoa, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_genoa, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e5m2, genoa, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_genoa, nk_euclidean_through_f32_from_dot_haswell_,
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

#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_
#endif // NK_SPATIALS_GENOA_H
