/**
 *  @file include/numkong/sets/icelake.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief AVX-512 VNNI/VBMI batched set operations for Ice Lake.
 *
 *  @sa include/numkong/sets.h
 */
#ifndef NUMKONG_SETS_ICELAKE_H
#define NUMKONG_SETS_ICELAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_ICELAKE_

#include "numkong/cast/haswell.h" // `nk_load_b128_haswell_`
#include "numkong/cast/skylake.h" // `nk_partial_load_b32x4_skylake_`
#include "numkong/set/icelake.h"
#include "numkong/dots/serial.h" // `nk_define_cross_normalized_packed_`, `nk_dots_reduce_sum_u1_`
#include "numkong/dots/icelake.h"
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512vpopcntdq,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512vpopcntdq", "f16c", \
                   "fma", "bmi", "bmi2")
#endif

#if NUMKONG_TARGET_ICELAKE

nk_define_cross_normalized_packed_(hamming, u1, icelake, u1x8, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                   nk_dots_packed_u1_icelake, nk_hamming_u32x4_from_dot_icelake_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_packed_(jaccard, u1, icelake, u1x8, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u1_icelake, nk_jaccard_f32x4_from_dot_icelake_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(hamming, u1, icelake, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_icelake, nk_hamming_u32x4_from_dot_icelake_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                                      /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(jaccard, u1, icelake, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_icelake, nk_jaccard_f32x4_from_dot_icelake_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_haswell_, nk_partial_load_b32x4_skylake_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                                      /*dimensions_per_value=*/8)

#endif // NUMKONG_TARGET_ICELAKE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_ICELAKE_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_SETS_ICELAKE_H
