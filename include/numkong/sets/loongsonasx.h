/**
 *  @file include/numkong/sets/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 25, 2026
 *  @brief 256-bit LASX batched set operations for LoongArch.
 *
 *  @sa include/numkong/sets.h
 */
#ifndef NUMKONG_SETS_LOONGSONASX_H
#define NUMKONG_SETS_LOONGSONASX_H

#if NUMKONG_ARCH_LOONGARCH64_
#if NUMKONG_TARGET_LOONGSONASX

#include "numkong/set/loongsonasx.h"
#include "numkong/dots/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

nk_define_cross_normalized_packed_(hamming, u1, loongsonasx, u1x8, u1x8, u32, /*norm_value_type=*/u32, u32,
                                   nk_b128_vec_t, nk_dots_packed_u1_loongsonasx, nk_hamming_u32x4_from_dot_loongsonasx_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_,
                                   /*dimensions_per_value=*/8)

nk_define_cross_normalized_packed_(jaccard, u1, loongsonasx, u1x8, u1x8, u32, /*norm_value_type=*/u32, f32,
                                   nk_b128_vec_t, nk_dots_packed_u1_loongsonasx, nk_jaccard_f32x4_from_dot_loongsonasx_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_,
                                   /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(hamming, u1, loongsonasx, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_loongsonasx, nk_hamming_u32x4_from_dot_loongsonasx_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_,
                                      /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(jaccard, u1, loongsonasx, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_loongsonasx, nk_jaccard_f32x4_from_dot_loongsonasx_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_,
                                      /*dimensions_per_value=*/8)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_LOONGSONASX
#endif // NUMKONG_ARCH_LOONGARCH64_
#endif // NUMKONG_SETS_LOONGSONASX_H
