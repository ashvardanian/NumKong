/**
 *  @brief Batched Set Operations for Serial (non-SIMD) Backends.
 *  @file include/numkong/sets/serial.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/sets.h
 */
#ifndef NK_SETS_SERIAL_H
#define NK_SETS_SERIAL_H

#include "numkong/set/serial.h"
#include "numkong/dots/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

nk_define_cross_normalized_packed_(hamming, u1, serial, u1x8, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                   nk_dots_packed_u1_serial, nk_hamming_u32x4_from_dot_serial_, nk_dots_reduce_sum_u1_,
                                   nk_load_b128_serial_, nk_partial_load_b32x4_serial_, nk_store_b128_serial_,
                                   nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_packed_(jaccard, u1, serial, u1x8, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u1_serial, nk_jaccard_f32x4_from_dot_serial_, nk_dots_reduce_sum_u1_,
                                   nk_load_b128_serial_, nk_partial_load_b32x4_serial_, nk_store_b128_serial_,
                                   nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(hamming, u1, serial, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_serial, nk_hamming_u32x4_from_dot_serial_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(jaccard, u1, serial, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_serial, nk_jaccard_f32x4_from_dot_serial_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SETS_SERIAL_H
