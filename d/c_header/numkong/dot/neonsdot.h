/**
 *  @brief SIMD-accelerated Dot Products for NEON SDOT.
 *  @file include/numkong/dot/neonsdot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_neonsdot_instructions ARM NEON SDOT/UDOT Instructions (ARMv8.4-DotProd)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vdotq_s32                   SDOT (V.4S, V.16B, V.16B)       3cy         2/cy        4/cy
 *      vdotq_u32                   UDOT (V.4S, V.16B, V.16B)       3cy         2/cy        4/cy
 *      vld1q_s8                    LD1 (V.16B)                     4cy         2/cy        3/cy
 *      vld1q_u8                    LD1 (V.16B)                     4cy         2/cy        3/cy
 *      vaddvq_s32                  ADDV (V.4S)                     4cy         1/cy        2/cy
 *      vaddvq_u32                  ADDV (V.4S)                     4cy         1/cy        2/cy
 *
 *  The ARMv8.4-DotProd extension provides SDOT/UDOT instructions critical for int8 quantized ML
 *  inference. Each instruction computes four dot products of 4-element int8 vectors, accumulating
 *  into int32 lanes, processing 16 multiply-accumulates per instruction.
 *
 *  SDOT handles signed int8 operands while UDOT handles unsigned. The 3-cycle latency with 2/cy
 *  throughput on A76 (4/cy on newer cores) enables int8 matrix multiplication for
 *  quantized neural network inference, where 8-bit weights reduce memory bandwidth by 4x vs FP32.
 *
 *  @section dot_neonsdot_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_i8x16 for 8-bit signed integer inputs using SDOT,
 *  - nk_dot_u8x16 for 8-bit unsigned integer inputs using UDOT,
 *  - nk_dot_i4x32 for 4-bit signed integer products,
 *  - nk_dot_u4x32 for 4-bit unsigned integer products.
 *
 *  @code{c}
 *  nk_dot_i8x16_state_neonsdot_t state_first, state_second, state_third, state_fourth;
 *  int8x16_t query_i8x16, target_first_i8x16, target_second_i8x16, target_third_i8x16, target_fourth_i8x16;
 *  nk_dot_i8x16_init_neonsdot(&state_first);
 *  nk_dot_i8x16_init_neonsdot(&state_second);
 *  nk_dot_i8x16_init_neonsdot(&state_third);
 *  nk_dot_i8x16_init_neonsdot(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 16 <= depth; idx += 16) {
 *      query_i8x16 = vld1q_s8(query_ptr + idx);
 *      target_first_i8x16 = vld1q_s8(target_first_ptr + idx);
 *      target_second_i8x16 = vld1q_s8(target_second_ptr + idx);
 *      target_third_i8x16 = vld1q_s8(target_third_ptr + idx);
 *      target_fourth_i8x16 = vld1q_s8(target_fourth_ptr + idx);
 *      nk_dot_i8x16_update_neonsdot(&state_first, query_i8x16, target_first_i8x16, idx, 16);
 *      nk_dot_i8x16_update_neonsdot(&state_second, query_i8x16, target_second_i8x16, idx, 16);
 *      nk_dot_i8x16_update_neonsdot(&state_third, query_i8x16, target_third_i8x16, idx, 16);
 *      nk_dot_i8x16_update_neonsdot(&state_fourth, query_i8x16, target_fourth_i8x16, idx, 16);
 *  }
 *  int32x4_t results_i32x4;
 *  nk_dot_i8x16_finalize_neonsdot(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 *
 *  For 4-bit integers, the state manages unpacking and accumulation:
 *
 *  @code{c}
 *  nk_dot_i4x32_state_neonsdot_t state_first, state_second, state_third, state_fourth;
 *  uint8x8_t query_packed, target_first_packed, target_second_packed, target_third_packed, target_fourth_packed;
 *  nk_dot_i4x32_init_neonsdot(&state_first);
 *  nk_dot_i4x32_init_neonsdot(&state_second);
 *  nk_dot_i4x32_init_neonsdot(&state_third);
 *  nk_dot_i4x32_init_neonsdot(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 16 <= depth; idx += 16) {
 *      query_packed = vld1_u8(query_ptr + idx / 2);
 *      target_first_packed = vld1_u8(target_first_ptr + idx / 2);
 *      target_second_packed = vld1_u8(target_second_ptr + idx / 2);
 *      target_third_packed = vld1_u8(target_third_ptr + idx / 2);
 *      target_fourth_packed = vld1_u8(target_fourth_ptr + idx / 2);
 *      nk_dot_i4x32_update_neonsdot(&state_first, query_packed, target_first_packed, idx, 16);
 *      nk_dot_i4x32_update_neonsdot(&state_second, query_packed, target_second_packed, idx, 16);
 *      nk_dot_i4x32_update_neonsdot(&state_third, query_packed, target_third_packed, idx, 16);
 *      nk_dot_i4x32_update_neonsdot(&state_fourth, query_packed, target_fourth_packed, idx, 16);
 *  }
 *  int32x4_t results_i32x4;
 *  nk_dot_i4x32_finalize_neonsdot(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 */
#ifndef NK_DOT_NEONSDOT_H
#define NK_DOT_NEONSDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONSDOT

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

NK_PUBLIC void nk_dot_i8_neonsdot(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                  nk_i32_t *result) {
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count_scalars; idx_scalars += 16) {
        int8x16_t a_i8x16 = vld1q_s8(a_scalars + idx_scalars);
        int8x16_t b_i8x16 = vld1q_s8(b_scalars + idx_scalars);
        sum_i32x4 = vdotq_s32(sum_i32x4, a_i8x16, b_i8x16);
    }
    nk_i32_t sum = vaddvq_s32(sum_i32x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_u8_neonsdot(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                  nk_u32_t *result) {
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count_scalars; idx_scalars += 16) {
        uint8x16_t a_u8x16 = vld1q_u8(a_scalars + idx_scalars);
        uint8x16_t b_u8x16 = vld1q_u8(b_scalars + idx_scalars);
        sum_u32x4 = vdotq_u32(sum_u32x4, a_u8x16, b_u8x16);
    }
    nk_u32_t sum = vaddvq_u32(sum_u32x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

/**
 *  @brief Running state for 128-bit dot accumulation over i8 scalars on NEON.
 */
typedef struct nk_dot_i8x16_state_neonsdot_t {
    int32x4_t sum_i32x4;
} nk_dot_i8x16_state_neonsdot_t;

NK_INTERNAL void nk_dot_i8x16_init_neonsdot(nk_dot_i8x16_state_neonsdot_t *state) { state->sum_i32x4 = vdupq_n_s32(0); }

NK_INTERNAL void nk_dot_i8x16_update_neonsdot(nk_dot_i8x16_state_neonsdot_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    int32x4_t sum_i32x4 = state->sum_i32x4;
    sum_i32x4 = vdotq_s32(sum_i32x4, vreinterpretq_s8_u32(a.u32x4), vreinterpretq_s8_u32(b.u32x4));
    state->sum_i32x4 = sum_i32x4;
}

NK_INTERNAL void nk_dot_i8x16_finalize_neonsdot(                                                //
    nk_dot_i8x16_state_neonsdot_t const *state_a, nk_dot_i8x16_state_neonsdot_t const *state_b, //
    nk_dot_i8x16_state_neonsdot_t const *state_c, nk_dot_i8x16_state_neonsdot_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->i32s[0] = vaddvq_s32(state_a->sum_i32x4);
    result->i32s[1] = vaddvq_s32(state_b->sum_i32x4);
    result->i32s[2] = vaddvq_s32(state_c->sum_i32x4);
    result->i32s[3] = vaddvq_s32(state_d->sum_i32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over u8 scalars on NEON.
 */
typedef struct nk_dot_u8x16_state_neonsdot_t {
    uint32x4_t sum_u32x4;
} nk_dot_u8x16_state_neonsdot_t;

NK_INTERNAL void nk_dot_u8x16_init_neonsdot(nk_dot_u8x16_state_neonsdot_t *state) { state->sum_u32x4 = vdupq_n_u32(0); }

NK_INTERNAL void nk_dot_u8x16_update_neonsdot(nk_dot_u8x16_state_neonsdot_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    uint32x4_t sum_u32x4 = state->sum_u32x4;
    sum_u32x4 = vdotq_u32(sum_u32x4, vreinterpretq_u8_u32(a.u32x4), vreinterpretq_u8_u32(b.u32x4));
    state->sum_u32x4 = sum_u32x4;
}

NK_INTERNAL void nk_dot_u8x16_finalize_neonsdot(                                                //
    nk_dot_u8x16_state_neonsdot_t const *state_a, nk_dot_u8x16_state_neonsdot_t const *state_b, //
    nk_dot_u8x16_state_neonsdot_t const *state_c, nk_dot_u8x16_state_neonsdot_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->u32s[0] = vaddvq_u32(state_a->sum_u32x4);
    result->u32s[1] = vaddvq_u32(state_b->sum_u32x4);
    result->u32s[2] = vaddvq_u32(state_c->sum_u32x4);
    result->u32s[3] = vaddvq_u32(state_d->sum_u32x4);
}

NK_PUBLIC void nk_dot_i4_neonsdot(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    //
    // ARM NEON SDOT handles signed×signed directly, so we use direct sign-extension:
    // Extract nibbles [0,15], sign-extend to i8 [-8,7] via shift trick, then SDOT.
    // No algebraic correction needed unlike x86 DPBUSD.
    //
    n = nk_size_round_up_to_multiple_(n, 2);
    nk_size_t n_bytes = n / 2;
    uint8x16_t const nibble_mask_u8x16 = vdupq_n_u8(0x0F);
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    uint8x16_t a_i4x32, b_i4x32;

nk_dot_i4_neonsdot_cycle:
    if (n_bytes < 16) {
        // Partial load for tail handling
        nk_b128_vec_t a_vec = {0}, b_vec = {0};
        nk_u8_t const *a_ptr = (nk_u8_t const *)a;
        nk_u8_t const *b_ptr = (nk_u8_t const *)b;
        for (nk_size_t i = 0; i < n_bytes; i++) {
            a_vec.u8s[i] = a_ptr[i];
            b_vec.u8s[i] = b_ptr[i];
        }
        a_i4x32 = a_vec.u8x16;
        b_i4x32 = b_vec.u8x16;
        n_bytes = 0;
    }
    else {
        a_i4x32 = vld1q_u8((nk_u8_t const *)a);
        b_i4x32 = vld1q_u8((nk_u8_t const *)b);
        a += 16, b += 16, n_bytes -= 16;
    }

    // Extract low and high nibbles as unsigned [0,15]
    uint8x16_t a_lo_u8x16 = vandq_u8(a_i4x32, nibble_mask_u8x16);
    uint8x16_t a_hi_u8x16 = vshrq_n_u8(a_i4x32, 4);
    uint8x16_t b_lo_u8x16 = vandq_u8(b_i4x32, nibble_mask_u8x16);
    uint8x16_t b_hi_u8x16 = vshrq_n_u8(b_i4x32, 4);

    // Sign-extend 4-bit to 8-bit: shift left 4, arithmetic shift right 4
    // This converts unsigned [0,15] to signed [-8,7]
    int8x16_t a_lo_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(a_lo_u8x16), 4), 4);
    int8x16_t a_hi_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(a_hi_u8x16), 4), 4);
    int8x16_t b_lo_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(b_lo_u8x16), 4), 4);
    int8x16_t b_hi_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(b_hi_u8x16), 4), 4);

    // SDOT for signed dot product - no correction needed!
    sum_i32x4 = vdotq_s32(sum_i32x4, a_lo_i8x16, b_lo_i8x16);
    sum_i32x4 = vdotq_s32(sum_i32x4, a_hi_i8x16, b_hi_i8x16);

    if (n_bytes) goto nk_dot_i4_neonsdot_cycle;

    *result = vaddvq_s32(sum_i32x4);
}

NK_PUBLIC void nk_dot_u4_neonsdot(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // Values are ∈ [0,15], so UDOT can be used directly.
    //
    n = nk_size_round_up_to_multiple_(n, 2);
    nk_size_t n_bytes = n / 2;
    uint8x16_t const nibble_mask_u8x16 = vdupq_n_u8(0x0F);
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    uint8x16_t a_u4x32, b_u4x32;

nk_dot_u4_neonsdot_cycle:
    if (n_bytes < 16) {
        // Partial load for tail handling
        nk_b128_vec_t a_vec = {0}, b_vec = {0};
        nk_u8_t const *a_ptr = (nk_u8_t const *)a;
        nk_u8_t const *b_ptr = (nk_u8_t const *)b;
        for (nk_size_t i = 0; i < n_bytes; i++) {
            a_vec.u8s[i] = a_ptr[i];
            b_vec.u8s[i] = b_ptr[i];
        }
        a_u4x32 = a_vec.u8x16;
        b_u4x32 = b_vec.u8x16;
        n_bytes = 0;
    }
    else {
        a_u4x32 = vld1q_u8((nk_u8_t const *)a);
        b_u4x32 = vld1q_u8((nk_u8_t const *)b);
        a += 16, b += 16, n_bytes -= 16;
    }

    // Extract low and high nibbles - values in [0,15] work directly with UDOT
    uint8x16_t a_lo_u8x16 = vandq_u8(a_u4x32, nibble_mask_u8x16);
    uint8x16_t a_hi_u8x16 = vshrq_n_u8(a_u4x32, 4);
    uint8x16_t b_lo_u8x16 = vandq_u8(b_u4x32, nibble_mask_u8x16);
    uint8x16_t b_hi_u8x16 = vshrq_n_u8(b_u4x32, 4);

    // UDOT directly on unsigned nibbles
    sum_u32x4 = vdotq_u32(sum_u32x4, a_lo_u8x16, b_lo_u8x16);
    sum_u32x4 = vdotq_u32(sum_u32x4, a_hi_u8x16, b_hi_u8x16);

    if (n_bytes) goto nk_dot_u4_neonsdot_cycle;

    *result = vaddvq_u32(sum_u32x4);
}

typedef struct nk_dot_i4x32_state_neonsdot_t {
    int32x4_t product_sum_i32x4;
} nk_dot_i4x32_state_neonsdot_t;

NK_INTERNAL void nk_dot_i4x32_init_neonsdot(nk_dot_i4x32_state_neonsdot_t *state) {
    state->product_sum_i32x4 = vdupq_n_s32(0);
}

NK_INTERNAL void nk_dot_i4x32_update_neonsdot(nk_dot_i4x32_state_neonsdot_t *state, nk_b128_vec_t a_i4x32,
                                              nk_b128_vec_t b_i4x32, nk_size_t depth_offset,
                                              nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    uint8x16_t const nibble_mask_u8x16 = vdupq_n_u8(0x0F);

    // Extract nibbles as unsigned first
    uint8x16_t a_low_u8x16 = vandq_u8(a_i4x32.u8x16, nibble_mask_u8x16);
    uint8x16_t a_high_u8x16 = vshrq_n_u8(a_i4x32.u8x16, 4);
    uint8x16_t b_low_u8x16 = vandq_u8(b_i4x32.u8x16, nibble_mask_u8x16);
    uint8x16_t b_high_u8x16 = vshrq_n_u8(b_i4x32.u8x16, 4);

    // Sign-extend 4-bit to 8-bit: shift left 4, arithmetic shift right 4
    int8x16_t a_low_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(a_low_u8x16), 4), 4);
    int8x16_t a_high_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(a_high_u8x16), 4), 4);
    int8x16_t b_low_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(b_low_u8x16), 4), 4);
    int8x16_t b_high_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(b_high_u8x16), 4), 4);

    // SDOT for signed dot product - no correction needed!
    int32x4_t product_sum_i32x4 = state->product_sum_i32x4;
    product_sum_i32x4 = vdotq_s32(product_sum_i32x4, a_low_i8x16, b_low_i8x16);
    product_sum_i32x4 = vdotq_s32(product_sum_i32x4, a_high_i8x16, b_high_i8x16);
    state->product_sum_i32x4 = product_sum_i32x4;
}

NK_INTERNAL void nk_dot_i4x32_finalize_neonsdot(                                                //
    nk_dot_i4x32_state_neonsdot_t const *state_a, nk_dot_i4x32_state_neonsdot_t const *state_b, //
    nk_dot_i4x32_state_neonsdot_t const *state_c, nk_dot_i4x32_state_neonsdot_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // Simple reduction - no correction formula needed with sign-extension approach!
    result->i32s[0] = vaddvq_s32(state_a->product_sum_i32x4);
    result->i32s[1] = vaddvq_s32(state_b->product_sum_i32x4);
    result->i32s[2] = vaddvq_s32(state_c->product_sum_i32x4);
    result->i32s[3] = vaddvq_s32(state_d->product_sum_i32x4);
}

typedef struct nk_dot_u4x32_state_neonsdot_t {
    uint32x4_t product_sum_u32x4;
} nk_dot_u4x32_state_neonsdot_t;

NK_INTERNAL void nk_dot_u4x32_init_neonsdot(nk_dot_u4x32_state_neonsdot_t *state) {
    state->product_sum_u32x4 = vdupq_n_u32(0);
}

NK_INTERNAL void nk_dot_u4x32_update_neonsdot(nk_dot_u4x32_state_neonsdot_t *state, nk_b128_vec_t a_u4x32,
                                              nk_b128_vec_t b_u4x32, nk_size_t depth_offset,
                                              nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    uint8x16_t const nibble_mask_u8x16 = vdupq_n_u8(0x0F);

    // Extract nibbles - values in [0,15] work directly with UDOT
    uint8x16_t a_low_u8x16 = vandq_u8(a_u4x32.u8x16, nibble_mask_u8x16);
    uint8x16_t a_high_u8x16 = vshrq_n_u8(a_u4x32.u8x16, 4);
    uint8x16_t b_low_u8x16 = vandq_u8(b_u4x32.u8x16, nibble_mask_u8x16);
    uint8x16_t b_high_u8x16 = vshrq_n_u8(b_u4x32.u8x16, 4);

    // UDOT directly on unsigned nibbles
    uint32x4_t product_sum_u32x4 = state->product_sum_u32x4;
    product_sum_u32x4 = vdotq_u32(product_sum_u32x4, a_low_u8x16, b_low_u8x16);
    product_sum_u32x4 = vdotq_u32(product_sum_u32x4, a_high_u8x16, b_high_u8x16);
    state->product_sum_u32x4 = product_sum_u32x4;
}

NK_INTERNAL void nk_dot_u4x32_finalize_neonsdot(                                                //
    nk_dot_u4x32_state_neonsdot_t const *state_a, nk_dot_u4x32_state_neonsdot_t const *state_b, //
    nk_dot_u4x32_state_neonsdot_t const *state_c, nk_dot_u4x32_state_neonsdot_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // Simple reduction - no correction formula needed!
    result->u32s[0] = vaddvq_u32(state_a->product_sum_u32x4);
    result->u32s[1] = vaddvq_u32(state_b->product_sum_u32x4);
    result->u32s[2] = vaddvq_u32(state_c->product_sum_u32x4);
    result->u32s[3] = vaddvq_u32(state_d->product_sum_u32x4);
}

NK_PUBLIC void nk_dot_e2m3_neonsdot(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    // Integer dot product for e2m3 using SDOT (signed×signed i8 → i32).
    // Every e2m3 value × 16 is an exact integer in [-120, +120], fits signed i8.
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // 32-entry LUT via vqtbl2q_u8 (handles 0-31 indices in one instruction).
    static nk_u8_t const lut_data[32] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,  28,  30,
                                         32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120};
    uint8x16x2_t lut_magnitude_u8x16x2 = vld1q_u8_x2(lut_data);
    uint8x16_t magnitude_mask_u8x16 = vdupq_n_u8(0x1F);
    uint8x16_t sign_mask_u8x16 = vdupq_n_u8(0x20);
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    uint8x16_t a_e2m3_u8x16, b_e2m3_u8x16;

nk_dot_e2m3_neonsdot_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_e2m3_u8x16 = a_vec.u8x16;
        b_e2m3_u8x16 = b_vec.u8x16;
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x16 = vld1q_u8((nk_u8_t const *)a_scalars);
        b_e2m3_u8x16 = vld1q_u8((nk_u8_t const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // Extract 5-bit magnitude indices and LUT lookup
    uint8x16_t a_magnitude_u8x16 = vandq_u8(a_e2m3_u8x16, magnitude_mask_u8x16);
    uint8x16_t b_magnitude_u8x16 = vandq_u8(b_e2m3_u8x16, magnitude_mask_u8x16);
    uint8x16_t a_unsigned_u8x16 = vqtbl2q_u8(lut_magnitude_u8x16x2, a_magnitude_u8x16);
    uint8x16_t b_unsigned_u8x16 = vqtbl2q_u8(lut_magnitude_u8x16x2, b_magnitude_u8x16);

    // Combined sign: (a ^ b) & 0x20 — nonzero means negative product
    uint8x16_t sign_combined_u8x16 = vandq_u8(veorq_u8(a_e2m3_u8x16, b_e2m3_u8x16), sign_mask_u8x16);
    uint8x16_t negate_mask_u8x16 = vceqq_u8(sign_combined_u8x16, sign_mask_u8x16);

    // Negate b where signs differ, keep positive otherwise
    int8x16_t b_signed_i8x16 = vbslq_s8(negate_mask_u8x16, vnegq_s8(vreinterpretq_s8_u8(b_unsigned_u8x16)),
                                        vreinterpretq_s8_u8(b_unsigned_u8x16));

    // SDOT: signed×signed, 4 bytes → i32
    sum_i32x4 = vdotq_s32(sum_i32x4, vreinterpretq_s8_u8(a_unsigned_u8x16), b_signed_i8x16);

    if (count_scalars) goto nk_dot_e2m3_neonsdot_cycle;
    *result = (nk_f32_t)vaddvq_s32(sum_i32x4) / 256.0f;
}

NK_PUBLIC void nk_dot_e3m2_neonsdot(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    // Integer dot product for e3m2 using i16 LUT via vqtbl2q_u8 (low bytes) + comparison (high byte) + SMLAL.
    // Every e3m2 value × 16 is an exact integer, but magnitudes reach 448, requiring i16.
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // The 32-entry magnitude LUT low bytes are looked up via vqtbl2q_u8.
    // High byte is 1 only for indices 28-31 (values 256-448), replaced by a >= 28 comparison.
    static nk_u8_t const lut_data[32] = {0,  1,  2,  3,  4,  5,  6,  7,   8,   10,  12,  14,  16, 20, 24,  28,
                                         32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 0,  64, 128, 192};
    uint8x16x2_t lut = vld1q_u8_x2(lut_data);
    uint8x16_t high_threshold_u8x16 = vdupq_n_u8(28);
    uint8x16_t magnitude_mask_u8x16 = vdupq_n_u8(0x1F);
    uint8x16_t sign_mask_u8x16 = vdupq_n_u8(0x20);
    int32x4_t sum0_i32x4 = vdupq_n_s32(0);
    int32x4_t sum1_i32x4 = vdupq_n_s32(0);
    uint8x16_t a_e3m2_u8x16, b_e3m2_u8x16;

nk_dot_e3m2_neonsdot_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_e3m2_u8x16 = a_vec.u8x16;
        b_e3m2_u8x16 = b_vec.u8x16;
        count_scalars = 0;
    }
    else {
        a_e3m2_u8x16 = vld1q_u8((nk_u8_t const *)a_scalars);
        b_e3m2_u8x16 = vld1q_u8((nk_u8_t const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // Extract 5-bit magnitude indices
    uint8x16_t a_mag_u8x16 = vandq_u8(a_e3m2_u8x16, magnitude_mask_u8x16);
    uint8x16_t b_mag_u8x16 = vandq_u8(b_e3m2_u8x16, magnitude_mask_u8x16);

    // LUT lookup for low bytes; high byte via comparison (1 iff index >= 28)
    uint8x16_t a_lo_u8x16 = vqtbl2q_u8(lut, a_mag_u8x16);
    uint8x16_t b_lo_u8x16 = vqtbl2q_u8(lut, b_mag_u8x16);
    uint8x16_t a_hi_u8x16 = vandq_u8(vcgeq_u8(a_mag_u8x16, high_threshold_u8x16), vdupq_n_u8(1));
    uint8x16_t b_hi_u8x16 = vandq_u8(vcgeq_u8(b_mag_u8x16, high_threshold_u8x16), vdupq_n_u8(1));

    // Combine low and high bytes into i16 via byte interleave (little-endian: low byte first)
    int16x8_t a_unsigned_low_i16x8 = vreinterpretq_s16_u8(vzip1q_u8(a_lo_u8x16, a_hi_u8x16));
    int16x8_t a_unsigned_high_i16x8 = vreinterpretq_s16_u8(vzip2q_u8(a_lo_u8x16, a_hi_u8x16));
    int16x8_t b_unsigned_low_i16x8 = vreinterpretq_s16_u8(vzip1q_u8(b_lo_u8x16, b_hi_u8x16));
    int16x8_t b_unsigned_high_i16x8 = vreinterpretq_s16_u8(vzip2q_u8(b_lo_u8x16, b_hi_u8x16));

    // Combined sign: XOR sign bits, negate only b (saves ~15 ops vs independent negation)
    uint8x16_t sign_combined_u8x16 = vandq_u8(veorq_u8(a_e3m2_u8x16, b_e3m2_u8x16), sign_mask_u8x16);
    uint8x16_t negate_mask_u8x16 = vceqq_u8(sign_combined_u8x16, sign_mask_u8x16);
    uint16x8_t negate_low_u16x8 = vreinterpretq_u16_u8(vzip1q_u8(negate_mask_u8x16, negate_mask_u8x16));
    uint16x8_t negate_high_u16x8 = vreinterpretq_u16_u8(vzip2q_u8(negate_mask_u8x16, negate_mask_u8x16));
    b_unsigned_low_i16x8 = vbslq_s16(negate_low_u16x8, vnegq_s16(b_unsigned_low_i16x8), b_unsigned_low_i16x8);
    b_unsigned_high_i16x8 = vbslq_s16(negate_high_u16x8, vnegq_s16(b_unsigned_high_i16x8), b_unsigned_high_i16x8);

    // Widening multiply-accumulate: i16×i16 → i32
    sum0_i32x4 = vmlal_s16(sum0_i32x4, vget_low_s16(a_unsigned_low_i16x8), vget_low_s16(b_unsigned_low_i16x8));
    sum0_i32x4 = vmlal_high_s16(sum0_i32x4, a_unsigned_low_i16x8, b_unsigned_low_i16x8);
    sum1_i32x4 = vmlal_s16(sum1_i32x4, vget_low_s16(a_unsigned_high_i16x8), vget_low_s16(b_unsigned_high_i16x8));
    sum1_i32x4 = vmlal_high_s16(sum1_i32x4, a_unsigned_high_i16x8, b_unsigned_high_i16x8);

    if (count_scalars) goto nk_dot_e3m2_neonsdot_cycle;
    int32x4_t total_i32x4 = vaddq_s32(sum0_i32x4, sum1_i32x4);
    *result = (nk_f32_t)vaddvq_s32(total_i32x4) / 256.0f;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM_
#endif // NK_DOT_NEONSDOT_H
