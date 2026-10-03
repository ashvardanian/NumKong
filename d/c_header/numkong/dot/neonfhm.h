/**
 *  @brief SIMD-accelerated Dot Products for NEON FHM.
 *  @file include/numkong/dot/neonfhm.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_neonfhm_instructions ARM NEON FP16 Matrix Instructions (ARMv8.4-FHM)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vfmlalq_low_f16             FMLAL (V.4S, V.8H, V.8H)        4cy         2/cy        4/cy
 *      vfmlalq_high_f16            FMLAL2 (V.4S, V.8H, V.8H)       4cy         2/cy        4/cy
 *      vfmlslq_low_f16             FMLSL (V.4S, V.8H, V.8H)        4cy         2/cy        4/cy
 *      vfmlslq_high_f16            FMLSL2 (V.4S, V.8H, V.8H)       4cy         2/cy        4/cy
 *      vld1q_f16                   LD1 (V.8H)                      4cy         2/cy        3/cy
 *      vaddvq_f32                  FADDP+FADDP (V.4S)              4cy         1/cy        2/cy
 *
 *  The ARMv8.4-FHM extension (FEAT_FHM) provides FMLAL/FMLSL instructions that fuse FP16 to FP32
 *  widening with multiply-accumulate in a single operation. FMLAL executes as a single fused op
 *  (4cy latency, 2/cy throughput on A76, 4/cy on M4+/V1+/Oryon) rather than separate FCVTL + FMLA.
 *
 *  FMLAL preserves FP32 accumulator precision while accepting FP16 inputs, ideal for mixed-precision
 *  workloads. The _low variants process elements 0-3, _high variants process elements 4-7, enabling
 *  processing of 8 FP16 elements per iteration with full precision accumulation.
 *
 *  @section dot_neonfhm_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_f16x8 state with native FMLAL f16 dot-products.
 *
 *  @code{c}
 *  nk_dot_f16x8_state_neonfhm_t state_first, state_second, state_third, state_fourth;
 *  float16x8_t query_f16x8, target_first_f16x8, target_second_f16x8, target_third_f16x8, target_fourth_f16x8;
 *  nk_dot_f16x8_init_neonfhm(&state_first);
 *  nk_dot_f16x8_init_neonfhm(&state_second);
 *  nk_dot_f16x8_init_neonfhm(&state_third);
 *  nk_dot_f16x8_init_neonfhm(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 8 <= depth; idx += 8) {
 *      query_f16x8 = vld1q_f16(query_ptr + idx);
 *      target_first_f16x8 = vld1q_f16(target_first_ptr + idx);
 *      target_second_f16x8 = vld1q_f16(target_second_ptr + idx);
 *      target_third_f16x8 = vld1q_f16(target_third_ptr + idx);
 *      target_fourth_f16x8 = vld1q_f16(target_fourth_ptr + idx);
 *      nk_dot_f16x8_update_neonfhm(&state_first, query_f16x8, target_first_f16x8, idx, 8);
 *      nk_dot_f16x8_update_neonfhm(&state_second, query_f16x8, target_second_f16x8, idx, 8);
 *      nk_dot_f16x8_update_neonfhm(&state_third, query_f16x8, target_third_f16x8, idx, 8);
 *      nk_dot_f16x8_update_neonfhm(&state_fourth, query_f16x8, target_fourth_f16x8, idx, 8);
 *  }
 *  float32x4_t results_f32x4;
 *  nk_dot_f16x8_finalize_neonfhm(&state_first, &state_second, &state_third, &state_fourth, depth, &results_f32x4);
 *  @endcode
 *
 */
#ifndef NK_DOT_NEONFHM_H
#define NK_DOT_NEONFHM_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b8x8_serial_`
#include "numkong/cast/neon.h"   // `nk_e4m3x8_to_f16x8_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

NK_PUBLIC void nk_dot_f16_neonfhm(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_f16_neonfhm_cycle:
    if (count_scalars < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_vec, count_scalars);
        a_f16x8 = vreinterpretq_f16_u16(a_vec.u16x8);
        b_f16x8 = vreinterpretq_f16_u16(b_vec.u16x8);
        count_scalars = 0;
    }
    else {
        a_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(a_scalars));
        b_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(b_scalars));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    // FMLAL: widening multiply-accumulate fp16 → f32
    // low: processes elements 0-3, high: processes elements 4-7
    sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_f16x8, b_f16x8);
    sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_f16x8, b_f16x8);
    if (count_scalars) goto nk_dot_f16_neonfhm_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

typedef struct nk_dot_f16x8_state_neonfhm_t {
    float32x4_t sum_f32x4;
} nk_dot_f16x8_state_neonfhm_t;

NK_INTERNAL void nk_dot_f16x8_init_neonfhm(nk_dot_f16x8_state_neonfhm_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NK_INTERNAL void nk_dot_f16x8_update_neonfhm(nk_dot_f16x8_state_neonfhm_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    float16x8_t a_f16x8 = vreinterpretq_f16_u16(a.u16x8);
    float16x8_t b_f16x8 = vreinterpretq_f16_u16(b.u16x8);
    // FMLAL: widening multiply-accumulate fp16 → f32
    state->sum_f32x4 = vfmlalq_low_f16(state->sum_f32x4, a_f16x8, b_f16x8);
    state->sum_f32x4 = vfmlalq_high_f16(state->sum_f32x4, a_f16x8, b_f16x8);
}

NK_INTERNAL void nk_dot_f16x8_finalize_neonfhm(                                               //
    nk_dot_f16x8_state_neonfhm_t const *state_a, nk_dot_f16x8_state_neonfhm_t const *state_b, //
    nk_dot_f16x8_state_neonfhm_t const *state_c, nk_dot_f16x8_state_neonfhm_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = vaddvq_f32(state_a->sum_f32x4);
    result->f32s[1] = vaddvq_f32(state_b->sum_f32x4);
    result->f32s[2] = vaddvq_f32(state_c->sum_f32x4);
    result->f32s[3] = vaddvq_f32(state_d->sum_f32x4);
}

NK_PUBLIC void nk_dot_f16c_neonfhm(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *result) {
    // Accumulate into 4 float32x2_t vectors (low/high for real/imag)
    float32x2_t sum_real_low_f32x2 = vdup_n_f32(0);
    float32x2_t sum_real_high_f32x2 = vdup_n_f32(0);
    float32x2_t sum_imag_low_f32x2 = vdup_n_f32(0);
    float32x2_t sum_imag_high_f32x2 = vdup_n_f32(0);

    while (count_pairs >= 4) {
        // Load and deinterleave: vld2 loads 4 complex pairs as 2 x float16x4_t
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);

        float16x4_t a_real_f16x4 = vreinterpret_f16_s16(a_i16x4x2.val[0]);
        float16x4_t a_imag_f16x4 = vreinterpret_f16_s16(a_i16x4x2.val[1]);
        float16x4_t b_real_f16x4 = vreinterpret_f16_s16(b_i16x4x2.val[0]);
        float16x4_t b_imag_f16x4 = vreinterpret_f16_s16(b_i16x4x2.val[1]);

        // Real: aᵣ × bᵣ - aᵢ × bᵢ (FMLAL then FMLSL)
        sum_real_low_f32x2 = vfmlal_low_f16(sum_real_low_f32x2, a_real_f16x4, b_real_f16x4);
        sum_real_low_f32x2 = vfmlsl_low_f16(sum_real_low_f32x2, a_imag_f16x4, b_imag_f16x4);
        sum_real_high_f32x2 = vfmlal_high_f16(sum_real_high_f32x2, a_real_f16x4, b_real_f16x4);
        sum_real_high_f32x2 = vfmlsl_high_f16(sum_real_high_f32x2, a_imag_f16x4, b_imag_f16x4);

        // Imag: aᵣ × bᵢ + aᵢ × bᵣ (FMLAL for both)
        sum_imag_low_f32x2 = vfmlal_low_f16(sum_imag_low_f32x2, a_real_f16x4, b_imag_f16x4);
        sum_imag_low_f32x2 = vfmlal_low_f16(sum_imag_low_f32x2, a_imag_f16x4, b_real_f16x4);
        sum_imag_high_f32x2 = vfmlal_high_f16(sum_imag_high_f32x2, a_real_f16x4, b_imag_f16x4);
        sum_imag_high_f32x2 = vfmlal_high_f16(sum_imag_high_f32x2, a_imag_f16x4, b_real_f16x4);

        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Combine and reduce
    float32x4_t sum_real_f32x4 = vcombine_f32(sum_real_low_f32x2, sum_real_high_f32x2);
    float32x4_t sum_imag_f32x4 = vcombine_f32(sum_imag_low_f32x2, sum_imag_high_f32x2);

    // Handle tail with serial fallback
    nk_f32c_t tail_result;
    nk_dot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = vaddvq_f32(sum_real_f32x4) + tail_result.real;
    result->imag = vaddvq_f32(sum_imag_f32x4) + tail_result.imag;
}

NK_PUBLIC void nk_vdot_f16c_neonfhm(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    // Accumulate into 4 float32x2_t vectors (low/high for real/imag)
    float32x2_t sum_real_low_f32x2 = vdup_n_f32(0);
    float32x2_t sum_real_high_f32x2 = vdup_n_f32(0);
    float32x2_t sum_imag_low_f32x2 = vdup_n_f32(0);
    float32x2_t sum_imag_high_f32x2 = vdup_n_f32(0);

    while (count_pairs >= 4) {
        // Load and deinterleave: vld2 loads 4 complex pairs as 2 x float16x4_t
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);

        float16x4_t a_real_f16x4 = vreinterpret_f16_s16(a_i16x4x2.val[0]);
        float16x4_t a_imag_f16x4 = vreinterpret_f16_s16(a_i16x4x2.val[1]);
        float16x4_t b_real_f16x4 = vreinterpret_f16_s16(b_i16x4x2.val[0]);
        float16x4_t b_imag_f16x4 = vreinterpret_f16_s16(b_i16x4x2.val[1]);

        // Real: aᵣ × bᵣ + aᵢ × bᵢ (FMLAL for both)
        sum_real_low_f32x2 = vfmlal_low_f16(sum_real_low_f32x2, a_real_f16x4, b_real_f16x4);
        sum_real_low_f32x2 = vfmlal_low_f16(sum_real_low_f32x2, a_imag_f16x4, b_imag_f16x4);
        sum_real_high_f32x2 = vfmlal_high_f16(sum_real_high_f32x2, a_real_f16x4, b_real_f16x4);
        sum_real_high_f32x2 = vfmlal_high_f16(sum_real_high_f32x2, a_imag_f16x4, b_imag_f16x4);

        // Imag: aᵣ × bᵢ - aᵢ × bᵣ (FMLAL then FMLSL)
        sum_imag_low_f32x2 = vfmlal_low_f16(sum_imag_low_f32x2, a_real_f16x4, b_imag_f16x4);
        sum_imag_low_f32x2 = vfmlsl_low_f16(sum_imag_low_f32x2, a_imag_f16x4, b_real_f16x4);
        sum_imag_high_f32x2 = vfmlal_high_f16(sum_imag_high_f32x2, a_real_f16x4, b_imag_f16x4);
        sum_imag_high_f32x2 = vfmlsl_high_f16(sum_imag_high_f32x2, a_imag_f16x4, b_real_f16x4);

        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Combine and reduce
    float32x4_t sum_real_f32x4 = vcombine_f32(sum_real_low_f32x2, sum_real_high_f32x2);
    float32x4_t sum_imag_f32x4 = vcombine_f32(sum_imag_low_f32x2, sum_imag_high_f32x2);

    // Handle tail with serial fallback
    nk_f32c_t tail_result;
    nk_vdot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = vaddvq_f32(sum_real_f32x4) + tail_result.real;
    result->imag = vaddvq_f32(sum_imag_f32x4) + tail_result.imag;
}

NK_PUBLIC void nk_dot_e4m3_neonfhm(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    float16x8_t a_low, a_high, b_low, b_high;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e4m3_neonfhm_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        nk_e4m3x16_to_f16x8x2_neon_(a_vec.u8x16, &a_low, &a_high);
        nk_e4m3x16_to_f16x8x2_neon_(b_vec.u8x16, &b_low, &b_high);
        count_scalars = 0;
    }
    else {
        nk_e4m3x16_to_f16x8x2_neon_(vld1q_u8(a_scalars), &a_low, &a_high);
        nk_e4m3x16_to_f16x8x2_neon_(vld1q_u8(b_scalars), &b_low, &b_high);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_low, b_low);
    sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_low, b_low);
    sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_high, b_high);
    sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_high, b_high);
    if (count_scalars) goto nk_dot_e4m3_neonfhm_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_e5m2_neonfhm(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    float16x8_t a_low, a_high, b_low, b_high;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e5m2_neonfhm_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_low = vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(a_vec.u8x16), 8));
        a_high = vreinterpretq_f16_u16(vshll_n_u8(vget_high_u8(a_vec.u8x16), 8));
        b_low = vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(b_vec.u8x16), 8));
        b_high = vreinterpretq_f16_u16(vshll_n_u8(vget_high_u8(b_vec.u8x16), 8));
        count_scalars = 0;
    }
    else {
        uint8x16_t a_u8x16 = vld1q_u8(a_scalars);
        uint8x16_t b_u8x16 = vld1q_u8(b_scalars);
        a_low = vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(a_u8x16), 8));
        a_high = vreinterpretq_f16_u16(vshll_n_u8(vget_high_u8(a_u8x16), 8));
        b_low = vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(b_u8x16), 8));
        b_high = vreinterpretq_f16_u16(vshll_n_u8(vget_high_u8(b_u8x16), 8));
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_low, b_low);
    sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_low, b_low);
    sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_high, b_high);
    sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_high, b_high);
    if (count_scalars) goto nk_dot_e5m2_neonfhm_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

typedef struct nk_dot_e4m3x16_state_neonfhm_t {
    float32x4_t sum_f32x4;
} nk_dot_e4m3x16_state_neonfhm_t;

NK_INTERNAL void nk_dot_e4m3x16_init_neonfhm(nk_dot_e4m3x16_state_neonfhm_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

NK_INTERNAL void nk_dot_e4m3x16_update_neonfhm(nk_dot_e4m3x16_state_neonfhm_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Convert e4m3 → f16 using 16-element LUT path (4× VQTBL4)
    float16x8_t a_low_f16x8, a_high_f16x8, b_low_f16x8, b_high_f16x8;
    nk_e4m3x16_to_f16x8x2_neon_(a.u8x16, &a_low_f16x8, &a_high_f16x8);
    nk_e4m3x16_to_f16x8x2_neon_(b.u8x16, &b_low_f16x8, &b_high_f16x8);
    // FMLAL: widening multiply-accumulate fp16 → f32
    state->sum_f32x4 = vfmlalq_low_f16(state->sum_f32x4, a_low_f16x8, b_low_f16x8);
    state->sum_f32x4 = vfmlalq_high_f16(state->sum_f32x4, a_low_f16x8, b_low_f16x8);
    state->sum_f32x4 = vfmlalq_low_f16(state->sum_f32x4, a_high_f16x8, b_high_f16x8);
    state->sum_f32x4 = vfmlalq_high_f16(state->sum_f32x4, a_high_f16x8, b_high_f16x8);
}

NK_INTERNAL void nk_dot_e4m3x16_finalize_neonfhm(                                                 //
    nk_dot_e4m3x16_state_neonfhm_t const *state_a, nk_dot_e4m3x16_state_neonfhm_t const *state_b, //
    nk_dot_e4m3x16_state_neonfhm_t const *state_c, nk_dot_e4m3x16_state_neonfhm_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = vaddvq_f32(state_a->sum_f32x4);
    result->f32s[1] = vaddvq_f32(state_b->sum_f32x4);
    result->f32s[2] = vaddvq_f32(state_c->sum_f32x4);
    result->f32s[3] = vaddvq_f32(state_d->sum_f32x4);
}

typedef struct nk_dot_e5m2x16_state_neonfhm_t {
    float32x4_t sum_f32x4;
} nk_dot_e5m2x16_state_neonfhm_t;

NK_INTERNAL void nk_dot_e5m2x16_init_neonfhm(nk_dot_e5m2x16_state_neonfhm_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

NK_INTERNAL void nk_dot_e5m2x16_update_neonfhm(nk_dot_e5m2x16_state_neonfhm_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Convert e5m2 → f16 via SHLL: widen u8→u16 and shift left 8 in one instruction
    float16x8_t a_low_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(a.u8x16), 8));
    float16x8_t a_high_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(vget_high_u8(a.u8x16), 8));
    float16x8_t b_low_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(b.u8x16), 8));
    float16x8_t b_high_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(vget_high_u8(b.u8x16), 8));
    // FMLAL: widening multiply-accumulate fp16 → f32
    state->sum_f32x4 = vfmlalq_low_f16(state->sum_f32x4, a_low_f16x8, b_low_f16x8);
    state->sum_f32x4 = vfmlalq_high_f16(state->sum_f32x4, a_low_f16x8, b_low_f16x8);
    state->sum_f32x4 = vfmlalq_low_f16(state->sum_f32x4, a_high_f16x8, b_high_f16x8);
    state->sum_f32x4 = vfmlalq_high_f16(state->sum_f32x4, a_high_f16x8, b_high_f16x8);
}

NK_INTERNAL void nk_dot_e5m2x16_finalize_neonfhm(                                                 //
    nk_dot_e5m2x16_state_neonfhm_t const *state_a, nk_dot_e5m2x16_state_neonfhm_t const *state_b, //
    nk_dot_e5m2x16_state_neonfhm_t const *state_c, nk_dot_e5m2x16_state_neonfhm_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = vaddvq_f32(state_a->sum_f32x4);
    result->f32s[1] = vaddvq_f32(state_b->sum_f32x4);
    result->f32s[2] = vaddvq_f32(state_c->sum_f32x4);
    result->f32s[3] = vaddvq_f32(state_d->sum_f32x4);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_
#endif // NK_DOT_NEONFHM_H
