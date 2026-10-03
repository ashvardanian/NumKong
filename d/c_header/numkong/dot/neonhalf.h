/**
 *  @brief SIMD-accelerated Dot Products for NEON FP16.
 *  @file include/numkong/dot/neonhalf.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_neonhalf_instructions ARM NEON FP16 Instructions (ARMv8.2-FP16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vfmaq_f16                   FMLA (V.8H, V.8H, V.8H)         4cy         2/cy        4/cy
 *      vcvt_f32_f16                FCVTL (V.4S, V.4H)              3cy         2/cy        4/cy
 *      vld1q_f16                   LD1 (V.8H)                      4cy         2/cy        3/cy
 *      vaddvq_f32                  FADDP+FADDP (V.4S)              4cy         1/cy        2/cy
 *      vfmsq_f16                   FMLS (V.8H, V.8H, V.8H)         4cy         2/cy        4/cy
 *
 *  The ARMv8.2-FP16 extension enables native half-precision arithmetic, doubling the element count
 *  per vector register (8x F16 vs 4x F32). This doubles theoretical throughput for bandwidth-bound
 *  workloads while halving memory footprint.
 *
 *  For dot products, inputs are widened from F16 to F32 for accumulation to preserve numerical
 *  precision. The FCVTL instruction handles this widening, allowing the FMA operations
 *  to maintain full F32 precision in the accumulator.
 *
 *  @section dot_neonhalf_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_f16x4 state with f16 inputs widened to f32 for accumulation.
 *
 *  @code{c}
 *  nk_dot_f16x4_state_neonhalf_t state_first, state_second, state_third, state_fourth;
 *  float16x4_t query_f16x4, target_first_f16x4, target_second_f16x4, target_third_f16x4, target_fourth_f16x4;
 *  nk_dot_f16x4_init_neonhalf(&state_first);
 *  nk_dot_f16x4_init_neonhalf(&state_second);
 *  nk_dot_f16x4_init_neonhalf(&state_third);
 *  nk_dot_f16x4_init_neonhalf(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 4 <= depth; idx += 4) {
 *      query_f16x4 = vld1_f16(query_ptr + idx);
 *      target_first_f16x4 = vld1_f16(target_first_ptr + idx);
 *      target_second_f16x4 = vld1_f16(target_second_ptr + idx);
 *      target_third_f16x4 = vld1_f16(target_third_ptr + idx);
 *      target_fourth_f16x4 = vld1_f16(target_fourth_ptr + idx);
 *      nk_dot_f16x4_update_neonhalf(&state_first, query_f16x4, target_first_f16x4, idx, 4);
 *      nk_dot_f16x4_update_neonhalf(&state_second, query_f16x4, target_second_f16x4, idx, 4);
 *      nk_dot_f16x4_update_neonhalf(&state_third, query_f16x4, target_third_f16x4, idx, 4);
 *      nk_dot_f16x4_update_neonhalf(&state_fourth, query_f16x4, target_fourth_f16x4, idx, 4);
 *  }
 *  float32x4_t results_f32x4;
 *  nk_dot_f16x4_finalize_neonhalf(&state_first, &state_second, &state_third, &state_fourth, depth, &results_f32x4);
 *  @endcode
 */
#ifndef NK_DOT_NEONHALF_H
#define NK_DOT_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b16x4_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NK_PUBLIC void nk_dot_f16_neonhalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    float32x4_t a_f32x4, b_f32x4;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_f16_neonhalf_cycle:
    if (count_scalars < 4) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b16x4_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_vec, count_scalars);
        a_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(a_vec.u16x4));
        b_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(b_vec.u16x4));
        count_scalars = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)a_scalars));
        b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)b_scalars));
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_f32x4, b_f32x4);
    if (count_scalars) goto nk_dot_f16_neonhalf_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_f16c_neonhalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_f16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_f16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmsq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    nk_f32c_t tail_result;
    nk_dot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

NK_PUBLIC void nk_vdot_f16c_neonhalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                     nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_f16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_f16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmsq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    nk_f32c_t tail_result;
    nk_vdot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

/**
 *  @brief Running state for 64-bit dot accumulation over f16 scalars on NEON with FP16 extension.
 *
 *  Processes 4 f16 values at a time (64 bits), converting directly to f32 without
 *  the overhead of vget_low/vget_high operations on 128-bit vectors.
 */
typedef struct nk_dot_f16x4_state_neonhalf_t {
    float32x4_t sum_f32x4;
} nk_dot_f16x4_state_neonhalf_t;

NK_INTERNAL void nk_dot_f16x4_init_neonhalf(nk_dot_f16x4_state_neonhalf_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NK_INTERNAL void nk_dot_f16x4_update_neonhalf(nk_dot_f16x4_state_neonhalf_t *state, nk_b64_vec_t a, nk_b64_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // 4 f16s = 64 bits, direct conversion without low/high split
    float16x4_t a_f16x4 = vreinterpret_f16_u16(a.u16x4);
    float16x4_t b_f16x4 = vreinterpret_f16_u16(b.u16x4);
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, vcvt_f32_f16(a_f16x4), vcvt_f32_f16(b_f16x4));
}

NK_INTERNAL void nk_dot_f16x4_finalize_neonhalf(                                                //
    nk_dot_f16x4_state_neonhalf_t const *state_a, nk_dot_f16x4_state_neonhalf_t const *state_b, //
    nk_dot_f16x4_state_neonhalf_t const *state_c, nk_dot_f16x4_state_neonhalf_t const *state_d, //
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

#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_
#endif // NK_DOT_NEONHALF_H
