/**
 *  @file include/numkong/dot/neon.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief SIMD-accelerated dot products for NEON.
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_neon_instructions NEON Dot Product Instructions
 *
 *  Key NEON instructions for dot products:
 *
 *  @verbatim
 *  Intrinsic     Instruction                  A76       M5
 *  vfmaq_f32     FMLA (V.4S, V.4S, V.4S)      4cy @ 2p  3cy @ 4p
 *  vfmaq_f64     FMLA (V.2D, V.2D, V.2D)      4cy @ 2p  4cy @ 4p
 *  vfmsq_f64     FMLS (V.2D, V.2D, V.2D)      4cy @ 2p  4cy @ 4p
 *  vmulq_f32     FMUL (V.4S, V.4S, V.4S)      3cy @ 2p  3cy @ 4p
 *  vmulq_f64     FMUL (V.2D, V.2D, V.2D)      3cy @ 2p  3cy @ 4p
 *  vaddvq_f32    FADDP+FADDP (reduce)         5cy @ 1p  8cy @ 1p
 *  vaddvq_f64    FADDP (V.2D to scalar)       3cy @ 1p  3cy @ 1p
 *  vpaddq_f32    FADDP (V.4S, V.4S, V.4S)     2cy @ 2p  3cy @ 4p
 *  vpaddq_f64    FADDP (V.2D, V.2D, V.2D)     2cy @ 2p  3cy @ 4p
 *  vcvt_f64_f32  FCVTL (V.2D, V.2S)           3cy @ 2p  3cy @ 2p
 *  vld2_f32      LD2 ({Vt.2S, Vt2.2S}, [Xn])  4cy @ 1p  4cy @ 1p
 *  @endverbatim
 *
 *  FMA throughput doubles on cores with 4 SIMD pipes (Apple M4+, Graviton3+, Oryon), but horizontal
 *  reductions remain at 1/cy on all cores and become the main bottleneck.
 *
 *  For f32 dot products, we upcast to f64 for accumulation to preserve precision and avoid
 *  catastrophic cancellation in large-magnitude sums.
 *
 *  @section dot_neon_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and
 *  force-inlined @c NUMKONG_INLINE functions:
 *
 *  - nk_dot_f32x2 state for f32 inputs with double-precision accumulation,
 *  - nk_dot_f64x2 state with Dot2 stable dot-products for f64 inputs.
 *
 *  @code{.c}
 *  nk_dot_f32x2_state_neon_t state_first, state_second, state_third, state_fourth;
 *  float32x2_t query_f32x2, target_first_f32x2, target_second_f32x2, target_third_f32x2, target_fourth_f32x2;
 *  nk_dot_f32x2_init_neon(&state_first);
 *  nk_dot_f32x2_init_neon(&state_second);
 *  nk_dot_f32x2_init_neon(&state_third);
 *  nk_dot_f32x2_init_neon(&state_fourth);
 *  for (nk_size_t index = 0; index + 2 <= depth; index += 2) {
 *      query_f32x2 = vld1_f32(query_ptr + index);
 *      target_first_f32x2 = vld1_f32(target_first_ptr + index);
 *      target_second_f32x2 = vld1_f32(target_second_ptr + index);
 *      target_third_f32x2 = vld1_f32(target_third_ptr + index);
 *      target_fourth_f32x2 = vld1_f32(target_fourth_ptr + index);
 *      nk_dot_f32x2_update_neon(&state_first, query_f32x2, target_first_f32x2, index, 2);
 *      nk_dot_f32x2_update_neon(&state_second, query_f32x2, target_second_f32x2, index, 2);
 *      nk_dot_f32x2_update_neon(&state_third, query_f32x2, target_third_f32x2, index, 2);
 *      nk_dot_f32x2_update_neon(&state_fourth, query_f32x2, target_fourth_f32x2, index, 2);
 *  }
 *  float32x4_t results_f32x4;
 *  nk_dot_f32x2_finalize_neon(&state_first, &state_second, &state_third, &state_fourth, depth, &results_f32x4);
 *  @endcode
 *
 *  For f64 inputs, Dot2 compensated summation provides numerical stability:
 *
 *  @code{.c}
 *  nk_dot_f64x2_state_neon_t state_first, state_second, state_third, state_fourth;
 *  float64x2_t query_f64x2, target_first_f64x2, target_second_f64x2, target_third_f64x2, target_fourth_f64x2;
 *  nk_dot_f64x2_init_neon(&state_first);
 *  nk_dot_f64x2_init_neon(&state_second);
 *  nk_dot_f64x2_init_neon(&state_third);
 *  nk_dot_f64x2_init_neon(&state_fourth);
 *  for (nk_size_t index = 0; index + 2 <= depth; index += 2) {
 *      query_f64x2 = vld1q_f64(query_ptr + index);
 *      target_first_f64x2 = vld1q_f64(target_first_ptr + index);
 *      target_second_f64x2 = vld1q_f64(target_second_ptr + index);
 *      target_third_f64x2 = vld1q_f64(target_third_ptr + index);
 *      target_fourth_f64x2 = vld1q_f64(target_fourth_ptr + index);
 *      nk_dot_f64x2_update_neon(&state_first, query_f64x2, target_first_f64x2, index, 2);
 *      nk_dot_f64x2_update_neon(&state_second, query_f64x2, target_second_f64x2, index, 2);
 *      nk_dot_f64x2_update_neon(&state_third, query_f64x2, target_third_f64x2, index, 2);
 *      nk_dot_f64x2_update_neon(&state_fourth, query_f64x2, target_fourth_f64x2, index, 2);
 *  }
 *  float64x4_t results_f64x4;
 *  nk_dot_f64x2_finalize_neon(&state_first, &state_second, &state_third, &state_fourth, depth, &results_f64x4);
 *  @endcode
 */
#ifndef NUMKONG_DOT_NEON_H
#define NUMKONG_DOT_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/cast/neon.h"  // `nk_e4m3x8_to_f16x8_neon_`
#include "numkong/dot/serial.h" // `nk_dot_f16c_`, `nk_vdot_f16c_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/** Compensated horizontal sum of 2 f64 lanes via TwoSum. */
NUMKONG_INLINE nk_f64_t nk_dot_stable_sum_f64x2_neon_(float64x2_t sum_f64x2, float64x2_t compensation_f64x2) {
    // TwoSum merge of sum + compensation (2-wide)
    float64x2_t tentative_sum_f64x2 = vaddq_f64(sum_f64x2, compensation_f64x2);
    float64x2_t virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, sum_f64x2);
    float64x2_t rounding_error_f64x2 = vaddq_f64(
        vsubq_f64(sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
        vsubq_f64(compensation_f64x2, virtual_addend_f64x2));
    // Scalar TwoSum 2→1
    nk_f64_t lower_sum = vgetq_lane_f64(tentative_sum_f64x2, 0);
    nk_f64_t upper_sum = vgetq_lane_f64(tentative_sum_f64x2, 1);
    nk_f64_t lower_error = vgetq_lane_f64(rounding_error_f64x2, 0);
    nk_f64_t upper_error = vgetq_lane_f64(rounding_error_f64x2, 1);
    nk_f64_t tentative_sum = lower_sum + upper_sum;
    nk_f64_t virtual_addend = tentative_sum - lower_sum;
    nk_f64_t rounding_error = (lower_sum - (tentative_sum - virtual_addend)) + (upper_sum - virtual_addend);
    return tentative_sum + (lower_error + upper_error + rounding_error);
}

/** Dot2 step, sum += a × b, mirroring @c nk_f64_dot2_: TwoProd through FMA, then TwoSum. */
NUMKONG_INLINE void nk_dot2_f64x2_neon_(float64x2_t *sum_f64x2, float64x2_t *compensation_f64x2, float64x2_t a_f64x2,
                                        float64x2_t b_f64x2) {
    float64x2_t product_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
    float64x2_t product_error_f64x2 = vnegq_f64(vfmsq_f64(product_f64x2, a_f64x2, b_f64x2));
    float64x2_t tentative_sum_f64x2 = vaddq_f64(*sum_f64x2, product_f64x2);
    float64x2_t virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, *sum_f64x2);
    float64x2_t sum_error_f64x2 = vaddq_f64(vsubq_f64(*sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
                                            vsubq_f64(product_f64x2, virtual_addend_f64x2));
    *sum_f64x2 = tentative_sum_f64x2;
    *compensation_f64x2 = vaddq_f64(*compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));
}

/** Scaled partial sums stay in vector lanes until the output tile is finalized. */
typedef struct nk_dot_scaled_state_neon_t {
    float64x2_t sum_f64x2;
} nk_dot_scaled_state_neon_t;

NUMKONG_INLINE void nk_dot_scaled_init_neon(nk_dot_scaled_state_neon_t *state) { state->sum_f64x2 = vdupq_n_f64(0); }

NUMKONG_INLINE void nk_dot_scaled_finalize_neon(                                          //
    nk_dot_scaled_state_neon_t const *state_a, nk_dot_scaled_state_neon_t const *state_b, //
    nk_dot_scaled_state_neon_t const *state_c, nk_dot_scaled_state_neon_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    float64x2_t const sums_ab_f64x2 = vpaddq_f64(state_a->sum_f64x2, state_b->sum_f64x2);
    float64x2_t const sums_cd_f64x2 = vpaddq_f64(state_c->sum_f64x2, state_d->sum_f64x2);
    result->f32x4 = vcombine_f32(vcvt_f32_f64(sums_ab_f64x2), vcvt_f32_f64(sums_cd_f64x2));
}

typedef struct nk_dot_scaled_i8x16_operand_neon_t {
    int8x16_t values_i8x16;
    nk_f64_t scale;
} nk_dot_scaled_i8x16_operand_neon_t;

typedef struct nk_dot_scaled_i8x32_operand_neon_t {
    int8x16_t values_i8x16[2];
    nk_f64_t scale;
} nk_dot_scaled_i8x32_operand_neon_t;

typedef struct nk_dot_scaled_i16x32_operand_neon_t {
    int16x8_t values_i16x8[4];
    nk_f64_t scale;
} nk_dot_scaled_i16x32_operand_neon_t;

typedef struct nk_dot_scaled_f16x32_operand_neon_t {
    float16x8_t values_f16x8[4];
    nk_f64_t scale;
} nk_dot_scaled_f16x32_operand_neon_t;

/** Doubled E2M1 values fit in signed bytes, including the signed-zero code. */
NUMKONG_INLINE int8x16_t nk_e2m1x16_to_i8x16_neon_(uint8x8_t packed_u8x8) {
    uint8x16_t codes_u8x16 = vcombine_u8(vshr_n_u8(packed_u8x8, 4), vand_u8(packed_u8x8, vdup_n_u8(15)));
    uint8x16_t exponent_u8x16 = vandq_u8(vshrq_n_u8(codes_u8x16, 1), vdupq_n_u8(3));
    uint8x16_t mantissa_u8x16 = vorrq_u8(vandq_u8(codes_u8x16, vdupq_n_u8(1)),
                                         vandq_u8(vcgtq_u8(exponent_u8x16, vdupq_n_u8(0)), vdupq_n_u8(2)));
    int8x16_t values_i8x16 = vreinterpretq_s8_u8(
        vshlq_u8(mantissa_u8x16, vreinterpretq_s8_u8(vqsubq_u8(exponent_u8x16, vdupq_n_u8(1)))));
    return vbslq_s8(vtstq_u8(codes_u8x16, vdupq_n_u8(8)), vnegq_s8(values_i8x16), values_i8x16);
}

/** E2M3 values multiplied by eight fit in signed bytes. */
NUMKONG_INLINE int8x16_t nk_e2m3x16_to_i8x16_neon_(uint8x16_t codes_u8x16) {
    uint8x16_t exponent_u8x16 = vandq_u8(vshrq_n_u8(codes_u8x16, 3), vdupq_n_u8(3));
    uint8x16_t mantissa_u8x16 = vorrq_u8(vandq_u8(codes_u8x16, vdupq_n_u8(7)),
                                         vandq_u8(vcgtq_u8(exponent_u8x16, vdupq_n_u8(0)), vdupq_n_u8(8)));
    int8x16_t values_i8x16 = vreinterpretq_s8_u8(
        vshlq_u8(mantissa_u8x16, vreinterpretq_s8_u8(vqsubq_u8(exponent_u8x16, vdupq_n_u8(1)))));
    return vbslq_s8(vtstq_u8(codes_u8x16, vdupq_n_u8(32)), vnegq_s8(values_i8x16), values_i8x16);
}

/** E3M2 values multiplied by sixteen fit in signed halfwords. */
NUMKONG_INLINE int16x8_t nk_e3m2x8_to_i16x8_neon_(uint8x8_t codes_u8x8) {
    uint16x8_t codes_u16x8 = vmovl_u8(codes_u8x8);
    uint16x8_t exponent_u16x8 = vandq_u16(vshrq_n_u16(codes_u16x8, 2), vdupq_n_u16(7));
    uint16x8_t mantissa_u16x8 = vorrq_u16(vandq_u16(codes_u16x8, vdupq_n_u16(3)),
                                          vandq_u16(vcgtq_u16(exponent_u16x8, vdupq_n_u16(0)), vdupq_n_u16(4)));
    int16x8_t values_i16x8 = vreinterpretq_s16_u16(
        vshlq_u16(mantissa_u16x8, vreinterpretq_s16_u16(vqsubq_u16(exponent_u16x8, vdupq_n_u16(1)))));
    return vbslq_s16(vtstq_u16(codes_u16x8, vdupq_n_u16(32)), vnegq_s16(values_i16x8), values_i16x8);
}

NUMKONG_INLINE void nk_load_nvfp4x1_to_i8x16_neon_(void const *codes, nk_u8_t const *scales, nk_size_t offset,
                                                   nk_dot_scaled_i8x16_operand_neon_t *dst) {
    nk_u8_t const *src = (nk_u8_t const *)codes + offset;
    dst->values_i8x16 = nk_e2m1x16_to_i8x16_neon_(vld1_u8(src));
    dst->scale = 0.5 * (nk_f64_t)nk_block_scaled_decode_scale_serial_(scales[offset / 8], nk_ue4m3_k);
}

NUMKONG_INLINE void nk_partial_load_nvfp4x1_to_i8x16_neon_(void const *codes, nk_u8_t const *scales, nk_size_t offset,
                                                           nk_dot_scaled_i8x16_operand_neon_t *dst, nk_size_t n) {
    nk_assert_(n == 16);
    nk_load_nvfp4x1_to_i8x16_neon_(codes, scales, offset, dst);
}

NUMKONG_INLINE void nk_load_mxfp4x1_to_i8x32_neon_(void const *codes, nk_u8_t const *scales, nk_size_t offset,
                                                   nk_dot_scaled_i8x32_operand_neon_t *dst) {
    nk_u8_t const *src = (nk_u8_t const *)codes + offset;
    dst->values_i8x16[0] = nk_e2m1x16_to_i8x16_neon_(vld1_u8(src));
    dst->values_i8x16[1] = nk_e2m1x16_to_i8x16_neon_(vld1_u8(src + 8));
    dst->scale = 0.5 * (nk_f64_t)nk_block_scaled_decode_scale_serial_(scales[offset / 16], nk_ue8m0_k);
}

NUMKONG_INLINE void nk_partial_load_mxfp4x1_to_i8x32_neon_(void const *codes, nk_u8_t const *scales, nk_size_t offset,
                                                           nk_dot_scaled_i8x32_operand_neon_t *dst, nk_size_t n) {
    nk_assert_(n == 32);
    nk_load_mxfp4x1_to_i8x32_neon_(codes, scales, offset, dst);
}

NUMKONG_INLINE void nk_load_mxfp6e2m3x1_to_i8x32_neon_(void const *codes, nk_u8_t const *scales, nk_size_t offset,
                                                       nk_dot_scaled_i8x32_operand_neon_t *dst) {
    nk_u8_t const *src = (nk_u8_t const *)codes + offset;
    dst->values_i8x16[0] = nk_e2m3x16_to_i8x16_neon_(vld1q_u8(src));
    dst->values_i8x16[1] = nk_e2m3x16_to_i8x16_neon_(vld1q_u8(src + 16));
    dst->scale = 0.125 * (nk_f64_t)nk_block_scaled_decode_scale_serial_(scales[offset / 32], nk_ue8m0_k);
}

NUMKONG_INLINE void nk_partial_load_mxfp6e2m3x1_to_i8x32_neon_(void const *codes, nk_u8_t const *scales,
                                                               nk_size_t offset,
                                                               nk_dot_scaled_i8x32_operand_neon_t *dst, nk_size_t n) {
    nk_assert_(n == 32);
    nk_load_mxfp6e2m3x1_to_i8x32_neon_(codes, scales, offset, dst);
}

NUMKONG_INLINE void nk_load_mxfp6e3m2x1_to_i16x32_neon_(void const *codes, nk_u8_t const *scales, nk_size_t offset,
                                                        nk_dot_scaled_i16x32_operand_neon_t *dst) {
    nk_u8_t const *src = (nk_u8_t const *)codes + offset;
    for (nk_size_t i = 0; i != 4; ++i) dst->values_i16x8[i] = nk_e3m2x8_to_i16x8_neon_(vld1_u8(src + i * 8));
    dst->scale = 0.0625 * (nk_f64_t)nk_block_scaled_decode_scale_serial_(scales[offset / 32], nk_ue8m0_k);
}

NUMKONG_INLINE void nk_partial_load_mxfp6e3m2x1_to_i16x32_neon_(void const *codes, nk_u8_t const *scales,
                                                                nk_size_t offset,
                                                                nk_dot_scaled_i16x32_operand_neon_t *dst, nk_size_t n) {
    nk_assert_(n == 32);
    nk_load_mxfp6e3m2x1_to_i16x32_neon_(codes, scales, offset, dst);
}

NUMKONG_INLINE void nk_load_mxfp8e4m3x1_to_f16x32_neon_(void const *codes, nk_u8_t const *scales, nk_size_t offset,
                                                        nk_dot_scaled_f16x32_operand_neon_t *dst) {
    nk_u8_t const *src = (nk_u8_t const *)codes + offset;
    nk_e4m3x16_to_f16x8x2_neon_(vld1q_u8(src), &dst->values_f16x8[0], &dst->values_f16x8[1]);
    nk_e4m3x16_to_f16x8x2_neon_(vld1q_u8(src + 16), &dst->values_f16x8[2], &dst->values_f16x8[3]);
    dst->scale = (nk_f64_t)nk_block_scaled_decode_scale_serial_(scales[offset / 32], nk_ue8m0_k);
}

NUMKONG_INLINE void nk_partial_load_mxfp8e4m3x1_to_f16x32_neon_(void const *codes, nk_u8_t const *scales,
                                                                nk_size_t offset,
                                                                nk_dot_scaled_f16x32_operand_neon_t *dst, nk_size_t n) {
    nk_assert_(n == 32);
    nk_load_mxfp8e4m3x1_to_f16x32_neon_(codes, scales, offset, dst);
}

NUMKONG_INLINE void nk_load_mxfp8e5m2x1_to_f16x32_neon_(void const *codes, nk_u8_t const *scales, nk_size_t offset,
                                                        nk_dot_scaled_f16x32_operand_neon_t *dst) {
    nk_u8_t const *src = (nk_u8_t const *)codes + offset;
    for (nk_size_t i = 0; i != 4; ++i) dst->values_f16x8[i] = nk_e5m2x8_to_f16x8_neon_(vld1_u8(src + i * 8));
    dst->scale = (nk_f64_t)nk_block_scaled_decode_scale_serial_(scales[offset / 32], nk_ue8m0_k);
}

NUMKONG_INLINE void nk_partial_load_mxfp8e5m2x1_to_f16x32_neon_(void const *codes, nk_u8_t const *scales,
                                                                nk_size_t offset,
                                                                nk_dot_scaled_f16x32_operand_neon_t *dst, nk_size_t n) {
    nk_assert_(n == 32);
    nk_load_mxfp8e5m2x1_to_f16x32_neon_(codes, scales, offset, dst);
}

NUMKONG_INLINE float64x2_t nk_i32x4_sum_as_f64x2_neon_(int32x4_t sums_i32x4) {
    int64x2_t sums_i64x2 = vpaddlq_s32(sums_i32x4);
    return vcvtq_f64_s64(sums_i64x2);
}

NUMKONG_INLINE void nk_dot_scaled_i8x16_update_neon_(nk_dot_scaled_state_neon_t *state,
                                                     nk_dot_scaled_i8x16_operand_neon_t a,
                                                     nk_dot_scaled_i8x16_operand_neon_t b, nk_size_t depth_offset,
                                                     nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    int32x4_t sums_i32x4 = vdupq_n_s32(0);
    sums_i32x4 = vpadalq_s16(sums_i32x4, vmull_s8(vget_low_s8(a.values_i8x16), vget_low_s8(b.values_i8x16)));
    sums_i32x4 = vpadalq_s16(sums_i32x4, vmull_high_s8(a.values_i8x16, b.values_i8x16));
    float64x2_t const sums_f64x2 = nk_i32x4_sum_as_f64x2_neon_(sums_i32x4);
    state->sum_f64x2 = vaddq_f64(state->sum_f64x2, vmulq_n_f64(sums_f64x2, a.scale * b.scale));
}

NUMKONG_INLINE void nk_dot_scaled_i8x32_update_neon_(nk_dot_scaled_state_neon_t *state,
                                                     nk_dot_scaled_i8x32_operand_neon_t a,
                                                     nk_dot_scaled_i8x32_operand_neon_t b, nk_size_t depth_offset,
                                                     nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    int32x4_t sums_i32x4 = vdupq_n_s32(0);
    for (nk_size_t i = 0; i != 2; ++i) {
        sums_i32x4 = vpadalq_s16(sums_i32x4, vmull_s8(vget_low_s8(a.values_i8x16[i]), vget_low_s8(b.values_i8x16[i])));
        sums_i32x4 = vpadalq_s16(sums_i32x4, vmull_high_s8(a.values_i8x16[i], b.values_i8x16[i]));
    }
    float64x2_t const sums_f64x2 = nk_i32x4_sum_as_f64x2_neon_(sums_i32x4);
    state->sum_f64x2 = vaddq_f64(state->sum_f64x2, vmulq_n_f64(sums_f64x2, a.scale * b.scale));
}

NUMKONG_INLINE void nk_dot_scaled_i16x32_update_neon_(nk_dot_scaled_state_neon_t *state,
                                                      nk_dot_scaled_i16x32_operand_neon_t a,
                                                      nk_dot_scaled_i16x32_operand_neon_t b, nk_size_t depth_offset,
                                                      nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    int32x4_t sums_i32x4 = vdupq_n_s32(0);
    for (nk_size_t i = 0; i != 4; ++i) {
        sums_i32x4 = vmlal_s16(sums_i32x4, vget_low_s16(a.values_i16x8[i]), vget_low_s16(b.values_i16x8[i]));
        sums_i32x4 = vmlal_high_s16(sums_i32x4, a.values_i16x8[i], b.values_i16x8[i]);
    }
    float64x2_t const sums_f64x2 = nk_i32x4_sum_as_f64x2_neon_(sums_i32x4);
    state->sum_f64x2 = vaddq_f64(state->sum_f64x2, vmulq_n_f64(sums_f64x2, a.scale * b.scale));
}

NUMKONG_INLINE void nk_dot_scaled_f16x32_update_neon_(nk_dot_scaled_state_neon_t *state,
                                                      nk_dot_scaled_f16x32_operand_neon_t a,
                                                      nk_dot_scaled_f16x32_operand_neon_t b, nk_size_t depth_offset,
                                                      nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    float64x2_t sums_f64x2 = vdupq_n_f64(0);
    for (nk_size_t i = 0; i != 4; ++i) {
        float32x4_t low_f32x4 = vmulq_f32(vcvt_f32_f16(vget_low_f16(a.values_f16x8[i])),
                                          vcvt_f32_f16(vget_low_f16(b.values_f16x8[i])));
        float32x4_t high_f32x4 = vmulq_f32(vcvt_high_f32_f16(a.values_f16x8[i]), vcvt_high_f32_f16(b.values_f16x8[i]));
        sums_f64x2 = vaddq_f64(sums_f64x2, vcvt_f64_f32(vget_low_f32(low_f32x4)));
        sums_f64x2 = vaddq_f64(sums_f64x2, vcvt_high_f64_f32(low_f32x4));
        sums_f64x2 = vaddq_f64(sums_f64x2, vcvt_f64_f32(vget_low_f32(high_f32x4)));
        sums_f64x2 = vaddq_f64(sums_f64x2, vcvt_high_f64_f32(high_f32x4));
    }
    state->sum_f64x2 = vaddq_f64(state->sum_f64x2, vmulq_n_f64(sums_f64x2, a.scale * b.scale));
}

#pragma region F32 and F64 Floats

/** Dot product of F32 vectors, widened to F64 and accumulated in F64. */
NUMKONG_INLINE void nk_dot_f32_through_f64_neon_(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars,
                                                 nk_size_t count_scalars, nk_f64_t *result) {
    // Upcast f32 to f64 via FCVTL/FCVTL2, two independent FMA chains for ILP
    float64x2_t sum_low_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_high_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count_scalars; idx_scalars += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a_scalars + idx_scalars);
        float32x4_t b_f32x4 = vld1q_f32(b_scalars + idx_scalars);
        float64x2_t a_low_f64x2 = vcvt_f64_f32(vget_low_f32(a_f32x4));
        float64x2_t a_high_f64x2 = vcvt_high_f64_f32(a_f32x4);
        float64x2_t b_low_f64x2 = vcvt_f64_f32(vget_low_f32(b_f32x4));
        float64x2_t b_high_f64x2 = vcvt_high_f64_f32(b_f32x4);
        sum_low_f64x2 = vfmaq_f64(sum_low_f64x2, a_low_f64x2, b_low_f64x2);
        sum_high_f64x2 = vfmaq_f64(sum_high_f64x2, a_high_f64x2, b_high_f64x2);
    }
    nk_f64_t sum_f64 = vaddvq_f64(vaddq_f64(sum_low_f64x2, sum_high_f64x2));
    for (; idx_scalars < count_scalars; ++idx_scalars)
        sum_f64 += (nk_f64_t)a_scalars[idx_scalars] * (nk_f64_t)b_scalars[idx_scalars];
    *result = sum_f64;
}

/**
 *  @brief Running state for 64-bit dot accumulation over f32 scalars on NEON.
 *
 *  Processes 2 f32 values at a time, upcasting to f64 for accumulation to avoid
 *  catastrophic cancellation in long reductions.
 */
typedef struct nk_dot_f32x2_state_neon_t {
    float64x2_t sum_f64x2;
} nk_dot_f32x2_state_neon_t;

NUMKONG_INLINE void nk_dot_f32x2_init_neon(nk_dot_f32x2_state_neon_t *state) { state->sum_f64x2 = vdupq_n_f64(0); }

NUMKONG_INLINE void nk_dot_f32x2_update_neon(nk_dot_f32x2_state_neon_t *state, nk_b64_vec_t a, nk_b64_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Upcast 2 f32s to f64s for high-precision accumulation
    float32x2_t a_f32x2 = vreinterpret_f32_u32(a.u32x2);
    float32x2_t b_f32x2 = vreinterpret_f32_u32(b.u32x2);
    float64x2_t a_f64x2 = vcvt_f64_f32(a_f32x2);
    float64x2_t b_f64x2 = vcvt_f64_f32(b_f32x2);
    state->sum_f64x2 = vfmaq_f64(state->sum_f64x2, a_f64x2, b_f64x2);
}

NUMKONG_INLINE void nk_dot_f32x2_finalize_neon(                                         //
    nk_dot_f32x2_state_neon_t const *state_a, nk_dot_f32x2_state_neon_t const *state_b, //
    nk_dot_f32x2_state_neon_t const *state_c, nk_dot_f32x2_state_neon_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    float64x2_t ab_f64x2 = vpaddq_f64(state_a->sum_f64x2, state_b->sum_f64x2);
    float64x2_t cd_f64x2 = vpaddq_f64(state_c->sum_f64x2, state_d->sum_f64x2);
    vst1q_f64(&result->f64s[0], ab_f64x2);
    vst1q_f64(&result->f64s[2], cd_f64x2);
}

/**
 *  @brief Running state for 128-bit dot accumulation over f64 scalars on NEON.
 *
 *  Uses the Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product.
 */
typedef struct nk_dot_f64x2_state_neon_t {
    float64x2_t sum_f64x2;
    float64x2_t compensation_f64x2;
} nk_dot_f64x2_state_neon_t;

NUMKONG_INLINE void nk_dot_f64x2_init_neon(nk_dot_f64x2_state_neon_t *state) {
    state->sum_f64x2 = vdupq_n_f64(0);
    state->compensation_f64x2 = vdupq_n_f64(0);
}

NUMKONG_INLINE void nk_dot_f64x2_update_neon(nk_dot_f64x2_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    float64x2_t sum_f64x2 = state->sum_f64x2;
    float64x2_t compensation_f64x2 = state->compensation_f64x2;
    float64x2_t a_f64x2 = vreinterpretq_f64_u64(a.u64x2);
    float64x2_t b_f64x2 = vreinterpretq_f64_u64(b.u64x2);

    // TwoProd: h = a × b, r = fma(a, b, -h) captures the rounding error
    float64x2_t product_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
    float64x2_t product_error_f64x2 = vnegq_f64(vfmsq_f64(product_f64x2, a_f64x2, b_f64x2));

    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    float64x2_t tentative_sum_f64x2 = vaddq_f64(sum_f64x2, product_f64x2);
    float64x2_t virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, sum_f64x2);
    float64x2_t sum_error_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
                                            vsubq_f64(product_f64x2, virtual_addend_f64x2));

    // Update: sum = t, compensation += q + r
    state->sum_f64x2 = tentative_sum_f64x2;
    state->compensation_f64x2 = vaddq_f64(compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));
}

NUMKONG_INLINE void nk_dot_f64x2_finalize_neon(                                         //
    nk_dot_f64x2_state_neon_t const *state_a, nk_dot_f64x2_state_neon_t const *state_b, //
    nk_dot_f64x2_state_neon_t const *state_c, nk_dot_f64x2_state_neon_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    // Compensated horizontal reduction preserving Dot2 error tracking per state
    result->f64s[0] = nk_dot_stable_sum_f64x2_neon_(state_a->sum_f64x2, state_a->compensation_f64x2);
    result->f64s[1] = nk_dot_stable_sum_f64x2_neon_(state_b->sum_f64x2, state_b->compensation_f64x2);
    result->f64s[2] = nk_dot_stable_sum_f64x2_neon_(state_c->sum_f64x2, state_c->compensation_f64x2);
    result->f64s[3] = nk_dot_stable_sum_f64x2_neon_(state_d->sum_f64x2, state_d->compensation_f64x2);
}

#pragma endregion F32 and F64 Floats

#pragma region F16 and BF16 Floats

/** Dot product of BF16 vectors, widened to F32 by shifts and accumulated in F32. */
NUMKONG_INLINE void nk_dot_bf16_through_f32_neon_(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars,
                                                  nk_size_t count_scalars, nk_f32_t *result) {
    uint16x8_t a_u16x8, b_u16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_bf16_through_f32_neon_cycle:
    if (count_scalars < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_vec, count_scalars);
        a_u16x8 = a_vec.u16x8;
        b_u16x8 = b_vec.u16x8;
        count_scalars = 0;
    }
    else {
        a_u16x8 = vld1q_u16((nk_u16_t const *)a_scalars);
        b_u16x8 = vld1q_u16((nk_u16_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    float32x4_t a_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a_u16x8), 16));
    float32x4_t a_high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(a_u16x8, 16));
    float32x4_t b_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b_u16x8), 16));
    float32x4_t b_high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(b_u16x8, 16));
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, b_high_f32x4);
    if (count_scalars) goto nk_dot_bf16_through_f32_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars on plain NEON.
 *
 *  Processes 8 bf16 values at a time (128 bits), converting to f32 via USHLL shift-16
 *  for accumulation without requiring the ARMv8.6-BF16 extension.
 */
typedef struct nk_dot_bf16x8_state_neon_t {
    float32x4_t sum_f32x4;
} nk_dot_bf16x8_state_neon_t;

NUMKONG_INLINE void nk_dot_bf16x8_init_neon(nk_dot_bf16x8_state_neon_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NUMKONG_INLINE void nk_dot_bf16x8_update_neon(nk_dot_bf16x8_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Convert bf16 to f32 via USHLL shift-16 (low and high halves)
    float32x4_t a_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a.u16x8), 16));
    float32x4_t a_high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(a.u16x8, 16));
    float32x4_t b_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b.u16x8), 16));
    float32x4_t b_high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(b.u16x8, 16));
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, a_low_f32x4, b_low_f32x4);
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, a_high_f32x4, b_high_f32x4);
}

NUMKONG_INLINE void nk_dot_bf16x8_finalize_neon(                                          //
    nk_dot_bf16x8_state_neon_t const *state_a, nk_dot_bf16x8_state_neon_t const *state_b, //
    nk_dot_bf16x8_state_neon_t const *state_c, nk_dot_bf16x8_state_neon_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    float32x4_t ab_f32x4 = vpaddq_f32(state_a->sum_f32x4, state_b->sum_f32x4);
    float32x4_t cd_f32x4 = vpaddq_f32(state_c->sum_f32x4, state_d->sum_f32x4);
    result->f32x4 = vpaddq_f32(ab_f32x4, cd_f32x4);
}

/** Dot product of F16 vectors, widened to F32 and accumulated in F32. */
NUMKONG_INLINE void nk_dot_f16_through_f32_neon_(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars,
                                                 nk_size_t count_scalars, nk_f32_t *result) {
    uint16x8_t a_u16x8, b_u16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_f16_through_f32_neon_cycle:
    if (count_scalars < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_vec, count_scalars);
        a_u16x8 = a_vec.u16x8;
        b_u16x8 = b_vec.u16x8;
        count_scalars = 0;
    }
    else {
        a_u16x8 = vld1q_u16((nk_u16_t const *)a_scalars);
        b_u16x8 = vld1q_u16((nk_u16_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    float16x8_t a_f16x8 = vreinterpretq_f16_u16(a_u16x8);
    float16x8_t b_f16x8 = vreinterpretq_f16_u16(b_u16x8);
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, b_high_f32x4);
    if (count_scalars) goto nk_dot_f16_through_f32_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars on plain NEON.
 *
 *  Processes 8 f16 values at a time (128 bits), converting to f32 via FCVTL
 *  for accumulation without requiring the ARMv8.2-A FP16 arithmetic extension.
 */
typedef struct nk_dot_f16x8_state_neon_t {
    float32x4_t sum_f32x4;
} nk_dot_f16x8_state_neon_t;

NUMKONG_INLINE void nk_dot_f16x8_init_neon(nk_dot_f16x8_state_neon_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NUMKONG_INLINE void nk_dot_f16x8_update_neon(nk_dot_f16x8_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Convert f16 to f32 via FCVTL / FCVTL2 (low and high halves)
    float16x8_t a_f16x8 = vreinterpretq_f16_u16(a.u16x8);
    float16x8_t b_f16x8 = vreinterpretq_f16_u16(b.u16x8);
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, a_low_f32x4, b_low_f32x4);
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, a_high_f32x4, b_high_f32x4);
}

NUMKONG_INLINE void nk_dot_f16x8_finalize_neon(                                         //
    nk_dot_f16x8_state_neon_t const *state_a, nk_dot_f16x8_state_neon_t const *state_b, //
    nk_dot_f16x8_state_neon_t const *state_c, nk_dot_f16x8_state_neon_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    float32x4_t ab_f32x4 = vpaddq_f32(state_a->sum_f32x4, state_b->sum_f32x4);
    float32x4_t cd_f32x4 = vpaddq_f32(state_c->sum_f32x4, state_d->sum_f32x4);
    result->f32x4 = vpaddq_f32(ab_f32x4, cd_f32x4);
}

#pragma endregion F16 and BF16 Floats

#pragma region Binary

typedef struct nk_dot_u1x128_state_neon_t {
    uint32x4_t dot_count_u32x4;
} nk_dot_u1x128_state_neon_t;

NUMKONG_INLINE void nk_dot_u1x128_init_neon(nk_dot_u1x128_state_neon_t *state) {
    state->dot_count_u32x4 = vdupq_n_u32(0);
}

NUMKONG_INLINE void nk_dot_u1x128_update_neon(nk_dot_u1x128_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    uint8x16_t and_u8x16 = vandq_u8(a.u8x16, b.u8x16);
    uint8x16_t popcount_u8x16 = vcntq_u8(and_u8x16);
    uint16x8_t popcount_u16x8 = vpaddlq_u8(popcount_u8x16);
    uint32x4_t popcount_u32x4 = vpaddlq_u16(popcount_u16x8);
    state->dot_count_u32x4 = vaddq_u32(state->dot_count_u32x4, popcount_u32x4);
}

NUMKONG_INLINE void nk_dot_u1x128_finalize_neon( //
    nk_dot_u1x128_state_neon_t const *state_a, nk_dot_u1x128_state_neon_t const *state_b,
    nk_dot_u1x128_state_neon_t const *state_c, nk_dot_u1x128_state_neon_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    uint32x4_t ab_sum_u32x4 = vpaddq_u32(state_a->dot_count_u32x4, state_b->dot_count_u32x4);
    uint32x4_t cd_sum_u32x4 = vpaddq_u32(state_c->dot_count_u32x4, state_d->dot_count_u32x4);
    result->u32x4 = vpaddq_u32(ab_sum_u32x4, cd_sum_u32x4);
}

#pragma endregion Binary

#if NUMKONG_TARGET_NEON

#pragma region F32 and F64 Floats

NUMKONG_API nk_status_t nk_dot_f32_neon(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dot_f32_through_f64_neon_(a_scalars, b_scalars, count_scalars, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dot_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                         nk_f64c_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Upcast f32 to f64 for accumulation (2 complex pairs per iteration, avoids slow vget_low/high)
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_pairs = 0;
    // ARMv8.3-A FCMLA (`vcmlaq_rot0/rot90_f32`) was benchmarked as an alternative to the
    // deinterleave+4FMA pattern below. FCMLA processes only 2 complex pairs per iteration
    // (interleaved 128-bit operands, 2x `vcmlaq`), while `vld2_f32` deinterleaves 2 pairs
    // with 4 independent FMA instructions that fully utilize M4's 4 SIMD pipes. Result on
    // Apple M4 at n=4096: manual f32 39.7 GB/s, FCMLA 17.1 GB/s (2.3x slower).
    // The f64 upcast here trades throughput for precision — FCMLA offers neither advantage.
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        // Unpack 2 complex pairs into real and imaginary parts:
        float32x2x2_t a_f32x2x2 = vld2_f32((nk_f32_t const *)(a_pairs + idx_pairs));
        float32x2x2_t b_f32x2x2 = vld2_f32((nk_f32_t const *)(b_pairs + idx_pairs));
        // Upcast to f64
        float64x2_t a_real_f64x2 = vcvt_f64_f32(a_f32x2x2.val[0]);
        float64x2_t a_imag_f64x2 = vcvt_f64_f32(a_f32x2x2.val[1]);
        float64x2_t b_real_f64x2 = vcvt_f64_f32(b_f32x2x2.val[0]);
        float64x2_t b_imag_f64x2 = vcvt_f64_f32(b_f32x2x2.val[1]);
        // Compute the dot product: real = aᵣ × bᵣ - aᵢ × bᵢ, imag = aᵣ × bᵢ + aᵢ × bᵣ
        sum_real_f64x2 = vfmaq_f64(sum_real_f64x2, a_real_f64x2, b_real_f64x2);
        sum_real_f64x2 = vfmsq_f64(sum_real_f64x2, a_imag_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmaq_f64(sum_imag_f64x2, a_real_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmaq_f64(sum_imag_f64x2, a_imag_f64x2, b_real_f64x2);
    }
    // Reduce horizontal sums:
    nk_f64_t sum_real_f64 = vaddvq_f64(sum_real_f64x2);
    nk_f64_t sum_imag_f64 = vaddvq_f64(sum_imag_f64x2);
    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f64_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real_f64 += ar * br - ai * bi;
        sum_imag_f64 += ar * bi + ai * br;
    }
    result->real = sum_real_f64;
    result->imag = sum_imag_f64;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vdot_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                          nk_f64c_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Upcast f32 to f64 for accumulation (2 complex pairs per iteration, avoids slow vget_low/high)
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        // Unpack 2 complex pairs into real and imaginary parts:
        float32x2x2_t a_f32x2x2 = vld2_f32((nk_f32_t const *)(a_pairs + idx_pairs));
        float32x2x2_t b_f32x2x2 = vld2_f32((nk_f32_t const *)(b_pairs + idx_pairs));
        // Upcast to f64
        float64x2_t a_real_f64x2 = vcvt_f64_f32(a_f32x2x2.val[0]);
        float64x2_t a_imag_f64x2 = vcvt_f64_f32(a_f32x2x2.val[1]);
        float64x2_t b_real_f64x2 = vcvt_f64_f32(b_f32x2x2.val[0]);
        float64x2_t b_imag_f64x2 = vcvt_f64_f32(b_f32x2x2.val[1]);
        // Compute conjugate dot product: real = aᵣ × bᵣ + aᵢ × bᵢ, imag = aᵣ × bᵢ - aᵢ × bᵣ
        sum_real_f64x2 = vfmaq_f64(sum_real_f64x2, a_real_f64x2, b_real_f64x2);
        sum_real_f64x2 = vfmaq_f64(sum_real_f64x2, a_imag_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmaq_f64(sum_imag_f64x2, a_real_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmsq_f64(sum_imag_f64x2, a_imag_f64x2, b_real_f64x2);
    }
    // Reduce horizontal sums:
    nk_f64_t sum_real_f64 = vaddvq_f64(sum_real_f64x2);
    nk_f64_t sum_imag_f64 = vaddvq_f64(sum_imag_f64x2);
    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f64_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real_f64 += ar * br + ai * bi;
        sum_imag_f64 += ar * bi - ai * br;
    }
    result->real = sum_real_f64;
    result->imag = sum_imag_f64;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dot_f64_neon(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t a_f64x2, b_f64x2;

nk_dot_f64_neon_cycle:
    if (count_scalars < 2) {
        nk_b128_vec_t a_tail, b_tail;
        nk_partial_load_b64x2_serial_(a_scalars, &a_tail, count_scalars);
        nk_partial_load_b64x2_serial_(b_scalars, &b_tail, count_scalars);
        a_f64x2 = a_tail.f64x2;
        b_f64x2 = b_tail.f64x2;
        count_scalars = 0;
    }
    else {
        a_f64x2 = vld1q_f64(a_scalars);
        b_f64x2 = vld1q_f64(b_scalars);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }

    // TwoProd: h = a × b, r = fma(a, b, -h) captures the rounding error
    float64x2_t product_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
    float64x2_t product_error_f64x2 = vnegq_f64(vfmsq_f64(product_f64x2, a_f64x2, b_f64x2));
    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    float64x2_t tentative_sum_f64x2 = vaddq_f64(sum_f64x2, product_f64x2);
    float64x2_t virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, sum_f64x2);
    float64x2_t sum_error_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
                                            vsubq_f64(product_f64x2, virtual_addend_f64x2));
    // Update: sum = t, compensation += q + r
    sum_f64x2 = tentative_sum_f64x2;
    compensation_f64x2 = vaddq_f64(compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));

    if (count_scalars) goto nk_dot_f64_neon_cycle;
    // Compensated horizontal reduction preserving Dot2 error tracking
    *result = nk_dot_stable_sum_f64x2_neon_(sum_f64x2, compensation_f64x2);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dot_f64c_neon(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                         nk_f64c_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated complex dot product
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_real_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t a_real_f64x2, a_imag_f64x2, b_real_f64x2, b_imag_f64x2;

nk_dot_f64c_neon_cycle:
    if (count_pairs < 2) {
        nk_b128_vec_t a_tail, b_tail;
        nk_partial_load_b64x2_serial_(a_pairs, &a_tail, count_pairs * 2);
        nk_partial_load_b64x2_serial_(b_pairs, &b_tail, count_pairs * 2);
        float64x2_t zeros_f64x2 = vdupq_n_f64(0);
        a_real_f64x2 = vzip1q_f64(a_tail.f64x2, zeros_f64x2);
        a_imag_f64x2 = vzip2q_f64(a_tail.f64x2, zeros_f64x2);
        b_real_f64x2 = vzip1q_f64(b_tail.f64x2, zeros_f64x2);
        b_imag_f64x2 = vzip2q_f64(b_tail.f64x2, zeros_f64x2);
        count_pairs = 0;
    }
    else {
        float64x2x2_t a_f64x2x2 = vld2q_f64((nk_f64_t const *)a_pairs);
        float64x2x2_t b_f64x2x2 = vld2q_f64((nk_f64_t const *)b_pairs);
        a_real_f64x2 = a_f64x2x2.val[0];
        a_imag_f64x2 = a_f64x2x2.val[1];
        b_real_f64x2 = b_f64x2x2.val[0];
        b_imag_f64x2 = b_f64x2x2.val[1];
        a_pairs += 2, b_pairs += 2, count_pairs -= 2;
    }

    // Real part: aᵣ × bᵣ - aᵢ × bᵢ (using TwoProd and TwoSum)
    // First term: +aᵣ × bᵣ
    float64x2_t product_rr_f64x2 = vmulq_f64(a_real_f64x2, b_real_f64x2);
    float64x2_t error_rr_f64x2 = vnegq_f64(vfmsq_f64(product_rr_f64x2, a_real_f64x2, b_real_f64x2));
    float64x2_t tentative_sum_real_f64x2 = vaddq_f64(sum_real_f64x2, product_rr_f64x2);
    float64x2_t virtual_addend_real_f64x2 = vsubq_f64(tentative_sum_real_f64x2, sum_real_f64x2);
    float64x2_t error_sum_real_f64x2 = vaddq_f64(
        vsubq_f64(sum_real_f64x2, vsubq_f64(tentative_sum_real_f64x2, virtual_addend_real_f64x2)),
        vsubq_f64(product_rr_f64x2, virtual_addend_real_f64x2));
    sum_real_f64x2 = tentative_sum_real_f64x2;
    compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(error_sum_real_f64x2, error_rr_f64x2));
    // Second term: -aᵢ × bᵢ (negate product and error, then standard TwoSum)
    float64x2_t product_ii_f64x2 = vmulq_f64(a_imag_f64x2, b_imag_f64x2);
    float64x2_t error_ii_f64x2 = vnegq_f64(vfmsq_f64(product_ii_f64x2, a_imag_f64x2, b_imag_f64x2));
    float64x2_t neg_product_ii_f64x2 = vnegq_f64(product_ii_f64x2);
    float64x2_t neg_error_ii_f64x2 = vnegq_f64(error_ii_f64x2);
    tentative_sum_real_f64x2 = vaddq_f64(sum_real_f64x2, neg_product_ii_f64x2);
    virtual_addend_real_f64x2 = vsubq_f64(tentative_sum_real_f64x2, sum_real_f64x2);
    error_sum_real_f64x2 = vaddq_f64(
        vsubq_f64(sum_real_f64x2, vsubq_f64(tentative_sum_real_f64x2, virtual_addend_real_f64x2)),
        vsubq_f64(neg_product_ii_f64x2, virtual_addend_real_f64x2));
    sum_real_f64x2 = tentative_sum_real_f64x2;
    compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(error_sum_real_f64x2, neg_error_ii_f64x2));

    // Imag part: aᵣ × bᵢ + aᵢ × bᵣ (using TwoProd and TwoSum)
    // First term: +aᵣ × bᵢ
    float64x2_t product_ri_f64x2 = vmulq_f64(a_real_f64x2, b_imag_f64x2);
    float64x2_t error_ri_f64x2 = vnegq_f64(vfmsq_f64(product_ri_f64x2, a_real_f64x2, b_imag_f64x2));
    float64x2_t tentative_sum_imag_f64x2 = vaddq_f64(sum_imag_f64x2, product_ri_f64x2);
    float64x2_t virtual_addend_imag_f64x2 = vsubq_f64(tentative_sum_imag_f64x2, sum_imag_f64x2);
    float64x2_t error_sum_imag_f64x2 = vaddq_f64(
        vsubq_f64(sum_imag_f64x2, vsubq_f64(tentative_sum_imag_f64x2, virtual_addend_imag_f64x2)),
        vsubq_f64(product_ri_f64x2, virtual_addend_imag_f64x2));
    sum_imag_f64x2 = tentative_sum_imag_f64x2;
    compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(error_sum_imag_f64x2, error_ri_f64x2));
    // Second term: +aᵢ × bᵣ
    float64x2_t product_ir_f64x2 = vmulq_f64(a_imag_f64x2, b_real_f64x2);
    float64x2_t error_ir_f64x2 = vnegq_f64(vfmsq_f64(product_ir_f64x2, a_imag_f64x2, b_real_f64x2));
    tentative_sum_imag_f64x2 = vaddq_f64(sum_imag_f64x2, product_ir_f64x2);
    virtual_addend_imag_f64x2 = vsubq_f64(tentative_sum_imag_f64x2, sum_imag_f64x2);
    error_sum_imag_f64x2 = vaddq_f64(
        vsubq_f64(sum_imag_f64x2, vsubq_f64(tentative_sum_imag_f64x2, virtual_addend_imag_f64x2)),
        vsubq_f64(product_ir_f64x2, virtual_addend_imag_f64x2));
    sum_imag_f64x2 = tentative_sum_imag_f64x2;
    compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(error_sum_imag_f64x2, error_ir_f64x2));

    if (count_pairs) goto nk_dot_f64c_neon_cycle;
    // Compensated horizontal reduction preserving Dot2 error tracking
    result->real = nk_dot_stable_sum_f64x2_neon_(sum_real_f64x2, compensation_real_f64x2);
    result->imag = nk_dot_stable_sum_f64x2_neon_(sum_imag_f64x2, compensation_imag_f64x2);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vdot_f64c_neon(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                          nk_f64c_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated conjugate dot product
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_real_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t a_real_f64x2, a_imag_f64x2, b_real_f64x2, b_imag_f64x2;

nk_vdot_f64c_neon_cycle:
    if (count_pairs < 2) {
        nk_b128_vec_t a_tail, b_tail;
        nk_partial_load_b64x2_serial_(a_pairs, &a_tail, count_pairs * 2);
        nk_partial_load_b64x2_serial_(b_pairs, &b_tail, count_pairs * 2);
        float64x2_t zeros_f64x2 = vdupq_n_f64(0);
        a_real_f64x2 = vzip1q_f64(a_tail.f64x2, zeros_f64x2);
        a_imag_f64x2 = vzip2q_f64(a_tail.f64x2, zeros_f64x2);
        b_real_f64x2 = vzip1q_f64(b_tail.f64x2, zeros_f64x2);
        b_imag_f64x2 = vzip2q_f64(b_tail.f64x2, zeros_f64x2);
        count_pairs = 0;
    }
    else {
        float64x2x2_t a_f64x2x2 = vld2q_f64((nk_f64_t const *)a_pairs);
        float64x2x2_t b_f64x2x2 = vld2q_f64((nk_f64_t const *)b_pairs);
        a_real_f64x2 = a_f64x2x2.val[0];
        a_imag_f64x2 = a_f64x2x2.val[1];
        b_real_f64x2 = b_f64x2x2.val[0];
        b_imag_f64x2 = b_f64x2x2.val[1];
        a_pairs += 2, b_pairs += 2, count_pairs -= 2;
    }

    // Real part: aᵣ × bᵣ + aᵢ × bᵢ (using TwoProd and TwoSum)
    // First term: +aᵣ × bᵣ
    float64x2_t product_rr_f64x2 = vmulq_f64(a_real_f64x2, b_real_f64x2);
    float64x2_t error_rr_f64x2 = vnegq_f64(vfmsq_f64(product_rr_f64x2, a_real_f64x2, b_real_f64x2));
    float64x2_t tentative_sum_real_f64x2 = vaddq_f64(sum_real_f64x2, product_rr_f64x2);
    float64x2_t virtual_addend_real_f64x2 = vsubq_f64(tentative_sum_real_f64x2, sum_real_f64x2);
    float64x2_t error_sum_real_f64x2 = vaddq_f64(
        vsubq_f64(sum_real_f64x2, vsubq_f64(tentative_sum_real_f64x2, virtual_addend_real_f64x2)),
        vsubq_f64(product_rr_f64x2, virtual_addend_real_f64x2));
    sum_real_f64x2 = tentative_sum_real_f64x2;
    compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(error_sum_real_f64x2, error_rr_f64x2));
    // Second term: +aᵢ × bᵢ (conjugate: add instead of subtract)
    float64x2_t product_ii_f64x2 = vmulq_f64(a_imag_f64x2, b_imag_f64x2);
    float64x2_t error_ii_f64x2 = vnegq_f64(vfmsq_f64(product_ii_f64x2, a_imag_f64x2, b_imag_f64x2));
    tentative_sum_real_f64x2 = vaddq_f64(sum_real_f64x2, product_ii_f64x2);
    virtual_addend_real_f64x2 = vsubq_f64(tentative_sum_real_f64x2, sum_real_f64x2);
    error_sum_real_f64x2 = vaddq_f64(
        vsubq_f64(sum_real_f64x2, vsubq_f64(tentative_sum_real_f64x2, virtual_addend_real_f64x2)),
        vsubq_f64(product_ii_f64x2, virtual_addend_real_f64x2));
    sum_real_f64x2 = tentative_sum_real_f64x2;
    compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(error_sum_real_f64x2, error_ii_f64x2));

    // Imag part: aᵣ × bᵢ - aᵢ × bᵣ (using TwoProd and TwoSum)
    // First term: +aᵣ × bᵢ
    float64x2_t product_ri_f64x2 = vmulq_f64(a_real_f64x2, b_imag_f64x2);
    float64x2_t error_ri_f64x2 = vnegq_f64(vfmsq_f64(product_ri_f64x2, a_real_f64x2, b_imag_f64x2));
    float64x2_t tentative_sum_imag_f64x2 = vaddq_f64(sum_imag_f64x2, product_ri_f64x2);
    float64x2_t virtual_addend_imag_f64x2 = vsubq_f64(tentative_sum_imag_f64x2, sum_imag_f64x2);
    float64x2_t error_sum_imag_f64x2 = vaddq_f64(
        vsubq_f64(sum_imag_f64x2, vsubq_f64(tentative_sum_imag_f64x2, virtual_addend_imag_f64x2)),
        vsubq_f64(product_ri_f64x2, virtual_addend_imag_f64x2));
    sum_imag_f64x2 = tentative_sum_imag_f64x2;
    compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(error_sum_imag_f64x2, error_ri_f64x2));
    // Second term: -aᵢ × bᵣ (conjugate: negate product and error, then standard TwoSum)
    float64x2_t product_ir_f64x2 = vmulq_f64(a_imag_f64x2, b_real_f64x2);
    float64x2_t error_ir_f64x2 = vnegq_f64(vfmsq_f64(product_ir_f64x2, a_imag_f64x2, b_real_f64x2));
    float64x2_t neg_product_ir_f64x2 = vnegq_f64(product_ir_f64x2);
    float64x2_t neg_error_ir_f64x2 = vnegq_f64(error_ir_f64x2);
    tentative_sum_imag_f64x2 = vaddq_f64(sum_imag_f64x2, neg_product_ir_f64x2);
    virtual_addend_imag_f64x2 = vsubq_f64(tentative_sum_imag_f64x2, sum_imag_f64x2);
    error_sum_imag_f64x2 = vaddq_f64(
        vsubq_f64(sum_imag_f64x2, vsubq_f64(tentative_sum_imag_f64x2, virtual_addend_imag_f64x2)),
        vsubq_f64(neg_product_ir_f64x2, virtual_addend_imag_f64x2));
    sum_imag_f64x2 = tentative_sum_imag_f64x2;
    compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(error_sum_imag_f64x2, neg_error_ir_f64x2));

    if (count_pairs) goto nk_vdot_f64c_neon_cycle;
    // Compensated horizontal reduction preserving Dot2 error tracking
    result->real = nk_dot_stable_sum_f64x2_neon_(sum_real_f64x2, compensation_real_f64x2);
    result->imag = nk_dot_stable_sum_f64x2_neon_(sum_imag_f64x2, compensation_imag_f64x2);
    return nk_success_k;
}
#pragma endregion F32 and F64 Floats

#pragma region F16 and BF16 Floats

NUMKONG_API nk_status_t nk_dot_bf16_neon(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dot_bf16_through_f32_neon_(a_scalars, b_scalars, count_scalars, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dot_f16_neon(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dot_f16_through_f32_neon_(a_scalars, b_scalars, count_scalars, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dot_e4m3_neon(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e4m3_neon_cycle:
    if (count_scalars < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x8_serial_(b_scalars, &b_vec, count_scalars);
        a_f16x8 = nk_e4m3x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e4m3x8_to_f16x8_neon_(b_vec.u8x8);
        count_scalars = 0;
    }
    else {
        a_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(a_scalars));
        b_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(b_scalars));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, b_high_f32x4);
    if (count_scalars) goto nk_dot_e4m3_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dot_e5m2_neon(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e5m2_neon_cycle:
    if (count_scalars < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x8_serial_(b_scalars, &b_vec, count_scalars);
        a_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(a_vec.u8x8, 8));
        b_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(b_vec.u8x8, 8));
        count_scalars = 0;
    }
    else {
        a_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(vld1_u8(a_scalars), 8));
        b_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(vld1_u8(b_scalars), 8));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, b_high_f32x4);
    if (count_scalars) goto nk_dot_e5m2_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dot_e2m3_neon(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float16x8_t a_low_f16x8, a_high_f16x8, b_low_f16x8, b_high_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    // x16 TBL path: process 16 elements per iteration via lookup table upcast
nk_dot_e2m3_neon_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        nk_e2m3x16_to_f16x8x2_neon_(a_vec.u8x16, &a_low_f16x8, &a_high_f16x8);
        nk_e2m3x16_to_f16x8x2_neon_(b_vec.u8x16, &b_low_f16x8, &b_high_f16x8);
        count_scalars = 0;
    }
    else {
        nk_e2m3x16_to_f16x8x2_neon_(vld1q_u8(a_scalars), &a_low_f16x8, &a_high_f16x8);
        nk_e2m3x16_to_f16x8x2_neon_(vld1q_u8(b_scalars), &b_low_f16x8, &b_high_f16x8);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_low_f16(a_low_f16x8)), vcvt_f32_f16(vget_low_f16(b_low_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_high_f32_f16(a_low_f16x8), vcvt_high_f32_f16(b_low_f16x8));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_low_f16(a_high_f16x8)),
                          vcvt_f32_f16(vget_low_f16(b_high_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_high_f32_f16(a_high_f16x8), vcvt_high_f32_f16(b_high_f16x8));
    if (count_scalars) goto nk_dot_e2m3_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dot_e3m2_neon(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float16x8_t a_low_f16x8, a_high_f16x8, b_low_f16x8, b_high_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    // x16 TBL path: process 16 elements per iteration via lookup table upcast
nk_dot_e3m2_neon_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        nk_e3m2x16_to_f16x8x2_neon_(a_vec.u8x16, &a_low_f16x8, &a_high_f16x8);
        nk_e3m2x16_to_f16x8x2_neon_(b_vec.u8x16, &b_low_f16x8, &b_high_f16x8);
        count_scalars = 0;
    }
    else {
        nk_e3m2x16_to_f16x8x2_neon_(vld1q_u8(a_scalars), &a_low_f16x8, &a_high_f16x8);
        nk_e3m2x16_to_f16x8x2_neon_(vld1q_u8(b_scalars), &b_low_f16x8, &b_high_f16x8);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_low_f16(a_low_f16x8)), vcvt_f32_f16(vget_low_f16(b_low_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_high_f32_f16(a_low_f16x8), vcvt_high_f32_f16(b_low_f16x8));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_low_f16(a_high_f16x8)),
                          vcvt_f32_f16(vget_low_f16(b_high_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_high_f32_f16(a_high_f16x8), vcvt_high_f32_f16(b_high_f16x8));
    if (count_scalars) goto nk_dot_e3m2_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
    return nk_success_k;
}
#pragma endregion F16 and BF16 Floats

#pragma region Binary

NUMKONG_API nk_status_t nk_dot_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result,
                                       void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_bytes = n_bits / NUMKONG_BITS_PER_BYTE;
    nk_u32_t dot = 0;
    nk_size_t i = 0;
    while (i + 16 <= n_bytes) {
        uint8x16_t popcount_u8x16 = vdupq_n_u8(0);
        for (nk_size_t cycle = 0; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            popcount_u8x16 = vaddq_u8(popcount_u8x16, vcntq_u8(vandq_u8(a_u8x16, b_u8x16)));
        }
        dot += (nk_u32_t)vaddlvq_u8(popcount_u8x16);
    }
    for (; i != n_bytes; ++i) dot += nk_u1x8_popcount_(a[i] & b[i]);
    *result = dot;
    return nk_success_k;
}
#pragma endregion Binary

NUMKONG_API nk_status_t nk_dot_f16c_neon(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                         nk_f32c_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
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
    nk_f32c_t tail_result;
    nk_dot_f16c_(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vdot_f16c_neon(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                          nk_f32c_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
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
    nk_f32c_t tail_result;
    nk_vdot_f16c_(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_DOT_NEON_H
