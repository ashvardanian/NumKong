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
 *  Block-scaled updates flush one exact integer or F32 block partial per block in F32.
 *  MX scales rebase to each row's and column's largest exponent less 31, so every scale product
 *  stays a normal F32 while a row and a column span 156 binades together.
 *  The epilogue applies both bases once.
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

#pragma region Block Scaled Floats

/** Block partials summed in F32 lanes, relative to their row and column bases until the end. */
typedef struct nk_dot_scaled_f32_state_neon_t {
    float32x4_t sum_f32x4;
} nk_dot_scaled_f32_state_neon_t;

/** Starts a sum at @p seed, the raw partial an earlier depth chunk stored. */
NUMKONG_INLINE void nk_dot_scaled_f32_init_neon(nk_dot_scaled_f32_state_neon_t *state, nk_f32_t seed) {
    state->sum_f32x4 = vsetq_lane_f32(seed, vdupq_n_f32(0), 0);
}

/** Four sums with their lanes added. */
NUMKONG_INLINE void nk_dot_scaled_f32_finalize_neon(nk_dot_scaled_f32_state_neon_t const *state_a,
                                                    nk_dot_scaled_f32_state_neon_t const *state_b,
                                                    nk_dot_scaled_f32_state_neon_t const *state_c,
                                                    nk_dot_scaled_f32_state_neon_t const *state_d,
                                                    nk_b128_vec_t *result) {
    result->f32x4 = vpaddq_f32(vpaddq_f32(state_a->sum_f32x4, state_b->sum_f32x4),
                               vpaddq_f32(state_c->sum_f32x4, state_d->sum_f32x4));
}

/** Multiplies lane i of @p values by @p mantissa and by two to the power of @p exponent plus
 *  @p lane_exponents[i] in three power-of-two steps, so only subnormal results round twice. */
NUMKONG_INLINE void nk_f32x4_scale_neon_(nk_b128_vec_t *values, nk_f32_t mantissa, nk_i32_t exponent,
                                         nk_i32_t const *lane_exponents) {
    int32x4_t exponents_i32x4 = vaddq_s32(vld1q_s32(lane_exponents), vdupq_n_s32(exponent));
    exponents_i32x4 = vmaxq_s32(vminq_s32(exponents_i32x4, vdupq_n_s32(381)), vdupq_n_s32(-378));
    float32x4_t values_f32x4 = vmulq_n_f32(values->f32x4, mantissa);
    for (int step = 0; step != 3; ++step) {
        int32x4_t const part_i32x4 = vmaxq_s32(vminq_s32(exponents_i32x4, vdupq_n_s32(127)), vdupq_n_s32(-126));
        int32x4_t const bits_i32x4 = vshlq_n_s32(vaddq_s32(part_i32x4, vdupq_n_s32(127)), 23);
        values_f32x4 = vmulq_f32(values_f32x4, vreinterpretq_f32_s32(bits_i32x4));
        exponents_i32x4 = vsubq_s32(exponents_i32x4, part_i32x4);
    }
    values->f32x4 = values_f32x4;
}

/** Lane i is two to the power of lane i of @p exponents_i32x4, each in [−126, 127]. */
NUMKONG_INLINE float32x4_t nk_f32x4_powers_of_two_neon_(int32x4_t exponents_i32x4) {
    return vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(exponents_i32x4, vdupq_n_s32(127)), 23));
}

/** Squared norms as mantissas times four to the power of @p halves_i32x4, clamped to [−63, 63]:
 *  normal norms take mantissas in [1, 4), and zeros, NaNs and infinities stay themselves. */
NUMKONG_INLINE float32x4_t nk_f32x4_split_squares_neon_(float32x4_t squares_f32x4, int32x4_t *halves_i32x4) {
    uint32x4_t const biased_u32x4 = vandq_u32(vshrq_n_u32(vreinterpretq_u32_f32(squares_f32x4), 23), vdupq_n_u32(0xFF));
    int32x4_t const exponents_i32x4 = vshrq_n_s32(vsubq_s32(vreinterpretq_s32_u32(biased_u32x4), vdupq_n_s32(127)), 1);
    *halves_i32x4 = vmaxq_s32(vminq_s32(exponents_i32x4, vdupq_n_s32(63)), vdupq_n_s32(-63));
    return vmulq_f32(squares_f32x4, nk_f32x4_powers_of_two_neon_(vmulq_n_s32(*halves_i32x4, -2)));
}

/*  Finishers of four relative sums, for rebased kernels and the SME epilogue alike: lane i's dot is
 *  values[i] · mantissa · 2^(row_exponent + column_exponents[i]), and its squared norms are
 *  row_norm · 4^row_exponent and column_norms[i] · 4^column_exponents[i]. */

/** Dot products; the norms go unused. */
NUMKONG_INLINE void nk_dot_f32x4_from_relative_neon_(nk_b128_vec_t *values, nk_f32_t mantissa, nk_i32_t row_exponent,
                                                     nk_i32_t const *column_exponents, nk_f32_t row_norm,
                                                     nk_f32_t const *column_norms) {
    nk_unused_(row_norm), nk_unused_(column_norms);
    nk_f32x4_scale_neon_(values, mantissa, row_exponent, column_exponents);
}

/** Angular distances, where the exponents cancel: 0 for two zero vectors, else 1 for a zero dot,
 *  else max(0, 1 − cosine); NaNs propagate. */
NUMKONG_INLINE void nk_angular_f32x4_from_relative_neon_(nk_b128_vec_t *values, nk_f32_t mantissa,
                                                         nk_i32_t row_exponent, nk_i32_t const *column_exponents,
                                                         nk_f32_t row_norm, nk_f32_t const *column_norms) {
    nk_unused_(row_exponent), nk_unused_(column_exponents);
    int32x4_t row_halves_i32x4, column_halves_i32x4;
    float32x4_t const row_norms_f32x4 = vdupq_n_f32(row_norm), column_norms_f32x4 = vld1q_f32(column_norms);
    float32x4_t const row_mantissas_f32x4 = nk_f32x4_split_squares_neon_(row_norms_f32x4, &row_halves_i32x4);
    float32x4_t const column_mantissas_f32x4 = nk_f32x4_split_squares_neon_(column_norms_f32x4, &column_halves_i32x4);
    float32x4_t const dots_f32x4 = vmulq_n_f32(values->f32x4, mantissa);
    // Over both halves the dot is the cosine times the root of the mantissas' product, at most 4
    float32x4_t const scaled_f32x4 = vmulq_f32(
        dots_f32x4, nk_f32x4_powers_of_two_neon_(vnegq_s32(vaddq_s32(row_halves_i32x4, column_halves_i32x4))));
    float32x4_t const cosines_f32x4 = vdivq_f32(scaled_f32x4,
                                                vsqrtq_f32(vmulq_f32(row_mantissas_f32x4, column_mantissas_f32x4)));
    float32x4_t angular_f32x4 = vmaxq_f32(vsubq_f32(vdupq_n_f32(1), cosines_f32x4), vdupq_n_f32(0));
    angular_f32x4 = vbslq_f32(vceqzq_f32(dots_f32x4), vdupq_n_f32(1), angular_f32x4);
    uint32x4_t const empty_u32x4 = vandq_u32(vceqzq_f32(row_norms_f32x4), vceqzq_f32(column_norms_f32x4));
    values->f32x4 = vbslq_f32(empty_u32x4, vdupq_n_f32(0), angular_f32x4);
}

/** Euclidean distances from dots over the leading squared norm's even power of two, clamped
 *  at zero, with the root scaled back by half that power and rounded once. A NaN in any
 *  operand gives NaN. */
NUMKONG_INLINE void nk_euclidean_f32x4_from_relative_neon_(nk_b128_vec_t *values, nk_f32_t mantissa,
                                                           nk_i32_t row_exponent, nk_i32_t const *column_exponents,
                                                           nk_f32_t row_norm, nk_f32_t const *column_norms) {
    int32x4_t row_halves_i32x4, column_halves_i32x4;
    float32x4_t const row_norms_f32x4 = vdupq_n_f32(row_norm), column_norms_f32x4 = vld1q_f32(column_norms);
    float32x4_t const row_mantissas_f32x4 = nk_f32x4_split_squares_neon_(row_norms_f32x4, &row_halves_i32x4);
    float32x4_t const column_mantissas_f32x4 = nk_f32x4_split_squares_neon_(column_norms_f32x4, &column_halves_i32x4);
    // Squared norms are their mantissas times 2^powers, even powers, and zero norms never lead
    int32x4_t const lowest_i32x4 = vdupq_n_s32(-126), floor_i32x4 = vdupq_n_s32(-(1 << 20));
    int32x4_t const row_powers_i32x4 = vbslq_s32(
        vceqzq_f32(row_norms_f32x4), floor_i32x4,
        vshlq_n_s32(vaddq_s32(vdupq_n_s32(row_exponent), row_halves_i32x4), 1));
    int32x4_t const column_powers_i32x4 = vbslq_s32(
        vceqzq_f32(column_norms_f32x4), floor_i32x4,
        vshlq_n_s32(vaddq_s32(vld1q_s32(column_exponents), column_halves_i32x4), 1));
    int32x4_t const top_i32x4 = vmaxq_s32(row_powers_i32x4, column_powers_i32x4);
    // Each term over 2^top is at most 8, so shifts clamped at 2^-126 only drop negligible bits
    float32x4_t const row_terms_f32x4 = vmulq_f32(
        row_mantissas_f32x4,
        nk_f32x4_powers_of_two_neon_(vmaxq_s32(vsubq_s32(row_powers_i32x4, top_i32x4), lowest_i32x4)));
    float32x4_t const column_terms_f32x4 = vmulq_f32(
        column_mantissas_f32x4,
        nk_f32x4_powers_of_two_neon_(vmaxq_s32(vsubq_s32(column_powers_i32x4, top_i32x4), lowest_i32x4)));
    int32x4_t const mean_powers_i32x4 = vshrq_n_s32(vaddq_s32(row_powers_i32x4, column_powers_i32x4), 1);
    float32x4_t dot_terms_f32x4 = vmulq_n_f32(values->f32x4, 2 * mantissa);
    dot_terms_f32x4 = vmulq_f32(
        dot_terms_f32x4, nk_f32x4_powers_of_two_neon_(vnegq_s32(vaddq_s32(row_halves_i32x4, column_halves_i32x4))));
    dot_terms_f32x4 = vmulq_f32(dot_terms_f32x4, nk_f32x4_powers_of_two_neon_(
                                                     vmaxq_s32(vsubq_s32(mean_powers_i32x4, top_i32x4), lowest_i32x4)));
    float32x4_t const squares_f32x4 = vmaxq_f32(
        vsubq_f32(vaddq_f32(row_terms_f32x4, column_terms_f32x4), dot_terms_f32x4), vdupq_n_f32(0));
    // Nonzero roots lie in 2^-75 … 4, so two powers of two reach every finite result
    int32x4_t const halves_i32x4 = vmaxq_s32(vminq_s32(vshrq_n_s32(top_i32x4, 1), vdupq_n_s32(254)), vdupq_n_s32(-252));
    int32x4_t const first_i32x4 = vmaxq_s32(vminq_s32(halves_i32x4, vdupq_n_s32(127)), lowest_i32x4);
    float32x4_t const roots_f32x4 = vmulq_f32(vsqrtq_f32(squares_f32x4), nk_f32x4_powers_of_two_neon_(first_i32x4));
    values->f32x4 = vmulq_f32(roots_f32x4, nk_f32x4_powers_of_two_neon_(vsubq_s32(halves_i32x4, first_i32x4)));
}

/** 64 i8-lifted elements, SDOT lane j of every register summing a 16-element group of one block,
 *  and their four F32 lane scales: one per NVFP4 block, or each MX block scale twice. */
typedef struct nk_dot_scaled_i8x64_operand_neon_t {
    int8x16_t values_i8x16[4];
    float32x4_t scales_f32x4;
} nk_dot_scaled_i8x64_operand_neon_t;

/** One MX block of 32 E3M2 elements times sixteen, with its F32 scale in every lane. */
typedef struct nk_dot_scaled_i16x32_operand_neon_t {
    int16x8_t values_i16x8[4];
    float32x4_t scales_f32x4;
} nk_dot_scaled_i16x32_operand_neon_t;

/** One MX block of 32 FP8 elements widened to F16, with its F32 scale in every lane. */
typedef struct nk_dot_scaled_f16x32_operand_neon_t {
    float16x8_t values_f16x8[4];
    float32x4_t scales_f32x4;
} nk_dot_scaled_f16x32_operand_neon_t;

/** Doubled E2M1 values of 16 code bytes as signed bytes: the low nibbles, odd elements, in one
 *  register and the high nibbles, even elements, in the other. */
NUMKONG_INLINE void nk_e2m1x32_to_i8x16x2_neon_(uint8x16_t codes_u8x16, int8x16_t *low_i8x16, int8x16_t *high_i8x16) {
    static nk_i8_t const lut_data[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int8x16_t const lut_i8x16 = vld1q_s8(lut_data);
    *low_i8x16 = vqtbl1q_s8(lut_i8x16, vandq_u8(codes_u8x16, vdupq_n_u8(0x0F)));
    *high_i8x16 = vqtbl1q_s8(lut_i8x16, vshrq_n_u8(codes_u8x16, 4));
}

/** E2M3 values times eight as signed bytes, sign included, through one 64-entry table. */
NUMKONG_INLINE int8x16_t nk_e2m3x16_to_i8x16_neon_(uint8x16_t codes_u8x16) {
    static nk_i8_t const lut_data[64] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  //
        16,  18,  20,  22,  24,  26,  28,  30,  32,  36,  40,  44,  48,  52,  56,  60,  //
        0,   -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9,  -10, -11, -12, -13, -14, -15, //
        -16, -18, -20, -22, -24, -26, -28, -30, -32, -36, -40, -44, -48, -52, -56, -60};
    return vqtbl4q_s8(vld1q_s8_x4(lut_data), vandq_u8(codes_u8x16, vdupq_n_u8(0x3F)));
}

/** 32 E2M1 code bytes, 64 elements, as doubled bytes in lane order: word w of every register holds
 *  four elements of 16-element group w, which lies inside one NVFP4 or MXFP4 block. */
NUMKONG_INLINE void nk_e2m1x64_to_i8x64_neon_(void const *codes, nk_b512_vec_t *dst) {
    nk_u8_t const *source = (nk_u8_t const *)codes;
    int8x16_t low_first_i8x16, high_first_i8x16, low_second_i8x16, high_second_i8x16;
    nk_e2m1x32_to_i8x16x2_neon_(vld1q_u8(source), &low_first_i8x16, &high_first_i8x16);
    nk_e2m1x32_to_i8x16x2_neon_(vld1q_u8(source + 16), &low_second_i8x16, &high_second_i8x16);
    uint32x4_t const low_first_u32x4 = vreinterpretq_u32_s8(low_first_i8x16);
    uint32x4_t const low_second_u32x4 = vreinterpretq_u32_s8(low_second_i8x16);
    uint32x4_t const high_first_u32x4 = vreinterpretq_u32_s8(high_first_i8x16);
    uint32x4_t const high_second_u32x4 = vreinterpretq_u32_s8(high_second_i8x16);
    dst->u8x16s[0] = vreinterpretq_u8_u32(vuzp1q_u32(low_first_u32x4, low_second_u32x4));
    dst->u8x16s[1] = vreinterpretq_u8_u32(vuzp2q_u32(low_first_u32x4, low_second_u32x4));
    dst->u8x16s[2] = vreinterpretq_u8_u32(vuzp1q_u32(high_first_u32x4, high_second_u32x4));
    dst->u8x16s[3] = vreinterpretq_u8_u32(vuzp2q_u32(high_first_u32x4, high_second_u32x4));
}

/** The first @p n code bytes of a 64-element E2M1 step, zero-padded. */
NUMKONG_INLINE void nk_partial_e2m1x64_to_i8x64_neon_(void const *codes, nk_b512_vec_t *dst, nk_size_t n) {
    nk_b256_vec_t codes_vec;
    nk_partial_load_b8x32_serial_(codes, &codes_vec, n);
    nk_e2m1x64_to_i8x64_neon_(&codes_vec, dst);
}

/** 64 E2M3 codes as bytes times eight in lane order: lanes 0 and 1 of every register hold the first
 *  MX block, lanes 2 and 3 the second. */
NUMKONG_INLINE void nk_e2m3x64_to_i8x64_neon_(void const *codes, nk_b512_vec_t *dst) {
    nk_u8_t const *source = (nk_u8_t const *)codes;
    uint32x4_t const first_u32x4 = vreinterpretq_u32_s8(nk_e2m3x16_to_i8x16_neon_(vld1q_u8(source)));
    uint32x4_t const second_u32x4 = vreinterpretq_u32_s8(nk_e2m3x16_to_i8x16_neon_(vld1q_u8(source + 16)));
    uint32x4_t const third_u32x4 = vreinterpretq_u32_s8(nk_e2m3x16_to_i8x16_neon_(vld1q_u8(source + 32)));
    uint32x4_t const fourth_u32x4 = vreinterpretq_u32_s8(nk_e2m3x16_to_i8x16_neon_(vld1q_u8(source + 48)));
    dst->u8x16s[0] = vreinterpretq_u8_u32(vuzp1q_u32(first_u32x4, third_u32x4));
    dst->u8x16s[1] = vreinterpretq_u8_u32(vuzp2q_u32(first_u32x4, third_u32x4));
    dst->u8x16s[2] = vreinterpretq_u8_u32(vuzp1q_u32(second_u32x4, fourth_u32x4));
    dst->u8x16s[3] = vreinterpretq_u8_u32(vuzp2q_u32(second_u32x4, fourth_u32x4));
}

/** The first @p n codes of a 64-element E2M3 step, zero-padded. */
NUMKONG_INLINE void nk_partial_e2m3x64_to_i8x64_neon_(void const *codes, nk_b512_vec_t *dst, nk_size_t n) {
    nk_b512_vec_t codes_vec;
    nk_u8_t const *source = (nk_u8_t const *)codes;
    for (nk_size_t chunk = 0; chunk != 4; ++chunk) {
        nk_size_t const offset = chunk * 16;
        if (n >= offset + 16) codes_vec.u8x16s[chunk] = vld1q_u8(source + offset);
        else {
            nk_b128_vec_t tail_vec;
            nk_partial_load_b8x16_serial_(source + offset, &tail_vec, n > offset ? n - offset : 0);
            codes_vec.u8x16s[chunk] = tail_vec.u8x16;
        }
    }
    nk_e2m3x64_to_i8x64_neon_(&codes_vec, dst);
}

/** Stores 64 lane-ordered bytes. A tail step stores all 64 too: packs zero a row's padding first,
 *  and every row spans whole 64-element steps. */
NUMKONG_INLINE void nk_partial_store_i8x64_neon_(nk_b512_vec_t const *src, void *dst, nk_size_t n) {
    nk_unused_(n);
    nk_store_b512_neon_(src, dst);
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

/*  Loaders of one depth step of a pack or panel, @p offset in dims: lifted values and their F32
 *  scales, one per 16 elements for the lane layout and one per block otherwise. */

NUMKONG_INLINE void nk_load_scaled_i8x64_neon_(void const *values, nk_u8_t const *scales, nk_size_t offset,
                                               nk_dot_scaled_i8x64_operand_neon_t *dst) {
    nk_i8_t const *source = (nk_i8_t const *)values + offset;
    dst->values_i8x16[0] = vld1q_s8(source), dst->values_i8x16[1] = vld1q_s8(source + 16);
    dst->values_i8x16[2] = vld1q_s8(source + 32), dst->values_i8x16[3] = vld1q_s8(source + 48);
    dst->scales_f32x4 = vld1q_f32((nk_f32_t const *)scales + offset / 16);
}

NUMKONG_INLINE void nk_load_scaled_i16x32_neon_(void const *values, nk_u8_t const *scales, nk_size_t offset,
                                                nk_dot_scaled_i16x32_operand_neon_t *dst) {
    nk_i16_t const *source = (nk_i16_t const *)values + offset;
    for (nk_size_t i = 0; i != 4; ++i) dst->values_i16x8[i] = vld1q_s16(source + i * 8);
    dst->scales_f32x4 = vld1q_dup_f32((nk_f32_t const *)scales + offset / 32);
}

NUMKONG_INLINE void nk_load_scaled_f16x32_neon_(void const *values, nk_u8_t const *scales, nk_size_t offset,
                                                nk_dot_scaled_f16x32_operand_neon_t *dst) {
    float16_t const *source = (float16_t const *)values + offset;
    for (nk_size_t i = 0; i != 4; ++i) dst->values_f16x8[i] = vld1q_f16(source + i * 8);
    dst->scales_f32x4 = vld1q_dup_f32((nk_f32_t const *)scales + offset / 32);
}

/** Pack-time decoders of 16 codes per call, for the formats that keep one block per depth step. */
NUMKONG_INLINE void nk_load_e3m2x16_to_i16x16_neon_(void const *src, nk_b256_vec_t *dst) {
    uint8x16_t const codes_u8x16 = vld1q_u8((nk_u8_t const *)src);
    dst->i16x8s[0] = nk_e3m2x8_to_i16x8_neon_(vget_low_u8(codes_u8x16));
    dst->i16x8s[1] = nk_e3m2x8_to_i16x8_neon_(vget_high_u8(codes_u8x16));
}

NUMKONG_INLINE void nk_partial_load_e3m2x16_to_i16x16_neon_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t codes_vec;
    nk_partial_load_b8x16_serial_(src, &codes_vec, n);
    dst->i16x8s[0] = nk_e3m2x8_to_i16x8_neon_(vget_low_u8(codes_vec.u8x16));
    dst->i16x8s[1] = nk_e3m2x8_to_i16x8_neon_(vget_high_u8(codes_vec.u8x16));
}

NUMKONG_INLINE void nk_load_e4m3x16_to_f16x16_neon_(void const *src, nk_b256_vec_t *dst) {
    float16x8_t low_f16x8, high_f16x8;
    nk_e4m3x16_to_f16x8x2_neon_(vld1q_u8((nk_u8_t const *)src), &low_f16x8, &high_f16x8);
    dst->u16x8s[0] = vreinterpretq_u16_f16(low_f16x8), dst->u16x8s[1] = vreinterpretq_u16_f16(high_f16x8);
}

NUMKONG_INLINE void nk_partial_load_e4m3x16_to_f16x16_neon_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t codes_vec;
    nk_partial_load_b8x16_serial_(src, &codes_vec, n);
    nk_load_e4m3x16_to_f16x16_neon_(&codes_vec, dst);
}

NUMKONG_INLINE void nk_load_e5m2x16_to_f16x16_neon_(void const *src, nk_b256_vec_t *dst) {
    uint8x16_t const codes_u8x16 = vld1q_u8((nk_u8_t const *)src);
    dst->u16x8s[0] = vreinterpretq_u16_f16(nk_e5m2x8_to_f16x8_neon_(vget_low_u8(codes_u8x16)));
    dst->u16x8s[1] = vreinterpretq_u16_f16(nk_e5m2x8_to_f16x8_neon_(vget_high_u8(codes_u8x16)));
}

NUMKONG_INLINE void nk_partial_load_e5m2x16_to_f16x16_neon_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t codes_vec;
    nk_partial_load_b8x16_serial_(src, &codes_vec, n);
    nk_load_e5m2x16_to_f16x16_neon_(&codes_vec, dst);
}

/** One 64-element step through 8 SMULL and SMLAL pairs into SADALP, the four lane partials SDOT
 *  would give: lifted products reach 3600, so each pair fits a halfword.
 *  One F32 flush per step applies the lane scales. */
NUMKONG_INLINE void nk_dot_scaled_i8x64_update_neon(nk_dot_scaled_f32_state_neon_t *state,
                                                    nk_dot_scaled_i8x64_operand_neon_t a,
                                                    nk_dot_scaled_i8x64_operand_neon_t b) {
    int32x4_t low_i32x4 = vdupq_n_s32(0), high_i32x4 = vdupq_n_s32(0);
    for (nk_size_t i = 0; i != 4; i += 2) {
        int16x8_t low_i16x8 = vmull_s8(vget_low_s8(a.values_i8x16[i]), vget_low_s8(b.values_i8x16[i]));
        low_i16x8 = vmlal_s8(low_i16x8, vget_low_s8(a.values_i8x16[i + 1]), vget_low_s8(b.values_i8x16[i + 1]));
        int16x8_t high_i16x8 = vmull_high_s8(a.values_i8x16[i], b.values_i8x16[i]);
        high_i16x8 = vmlal_high_s8(high_i16x8, a.values_i8x16[i + 1], b.values_i8x16[i + 1]);
        low_i32x4 = vpadalq_s16(low_i32x4, low_i16x8), high_i32x4 = vpadalq_s16(high_i32x4, high_i16x8);
    }
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, vcvtq_f32_s32(vpaddq_s32(low_i32x4, high_i32x4)),
                                 vmulq_f32(a.scales_f32x4, b.scales_f32x4));
}

/** One E3M2 block: lane sums of 8 products under 2^20.6 convert exactly, then one F32 flush. */
NUMKONG_INLINE void nk_dot_scaled_i16x32_update_neon(nk_dot_scaled_f32_state_neon_t *state,
                                                     nk_dot_scaled_i16x32_operand_neon_t a,
                                                     nk_dot_scaled_i16x32_operand_neon_t b) {
    int32x4_t sums_i32x4 = vdupq_n_s32(0);
    for (nk_size_t i = 0; i != 4; ++i) {
        sums_i32x4 = vmlal_s16(sums_i32x4, vget_low_s16(a.values_i16x8[i]), vget_low_s16(b.values_i16x8[i]));
        sums_i32x4 = vmlal_high_s16(sums_i32x4, a.values_i16x8[i], b.values_i16x8[i]);
    }
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, vcvtq_f32_s32(sums_i32x4),
                                 vmulq_f32(a.scales_f32x4, b.scales_f32x4));
}

/** One FP8 block widened to F32: products of E4M3 or E5M2 values are exact, the block sum rounds,
 *  and one F32 flush applies the scales. */
NUMKONG_INLINE void nk_dot_scaled_f16x32_update_neon(nk_dot_scaled_f32_state_neon_t *state,
                                                     nk_dot_scaled_f16x32_operand_neon_t a,
                                                     nk_dot_scaled_f16x32_operand_neon_t b) {
    float32x4_t sums_f32x4 = vdupq_n_f32(0);
    for (nk_size_t i = 0; i != 4; ++i) {
        sums_f32x4 = vfmaq_f32(sums_f32x4, vcvt_f32_f16(vget_low_f16(a.values_f16x8[i])),
                               vcvt_f32_f16(vget_low_f16(b.values_f16x8[i])));
        sums_f32x4 = vfmaq_f32(sums_f32x4, vcvt_high_f32_f16(a.values_f16x8[i]), vcvt_high_f32_f16(b.values_f16x8[i]));
    }
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, sums_f32x4, vmulq_f32(a.scales_f32x4, b.scales_f32x4));
}

#pragma endregion Block Scaled Floats

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
