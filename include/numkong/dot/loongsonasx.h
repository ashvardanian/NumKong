/**
 *  @brief SIMD-accelerated Dot Products for LoongArch LASX (256-bit).
 *  @file include/numkong/dot/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_loongsonasx_instructions Key LASX Dot Product Instructions
 *
 *  LASX provides 256-bit SIMD operations using __m256i as the universal vector type.
 *  All intrinsics are prefixed with __lasx_. Float operations reinterpret __m256i as
 *  f32x8 or f64x4. Integer widening multiply-accumulate chains handle i8/u8 dot products.
 *
 *  For F32 dot products, upcasting to F64 and downcasting back is faster than stable
 *  summation algorithms. For F64 we use the Dot2 algorithm (Ogita-Rump-Oishi, 2005)
 *  for compensated accumulation via TwoSum/TwoProd.
 *
 *  @section dot_loongsonasx_stateful Stateful Streaming Logic
 *
 *  - nk_dot_f64x4 state with Dot2 stable dot-products,
 *  - nk_dot_f32x8 state with double-precision numerics,
 *  - nk_dot_through_i32 state for 8-bit signed and unsigned integer inputs.
 */
#ifndef NK_DOT_LOONGSONASX_H
#define NK_DOT_LOONGSONASX_H

#if NK_TARGET_LOONGARCH64_
#if NK_TARGET_LOONGSONASX

#include "numkong/types.h"
#include "numkong/dot/serial.h"
#include "numkong/cast/loongsonasx.h" // `nk_bf16x8_to_f32x8_loongsonasx_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Horizontal Reduction Helpers

/** @brief Horizontal sum of 4 f64 lanes in a 256-bit LASX register. */
NK_INTERNAL nk_f64_t nk_reduce_add_f64x4_loongsonasx_(__m256d sum_f64x4) {
    // Add high 128-bit lane to low 128-bit lane
    __m256d high_f64x4 = (__m256d)__lasx_xvpermi_q((__m256i)sum_f64x4, (__m256i)sum_f64x4, 0x11);
    __m256d sum_f64x2 = __lasx_xvfadd_d(sum_f64x4, high_f64x4);
    // Swap lanes 0↔1, add to reduce to 1 value, then extract
    __m256d swapped_f64x2 = (__m256d)__lasx_xvshuf4i_d((__m256i)sum_f64x2, (__m256i)sum_f64x2, 0b0001);
    __m256d reduced_f64x2 = __lasx_xvfadd_d(sum_f64x2, swapped_f64x2);
    nk_fui64_t c;
    c.u = (nk_u64_t)__lasx_xvpickve2gr_du((__m256i)reduced_f64x2, 0);
    return c.f;
}

/** @brief Horizontal sum of 8 i32 lanes in a 256-bit LASX register. */
NK_INTERNAL nk_i32_t nk_reduce_add_i32x8_loongsonasx_(__m256i sum_i32x8) {
    __m256i high_i32x8 = __lasx_xvpermi_q(sum_i32x8, sum_i32x8, 0x11);
    __m256i sum_i32x4 = __lasx_xvadd_w(sum_i32x8, high_i32x8);
    // Pairwise widen i32 → i64, then extract and add
    __m256i sum_i64x2 = __lasx_xvhaddw_d_w(sum_i32x4, sum_i32x4);
    return (nk_i32_t)(__lasx_xvpickve2gr_d(sum_i64x2, 0) + __lasx_xvpickve2gr_d(sum_i64x2, 1));
}

/** @brief Compensated horizontal sum of 4 f64 lanes via TwoSum tree reduction.
 *  @sa nk_reduce_sum_f64_serial_ for the serial equivalent
 */
NK_INTERNAL nk_f64_t nk_dot_stable_sum_f64x4_loongsonasx_(__m256d sum_f64x4, __m256d compensation_f64x4) {
    // Stage 0: TwoSum merge of sum + compensation (4-wide, parallel)
    __m256d tentative_sum_f64x4 = __lasx_xvfadd_d(sum_f64x4, compensation_f64x4);
    __m256d virtual_addend_f64x4 = __lasx_xvfsub_d(tentative_sum_f64x4, sum_f64x4);
    __m256d rounding_error_f64x4 = __lasx_xvfadd_d(
        __lasx_xvfsub_d(sum_f64x4, __lasx_xvfsub_d(tentative_sum_f64x4, virtual_addend_f64x4)),
        __lasx_xvfsub_d(compensation_f64x4, virtual_addend_f64x4));

    // Stage 1: TwoSum halving 4 → 2 by adding high 128-bit lane to low 128-bit lane
    __m256d upper_sum_f64x4 = (__m256d)__lasx_xvpermi_q((__m256i)tentative_sum_f64x4, (__m256i)tentative_sum_f64x4,
                                                        0x11);
    __m256d lower_sum_f64x4 = tentative_sum_f64x4; // low 128 bits are already there
    __m256d tentative_sum_f64x2 = __lasx_xvfadd_d(lower_sum_f64x4, upper_sum_f64x4);
    __m256d virtual_addend_f64x2 = __lasx_xvfsub_d(tentative_sum_f64x2, lower_sum_f64x4);
    __m256d rounding_error_f64x2 = __lasx_xvfadd_d(
        __lasx_xvfsub_d(lower_sum_f64x4, __lasx_xvfsub_d(tentative_sum_f64x2, virtual_addend_f64x2)),
        __lasx_xvfsub_d(upper_sum_f64x4, virtual_addend_f64x2));
    // Accumulate errors: stage 0 errors (halved) + stage 1 rounding error
    __m256d upper_error_f64x4 = (__m256d)__lasx_xvpermi_q((__m256i)rounding_error_f64x4, (__m256i)rounding_error_f64x4,
                                                          0x11);
    __m256d lower_error_f64x4 = rounding_error_f64x4; // low 128 bits are already there
    __m256d accumulated_error_f64x2 = __lasx_xvfadd_d(__lasx_xvfadd_d(lower_error_f64x4, upper_error_f64x4),
                                                      rounding_error_f64x2);

    // Stage 2: Scalar TwoSum 2 → 1
    nk_fui64_t c;
    c.u = (nk_u64_t)__lasx_xvpickve2gr_du((__m256i)tentative_sum_f64x2, 0);
    nk_f64_t sum_low = c.f;
    c.u = (nk_u64_t)__lasx_xvpickve2gr_du((__m256i)tentative_sum_f64x2, 1);
    nk_f64_t sum_high = c.f;
    c.u = (nk_u64_t)__lasx_xvpickve2gr_du((__m256i)accumulated_error_f64x2, 0);
    nk_f64_t error_low = c.f;
    c.u = (nk_u64_t)__lasx_xvpickve2gr_du((__m256i)accumulated_error_f64x2, 1);
    nk_f64_t error_high = c.f;
    nk_f64_t tentative_sum = sum_low + sum_high;
    nk_f64_t virtual_addend = tentative_sum - sum_low;
    nk_f64_t rounding_error = (sum_low - (tentative_sum - virtual_addend)) + (sum_high - virtual_addend);
    return tentative_sum + (error_low + error_high + rounding_error);
}

#pragma endregion Horizontal Reduction Helpers

#pragma region F32 and F64 Floats

NK_PUBLIC void nk_dot_f32_loongsonasx(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f64_t *result) {
    // LASX is 256-bit = 8 × f32. Load 8 f32, split into low/high 4, widen each to f64, FMA in f64.
    __m256d sum_low_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);  // 4 f64 accumulators (from low 4 f32)
    __m256d sum_high_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0); // 4 f64 accumulators (from high 4 f32)
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count_scalars; idx_scalars += 8) {
        __m256i a_f32x8 = __lasx_xvld(a_scalars + idx_scalars, 0);
        __m256i b_f32x8 = __lasx_xvld(b_scalars + idx_scalars, 0);
        // Widen low 4 f32 → f64
        __m256d a_low_f64x4 = __lasx_xvfcvtl_d_s((__m256)a_f32x8);
        __m256d b_low_f64x4 = __lasx_xvfcvtl_d_s((__m256)b_f32x8);
        // Widen high 4 f32 → f64
        __m256d a_high_f64x4 = __lasx_xvfcvth_d_s((__m256)a_f32x8);
        __m256d b_high_f64x4 = __lasx_xvfcvth_d_s((__m256)b_f32x8);
        // FMA in f64
        sum_low_f64x4 = __lasx_xvfmadd_d(a_low_f64x4, b_low_f64x4, sum_low_f64x4);
        sum_high_f64x4 = __lasx_xvfmadd_d(a_high_f64x4, b_high_f64x4, sum_high_f64x4);
    }
    __m256d combined_f64x4 = __lasx_xvfadd_d(sum_low_f64x4, sum_high_f64x4);
    nk_f64_t sum = nk_reduce_add_f64x4_loongsonasx_(combined_f64x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_f64_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_f64_loongsonasx(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f64_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product
    __m256d sum_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d compensation_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count_scalars; idx_scalars += 4) {
        __m256d a_f64x4 = (__m256d)__lasx_xvld(a_scalars + idx_scalars, 0);
        __m256d b_f64x4 = (__m256d)__lasx_xvld(b_scalars + idx_scalars, 0);

        // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
        __m256d product_f64x4 = __lasx_xvfmul_d(a_f64x4, b_f64x4);
        __m256d product_error_f64x4 = __lasx_xvfmsub_d(a_f64x4, b_f64x4, product_f64x4);

        // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
        __m256d tentative_sum_f64x4 = __lasx_xvfadd_d(sum_f64x4, product_f64x4);
        __m256d virtual_addend_f64x4 = __lasx_xvfsub_d(tentative_sum_f64x4, sum_f64x4);
        __m256d sum_error_f64x4 = __lasx_xvfadd_d(
            __lasx_xvfsub_d(sum_f64x4, __lasx_xvfsub_d(tentative_sum_f64x4, virtual_addend_f64x4)),
            __lasx_xvfsub_d(product_f64x4, virtual_addend_f64x4));

        // Update: sum = t, compensation += q + r
        sum_f64x4 = tentative_sum_f64x4;
        compensation_f64x4 = __lasx_xvfadd_d(compensation_f64x4, __lasx_xvfadd_d(sum_error_f64x4, product_error_f64x4));
    }
    // Scalar tail
    nk_f64_t sum = nk_dot_stable_sum_f64x4_loongsonasx_(sum_f64x4, compensation_f64x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_f64_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_i8_loongsonasx(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                     nk_i32_t *result) {
    __m256i sum_i32x8 = __lasx_xvreplgr2vr_w(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_i8x32 = __lasx_xvld(a_scalars + idx_scalars, 0);
        __m256i b_i8x32 = __lasx_xvld(b_scalars + idx_scalars, 0);
        // Widening multiply i8 × i8 → i16 (even and odd elements separately)
        __m256i acc_i16x16 = __lasx_xvreplgr2vr_h(0);
        acc_i16x16 = __lasx_xvmaddwev_h_b(acc_i16x16, a_i8x32, b_i8x32);
        acc_i16x16 = __lasx_xvmaddwod_h_b(acc_i16x16, a_i8x32, b_i8x32);
        // Horizontal pairwise i16 → i32, then accumulate
        __m256i widened_i32x8 = __lasx_xvhaddw_w_h(acc_i16x16, acc_i16x16);
        sum_i32x8 = __lasx_xvadd_w(sum_i32x8, widened_i32x8);
    }
    nk_i32_t sum = nk_reduce_add_i32x8_loongsonasx_(sum_i32x8);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_u8_loongsonasx(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                     nk_u32_t *result) {
    __m256i sum_i32x8 = __lasx_xvreplgr2vr_w(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_u8x32 = __lasx_xvld(a_scalars + idx_scalars, 0);
        __m256i b_u8x32 = __lasx_xvld(b_scalars + idx_scalars, 0);
        // Unsigned widening multiply u8 × u8 → u16 (even and odd elements separately)
        __m256i acc_u16x16 = __lasx_xvreplgr2vr_h(0);
        acc_u16x16 = __lasx_xvmaddwev_h_bu(acc_u16x16, a_u8x32, b_u8x32);
        acc_u16x16 = __lasx_xvmaddwod_h_bu(acc_u16x16, a_u8x32, b_u8x32);
        // Unsigned horizontal pairwise u16 → u32, then accumulate
        __m256i widened_u32x8 = __lasx_xvhaddw_wu_hu(acc_u16x16, acc_u16x16);
        sum_i32x8 = __lasx_xvadd_w(sum_i32x8, widened_u32x8);
    }
    nk_u32_t sum = (nk_u32_t)nk_reduce_add_i32x8_loongsonasx_(sum_i32x8);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_bf16_loongsonasx(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    __m256 sum_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256i mask_high_u32x8 = __lasx_xvreplgr2vr_w((int)0xFFFF0000);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count_scalars; idx_scalars += 16) {
        __m256i a_bf16x16 = __lasx_xvld(a_scalars + idx_scalars, 0);
        __m256i b_bf16x16 = __lasx_xvld(b_scalars + idx_scalars, 0);
        __m256 a_even_f32x8 = (__m256)__lasx_xvslli_w(a_bf16x16, 16);
        __m256 b_even_f32x8 = (__m256)__lasx_xvslli_w(b_bf16x16, 16);
        sum_f32x8 = __lasx_xvfmadd_s(a_even_f32x8, b_even_f32x8, sum_f32x8);
        __m256 a_odd_f32x8 = (__m256)__lasx_xvand_v(a_bf16x16, mask_high_u32x8);
        __m256 b_odd_f32x8 = (__m256)__lasx_xvand_v(b_bf16x16, mask_high_u32x8);
        sum_f32x8 = __lasx_xvfmadd_s(a_odd_f32x8, b_odd_f32x8, sum_f32x8);
    }
    // Horizontal reduce 8 × f32 → 1 × f32
    __m256 high_f32x4 = (__m256)__lasx_xvpermi_q((__m256i)sum_f32x8, (__m256i)sum_f32x8, 0x11);
    __m256 sum_f32x4 = __lasx_xvfadd_s(sum_f32x8, high_f32x4);
    __m256 swapped_f32x4 = (__m256)__lasx_xvshuf4i_w((__m256i)sum_f32x4, 0b01001110);
    __m256 reduced_f32x4 = __lasx_xvfadd_s(sum_f32x4, swapped_f32x4);
    __m256 swapped_f32x2 = (__m256)__lasx_xvshuf4i_w((__m256i)reduced_f32x4, 0b10110001);
    __m256 reduced_f32x2 = __lasx_xvfadd_s(reduced_f32x4, swapped_f32x2);
    nk_fui32_t c;
    c.u = (nk_u32_t)__lasx_xvpickve2gr_w((__m256i)reduced_f32x2, 0);
    nk_f32_t sum = c.f;
    for (; idx_scalars < count_scalars; ++idx_scalars) {
        nk_f32_t a_val, b_val;
        nk_bf16_to_f32_serial(&a_scalars[idx_scalars], &a_val);
        nk_bf16_to_f32_serial(&b_scalars[idx_scalars], &b_val);
        sum += a_val * b_val;
    }
    *result = sum;
}

typedef struct nk_dot_f64x4_state_loongsonasx_t {
    __m256i sum_f64x4;
    __m256i compensation_f64x4; // Error accumulator for Dot2
} nk_dot_f64x4_state_loongsonasx_t;

NK_INTERNAL void nk_dot_f64x4_init_loongsonasx(nk_dot_f64x4_state_loongsonasx_t *state) {
    state->sum_f64x4 = __lasx_xvreplgr2vr_d(0);
    state->compensation_f64x4 = __lasx_xvreplgr2vr_d(0);
}

NK_INTERNAL void nk_dot_f64x4_update_loongsonasx(nk_dot_f64x4_state_loongsonasx_t *state, nk_b256_vec_t a,
                                                 nk_b256_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256d sum_f64x4 = (__m256d)state->sum_f64x4;
    __m256d compensation_f64x4 = (__m256d)state->compensation_f64x4;
    __m256d a_f64x4 = a.ymm_pd;
    __m256d b_f64x4 = b.ymm_pd;

    // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
    __m256d product_f64x4 = __lasx_xvfmul_d(a_f64x4, b_f64x4);
    __m256d product_error_f64x4 = __lasx_xvfmsub_d(a_f64x4, b_f64x4, product_f64x4);

    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    __m256d tentative_sum_f64x4 = __lasx_xvfadd_d(sum_f64x4, product_f64x4);
    __m256d virtual_addend_f64x4 = __lasx_xvfsub_d(tentative_sum_f64x4, sum_f64x4);
    __m256d sum_error_f64x4 = __lasx_xvfadd_d(
        __lasx_xvfsub_d(sum_f64x4, __lasx_xvfsub_d(tentative_sum_f64x4, virtual_addend_f64x4)),
        __lasx_xvfsub_d(product_f64x4, virtual_addend_f64x4));

    // Update: sum = t, compensation += q + r
    state->sum_f64x4 = (__m256i)tentative_sum_f64x4;
    state->compensation_f64x4 = (__m256i)__lasx_xvfadd_d(compensation_f64x4,
                                                         __lasx_xvfadd_d(sum_error_f64x4, product_error_f64x4));
}

NK_INTERNAL void nk_dot_f64x4_finalize_loongsonasx(                                                   //
    nk_dot_f64x4_state_loongsonasx_t const *state_a, nk_dot_f64x4_state_loongsonasx_t const *state_b, //
    nk_dot_f64x4_state_loongsonasx_t const *state_c, nk_dot_f64x4_state_loongsonasx_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    // Compensated horizontal reduction preserving Dot2 error tracking per state
    result->f64s[0] = nk_dot_stable_sum_f64x4_loongsonasx_((__m256d)state_a->sum_f64x4,
                                                           (__m256d)state_a->compensation_f64x4);
    result->f64s[1] = nk_dot_stable_sum_f64x4_loongsonasx_((__m256d)state_b->sum_f64x4,
                                                           (__m256d)state_b->compensation_f64x4);
    result->f64s[2] = nk_dot_stable_sum_f64x4_loongsonasx_((__m256d)state_c->sum_f64x4,
                                                           (__m256d)state_c->compensation_f64x4);
    result->f64s[3] = nk_dot_stable_sum_f64x4_loongsonasx_((__m256d)state_d->sum_f64x4,
                                                           (__m256d)state_d->compensation_f64x4);
}

typedef struct nk_dot_f32x8_state_loongsonasx_t {
    __m256i sum_f64x4;
} nk_dot_f32x8_state_loongsonasx_t;

NK_INTERNAL void nk_dot_f32x8_init_loongsonasx(nk_dot_f32x8_state_loongsonasx_t *state) {
    state->sum_f64x4 = __lasx_xvreplgr2vr_d(0);
}

NK_INTERNAL void nk_dot_f32x8_update_loongsonasx(nk_dot_f32x8_state_loongsonasx_t *state, nk_b256_vec_t a,
                                                 nk_b256_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256d a_low_f64x4 = __lasx_xvfcvtl_d_s(a.ymm_ps);
    __m256d b_low_f64x4 = __lasx_xvfcvtl_d_s(b.ymm_ps);
    state->sum_f64x4 = (__m256i)__lasx_xvfmadd_d(a_low_f64x4, b_low_f64x4, (__m256d)state->sum_f64x4);
    __m256d a_high_f64x4 = __lasx_xvfcvth_d_s(a.ymm_ps);
    __m256d b_high_f64x4 = __lasx_xvfcvth_d_s(b.ymm_ps);
    state->sum_f64x4 = (__m256i)__lasx_xvfmadd_d(a_high_f64x4, b_high_f64x4, (__m256d)state->sum_f64x4);
}

NK_INTERNAL void nk_dot_f32x8_finalize_loongsonasx(                                                   //
    nk_dot_f32x8_state_loongsonasx_t const *state_a, nk_dot_f32x8_state_loongsonasx_t const *state_b, //
    nk_dot_f32x8_state_loongsonasx_t const *state_c, nk_dot_f32x8_state_loongsonasx_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    // Horizontal reduction: 4 f64s → 1 f64 for each state, packed into result via SIMD
    __m256d sum_a_f64x4 = (__m256d)state_a->sum_f64x4;
    __m256d sum_b_f64x4 = (__m256d)state_b->sum_f64x4;
    __m256d sum_c_f64x4 = (__m256d)state_c->sum_f64x4;
    __m256d sum_d_f64x4 = (__m256d)state_d->sum_f64x4;

    // 4 → 2: add high 128-bit lane to low lane
    __m256d sum_a_f64x2 = __lasx_xvfadd_d(sum_a_f64x4,
                                          (__m256d)__lasx_xvpermi_q((__m256i)sum_a_f64x4, (__m256i)sum_a_f64x4, 0x11));
    __m256d sum_b_f64x2 = __lasx_xvfadd_d(sum_b_f64x4,
                                          (__m256d)__lasx_xvpermi_q((__m256i)sum_b_f64x4, (__m256i)sum_b_f64x4, 0x11));
    __m256d sum_c_f64x2 = __lasx_xvfadd_d(sum_c_f64x4,
                                          (__m256d)__lasx_xvpermi_q((__m256i)sum_c_f64x4, (__m256i)sum_c_f64x4, 0x11));
    __m256d sum_d_f64x2 = __lasx_xvfadd_d(sum_d_f64x4,
                                          (__m256d)__lasx_xvpermi_q((__m256i)sum_d_f64x4, (__m256i)sum_d_f64x4, 0x11));

    // 2 → 1: interleave then horizontal add (xvilvl_d/xvilvh_d are integer intrinsics)
    __m256d ab_low_f64x2 = (__m256d)__lasx_xvilvl_d((__m256i)sum_b_f64x2, (__m256i)sum_a_f64x2);
    __m256d ab_high_f64x2 = (__m256d)__lasx_xvilvh_d((__m256i)sum_b_f64x2, (__m256i)sum_a_f64x2);
    __m256d cd_low_f64x2 = (__m256d)__lasx_xvilvl_d((__m256i)sum_d_f64x2, (__m256i)sum_c_f64x2);
    __m256d cd_high_f64x2 = (__m256d)__lasx_xvilvh_d((__m256i)sum_d_f64x2, (__m256i)sum_c_f64x2);
    __m256d sum_ab_f64x2 = __lasx_xvfadd_d(ab_low_f64x2, ab_high_f64x2);
    __m256d sum_cd_f64x2 = __lasx_xvfadd_d(cd_low_f64x2, cd_high_f64x2);

    // Pack [sum_a, sum_b, sum_c, sum_d] into one 256-bit result
    result->ymm = __lasx_xvpermi_q((__m256i)sum_cd_f64x2, (__m256i)sum_ab_f64x2, 0x20);
}

#pragma endregion F32 and F64 Floats

#pragma region I8 and U8 Integers

/**
 *  @brief Internal helper state for dot-products of integer types, where 32-bit accumulation is enough.
 *  @sa nk_dot_i8x32_state_loongsonasx_t, nk_dot_u8x32_state_loongsonasx_t
 */
typedef struct nk_dot_through_i32_state_loongsonasx_t_ {
    __m256i sum_i32x8;
} nk_dot_through_i32_state_loongsonasx_t_;

/**
 *  @brief Initializes 32-bit accumulators for integer dot-products.
 *  @sa nk_dot_i8x32_update_loongsonasx, nk_dot_u8x32_update_loongsonasx
 */
NK_INTERNAL void nk_dot_through_i32_init_loongsonasx_(nk_dot_through_i32_state_loongsonasx_t_ *state) {
    state->sum_i32x8 = __lasx_xvreplgr2vr_w(0);
}

/**
 *  @brief Finalizes 4x integer dot-products placing them into 4x consecutive 32-bit slots.
 *  @sa nk_dot_i8x32_update_loongsonasx, nk_dot_u8x32_update_loongsonasx
 */
NK_INTERNAL void nk_dot_through_i32_finalize_loongsonasx_(                                                          //
    nk_dot_through_i32_state_loongsonasx_t_ const *state_a, nk_dot_through_i32_state_loongsonasx_t_ const *state_b, //
    nk_dot_through_i32_state_loongsonasx_t_ const *state_c, nk_dot_through_i32_state_loongsonasx_t_ const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // ILP-optimized 4-way horizontal reduction for i32 using LASX interleave
    // Step 1: 8 → 4 for all 4 states (add high 128-bit lane to low lane)
    __m256i sum_a_i32x4 = __lasx_xvadd_w(state_a->sum_i32x8,
                                         __lasx_xvpermi_q(state_a->sum_i32x8, state_a->sum_i32x8, 0x11));
    __m256i sum_b_i32x4 = __lasx_xvadd_w(state_b->sum_i32x8,
                                         __lasx_xvpermi_q(state_b->sum_i32x8, state_b->sum_i32x8, 0x11));
    __m256i sum_c_i32x4 = __lasx_xvadd_w(state_c->sum_i32x8,
                                         __lasx_xvpermi_q(state_c->sum_i32x8, state_c->sum_i32x8, 0x11));
    __m256i sum_d_i32x4 = __lasx_xvadd_w(state_d->sum_i32x8,
                                         __lasx_xvpermi_q(state_d->sum_i32x8, state_d->sum_i32x8, 0x11));
    // Step 2: Transpose 4×4 matrix via interleave
    __m256i transpose_ab_low_i32x4 = __lasx_xvilvl_w(sum_b_i32x4, sum_a_i32x4);
    __m256i transpose_cd_low_i32x4 = __lasx_xvilvl_w(sum_d_i32x4, sum_c_i32x4);
    __m256i transpose_ab_high_i32x4 = __lasx_xvilvh_w(sum_b_i32x4, sum_a_i32x4);
    __m256i transpose_cd_high_i32x4 = __lasx_xvilvh_w(sum_d_i32x4, sum_c_i32x4);
    __m256i sum_lane0_i32x4 = __lasx_xvilvl_d(transpose_cd_low_i32x4, transpose_ab_low_i32x4);
    __m256i sum_lane1_i32x4 = __lasx_xvilvh_d(transpose_cd_low_i32x4, transpose_ab_low_i32x4);
    __m256i sum_lane2_i32x4 = __lasx_xvilvl_d(transpose_cd_high_i32x4, transpose_ab_high_i32x4);
    __m256i sum_lane3_i32x4 = __lasx_xvilvh_d(transpose_cd_high_i32x4, transpose_ab_high_i32x4);
    // Step 3: Vertical sum
    __m256i sum_i32x4 = __lasx_xvadd_w(__lasx_xvadd_w(sum_lane0_i32x4, sum_lane1_i32x4),
                                       __lasx_xvadd_w(sum_lane2_i32x4, sum_lane3_i32x4));
    result->xmm = nk_lasx_castsi256_si128_(sum_i32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over i8 scalars on LASX.
 *  @note Alias of nk_dot_through_i32_state_loongsonasx_t_
 */
typedef struct nk_dot_through_i32_state_loongsonasx_t_ nk_dot_i8x32_state_loongsonasx_t;

NK_INTERNAL void nk_dot_i8x32_init_loongsonasx(nk_dot_i8x32_state_loongsonasx_t *state) {
    nk_dot_through_i32_init_loongsonasx_(state);
}

NK_INTERNAL void nk_dot_i8x32_update_loongsonasx(nk_dot_i8x32_state_loongsonasx_t *state, nk_b256_vec_t a,
                                                 nk_b256_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256i acc_i16x16 = __lasx_xvreplgr2vr_h(0);
    acc_i16x16 = __lasx_xvmaddwev_h_b(acc_i16x16, a.ymm, b.ymm);
    acc_i16x16 = __lasx_xvmaddwod_h_b(acc_i16x16, a.ymm, b.ymm);
    __m256i widened_i32x8 = __lasx_xvhaddw_w_h(acc_i16x16, acc_i16x16);
    state->sum_i32x8 = __lasx_xvadd_w(state->sum_i32x8, widened_i32x8);
}

NK_INTERNAL void nk_dot_i8x32_finalize_loongsonasx(                                                   //
    nk_dot_i8x32_state_loongsonasx_t const *state_a, nk_dot_i8x32_state_loongsonasx_t const *state_b, //
    nk_dot_i8x32_state_loongsonasx_t const *state_c, nk_dot_i8x32_state_loongsonasx_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_dot_through_i32_finalize_loongsonasx_(state_a, state_b, state_c, state_d, total_dimensions, result);
}

/**
 *  @brief Running state for 256-bit dot accumulation over u8 scalars on LASX.
 *  @note Alias of nk_dot_through_i32_state_loongsonasx_t_
 */
typedef struct nk_dot_through_i32_state_loongsonasx_t_ nk_dot_u8x32_state_loongsonasx_t;

NK_INTERNAL void nk_dot_u8x32_init_loongsonasx(nk_dot_u8x32_state_loongsonasx_t *state) {
    nk_dot_through_i32_init_loongsonasx_(state);
}

NK_INTERNAL void nk_dot_u8x32_update_loongsonasx(nk_dot_u8x32_state_loongsonasx_t *state, nk_b256_vec_t a,
                                                 nk_b256_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256i acc_u16x16 = __lasx_xvreplgr2vr_h(0);
    acc_u16x16 = __lasx_xvmaddwev_h_bu(acc_u16x16, a.ymm, b.ymm);
    acc_u16x16 = __lasx_xvmaddwod_h_bu(acc_u16x16, a.ymm, b.ymm);
    __m256i widened_u32x8 = __lasx_xvhaddw_wu_hu(acc_u16x16, acc_u16x16);
    state->sum_i32x8 = __lasx_xvadd_w(state->sum_i32x8, widened_u32x8);
}

NK_INTERNAL void nk_dot_u8x32_finalize_loongsonasx(                                                   //
    nk_dot_u8x32_state_loongsonasx_t const *state_a, nk_dot_u8x32_state_loongsonasx_t const *state_b, //
    nk_dot_u8x32_state_loongsonasx_t const *state_c, nk_dot_u8x32_state_loongsonasx_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_dot_through_i32_finalize_loongsonasx_(state_a, state_b, state_c, state_d, total_dimensions, result);
}

#pragma endregion I8 and U8 Integers

#pragma region F16 and BF16 Floats

/**
 *  @brief Internal helper state for dot-products of low-precision types, where 32-bit accumulation is enough.
 *  @sa nk_dot_bf16x16_state_loongsonasx_t
 */
typedef struct nk_dot_through_f32_state_loongsonasx_t_ {
    __m256i sum_f32x8;
} nk_dot_through_f32_state_loongsonasx_t_;

/**
 *  @brief Initializes 32-bit accumulators for low-precision dot-products.
 *  @sa nk_dot_bf16x16_init_loongsonasx
 */
NK_INTERNAL void nk_dot_through_f32_init_loongsonasx_(nk_dot_through_f32_state_loongsonasx_t_ *state) {
    state->sum_f32x8 = __lasx_xvreplgr2vr_w(0);
}

/**
 *  @brief Fuses 32-bit multiplication and accumulation for pre-converted f32 vectors.
 *  @sa nk_dot_bf16x16_update_loongsonasx
 */
NK_INTERNAL void nk_dot_through_f32_update_loongsonasx_(nk_dot_through_f32_state_loongsonasx_t_ *state, nk_b256_vec_t a,
                                                        nk_b256_vec_t b, nk_size_t depth_offset,
                                                        nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->sum_f32x8 = (__m256i)__lasx_xvfmadd_s(a.ymm_ps, b.ymm_ps, (__m256)state->sum_f32x8);
}

/**
 *  @brief Finalizes 4x low-precision dot-products placing them into 4x consecutive 32-bit slots.
 *  @sa nk_dot_bf16x16_finalize_loongsonasx
 *
 *  Computes 4x horizontal reductions, each involving 8x floats, using LASX interleave instructions.
 */
NK_INTERNAL void nk_dot_through_f32_finalize_loongsonasx_(                                                          //
    nk_dot_through_f32_state_loongsonasx_t_ const *state_a, nk_dot_through_f32_state_loongsonasx_t_ const *state_b, //
    nk_dot_through_f32_state_loongsonasx_t_ const *state_c, nk_dot_through_f32_state_loongsonasx_t_ const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // Step 1: 8 → 4 for all 4 states (add high 128-bit lane to low lane)
    __m256 sum_a_f32x4 = __lasx_xvfadd_s((__m256)state_a->sum_f32x8,
                                         (__m256)__lasx_xvpermi_q(state_a->sum_f32x8, state_a->sum_f32x8, 0x11));
    __m256 sum_b_f32x4 = __lasx_xvfadd_s((__m256)state_b->sum_f32x8,
                                         (__m256)__lasx_xvpermi_q(state_b->sum_f32x8, state_b->sum_f32x8, 0x11));
    __m256 sum_c_f32x4 = __lasx_xvfadd_s((__m256)state_c->sum_f32x8,
                                         (__m256)__lasx_xvpermi_q(state_c->sum_f32x8, state_c->sum_f32x8, 0x11));
    __m256 sum_d_f32x4 = __lasx_xvfadd_s((__m256)state_d->sum_f32x8,
                                         (__m256)__lasx_xvpermi_q(state_d->sum_f32x8, state_d->sum_f32x8, 0x11));
    // Step 2: Transpose 4×4 matrix via interleave (integer intrinsics, cast at boundaries)
    __m256i transpose_ab_low_f32x4 = __lasx_xvilvl_w((__m256i)sum_b_f32x4, (__m256i)sum_a_f32x4);
    __m256i transpose_cd_low_f32x4 = __lasx_xvilvl_w((__m256i)sum_d_f32x4, (__m256i)sum_c_f32x4);
    __m256i transpose_ab_high_f32x4 = __lasx_xvilvh_w((__m256i)sum_b_f32x4, (__m256i)sum_a_f32x4);
    __m256i transpose_cd_high_f32x4 = __lasx_xvilvh_w((__m256i)sum_d_f32x4, (__m256i)sum_c_f32x4);
    __m256i sum_lane0_f32x4 = __lasx_xvilvl_d(transpose_cd_low_f32x4, transpose_ab_low_f32x4);
    __m256i sum_lane1_f32x4 = __lasx_xvilvh_d(transpose_cd_low_f32x4, transpose_ab_low_f32x4);
    __m256i sum_lane2_f32x4 = __lasx_xvilvl_d(transpose_cd_high_f32x4, transpose_ab_high_f32x4);
    __m256i sum_lane3_f32x4 = __lasx_xvilvh_d(transpose_cd_high_f32x4, transpose_ab_high_f32x4);
    // Step 3: Vertical sum
    __m256 sum_f32x4 = __lasx_xvfadd_s(__lasx_xvfadd_s((__m256)sum_lane0_f32x4, (__m256)sum_lane1_f32x4),
                                       __lasx_xvfadd_s((__m256)sum_lane2_f32x4, (__m256)sum_lane3_f32x4));
    result->xmm_ps = nk_lasx_castps256_ps128_(sum_f32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars on LASX.
 *  @note Alias of nk_dot_through_f32_state_loongsonasx_t_
 */
typedef struct nk_dot_through_f32_state_loongsonasx_t_ nk_dot_bf16x16_state_loongsonasx_t;

NK_INTERNAL void nk_dot_bf16x16_init_loongsonasx(nk_dot_bf16x16_state_loongsonasx_t *state) {
    nk_dot_through_f32_init_loongsonasx_(state);
}

NK_INTERNAL void nk_dot_bf16x16_update_loongsonasx(nk_dot_bf16x16_state_loongsonasx_t *state, nk_b256_vec_t a,
                                                   nk_b256_vec_t b, nk_size_t depth_offset,
                                                   nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Even bf16 elements → slli_epi32 by 16 places them in f32 upper bits.
    // Odd bf16 elements → AND with 0xFFFF0000 keeps them in f32 upper bits.
    __m256i mask_high_u32x8 = __lasx_xvreplgr2vr_w((int)0xFFFF0000);
    __m256 a_even_f32x8 = (__m256)__lasx_xvslli_w(a.ymm, 16);
    __m256 b_even_f32x8 = (__m256)__lasx_xvslli_w(b.ymm, 16);
    state->sum_f32x8 = (__m256i)__lasx_xvfmadd_s(a_even_f32x8, b_even_f32x8, (__m256)state->sum_f32x8);
    __m256 a_odd_f32x8 = (__m256)__lasx_xvand_v(a.ymm, mask_high_u32x8);
    __m256 b_odd_f32x8 = (__m256)__lasx_xvand_v(b.ymm, mask_high_u32x8);
    state->sum_f32x8 = (__m256i)__lasx_xvfmadd_s(a_odd_f32x8, b_odd_f32x8, (__m256)state->sum_f32x8);
}

NK_INTERNAL void nk_dot_bf16x16_finalize_loongsonasx(                                                     //
    nk_dot_bf16x16_state_loongsonasx_t const *state_a, nk_dot_bf16x16_state_loongsonasx_t const *state_b, //
    nk_dot_bf16x16_state_loongsonasx_t const *state_c, nk_dot_bf16x16_state_loongsonasx_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_dot_through_f32_finalize_loongsonasx_(state_a, state_b, state_c, state_d, total_dimensions, result);
}

NK_PUBLIC void nk_dot_f16_loongsonasx(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f32_t *result) {
    __m256 sum_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count_scalars; idx_scalars += 8) {
        __m128i a_f16x8 = __lsx_vld(a_scalars + idx_scalars, 0);
        __m128i b_f16x8 = __lsx_vld(b_scalars + idx_scalars, 0);
        __m256 a_f32x8 = (__m256)nk_f16x8_to_f32x8_loongsonasx_(a_f16x8);
        __m256 b_f32x8 = (__m256)nk_f16x8_to_f32x8_loongsonasx_(b_f16x8);
        sum_f32x8 = __lasx_xvfmadd_s(a_f32x8, b_f32x8, sum_f32x8);
    }
    __m256 high_f32x4 = (__m256)__lasx_xvpermi_q((__m256i)sum_f32x8, (__m256i)sum_f32x8, 0x11);
    __m256 sum_f32x4 = __lasx_xvfadd_s(sum_f32x8, high_f32x4);
    __m256 swapped_f32x4 = (__m256)__lasx_xvshuf4i_w((__m256i)sum_f32x4, 0b01001110);
    __m256 reduced_f32x4 = __lasx_xvfadd_s(sum_f32x4, swapped_f32x4);
    __m256 swapped_f32x2 = (__m256)__lasx_xvshuf4i_w((__m256i)reduced_f32x4, 0b10110001);
    __m256 reduced_f32x2 = __lasx_xvfadd_s(reduced_f32x4, swapped_f32x2);
    nk_fui32_t c;
    c.u = (nk_u32_t)__lasx_xvpickve2gr_w((__m256i)reduced_f32x2, 0);
    nk_f32_t sum = c.f;
    for (; idx_scalars < count_scalars; ++idx_scalars) {
        nk_f32_t a_val, b_val;
        nk_f16_to_f32_serial(&a_scalars[idx_scalars], &a_val);
        nk_f16_to_f32_serial(&b_scalars[idx_scalars], &b_val);
        sum += a_val * b_val;
    }
    *result = sum;
}

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars on LASX.
 *  @note Alias of nk_dot_through_f32_state_loongsonasx_t_
 */
typedef struct nk_dot_through_f32_state_loongsonasx_t_ nk_dot_f16x16_state_loongsonasx_t;

NK_INTERNAL void nk_dot_f16x16_init_loongsonasx(nk_dot_f16x16_state_loongsonasx_t *state) {
    nk_dot_through_f32_init_loongsonasx_(state);
}

NK_INTERNAL void nk_dot_f16x16_update_loongsonasx(nk_dot_f16x16_state_loongsonasx_t *state, nk_b256_vec_t a,
                                                  nk_b256_vec_t b, nk_size_t depth_offset,
                                                  nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256 a_low_f32x8 = __lasx_xvfcvtl_s_h(a.ymm);
    __m256 b_low_f32x8 = __lasx_xvfcvtl_s_h(b.ymm);
    state->sum_f32x8 = (__m256i)__lasx_xvfmadd_s(a_low_f32x8, b_low_f32x8, (__m256)state->sum_f32x8);
    __m256 a_high_f32x8 = __lasx_xvfcvth_s_h(a.ymm);
    __m256 b_high_f32x8 = __lasx_xvfcvth_s_h(b.ymm);
    state->sum_f32x8 = (__m256i)__lasx_xvfmadd_s(a_high_f32x8, b_high_f32x8, (__m256)state->sum_f32x8);
}

NK_INTERNAL void nk_dot_f16x16_finalize_loongsonasx(                                                    //
    nk_dot_f16x16_state_loongsonasx_t const *state_a, nk_dot_f16x16_state_loongsonasx_t const *state_b, //
    nk_dot_f16x16_state_loongsonasx_t const *state_c, nk_dot_f16x16_state_loongsonasx_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_dot_through_f32_finalize_loongsonasx_(state_a, state_b, state_c, state_d, total_dimensions, result);
}

#pragma endregion F16 and BF16 Floats

#pragma region Binary

typedef struct nk_dot_u1x256_state_loongsonasx_t {
    __m256i dot_count_u32x8;
} nk_dot_u1x256_state_loongsonasx_t;

NK_INTERNAL void nk_dot_u1x256_init_loongsonasx(nk_dot_u1x256_state_loongsonasx_t *state) {
    state->dot_count_u32x8 = __lasx_xvreplgr2vr_w(0);
}

NK_INTERNAL void nk_dot_u1x256_update_loongsonasx(nk_dot_u1x256_state_loongsonasx_t *state, nk_b256_vec_t a,
                                                  nk_b256_vec_t b, nk_size_t depth_offset,
                                                  nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256i and_u8x32 = __lasx_xvand_v(a.ymm, b.ymm);
    state->dot_count_u32x8 = __lasx_xvadd_w(state->dot_count_u32x8, __lasx_xvpcnt_w(and_u8x32));
}

NK_INTERNAL void nk_dot_u1x256_finalize_loongsonasx(                                                    //
    nk_dot_u1x256_state_loongsonasx_t const *state_a, nk_dot_u1x256_state_loongsonasx_t const *state_b, //
    nk_dot_u1x256_state_loongsonasx_t const *state_c, nk_dot_u1x256_state_loongsonasx_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // Step 1: Fold 8→4 in 256-bit (add high 128-bit lane to low), extract low 128 bits
    __m128i sum_a_u32x4 = nk_lasx_castsi256_si128_(__lasx_xvadd_w(
        state_a->dot_count_u32x8, __lasx_xvpermi_q(state_a->dot_count_u32x8, state_a->dot_count_u32x8, 0x11)));
    __m128i sum_b_u32x4 = nk_lasx_castsi256_si128_(__lasx_xvadd_w(
        state_b->dot_count_u32x8, __lasx_xvpermi_q(state_b->dot_count_u32x8, state_b->dot_count_u32x8, 0x11)));
    __m128i sum_c_u32x4 = nk_lasx_castsi256_si128_(__lasx_xvadd_w(
        state_c->dot_count_u32x8, __lasx_xvpermi_q(state_c->dot_count_u32x8, state_c->dot_count_u32x8, 0x11)));
    __m128i sum_d_u32x4 = nk_lasx_castsi256_si128_(__lasx_xvadd_w(
        state_d->dot_count_u32x8, __lasx_xvpermi_q(state_d->dot_count_u32x8, state_d->dot_count_u32x8, 0x11)));
    // Step 2: Transpose 4×4 in 128-bit via LSX interleave
    __m128i transpose_ab_low_u32x4 = __lsx_vilvl_w(sum_b_u32x4, sum_a_u32x4);
    __m128i transpose_cd_low_u32x4 = __lsx_vilvl_w(sum_d_u32x4, sum_c_u32x4);
    __m128i transpose_ab_high_u32x4 = __lsx_vilvh_w(sum_b_u32x4, sum_a_u32x4);
    __m128i transpose_cd_high_u32x4 = __lsx_vilvh_w(sum_d_u32x4, sum_c_u32x4);
    __m128i sum_lane0_u32x4 = __lsx_vilvl_d(transpose_cd_low_u32x4, transpose_ab_low_u32x4);
    __m128i sum_lane1_u32x4 = __lsx_vilvh_d(transpose_cd_low_u32x4, transpose_ab_low_u32x4);
    __m128i sum_lane2_u32x4 = __lsx_vilvl_d(transpose_cd_high_u32x4, transpose_ab_high_u32x4);
    __m128i sum_lane3_u32x4 = __lsx_vilvh_d(transpose_cd_high_u32x4, transpose_ab_high_u32x4);
    // Step 3: Vertical sum in 128-bit
    result->xmm = __lsx_vadd_w(__lsx_vadd_w(sum_lane0_u32x4, sum_lane1_u32x4),
                               __lsx_vadd_w(sum_lane2_u32x4, sum_lane3_u32x4));
}

#pragma endregion Binary

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH64_
#endif // NK_DOT_LOONGSONASX_H
