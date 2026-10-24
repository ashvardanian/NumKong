/**
 *  @brief SWAR-accelerated Dot Products for SIMD-free CPUs.
 *  @file include/numkong/dot/serial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_serial_instructions Serial Fallback Implementation
 *
 *  The serial backend provides portable scalar implementations for all numeric types without requiring
 *  any SIMD extensions. While significantly slower than vectorized implementations, these serve as:
 *
 *  - Reference implementations for correctness validation
 *  - Fallbacks for platforms without SIMD support (WASM, older CPUs)
 *  - Baseline for benchmarking vectorized speedups
 *
 *  For f64 dot products, compensated (Kahan-style) summation is used to minimize floating-point
 *  accumulation errors. For smaller types (f16, bf16, FP8), values are upcast to f32 for accumulation.
 *
 *  @section dot_serial_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_f64x2 state with compensated summation for numerical stability,
 *  - nk_dot_f32x4 state with simple f32 accumulation,
 *  - nk_dot_f16x8 state for f16 inputs via f32 upcasting,
 *  - nk_dot_bf16x8 state for bf16 inputs via f32 upcasting,
 *  - nk_dot_i8x16 for 8-bit signed integer inputs,
 *  - nk_dot_u8x16 for 8-bit unsigned integer inputs,
 *  - nk_dot_e4m3x16, nk_dot_e5m2x16, nk_dot_e2m3x16, nk_dot_e3m2x16 for FP8/FP6 inputs,
 *  - nk_dot_i4x16, nk_dot_u4x16 for 4-bit integer inputs.
 *
 *  @code{c}
 *  nk_dot_f64x2_state_serial_t state_first, state_second, state_third, state_fourth;
 *  nk_b128_vec_t query_f64x2, target_first_f64x2, target_second_f64x2, target_third_f64x2, target_fourth_f64x2;
 *  nk_dot_f64x2_init_serial(&state_first);
 *  nk_dot_f64x2_init_serial(&state_second);
 *  nk_dot_f64x2_init_serial(&state_third);
 *  nk_dot_f64x2_init_serial(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 2 <= depth; idx += 2) {
 *      query_f64x2.f64s[0] = query_ptr[idx], query_f64x2.f64s[1] = query_ptr[idx + 1];
 *      target_first_f64x2.f64s[0] = target_first_ptr[idx], target_first_f64x2.f64s[1] = target_first_ptr[idx + 1];
 *      target_second_f64x2.f64s[0] = target_second_ptr[idx], target_second_f64x2.f64s[1] = target_second_ptr[idx + 1];
 *      target_third_f64x2.f64s[0] = target_third_ptr[idx], target_third_f64x2.f64s[1] = target_third_ptr[idx + 1];
 *      target_fourth_f64x2.f64s[0] = target_fourth_ptr[idx], target_fourth_f64x2.f64s[1] = target_fourth_ptr[idx + 1];
 *      nk_dot_f64x2_update_serial(&state_first, query_f64x2, target_first_f64x2, idx, 2);
 *      nk_dot_f64x2_update_serial(&state_second, query_f64x2, target_second_f64x2, idx, 2);
 *      nk_dot_f64x2_update_serial(&state_third, query_f64x2, target_third_f64x2, idx, 2);
 *      nk_dot_f64x2_update_serial(&state_fourth, query_f64x2, target_fourth_f64x2, idx, 2);
 *  }
 *  nk_b256_vec_t results_f64x4;
 *  nk_dot_f64x2_finalize_serial(&state_first, &state_second, &state_third, &state_fourth, depth, &results_f64x4);
 *  @endcode
 *
 *  Integer types follow a similar pattern with appropriate type changes:
 *
 *  @code{c}
 *  nk_dot_i8x16_state_serial_t state_first, state_second, state_third, state_fourth;
 *  nk_b128_vec_t query_i8x16, target_first_i8x16, target_second_i8x16, target_third_i8x16, target_fourth_i8x16;
 *  nk_dot_i8x16_init_serial(&state_first);
 *  nk_dot_i8x16_init_serial(&state_second);
 *  nk_dot_i8x16_init_serial(&state_third);
 *  nk_dot_i8x16_init_serial(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 16 <= depth; idx += 16) {
 *      memcpy(query_i8x16.i8s, query_ptr + idx, 16);
 *      memcpy(target_first_i8x16.i8s, target_first_ptr + idx, 16);
 *      memcpy(target_second_i8x16.i8s, target_second_ptr + idx, 16);
 *      memcpy(target_third_i8x16.i8s, target_third_ptr + idx, 16);
 *      memcpy(target_fourth_i8x16.i8s, target_fourth_ptr + idx, 16);
 *      nk_dot_i8x16_update_serial(&state_first, query_i8x16, target_first_i8x16, idx, 16);
 *      nk_dot_i8x16_update_serial(&state_second, query_i8x16, target_second_i8x16, idx, 16);
 *      nk_dot_i8x16_update_serial(&state_third, query_i8x16, target_third_i8x16, idx, 16);
 *      nk_dot_i8x16_update_serial(&state_fourth, query_i8x16, target_fourth_i8x16, idx, 16);
 *  }
 *  nk_b128_vec_t results_i32x4;
 *  nk_dot_i8x16_finalize_serial(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 */
#ifndef NK_DOT_SERIAL_H
#define NK_DOT_SERIAL_H

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // `nk_f64_abs_`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Macro for dot product with simple accumulation.
 */
#define nk_define_dot_(input_type, accumulator_type, output_type, load_and_convert)                         \
    NK_PUBLIC void nk_dot_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, nk_##output_type##_t *result) {                \
        nk_##accumulator_type##_t sum = 0, a_val, b_val;                                                    \
        for (nk_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &a_val);                                                                \
            load_and_convert(b + i, &b_val);                                                                \
            sum += a_val * b_val;                                                                           \
        }                                                                                                   \
        *result = (nk_##output_type##_t)sum;                                                                \
    }

#define nk_define_dot_complex_(input_type, accumulator_type, output_complex_type, load_and_convert)        \
    NK_PUBLIC void nk_dot_##input_type##_serial(nk_##input_type##_t const *a_pairs,                        \
                                                nk_##input_type##_t const *b_pairs, nk_size_t count_pairs, \
                                                nk_##output_complex_type##_t *result) {                    \
        nk_##accumulator_type##_t sum_real = 0, sum_imag = 0;                                              \
        nk_##accumulator_type##_t a_real, b_real, a_imag, b_imag;                                          \
        for (nk_size_t i = 0; i != count_pairs; ++i) {                                                     \
            load_and_convert(&(a_pairs + i)->real, &a_real);                                               \
            load_and_convert(&(b_pairs + i)->real, &b_real);                                               \
            load_and_convert(&(a_pairs + i)->imag, &a_imag);                                               \
            load_and_convert(&(b_pairs + i)->imag, &b_imag);                                               \
            sum_real += a_real * b_real - a_imag * b_imag;                                                 \
            sum_imag += a_real * b_imag + a_imag * b_real;                                                 \
        }                                                                                                  \
        result->real = sum_real;                                                                           \
        result->imag = sum_imag;                                                                           \
    }

#define nk_define_vdot_complex_(input_type, accumulator_type, output_complex_type, load_and_convert)        \
    NK_PUBLIC void nk_vdot_##input_type##_serial(nk_##input_type##_t const *a_pairs,                        \
                                                 nk_##input_type##_t const *b_pairs, nk_size_t count_pairs, \
                                                 nk_##output_complex_type##_t *result) {                    \
        nk_##accumulator_type##_t sum_real = 0, sum_imag = 0;                                               \
        nk_##accumulator_type##_t a_real, b_real, a_imag, b_imag;                                           \
        for (nk_size_t i = 0; i != count_pairs; ++i) {                                                      \
            load_and_convert(&(a_pairs + i)->real, &a_real);                                                \
            load_and_convert(&(b_pairs + i)->real, &b_real);                                                \
            load_and_convert(&(a_pairs + i)->imag, &a_imag);                                                \
            load_and_convert(&(b_pairs + i)->imag, &b_imag);                                                \
            sum_real += a_real * b_real + a_imag * b_imag;                                                  \
            sum_imag += a_real * b_imag - a_imag * b_real;                                                  \
        }                                                                                                   \
        result->real = sum_real;                                                                            \
        result->imag = sum_imag;                                                                            \
    }

#pragma region - Traditional Floats

nk_define_dot_(f32, f64, f64, nk_assign_from_to_)            // nk_dot_f32_serial
nk_define_dot_complex_(f32c, f64, f64c, nk_assign_from_to_)  // nk_dot_f32c_serial
nk_define_vdot_complex_(f32c, f64, f64c, nk_assign_from_to_) // nk_vdot_f32c_serial

#pragma endregion - Traditional Floats

#pragma region - Smaller Floats

nk_define_dot_(f16, f32, f32, nk_f16_to_f32_serial)            // nk_dot_f16_serial
nk_define_dot_complex_(f16c, f32, f32c, nk_f16_to_f32_serial)  // nk_dot_f16c_serial
nk_define_vdot_complex_(f16c, f32, f32c, nk_f16_to_f32_serial) // nk_vdot_f16c_serial

nk_define_dot_(bf16, f32, f32, nk_bf16_to_f32_serial)            // nk_dot_bf16_serial
nk_define_dot_complex_(bf16c, f32, f32c, nk_bf16_to_f32_serial)  // nk_dot_bf16c_serial
nk_define_vdot_complex_(bf16c, f32, f32c, nk_bf16_to_f32_serial) // nk_vdot_bf16c_serial

nk_define_dot_(e4m3, f32, f32, nk_e4m3_to_f32_serial) // nk_dot_e4m3_serial
nk_define_dot_(e5m2, f32, f32, nk_e5m2_to_f32_serial) // nk_dot_e5m2_serial
nk_define_dot_(e2m3, f32, f32, nk_e2m3_to_f32_serial) // nk_dot_e2m3_serial
nk_define_dot_(e3m2, f32, f32, nk_e3m2_to_f32_serial) // nk_dot_e3m2_serial

#pragma endregion - Smaller Floats

#pragma region - Small Integers

nk_define_dot_(i8, i32, i32, nk_assign_from_to_) // nk_dot_i8_serial
nk_define_dot_(u8, u32, u32, nk_assign_from_to_) // nk_dot_u8_serial

#undef nk_define_dot_
#undef nk_define_dot_complex_
#undef nk_define_vdot_complex_

NK_PUBLIC void nk_dot_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // Sign extension: (nibble ^ 8) - 8 maps [0,15] to [-8,7]
    n = nk_size_round_up_to_multiple_(n, 2);
    nk_size_t n_bytes = n / 2;
    nk_i32_t sum = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        nk_i32_t a_low = (nk_i32_t)nk_i4x2_low_(a[i]);
        nk_i32_t b_low = (nk_i32_t)nk_i4x2_low_(b[i]);
        nk_i32_t a_high = (nk_i32_t)nk_i4x2_high_(a[i]);
        nk_i32_t b_high = (nk_i32_t)nk_i4x2_high_(b[i]);
        sum += a_low * b_low + a_high * b_high;
    }
    *result = sum;
}

NK_PUBLIC void nk_dot_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // No sign extension needed - values are ∈ [0,15].
    n = nk_size_round_up_to_multiple_(n, 2);
    nk_size_t n_bytes = n / 2;
    nk_u32_t sum = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        nk_u32_t a_low = (nk_u32_t)nk_u4x2_low_(a[i]);
        nk_u32_t b_low = (nk_u32_t)nk_u4x2_low_(b[i]);
        nk_u32_t a_high = (nk_u32_t)nk_u4x2_high_(a[i]);
        nk_u32_t b_high = (nk_u32_t)nk_u4x2_high_(b[i]);
        sum += a_low * b_low + a_high * b_high;
    }
    *result = sum;
}

#pragma endregion - Small Integers

#pragma region - Traditional Floats

/*  Double-precision dot-produce variants
 *
 *  Implements Neumaier's Kahan-Babuška variant to minimize floating-point rounding errors.
 *  Unlike Kahan, Neumaier handles the case where the term being added is larger than the
 *  running sum. Achieves O(1) error growth regardless of vector dimension.
 *
 *  Algorithm: For each term, compute t = sum + term, then:
 *    - If ‖sum‖ ≥ ‖term‖: c += (sum - t) + term  (lost low-order bits of term)
 *    - Else:              c += (term - t) + sum  (lost low-order bits of sum)
 *
 *  @see Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen"
 */
NK_PUBLIC void nk_dot_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t i = 0; i != n; ++i) nk_f64_dot2_(&sum, &compensation, a[i], b[i]);
    *result = sum + compensation;
}

NK_PUBLIC void nk_dot_f64c_serial(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                  nk_f64c_t *result) {
    nk_f64_t sum_real = 0, sum_imag = 0, compensation_real = 0, compensation_imag = 0;
    for (nk_size_t i = 0; i != count_pairs; ++i) {
        nk_f64_t a_real = a_pairs[i].real, b_real = b_pairs[i].real;
        nk_f64_t a_imag = a_pairs[i].imag, b_imag = b_pairs[i].imag;
        nk_f64_dot2_(&sum_real, &compensation_real, a_real, b_real);
        nk_f64_dot2_(&sum_real, &compensation_real, -a_imag, b_imag);
        nk_f64_dot2_(&sum_imag, &compensation_imag, a_real, b_imag);
        nk_f64_dot2_(&sum_imag, &compensation_imag, a_imag, b_real);
    }
    result->real = sum_real + compensation_real;
    result->imag = sum_imag + compensation_imag;
}

NK_PUBLIC void nk_vdot_f64c_serial(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f64c_t *result) {
    nk_f64_t sum_real = 0, sum_imag = 0, compensation_real = 0, compensation_imag = 0;
    for (nk_size_t i = 0; i != count_pairs; ++i) {
        nk_f64_t a_real = a_pairs[i].real, b_real = b_pairs[i].real;
        nk_f64_t a_imag = a_pairs[i].imag, b_imag = b_pairs[i].imag;
        nk_f64_dot2_(&sum_real, &compensation_real, a_real, b_real);
        nk_f64_dot2_(&sum_real, &compensation_real, a_imag, b_imag);
        nk_f64_dot2_(&sum_imag, &compensation_imag, a_real, b_imag);
        nk_f64_dot2_(&sum_imag, &compensation_imag, -a_imag, b_real);
    }
    result->real = sum_real + compensation_real;
    result->imag = sum_imag + compensation_imag;
}

typedef struct nk_dot_f64x2_state_serial_t {
    nk_f64_t sums[2];
    nk_f64_t compensations[2];
} nk_dot_f64x2_state_serial_t;

NK_INTERNAL void nk_dot_f64x2_init_serial(nk_dot_f64x2_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0;
    state->compensations[0] = 0, state->compensations[1] = 0;
}

NK_INTERNAL void nk_dot_f64x2_update_serial(nk_dot_f64x2_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_f64_t sum0 = state->sums[0], compensation0 = state->compensations[0];
    nk_f64_t sum1 = state->sums[1], compensation1 = state->compensations[1];
    nk_f64_dot2_(&sum0, &compensation0, a.f64s[0], b.f64s[0]);
    nk_f64_dot2_(&sum1, &compensation1, a.f64s[1], b.f64s[1]);

    state->sums[0] = sum0, state->sums[1] = sum1;
    state->compensations[0] = compensation0, state->compensations[1] = compensation1;
}

NK_INTERNAL void nk_dot_f64x2_finalize_serial(                                              //
    nk_dot_f64x2_state_serial_t const *state_a, nk_dot_f64x2_state_serial_t const *state_b, //
    nk_dot_f64x2_state_serial_t const *state_c, nk_dot_f64x2_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f64s[0] = nk_reduce_sum_f64_serial_(state_a->sums, state_a->compensations, 2);
    result->f64s[1] = nk_reduce_sum_f64_serial_(state_b->sums, state_b->compensations, 2);
    result->f64s[2] = nk_reduce_sum_f64_serial_(state_c->sums, state_c->compensations, 2);
    result->f64s[3] = nk_reduce_sum_f64_serial_(state_d->sums, state_d->compensations, 2);
}

typedef struct nk_dot_f32x4_state_serial_t {
    nk_f64_t sums[4];
} nk_dot_f32x4_state_serial_t;

NK_INTERNAL void nk_dot_f32x4_init_serial(nk_dot_f32x4_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_f32x4_update_serial(nk_dot_f32x4_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_f64_t sum0 = state->sums[0];
    nk_f64_t sum1 = state->sums[1];
    nk_f64_t sum2 = state->sums[2];
    nk_f64_t sum3 = state->sums[3];
    sum0 += (nk_f64_t)a.f32s[0] * b.f32s[0], sum1 += (nk_f64_t)a.f32s[1] * b.f32s[1];
    sum2 += (nk_f64_t)a.f32s[2] * b.f32s[2], sum3 += (nk_f64_t)a.f32s[3] * b.f32s[3];
    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_f32x4_finalize_serial(                                              //
    nk_dot_f32x4_state_serial_t const *state_a, nk_dot_f32x4_state_serial_t const *state_b, //
    nk_dot_f32x4_state_serial_t const *state_c, nk_dot_f32x4_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f64s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f64s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f64s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f64s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

#pragma endregion - Traditional Floats

#pragma region - Smaller Floats

typedef struct nk_dot_f16x8_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_f16x8_state_serial_t;

NK_INTERNAL void nk_dot_f16x8_init_serial(nk_dot_f16x8_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_f16x8_update_serial(nk_dot_f16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (nk_size_t i = 0; i < 8; i += 4) {
        nk_f32_t a0, a1, a2, a3, b0, b1, b2, b3;
        nk_f16_to_f32_serial(a.f16s + i + 0, &a0), nk_f16_to_f32_serial(a.f16s + i + 1, &a1);
        nk_f16_to_f32_serial(a.f16s + i + 2, &a2), nk_f16_to_f32_serial(a.f16s + i + 3, &a3);
        nk_f16_to_f32_serial(b.f16s + i + 0, &b0), nk_f16_to_f32_serial(b.f16s + i + 1, &b1);
        nk_f16_to_f32_serial(b.f16s + i + 2, &b2), nk_f16_to_f32_serial(b.f16s + i + 3, &b3);
        sum0 += a0 * b0, sum1 += a1 * b1, sum2 += a2 * b2, sum3 += a3 * b3;
    }
    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_f16x8_finalize_serial(                                              //
    nk_dot_f16x8_state_serial_t const *state_a, nk_dot_f16x8_state_serial_t const *state_b, //
    nk_dot_f16x8_state_serial_t const *state_c, nk_dot_f16x8_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_bf16x8_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_bf16x8_state_serial_t;

NK_INTERNAL void nk_dot_bf16x8_init_serial(nk_dot_bf16x8_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_bf16x8_update_serial(nk_dot_bf16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (nk_size_t i = 0; i < 8; i += 4) {
        nk_f32_t a0, a1, a2, a3, b0, b1, b2, b3;
        nk_bf16_to_f32_serial(a.bf16s + i + 0, &a0), nk_bf16_to_f32_serial(a.bf16s + i + 1, &a1);
        nk_bf16_to_f32_serial(a.bf16s + i + 2, &a2), nk_bf16_to_f32_serial(a.bf16s + i + 3, &a3);
        nk_bf16_to_f32_serial(b.bf16s + i + 0, &b0), nk_bf16_to_f32_serial(b.bf16s + i + 1, &b1);
        nk_bf16_to_f32_serial(b.bf16s + i + 2, &b2), nk_bf16_to_f32_serial(b.bf16s + i + 3, &b3);
        sum0 += a0 * b0, sum1 += a1 * b1, sum2 += a2 * b2, sum3 += a3 * b3;
    }
    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_bf16x8_finalize_serial(                                               //
    nk_dot_bf16x8_state_serial_t const *state_a, nk_dot_bf16x8_state_serial_t const *state_b, //
    nk_dot_bf16x8_state_serial_t const *state_c, nk_dot_bf16x8_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

#pragma endregion - Smaller Floats

#pragma region - Small Integers

typedef struct nk_dot_i8x16_state_serial_t {
    nk_i64_t sums[2];
} nk_dot_i8x16_state_serial_t;

NK_INTERNAL void nk_dot_i8x16_init_serial(nk_dot_i8x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0;
}

NK_INTERNAL void nk_dot_i8x16_update_serial(nk_dot_i8x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_i64_t sum0 = state->sums[0];
    nk_i64_t sum1 = state->sums[1];
    sum0 += (nk_i16_t)a.i8s[0] * (nk_i16_t)b.i8s[0], sum1 += (nk_i16_t)a.i8s[1] * (nk_i16_t)b.i8s[1];
    sum0 += (nk_i16_t)a.i8s[2] * (nk_i16_t)b.i8s[2], sum1 += (nk_i16_t)a.i8s[3] * (nk_i16_t)b.i8s[3];
    sum0 += (nk_i16_t)a.i8s[4] * (nk_i16_t)b.i8s[4], sum1 += (nk_i16_t)a.i8s[5] * (nk_i16_t)b.i8s[5];
    sum0 += (nk_i16_t)a.i8s[6] * (nk_i16_t)b.i8s[6], sum1 += (nk_i16_t)a.i8s[7] * (nk_i16_t)b.i8s[7];
    sum0 += (nk_i16_t)a.i8s[8] * (nk_i16_t)b.i8s[8], sum1 += (nk_i16_t)a.i8s[9] * (nk_i16_t)b.i8s[9];
    sum0 += (nk_i16_t)a.i8s[10] * (nk_i16_t)b.i8s[10], sum1 += (nk_i16_t)a.i8s[11] * (nk_i16_t)b.i8s[11];
    sum0 += (nk_i16_t)a.i8s[12] * (nk_i16_t)b.i8s[12], sum1 += (nk_i16_t)a.i8s[13] * (nk_i16_t)b.i8s[13];
    sum0 += (nk_i16_t)a.i8s[14] * (nk_i16_t)b.i8s[14], sum1 += (nk_i16_t)a.i8s[15] * (nk_i16_t)b.i8s[15];
    state->sums[0] = sum0, state->sums[1] = sum1;
}

NK_INTERNAL void nk_dot_i8x16_finalize_serial(                                              //
    nk_dot_i8x16_state_serial_t const *state_a, nk_dot_i8x16_state_serial_t const *state_b, //
    nk_dot_i8x16_state_serial_t const *state_c, nk_dot_i8x16_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->i32s[0] = (nk_i32_t)(state_a->sums[0] + state_a->sums[1]);
    result->i32s[1] = (nk_i32_t)(state_b->sums[0] + state_b->sums[1]);
    result->i32s[2] = (nk_i32_t)(state_c->sums[0] + state_c->sums[1]);
    result->i32s[3] = (nk_i32_t)(state_d->sums[0] + state_d->sums[1]);
}

typedef struct nk_dot_u8x16_state_serial_t {
    nk_u64_t sums[2];
} nk_dot_u8x16_state_serial_t;

NK_INTERNAL void nk_dot_u8x16_init_serial(nk_dot_u8x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0;
}

NK_INTERNAL void nk_dot_u8x16_update_serial(nk_dot_u8x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_u64_t sum0 = state->sums[0];
    nk_u64_t sum1 = state->sums[1];

    sum0 += (nk_u16_t)a.u8s[0] * (nk_u16_t)b.u8s[0], sum1 += (nk_u16_t)a.u8s[1] * (nk_u16_t)b.u8s[1];
    sum0 += (nk_u16_t)a.u8s[2] * (nk_u16_t)b.u8s[2], sum1 += (nk_u16_t)a.u8s[3] * (nk_u16_t)b.u8s[3];
    sum0 += (nk_u16_t)a.u8s[4] * (nk_u16_t)b.u8s[4], sum1 += (nk_u16_t)a.u8s[5] * (nk_u16_t)b.u8s[5];
    sum0 += (nk_u16_t)a.u8s[6] * (nk_u16_t)b.u8s[6], sum1 += (nk_u16_t)a.u8s[7] * (nk_u16_t)b.u8s[7];
    sum0 += (nk_u16_t)a.u8s[8] * (nk_u16_t)b.u8s[8], sum1 += (nk_u16_t)a.u8s[9] * (nk_u16_t)b.u8s[9];
    sum0 += (nk_u16_t)a.u8s[10] * (nk_u16_t)b.u8s[10], sum1 += (nk_u16_t)a.u8s[11] * (nk_u16_t)b.u8s[11];
    sum0 += (nk_u16_t)a.u8s[12] * (nk_u16_t)b.u8s[12], sum1 += (nk_u16_t)a.u8s[13] * (nk_u16_t)b.u8s[13];
    sum0 += (nk_u16_t)a.u8s[14] * (nk_u16_t)b.u8s[14], sum1 += (nk_u16_t)a.u8s[15] * (nk_u16_t)b.u8s[15];
    state->sums[0] = sum0, state->sums[1] = sum1;
}

NK_INTERNAL void nk_dot_u8x16_finalize_serial(                                              //
    nk_dot_u8x16_state_serial_t const *state_a, nk_dot_u8x16_state_serial_t const *state_b, //
    nk_dot_u8x16_state_serial_t const *state_c, nk_dot_u8x16_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->u32s[0] = (nk_u32_t)(state_a->sums[0] + state_a->sums[1]);
    result->u32s[1] = (nk_u32_t)(state_b->sums[0] + state_b->sums[1]);
    result->u32s[2] = (nk_u32_t)(state_c->sums[0] + state_c->sums[1]);
    result->u32s[3] = (nk_u32_t)(state_d->sums[0] + state_d->sums[1]);
}

#pragma endregion - Small Integers

#pragma region - Smaller Floats

typedef struct nk_dot_e4m3x16_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_e4m3x16_state_serial_t;

NK_INTERNAL void nk_dot_e4m3x16_init_serial(nk_dot_e4m3x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_e4m3x16_update_serial(nk_dot_e4m3x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_f32_t sum0 = state->sums[0];
    nk_f32_t sum1 = state->sums[1];
    nk_f32_t sum2 = state->sums[2];
    nk_f32_t sum3 = state->sums[3];
    nk_f32_t ai0, ai1, ai2, ai3;
    nk_f32_t bi0, bi1, bi2, bi3;
    for (nk_size_t i = 0; i != 16; i += 4) {
        nk_e4m3_to_f32_serial(a.e4m3s + i, &ai0), nk_e4m3_to_f32_serial(b.e4m3s + i, &bi0);
        nk_e4m3_to_f32_serial(a.e4m3s + i + 1, &ai1), nk_e4m3_to_f32_serial(b.e4m3s + i + 1, &bi1);
        nk_e4m3_to_f32_serial(a.e4m3s + i + 2, &ai2), nk_e4m3_to_f32_serial(b.e4m3s + i + 2, &bi2);
        nk_e4m3_to_f32_serial(a.e4m3s + i + 3, &ai3), nk_e4m3_to_f32_serial(b.e4m3s + i + 3, &bi3);
        sum0 += ai0 * bi0, sum1 += ai1 * bi1, sum2 += ai2 * bi2, sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_e4m3x16_finalize_serial(                                                //
    nk_dot_e4m3x16_state_serial_t const *state_a, nk_dot_e4m3x16_state_serial_t const *state_b, //
    nk_dot_e4m3x16_state_serial_t const *state_c, nk_dot_e4m3x16_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_e5m2x16_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_e5m2x16_state_serial_t;

NK_INTERNAL void nk_dot_e5m2x16_init_serial(nk_dot_e5m2x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_e5m2x16_update_serial(nk_dot_e5m2x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_f32_t sum0 = state->sums[0];
    nk_f32_t sum1 = state->sums[1];
    nk_f32_t sum2 = state->sums[2];
    nk_f32_t sum3 = state->sums[3];
    nk_f32_t ai0, ai1, ai2, ai3;
    nk_f32_t bi0, bi1, bi2, bi3;
    for (nk_size_t i = 0; i != 16; i += 4) {
        nk_e5m2_to_f32_serial(a.e5m2s + i, &ai0), nk_e5m2_to_f32_serial(b.e5m2s + i, &bi0);
        nk_e5m2_to_f32_serial(a.e5m2s + i + 1, &ai1), nk_e5m2_to_f32_serial(b.e5m2s + i + 1, &bi1);
        nk_e5m2_to_f32_serial(a.e5m2s + i + 2, &ai2), nk_e5m2_to_f32_serial(b.e5m2s + i + 2, &bi2);
        nk_e5m2_to_f32_serial(a.e5m2s + i + 3, &ai3), nk_e5m2_to_f32_serial(b.e5m2s + i + 3, &bi3);
        sum0 += ai0 * bi0, sum1 += ai1 * bi1, sum2 += ai2 * bi2, sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_e5m2x16_finalize_serial(                                                //
    nk_dot_e5m2x16_state_serial_t const *state_a, nk_dot_e5m2x16_state_serial_t const *state_b, //
    nk_dot_e5m2x16_state_serial_t const *state_c, nk_dot_e5m2x16_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_e2m3x16_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_e2m3x16_state_serial_t;

NK_INTERNAL void nk_dot_e2m3x16_init_serial(nk_dot_e2m3x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_e2m3x16_update_serial(nk_dot_e2m3x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_f32_t sum0 = state->sums[0];
    nk_f32_t sum1 = state->sums[1];
    nk_f32_t sum2 = state->sums[2];
    nk_f32_t sum3 = state->sums[3];
    nk_f32_t ai0, ai1, ai2, ai3;
    nk_f32_t bi0, bi1, bi2, bi3;
    for (nk_size_t i = 0; i != 16; i += 4) {
        nk_e2m3_to_f32_serial(a.e2m3s + i, &ai0), nk_e2m3_to_f32_serial(b.e2m3s + i, &bi0);
        nk_e2m3_to_f32_serial(a.e2m3s + i + 1, &ai1), nk_e2m3_to_f32_serial(b.e2m3s + i + 1, &bi1);
        nk_e2m3_to_f32_serial(a.e2m3s + i + 2, &ai2), nk_e2m3_to_f32_serial(b.e2m3s + i + 2, &bi2);
        nk_e2m3_to_f32_serial(a.e2m3s + i + 3, &ai3), nk_e2m3_to_f32_serial(b.e2m3s + i + 3, &bi3);
        sum0 += ai0 * bi0, sum1 += ai1 * bi1, sum2 += ai2 * bi2, sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_e2m3x16_finalize_serial(                                                //
    nk_dot_e2m3x16_state_serial_t const *state_a, nk_dot_e2m3x16_state_serial_t const *state_b, //
    nk_dot_e2m3x16_state_serial_t const *state_c, nk_dot_e2m3x16_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_e3m2x16_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_e3m2x16_state_serial_t;

NK_INTERNAL void nk_dot_e3m2x16_init_serial(nk_dot_e3m2x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_e3m2x16_update_serial(nk_dot_e3m2x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_f32_t sum0 = state->sums[0];
    nk_f32_t sum1 = state->sums[1];
    nk_f32_t sum2 = state->sums[2];
    nk_f32_t sum3 = state->sums[3];
    nk_f32_t ai0, ai1, ai2, ai3;
    nk_f32_t bi0, bi1, bi2, bi3;
    for (nk_size_t i = 0; i != 16; i += 4) {
        nk_e3m2_to_f32_serial(a.e3m2s + i, &ai0), nk_e3m2_to_f32_serial(b.e3m2s + i, &bi0);
        nk_e3m2_to_f32_serial(a.e3m2s + i + 1, &ai1), nk_e3m2_to_f32_serial(b.e3m2s + i + 1, &bi1);
        nk_e3m2_to_f32_serial(a.e3m2s + i + 2, &ai2), nk_e3m2_to_f32_serial(b.e3m2s + i + 2, &bi2);
        nk_e3m2_to_f32_serial(a.e3m2s + i + 3, &ai3), nk_e3m2_to_f32_serial(b.e3m2s + i + 3, &bi3);
        sum0 += ai0 * bi0, sum1 += ai1 * bi1, sum2 += ai2 * bi2, sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_e3m2x16_finalize_serial(                                                //
    nk_dot_e3m2x16_state_serial_t const *state_a, nk_dot_e3m2x16_state_serial_t const *state_b, //
    nk_dot_e3m2x16_state_serial_t const *state_c, nk_dot_e3m2x16_state_serial_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

#pragma endregion - Smaller Floats

#pragma region - Small Integers

// U4x2 state: processes 16 nibbles (8 bytes = 64 bits) per update
typedef struct nk_dot_u4x16_state_serial_t {
    nk_u64_t sums[2]; // sums[0]: low nibbles, sums[1]: high nibbles
} nk_dot_u4x16_state_serial_t;

NK_INTERNAL void nk_dot_u4x16_init_serial(nk_dot_u4x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0;
}

NK_INTERNAL void nk_dot_u4x16_update_serial(nk_dot_u4x16_state_serial_t *state, nk_b64_vec_t a, nk_b64_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Process 8 bytes (16 nibbles total) using SWAR
    // Separate accumulators for low and high nibbles
    nk_u64_t sum_low = state->sums[0];
    nk_u64_t sum_high = state->sums[1];

    // Process all 8 bytes, extracting and multiplying nibbles
    for (nk_size_t i = 0; i < 8; i++) {
        nk_u8_t a_byte = a.u8s[i];
        nk_u8_t b_byte = b.u8s[i];

        // Extract low and high nibbles using SWAR masks
        nk_u8_t a_low = a_byte & 0x0F;
        nk_u8_t b_low = b_byte & 0x0F;
        nk_u8_t a_high = (a_byte >> 4) & 0x0F;
        nk_u8_t b_high = (b_byte >> 4) & 0x0F;

        // Accumulate products into separate accumulators
        sum_low += (nk_u32_t)a_low * (nk_u32_t)b_low;
        sum_high += (nk_u32_t)a_high * (nk_u32_t)b_high;
    }

    state->sums[0] = sum_low, state->sums[1] = sum_high;
}

NK_INTERNAL void nk_dot_u4x16_finalize_serial(nk_dot_u4x16_state_serial_t const *state_a,
                                              nk_dot_u4x16_state_serial_t const *state_b,
                                              nk_dot_u4x16_state_serial_t const *state_c,
                                              nk_dot_u4x16_state_serial_t const *state_d, nk_size_t total_dimensions,
                                              nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->u32s[0] = (nk_u32_t)(state_a->sums[0] + state_a->sums[1]);
    result->u32s[1] = (nk_u32_t)(state_b->sums[0] + state_b->sums[1]);
    result->u32s[2] = (nk_u32_t)(state_c->sums[0] + state_c->sums[1]);
    result->u32s[3] = (nk_u32_t)(state_d->sums[0] + state_d->sums[1]);
}

NK_INTERNAL void nk_load_i4x16_to_i8x16_serial_(void const *src, nk_b128_vec_t *dst) {
    nk_i4_to_i8_serial_((nk_i4x2_t const *)src, dst->i8s, 16);
}

NK_INTERNAL void nk_partial_load_i4x16_to_i8x16_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    nk_i4_to_i8_serial_((nk_i4x2_t const *)src, dst->i8s, n);
    for (nk_size_t i = n; i < 16; ++i) dst->i8s[i] = 0;
}

NK_INTERNAL void nk_load_u4x16_to_u8x16_serial_(void const *src, nk_b128_vec_t *dst) {
    nk_u4_to_u8_serial_((nk_u4x2_t const *)src, dst->u8s, 16);
}

NK_INTERNAL void nk_partial_load_u4x16_to_u8x16_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    nk_u4_to_u8_serial_((nk_u4x2_t const *)src, dst->u8s, n);
    for (nk_size_t i = n; i < 16; ++i) dst->u8s[i] = 0;
}

typedef struct nk_dot_i4x16_state_serial_t {
    nk_i64_t sums[2]; // sums[0]: low nibbles, sums[1]: high nibbles
} nk_dot_i4x16_state_serial_t;

NK_INTERNAL void nk_dot_i4x16_init_serial(nk_dot_i4x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0;
}

NK_INTERNAL void nk_dot_i4x16_update_serial(nk_dot_i4x16_state_serial_t *state, nk_b64_vec_t a, nk_b64_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Process 8 bytes (16 nibbles total) using SWAR with sign extension
    // Separate accumulators for low and high nibbles
    nk_i64_t sum_low = state->sums[0];
    nk_i64_t sum_high = state->sums[1];

    // Process all 8 bytes, extracting and multiplying signed nibbles
    for (nk_size_t i = 0; i < 8; i++) {
        nk_u8_t a_byte = a.u8s[i];
        nk_u8_t b_byte = b.u8s[i];

        // Extract nibbles and sign extend: (nibble ^ 8) - 8 maps [0,15] → [-8,7]
        nk_i8_t a_low = (nk_i8_t)(((a_byte & 0x0F) ^ 8) - 8);
        nk_i8_t b_low = (nk_i8_t)(((b_byte & 0x0F) ^ 8) - 8);
        nk_i8_t a_high = (nk_i8_t)((((a_byte >> 4) & 0x0F) ^ 8) - 8);
        nk_i8_t b_high = (nk_i8_t)((((b_byte >> 4) & 0x0F) ^ 8) - 8);

        // Accumulate products into separate accumulators
        sum_low += (nk_i32_t)a_low * (nk_i32_t)b_low;
        sum_high += (nk_i32_t)a_high * (nk_i32_t)b_high;
    }

    state->sums[0] = sum_low, state->sums[1] = sum_high;
}

NK_INTERNAL void nk_dot_i4x16_finalize_serial(nk_dot_i4x16_state_serial_t const *state_a,
                                              nk_dot_i4x16_state_serial_t const *state_b,
                                              nk_dot_i4x16_state_serial_t const *state_c,
                                              nk_dot_i4x16_state_serial_t const *state_d, nk_size_t total_dimensions,
                                              nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->i32s[0] = (nk_i32_t)(state_a->sums[0] + state_a->sums[1]);
    result->i32s[1] = (nk_i32_t)(state_b->sums[0] + state_b->sums[1]);
    result->i32s[2] = (nk_i32_t)(state_c->sums[0] + state_c->sums[1]);
    result->i32s[3] = (nk_i32_t)(state_d->sums[0] + state_d->sums[1]);
}

#pragma endregion - Small Integers

#pragma region - Binary

NK_PUBLIC void nk_dot_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result) {
    nk_u32_t dot = 0;
    nk_size_t bytes = nk_size_divide_round_up_(n_bits, NK_BITS_PER_BYTE);
    for (nk_size_t i = 0; i < bytes; ++i) dot += nk_u1x8_popcount_(((nk_u8_t const *)a)[i] & ((nk_u8_t const *)b)[i]);
    *result = dot;
}

typedef struct nk_dot_u1x128_state_serial_t {
    nk_u32_t dot_count;
} nk_dot_u1x128_state_serial_t;

NK_INTERNAL void nk_dot_u1x128_init_serial(nk_dot_u1x128_state_serial_t *state) { state->dot_count = 0; }

NK_INTERNAL void nk_dot_u1x128_update_serial(nk_dot_u1x128_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_u64_t and_low = a.u64s[0] & b.u64s[0];
    nk_u64_t and_high = a.u64s[1] & b.u64s[1];
    state->dot_count += (nk_u32_t)nk_u64_popcount_(and_low);
    state->dot_count += (nk_u32_t)nk_u64_popcount_(and_high);
}

NK_INTERNAL void nk_dot_u1x128_finalize_serial(nk_dot_u1x128_state_serial_t const *state_a,
                                               nk_dot_u1x128_state_serial_t const *state_b,
                                               nk_dot_u1x128_state_serial_t const *state_c,
                                               nk_dot_u1x128_state_serial_t const *state_d, nk_size_t total_dimensions,
                                               nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->u32s[0] = state_a->dot_count;
    result->u32s[1] = state_b->dot_count;
    result->u32s[2] = state_c->dot_count;
    result->u32s[3] = state_d->dot_count;
}

#pragma endregion - Binary

/**
 *  Serial fallback sum helpers for progressive element-sum accumulation.
 *  Used by the compensated symmetric GEMM macro to piggyback sum computation
 *  on the depth loop's already-loaded vectors, avoiding a separate sum pass.
 */

#pragma region - Stateful Element Sum Helpers (for compensated GEMM)

/* i4x32: Haswell i4 (nk_b128_vec_t containing 32 nibbles in 16 bytes) */
typedef struct nk_sum_i4x32_state_serial_t {
    nk_i64_t sum;
} nk_sum_i4x32_state_serial_t;

NK_INTERNAL void nk_sum_i4x32_init_serial(nk_sum_i4x32_state_serial_t *state) { state->sum = 0; }

NK_INTERNAL void nk_sum_i4x32_update_serial(nk_sum_i4x32_state_serial_t *state, nk_b128_vec_t v) {
    nk_u8_t const *d = (nk_u8_t const *)&v;
    for (int i = 0; i < 16; i++) {
        nk_i8_t low = (nk_i8_t)((d[i] & 0x0F) ^ 0x08) - 8; /* sign-extend low nibble */
        nk_i8_t high = (nk_i8_t)((d[i] >> 4) ^ 0x08) - 8;  /* sign-extend high nibble */
        state->sum += low + high;
    }
}

NK_INTERNAL nk_i32_t nk_sum_i4x32_finalize_serial(nk_sum_i4x32_state_serial_t const *state, nk_size_t count) {
    nk_unused_(count);
    return (nk_i32_t)state->sum;
}

#pragma endregion - Stateful Element Sum Helpers

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_DOT_SERIAL_H
