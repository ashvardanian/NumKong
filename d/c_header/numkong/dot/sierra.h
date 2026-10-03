/**
 *  @brief SIMD-accelerated Dot Products for Sierra Forest.
 *  @file include/numkong/dot/sierra.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_sierra_instructions AVX-VNNI-INT8 Instructions
 *
 *      Intrinsic                   Instruction
 *      _mm256_dpbssd_epi32         VPDPBSSD (YMM, YMM, YMM)   i8 x i8 -> i32
 *      _mm256_dpbuud_epi32         VPDPBUUD (YMM, YMM, YMM)   u8 x u8 -> u32
 *
 *  Sierra Forest CPUs support AVX-VNNI-INT8, adding native signed*signed and
 *  unsigned*unsigned 8-bit dot products. This eliminates the algebraic sign
 *  transformations required on Alder Lake (AVX-VNNI only).
 *
 *  @section dot_sierra_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_i8x32 for 8-bit signed integer inputs using native DPBSSD (no algebraic transform),
 *  - nk_dot_u8x32 for 8-bit unsigned integer inputs using native DPBUUD (no algebraic transform).
 *
 *  Each state struct contains only a single accumulator field (no correction terms needed).
 *
 *  @code{c}
 *  nk_dot_i8x32_state_sierra_t state_first, state_second, state_third, state_fourth;
 *  nk_b256_vec_t query_i8x32, target_first_i8x32, target_second_i8x32, target_third_i8x32, target_fourth_i8x32;
 *  nk_dot_i8x32_init_sierra(&state_first);
 *  nk_dot_i8x32_init_sierra(&state_second);
 *  nk_dot_i8x32_init_sierra(&state_third);
 *  nk_dot_i8x32_init_sierra(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 32 <= depth; idx += 32) {
 *      query_i8x32.ymm = _mm256_loadu_si256(query_ptr + idx);
 *      target_first_i8x32.ymm = _mm256_loadu_si256(target_first_ptr + idx);
 *      target_second_i8x32.ymm = _mm256_loadu_si256(target_second_ptr + idx);
 *      target_third_i8x32.ymm = _mm256_loadu_si256(target_third_ptr + idx);
 *      target_fourth_i8x32.ymm = _mm256_loadu_si256(target_fourth_ptr + idx);
 *      nk_dot_i8x32_update_sierra(&state_first, query_i8x32, target_first_i8x32, idx, 32);
 *      nk_dot_i8x32_update_sierra(&state_second, query_i8x32, target_second_i8x32, idx, 32);
 *      nk_dot_i8x32_update_sierra(&state_third, query_i8x32, target_third_i8x32, idx, 32);
 *      nk_dot_i8x32_update_sierra(&state_fourth, query_i8x32, target_fourth_i8x32, idx, 32);
 *  }
 *  nk_b128_vec_t results_i32x4;
 *  nk_dot_i8x32_finalize_sierra(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 *
 *  The unsigned variant follows the same pattern with appropriate type changes:
 *
 *  @code{c}
 *  nk_dot_u8x32_state_sierra_t state_first, state_second, state_third, state_fourth;
 *  nk_b256_vec_t query_u8x32, target_first_u8x32, target_second_u8x32, target_third_u8x32, target_fourth_u8x32;
 *  nk_dot_u8x32_init_sierra(&state_first);
 *  nk_dot_u8x32_init_sierra(&state_second);
 *  nk_dot_u8x32_init_sierra(&state_third);
 *  nk_dot_u8x32_init_sierra(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 32 <= depth; idx += 32) {
 *      query_u8x32.ymm = _mm256_loadu_si256(query_ptr + idx);
 *      target_first_u8x32.ymm = _mm256_loadu_si256(target_first_ptr + idx);
 *      target_second_u8x32.ymm = _mm256_loadu_si256(target_second_ptr + idx);
 *      target_third_u8x32.ymm = _mm256_loadu_si256(target_third_ptr + idx);
 *      target_fourth_u8x32.ymm = _mm256_loadu_si256(target_fourth_ptr + idx);
 *      nk_dot_u8x32_update_sierra(&state_first, query_u8x32, target_first_u8x32, idx, 32);
 *      nk_dot_u8x32_update_sierra(&state_second, query_u8x32, target_second_u8x32, idx, 32);
 *      nk_dot_u8x32_update_sierra(&state_third, query_u8x32, target_third_u8x32, idx, 32);
 *      nk_dot_u8x32_update_sierra(&state_fourth, query_u8x32, target_fourth_u8x32, idx, 32);
 *  }
 *  nk_b128_vec_t results_u32x4;
 *  nk_dot_u8x32_finalize_sierra(&state_first, &state_second, &state_third, &state_fourth, depth, &results_u32x4);
 *  @endcode
 */
#ifndef NK_DOT_SIERRA_H
#define NK_DOT_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA

#include "numkong/types.h"
#include "numkong/cast/serial.h"    // `nk_partial_load_b8x32_serial_`
#include "numkong/reduce/haswell.h" // `nk_reduce_add_i32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

NK_PUBLIC void nk_dot_i8_sierra(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                nk_i32_t *result) {
    // Native i8*i8 dot product using DPBSSD (signed * signed -> i32)
    // No algebraic transformation needed - dpbssd handles signed*signed directly.
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i a_i8x32, b_i8x32;

nk_dot_i8_sierra_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_i8x32 = _mm256_load_si256(&a_vec.ymm);
        b_i8x32 = _mm256_load_si256(&b_vec.ymm);
        count_scalars = 0;
    }
    else {
        a_i8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_i8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // VPDPBSSD: signed i8 * signed i8 -> i32 accumulation
    sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, a_i8x32, b_i8x32);

    if (count_scalars) goto nk_dot_i8_sierra_cycle;

    *result = nk_reduce_add_i32x8_haswell_(sum_i32x8);
}

typedef struct nk_dot_i8x32_state_sierra_t {
    __m256i sum_i32x8; // DPBSSD accumulator: i8 * i8 -> i32
} nk_dot_i8x32_state_sierra_t;

NK_INTERNAL void nk_dot_i8x32_init_sierra(nk_dot_i8x32_state_sierra_t *state) {
    state->sum_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_i8x32_update_sierra(nk_dot_i8x32_state_sierra_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->sum_i32x8 = _mm256_dpbssd_epi32(state->sum_i32x8, a.ymm, b.ymm);
}

NK_INTERNAL void nk_dot_i8x32_finalize_sierra(                                              //
    nk_dot_i8x32_state_sierra_t const *state_a, nk_dot_i8x32_state_sierra_t const *state_b, //
    nk_dot_i8x32_state_sierra_t const *state_c, nk_dot_i8x32_state_sierra_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *results) {
    nk_unused_(total_dimensions);

    // ILP-optimized 4-way horizontal reduction: i32x8 -> scalar i32
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->sum_i32x8),
                                        _mm256_extracti128_si256(state_a->sum_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->sum_i32x8),
                                        _mm256_extracti128_si256(state_b->sum_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->sum_i32x8),
                                        _mm256_extracti128_si256(state_c->sum_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->sum_i32x8),
                                        _mm256_extracti128_si256(state_d->sum_i32x8, 1));

    // Transpose and reduce
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    results->xmm = _mm_add_epi32(_mm_add_epi32(lane0_i32x4, lane1_i32x4), _mm_add_epi32(lane2_i32x4, lane3_i32x4));
}

NK_PUBLIC void nk_dot_u8_sierra(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                nk_u32_t *result) {
    // Native u8*u8 dot product using DPBUUD (unsigned * unsigned -> u32)
    // No algebraic transformation needed - dpbuud handles unsigned*unsigned directly.
    __m256i sum_u32x8 = _mm256_setzero_si256();
    __m256i a_u8x32, b_u8x32;

nk_dot_u8_sierra_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_u8x32 = _mm256_load_si256(&a_vec.ymm);
        b_u8x32 = _mm256_load_si256(&b_vec.ymm);
        count_scalars = 0;
    }
    else {
        a_u8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_u8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // VPDPBUUD: unsigned u8 * unsigned u8 -> u32 accumulation
    sum_u32x8 = _mm256_dpbuud_epi32(sum_u32x8, a_u8x32, b_u8x32);

    if (count_scalars) goto nk_dot_u8_sierra_cycle;

    // Reduce u32x8 to scalar - reinterpret as i32 for reduction, cast back
    *result = (nk_u32_t)(nk_i32_t)nk_reduce_add_i32x8_haswell_(sum_u32x8);
}

typedef struct nk_dot_u8x32_state_sierra_t {
    __m256i sum_u32x8; // DPBUUD accumulator: u8 * u8 -> u32
} nk_dot_u8x32_state_sierra_t;

NK_INTERNAL void nk_dot_u8x32_init_sierra(nk_dot_u8x32_state_sierra_t *state) {
    state->sum_u32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_u8x32_update_sierra(nk_dot_u8x32_state_sierra_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->sum_u32x8 = _mm256_dpbuud_epi32(state->sum_u32x8, a.ymm, b.ymm);
}

NK_INTERNAL void nk_dot_u8x32_finalize_sierra(                                              //
    nk_dot_u8x32_state_sierra_t const *state_a, nk_dot_u8x32_state_sierra_t const *state_b, //
    nk_dot_u8x32_state_sierra_t const *state_c, nk_dot_u8x32_state_sierra_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    // Same transpose+reduce pattern but simpler - no correction term
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->sum_u32x8),
                                        _mm256_extracti128_si256(state_a->sum_u32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->sum_u32x8),
                                        _mm256_extracti128_si256(state_b->sum_u32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->sum_u32x8),
                                        _mm256_extracti128_si256(state_c->sum_u32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->sum_u32x8),
                                        _mm256_extracti128_si256(state_d->sum_u32x8, 1));

    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    result->xmm = _mm_add_epi32(_mm_add_epi32(lane0_i32x4, lane1_i32x4), _mm_add_epi32(lane2_i32x4, lane3_i32x4));
}

NK_PUBLIC void nk_dot_e2m3_sierra(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    // Integer dot product for e2m3 using dual-VPSHUFB (LUT) + VPDPBSSD (signed*signed).
    // Every e2m3 value * 16 is an exact integer in [-120, +120].
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // Uses dpbssd instead of dpbusd — both operands are already signed i8 after
    // LUT + sign application, so no unsigned conversion is needed.
    //
    __m256i const lut_lower_u8x32 = _mm256_set_epi8(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28,
                                                    26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i const lut_upper_u8x32 = _mm256_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                    120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i a_e2m3_u8x32, b_e2m3_u8x32;

nk_dot_e2m3_sierra_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_e2m3_u8x32 = a_vec.ymm;
        b_e2m3_u8x32 = b_vec.ymm;
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_e2m3_u8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Decode a: extract magnitude, dual-VPSHUFB LUT, apply sign
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_index_u8x32 = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);
    __m256i a_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, a_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, a_shuffle_index_u8x32),
                                                  a_upper_select_u8x32);
    __m256i a_negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(a_e2m3_u8x32, sign_mask_u8x32), sign_mask_u8x32);
    __m256i a_signed_i8x32 = _mm256_blendv_epi8(
        a_unsigned_u8x32, _mm256_sub_epi8(_mm256_setzero_si256(), a_unsigned_u8x32), a_negate_mask_u8x32);

    // Decode b: same LUT decode + sign
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i b_shuffle_index_u8x32 = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);
    __m256i b_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, b_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, b_shuffle_index_u8x32),
                                                  b_upper_select_u8x32);
    __m256i b_negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(b_e2m3_u8x32, sign_mask_u8x32), sign_mask_u8x32);
    __m256i b_signed_i8x32 = _mm256_blendv_epi8(
        b_unsigned_u8x32, _mm256_sub_epi8(_mm256_setzero_si256(), b_unsigned_u8x32), b_negate_mask_u8x32);

    // VPDPBSSD: signed i8 * signed i8 -> i32
    sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, a_signed_i8x32, b_signed_i8x32);

    if (count_scalars) goto nk_dot_e2m3_sierra_cycle;
    *result = (nk_f32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8) / 256.0f;
}

typedef struct nk_dot_e2m3x32_state_sierra_t {
    __m256i sum_i32x8; // DPBSSD accumulator: i8_signed * i8_signed -> i32
} nk_dot_e2m3x32_state_sierra_t;

NK_INTERNAL void nk_dot_e2m3x32_init_sierra(nk_dot_e2m3x32_state_sierra_t *state) {
    state->sum_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_e2m3x32_update_sierra(nk_dot_e2m3x32_state_sierra_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Same LUT constants...
    __m256i const lut_lower_u8x32 = _mm256_set_epi8(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28,
                                                    26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i const lut_upper_u8x32 = _mm256_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                    120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);

    __m256i a_e2m3_u8x32 = a.ymm;
    __m256i b_e2m3_u8x32 = b.ymm;

    // Decode a
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_index_u8x32 = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);
    __m256i a_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, a_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, a_shuffle_index_u8x32),
                                                  a_upper_select_u8x32);
    __m256i a_negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(a_e2m3_u8x32, sign_mask_u8x32), sign_mask_u8x32);
    __m256i a_signed_i8x32 = _mm256_blendv_epi8(
        a_unsigned_u8x32, _mm256_sub_epi8(_mm256_setzero_si256(), a_unsigned_u8x32), a_negate_mask_u8x32);

    // Decode b
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i b_shuffle_index_u8x32 = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);
    __m256i b_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, b_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, b_shuffle_index_u8x32),
                                                  b_upper_select_u8x32);
    __m256i b_negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(b_e2m3_u8x32, sign_mask_u8x32), sign_mask_u8x32);
    __m256i b_signed_i8x32 = _mm256_blendv_epi8(
        b_unsigned_u8x32, _mm256_sub_epi8(_mm256_setzero_si256(), b_unsigned_u8x32), b_negate_mask_u8x32);

    // VPDPBSSD: signed * signed -> i32
    state->sum_i32x8 = _mm256_dpbssd_epi32(state->sum_i32x8, a_signed_i8x32, b_signed_i8x32);
}

NK_INTERNAL void nk_dot_e2m3x32_finalize_sierra(                                                //
    nk_dot_e2m3x32_state_sierra_t const *state_a, nk_dot_e2m3x32_state_sierra_t const *state_b, //
    nk_dot_e2m3x32_state_sierra_t const *state_c, nk_dot_e2m3x32_state_sierra_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *results) {
    nk_unused_(total_dimensions);

    // ILP-optimized 4-way horizontal reduction: i32x8 -> scalar i32, then -> f32 with /256
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->sum_i32x8),
                                        _mm256_extracti128_si256(state_a->sum_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->sum_i32x8),
                                        _mm256_extracti128_si256(state_b->sum_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->sum_i32x8),
                                        _mm256_extracti128_si256(state_c->sum_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->sum_i32x8),
                                        _mm256_extracti128_si256(state_d->sum_i32x8, 1));

    // Transpose for SIMD reduction
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_i32x4 = _mm_add_epi32(_mm_add_epi32(lane0_i32x4, lane1_i32x4), _mm_add_epi32(lane2_i32x4, lane3_i32x4));

    // Convert i32 -> f32 and scale by 1/256
    __m128 sum_f32x4 = _mm_mul_ps(_mm_cvtepi32_ps(sum_i32x4), _mm_set1_ps(1.0f / 256.0f));
    results->xmm = _mm_castps_si128(sum_f32x4);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_
#endif // NK_DOT_SIERRA_H
