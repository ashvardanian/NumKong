/**
 *  @brief SIMD-accelerated Dot Products for Skylake.
 *  @file include/numkong/dot/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_skylake_instructions Key AVX-512 Instructions
 *
 *      Intrinsic             Instruction                  Skylake-X  Genoa
 *      _mm512_madd_epi16     VPMADDWD (ZMM, ZMM, ZMM)     5cy @ p05  3cy @ p01
 *      _mm512_add_epi32      VPADDD (ZMM, ZMM, ZMM)       1cy @ p05  1cy @ p0123
 *      _mm512_fmadd_ps       VFMADD132PS (ZMM, ZMM, ZMM)  4cy @ p05  4cy @ p01
 *      _mm512_cvtepi8_epi16  VPMOVSXBW (ZMM, YMM)         3cy @ p5   4cy @ p12
 *
 *  Skylake-X server chips feature dual 512-bit FMA units on ports 0 and 5, enabling 0.5cy throughput for
 *  VFMADD and arithmetic operations. Client Skylake variants have only one FMA unit with 1cy throughput.
 *  Without VNNI support, integer dot products use VPMADDWD for i16 pair multiplication with i32 accumulation.
 *
 *  @section dot_skylake_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_f64x8 state with Dot2 stable dot-products,
 *  - nk_dot_f32x8 state with double-precision numerics,
 *  - nk_dot_through_f32 state for 16-bit float inputs with single-precision numerics.
 *
 *  @code{c}
 *  nk_dot_f64x8_state_skylake_t state_first, state_second, state_third, state_fourth;
 *  nk_b512_vec_t query_f64x8, target_first_f64x8, target_second_f64x8, target_third_f64x8, target_fourth_f64x8;
 *  nk_dot_f64x8_init_skylake(&state_first);
 *  nk_dot_f64x8_init_skylake(&state_second);
 *  nk_dot_f64x8_init_skylake(&state_third);
 *  nk_dot_f64x8_init_skylake(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 8 <= depth; idx += 8) {
 *      query_f64x8.zmm_pd = _mm512_loadu_pd(query_ptr + idx);
 *      target_first_f64x8.zmm_pd = _mm512_loadu_pd(target_first_ptr + idx);
 *      target_second_f64x8.zmm_pd = _mm512_loadu_pd(target_second_ptr + idx);
 *      target_third_f64x8.zmm_pd = _mm512_loadu_pd(target_third_ptr + idx);
 *      target_fourth_f64x8.zmm_pd = _mm512_loadu_pd(target_fourth_ptr + idx);
 *      nk_dot_f64x8_update_skylake(&state_first, query_f64x8, target_first_f64x8, idx, 8);
 *      nk_dot_f64x8_update_skylake(&state_second, query_f64x8, target_second_f64x8, idx, 8);
 *      nk_dot_f64x8_update_skylake(&state_third, query_f64x8, target_third_f64x8, idx, 8);
 *      nk_dot_f64x8_update_skylake(&state_fourth, query_f64x8, target_fourth_f64x8, idx, 8);
 *  }
 *  nk_b256_vec_t results_f64x4;
 *  nk_dot_f64x8_finalize_skylake(&state_first, &state_second, &state_third, &state_fourth, depth, &results_f64x4);
 *  @endcode
 *
 *  Smaller float types like f16 and bf16 on Skylake use ISA-specific upcasting to f32 combined with native
 *  FMA instructions, sharing the `nk_dot_through_f32` accumulation logic:
 *
 *  @code{c}
 *  nk_dot_f16x16_state_skylake_t state_first, state_second, state_third, state_fourth;
 *  nk_b512_vec_t query_f32x16, target_first_f32x16, target_second_f32x16, target_third_f32x16, target_fourth_f32x16;
 *  nk_dot_through_f32_init_skylake_(&state_first);
 *  nk_dot_through_f32_init_skylake_(&state_second);
 *  nk_dot_through_f32_init_skylake_(&state_third);
 *  nk_dot_through_f32_init_skylake_(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 16 <= depth; idx += 16) {
 *      nk_load_f16x16_to_f32x16_skylake_(query_ptr + idx, &query_f32x16);
 *      nk_load_f16x16_to_f32x16_skylake_(target_first_ptr + idx, &target_first_f32x16);
 *      nk_load_f16x16_to_f32x16_skylake_(target_second_ptr + idx, &target_second_f32x16);
 *      nk_load_f16x16_to_f32x16_skylake_(target_third_ptr + idx, &target_third_f32x16);
 *      nk_load_f16x16_to_f32x16_skylake_(target_fourth_ptr + idx, &target_fourth_f32x16);
 *      nk_dot_through_f32_update_skylake_(&state_first, query_f32x16, target_first_f32x16, idx, 16);
 *      nk_dot_through_f32_update_skylake_(&state_second, query_f32x16, target_second_f32x16, idx, 16);
 *      nk_dot_through_f32_update_skylake_(&state_third, query_f32x16, target_third_f32x16, idx, 16);
 *      nk_dot_through_f32_update_skylake_(&state_fourth, query_f32x16, target_fourth_f32x16, idx, 16);
 *  }
 *  nk_b128_vec_t results_f32x4;
 *  nk_dot_through_f32_finalize_skylake_(&state_first, &state_second, &state_third, &state_fourth,
 *      depth, &results_f32x4);
 *  @endcode
 */
#ifndef NK_DOT_SKYLAKE_H
#define NK_DOT_SKYLAKE_H

#if NK_TARGET_X8664_
#if NK_TARGET_SKYLAKE

#include "numkong/cast/skylake.h"   // `nk_bf16x16_to_f32x16_skylake_`
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`
#include "numkong/dot/haswell.h"    // `nk_dot_stable_sum_f64x4_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

/** @brief Compensated horizontal sum of 8 f64 lanes via TwoSum tree reduction. */
NK_INTERNAL nk_f64_t nk_dot_stable_sum_f64x8_skylake_(__m512d sum_f64x8, __m512d compensation_f64x8) {
    // Stage 0: TwoSum merge of sum + compensation (8-wide)
    __m512d tentative_sum_f64x8 = _mm512_add_pd(sum_f64x8, compensation_f64x8);
    __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, sum_f64x8);
    __m512d rounding_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
        _mm512_sub_pd(compensation_f64x8, virtual_addend_f64x8));

    // Stage 1: TwoSum halving 8→4
    __m256d lower_sum_f64x4 = _mm512_castpd512_pd256(tentative_sum_f64x8);
    __m256d upper_sum_f64x4 = _mm512_extractf64x4_pd(tentative_sum_f64x8, 1);
    __m256d tentative_sum_f64x4 = _mm256_add_pd(lower_sum_f64x4, upper_sum_f64x4);
    __m256d virtual_addend_f64x4 = _mm256_sub_pd(tentative_sum_f64x4, lower_sum_f64x4);
    __m256d rounding_error_f64x4 = _mm256_add_pd(
        _mm256_sub_pd(lower_sum_f64x4, _mm256_sub_pd(tentative_sum_f64x4, virtual_addend_f64x4)),
        _mm256_sub_pd(upper_sum_f64x4, virtual_addend_f64x4));
    __m256d lower_error_f64x4 = _mm512_castpd512_pd256(rounding_error_f64x8);
    __m256d upper_error_f64x4 = _mm512_extractf64x4_pd(rounding_error_f64x8, 1);
    __m256d accumulated_error_f64x4 = _mm256_add_pd(_mm256_add_pd(lower_error_f64x4, upper_error_f64x4),
                                                    rounding_error_f64x4);

    // Stages 2-3: Delegate to Haswell for 4→2→1 reduction
    return nk_dot_stable_sum_f64x4_haswell_(tentative_sum_f64x4, accumulated_error_f64x4);
}

#pragma region F32 and F64 Floats

/**
 *  @brief Internal helper state for dot-products of low-precision types, where 32-bit accumulation is enough.
 *  @sa nk_dot_f16x16_state_skylake_t, nk_dot_bf16x16_state_skylake_t
 *  @sa nk_dot_e4m3x16_state_skylake_t, nk_dot_e5m2x16_state_skylake_t
 */
typedef struct nk_dot_through_f32_state_skylake_t_ {
    __m512 sum_f32x16;
} nk_dot_through_f32_state_skylake_t_;

/**
 *  @brief Initializes 32-bit accumulators for low-precision dot-products.
 *  @sa nk_dot_f16x16_init_skylake, nk_dot_bf16x16_init_skylake
 *  @sa nk_dot_e4m3x16_init_skylake, nk_dot_e5m2x16_init_skylake
 */
NK_INTERNAL void nk_dot_through_f32_init_skylake_(nk_dot_through_f32_state_skylake_t_ *state) {
    state->sum_f32x16 = _mm512_setzero_ps();
}

/**
 *  @brief Fuses 32-bit multiplication and accumulation for low-precision dot-products.
 *  @sa nk_dot_f16x16_udpate_skylake, nk_dot_bf16x16_udpate_skylake
 *  @sa nk_dot_e4m3x16_udpate_skylake, nk_dot_e5m2x16_udpate_skylake
 */
NK_INTERNAL void nk_dot_through_f32_update_skylake_(nk_dot_through_f32_state_skylake_t_ *state, nk_b512_vec_t a,
                                                    nk_b512_vec_t b, nk_size_t depth_offset,
                                                    nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->sum_f32x16 = _mm512_fmadd_ps(a.zmm_ps, b.zmm_ps, state->sum_f32x16);
}

/**
 *  @brief E5M2 byte-batched update: consumes 64 raw E5M2 bytes per call and widens inline.
 *  Two independent FMA chains (each 2-deep) merge into the single state accumulator at exit.
 *  Keeps register pressure at one __m512 across calls while breaking the FMA dep chain.
 */
NK_INTERNAL void nk_dot_e5m2x64_update_skylake_(nk_dot_through_f32_state_skylake_t_ *state, nk_b512_vec_t a_bytes,
                                                nk_b512_vec_t b_bytes, nk_size_t depth_offset,
                                                nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m512i const zero_u8x64 = _mm512_setzero_si512();
    __m512i a_even_f16x32 = _mm512_unpacklo_epi8(zero_u8x64, a_bytes.zmm);
    __m512i a_odd_f16x32 = _mm512_unpackhi_epi8(zero_u8x64, a_bytes.zmm);
    __m512i b_even_f16x32 = _mm512_unpacklo_epi8(zero_u8x64, b_bytes.zmm);
    __m512i b_odd_f16x32 = _mm512_unpackhi_epi8(zero_u8x64, b_bytes.zmm);
    __m512 a_first_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(a_even_f16x32));
    __m512 a_second_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a_even_f16x32, 1));
    __m512 a_third_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(a_odd_f16x32));
    __m512 a_fourth_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a_odd_f16x32, 1));
    __m512 b_first_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(b_even_f16x32));
    __m512 b_second_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b_even_f16x32, 1));
    __m512 b_third_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(b_odd_f16x32));
    __m512 b_fourth_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b_odd_f16x32, 1));
    __m512 first_chain_f32x16 = _mm512_mul_ps(a_first_f32x16, b_first_f32x16);
    __m512 second_chain_f32x16 = _mm512_mul_ps(a_second_f32x16, b_second_f32x16);
    first_chain_f32x16 = _mm512_fmadd_ps(a_third_f32x16, b_third_f32x16, first_chain_f32x16);
    second_chain_f32x16 = _mm512_fmadd_ps(a_fourth_f32x16, b_fourth_f32x16, second_chain_f32x16);
    state->sum_f32x16 = _mm512_add_ps(state->sum_f32x16, _mm512_add_ps(first_chain_f32x16, second_chain_f32x16));
}

/**
 *  @brief Finalizes 4x low-precision dot-products placing them into 4x consecutive 32-bit slots.
 *  @sa nk_dot_f16x16_udpate_skylake, nk_dot_bf16x16_udpate_skylake
 *  @sa nk_dot_e4m3x16_udpate_skylake, nk_dot_e5m2x16_udpate_skylake
 *
 *  The goal of this kernel is simple - compute 4x horizontal reductions, each involing 16x floats.
 *  The lack of vectorized horizontal instruction implies many consecutive shuffles producing a tree-like
 *  reduction. This kernel allow combinding some of those operations between different dot products.
 */
NK_INTERNAL void nk_dot_through_f32_finalize_skylake_(                                                      //
    nk_dot_through_f32_state_skylake_t_ const *state_a, nk_dot_through_f32_state_skylake_t_ const *state_b, //
    nk_dot_through_f32_state_skylake_t_ const *state_c, nk_dot_through_f32_state_skylake_t_ const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    __m512 const sum_a_f32x16 = state_a->sum_f32x16, sum_b_f32x16 = state_b->sum_f32x16,
                 sum_c_f32x16 = state_c->sum_f32x16, sum_d_f32x16 = state_d->sum_f32x16;

    // ILP-optimized 4-way horizontal reduction for f32x16 in AVX-512
    // Step 1: 16 → 8 for all 4 states (extract high 256-bit half and add to low half)
    __m256 sum_a_f32x8 = _mm256_add_ps(_mm512_castps512_ps256(sum_a_f32x16), _mm512_extractf32x8_ps(sum_a_f32x16, 1));
    __m256 sum_b_f32x8 = _mm256_add_ps(_mm512_castps512_ps256(sum_b_f32x16), _mm512_extractf32x8_ps(sum_b_f32x16, 1));
    __m256 sum_c_f32x8 = _mm256_add_ps(_mm512_castps512_ps256(sum_c_f32x16), _mm512_extractf32x8_ps(sum_c_f32x16, 1));
    __m256 sum_d_f32x8 = _mm256_add_ps(_mm512_castps512_ps256(sum_d_f32x16), _mm512_extractf32x8_ps(sum_d_f32x16, 1));
    // Step 2: 8 → 4 for all 4 states (extract high 128-bit half and add to low half)
    __m128 sum_a_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_a_f32x8), _mm256_extractf128_ps(sum_a_f32x8, 1));
    __m128 sum_b_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_b_f32x8), _mm256_extractf128_ps(sum_b_f32x8, 1));
    __m128 sum_c_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_c_f32x8), _mm256_extractf128_ps(sum_c_f32x8, 1));
    __m128 sum_d_f32x4 = _mm_add_ps(_mm256_castps256_ps128(sum_d_f32x8), _mm256_extractf128_ps(sum_d_f32x8, 1));
    // Step 3: Transpose 4x4 and reduce to get final 4 scalars
    __m128 transpose_ab_low_f32x4 = _mm_unpacklo_ps(sum_a_f32x4, sum_b_f32x4);
    __m128 transpose_cd_low_f32x4 = _mm_unpacklo_ps(sum_c_f32x4, sum_d_f32x4);
    __m128 transpose_ab_high_f32x4 = _mm_unpackhi_ps(sum_a_f32x4, sum_b_f32x4);
    __m128 transpose_cd_high_f32x4 = _mm_unpackhi_ps(sum_c_f32x4, sum_d_f32x4);
    __m128 sum_lane0_f32x4 = _mm_movelh_ps(transpose_ab_low_f32x4, transpose_cd_low_f32x4);
    __m128 sum_lane1_f32x4 = _mm_movehl_ps(transpose_cd_low_f32x4, transpose_ab_low_f32x4);
    __m128 sum_lane2_f32x4 = _mm_movelh_ps(transpose_ab_high_f32x4, transpose_cd_high_f32x4);
    __m128 sum_lane3_f32x4 = _mm_movehl_ps(transpose_cd_high_f32x4, transpose_ab_high_f32x4);
    __m128 final_sum_f32x4 = _mm_add_ps(_mm_add_ps(sum_lane0_f32x4, sum_lane1_f32x4),
                                        _mm_add_ps(sum_lane2_f32x4, sum_lane3_f32x4));
    result->xmm = _mm_castps_si128(final_sum_f32x4);
}

NK_PUBLIC void nk_dot_f32_skylake(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m512d sum_f64x8 = _mm512_setzero_pd();

nk_dot_f32_skylake_cycle:
    if (count_scalars < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, a_scalars);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps(a_scalars);
        b_f32x8 = _mm256_loadu_ps(b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f64x8 = _mm512_fmadd_pd(_mm512_cvtps_pd(a_f32x8), _mm512_cvtps_pd(b_f32x8), sum_f64x8);
    if (count_scalars) goto nk_dot_f32_skylake_cycle;

    *result = _mm512_reduce_add_pd(sum_f64x8);
}

NK_PUBLIC void nk_dot_f64_skylake(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product
    __m512d a_f64x8, b_f64x8;
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d compensation_f64x8 = _mm512_setzero_pd();

nk_dot_f64_skylake_cycle:
    if (count_scalars < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a_scalars);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a_scalars);
        b_f64x8 = _mm512_loadu_pd(b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
    __m512d product_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, product_f64x8);
    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    __m512d tentative_sum_f64x8 = _mm512_add_pd(sum_f64x8, product_f64x8);
    __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, sum_f64x8);
    __m512d sum_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
        _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));
    // Update: sum = t, compensation += q + r
    sum_f64x8 = tentative_sum_f64x8;
    compensation_f64x8 = _mm512_add_pd(compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
    if (count_scalars) goto nk_dot_f64_skylake_cycle;

    // Compensated horizontal reduction preserving Dot2 error tracking
    *result = nk_dot_stable_sum_f64x8_skylake_(sum_f64x8, compensation_f64x8);
}

NK_PUBLIC void nk_dot_f32c_skylake(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f64c_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(0x8000000000000000, 0, 0x8000000000000000, 0, 0x8000000000000000,
                                                     0, 0x8000000000000000, 0);
nk_dot_f32c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, (nk_f32_t const *)a_pairs);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, (nk_f32_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps((nk_f32_t const *)a_pairs);
        b_f32x8 = _mm256_loadu_ps((nk_f32_t const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    __m512d a_f64x8 = _mm512_cvtps_pd(a_f32x8);
    __m512d b_f64x8 = _mm512_cvtps_pd(b_f32x8);
    __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0x55);
    sum_real_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_real_f64x8);
    sum_imag_f64x8 = _mm512_fmadd_pd(a_f64x8, b_swapped_f64x8, sum_imag_f64x8);
    if (count_pairs) goto nk_dot_f32c_skylake_cycle;

    // Flip the sign bit in every second f64 before accumulation:
    sum_real_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_real_f64x8), sign_flip_f64x8));

    // Reduce horizontal sums:
    result->real = _mm512_reduce_add_pd(sum_real_f64x8);
    result->imag = _mm512_reduce_add_pd(sum_imag_f64x8);
}

NK_PUBLIC void nk_vdot_f32c_skylake(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f64c_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(0x8000000000000000, 0, 0x8000000000000000, 0, 0x8000000000000000,
                                                     0, 0x8000000000000000, 0);
nk_vdot_f32c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, (nk_f32_t const *)a_pairs);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, (nk_f32_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps((nk_f32_t const *)a_pairs);
        b_f32x8 = _mm256_loadu_ps((nk_f32_t const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    __m512d a_f64x8 = _mm512_cvtps_pd(a_f32x8);
    __m512d b_f64x8 = _mm512_cvtps_pd(b_f32x8);
    sum_real_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_real_f64x8);
    __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0x55);
    sum_imag_f64x8 = _mm512_fmadd_pd(a_f64x8, b_swapped_f64x8, sum_imag_f64x8);
    if (count_pairs) goto nk_vdot_f32c_skylake_cycle;

    // Flip the sign bit in every second f64 before accumulation:
    sum_imag_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_imag_f64x8), sign_flip_f64x8));

    // Reduce horizontal sums:
    result->real = _mm512_reduce_add_pd(sum_real_f64x8);
    result->imag = _mm512_reduce_add_pd(sum_imag_f64x8);
}

NK_PUBLIC void nk_dot_f64c_skylake(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated complex dot product
    __m512d a_f64x8, b_f64x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();
    __m512d compensation_real_f64x8 = _mm512_setzero_pd();
    __m512d compensation_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
nk_dot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a_pairs);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a_pairs);
        b_f64x8 = _mm512_loadu_pd(b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0x55); //? Same as 0b01010101.

    // TwoProd for real part: a * b
    __m512d product_real_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_real_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, product_real_f64x8);
    // TwoSum for real part
    __m512d tentative_sum_real_f64x8 = _mm512_add_pd(sum_real_f64x8, product_real_f64x8);
    __m512d virtual_addend_real_f64x8 = _mm512_sub_pd(tentative_sum_real_f64x8, sum_real_f64x8);
    __m512d sum_real_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_real_f64x8, _mm512_sub_pd(tentative_sum_real_f64x8, virtual_addend_real_f64x8)),
        _mm512_sub_pd(product_real_f64x8, virtual_addend_real_f64x8));
    sum_real_f64x8 = tentative_sum_real_f64x8;
    compensation_real_f64x8 = _mm512_add_pd(compensation_real_f64x8,
                                            _mm512_add_pd(sum_real_error_f64x8, product_real_error_f64x8));

    // TwoProd for imag part: a * b_swapped
    __m512d product_imag_f64x8 = _mm512_mul_pd(a_f64x8, b_swapped_f64x8);
    __m512d product_imag_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_swapped_f64x8, product_imag_f64x8);
    // TwoSum for imag part
    __m512d tentative_sum_imag_f64x8 = _mm512_add_pd(sum_imag_f64x8, product_imag_f64x8);
    __m512d virtual_addend_imag_f64x8 = _mm512_sub_pd(tentative_sum_imag_f64x8, sum_imag_f64x8);
    __m512d sum_imag_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_imag_f64x8, _mm512_sub_pd(tentative_sum_imag_f64x8, virtual_addend_imag_f64x8)),
        _mm512_sub_pd(product_imag_f64x8, virtual_addend_imag_f64x8));
    sum_imag_f64x8 = tentative_sum_imag_f64x8;
    compensation_imag_f64x8 = _mm512_add_pd(compensation_imag_f64x8,
                                            _mm512_add_pd(sum_imag_error_f64x8, product_imag_error_f64x8));

    if (count_pairs) goto nk_dot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation (to get a_r*b_r - a_i*b_i):
    sum_real_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_real_f64x8), sign_flip_f64x8));
    compensation_real_f64x8 = _mm512_castsi512_pd(
        _mm512_xor_si512(_mm512_castpd_si512(compensation_real_f64x8), sign_flip_f64x8));

    // Compensated horizontal reduction preserving Dot2 error tracking
    result->real = nk_dot_stable_sum_f64x8_skylake_(sum_real_f64x8, compensation_real_f64x8);
    result->imag = nk_dot_stable_sum_f64x8_skylake_(sum_imag_f64x8, compensation_imag_f64x8);
}

NK_PUBLIC void nk_vdot_f64c_skylake(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated conjugate dot product
    __m512d a_f64x8, b_f64x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();
    __m512d compensation_real_f64x8 = _mm512_setzero_pd();
    __m512d compensation_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
nk_vdot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, (nk_f64_t const *)a_pairs);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, (nk_f64_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd((nk_f64_t const *)a_pairs);
        b_f64x8 = _mm512_loadu_pd((nk_f64_t const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0x55); //? Same as 0b01010101.

    // TwoProd for real part: a * b
    __m512d product_real_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_real_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, product_real_f64x8);
    // TwoSum for real part
    __m512d tentative_sum_real_f64x8 = _mm512_add_pd(sum_real_f64x8, product_real_f64x8);
    __m512d virtual_addend_real_f64x8 = _mm512_sub_pd(tentative_sum_real_f64x8, sum_real_f64x8);
    __m512d sum_real_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_real_f64x8, _mm512_sub_pd(tentative_sum_real_f64x8, virtual_addend_real_f64x8)),
        _mm512_sub_pd(product_real_f64x8, virtual_addend_real_f64x8));
    sum_real_f64x8 = tentative_sum_real_f64x8;
    compensation_real_f64x8 = _mm512_add_pd(compensation_real_f64x8,
                                            _mm512_add_pd(sum_real_error_f64x8, product_real_error_f64x8));

    // TwoProd for imag part: a * b_swapped
    __m512d product_imag_f64x8 = _mm512_mul_pd(a_f64x8, b_swapped_f64x8);
    __m512d product_imag_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_swapped_f64x8, product_imag_f64x8);
    // TwoSum for imag part
    __m512d tentative_sum_imag_f64x8 = _mm512_add_pd(sum_imag_f64x8, product_imag_f64x8);
    __m512d virtual_addend_imag_f64x8 = _mm512_sub_pd(tentative_sum_imag_f64x8, sum_imag_f64x8);
    __m512d sum_imag_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_imag_f64x8, _mm512_sub_pd(tentative_sum_imag_f64x8, virtual_addend_imag_f64x8)),
        _mm512_sub_pd(product_imag_f64x8, virtual_addend_imag_f64x8));
    sum_imag_f64x8 = tentative_sum_imag_f64x8;
    compensation_imag_f64x8 = _mm512_add_pd(compensation_imag_f64x8,
                                            _mm512_add_pd(sum_imag_error_f64x8, product_imag_error_f64x8));

    if (count_pairs) goto nk_vdot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation (to get a_r*b_i - a_i*b_r):
    sum_imag_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_imag_f64x8), sign_flip_f64x8));
    compensation_imag_f64x8 = _mm512_castsi512_pd(
        _mm512_xor_si512(_mm512_castpd_si512(compensation_imag_f64x8), sign_flip_f64x8));

    // Compensated horizontal reduction preserving Dot2 error tracking
    result->real = nk_dot_stable_sum_f64x8_skylake_(sum_real_f64x8, compensation_real_f64x8);
    result->imag = nk_dot_stable_sum_f64x8_skylake_(sum_imag_f64x8, compensation_imag_f64x8);
}

#pragma endregion F32 and F64 Floats
#pragma region F16 and BF16 Floats

NK_PUBLIC void nk_dot_f16_skylake(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    __m256i a_f16x16, b_f16x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_f16_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_f16x16 = _mm256_maskz_loadu_epi16(mask, a_scalars);
        b_f16x16 = _mm256_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f16x16 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_f16x16 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
    __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_f16_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_bf16_skylake(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m512i a_bf16_i16x32, b_bf16_i16x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512i mask_high_u32x16 = _mm512_set1_epi32((int)0xFFFF0000);

nk_dot_bf16_skylake_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_bf16_i16x32 = _mm512_maskz_loadu_epi16(mask, a_scalars);
        b_bf16_i16x32 = _mm512_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_bf16_i16x32 = _mm512_loadu_si512(a_scalars);
        b_bf16_i16x32 = _mm512_loadu_si512(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    __m512 a_even_f32x16 = _mm512_castsi512_ps(_mm512_slli_epi32(a_bf16_i16x32, 16));
    __m512 b_even_f32x16 = _mm512_castsi512_ps(_mm512_slli_epi32(b_bf16_i16x32, 16));
    sum_f32x16 = _mm512_fmadd_ps(a_even_f32x16, b_even_f32x16, sum_f32x16);
    __m512 a_odd_f32x16 = _mm512_castsi512_ps(_mm512_and_si512(a_bf16_i16x32, mask_high_u32x16));
    __m512 b_odd_f32x16 = _mm512_castsi512_ps(_mm512_and_si512(b_bf16_i16x32, mask_high_u32x16));
    sum_f32x16 = _mm512_fmadd_ps(a_odd_f32x16, b_odd_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_bf16_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_e4m3_skylake(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m128i a_e4m3_u8x16, b_e4m3_u8x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_e4m3_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)count_scalars);
        a_e4m3_u8x16 = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3_u8x16 = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3_u8x16 = _mm_loadu_si128((__m128i const *)a_scalars);
        b_e4m3_u8x16 = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3_u8x16);
    __m512 b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3_u8x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_e4m3_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_e5m2_skylake(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    // E5M2 shares F16 bias (15): vpunpck*bw against zero places the byte as F16 encoding,
    // so we inline the widen rather than calling the helper 4× — same ops, cleaner code.
    __m512 first_chain_f32x16 = _mm512_setzero_ps();
    __m512 second_chain_f32x16 = _mm512_setzero_ps();
    __m512i const zero_u8x64 = _mm512_setzero_si512();
    __m512i a_u8x64, b_u8x64;

nk_dot_e5m2_skylake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count_scalars);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512((__m512i const *)a_scalars);
        b_u8x64 = _mm512_loadu_si512((__m512i const *)b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }
    __m512i a_even_f16x32 = _mm512_unpacklo_epi8(zero_u8x64, a_u8x64);
    __m512i a_odd_f16x32 = _mm512_unpackhi_epi8(zero_u8x64, a_u8x64);
    __m512i b_even_f16x32 = _mm512_unpacklo_epi8(zero_u8x64, b_u8x64);
    __m512i b_odd_f16x32 = _mm512_unpackhi_epi8(zero_u8x64, b_u8x64);
    __m512 a_first_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(a_even_f16x32));
    __m512 a_second_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a_even_f16x32, 1));
    __m512 a_third_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(a_odd_f16x32));
    __m512 a_fourth_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a_odd_f16x32, 1));
    __m512 b_first_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(b_even_f16x32));
    __m512 b_second_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b_even_f16x32, 1));
    __m512 b_third_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(b_odd_f16x32));
    __m512 b_fourth_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b_odd_f16x32, 1));
    first_chain_f32x16 = _mm512_fmadd_ps(a_first_f32x16, b_first_f32x16, first_chain_f32x16);
    second_chain_f32x16 = _mm512_fmadd_ps(a_second_f32x16, b_second_f32x16, second_chain_f32x16);
    first_chain_f32x16 = _mm512_fmadd_ps(a_third_f32x16, b_third_f32x16, first_chain_f32x16);
    second_chain_f32x16 = _mm512_fmadd_ps(a_fourth_f32x16, b_fourth_f32x16, second_chain_f32x16);
    if (count_scalars) goto nk_dot_e5m2_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(_mm512_add_ps(first_chain_f32x16, second_chain_f32x16));
}

NK_PUBLIC void nk_dot_e2m3_skylake(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    // Integer dot product for e2m3 using dual-VPSHUFB (LUT) + VPMADDUBSW (unsigned×signed).
    // 64 elements per iteration using AVX-512BW. Result = i32_dot / 256.0f (exact).
    //
    // LUTs replicated 4× for 512-bit VPSHUFB (operates per 128-bit lane):
    __m512i const lut_low_u8x64 = _mm512_set_epi8(                 //
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, //
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, //
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, //
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i const lut_high_u8x64 = _mm512_set_epi8(                        //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32, //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32, //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32, //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const magnitude_mask_u8x64 = _mm512_set1_epi8(0x1F);
    __m512i const half_select_u8x64 = _mm512_set1_epi8(0x10);
    __m512i const sign_mask_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i a_e2m3_u8x64, b_e2m3_u8x64;

nk_dot_e2m3_skylake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_e2m3_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_e2m3_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x64 = _mm512_loadu_si512((__m512i const *)a_scalars);
        b_e2m3_u8x64 = _mm512_loadu_si512((__m512i const *)b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Extract 5-bit magnitude, then split into low 4 bits (VPSHUFB index) and bit 4 (hi/lo select)
    __m512i a_magnitude_u8x64 = _mm512_and_si512(a_e2m3_u8x64, magnitude_mask_u8x64);
    __m512i b_magnitude_u8x64 = _mm512_and_si512(b_e2m3_u8x64, magnitude_mask_u8x64);
    __m512i a_shuffle_index_u8x64 = _mm512_and_si512(a_magnitude_u8x64, nibble_mask_u8x64);
    __m512i b_shuffle_index_u8x64 = _mm512_and_si512(b_magnitude_u8x64, nibble_mask_u8x64);

    // Bit-4 select via kmask (cleaner than Haswell's vector compare)
    __mmask64 a_high_select = _mm512_test_epi8_mask(a_magnitude_u8x64, half_select_u8x64);
    __mmask64 b_high_select = _mm512_test_epi8_mask(b_magnitude_u8x64, half_select_u8x64);

    // Dual VPSHUFB + mask-blend for 32-entry LUT
    __m512i a_unsigned_u8x64 = _mm512_mask_blend_epi8(a_high_select,
                                                      _mm512_shuffle_epi8(lut_low_u8x64, a_shuffle_index_u8x64),
                                                      _mm512_shuffle_epi8(lut_high_u8x64, a_shuffle_index_u8x64));
    __m512i b_unsigned_u8x64 = _mm512_mask_blend_epi8(b_high_select,
                                                      _mm512_shuffle_epi8(lut_low_u8x64, b_shuffle_index_u8x64),
                                                      _mm512_shuffle_epi8(lut_high_u8x64, b_shuffle_index_u8x64));

    // Combined sign: (a ^ b) & 0x20, negate b where signs differ using kmask
    __m512i sign_combined_u8x64 = _mm512_and_si512(_mm512_xor_si512(a_e2m3_u8x64, b_e2m3_u8x64), sign_mask_u8x64);
    __mmask64 negate_mask = _mm512_test_epi8_mask(sign_combined_u8x64, sign_combined_u8x64);
    __m512i b_signed_i8x64 = _mm512_mask_sub_epi8(b_unsigned_u8x64, negate_mask, _mm512_setzero_si512(),
                                                  b_unsigned_u8x64);

    // VPMADDUBSW: a_unsigned[u8] × b_signed[i8] → i16 pairs
    __m512i products_i16x32 = _mm512_maddubs_epi16(a_unsigned_u8x64, b_signed_i8x64);
    // VPMADDWD with ones: i16 pairs → i32
    sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(products_i16x32, ones_i16x32));

    if (count_scalars) goto nk_dot_e2m3_skylake_cycle;
    *result = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16) / 256.0f;
}

NK_PUBLIC void nk_dot_e3m2_skylake(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    // Integer dot product for e3m2 using dual-VPSHUFB (low-byte LUT) + VPMADDWD (i16×i16→i32).
    // 64 elements per iteration using AVX-512BW. Magnitudes reach 448, requiring i16.
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    __m512i const lut_low_byte_first_u8x64 = _mm512_set_epi8(  //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const lut_low_byte_second_u8x64 = _mm512_set_epi8(                                                    //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32, //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32, //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32, //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32);
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const magnitude_mask_u8x64 = _mm512_set1_epi8(0x1F);
    __m512i const half_select_u8x64 = _mm512_set1_epi8(0x10);
    __m512i const sign_mask_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const ones_u8x64 = _mm512_set1_epi8(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i a_e3m2_u8x64, b_e3m2_u8x64;

nk_dot_e3m2_skylake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_e3m2_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_e3m2_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e3m2_u8x64 = _mm512_loadu_si512((__m512i const *)a_scalars);
        b_e3m2_u8x64 = _mm512_loadu_si512((__m512i const *)b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Extract 5-bit magnitude, split into low 4 bits and bit 4
    __m512i a_magnitude_u8x64 = _mm512_and_si512(a_e3m2_u8x64, magnitude_mask_u8x64);
    __m512i b_magnitude_u8x64 = _mm512_and_si512(b_e3m2_u8x64, magnitude_mask_u8x64);
    __m512i a_shuffle_index_u8x64 = _mm512_and_si512(a_magnitude_u8x64, nibble_mask_u8x64);
    __m512i b_shuffle_index_u8x64 = _mm512_and_si512(b_magnitude_u8x64, nibble_mask_u8x64);

    // Bit-4 select via kmask
    __mmask64 a_high_select = _mm512_test_epi8_mask(a_magnitude_u8x64, half_select_u8x64);
    __mmask64 b_high_select = _mm512_test_epi8_mask(b_magnitude_u8x64, half_select_u8x64);

    // Dual VPSHUFB + mask-blend for low bytes
    __m512i a_low_byte_u8x64 = _mm512_mask_blend_epi8(
        a_high_select, _mm512_shuffle_epi8(lut_low_byte_first_u8x64, a_shuffle_index_u8x64),
        _mm512_shuffle_epi8(lut_low_byte_second_u8x64, a_shuffle_index_u8x64));
    __m512i b_low_byte_u8x64 = _mm512_mask_blend_epi8(
        b_high_select, _mm512_shuffle_epi8(lut_low_byte_first_u8x64, b_shuffle_index_u8x64),
        _mm512_shuffle_epi8(lut_low_byte_second_u8x64, b_shuffle_index_u8x64));

    // High byte: 1 iff magnitude >= 28 (unsigned compare via _mm512_cmpge_epu8_mask)
    __mmask64 a_high_mask = _mm512_cmpge_epu8_mask(a_magnitude_u8x64, _mm512_set1_epi8(28));
    __mmask64 b_high_mask = _mm512_cmpge_epu8_mask(b_magnitude_u8x64, _mm512_set1_epi8(28));
    __m512i a_high_byte_u8x64 = _mm512_maskz_mov_epi8(a_high_mask, ones_u8x64);
    __m512i b_high_byte_u8x64 = _mm512_maskz_mov_epi8(b_high_mask, ones_u8x64);

    // Interleave low and high bytes into i16
    __m512i a_low_i16x32 = _mm512_unpacklo_epi8(a_low_byte_u8x64, a_high_byte_u8x64);
    __m512i a_high_i16x32 = _mm512_unpackhi_epi8(a_low_byte_u8x64, a_high_byte_u8x64);
    __m512i b_low_i16x32 = _mm512_unpacklo_epi8(b_low_byte_u8x64, b_high_byte_u8x64);
    __m512i b_high_i16x32 = _mm512_unpackhi_epi8(b_low_byte_u8x64, b_high_byte_u8x64);

    // Combined sign: (a ^ b) & 0x20, need to apply at i16 level
    // Compute sign mask at u8 level, widen to match unpacklo/unpackhi ordering via PEXT
    __m512i sign_combined_u8x64 = _mm512_and_si512(_mm512_xor_si512(a_e3m2_u8x64, b_e3m2_u8x64), sign_mask_u8x64);
    __mmask64 negate_u8_mask = _mm512_test_epi8_mask(sign_combined_u8x64, sign_combined_u8x64);
    // Extract bits matching unpacklo element ordering (bytes 0-7,16-23,32-39,48-55 per 64-byte vector)
    __mmask32 negate_low_i16 = (__mmask32)_pext_u64(negate_u8_mask, 0x00FF00FF00FF00FFULL);
    __mmask32 negate_high_i16 = (__mmask32)_pext_u64(negate_u8_mask, 0xFF00FF00FF00FF00ULL);
    // Negate b at i16 level using mask_sub
    __m512i b_signed_low_i16x32 = _mm512_mask_sub_epi16(b_low_i16x32, negate_low_i16, _mm512_setzero_si512(),
                                                        b_low_i16x32);
    __m512i b_signed_high_i16x32 = _mm512_mask_sub_epi16(b_high_i16x32, negate_high_i16, _mm512_setzero_si512(),
                                                         b_high_i16x32);

    // VPMADDWD: a_i16 × b_signed_i16 → i32 accumulator
    sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(a_low_i16x32, b_signed_low_i16x32));
    sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(a_high_i16x32, b_signed_high_i16x32));

    if (count_scalars) goto nk_dot_e3m2_skylake_cycle;
    *result = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16) / 256.0f;
}

#pragma endregion F16 and BF16 Floats

#pragma region I8 and U8 Integers

NK_PUBLIC void nk_dot_i8_skylake(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_i32_t *result) {
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        // Load 32 bytes at a time and widen to i16
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i const *)(b_scalars + idx_scalars));
        __m512i a_i16x32 = _mm512_cvtepi8_epi16(a_i8x32);
        __m512i b_i16x32 = _mm512_cvtepi8_epi16(b_i8x32);
        // VPMADDWD: 5cy (0.5/cy) @ p05 - multiply adjacent i16 pairs, add to i32
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(a_i16x32, b_i16x32));
    }
    nk_i32_t sum = _mm512_reduce_add_epi32(sum_i32x16);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_u8_skylake(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_u32_t *result) {
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        // Load 32 bytes and zero-extend to i16 (u8 → u16 via zero-extension)
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b_scalars + idx_scalars));
        __m512i a_u16x32 = _mm512_cvtepu8_epi16(a_u8x32);
        __m512i b_u16x32 = _mm512_cvtepu8_epi16(b_u8x32);
        // VPMADDWD: 5cy (0.5/cy) @ p05 - multiply adjacent i16 pairs, add to i32
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(a_u16x32, b_u16x32));
    }
    nk_u32_t sum = (nk_u32_t)_mm512_reduce_add_epi32(sum_i32x16);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

typedef struct nk_dot_f64x8_state_skylake_t {
    __m512d sum_f64x8;
    __m512d compensation_f64x8;
} nk_dot_f64x8_state_skylake_t;

NK_INTERNAL void nk_dot_f64x8_init_skylake(nk_dot_f64x8_state_skylake_t *state) {
    state->sum_f64x8 = _mm512_setzero_pd();
    state->compensation_f64x8 = _mm512_setzero_pd();
}

NK_INTERNAL void nk_dot_f64x8_update_skylake(nk_dot_f64x8_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m512d sum_f64x8 = state->sum_f64x8;
    __m512d compensation_f64x8 = state->compensation_f64x8;
    __m512d a_f64x8 = a.zmm_pd;
    __m512d b_f64x8 = b.zmm_pd;

    // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
    __m512d product_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, product_f64x8);

    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    __m512d tentative_sum_f64x8 = _mm512_add_pd(sum_f64x8, product_f64x8);
    __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, sum_f64x8);
    __m512d sum_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
        _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));

    // Update: sum = t, compensation += q + r
    state->sum_f64x8 = tentative_sum_f64x8;
    state->compensation_f64x8 = _mm512_add_pd(compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
}

NK_INTERNAL void nk_dot_f64x8_finalize_skylake(                                               //
    nk_dot_f64x8_state_skylake_t const *state_a, nk_dot_f64x8_state_skylake_t const *state_b, //
    nk_dot_f64x8_state_skylake_t const *state_c, nk_dot_f64x8_state_skylake_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    // Compensated horizontal reduction preserving Dot2 error tracking per state
    result->f64s[0] = nk_dot_stable_sum_f64x8_skylake_(state_a->sum_f64x8, state_a->compensation_f64x8);
    result->f64s[1] = nk_dot_stable_sum_f64x8_skylake_(state_b->sum_f64x8, state_b->compensation_f64x8);
    result->f64s[2] = nk_dot_stable_sum_f64x8_skylake_(state_c->sum_f64x8, state_c->compensation_f64x8);
    result->f64s[3] = nk_dot_stable_sum_f64x8_skylake_(state_d->sum_f64x8, state_d->compensation_f64x8);
}

typedef struct nk_dot_f32x8_state_skylake_t {
    __m512d sum_f64x8;
} nk_dot_f32x8_state_skylake_t;

NK_INTERNAL void nk_dot_f32x8_init_skylake(nk_dot_f32x8_state_skylake_t *state) {
    state->sum_f64x8 = _mm512_setzero_pd();
}

NK_INTERNAL void nk_dot_f32x8_update_skylake(nk_dot_f32x8_state_skylake_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Upcast 8 f32s to f64 for high-precision accumulation
    __m512d a_f64x8 = _mm512_cvtps_pd(a.ymm_ps);
    __m512d b_f64x8 = _mm512_cvtps_pd(b.ymm_ps);
    // Simple FMA accumulation in f64
    state->sum_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, state->sum_f64x8);
}

NK_INTERNAL void nk_dot_f32x8_finalize_skylake(                                               //
    nk_dot_f32x8_state_skylake_t const *state_a, nk_dot_f32x8_state_skylake_t const *state_b, //
    nk_dot_f32x8_state_skylake_t const *state_c, nk_dot_f32x8_state_skylake_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    // ILP-optimized 4-way horizontal reduction for f64
    // Step 1: 8->4 for all 4 states (extract high 256-bit half and add to low half)
    __m256d sum_a_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(state_a->sum_f64x8),
                                        _mm512_extractf64x4_pd(state_a->sum_f64x8, 1));
    __m256d sum_b_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(state_b->sum_f64x8),
                                        _mm512_extractf64x4_pd(state_b->sum_f64x8, 1));
    __m256d sum_c_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(state_c->sum_f64x8),
                                        _mm512_extractf64x4_pd(state_c->sum_f64x8, 1));
    __m256d sum_d_f64x4 = _mm256_add_pd(_mm512_castpd512_pd256(state_d->sum_f64x8),
                                        _mm512_extractf64x4_pd(state_d->sum_f64x8, 1));
    // Step 2: 4->2 for all 4 states (extract high 128-bit half and add to low half)
    __m128d sum_a_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_a_f64x4), _mm256_extractf128_pd(sum_a_f64x4, 1));
    __m128d sum_b_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_b_f64x4), _mm256_extractf128_pd(sum_b_f64x4, 1));
    __m128d sum_c_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_c_f64x4), _mm256_extractf128_pd(sum_c_f64x4, 1));
    __m128d sum_d_f64x2 = _mm_add_pd(_mm256_castpd256_pd128(sum_d_f64x4), _mm256_extractf128_pd(sum_d_f64x4, 1));
    // Step 3: Horizontal add pairs: [a0+a1, b0+b1] and [c0+c1, d0+d1]
    __m128d sum_ab_f64x2 = _mm_hadd_pd(sum_a_f64x2, sum_b_f64x2);
    __m128d sum_cd_f64x2 = _mm_hadd_pd(sum_c_f64x2, sum_d_f64x2);
    result->ymm_pd = _mm256_set_m128d(sum_cd_f64x2, sum_ab_f64x2);
}

typedef nk_dot_through_f32_state_skylake_t_ nk_dot_bf16x16_state_skylake_t;

typedef nk_dot_through_f32_state_skylake_t_ nk_dot_bf16x32_state_skylake_t;

NK_INTERNAL void nk_dot_bf16x32_init_skylake(nk_dot_bf16x32_state_skylake_t *state) {
    nk_dot_through_f32_init_skylake_(state);
}

NK_INTERNAL void nk_dot_bf16x32_update_skylake(nk_dot_bf16x32_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m512i mask_high_u32x16 = _mm512_set1_epi32((int)0xFFFF0000);
    __m512 a_even_f32x16 = _mm512_castsi512_ps(_mm512_slli_epi32(a.zmm, 16));
    __m512 b_even_f32x16 = _mm512_castsi512_ps(_mm512_slli_epi32(b.zmm, 16));
    state->sum_f32x16 = _mm512_fmadd_ps(a_even_f32x16, b_even_f32x16, state->sum_f32x16);
    __m512 a_odd_f32x16 = _mm512_castsi512_ps(_mm512_and_si512(a.zmm, mask_high_u32x16));
    __m512 b_odd_f32x16 = _mm512_castsi512_ps(_mm512_and_si512(b.zmm, mask_high_u32x16));
    state->sum_f32x16 = _mm512_fmadd_ps(a_odd_f32x16, b_odd_f32x16, state->sum_f32x16);
}

NK_INTERNAL void nk_dot_bf16x32_finalize_skylake(                                                 //
    nk_dot_bf16x32_state_skylake_t const *state_a, nk_dot_bf16x32_state_skylake_t const *state_b, //
    nk_dot_bf16x32_state_skylake_t const *state_c, nk_dot_bf16x32_state_skylake_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_dot_through_f32_finalize_skylake_(state_a, state_b, state_c, state_d, total_dimensions, result);
}

typedef nk_dot_through_f32_state_skylake_t_ nk_dot_f16x16_state_skylake_t;

typedef struct nk_dot_e2m3x64_state_skylake_t {
    __m512i sum_i32x16;
} nk_dot_e2m3x64_state_skylake_t;

NK_INTERNAL void nk_dot_e2m3x64_init_skylake(nk_dot_e2m3x64_state_skylake_t *state) {
    state->sum_i32x16 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_e2m3x64_update_skylake(nk_dot_e2m3x64_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m512i const lut_low_u8x64 = _mm512_set_epi8(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28, 26,
                                                  24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28, 26, 24, 22, 20,
                                                  18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28, 26, 24, 22, 20, 18, 16, 14,
                                                  12, 10, 8, 6, 4, 2, 0);
    __m512i const lut_high_u8x64 = _mm512_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                   120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                   120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                   120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const magnitude_mask_u8x64 = _mm512_set1_epi8(0x1F);
    __m512i const half_select_u8x64 = _mm512_set1_epi8(0x10);
    __m512i const sign_mask_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const ones_i16x32 = _mm512_set1_epi16(1);

    __m512i a_u8x64 = a.zmm;
    __m512i b_u8x64 = b.zmm;

    __m512i a_magnitude = _mm512_and_si512(a_u8x64, magnitude_mask_u8x64);
    __m512i b_magnitude = _mm512_and_si512(b_u8x64, magnitude_mask_u8x64);
    __m512i a_shuffle_idx = _mm512_and_si512(a_magnitude, nibble_mask_u8x64);
    __m512i b_shuffle_idx = _mm512_and_si512(b_magnitude, nibble_mask_u8x64);

    __mmask64 a_high = _mm512_test_epi8_mask(a_magnitude, half_select_u8x64);
    __mmask64 b_high = _mm512_test_epi8_mask(b_magnitude, half_select_u8x64);

    __m512i a_unsigned = _mm512_mask_blend_epi8(a_high, _mm512_shuffle_epi8(lut_low_u8x64, a_shuffle_idx),
                                                _mm512_shuffle_epi8(lut_high_u8x64, a_shuffle_idx));
    __m512i b_unsigned = _mm512_mask_blend_epi8(b_high, _mm512_shuffle_epi8(lut_low_u8x64, b_shuffle_idx),
                                                _mm512_shuffle_epi8(lut_high_u8x64, b_shuffle_idx));

    __m512i sign_combined = _mm512_and_si512(_mm512_xor_si512(a_u8x64, b_u8x64), sign_mask_u8x64);
    __mmask64 negate_mask = _mm512_test_epi8_mask(sign_combined, sign_combined);
    __m512i b_signed_i8x64 = _mm512_mask_sub_epi8(b_unsigned, negate_mask, _mm512_setzero_si512(), b_unsigned);

    __m512i products_i16x32 = _mm512_maddubs_epi16(a_unsigned, b_signed_i8x64);
    state->sum_i32x16 = _mm512_add_epi32(state->sum_i32x16, _mm512_madd_epi16(products_i16x32, ones_i16x32));
}

NK_INTERNAL void nk_dot_e2m3x64_finalize_skylake(                                                 //
    nk_dot_e2m3x64_state_skylake_t const *state_a, nk_dot_e2m3x64_state_skylake_t const *state_b, //
    nk_dot_e2m3x64_state_skylake_t const *state_c, nk_dot_e2m3x64_state_skylake_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *results) {
    nk_unused_(total_dimensions);

    // 16→8 for all 4 states (extract high 256-bit half and add to low half)
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->sum_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->sum_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->sum_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->sum_i32x16, 1));

    // 8→4: extract high 128-bit half and add to low half
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));

    // 4×4 transpose and reduce (same as Sierra/Haswell integer finalize)
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_i32x4 = _mm_add_epi32(_mm_add_epi32(lane0_i32x4, lane1_i32x4), _mm_add_epi32(lane2_i32x4, lane3_i32x4));

    __m128 sum_f32x4 = _mm_mul_ps(_mm_cvtepi32_ps(sum_i32x4), _mm_set1_ps(1.0f / 256.0f));
    results->xmm = _mm_castps_si128(sum_f32x4);
}

typedef struct nk_dot_e3m2x64_state_skylake_t {
    __m512i sum_a_i32x16;
    __m512i sum_b_i32x16;
} nk_dot_e3m2x64_state_skylake_t;

NK_INTERNAL void nk_dot_e3m2x64_init_skylake(nk_dot_e3m2x64_state_skylake_t *state) {
    state->sum_a_i32x16 = _mm512_setzero_si512();
    state->sum_b_i32x16 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_e3m2x64_update_skylake(nk_dot_e3m2x64_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m512i const lut_low_byte_first_u8x64 = _mm512_set_epi8(                                                         //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const lut_low_byte_second_u8x64 = _mm512_set_epi8(                                                    //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32, //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32, //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32, //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32);
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const magnitude_mask_u8x64 = _mm512_set1_epi8(0x1F);
    __m512i const half_select_u8x64 = _mm512_set1_epi8(0x10);
    __m512i const sign_mask_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const ones_u8x64 = _mm512_set1_epi8(1);

    __m512i a_u8x64 = a.zmm;
    __m512i b_u8x64 = b.zmm;

    __m512i a_magnitude = _mm512_and_si512(a_u8x64, magnitude_mask_u8x64);
    __m512i b_magnitude = _mm512_and_si512(b_u8x64, magnitude_mask_u8x64);
    __m512i a_shuffle_idx = _mm512_and_si512(a_magnitude, nibble_mask_u8x64);
    __m512i b_shuffle_idx = _mm512_and_si512(b_magnitude, nibble_mask_u8x64);

    __mmask64 a_high = _mm512_test_epi8_mask(a_magnitude, half_select_u8x64);
    __mmask64 b_high = _mm512_test_epi8_mask(b_magnitude, half_select_u8x64);

    __m512i a_low_byte = _mm512_mask_blend_epi8(a_high, _mm512_shuffle_epi8(lut_low_byte_first_u8x64, a_shuffle_idx),
                                                _mm512_shuffle_epi8(lut_low_byte_second_u8x64, a_shuffle_idx));
    __m512i b_low_byte = _mm512_mask_blend_epi8(b_high, _mm512_shuffle_epi8(lut_low_byte_first_u8x64, b_shuffle_idx),
                                                _mm512_shuffle_epi8(lut_low_byte_second_u8x64, b_shuffle_idx));

    __mmask64 a_high_mask = _mm512_cmpge_epu8_mask(a_magnitude, _mm512_set1_epi8(28));
    __mmask64 b_high_mask = _mm512_cmpge_epu8_mask(b_magnitude, _mm512_set1_epi8(28));
    __m512i a_high_byte = _mm512_maskz_mov_epi8(a_high_mask, ones_u8x64);
    __m512i b_high_byte = _mm512_maskz_mov_epi8(b_high_mask, ones_u8x64);

    __m512i a_low_i16x32 = _mm512_unpacklo_epi8(a_low_byte, a_high_byte);
    __m512i a_high_i16x32 = _mm512_unpackhi_epi8(a_low_byte, a_high_byte);
    __m512i b_low_i16x32 = _mm512_unpacklo_epi8(b_low_byte, b_high_byte);
    __m512i b_high_i16x32 = _mm512_unpackhi_epi8(b_low_byte, b_high_byte);

    // Combined sign: negate b at i16 level via PEXT + mask_sub
    __m512i sign_combined = _mm512_and_si512(_mm512_xor_si512(a_u8x64, b_u8x64), sign_mask_u8x64);
    __mmask64 negate_u8 = _mm512_test_epi8_mask(sign_combined, sign_combined);
    __mmask32 negate_low = (__mmask32)_pext_u64(negate_u8, 0x00FF00FF00FF00FFULL);
    __mmask32 negate_high = (__mmask32)_pext_u64(negate_u8, 0xFF00FF00FF00FF00ULL);
    __m512i b_signed_low_i16x32 = _mm512_mask_sub_epi16(b_low_i16x32, negate_low, _mm512_setzero_si512(), b_low_i16x32);
    __m512i b_signed_high_i16x32 = _mm512_mask_sub_epi16(b_high_i16x32, negate_high, _mm512_setzero_si512(),
                                                         b_high_i16x32);

    state->sum_a_i32x16 = _mm512_add_epi32(state->sum_a_i32x16, _mm512_madd_epi16(a_low_i16x32, b_signed_low_i16x32));
    state->sum_b_i32x16 = _mm512_add_epi32(state->sum_b_i32x16, _mm512_madd_epi16(a_high_i16x32, b_signed_high_i16x32));
}

NK_INTERNAL void nk_dot_e3m2x64_finalize_skylake(                                                 //
    nk_dot_e3m2x64_state_skylake_t const *state_a, nk_dot_e3m2x64_state_skylake_t const *state_b, //
    nk_dot_e3m2x64_state_skylake_t const *state_c, nk_dot_e3m2x64_state_skylake_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *results) {
    nk_unused_(total_dimensions);

    // Merge two accumulators per state
    __m512i merged_a = _mm512_add_epi32(state_a->sum_a_i32x16, state_a->sum_b_i32x16);
    __m512i merged_b = _mm512_add_epi32(state_b->sum_a_i32x16, state_b->sum_b_i32x16);
    __m512i merged_c = _mm512_add_epi32(state_c->sum_a_i32x16, state_c->sum_b_i32x16);
    __m512i merged_d = _mm512_add_epi32(state_d->sum_a_i32x16, state_d->sum_b_i32x16);

    // 16→8
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(merged_a), _mm512_extracti32x8_epi32(merged_a, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(merged_b), _mm512_extracti32x8_epi32(merged_b, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(merged_c), _mm512_extracti32x8_epi32(merged_c, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(merged_d), _mm512_extracti32x8_epi32(merged_d, 1));

    // 8→4
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));

    // 4×4 transpose and reduce
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_i32x4 = _mm_add_epi32(_mm_add_epi32(lane0_i32x4, lane1_i32x4), _mm_add_epi32(lane2_i32x4, lane3_i32x4));

    __m128 sum_f32x4 = _mm_mul_ps(_mm_cvtepi32_ps(sum_i32x4), _mm_set1_ps(1.0f / 256.0f));
    results->xmm = _mm_castps_si128(sum_f32x4);
}

#pragma endregion I8 and U8 Integers

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X8664_
#endif // NK_DOT_SKYLAKE_H
