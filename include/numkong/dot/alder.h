/**
 *  @brief SIMD-accelerated Dot Products for Alder Lake.
 *  @file include/numkong/dot/alder.h
 *  @author Ash Vardanian
 *  @date March 4, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_alder_instructions AVX-VNNI Instructions Performance
 *
 *      Intrinsic            Instruction               Alder Lake  Raptor Lake
 *      _mm256_dpbusd_epi32  VPDPBUSD (YMM, YMM, YMM)  4cy @ p05   4cy @ p05
 *      _mm256_madd_epi16    VPMADDWD (YMM, YMM, YMM)  4cy @ p05   4cy @ p05
 *      _mm256_sad_epu8      VPSADBW (YMM, YMM, YMM)   3cy @ p5    3cy @ p5
 *
 *  Alder Lake and Raptor Lake support AVX-VNNI (256-bit VNNI)
 *  for accelerated integer dot products. This is the 256-bit variant of AVX-512 VNNI found on Ice Lake.
 *  We use VPDPBUSD for asymmetric unsigned×signed multiplication with algebraic transformations to
 *  handle signed×signed (i8) and unsigned×unsigned (u8) cases.
 *
 *  Performance improvements over previous approaches:
 *    - i8×i8: 1.3-1.4× speedup using dpbusd with XOR transformation (a+128)×b - 128×sum(b)
 *    - u8×u8: 1.8-2.0× speedup using dpbusd with XOR transformation a×(b-128) + 128×sum(a)
 *  These match the speedups achieved on Ice Lake (AVX-512 VNNI) but with 256-bit vectors.
 *
 *  @section dot_alder_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_i8x32 for 8-bit signed integer inputs using DPBUSD with algebraic transformation,
 *  - nk_dot_u8x32 for 8-bit unsigned integer inputs using DPBUSD with algebraic transformation.
 *
 *  @code{c}
 *  nk_dot_i8x32_state_alder_t state_first, state_second, state_third, state_fourth;
 *  nk_b256_vec_t query_i8x32, target_first_i8x32, target_second_i8x32, target_third_i8x32, target_fourth_i8x32;
 *  nk_dot_i8x32_init_alder(&state_first);
 *  nk_dot_i8x32_init_alder(&state_second);
 *  nk_dot_i8x32_init_alder(&state_third);
 *  nk_dot_i8x32_init_alder(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 32 <= depth; idx += 32) {
 *      query_i8x32.ymm = _mm256_loadu_si256(query_ptr + idx);
 *      target_first_i8x32.ymm = _mm256_loadu_si256(target_first_ptr + idx);
 *      target_second_i8x32.ymm = _mm256_loadu_si256(target_second_ptr + idx);
 *      target_third_i8x32.ymm = _mm256_loadu_si256(target_third_ptr + idx);
 *      target_fourth_i8x32.ymm = _mm256_loadu_si256(target_fourth_ptr + idx);
 *      nk_dot_i8x32_update_alder(&state_first, query_i8x32, target_first_i8x32, idx, 32);
 *      nk_dot_i8x32_update_alder(&state_second, query_i8x32, target_second_i8x32, idx, 32);
 *      nk_dot_i8x32_update_alder(&state_third, query_i8x32, target_third_i8x32, idx, 32);
 *      nk_dot_i8x32_update_alder(&state_fourth, query_i8x32, target_fourth_i8x32, idx, 32);
 *  }
 *  nk_b128_vec_t results_i32x4;
 *  nk_dot_i8x32_finalize_alder(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 *
 *  The unsigned variant follows the same pattern with appropriate type changes:
 *
 *  @code{c}
 *  nk_dot_u8x32_state_alder_t state_first, state_second, state_third, state_fourth;
 *  nk_b256_vec_t query_u8x32, target_first_u8x32, target_second_u8x32, target_third_u8x32, target_fourth_u8x32;
 *  nk_dot_u8x32_init_alder(&state_first);
 *  nk_dot_u8x32_init_alder(&state_second);
 *  nk_dot_u8x32_init_alder(&state_third);
 *  nk_dot_u8x32_init_alder(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 32 <= depth; idx += 32) {
 *      query_u8x32.ymm = _mm256_loadu_si256(query_ptr + idx);
 *      target_first_u8x32.ymm = _mm256_loadu_si256(target_first_ptr + idx);
 *      target_second_u8x32.ymm = _mm256_loadu_si256(target_second_ptr + idx);
 *      target_third_u8x32.ymm = _mm256_loadu_si256(target_third_ptr + idx);
 *      target_fourth_u8x32.ymm = _mm256_loadu_si256(target_fourth_ptr + idx);
 *      nk_dot_u8x32_update_alder(&state_first, query_u8x32, target_first_u8x32, idx, 32);
 *      nk_dot_u8x32_update_alder(&state_second, query_u8x32, target_second_u8x32, idx, 32);
 *      nk_dot_u8x32_update_alder(&state_third, query_u8x32, target_third_u8x32, idx, 32);
 *      nk_dot_u8x32_update_alder(&state_fourth, query_u8x32, target_fourth_u8x32, idx, 32);
 *  }
 *  nk_b128_vec_t results_u32x4;
 *  nk_dot_u8x32_finalize_alder(&state_first, &state_second, &state_third, &state_fourth, depth, &results_u32x4);
 *  @endcode
 */
#ifndef NK_DOT_ALDER_H
#define NK_DOT_ALDER_H

#if NK_TARGET_X8664_
#if NK_TARGET_ALDER

#include "numkong/types.h"
#include "numkong/cast/serial.h"    // `nk_partial_load_b8x32_serial_`
#include "numkong/reduce/haswell.h" // `nk_reduce_add_i32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

// On GCC/Clang, VEX encoding is handled by target attributes.
// Alias the MSVC-specific _avx intrinsic names to standard names.
#if !defined(_MSC_VER)
#define _mm256_dpbusd_avx_epi32 _mm256_dpbusd_epi32
#define _mm256_dpwssd_avx_epi32 _mm256_dpwssd_epi32
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni")
#endif

NK_PUBLIC void nk_dot_i8_alder(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                               nk_i32_t *result) {
    // Optimized i8×i8 dot product using algebraic transformation with DPBUSD
    //
    // Algebraic transformation:
    //   Let a' = a XOR 0x80 (interpreted as unsigned, gives a+128 mod 256)
    //   dpbusd(a', b) computes: (a+128) × b  [unsigned × signed]
    //   Therefore: a×b = (a+128)×b - 128×sum(b)
    //
    // Where:
    //   - XOR with 0x80 converts signed i8 [-128,127] to unsigned [0,255]
    //   - dpbusd performs unsigned×signed multiply-accumulate
    //   - Correction term 128×sum(b) is computed and subtracted at the end
    //
    // Performance: ~1.3-1.4× speedup expected over cvtepi8_epi16 + dpwssd approach
    //   - Processes 32 elements/iteration (AVX2 width)
    //   - Lower latency per iteration: 4 cy (VPDPBUSD @ p05) vs 3+4 = 7 cy (VPMOVSXBW @ p5 + VPMADDWD @ p05)
    //   - Better port utilization: VPDPBUSD (p05) runs in parallel with VPMOVSXBW (p5) + VPMADDWD (p05) for
    //     correction term, enabling dual-issue execution on p0 and p5 simultaneously. Old approach bottlenecked
    //     on p5 for sign extension.
    //
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const zeros_u8x32 = _mm256_setzero_si256();
    __m256i sum_ab_i32x8 = _mm256_setzero_si256();
    __m256i sum_b_biased_i64x4 = _mm256_setzero_si256();
    __m256i a_i8x32, b_i8x32;
    nk_size_t total_elements = count_scalars;

nk_dot_i8_alder_cycle:
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

    // Convert a to unsigned [0,255] by XOR with 0x80: a_unsigned = a + 128
    __m256i a_unsigned_u8x32 = _mm256_xor_si256(a_i8x32, xor_mask_u8x32);

    // Compute (a+128) × b using dpbusd: unsigned × signed
    sum_ab_i32x8 = _mm256_dpbusd_avx_epi32(sum_ab_i32x8, a_unsigned_u8x32, b_i8x32);

    // Accumulate sum(b+128) using SAD (replaces cvtepi8_epi16 + madd)
    __m256i b_biased_u8x32 = _mm256_xor_si256(b_i8x32, xor_mask_u8x32);
    sum_b_biased_i64x4 = _mm256_add_epi64(sum_b_biased_i64x4, _mm256_sad_epu8(b_biased_u8x32, zeros_u8x32));

    if (count_scalars) goto nk_dot_i8_alder_cycle;

    // Apply algebraic correction: a×b = (a+128)×b - 128×sum(b)
    // With biased accumulator: sum(b) = sum_b_biased - 128×count
    // So: correction = 128×sum(b) = 128×sum_b_biased - 16384×count
    nk_i32_t ab_sum = nk_reduce_add_i32x8_haswell_(sum_ab_i32x8);
    nk_i64_t sum_b_biased = nk_reduce_add_i64x4_haswell_(sum_b_biased_i64x4);
    nk_size_t elements_rounded = nk_size_round_up_to_multiple_(total_elements, 32);
    nk_i64_t correction = 128LL * sum_b_biased - 16384LL * (nk_i64_t)elements_rounded;

    *result = (nk_i32_t)(ab_sum - correction);
}

typedef struct nk_dot_i8x32_state_alder_t {
    __m256i biased_product_sum_i32x8; // Single accumulator: (a+128)×b, correction applied at finalize
} nk_dot_i8x32_state_alder_t;

NK_INTERNAL void nk_dot_i8x32_init_alder(nk_dot_i8x32_state_alder_t *state) {
    state->biased_product_sum_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_i8x32_update_alder(nk_dot_i8x32_state_alder_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                           nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256i a_unsigned_u8x32 = _mm256_xor_si256(a.ymm, _mm256_set1_epi8((char)0x80));
    state->biased_product_sum_i32x8 = _mm256_dpbusd_avx_epi32(state->biased_product_sum_i32x8, a_unsigned_u8x32, b.ymm);
}

NK_INTERNAL void nk_dot_i8x32_finalize_alder(                                             //
    nk_dot_i8x32_state_alder_t const *state_a, nk_dot_i8x32_state_alder_t const *state_b, //
    nk_dot_i8x32_state_alder_t const *state_c, nk_dot_i8x32_state_alder_t const *state_d, //
    nk_size_t total_dimensions,                                                           //
    nk_i32_t a_sum, /* A row sum (unused for i8) */                                       //
    nk_b128_vec_t const *b_sums_vec, /* 4 × i32 B column sums */                          //
    nk_b128_vec_t *result_vec) {
    nk_unused_(total_dimensions);
    nk_unused_(a_sum);

    // Reduce biased products: ymm (i32x8) → xmm (i32x4)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->biased_product_sum_i32x8),
                                        _mm256_extracti128_si256(state_a->biased_product_sum_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->biased_product_sum_i32x8),
                                        _mm256_extracti128_si256(state_b->biased_product_sum_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->biased_product_sum_i32x8),
                                        _mm256_extracti128_si256(state_c->biased_product_sum_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->biased_product_sum_i32x8),
                                        _mm256_extracti128_si256(state_d->biased_product_sum_i32x8, 1));

    // 4-way transpose reduce
    __m128i t_ab_low = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i t_cd_low = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i t_ab_high = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i t_cd_high = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i biased_i32x4 = _mm_add_epi32(
        _mm_add_epi32(_mm_unpacklo_epi64(t_ab_low, t_cd_low), _mm_unpackhi_epi64(t_ab_low, t_cd_low)),
        _mm_add_epi32(_mm_unpacklo_epi64(t_ab_high, t_cd_high), _mm_unpackhi_epi64(t_ab_high, t_cd_high)));

    // Apply compensation: result = biased − 128 × Σb
    __m128i correction_i32x4 = _mm_slli_epi32(b_sums_vec->xmm, 7); // × 128
    result_vec->xmm = _mm_sub_epi32(biased_i32x4, correction_i32x4);
}

NK_PUBLIC void nk_dot_u8_alder(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                               nk_u32_t *result) {
    // Optimized u8×u8 dot product using algebraic transformation with DPBUSD
    //
    // Algebraic transformation:
    //   Let b' = b XOR 0x80 (converts unsigned to signed: b' = b - 128)
    //   dpbusd(a, b') computes: a × (b-128)  [unsigned × signed]
    //   Therefore: a×b = a×(b-128) + 128×sum(a)
    //
    // Where:
    //   - XOR with 0x80 converts unsigned u8 [0,255] to signed [-128,127]
    //   - dpbusd performs unsigned×signed multiply-accumulate
    //   - sad_epu8 computes sum(a) as correction term
    //   - Correction term 128×sum(a) is added at the end
    //
    // Performance: ~1.8-2.0× speedup expected over unpack + dpwssd approach
    //   - Processes 32 elements/iteration
    //   - Lower latency per iteration
    //   - Eliminates unpack operations
    //   - dpbusd runs in parallel with sad
    //
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const zeros_u8x32 = _mm256_setzero_si256();
    __m256i sum_ab_i32x8 = _mm256_setzero_si256();
    __m256i sum_a_i64x4 = _mm256_setzero_si256();
    __m256i a_u8x32, b_u8x32;

nk_dot_u8_alder_cycle:
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

    // Convert b to signed [-128,127] by XOR with 0x80: b_signed = b - 128
    __m256i b_signed_i8x32 = _mm256_xor_si256(b_u8x32, xor_mask_u8x32);

    // Compute a × (b-128) using dpbusd: unsigned × signed
    sum_ab_i32x8 = _mm256_dpbusd_avx_epi32(sum_ab_i32x8, a_u8x32, b_signed_i8x32);

    // Accumulate sum(a) for correction term using sad_epu8 (1cy @ p5)
    sum_a_i64x4 = _mm256_add_epi64(sum_a_i64x4, _mm256_sad_epu8(a_u8x32, zeros_u8x32));

    if (count_scalars) goto nk_dot_u8_alder_cycle;

    // Apply algebraic correction: a×b = a×(b-128) + 128×sum(a)
    nk_i32_t ab_dot_signed = nk_reduce_add_i32x8_haswell_(sum_ab_i32x8);

    // Reduce sum_a from 4 i64 values to scalar
    __m128i sum_a_low_i64x2 = _mm256_castsi256_si128(sum_a_i64x4);
    __m128i sum_a_high_i64x2 = _mm256_extracti128_si256(sum_a_i64x4, 1);
    __m128i sum_a_i64x2 = _mm_add_epi64(sum_a_low_i64x2, sum_a_high_i64x2);
    __m128i sum_a_shuffled = _mm_shuffle_epi32(sum_a_i64x2, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sum_a_final = _mm_add_epi64(sum_a_i64x2, sum_a_shuffled);
    nk_i64_t sum_a = _mm_cvtsi128_si64(sum_a_final);

    nk_i64_t correction = 128LL * sum_a;

    *result = (nk_u32_t)(ab_dot_signed + correction);
}

typedef struct nk_dot_u8x32_state_alder_t {
    __m256i biased_product_sum_i32x8; // Single accumulator: DPBUSD(b, a^0x80), correction applied at finalize
} nk_dot_u8x32_state_alder_t;

NK_INTERNAL void nk_dot_u8x32_init_alder(nk_dot_u8x32_state_alder_t *state) {
    state->biased_product_sum_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_u8x32_update_alder(nk_dot_u8x32_state_alder_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                           nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Operand swap: DPBUSD(b, a^0x80) = b·(a−128) → result = biased + 128·Σb
    __m256i a_signed_i8x32 = _mm256_xor_si256(a.ymm, _mm256_set1_epi8((char)0x80));
    state->biased_product_sum_i32x8 = _mm256_dpbusd_avx_epi32(state->biased_product_sum_i32x8, b.ymm, a_signed_i8x32);
}

NK_INTERNAL void nk_dot_u8x32_finalize_alder(                                             //
    nk_dot_u8x32_state_alder_t const *state_a, nk_dot_u8x32_state_alder_t const *state_b, //
    nk_dot_u8x32_state_alder_t const *state_c, nk_dot_u8x32_state_alder_t const *state_d, //
    nk_size_t total_dimensions,                                                           //
    nk_i32_t a_sum, /* A row sum (unused for u8) */                                       //
    nk_b128_vec_t const *b_sums_vec, /* 4 × u32 B column sums */                          //
    nk_b128_vec_t *result_vec) {
    nk_unused_(total_dimensions);
    nk_unused_(a_sum);

    // Reduce biased products: ymm (i32x8) → xmm (i32x4)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_a->biased_product_sum_i32x8),
                                        _mm256_extracti128_si256(state_a->biased_product_sum_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_b->biased_product_sum_i32x8),
                                        _mm256_extracti128_si256(state_b->biased_product_sum_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_c->biased_product_sum_i32x8),
                                        _mm256_extracti128_si256(state_c->biased_product_sum_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(state_d->biased_product_sum_i32x8),
                                        _mm256_extracti128_si256(state_d->biased_product_sum_i32x8, 1));

    // 4-way transpose reduce
    __m128i t_ab_low = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i t_cd_low = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i t_ab_high = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i t_cd_high = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i biased_i32x4 = _mm_add_epi32(
        _mm_add_epi32(_mm_unpacklo_epi64(t_ab_low, t_cd_low), _mm_unpackhi_epi64(t_ab_low, t_cd_low)),
        _mm_add_epi32(_mm_unpacklo_epi64(t_ab_high, t_cd_high), _mm_unpackhi_epi64(t_ab_high, t_cd_high)));

    // Apply compensation: result = biased + 128 × Σb
    __m128i correction_i32x4 = _mm_slli_epi32(b_sums_vec->xmm, 7); // × 128
    result_vec->xmm = _mm_add_epi32(biased_i32x4, correction_i32x4);
}

/**
 *  Stateful element-sum helpers for compensated symmetric GEMM.
 *  SAD runs on port 5 while DPBUSD runs on ports 0+1 — zero throughput cost when inlined.
 */

/* i8x32: signed i8 sum via XOR→unsigned + SAD, bias-corrected at finalize */
typedef struct nk_sum_i8x32_state_alder_t {
    __m256i biased_sum_u64x4; /* Accumulates SAD of (v ^ 0x80), needs bias correction at finalize */
} nk_sum_i8x32_state_alder_t;

NK_INTERNAL void nk_sum_i8x32_init_alder(nk_sum_i8x32_state_alder_t *state) {
    state->biased_sum_u64x4 = _mm256_setzero_si256();
}
NK_INTERNAL void nk_sum_i8x32_update_alder(nk_sum_i8x32_state_alder_t *state, nk_b256_vec_t vector) {
    // Convert signed→unsigned via XOR 0x80, then SAD against zero gives sum of unsigned values
    __m256i vector_unsigned_u8x32 = _mm256_xor_si256(vector.ymm, _mm256_set1_epi8((char)0x80));
    __m256i sad_result_u64x4 = _mm256_sad_epu8(vector_unsigned_u8x32, _mm256_setzero_si256());
    state->biased_sum_u64x4 = _mm256_add_epi64(state->biased_sum_u64x4, sad_result_u64x4);
}
NK_INTERNAL nk_i32_t nk_sum_i8x32_finalize_alder(nk_sum_i8x32_state_alder_t const *state, nk_size_t count) {
    // Horizontal reduce u64x4 → scalar
    __m128i low_u64x2 = _mm256_castsi256_si128(state->biased_sum_u64x4);
    __m128i high_u64x2 = _mm256_extracti128_si256(state->biased_sum_u64x4, 1);
    __m128i paired_u64x2 = _mm_add_epi64(low_u64x2, high_u64x2);
    __m128i shuffled_u64x2 = _mm_shuffle_epi32(paired_u64x2, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i total_u64x2 = _mm_add_epi64(paired_u64x2, shuffled_u64x2);
    nk_u64_t unsigned_sum = (nk_u64_t)_mm_cvtsi128_si64(total_u64x2);
    // Undo XOR bias: signed_sum = unsigned_sum - 128 * count
    return (nk_i32_t)((nk_i64_t)unsigned_sum - 128 * (nk_i64_t)count);
}

/* u8x32: unsigned u8 sum via plain SAD — no bias correction needed */
typedef struct nk_sum_u8x32_state_alder_t {
    __m256i sum_u64x4; /* Direct SAD accumulator */
} nk_sum_u8x32_state_alder_t;

NK_INTERNAL void nk_sum_u8x32_init_alder(nk_sum_u8x32_state_alder_t *state) {
    state->sum_u64x4 = _mm256_setzero_si256();
}
NK_INTERNAL void nk_sum_u8x32_update_alder(nk_sum_u8x32_state_alder_t *state, nk_b256_vec_t vector) {
    __m256i sad_result_u64x4 = _mm256_sad_epu8(vector.ymm, _mm256_setzero_si256());
    state->sum_u64x4 = _mm256_add_epi64(state->sum_u64x4, sad_result_u64x4);
}
NK_INTERNAL nk_u32_t nk_sum_u8x32_finalize_alder(nk_sum_u8x32_state_alder_t const *state, nk_size_t count) {
    nk_unused_(count);
    __m128i low_u64x2 = _mm256_castsi256_si128(state->sum_u64x4);
    __m128i high_u64x2 = _mm256_extracti128_si256(state->sum_u64x4, 1);
    __m128i paired_u64x2 = _mm_add_epi64(low_u64x2, high_u64x2);
    __m128i shuffled_u64x2 = _mm_shuffle_epi32(paired_u64x2, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i total_u64x2 = _mm_add_epi64(paired_u64x2, shuffled_u64x2);
    return (nk_u32_t)_mm_cvtsi128_si64(total_u64x2);
}

NK_PUBLIC void nk_dot_e2m3_alder(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    // Integer dot product for e2m3 using dual-VPSHUFB (LUT) + VPDPBUSD (unsigned×signed).
    // Every e2m3 value × 16 is an exact integer in [-120, +120].
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // This is the Alder Lake (256-bit AVX-VNNI) variant of the Ice Lake kernel.
    // DPBUSD replaces MADDUBS+MADD (2 instructions → 1), accumulating u8×i8→i32 directly.
    //
    __m256i const lut_low_u8x32 = _mm256_set_epi8(                 //
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, //
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i const lut_high_u8x32 = _mm256_set_epi8(                        //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32, //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i a_e2m3_u8x32, b_e2m3_u8x32;

nk_dot_e2m3_alder_cycle:
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

    // Extract 5-bit magnitude, then split into low 4 bits (VPSHUFB index) and bit 4 (hi/lo select)
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_index_u8x32 = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_shuffle_index_u8x32 = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_high_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32),
                                                    half_select_u8x32);
    __m256i b_high_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32),
                                                    half_select_u8x32);

    // Dual VPSHUFB: lookup in both halves, blend based on bit 4
    __m256i a_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_low_u8x32, a_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_high_u8x32, a_shuffle_index_u8x32),
                                                  a_high_select_u8x32);
    __m256i b_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_low_u8x32, b_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_high_u8x32, b_shuffle_index_u8x32),
                                                  b_high_select_u8x32);

    // Combined sign: (a ^ b) & 0x20, negate b where signs differ
    __m256i sign_combined_u8x32 = _mm256_and_si256(_mm256_xor_si256(a_e2m3_u8x32, b_e2m3_u8x32), sign_mask_u8x32);
    __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(sign_combined_u8x32, sign_mask_u8x32);
    __m256i b_negated_u8x32 = _mm256_sub_epi8(_mm256_setzero_si256(), b_unsigned_u8x32);
    __m256i b_signed_i8x32 = _mm256_blendv_epi8(b_unsigned_u8x32, b_negated_u8x32, negate_mask_u8x32);

    // VPDPBUSD: a_unsigned[u8] × b_signed[i8] → i32 (replaces VPMADDUBSW + VPMADDWD)
    sum_i32x8 = _mm256_dpbusd_avx_epi32(sum_i32x8, a_unsigned_u8x32, b_signed_i8x32);

    if (count_scalars) goto nk_dot_e2m3_alder_cycle;
    *result = (nk_f32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8) / 256.0f;
}

typedef struct nk_dot_e2m3x32_state_alder_t {
    __m256i sum_i32x8; // DPBUSD accumulator: u8_magnitude × i8_signed → i32
} nk_dot_e2m3x32_state_alder_t;

NK_INTERNAL void nk_dot_e2m3x32_init_alder(nk_dot_e2m3x32_state_alder_t *state) {
    state->sum_i32x8 = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_e2m3x32_update_alder(nk_dot_e2m3x32_state_alder_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m256i const lut_low_u8x32 = _mm256_set_epi8(                 //
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, //
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i const lut_high_u8x32 = _mm256_set_epi8(                        //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32, //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);

    __m256i a_e2m3_u8x32 = a.ymm;
    __m256i b_e2m3_u8x32 = b.ymm;

    // Extract 5-bit magnitude, split into low 4 bits and bit 4
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_index_u8x32 = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_shuffle_index_u8x32 = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_high_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32),
                                                    half_select_u8x32);
    __m256i b_high_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32),
                                                    half_select_u8x32);

    // Dual VPSHUFB + blend
    __m256i a_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_low_u8x32, a_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_high_u8x32, a_shuffle_index_u8x32),
                                                  a_high_select_u8x32);
    __m256i b_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_low_u8x32, b_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_high_u8x32, b_shuffle_index_u8x32),
                                                  b_high_select_u8x32);

    // Combined sign + conditional negate
    __m256i sign_combined_u8x32 = _mm256_and_si256(_mm256_xor_si256(a_e2m3_u8x32, b_e2m3_u8x32), sign_mask_u8x32);
    __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(sign_combined_u8x32, sign_mask_u8x32);
    __m256i b_negated_u8x32 = _mm256_sub_epi8(_mm256_setzero_si256(), b_unsigned_u8x32);
    __m256i b_signed_i8x32 = _mm256_blendv_epi8(b_unsigned_u8x32, b_negated_u8x32, negate_mask_u8x32);

    // VPDPBUSD: u8 × i8 → i32
    state->sum_i32x8 = _mm256_dpbusd_avx_epi32(state->sum_i32x8, a_unsigned_u8x32, b_signed_i8x32);
}

NK_INTERNAL void nk_dot_e2m3x32_finalize_alder(                                               //
    nk_dot_e2m3x32_state_alder_t const *state_a, nk_dot_e2m3x32_state_alder_t const *state_b, //
    nk_dot_e2m3x32_state_alder_t const *state_c, nk_dot_e2m3x32_state_alder_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *results) {
    nk_unused_(total_dimensions);

    // ILP-optimized 4-way horizontal reduction: i32x8 → scalar i32, then → f32 with ÷256
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

    // Convert i32 → f32 and scale by 1/256
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

#endif // NK_TARGET_ALDER
#endif // NK_TARGET_X8664_
#endif // NK_DOT_ALDER_H
