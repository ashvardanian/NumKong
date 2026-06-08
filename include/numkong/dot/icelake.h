/**
 *  @brief SIMD-accelerated Dot Products for Ice Lake.
 *  @file include/numkong/dot/icelake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_icelake_instructions VNNI Instructions Performance
 *
 *      Intrinsic            Instruction               Icelake    Genoa
 *      _mm512_dpwssd_epi32  VPDPWSSD (ZMM, ZMM, ZMM)  5cy @ p0   4cy @ p01
 *      _mm512_dpbusd_epi32  VPDPBUSD (ZMM, ZMM, ZMM)  5cy @ p0   4cy @ p01
 *      _mm512_madd_epi16    VPMADDWD (ZMM, ZMM, ZMM)  5cy @ p05  3cy @ p01
 *
 *  Ice Lake introduces AVX-512 VNNI for accelerated integer dot products. VNNI instructions bottleneck
 *  on port 0, limiting throughput to 1/cy. AMD Genoa dual-issues on ports 0-1, achieving 0.5/cy throughput.
 *
 *  @section dot_icelake_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_i8x64 for 8-bit signed integer inputs using DPBUSD with algebraic transformation,
 *  - nk_dot_u8x64 for 8-bit unsigned integer inputs using DPBUSD with algebraic transformation,
 *  - nk_dot_i4x128 for 4-bit signed integer products with correction terms,
 *  - nk_dot_u4x128 for 4-bit unsigned integer products.
 *
 *  @code{c}
 *  nk_dot_i8x64_state_icelake_t state_first, state_second, state_third, state_fourth;
 *  nk_b512_vec_t query_i8x64, target_first_i8x64, target_second_i8x64, target_third_i8x64, target_fourth_i8x64;
 *  nk_dot_i8x64_init_icelake(&state_first);
 *  nk_dot_i8x64_init_icelake(&state_second);
 *  nk_dot_i8x64_init_icelake(&state_third);
 *  nk_dot_i8x64_init_icelake(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 64 <= depth; idx += 64) {
 *      query_i8x64.zmm = _mm512_loadu_si512(query_ptr + idx);
 *      target_first_i8x64.zmm = _mm512_loadu_si512(target_first_ptr + idx);
 *      target_second_i8x64.zmm = _mm512_loadu_si512(target_second_ptr + idx);
 *      target_third_i8x64.zmm = _mm512_loadu_si512(target_third_ptr + idx);
 *      target_fourth_i8x64.zmm = _mm512_loadu_si512(target_fourth_ptr + idx);
 *      nk_dot_i8x64_update_icelake(&state_first, query_i8x64, target_first_i8x64, idx, 64);
 *      nk_dot_i8x64_update_icelake(&state_second, query_i8x64, target_second_i8x64, idx, 64);
 *      nk_dot_i8x64_update_icelake(&state_third, query_i8x64, target_third_i8x64, idx, 64);
 *      nk_dot_i8x64_update_icelake(&state_fourth, query_i8x64, target_fourth_i8x64, idx, 64);
 *  }
 *  nk_b128_vec_t results_i32x4;
 *  nk_dot_i8x64_finalize_icelake(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 *
 *  For 4-bit integers, the state manages the complex unpacking and correction term accumulation:
 *
 *  @code{c}
 *  nk_dot_i4x128_state_icelake_t state_first, state_second, state_third, state_fourth;
 *  nk_b512_vec_t query_i4x128, target_first_i4x128, target_second_i4x128, target_third_i4x128, target_fourth_i4x128;
 *  nk_dot_i4x128_init_icelake(&state_first);
 *  nk_dot_i4x128_init_icelake(&state_second);
 *  nk_dot_i4x128_init_icelake(&state_third);
 *  nk_dot_i4x128_init_icelake(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 128 <= depth; idx += 128) {
 *      query_i4x128.zmm = _mm512_loadu_si512(query_ptr + idx / 2);
 *      target_first_i4x128.zmm = _mm512_loadu_si512(target_first_ptr + idx / 2);
 *      target_second_i4x128.zmm = _mm512_loadu_si512(target_second_ptr + idx / 2);
 *      target_third_i4x128.zmm = _mm512_loadu_si512(target_third_ptr + idx / 2);
 *      target_fourth_i4x128.zmm = _mm512_loadu_si512(target_fourth_ptr + idx / 2);
 *      nk_dot_i4x128_update_icelake(&state_first, query_i4x128, target_first_i4x128, idx, 128);
 *      nk_dot_i4x128_update_icelake(&state_second, query_i4x128, target_second_i4x128, idx, 128);
 *      nk_dot_i4x128_update_icelake(&state_third, query_i4x128, target_third_i4x128, idx, 128);
 *      nk_dot_i4x128_update_icelake(&state_fourth, query_i4x128, target_fourth_i4x128, idx, 128);
 *  }
 *  nk_b128_vec_t results_i32x4;
 *  nk_dot_i4x128_finalize_icelake(&state_first, &state_second, &state_third, &state_fourth, depth, &results_i32x4);
 *  @endcode
 */
#ifndef NK_DOT_ICELAKE_H
#define NK_DOT_ICELAKE_H

#if NK_TARGET_X8664_
#if NK_TARGET_ICELAKE

#include "numkong/types.h"
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                        \
    __attribute__((                                                                                                  \
        target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512vbmi,avx512vpopcntdq,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512vbmi", \
                   "avx512vpopcntdq", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_dot_i8_icelake(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_i32_t *result) {
    // Optimized i8×i8 dot product using algebraic transformation with DPBUSD
    //
    // Old approach (Haswell/Skylake):
    //   - Sign-extend i8 → i16 using cvtepi8_epi16 (3cy latency @ p5, 32 elements/iteration)
    //   - Multiply i16×i16 using vpmaddwd + dpwssd
    //   - Bottleneck: cvtepi8_epi16 serializes on port 5
    //
    // New approach (Ice Lake+):
    //   - Use DPBUSD (unsigned×signed multiply-add) with algebraic transformation
    //   - Convert signed i8 to unsigned via XOR with 0x80: a' = a + 128
    //   - Compute dpbusd(a', b) = (a+128)×b, then correct: a×b = (a+128)×b - 128×sum(b)
    //   - Use SAD for fast correction term accumulation (1cy @ p5 vs 8-10cy with cvtepi8)
    //   - Processes 64 elements/iteration
    //
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();
    __m512i sum_ab_i32x16 = _mm512_setzero_si512();
    __m512i sum_b_biased_i64x8 = _mm512_setzero_si512();
    __m512i a_i8x64, b_i8x64;
    nk_size_t count_original = count_scalars;

nk_dot_i8_icelake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_i8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_i8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i8x64 = _mm512_loadu_si512(a_scalars);
        b_i8x64 = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Convert a to unsigned [0,255] by XOR with 0x80: a_biased = a + 128
    __m512i a_biased_u8x64 = _mm512_xor_si512(a_i8x64, xor_mask_u8x64);

    // Compute (a+128) × b using dpbusd: unsigned × signed
    sum_ab_i32x16 = _mm512_dpbusd_epi32(sum_ab_i32x16, a_biased_u8x64, b_i8x64);

    // Accumulate sum(b+128) using SAD (1cy @ p5 instead of 8-10cy with cvtepi8+madd)
    __m512i b_biased_u8x64 = _mm512_xor_si512(b_i8x64, xor_mask_u8x64);
    sum_b_biased_i64x8 = _mm512_add_epi64(sum_b_biased_i64x8, _mm512_sad_epu8(b_biased_u8x64, zeros_u8x64));

    if (count_scalars) goto nk_dot_i8_icelake_cycle;

    // Apply algebraic correction: a×b = (a+128)×b - 128×sum(b)
    // sum_b = sum_b_biased - 128×count_rounded
    // correction = 128×sum_b = 128×sum_b_biased - 16384×count_rounded
    nk_i32_t ab_sum = _mm512_reduce_add_epi32(sum_ab_i32x16);
    nk_i64_t sum_b_biased = _mm512_reduce_add_epi64(sum_b_biased_i64x8);
    nk_size_t count_rounded = nk_size_round_up_to_multiple_(count_original, 64);
    nk_i64_t correction = 128LL * sum_b_biased - 16384LL * (nk_i64_t)count_rounded;

    *result = (nk_i32_t)(ab_sum - correction);
}

NK_PUBLIC void nk_dot_u8_icelake(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
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
    // Performance: 1.92× speedup over unpack + dpwssd approach
    //   - Processes 64 elements/iteration
    //   - Lower latency: ~8cy vs ~16cy per iteration
    //   - Eliminates 4× unpack operations (1cy each @ p5)
    //   - dpbusd@p0 runs in parallel with sad@p5
    //
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();
    __m512i sum_ab_i32x16 = _mm512_setzero_si512();
    __m512i sum_a_i64x8 = _mm512_setzero_si512();
    __m512i a_u8x64, b_u8x64;

nk_dot_u8_icelake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512(a_scalars);
        b_u8x64 = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Convert b to signed [-128,127] by XOR with 0x80: b_signed = b - 128
    __m512i b_signed_i8x64 = _mm512_xor_si512(b_u8x64, xor_mask_u8x64);

    // Compute a × (b-128) using dpbusd: unsigned × signed
    sum_ab_i32x16 = _mm512_dpbusd_epi32(sum_ab_i32x16, a_u8x64, b_signed_i8x64);

    // Accumulate sum(a) for correction term using sad_epu8 (1cy @ p5)
    sum_a_i64x8 = _mm512_add_epi64(sum_a_i64x8, _mm512_sad_epu8(a_u8x64, zeros_u8x64));

    if (count_scalars) goto nk_dot_u8_icelake_cycle;

    // Apply algebraic correction: a×b = a×(b-128) + 128×sum(a)
    nk_i32_t ab_dot_signed = _mm512_reduce_add_epi32(sum_ab_i32x16);
    nk_i64_t sum_a = _mm512_reduce_add_epi64(sum_a_i64x8);
    nk_i64_t correction = 128LL * sum_a;

    *result = (nk_u32_t)(ab_dot_signed + correction);
}

typedef struct nk_dot_i8x64_state_icelake_t {
    __m512i biased_product_sum_i32x16; // Single accumulator: (a^0x80)×b
} nk_dot_i8x64_state_icelake_t;

NK_INTERNAL void nk_dot_i8x64_init_icelake(nk_dot_i8x64_state_icelake_t *state) {
    state->biased_product_sum_i32x16 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_i8x64_update_icelake(nk_dot_i8x64_state_icelake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Optimized i8×i8 using DPBUSD with algebraic transformation
    // DPBUSD(a^0x80, b) = (a+128)·b = a·b + 128·Σb
    // Correction applied at finalize: result = biased − 128·Σb
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);

    __m512i a_i8x64 = a.zmm;
    __m512i b_i8x64 = b.zmm;

    // Convert a to unsigned: a_unsigned = a ^ 0x80
    __m512i a_unsigned_u8x64 = _mm512_xor_si512(a_i8x64, xor_mask_u8x64);

    // Compute (a+128) × b using dpbusd — no correction accumulator needed
    state->biased_product_sum_i32x16 = _mm512_dpbusd_epi32(state->biased_product_sum_i32x16, a_unsigned_u8x64, b_i8x64);
}

NK_INTERNAL void nk_dot_i8x64_finalize_icelake(                                               //
    nk_dot_i8x64_state_icelake_t const *state_a, nk_dot_i8x64_state_icelake_t const *state_b, //
    nk_dot_i8x64_state_icelake_t const *state_c, nk_dot_i8x64_state_icelake_t const *state_d, //
    nk_size_t total_dimensions,                                                               //
    nk_i32_t a_sum, /* A row sum (unused for i8) */                                           //
    nk_b128_vec_t const *b_sums_vec, /* 4 × i32 B column sums */                              //
    nk_b128_vec_t *result_vec) {
    nk_unused_(total_dimensions);
    nk_unused_(a_sum);

    // Reduce biased products: zmm (i32x16) → ymm (i32x8)
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->biased_product_sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->biased_product_sum_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->biased_product_sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->biased_product_sum_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->biased_product_sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->biased_product_sum_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->biased_product_sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->biased_product_sum_i32x16, 1));

    // Reduce ymm (i32x8) → xmm (i32x4)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));

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

typedef struct nk_dot_u8x64_state_icelake_t {
    __m512i biased_product_sum_i32x16; // Single accumulator: DPBUSD(b, a^0x80)
} nk_dot_u8x64_state_icelake_t;

NK_INTERNAL void nk_dot_u8x64_init_icelake(nk_dot_u8x64_state_icelake_t *state) {
    state->biased_product_sum_i32x16 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_u8x64_update_icelake(nk_dot_u8x64_state_icelake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Optimized u8×u8 using operand swap: DPBUSD(b, a^0x80)
    // DPBUSD(b, a^0x80) = b·(a−128) = a·b − 128·Σb
    // Correction applied at finalize: result = biased + 128·Σb
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);

    __m512i a_u8x64 = a.zmm;
    __m512i b_u8x64 = b.zmm;

    // Convert a to signed: a_signed = a ^ 0x80 = a − 128
    __m512i a_signed_i8x64 = _mm512_xor_si512(a_u8x64, xor_mask_u8x64);

    // Operand swap: b (unsigned) in first slot, a−128 (signed) in second
    state->biased_product_sum_i32x16 = _mm512_dpbusd_epi32(state->biased_product_sum_i32x16, b_u8x64, a_signed_i8x64);
}

NK_INTERNAL void nk_dot_u8x64_finalize_icelake(                                               //
    nk_dot_u8x64_state_icelake_t const *state_a, nk_dot_u8x64_state_icelake_t const *state_b, //
    nk_dot_u8x64_state_icelake_t const *state_c, nk_dot_u8x64_state_icelake_t const *state_d, //
    nk_size_t total_dimensions,                                                               //
    nk_i32_t a_sum, /* A row sum (unused for u8) */                                           //
    nk_b128_vec_t const *b_sums_vec, /* 4 × u32 B column sums */                              //
    nk_b128_vec_t *result_vec) {
    nk_unused_(total_dimensions);
    nk_unused_(a_sum);

    // Reduce biased products: zmm (i32x16) → ymm (i32x8)
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->biased_product_sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->biased_product_sum_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->biased_product_sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->biased_product_sum_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->biased_product_sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->biased_product_sum_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->biased_product_sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->biased_product_sum_i32x16, 1));

    // Reduce ymm (i32x8) → xmm (i32x4)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));

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
 *  SAD512 runs on port 5 while DPBUSD runs on port 0 — zero throughput cost when inlined.
 */

/* i8x64: signed i8 sum via XOR→unsigned + SAD, bias-corrected at finalize */
typedef struct nk_sum_i8x64_state_icelake_t {
    __m512i biased_sum_u64x8;
} nk_sum_i8x64_state_icelake_t;

NK_INTERNAL void nk_sum_i8x64_init_icelake(nk_sum_i8x64_state_icelake_t *state) {
    state->biased_sum_u64x8 = _mm512_setzero_si512();
}
NK_INTERNAL void nk_sum_i8x64_update_icelake(nk_sum_i8x64_state_icelake_t *state, nk_b512_vec_t vector) {
    __m512i vector_unsigned_u8x64 = _mm512_xor_si512(vector.zmm, _mm512_set1_epi8((char)0x80));
    __m512i sad_result_u64x8 = _mm512_sad_epu8(vector_unsigned_u8x64, _mm512_setzero_si512());
    state->biased_sum_u64x8 = _mm512_add_epi64(state->biased_sum_u64x8, sad_result_u64x8);
}
NK_INTERNAL nk_i32_t nk_sum_i8x64_finalize_icelake(nk_sum_i8x64_state_icelake_t const *state, nk_size_t count) {
    nk_u64_t unsigned_sum = (nk_u64_t)_mm512_reduce_add_epi64(state->biased_sum_u64x8);
    return (nk_i32_t)((nk_i64_t)unsigned_sum - 128 * (nk_i64_t)count);
}

/* u8x64: unsigned u8 sum via plain SAD */
typedef struct nk_sum_u8x64_state_icelake_t {
    __m512i sum_u64x8;
} nk_sum_u8x64_state_icelake_t;

NK_INTERNAL void nk_sum_u8x64_init_icelake(nk_sum_u8x64_state_icelake_t *state) {
    state->sum_u64x8 = _mm512_setzero_si512();
}
NK_INTERNAL void nk_sum_u8x64_update_icelake(nk_sum_u8x64_state_icelake_t *state, nk_b512_vec_t vector) {
    __m512i sad_result_u64x8 = _mm512_sad_epu8(vector.zmm, _mm512_setzero_si512());
    state->sum_u64x8 = _mm512_add_epi64(state->sum_u64x8, sad_result_u64x8);
}
NK_INTERNAL nk_u32_t nk_sum_u8x64_finalize_icelake(nk_sum_u8x64_state_icelake_t const *state, nk_size_t count) {
    nk_unused_(count);
    return (nk_u32_t)_mm512_reduce_add_epi64(state->sum_u64x8);
}

/* i4x128: signed i4 sum — vectorized nibble extraction + SAD on 512-bit vector.
 * Each byte contains 2 nibbles in [0,15] representing signed values in [-8,7].
 * We XOR nibbles with 0x08 to get unsigned [0,15], SAD against zero, then bias-correct at finalize. */
typedef struct nk_sum_i4x128_state_icelake_t {
    __m512i biased_sum_u64x8; /* Accumulates SAD of (nibble ^ 0x08), needs bias correction */
} nk_sum_i4x128_state_icelake_t;

NK_INTERNAL void nk_sum_i4x128_init_icelake(nk_sum_i4x128_state_icelake_t *state) {
    state->biased_sum_u64x8 = _mm512_setzero_si512();
}
NK_INTERNAL void nk_sum_i4x128_update_icelake(nk_sum_i4x128_state_icelake_t *state, nk_b512_vec_t v) {
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8(0x08);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();
    // Extract low and high nibbles, XOR with 8 to get unsigned representation
    __m512i low_u8x64 = _mm512_and_si512(v.zmm, nibble_mask_u8x64);
    __m512i high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(v.zmm, 4), nibble_mask_u8x64);
    __m512i low_biased_u8x64 = _mm512_xor_si512(low_u8x64, xor_mask_u8x64);
    __m512i high_biased_u8x64 = _mm512_xor_si512(high_u8x64, xor_mask_u8x64);
    // SAD against zero gives sum of unsigned values, accumulate in u64 lanes
    state->biased_sum_u64x8 = _mm512_add_epi64(state->biased_sum_u64x8, _mm512_sad_epu8(low_biased_u8x64, zeros_u8x64));
    state->biased_sum_u64x8 = _mm512_add_epi64(state->biased_sum_u64x8,
                                               _mm512_sad_epu8(high_biased_u8x64, zeros_u8x64));
}
NK_INTERNAL nk_i32_t nk_sum_i4x128_finalize_icelake(nk_sum_i4x128_state_icelake_t const *state, nk_size_t count) {
    // Reduce u64x8 → scalar, then undo XOR bias: signed_sum = unsigned_sum - 8 * count
    nk_i64_t unsigned_sum = _mm512_reduce_add_epi64(state->biased_sum_u64x8);
    return (nk_i32_t)(unsigned_sum - 8 * (nk_i64_t)count);
}

NK_PUBLIC void nk_dot_i4_icelake(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    //
    // Algorithm: For signed i4, we use an algebraic transformation.
    // Let ax, bx be the unsigned [0,15] representation of signed values a, b in [-8,7].
    // Then: a = ax - 8, b = bx - 8 (the XOR trick gives signed = (unsigned ^ 8) - 8)
    // So: a * b = (ax - 8)(bx - 8) = ax * bx - 8 * ax - 8 * bx + 64
    //
    // We compute ax * bx using DPBUSD, then apply the correction:
    //   signed_dot = unsigned_dot - 8 * (sum_ax + sum_bx) + 64 * n
    //
    n = nk_size_round_up_to_multiple_(n, 2);
    nk_size_t n_bytes = n / 2;
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8(0x08);
    __m512i const zeros_u8x64 = _mm512_setzero_si512();
    __m512i sum_cd_i32x16 = _mm512_setzero_si512();
    __m512i sum_cx_i64x8 = _mm512_setzero_si512();
    __m512i sum_dx_i64x8 = _mm512_setzero_si512();
    __m512i a_i4x128, b_i4x128;

nk_dot_i4_icelake_cycle:
    if (n_bytes < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        a_i4x128 = _mm512_mask_loadu_epi8(_mm512_set1_epi8((char)0x88), mask, a);
        b_i4x128 = _mm512_mask_loadu_epi8(_mm512_set1_epi8((char)0x88), mask, b);
        n_bytes = 0;
    }
    else {
        a_i4x128 = _mm512_loadu_si512(a);
        b_i4x128 = _mm512_loadu_si512(b);
        a += 64, b += 64, n_bytes -= 64;
    }

    // Extract low and high nibbles
    __m512i a_low_u8x64 = _mm512_and_si512(a_i4x128, nibble_mask_u8x64);
    __m512i a_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_i4x128, 4), nibble_mask_u8x64);
    __m512i b_low_u8x64 = _mm512_and_si512(b_i4x128, nibble_mask_u8x64);
    __m512i b_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_i4x128, 4), nibble_mask_u8x64);

    // XOR with 8 to get cx, dx values for the algebraic transformation
    __m512i c_low_u8x64 = _mm512_xor_si512(a_low_u8x64, xor_mask_u8x64);
    __m512i c_high_u8x64 = _mm512_xor_si512(a_high_u8x64, xor_mask_u8x64);
    __m512i d_low_u8x64 = _mm512_xor_si512(b_low_u8x64, xor_mask_u8x64);
    __m512i d_high_u8x64 = _mm512_xor_si512(b_high_u8x64, xor_mask_u8x64);

    // Compute dot products of cx*dx for low and high nibbles
    sum_cd_i32x16 = _mm512_dpbusd_epi32(sum_cd_i32x16, c_low_u8x64, d_low_u8x64);
    sum_cd_i32x16 = _mm512_dpbusd_epi32(sum_cd_i32x16, c_high_u8x64, d_high_u8x64);

    // Accumulate sums of cx and dx using SAD against zeros
    sum_cx_i64x8 = _mm512_add_epi64(sum_cx_i64x8, _mm512_sad_epu8(c_low_u8x64, zeros_u8x64));
    sum_cx_i64x8 = _mm512_add_epi64(sum_cx_i64x8, _mm512_sad_epu8(c_high_u8x64, zeros_u8x64));
    sum_dx_i64x8 = _mm512_add_epi64(sum_dx_i64x8, _mm512_sad_epu8(d_low_u8x64, zeros_u8x64));
    sum_dx_i64x8 = _mm512_add_epi64(sum_dx_i64x8, _mm512_sad_epu8(d_high_u8x64, zeros_u8x64));
    if (n_bytes) goto nk_dot_i4_icelake_cycle;

    // Reduce partial sums and apply algebraic correction
    nk_i32_t cd_dot = _mm512_reduce_add_epi32(sum_cd_i32x16);
    nk_i64_t sum_cx = _mm512_reduce_add_epi64(sum_cx_i64x8);
    nk_i64_t sum_dx = _mm512_reduce_add_epi64(sum_dx_i64x8);
    *result = (nk_i32_t)(cd_dot - 8 * (sum_cx + sum_dx) + 64 * (nk_i64_t)n);
}

NK_PUBLIC void nk_dot_u4_icelake(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // Values are ∈ [0,15], so DPBUSD can be used directly.
    //
    n = nk_size_round_up_to_multiple_(n, 2);
    nk_size_t n_bytes = n / 2;
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i sum_i32x16 = _mm512_setzero_si512();

    __m512i a_u4x128, b_u4x128;

nk_dot_u4_icelake_cycle:
    if (n_bytes < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        a_u4x128 = _mm512_maskz_loadu_epi8(mask, a);
        b_u4x128 = _mm512_maskz_loadu_epi8(mask, b);
        n_bytes = 0;
    }
    else {
        a_u4x128 = _mm512_loadu_si512(a);
        b_u4x128 = _mm512_loadu_si512(b);
        a += 64, b += 64, n_bytes -= 64;
    }

    // Extract low and high nibbles
    __m512i a_low_u8x64 = _mm512_and_si512(a_u4x128, nibble_mask_u8x64);
    __m512i a_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_u4x128, 4), nibble_mask_u8x64);
    __m512i b_low_u8x64 = _mm512_and_si512(b_u4x128, nibble_mask_u8x64);
    __m512i b_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_u4x128, 4), nibble_mask_u8x64);

    // DPBUSD works directly for u4 since values are ∈ [0,15]
    // and the signed interpretation of [0,15] is the same as unsigned
    sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, a_low_u8x64, b_low_u8x64);
    sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, a_high_u8x64, b_high_u8x64);
    if (n_bytes) goto nk_dot_u4_icelake_cycle;

    *result = (nk_u32_t)_mm512_reduce_add_epi32(sum_i32x16);
}

typedef struct nk_dot_i4x128_state_icelake_t {
    __m512i biased_product_sum_i32x16; // Single accumulator: (a^8)×(b^8) products
} nk_dot_i4x128_state_icelake_t;

NK_INTERNAL void nk_dot_i4x128_init_icelake(nk_dot_i4x128_state_icelake_t *state) {
    state->biased_product_sum_i32x16 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_i4x128_update_icelake(nk_dot_i4x128_state_icelake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    // i4 values are packed as nibbles: 128 nibbles in 64 bytes (512 bits)
    // Algebraic transformation: a×b = (a^8)×(b^8) − 8×(Σa + Σb) − 64×n
    // Correction applied at finalize time using precomputed sums.
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const bias_xor_mask_u8x64 = _mm512_set1_epi8(0x08);

    __m512i a_i4x128 = a.zmm;
    __m512i b_i4x128 = b.zmm;

    // Extract low and high nibbles (all 128 nibbles from 64 bytes)
    __m512i a_low_u8x64 = _mm512_and_si512(a_i4x128, nibble_mask_u8x64);
    __m512i a_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_i4x128, 4), nibble_mask_u8x64);
    __m512i b_low_u8x64 = _mm512_and_si512(b_i4x128, nibble_mask_u8x64);
    __m512i b_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_i4x128, 4), nibble_mask_u8x64);

    // Apply bias transformation: XOR with 8
    __m512i a_biased_low_u8x64 = _mm512_xor_si512(a_low_u8x64, bias_xor_mask_u8x64);
    __m512i a_biased_high_u8x64 = _mm512_xor_si512(a_high_u8x64, bias_xor_mask_u8x64);
    __m512i b_biased_low_u8x64 = _mm512_xor_si512(b_low_u8x64, bias_xor_mask_u8x64);
    __m512i b_biased_high_u8x64 = _mm512_xor_si512(b_high_u8x64, bias_xor_mask_u8x64);

    // Compute dot products of a_biased×b_biased — no SAD correction accumulators
    state->biased_product_sum_i32x16 = _mm512_dpbusd_epi32(state->biased_product_sum_i32x16, a_biased_low_u8x64,
                                                           b_biased_low_u8x64);
    state->biased_product_sum_i32x16 = _mm512_dpbusd_epi32(state->biased_product_sum_i32x16, a_biased_high_u8x64,
                                                           b_biased_high_u8x64);
}

NK_INTERNAL void nk_dot_i4x128_finalize_icelake(                                                //
    nk_dot_i4x128_state_icelake_t const *state_a, nk_dot_i4x128_state_icelake_t const *state_b, //
    nk_dot_i4x128_state_icelake_t const *state_c, nk_dot_i4x128_state_icelake_t const *state_d, //
    nk_size_t total_dimensions,                                                                 //
    nk_i32_t a_sum, /* A row sum (signed sum of i4 values) */                                   //
    nk_b128_vec_t const *b_sums_vec, /* 4 × i32 B column sums */                                //
    nk_b128_vec_t *result_vec) {

    // Compensated 4-way reduction with external correction sums.
    // Formula: result = biased_product − 8×(Σa + Σb) − 64×depth_padded
    nk_size_t depth_nibbles = nk_size_round_up_to_multiple_(total_dimensions, 128);

    // Reduce main products: zmm (i32x16) → ymm (i32x8)
    __m256i product_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->biased_product_sum_i32x16),
                                               _mm512_extracti32x8_epi32(state_a->biased_product_sum_i32x16, 1));
    __m256i product_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->biased_product_sum_i32x16),
                                               _mm512_extracti32x8_epi32(state_b->biased_product_sum_i32x16, 1));
    __m256i product_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->biased_product_sum_i32x16),
                                               _mm512_extracti32x8_epi32(state_c->biased_product_sum_i32x16, 1));
    __m256i product_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->biased_product_sum_i32x16),
                                               _mm512_extracti32x8_epi32(state_d->biased_product_sum_i32x16, 1));

    // Reduce ymm (i32x8) → xmm (i32x4)
    __m128i product_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(product_a_i32x8),
                                            _mm256_extracti128_si256(product_a_i32x8, 1));
    __m128i product_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(product_b_i32x8),
                                            _mm256_extracti128_si256(product_b_i32x8, 1));
    __m128i product_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(product_c_i32x8),
                                            _mm256_extracti128_si256(product_c_i32x8, 1));
    __m128i product_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(product_d_i32x8),
                                            _mm256_extracti128_si256(product_d_i32x8, 1));

    // 4-way transpose reduce
    __m128i t_ab_low = _mm_unpacklo_epi32(product_a_i32x4, product_b_i32x4);
    __m128i t_cd_low = _mm_unpacklo_epi32(product_c_i32x4, product_d_i32x4);
    __m128i t_ab_high = _mm_unpackhi_epi32(product_a_i32x4, product_b_i32x4);
    __m128i t_cd_high = _mm_unpackhi_epi32(product_c_i32x4, product_d_i32x4);
    __m128i biased_i32x4 = _mm_add_epi32(
        _mm_add_epi32(_mm_unpacklo_epi64(t_ab_low, t_cd_low), _mm_unpackhi_epi64(t_ab_low, t_cd_low)),
        _mm_add_epi32(_mm_unpacklo_epi64(t_ab_high, t_cd_high), _mm_unpackhi_epi64(t_ab_high, t_cd_high)));

    // Apply compensation: result = biased − 8×(Σa + Σb) − 64×depth_padded
    __m128i a_sum_broadcast_i32x4 = _mm_set1_epi32(a_sum);
    __m128i ab_sums_i32x4 = _mm_add_epi32(a_sum_broadcast_i32x4, b_sums_vec->xmm);
    __m128i correction_i32x4 = _mm_slli_epi32(ab_sums_i32x4, 3); // × 8
    __m128i offset_i32x4 = _mm_set1_epi32((nk_i32_t)(-64LL * (nk_i64_t)depth_nibbles));
    result_vec->xmm = _mm_add_epi32(_mm_sub_epi32(biased_i32x4, correction_i32x4), offset_i32x4);
}

typedef struct nk_dot_u4x128_state_icelake_t {
    __m512i sum_i32x16; // Direct unsigned accumulator
} nk_dot_u4x128_state_icelake_t;

NK_INTERNAL void nk_dot_u4x128_init_icelake(nk_dot_u4x128_state_icelake_t *state) {
    state->sum_i32x16 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_u4x128_update_icelake(nk_dot_u4x128_state_icelake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // u4 values are packed as nibbles: 128 nibbles in 64 bytes (512 bits)
    // Values are ∈ [0,15], so DPBUSD can be used directly
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);

    // Load 64 bytes containing 128 nibbles (full 512-bit register)
    __m512i a_u4x128 = a.zmm;
    __m512i b_u4x128 = b.zmm;

    // Extract low and high nibbles (all 128 nibbles from 64 bytes)
    __m512i a_low_u8x64 = _mm512_and_si512(a_u4x128, nibble_mask_u8x64);
    __m512i a_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_u4x128, 4), nibble_mask_u8x64);
    __m512i b_low_u8x64 = _mm512_and_si512(b_u4x128, nibble_mask_u8x64);
    __m512i b_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_u4x128, 4), nibble_mask_u8x64);

    // DPBUSD works directly for u4 since values are ∈ [0,15]
    state->sum_i32x16 = _mm512_dpbusd_epi32(state->sum_i32x16, a_low_u8x64, b_low_u8x64);
    state->sum_i32x16 = _mm512_dpbusd_epi32(state->sum_i32x16, a_high_u8x64, b_high_u8x64);
}

NK_INTERNAL void nk_dot_u4x128_finalize_icelake(                                                //
    nk_dot_u4x128_state_icelake_t const *state_a, nk_dot_u4x128_state_icelake_t const *state_b, //
    nk_dot_u4x128_state_icelake_t const *state_c, nk_dot_u4x128_state_icelake_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // ILP-optimized 4-way hierarchical reduction for u4 (no correction needed)

    // Reduce zmm (i32x16) → ymm (i32x8)
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->sum_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->sum_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->sum_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->sum_i32x16, 1));

    // Reduce ymm (i32x8) → xmm (i32x4)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));

    // 4-way transpose to get [a,b,c,d] in lanes
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);

    __m128i final_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                        _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
    result->xmm = final_i32x4;
}

NK_PUBLIC void nk_dot_e2m3_icelake(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    // Integer dot product for e2m3 using VPERMB (LUT) + VPDPBUSD (unsigned×signed multiply-add).
    // Every e2m3 value × 16 is an exact integer in [-120, +120].
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // LUT maps 5-bit unsigned magnitude to (value × 16):
    //   exp=0 (sub): 2*mant,         exp=1: 16+2*mant
    //   exp=2:       32+4*mant,       exp=3: 64+8*mant
    //
    // VPERMB uses bits [5:0] of the index, so we need a 64-byte LUT with entries 0-31
    // replicated in the upper 32 bytes (VPERMB indexes mod 64, our indices are 0-31).
    // _mm512_set_epi8 lists bytes HIGH→LOW: byte63, byte62, ..., byte0
    __m512i const lut_magnitude_u8x64 = _mm512_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36,
                                                        32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0,
                                                        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36,
                                                        32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i const magnitude_mask_u8x64 = _mm512_set1_epi8(0x1F);
    __m512i const sign_mask_u8x64 = _mm512_set1_epi8(0x20);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i a_e2m3_u8x64, b_e2m3_u8x64;

nk_dot_e2m3_icelake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_e2m3_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_e2m3_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x64 = _mm512_loadu_si512(a_scalars);
        b_e2m3_u8x64 = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Extract 5-bit magnitude indices
    __m512i a_magnitude_u8x64 = _mm512_and_si512(a_e2m3_u8x64, magnitude_mask_u8x64);
    __m512i b_magnitude_u8x64 = _mm512_and_si512(b_e2m3_u8x64, magnitude_mask_u8x64);

    // VPERMB LUT lookup: unsigned magnitudes × 16
    __m512i a_unsigned_u8x64 = _mm512_permutexvar_epi8(a_magnitude_u8x64, lut_magnitude_u8x64);
    __m512i b_unsigned_u8x64 = _mm512_permutexvar_epi8(b_magnitude_u8x64, lut_magnitude_u8x64);

    // Combined sign: (a ^ b) & 0x20 — nonzero means negative product
    __m512i sign_combined_u8x64 = _mm512_and_si512(_mm512_xor_si512(a_e2m3_u8x64, b_e2m3_u8x64), sign_mask_u8x64);
    __mmask64 negate_mask = _mm512_test_epi8_mask(sign_combined_u8x64, sign_combined_u8x64);

    // Negate b where signs differ: b_signed = negate_mask ? (0 - b_unsigned) : b_unsigned
    // For VPDPBUSD: a=unsigned [0,120], b=signed [-120,+120]
    __m512i b_signed_i8x64 = _mm512_mask_sub_epi8(b_unsigned_u8x64, negate_mask, _mm512_setzero_si512(),
                                                  b_unsigned_u8x64);

    // VPDPBUSD: a_unsigned[unsigned] × b_signed[signed], 4 bytes → i32
    sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, a_unsigned_u8x64, b_signed_i8x64);

    if (count_scalars) goto nk_dot_e2m3_icelake_cycle;
    *result = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16) / 256.0f;
}

NK_PUBLIC void nk_dot_e3m2_icelake(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    // Integer dot product for e3m2 using VPERMW (i16 LUT) + VPMADDWD (i16×i16→i32).
    // Every e3m2 value × 16 is an exact integer, but magnitudes reach 448, requiring i16.
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // 32-entry i16 LUT for magnitude × 16:
    //   exp=0 (sub): mant,            exp=1: 4+mant
    //   exp=2:       8+2*mant,        exp=3: 16+4*mant
    //   exp=4:       32+8*mant,       exp=5: 64+16*mant
    //   exp=6:       128+32*mant,     exp=7: 256+64*mant
    //
    // VPERMW uses bits [4:0] of the index (mod 32), so 32 entries fit exactly in one ZMM.
    // _mm512_set_epi16 lists words HIGH→LOW: word31, word30, ..., word0
    __m512i const lut_magnitude_i16x32 = _mm512_set_epi16(                       //
        448, 384, 320, 256, 224, 192, 160, 128, 112, 96, 80, 64, 56, 48, 40, 32, //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const magnitude_mask_i16x32 = _mm512_set1_epi16(0x1F);
    __m512i const sign_mask_i16x32 = _mm512_set1_epi16(0x20);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m256i a_e3m2_u8x32, b_e3m2_u8x32;

nk_dot_e3m2_icelake_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)count_scalars);
        a_e3m2_u8x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e3m2_u8x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e3m2_u8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_e3m2_u8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Zero-extend u8x32 → u16x32
    __m512i a_u16x32 = _mm512_cvtepu8_epi16(a_e3m2_u8x32);
    __m512i b_u16x32 = _mm512_cvtepu8_epi16(b_e3m2_u8x32);

    // Extract 5-bit magnitude indices
    __m512i a_magnitude_u16x32 = _mm512_and_si512(a_u16x32, magnitude_mask_i16x32);
    __m512i b_magnitude_u16x32 = _mm512_and_si512(b_u16x32, magnitude_mask_i16x32);

    // VPERMW LUT lookup: unsigned magnitudes × 16
    __m512i a_unsigned_i16x32 = _mm512_permutexvar_epi16(a_magnitude_u16x32, lut_magnitude_i16x32);
    __m512i b_unsigned_i16x32 = _mm512_permutexvar_epi16(b_magnitude_u16x32, lut_magnitude_i16x32);

    // Apply signs: negate if bit 5 is set
    __mmask32 a_negate = _mm512_test_epi16_mask(a_u16x32, sign_mask_i16x32);
    __mmask32 b_negate = _mm512_test_epi16_mask(b_u16x32, sign_mask_i16x32);
    __m512i a_signed_i16x32 = _mm512_mask_sub_epi16(a_unsigned_i16x32, a_negate, _mm512_setzero_si512(),
                                                    a_unsigned_i16x32);
    __m512i b_signed_i16x32 = _mm512_mask_sub_epi16(b_unsigned_i16x32, b_negate, _mm512_setzero_si512(),
                                                    b_unsigned_i16x32);

    // VPMADDWD: i16×i16→i32, multiplies adjacent pairs and adds
    sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(a_signed_i16x32, b_signed_i16x32));

    if (count_scalars) goto nk_dot_e3m2_icelake_cycle;
    *result = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16) / 256.0f;
}

#pragma region F16 and BF16 Floats

NK_PUBLIC void nk_dot_e4m3_icelake(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    // E4M3 dot product via octave decomposition + VPDPBUSD integer MAC.
    // Splits 4-bit exponent into 2 octave bits + 2 remainder bits, maps low 5 bits via VPERMB
    // to u8 integers [0, 120], then 16 VPDPBUSD cross-products across 4×4 octave pairs.

    __m512i const lut_normal_u8x64 = _mm512_set_epi8(                      //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32, //
        30, 28, 26, 24, 22, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8,      //
        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32, //
        30, 28, 26, 24, 22, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8);     //
    __m512i const lut_subnorm_u8x64 = _mm512_set_epi8(                     //
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                    //
        0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0,                 //
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                    //
        0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0);                //
    __m512i const magnitude_mask_u8x64 = _mm512_set1_epi8(0x7F);
    __m512i const subnorm_threshold_u8x64 = _mm512_set1_epi8(0x08);
    __m512i const oct_threshold_20_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const oct_threshold_40_u8x64 = _mm512_set1_epi8(0x40);
    __m512i const oct_threshold_60_u8x64 = _mm512_set1_epi8(0x60);

    __m512i dot0_i32x16 = _mm512_setzero_si512();
    __m512i dot1_i32x16 = _mm512_setzero_si512();
    __m512i dot2_i32x16 = _mm512_setzero_si512();
    __m512i dot3_i32x16 = _mm512_setzero_si512();
    __m512i dot4_i32x16 = _mm512_setzero_si512();
    __m512i dot5_i32x16 = _mm512_setzero_si512();
    __m512i dot6_i32x16 = _mm512_setzero_si512();
    __m512i a_e4m3_u8x64, b_e4m3_u8x64;

nk_dot_e4m3_icelake_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_e4m3_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3_u8x64 = _mm512_loadu_si512(a_scalars);
        b_e4m3_u8x64 = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    __m512i a_magnitude_u8x64 = _mm512_and_si512(a_e4m3_u8x64, magnitude_mask_u8x64);
    __m512i b_magnitude_u8x64 = _mm512_and_si512(b_e4m3_u8x64, magnitude_mask_u8x64);
    __m512i a_base_u8x64 = _mm512_permutexvar_epi8(a_magnitude_u8x64, lut_normal_u8x64);
    __m512i b_base_u8x64 = _mm512_permutexvar_epi8(b_magnitude_u8x64, lut_normal_u8x64);

    // Subnormal fixup via VPERMB (avoids VPADDB on Zen 4 ports 8+9 / SPR port 0)
    a_base_u8x64 = _mm512_mask_permutexvar_epi8(a_base_u8x64,
                                                _mm512_cmplt_epu8_mask(a_magnitude_u8x64, subnorm_threshold_u8x64),
                                                a_magnitude_u8x64, lut_subnorm_u8x64);
    b_base_u8x64 = _mm512_mask_permutexvar_epi8(b_base_u8x64,
                                                _mm512_cmplt_epu8_mask(b_magnitude_u8x64, subnorm_threshold_u8x64),
                                                b_magnitude_u8x64, lut_subnorm_u8x64);

    // Sign via ternary logic: (a ^ b) & ~0x7F in one VPTERNLOGD (imm 0x14)
    __m512i sign_diff_u8x64 = _mm512_ternarylogic_epi64(a_e4m3_u8x64, b_e4m3_u8x64, magnitude_mask_u8x64, 0x14);
    __m512i b_signed_i8x64 = _mm512_mask_sub_epi8(b_base_u8x64, _mm512_test_epi8_mask(sign_diff_u8x64, sign_diff_u8x64),
                                                  _mm512_setzero_si512(), b_base_u8x64);

    // Octave masks via cascaded range compares on magnitude
    __mmask64 ka_lt20 = _mm512_cmplt_epu8_mask(a_magnitude_u8x64, oct_threshold_20_u8x64);
    __mmask64 ka_lt40 = _mm512_cmplt_epu8_mask(a_magnitude_u8x64, oct_threshold_40_u8x64);
    __mmask64 ka_lt60 = _mm512_cmplt_epu8_mask(a_magnitude_u8x64, oct_threshold_60_u8x64);
    __mmask64 kb_lt20 = _mm512_cmplt_epu8_mask(b_magnitude_u8x64, oct_threshold_20_u8x64);
    __mmask64 kb_lt40 = _mm512_cmplt_epu8_mask(b_magnitude_u8x64, oct_threshold_40_u8x64);
    __mmask64 kb_lt60 = _mm512_cmplt_epu8_mask(b_magnitude_u8x64, oct_threshold_60_u8x64);

    __m512i a0_u8x64 = _mm512_maskz_mov_epi8(ka_lt20, a_base_u8x64);
    __m512i a1_u8x64 = _mm512_maskz_mov_epi8(ka_lt40 & ~ka_lt20, a_base_u8x64);
    __m512i a2_u8x64 = _mm512_maskz_mov_epi8(ka_lt60 & ~ka_lt40, a_base_u8x64);
    __m512i a3_u8x64 = _mm512_maskz_mov_epi8(~ka_lt60, a_base_u8x64);

    __m512i b0_i8x64 = _mm512_maskz_mov_epi8(kb_lt20, b_signed_i8x64);
    __m512i b1_i8x64 = _mm512_maskz_mov_epi8(kb_lt40 & ~kb_lt20, b_signed_i8x64);
    __m512i b2_i8x64 = _mm512_maskz_mov_epi8(kb_lt60 & ~kb_lt40, b_signed_i8x64);
    __m512i b3_i8x64 = _mm512_maskz_mov_epi8(~kb_lt60, b_signed_i8x64);

    // 16 VPDPBUSD into 7 accumulators grouped by octave sum k = oa + ob
    dot0_i32x16 = _mm512_dpbusd_epi32(dot0_i32x16, a0_u8x64, b0_i8x64);
    dot1_i32x16 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(dot1_i32x16, a0_u8x64, b1_i8x64), a1_u8x64, b0_i8x64);
    dot2_i32x16 = _mm512_dpbusd_epi32(
        _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(dot2_i32x16, a0_u8x64, b2_i8x64), a1_u8x64, b1_i8x64), a2_u8x64,
        b0_i8x64);
    dot3_i32x16 = _mm512_dpbusd_epi32(
        _mm512_dpbusd_epi32(
            _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(dot3_i32x16, a0_u8x64, b3_i8x64), a1_u8x64, b2_i8x64), a2_u8x64,
            b1_i8x64),
        a3_u8x64, b0_i8x64);
    dot4_i32x16 = _mm512_dpbusd_epi32(
        _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(dot4_i32x16, a1_u8x64, b3_i8x64), a2_u8x64, b2_i8x64), a3_u8x64,
        b1_i8x64);
    dot5_i32x16 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(dot5_i32x16, a2_u8x64, b3_i8x64), a3_u8x64, b2_i8x64);
    dot6_i32x16 = _mm512_dpbusd_epi32(dot6_i32x16, a3_u8x64, b3_i8x64);

    if (count_scalars) goto nk_dot_e4m3_icelake_cycle;

    __m512 sum_f32x16 = _mm512_mul_ps(_mm512_cvtepi32_ps(dot0_i32x16), _mm512_set1_ps(9.5367431640625e-07f));
    sum_f32x16 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot1_i32x16), _mm512_set1_ps(1.52587890625e-05f), sum_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot2_i32x16), _mm512_set1_ps(2.44140625e-04f), sum_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot3_i32x16), _mm512_set1_ps(3.90625e-03f), sum_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot4_i32x16), _mm512_set1_ps(6.25e-02f), sum_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot5_i32x16), _mm512_set1_ps(1.0f), sum_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot6_i32x16), _mm512_set1_ps(16.0f), sum_f32x16);
    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

#pragma endregion F16 and BF16 Floats

#pragma region Binary

NK_PUBLIC void nk_dot_u1_icelake(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n_bits, NK_BITS_PER_BYTE);
    __m512i and_popcount_u64x8 = _mm512_setzero_si512();
    __m512i a_u8x64, b_u8x64;

nk_dot_u1_icelake_cycle:
    if (n_bytes < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        n_bytes = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_epi8(a);
        b_u8x64 = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_bytes -= 64;
    }
    and_popcount_u64x8 = _mm512_add_epi64(and_popcount_u64x8, _mm512_popcnt_epi64(_mm512_and_si512(a_u8x64, b_u8x64)));
    if (n_bytes) goto nk_dot_u1_icelake_cycle;

    *result = (nk_u32_t)_mm512_reduce_add_epi64(and_popcount_u64x8);
}

typedef struct nk_dot_u1x512_state_icelake_t {
    __m512i dot_count_i64x8;
} nk_dot_u1x512_state_icelake_t;

NK_INTERNAL void nk_dot_u1x512_init_icelake(nk_dot_u1x512_state_icelake_t *state) {
    state->dot_count_i64x8 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_dot_u1x512_update_icelake(nk_dot_u1x512_state_icelake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->dot_count_i64x8 = _mm512_add_epi64(state->dot_count_i64x8,
                                              _mm512_popcnt_epi64(_mm512_and_si512(a.zmm, b.zmm)));
}

NK_INTERNAL void nk_dot_u1x512_finalize_icelake( //
    nk_dot_u1x512_state_icelake_t const *state_a, nk_dot_u1x512_state_icelake_t const *state_b,
    nk_dot_u1x512_state_icelake_t const *state_c, nk_dot_u1x512_state_icelake_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    // VPMOVQD: truncate 8×i64 → 8×i32 per state
    __m256i a_i32x8 = _mm512_cvtepi64_epi32(state_a->dot_count_i64x8);
    __m256i b_i32x8 = _mm512_cvtepi64_epi32(state_b->dot_count_i64x8);
    __m256i c_i32x8 = _mm512_cvtepi64_epi32(state_c->dot_count_i64x8);
    __m256i d_i32x8 = _mm512_cvtepi64_epi32(state_d->dot_count_i64x8);

    // Fold 8×i32 → 4×i32 (add high 128-bit lane to low)
    __m128i a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(a_i32x8), _mm256_extracti128_si256(a_i32x8, 1));
    __m128i b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(b_i32x8), _mm256_extracti128_si256(b_i32x8, 1));
    __m128i c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(c_i32x8), _mm256_extracti128_si256(c_i32x8, 1));
    __m128i d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(d_i32x8), _mm256_extracti128_si256(d_i32x8, 1));

    // VPHADDD cascade: 4×i32 → 2×i32 → 1×i32 per state
    __m128i ab_i32x4 = _mm_hadd_epi32(a_i32x4, b_i32x4);
    __m128i cd_i32x4 = _mm_hadd_epi32(c_i32x4, d_i32x4);
    result->xmm = _mm_hadd_epi32(ab_i32x4, cd_i32x4);
}

#pragma endregion Binary

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X8664_
#endif // NK_DOT_ICELAKE_H
