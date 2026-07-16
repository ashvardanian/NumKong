/**
 *  @brief SIMD-accelerated Dot Products for Diamond Rapids.
 *  @file include/numkong/dot/diamond.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_diamond_instructions Key AVX10.2 FP8 + FP16 VNNI Instructions
 *
 *      Intrinsic            Instruction                   Diamond Rapids
 *      _mm512_cvthf8_ph     VCVTHF82PH (ZMM, YMM)         ~3cy (estimated)
 *      _mm512_cvtbf8_ph     VCVTBF82PH (ZMM, YMM)         ~3cy (estimated)
 *      _mm512_dpph_ps       VDPPHPS (ZMM, ZMM, ZMM)       ~6cy (estimated)
 *
 *  Diamond Rapids (AVX10.2) introduces native FP8→FP16 conversion via VCVTHF82PH (E4M3→FP16)
 *  and VCVTBF82PH (E5M2→FP16), replacing the multi-instruction arithmetic conversion used by
 *  Genoa's BF16 path. VDPPHPS then computes two FP16 dot products per 32-bit lane, accumulating
 *  into FP32 — providing the same 32-element throughput as Genoa's VDPBF16PS but with FP16
 *  intermediate precision (10-bit mantissa vs BF16's 7-bit).
 *
 *  @section dot_diamond_stateful Stateful Streaming Logic
 *
 *  Defines stateful init/update/finalize helpers for tiled GEMM via the dots/ macros:
 *  - nk_dot_through_f16_state_diamond_t_ shared by both E4M3 and E5M2 (FP16→VDPPHPS→FP32)
 */
#ifndef NK_DOT_DIAMOND_H
#define NK_DOT_DIAMOND_H

#if NK_TARGET_X8664_
#if NK_TARGET_DIAMOND

#include "numkong/types.h"
#include "numkong/cast/diamond.h"   // `nk_load_e4m3x32_to_f16x32_diamond_`
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`
#include "numkong/dot/skylake.h"    // `nk_dot_through_f32_finalize_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                    \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2-512,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx10.2-512", "f16c", "fma", \
                   "bmi", "bmi2")
#endif

NK_API_COMPTIME void nk_dot_e4m3_diamond(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    __m256i a_e4m3x32, b_e4m3x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_e4m3_diamond_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_e4m3x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x32 = _mm256_loadu_epi8(a_scalars);
        b_e4m3x32 = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    __m512h a_f16x32 = _mm512_cvthf8_ph(a_e4m3x32);
    __m512h b_f16x32 = _mm512_cvthf8_ph(b_e4m3x32);
    sum_f32x16 = _mm512_dpph_ps(sum_f32x16, a_f16x32, b_f16x32);
    if (count_scalars) goto nk_dot_e4m3_diamond_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_API_COMPTIME void nk_dot_e5m2_diamond(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    __m256i a_e5m2x32, b_e5m2x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_e5m2_diamond_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_e5m2x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e5m2x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e5m2x32 = _mm256_loadu_epi8(a_scalars);
        b_e5m2x32 = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    __m512h a_f16x32 = _mm512_cvtbf8_ph(a_e5m2x32);
    __m512h b_f16x32 = _mm512_cvtbf8_ph(b_e5m2x32);
    sum_f32x16 = _mm512_dpph_ps(sum_f32x16, a_f16x32, b_f16x32);
    if (count_scalars) goto nk_dot_e5m2_diamond_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_API_COMPTIME void nk_dot_f16_diamond(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f32_t *result) {
    __m512h a_f16x32, b_f16x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_f16_diamond_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a_scalars));
        b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b_scalars));
        count_scalars = 0;
    }
    else {
        a_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(a_scalars));
        b_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(b_scalars));
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    sum_f32x16 = _mm512_dpph_ps(sum_f32x16, a_f16x32, b_f16x32);
    if (count_scalars) goto nk_dot_f16_diamond_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

typedef nk_dot_through_f32_state_skylake_t_ nk_dot_through_f16_state_diamond_t_;

NK_HELPER_INLINE void nk_dot_through_f16_init_diamond_(nk_dot_through_f16_state_diamond_t_ *state) {
    state->sum_f32x16 = _mm512_setzero();
}

NK_HELPER_INLINE void nk_dot_through_f16_update_diamond_(nk_dot_through_f16_state_diamond_t_ *state, nk_b512_vec_t a,
                                                         nk_b512_vec_t b, nk_size_t depth_offset,
                                                         nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->sum_f32x16 = _mm512_dpph_ps(state->sum_f32x16, nk_m512h_from_m512i_(a.zmm), nk_m512h_from_m512i_(b.zmm));
}

NK_HELPER_INLINE void nk_dot_through_f16_finalize_diamond_(                                                 //
    nk_dot_through_f16_state_diamond_t_ const *state_a, nk_dot_through_f16_state_diamond_t_ const *state_b, //
    nk_dot_through_f16_state_diamond_t_ const *state_c, nk_dot_through_f16_state_diamond_t_ const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_dot_through_f32_finalize_skylake_(state_a, state_b, state_c, state_d, total_dimensions, result);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_DIAMOND
#endif // NK_TARGET_X8664_
#endif // NK_DOT_DIAMOND_H
