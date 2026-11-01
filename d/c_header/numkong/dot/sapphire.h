/**
 *  @brief SIMD-accelerated Dot Products for Sapphire Rapids.
 *  @file include/numkong/dot/sapphire.h
 *  @author Ash Vardanian
 *  @date February 7, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_sapphire_instructions Key AVX-512 FP16 Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm512_fmadd_ph             VFMADDPH (ZMM, ZMM, ZMM)        4cy         0.5/cy      p01
 *      _mm512_fmadd_ps             VFMADD132PS (ZMM, ZMM, ZMM)     4cy         0.5/cy      p01
 *      _mm512_cvtph_ps             VCVTPH2PS (ZMM, YMM)            7cy         1/cy        p01
 *
 *  Sapphire Rapids introduces native AVX-512 FP16 support, enabling 32 FP16 FMAs per instruction at the same
 *  throughput as 16 FP32 FMAs — effectively 2x compute density. For FP6 types (E2M3 and E3M2) whose products
 *  are small enough to accumulate safely in FP16, this provides near-2x speedup over the Genoa BF16 path.
 *
 *  @section dot_sapphire_accumulation Safe FP16 Accumulation
 *
 *  E2M3 max product: 7.5² = 56.25; flush every 4 iterations → max lane sum ~225, FP16 ULP ~0.125.
 *  E3M2 max product: 28² = 784; flush every 4 iterations → max lane sum ~3136, FP16 ULP ~2.0.
 *  After the flush window, we widen the FP16 accumulator to FP32 and reset.
 *
 *  @section dot_sapphire_stateful Stateful Streaming Logic
 *
 *  Typed wrappers control the flush cadence:
 *  - nk_dot_e2m3x32_state_sapphire_t flushes every 4 iterations (128 elements)
 *  - nk_dot_e3m2x32_state_sapphire_t flushes every 4 iterations (128 elements)
 */
#ifndef NK_DOT_SAPPHIRE_H
#define NK_DOT_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE

#include "numkong/types.h"
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`
#include "numkong/dot/skylake.h"    // `nk_dot_through_f32_finalize_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

/** @brief Convert 32x e2m3 → 32x f16 via 64-entry signed LUT lookup (AVX-512BW).
 *  E2M3 format: S EE MMM (bias=1, 6 bits total: sign at bit 5, magnitude bits 4-0).
 *  F16: S EEEEE MMMMMMMMMM (bias=15).
 *
 *  Uses permutex2var with two 32-entry LUTs (positive and negative F16 values).
 *  The E2M3 sign bit (bit 5) naturally becomes the source-select bit of the 6-bit index,
 *  so no separate sign extraction, shift, or OR is needed. After cvtepu8_epi16, bits 15:6
 *  are zero and permutex2var only reads bits 5:0, so no AND mask is required either. */
NK_INTERNAL __m512h nk_e2m3x32_to_f16x32_sapphire_(__m256i e2m3x32) {
    __m512i idx_i16x32 = _mm512_cvtepu8_epi16(e2m3x32);

    // 32-entry LUT for positive E2M3 magnitudes → F16
    __m512i const lut_pos_i16x32 = _mm512_set_epi16(                     //
        0x4780, 0x4700, 0x4680, 0x4600, 0x4580, 0x4500, 0x4480, 0x4400,  // [31-24] exp=3: f16_exp=17
        0x4380, 0x4300, 0x4280, 0x4200, 0x4180, 0x4100, 0x4080, 0x4000,  // [23-16] exp=2: f16_exp=16
        0x3F80, 0x3F00, 0x3E80, 0x3E00, 0x3D80, 0x3D00, 0x3C80, 0x3C00,  // [15-8] exp=1: f16_exp=15
        0x3B00, 0x3A00, 0x3900, 0x3800, 0x3600, 0x3400, 0x3000, 0x0000); // [7-0] exp=0: subnormals (0, 1/8..7/8)

    // 32-entry LUT for negative E2M3 magnitudes → F16 (= positive | 0x8000)
    __m512i const lut_neg_i16x32 = _mm512_set_epi16(                 //
        (short)0xC780, (short)0xC700, (short)0xC680, (short)0xC600,  //
        (short)0xC580, (short)0xC500, (short)0xC480, (short)0xC400,  // [31-24] exp=3
        (short)0xC380, (short)0xC300, (short)0xC280, (short)0xC200,  //
        (short)0xC180, (short)0xC100, (short)0xC080, (short)0xC000,  // [23-16] exp=2
        (short)0xBF80, (short)0xBF00, (short)0xBE80, (short)0xBE00,  //
        (short)0xBD80, (short)0xBD00, (short)0xBC80, (short)0xBC00,  // [15-8] exp=1
        (short)0xBB00, (short)0xBA00, (short)0xB900, (short)0xB800,  //
        (short)0xB600, (short)0xB400, (short)0xB000, (short)0x8000); // [7-0] exp=0

    return nk_m512h_from_m512i_(_mm512_permutex2var_epi16(lut_pos_i16x32, idx_i16x32, lut_neg_i16x32));
}

/** @brief Convert 32x e3m2 → 32x f16 via 64-entry signed LUT lookup (AVX-512BW).
 *  E3M2 format: S EEE MM (bias=3, 6 bits total: sign at bit 5, magnitude bits 4-0).
 *  F16: S EEEEE MMMMMMMMMM (bias=15).
 *
 *  Same permutex2var technique as E2M3 — sign bit 5 selects the LUT source. */
NK_INTERNAL __m512h nk_e3m2x32_to_f16x32_sapphire_(__m256i e3m2x32) {
    __m512i idx_i16x32 = _mm512_cvtepu8_epi16(e3m2x32);

    // 32-entry LUT for positive E3M2 magnitudes → F16
    __m512i const lut_pos_i16x32 = _mm512_set_epi16( //
        0x4F00, 0x4E00, 0x4D00, 0x4C00,              // [31-28] exp=7: f16_exp=19
        0x4B00, 0x4A00, 0x4900, 0x4800,              // [27-24] exp=6: f16_exp=18
        0x4700, 0x4600, 0x4500, 0x4400,              // [23-20] exp=5: f16_exp=17
        0x4300, 0x4200, 0x4100, 0x4000,              // [19-16] exp=4: f16_exp=16
        0x3F00, 0x3E00, 0x3D00, 0x3C00,              // [15-12] exp=3: f16_exp=15
        0x3B00, 0x3A00, 0x3900, 0x3800,              // [11-8] exp=2: f16_exp=14
        0x3700, 0x3600, 0x3500, 0x3400,              // [7-4] exp=1: f16_exp=13
        0x3200, 0x3000, 0x2C00, 0x0000);             // [3-0] exp=0: subnormals

    // 32-entry LUT for negative E3M2 magnitudes → F16 (= positive | 0x8000)
    __m512i const lut_neg_i16x32 = _mm512_set_epi16(                 //
        (short)0xCF00, (short)0xCE00, (short)0xCD00, (short)0xCC00,  // [31-28] exp=7
        (short)0xCB00, (short)0xCA00, (short)0xC900, (short)0xC800,  // [27-24] exp=6
        (short)0xC700, (short)0xC600, (short)0xC500, (short)0xC400,  // [23-20] exp=5
        (short)0xC300, (short)0xC200, (short)0xC100, (short)0xC000,  // [19-16] exp=4
        (short)0xBF00, (short)0xBE00, (short)0xBD00, (short)0xBC00,  // [15-12] exp=3
        (short)0xBB00, (short)0xBA00, (short)0xB900, (short)0xB800,  // [11-8] exp=2
        (short)0xB700, (short)0xB600, (short)0xB500, (short)0xB400,  // [7-4] exp=1
        (short)0xB200, (short)0xB000, (short)0xAC00, (short)0x8000); // [3-0] exp=0

    return nk_m512h_from_m512i_(_mm512_permutex2var_epi16(lut_pos_i16x32, idx_i16x32, lut_neg_i16x32));
}

/** @brief Flush 32 FP16 values to FP32 accumulator by splitting into 2x16 halves. */
NK_INTERNAL __m512 nk_flush_f16_to_f32_sapphire_(__m512h acc_f16x32, __m512 sum_f32x16) {
    __m256i low_f16x16 = _mm512_castsi512_si256(nk_m512i_from_m512h_(acc_f16x32));
    __m256i high_f16x16 = _mm512_extracti64x4_epi64(nk_m512i_from_m512h_(acc_f16x32), 1);
    sum_f32x16 = _mm512_add_ps(sum_f32x16, _mm512_cvtph_ps(low_f16x16));
    sum_f32x16 = _mm512_add_ps(sum_f32x16, _mm512_cvtph_ps(high_f16x16));
    return sum_f32x16;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_
#endif // NK_DOT_SAPPHIRE_H
