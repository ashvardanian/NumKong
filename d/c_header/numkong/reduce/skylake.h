/**
 *  @brief AVX-512 implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/skylake.h
 *  @author Ash Vardanian
 *  @date February 11, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section tail_nan_fill  Tail Handling via NaN Fill
 *
 *  In floating-point minmax contiguous kernels (f32, f64), the tail block fills
 *  unloaded lanes with NaN via `_mm512_mask_loadu_ps(nan, mask, ptr)` instead of
 *  `_mm512_maskz_loadu_ps(mask, ptr)`.  This allows the subsequent `_CMP_LT_OQ` /
 *  `_CMP_GT_OQ` comparisons to run without the tail-load mask predicate, because
 *  IEEE-754 ordered-quiet comparisons return false for NaN operands.
 *
 *  @section reduce_block_caps Block-Cap Overflow Thresholds
 *
 *  Dispatch functions use pairwise recursion when `count` exceeds a block cap.
 *  The cap is sized so the iteration counter in the contiguous kernel never wraps.
 *
 *  Iteration counters start at 0 (initial load) and increment by 1 per SIMD chunk.
 *  A u8 counter holds 0..255 → 256 iterations → processes 256 × lanes elements.
 *  A u16 counter holds 0..65535 → 65536 iterations → processes 65536 × lanes elements.
 *  A u32 counter holds 0..4294967295 → ~4.3 billion iterations.
 *
 *  Threshold formula: count > (COUNTER_MAX + 1) × lanes_per_chunk
 *    - u8 minmax:  (NK_U8_MAX  + 1) × lanes   (e.g. 256 × 64 = 16384 for i8x64)
 *    - u16 minmax: (NK_U16_MAX + 1) × lanes   (e.g. 65536 × 32 = 2097152 for i16x32)
 *    - u32 minmax: NK_U32_MAX × lanes          (no +1: NK_U32_MAX + 1 overflows unsigned)
 *
 *  Moments block caps are sized for accumulator overflow, not counter overflow.
 *  See individual dispatch functions for type-specific derivations.
 */
#ifndef NK_REDUCE_SKYLAKE_H
#define NK_REDUCE_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/cast/skylake.h"
#include "numkong/reduce/serial.h"

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

/** @brief Horizontal sum of 16 floats in a ZMM register (native f32 precision). */
NK_INTERNAL nk_f32_t nk_reduce_add_f32x16_skylake_(__m512 sum_f32x16) {
    __m256 lo_f32x8 = _mm512_castps512_ps256(sum_f32x16);
    __m256 hi_f32x8 = _mm512_extractf32x8_ps(sum_f32x16, 1);
    __m256 sum_f32x8 = _mm256_add_ps(lo_f32x8, hi_f32x8);
    __m128 lo_f32x4 = _mm256_castps256_ps128(sum_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(sum_f32x8, 1);
    __m128 sum_f32x4 = _mm_add_ps(lo_f32x4, hi_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    return _mm_cvtss_f32(sum_f32x4);
}

/** @brief Horizontal sum of 8 doubles in a ZMM register. */
NK_INTERNAL nk_f64_t nk_reduce_add_f64x8_skylake_(__m512d sum_f64x8) {
    __m256d lo_f64x4 = _mm512_castpd512_pd256(sum_f64x8);
    __m256d hi_f64x4 = _mm512_extractf64x4_pd(sum_f64x8, 1);
    __m256d sum_f64x4 = _mm256_add_pd(lo_f64x4, hi_f64x4);
    __m128d lo_f64x2 = _mm256_castpd256_pd128(sum_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(sum_f64x4, 1);
    __m128d sum_f64x2 = _mm_add_pd(lo_f64x2, hi_f64x2);
    sum_f64x2 = _mm_hadd_pd(sum_f64x2, sum_f64x2);
    return _mm_cvtsd_f64(sum_f64x2);
}

/** @brief Horizontal min of 16 floats in a ZMM register. */
NK_INTERNAL nk_f32_t nk_reduce_min_f32x16_skylake_(__m512 min_f32x16) {
    __m256 lo_f32x8 = _mm512_castps512_ps256(min_f32x16);
    __m256 hi_f32x8 = _mm512_extractf32x8_ps(min_f32x16, 1);
    __m256 min_f32x8 = _mm256_min_ps(lo_f32x8, hi_f32x8);
    __m128 lo_f32x4 = _mm256_castps256_ps128(min_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(min_f32x8, 1);
    __m128 min_f32x4 = _mm_min_ps(lo_f32x4, hi_f32x4);
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(min_f32x4);
}

/** @brief Horizontal max of 16 floats in a ZMM register. */
NK_INTERNAL nk_f32_t nk_reduce_max_f32x16_skylake_(__m512 max_f32x16) {
    __m256 lo_f32x8 = _mm512_castps512_ps256(max_f32x16);
    __m256 hi_f32x8 = _mm512_extractf32x8_ps(max_f32x16, 1);
    __m256 max_f32x8 = _mm256_max_ps(lo_f32x8, hi_f32x8);
    __m128 lo_f32x4 = _mm256_castps256_ps128(max_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(max_f32x8, 1);
    __m128 max_f32x4 = _mm_max_ps(lo_f32x4, hi_f32x4);
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max_f32x4);
}

/** @brief Horizontal min of 8 doubles in a ZMM register. */
NK_INTERNAL nk_f64_t nk_reduce_min_f64x8_skylake_(__m512d min_f64x8) {
    __m256d lo_f64x4 = _mm512_castpd512_pd256(min_f64x8);
    __m256d hi_f64x4 = _mm512_extractf64x4_pd(min_f64x8, 1);
    __m256d min_f64x4 = _mm256_min_pd(lo_f64x4, hi_f64x4);
    __m128d lo_f64x2 = _mm256_castpd256_pd128(min_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(min_f64x4, 1);
    __m128d min_f64x2 = _mm_min_pd(lo_f64x2, hi_f64x2);
    min_f64x2 = _mm_min_pd(min_f64x2, _mm_shuffle_pd(min_f64x2, min_f64x2, 1));
    return _mm_cvtsd_f64(min_f64x2);
}

/** @brief Horizontal max of 8 doubles in a ZMM register. */
NK_INTERNAL nk_f64_t nk_reduce_max_f64x8_skylake_(__m512d max_f64x8) {
    __m256d lo_f64x4 = _mm512_castpd512_pd256(max_f64x8);
    __m256d hi_f64x4 = _mm512_extractf64x4_pd(max_f64x8, 1);
    __m256d max_f64x4 = _mm256_max_pd(lo_f64x4, hi_f64x4);
    __m128d lo_f64x2 = _mm256_castpd256_pd128(max_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(max_f64x4, 1);
    __m128d max_f64x2 = _mm_max_pd(lo_f64x2, hi_f64x2);
    max_f64x2 = _mm_max_pd(max_f64x2, _mm_shuffle_pd(max_f64x2, max_f64x2, 1));
    return _mm_cvtsd_f64(max_f64x2);
}

/** @brief Horizontal sum of 16 i32s in a ZMM register. */
NK_INTERNAL nk_i32_t nk_reduce_add_i32x16_skylake_(__m512i sum_i32x16) {
    __m256i lo_i32x8 = _mm512_castsi512_si256(sum_i32x16);
    __m256i hi_i32x8 = _mm512_extracti32x8_epi32(sum_i32x16, 1);
    __m256i sum_i32x8 = _mm256_add_epi32(lo_i32x8, hi_i32x8);
    __m128i lo_i32x4 = _mm256_castsi256_si128(sum_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(sum_i32x8, 1);
    __m128i sum_i32x4 = _mm_add_epi32(lo_i32x4, hi_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    return _mm_cvtsi128_si32(sum_i32x4);
}

/** @brief Horizontal sum of 8 i64s in a ZMM register. */
NK_INTERNAL nk_i64_t nk_reduce_add_i64x8_skylake_(__m512i sum_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(sum_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(sum_i64x8, 1);
    __m256i sum_i64x4 = _mm256_add_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(sum_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(sum_i64x4, 1);
    __m128i sum_i64x2 = _mm_add_epi64(lo_i64x2, hi_i64x2);
    sum_i64x2 = _mm_add_epi64(sum_i64x2, _mm_shuffle_epi32(sum_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si64(sum_i64x2);
}

/**
 *  @brief Returns AVX-512 mask for strided access of 8-bit elements (64-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  With 64 elements per register, useful for strides 2-16 (yielding 4+ elements per load).
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask64 nk_stride_mask_u1x64_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask64)0x5555555555555555ull;  // 32 elems
    case 3: return (__mmask64)0x9249249249249249ull;  // 22 elems
    case 4: return (__mmask64)0x1111111111111111ull;  // 16 elems
    case 5: return (__mmask64)0x1084210842108421ull;  // 13 elems
    case 6: return (__mmask64)0x1041041041041041ull;  // 11 elems
    case 7: return (__mmask64)0x0102040810204081ull;  // 9 elems
    case 8: return (__mmask64)0x0101010101010101ull;  // 8 elems
    case 9: return (__mmask64)0x8040201008040201ull;  // 8 elems
    case 10: return (__mmask64)0x1004010040100401ull; // 7 elems
    case 11: return (__mmask64)0x0080100200400801ull; // 6 elems
    case 12: return (__mmask64)0x1001001001001001ull; // 6 elems
    case 13: return (__mmask64)0x0010008004002001ull; // 5 elems
    case 14: return (__mmask64)0x0100040010004001ull; // 5 elems
    case 15: return (__mmask64)0x1000200040008001ull; // 5 elems
    case 16: return (__mmask64)0x0001000100010001ull; // 4 elems
    default: return (__mmask64)0;
    }
}

/**
 *  @brief Returns AVX-512 mask for strided access of 32-bit elements (16-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Example: stride 4 extracts column 0 from a 4-column matrix.
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask16 nk_stride_mask_b32x16_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask16)0x5555; // 8 elems
    case 3: return (__mmask16)0x9249; // 6 elems
    case 4: return (__mmask16)0x1111; // 4 elems
    case 5: return (__mmask16)0x8421; // 4 elems
    case 6: return (__mmask16)0x1041; // 3 elems
    case 7: return (__mmask16)0x4081; // 3 elems
    case 8: return (__mmask16)0x0101; // 2 elems
    default: return (__mmask16)0;     // Invalid stride - caller should use gather or serial
    }
}

/**
 *  @brief Returns AVX-512 mask for strided access of 16-bit elements (32-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Example: stride 4 extracts column 0 from a 4-column int16 matrix.
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask32 nk_stride_mask_b16x32_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask32)0x55555555;  // 16 elems
    case 3: return (__mmask32)0x49249249;  // 11 elems
    case 4: return (__mmask32)0x11111111;  // 8 elems
    case 5: return (__mmask32)0x42108421;  // 7 elems
    case 6: return (__mmask32)0x41041041;  // 6 elems
    case 7: return (__mmask32)0x10204081;  // 5 elems
    case 8: return (__mmask32)0x01010101;  // 4 elems
    case 9: return (__mmask32)0x08040201;  // 4 elems
    case 10: return (__mmask32)0x40100401; // 4 elems
    case 11: return (__mmask32)0x00400801; // 3 elems
    case 12: return (__mmask32)0x01001001; // 3 elems
    case 13: return (__mmask32)0x04002001; // 3 elems
    case 14: return (__mmask32)0x10004001; // 3 elems
    case 15: return (__mmask32)0x40008001; // 3 elems
    case 16: return (__mmask32)0x00010001; // 2 elems
    default: return (__mmask32)0;
    }
}

/**
 *  @brief Returns AVX-512 mask for strided access of 64-bit elements (8-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Example: stride 4 extracts column 0 from a 4-column matrix.
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask8 nk_stride_mask_b64x8_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask8)0x55; // [1,0,1,0,1,0,1,0] → 4 elems
    case 3: return (__mmask8)0x49; // [1,0,0,1,0,0,1,0] → 3 elems
    case 4: return (__mmask8)0x11; // [1,0,0,0,1,0,0,0] → 2 elems
    case 5: return (__mmask8)0x21; // [1,0,0,0,0,1,0,0] → 2 elems
    case 6: return (__mmask8)0x41; // [1,0,0,0,0,0,1,0] → 2 elems
    case 7: return (__mmask8)0x01; // [1,0,0,0,0,0,0,0] → 1 elem
    case 8: return (__mmask8)0x01; // [1,0,0,0,0,0,0,0] → 1 elem
    default: return (__mmask8)0;
    }
}

/**
 *  @brief Returns number of logical elements per 16-scalar chunk for given stride.
 */
NK_INTERNAL nk_size_t nk_stride_elems_b32x16_(nk_size_t stride) {
    switch (stride) {
    case 2: return 8;
    case 3: return 6;
    case 4: return 4;
    case 5: return 4;
    case 6: return 3;
    case 7: return 3;
    case 8: return 2;
    default: return 0;
    }
}

/**
 *  @brief Returns number of logical elements per 8-scalar chunk for given stride.
 */
NK_INTERNAL nk_size_t nk_stride_elems_b64x8_(nk_size_t stride) {
    switch (stride) {
    case 2: return 4;
    case 3: return 3;
    case 4: return 2;
    case 5: return 2;
    case 6: return 2;
    case 7: return 1;
    case 8: return 1;
    default: return 0;
    }
}

/** @brief Horizontal min of 64 signed i8s in a ZMM register. */
NK_INTERNAL nk_i8_t nk_reduce_min_i8x64_skylake_(__m512i min_i8x64) {
    __m256i lo_i8x32 = _mm512_castsi512_si256(min_i8x64);
    __m256i hi_i8x32 = _mm512_extracti64x4_epi64(min_i8x64, 1);
    __m256i min_i8x32 = _mm256_min_epi8(lo_i8x32, hi_i8x32);
    __m128i lo_i8x16 = _mm256_castsi256_si128(min_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(min_i8x32, 1);
    __m128i min_i8x16 = _mm_min_epi8(lo_i8x16, hi_i8x16);
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_srli_si128(min_i8x16, 2));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_srli_si128(min_i8x16, 1));
    return (nk_i8_t)_mm_cvtsi128_si32(min_i8x16);
}

/** @brief Horizontal max of 64 signed i8s in a ZMM register. */
NK_INTERNAL nk_i8_t nk_reduce_max_i8x64_skylake_(__m512i max_i8x64) {
    __m256i lo_i8x32 = _mm512_castsi512_si256(max_i8x64);
    __m256i hi_i8x32 = _mm512_extracti64x4_epi64(max_i8x64, 1);
    __m256i max_i8x32 = _mm256_max_epi8(lo_i8x32, hi_i8x32);
    __m128i lo_i8x16 = _mm256_castsi256_si128(max_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(max_i8x32, 1);
    __m128i max_i8x16 = _mm_max_epi8(lo_i8x16, hi_i8x16);
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_srli_si128(max_i8x16, 2));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_srli_si128(max_i8x16, 1));
    return (nk_i8_t)_mm_cvtsi128_si32(max_i8x16);
}

/** @brief Horizontal min of 64 unsigned u8s in a ZMM register. */
NK_INTERNAL nk_u8_t nk_reduce_min_u8x64_skylake_(__m512i min_u8x64) {
    __m256i lo_u8x32 = _mm512_castsi512_si256(min_u8x64);
    __m256i hi_u8x32 = _mm512_extracti64x4_epi64(min_u8x64, 1);
    __m256i min_u8x32 = _mm256_min_epu8(lo_u8x32, hi_u8x32);
    __m128i lo_u8x16 = _mm256_castsi256_si128(min_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(min_u8x32, 1);
    __m128i min_u8x16 = _mm_min_epu8(lo_u8x16, hi_u8x16);
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_srli_si128(min_u8x16, 2));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_srli_si128(min_u8x16, 1));
    return (nk_u8_t)_mm_cvtsi128_si32(min_u8x16);
}

/** @brief Horizontal max of 64 unsigned u8s in a ZMM register. */
NK_INTERNAL nk_u8_t nk_reduce_max_u8x64_skylake_(__m512i max_u8x64) {
    __m256i lo_u8x32 = _mm512_castsi512_si256(max_u8x64);
    __m256i hi_u8x32 = _mm512_extracti64x4_epi64(max_u8x64, 1);
    __m256i max_u8x32 = _mm256_max_epu8(lo_u8x32, hi_u8x32);
    __m128i lo_u8x16 = _mm256_castsi256_si128(max_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(max_u8x32, 1);
    __m128i max_u8x16 = _mm_max_epu8(lo_u8x16, hi_u8x16);
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_srli_si128(max_u8x16, 2));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_srli_si128(max_u8x16, 1));
    return (nk_u8_t)_mm_cvtsi128_si32(max_u8x16);
}

/** @brief Horizontal min of 32 signed i16s in a ZMM register. */
NK_INTERNAL nk_i16_t nk_reduce_min_i16x32_skylake_(__m512i min_i16x32) {
    __m256i lo_i16x16 = _mm512_castsi512_si256(min_i16x32);
    __m256i hi_i16x16 = _mm512_extracti64x4_epi64(min_i16x32, 1);
    __m256i min_i16x16 = _mm256_min_epi16(lo_i16x16, hi_i16x16);
    __m128i lo_i16x8 = _mm256_castsi256_si128(min_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(min_i16x16, 1);
    __m128i min_i16x8 = _mm_min_epi16(lo_i16x8, hi_i16x8);
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_srli_si128(min_i16x8, 2));
    return (nk_i16_t)_mm_cvtsi128_si32(min_i16x8);
}

/** @brief Horizontal max of 32 signed i16s in a ZMM register. */
NK_INTERNAL nk_i16_t nk_reduce_max_i16x32_skylake_(__m512i max_i16x32) {
    __m256i lo_i16x16 = _mm512_castsi512_si256(max_i16x32);
    __m256i hi_i16x16 = _mm512_extracti64x4_epi64(max_i16x32, 1);
    __m256i max_i16x16 = _mm256_max_epi16(lo_i16x16, hi_i16x16);
    __m128i lo_i16x8 = _mm256_castsi256_si128(max_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(max_i16x16, 1);
    __m128i max_i16x8 = _mm_max_epi16(lo_i16x8, hi_i16x8);
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_srli_si128(max_i16x8, 2));
    return (nk_i16_t)_mm_cvtsi128_si32(max_i16x8);
}

/** @brief Horizontal min of 32 unsigned u16s in a ZMM register. */
NK_INTERNAL nk_u16_t nk_reduce_min_u16x32_skylake_(__m512i min_u16x32) {
    __m256i lo_u16x16 = _mm512_castsi512_si256(min_u16x32);
    __m256i hi_u16x16 = _mm512_extracti64x4_epi64(min_u16x32, 1);
    __m256i min_u16x16 = _mm256_min_epu16(lo_u16x16, hi_u16x16);
    __m128i lo_u16x8 = _mm256_castsi256_si128(min_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(min_u16x16, 1);
    __m128i min_u16x8 = _mm_min_epu16(lo_u16x8, hi_u16x8);
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_srli_si128(min_u16x8, 2));
    return (nk_u16_t)_mm_cvtsi128_si32(min_u16x8);
}

/** @brief Horizontal max of 32 unsigned u16s in a ZMM register. */
NK_INTERNAL nk_u16_t nk_reduce_max_u16x32_skylake_(__m512i max_u16x32) {
    __m256i lo_u16x16 = _mm512_castsi512_si256(max_u16x32);
    __m256i hi_u16x16 = _mm512_extracti64x4_epi64(max_u16x32, 1);
    __m256i max_u16x16 = _mm256_max_epu16(lo_u16x16, hi_u16x16);
    __m128i lo_u16x8 = _mm256_castsi256_si128(max_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(max_u16x16, 1);
    __m128i max_u16x8 = _mm_max_epu16(lo_u16x8, hi_u16x8);
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_srli_si128(max_u16x8, 2));
    return (nk_u16_t)_mm_cvtsi128_si32(max_u16x8);
}

/** @brief Horizontal min of 16 signed i32s in a ZMM register. */
NK_INTERNAL nk_i32_t nk_reduce_min_i32x16_skylake_(__m512i min_i32x16) {
    __m256i lo_i32x8 = _mm512_castsi512_si256(min_i32x16);
    __m256i hi_i32x8 = _mm512_extracti64x4_epi64(min_i32x16, 1);
    __m256i min_i32x8 = _mm256_min_epi32(lo_i32x8, hi_i32x8);
    __m128i lo_i32x4 = _mm256_castsi256_si128(min_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(min_i32x8, 1);
    __m128i min_i32x4 = _mm_min_epi32(lo_i32x4, hi_i32x4);
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(min_i32x4);
}

/** @brief Horizontal max of 16 signed i32s in a ZMM register. */
NK_INTERNAL nk_i32_t nk_reduce_max_i32x16_skylake_(__m512i max_i32x16) {
    __m256i lo_i32x8 = _mm512_castsi512_si256(max_i32x16);
    __m256i hi_i32x8 = _mm512_extracti64x4_epi64(max_i32x16, 1);
    __m256i max_i32x8 = _mm256_max_epi32(lo_i32x8, hi_i32x8);
    __m128i lo_i32x4 = _mm256_castsi256_si128(max_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(max_i32x8, 1);
    __m128i max_i32x4 = _mm_max_epi32(lo_i32x4, hi_i32x4);
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(max_i32x4);
}

/** @brief Horizontal min of 16 unsigned u32s in a ZMM register. */
NK_INTERNAL nk_u32_t nk_reduce_min_u32x16_skylake_(__m512i min_u32x16) {
    __m256i lo_u32x8 = _mm512_castsi512_si256(min_u32x16);
    __m256i hi_u32x8 = _mm512_extracti64x4_epi64(min_u32x16, 1);
    __m256i min_u32x8 = _mm256_min_epu32(lo_u32x8, hi_u32x8);
    __m128i lo_u32x4 = _mm256_castsi256_si128(min_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(min_u32x8, 1);
    __m128i min_u32x4 = _mm_min_epu32(lo_u32x4, hi_u32x4);
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(min_u32x4);
}

/** @brief Horizontal max of 16 unsigned u32s in a ZMM register. */
NK_INTERNAL nk_u32_t nk_reduce_max_u32x16_skylake_(__m512i max_u32x16) {
    __m256i lo_u32x8 = _mm512_castsi512_si256(max_u32x16);
    __m256i hi_u32x8 = _mm512_extracti64x4_epi64(max_u32x16, 1);
    __m256i max_u32x8 = _mm256_max_epu32(lo_u32x8, hi_u32x8);
    __m128i lo_u32x4 = _mm256_castsi256_si128(max_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(max_u32x8, 1);
    __m128i max_u32x4 = _mm_max_epu32(lo_u32x4, hi_u32x4);
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(max_u32x4);
}

/** @brief Horizontal min of 8 signed i64s in a ZMM register. */
NK_INTERNAL nk_i64_t nk_reduce_min_i64x8_skylake_(__m512i min_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(min_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(min_i64x8, 1);
    __m256i min_i64x4 = _mm256_min_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(min_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(min_i64x4, 1);
    __m128i min_i64x2 = _mm_min_epi64(lo_i64x2, hi_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(min_i64x2, min_i64x2);
    __m128i final_i64 = _mm_min_epi64(min_i64x2, hi_lane_i64);
    return _mm_cvtsi128_si64(final_i64);
}

/** @brief Horizontal max of 8 signed i64s in a ZMM register. */
NK_INTERNAL nk_i64_t nk_reduce_max_i64x8_skylake_(__m512i max_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(max_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(max_i64x8, 1);
    __m256i max_i64x4 = _mm256_max_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(max_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(max_i64x4, 1);
    __m128i max_i64x2 = _mm_max_epi64(lo_i64x2, hi_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(max_i64x2, max_i64x2);
    __m128i final_i64 = _mm_max_epi64(max_i64x2, hi_lane_i64);
    return _mm_cvtsi128_si64(final_i64);
}

/** @brief Horizontal min of 8 unsigned u64s in a ZMM register. */
NK_INTERNAL nk_u64_t nk_reduce_min_u64x8_skylake_(__m512i min_u64x8) {
    __m256i lo_u64x4 = _mm512_castsi512_si256(min_u64x8);
    __m256i hi_u64x4 = _mm512_extracti64x4_epi64(min_u64x8, 1);
    __m256i min_u64x4 = _mm256_min_epu64(lo_u64x4, hi_u64x4);
    __m128i lo_u64x2 = _mm256_castsi256_si128(min_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(min_u64x4, 1);
    __m128i min_u64x2 = _mm_min_epu64(lo_u64x2, hi_u64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(min_u64x2, min_u64x2);
    __m128i final_u64 = _mm_min_epu64(min_u64x2, hi_lane_u64);
    return (nk_u64_t)_mm_cvtsi128_si64(final_u64);
}

/** @brief Horizontal max of 8 unsigned u64s in a ZMM register. */
NK_INTERNAL nk_u64_t nk_reduce_max_u64x8_skylake_(__m512i max_u64x8) {
    __m256i lo_u64x4 = _mm512_castsi512_si256(max_u64x8);
    __m256i hi_u64x4 = _mm512_extracti64x4_epi64(max_u64x8, 1);
    __m256i max_u64x4 = _mm256_max_epu64(lo_u64x4, hi_u64x4);
    __m128i lo_u64x2 = _mm256_castsi256_si128(max_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(max_u64x4, 1);
    __m128i max_u64x2 = _mm_max_epu64(lo_u64x2, hi_u64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(max_u64x2, max_u64x2);
    __m128i final_u64 = _mm_max_epu64(max_u64x2, hi_lane_u64);
    return (nk_u64_t)_mm_cvtsi128_si64(final_u64);
}

/** @brief Horizontal sum of 8 unsigned u64s in a ZMM register. */
NK_INTERNAL nk_u64_t nk_reduce_add_u64x8_skylake_(__m512i sum_u64x8) {
    __m256i lo_u64x4 = _mm512_castsi512_si256(sum_u64x8);
    __m256i hi_u64x4 = _mm512_extracti64x4_epi64(sum_u64x8, 1);
    __m256i sum_u64x4 = _mm256_add_epi64(lo_u64x4, hi_u64x4);
    __m128i lo_u64x2 = _mm256_castsi256_si128(sum_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(sum_u64x4, 1);
    __m128i sum_u64x2 = _mm_add_epi64(lo_u64x2, hi_u64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(sum_u64x2, sum_u64x2);
    __m128i final_u64 = _mm_add_epi64(sum_u64x2, hi_lane_u64);
    return (nk_u64_t)_mm_cvtsi128_si64(final_u64);
}

NK_INTERNAL __m512i nk_fp8x64_to_u8x64_comparable_skylake_(__m512i raw_i8x64) {
    __mmask64 neg_m64 = _mm512_test_epi8_mask(raw_i8x64, _mm512_set1_epi8((char)0x80));
    __m512i pos_xor_i8x64 = _mm512_set1_epi8((char)0x80);
    __m512i neg_xor_i8x64 = _mm512_set1_epi8((char)0xFF);
    __m512i xor_i8x64 = _mm512_mask_mov_epi8(pos_xor_i8x64, neg_m64, neg_xor_i8x64);
    return _mm512_xor_si512(raw_i8x64, xor_i8x64);
}

NK_INTERNAL __m512i nk_u8x64_comparable_to_fp8x64_skylake_(__m512i cmp_i8x64) {
    __mmask64 was_neg_m64 = _mm512_cmplt_epu8_mask(cmp_i8x64, _mm512_set1_epi8((char)0x80));
    __m512i neg_xor_i8x64 = _mm512_set1_epi8((char)0xFF);
    __m512i pos_xor_i8x64 = _mm512_set1_epi8((char)0x80);
    __m512i xor_i8x64 = _mm512_mask_mov_epi8(pos_xor_i8x64, was_neg_m64, neg_xor_i8x64);
    return _mm512_xor_si512(cmp_i8x64, xor_i8x64);
}

/** @brief Horizontal argmin: returns index of first minimum unsigned byte in ZMM register. */
NK_INTERNAL nk_size_t nk_argmin_u8x64_skylake_(__m512i data_u8x64) {
    nk_u8_t min_val = nk_reduce_min_u8x64_skylake_(data_u8x64);
    __mmask64 eq_m64 = _mm512_cmpeq_epi8_mask(data_u8x64, _mm512_set1_epi8((char)min_val));
    return (nk_size_t)_tzcnt_u64(eq_m64);
}

/** @brief Horizontal argmax: returns index of first maximum unsigned byte in ZMM register. */
NK_INTERNAL nk_size_t nk_argmax_u8x64_skylake_(__m512i data_u8x64) {
    nk_u8_t max_val = nk_reduce_max_u8x64_skylake_(data_u8x64);
    __mmask64 eq_m64 = _mm512_cmpeq_epi8_mask(data_u8x64, _mm512_set1_epi8((char)max_val));
    return (nk_size_t)_tzcnt_u64(eq_m64);
}

NK_INTERNAL __m512i nk_fp6x64_to_u8x64_comparable_skylake_(__m512i raw_i8x64) {
    raw_i8x64 = _mm512_and_si512(raw_i8x64, _mm512_set1_epi8(0x3F)); // mask to 6 valid bits
    __mmask64 neg_m64 = _mm512_test_epi8_mask(raw_i8x64, _mm512_set1_epi8(0x20));
    __m512i pos_xor_i8x64 = _mm512_set1_epi8(0x20);
    __m512i neg_xor_i8x64 = _mm512_set1_epi8(0x3F);
    __m512i xor_i8x64 = _mm512_mask_mov_epi8(pos_xor_i8x64, neg_m64, neg_xor_i8x64);
    return _mm512_xor_si512(raw_i8x64, xor_i8x64);
}

NK_INTERNAL __m512i nk_u8x64_comparable_to_fp6x64_skylake_(__m512i cmp_i8x64) {
    __mmask64 was_neg_m64 = _mm512_cmplt_epu8_mask(cmp_i8x64, _mm512_set1_epi8(0x20));
    __m512i neg_xor_i8x64 = _mm512_set1_epi8(0x3F);
    __m512i pos_xor_i8x64 = _mm512_set1_epi8(0x20);
    __m512i xor_i8x64 = _mm512_mask_mov_epi8(pos_xor_i8x64, was_neg_m64, neg_xor_i8x64);
    return _mm512_xor_si512(cmp_i8x64, xor_i8x64);
}

NK_INTERNAL void nk_reduce_moments_f32_skylake_contiguous_( //
    nk_f32_t const *data_ptr, nk_size_t count,              //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    __m512d sum_low_f64x8 = _mm512_setzero_pd(), sum_high_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_low_f64x8 = _mm512_setzero_pd(), sumsq_high_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512d low_f64x8 = _mm512_cvtps_pd(_mm256_loadu_ps(data_ptr + idx));
        __m512d high_f64x8 = _mm512_cvtps_pd(_mm256_loadu_ps(data_ptr + idx + 8));
        sum_low_f64x8 = _mm512_add_pd(sum_low_f64x8, low_f64x8);
        sum_high_f64x8 = _mm512_add_pd(sum_high_f64x8, high_f64x8);
        sumsq_low_f64x8 = _mm512_fmadd_pd(low_f64x8, low_f64x8, sumsq_low_f64x8);
        sumsq_high_f64x8 = _mm512_fmadd_pd(high_f64x8, high_f64x8, sumsq_high_f64x8);
    }
    __m512d sum_f64x8 = _mm512_add_pd(sum_low_f64x8, sum_high_f64x8);
    __m512d sumsq_f64x8 = _mm512_add_pd(sumsq_low_f64x8, sumsq_high_f64x8);
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data_ptr + idx);
        __m512d low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(tail_f32x16));
        __m512d high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(tail_f32x16, 1));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, low_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(low_f64x8, low_f64x8, sumsq_f64x8);
        if (remaining > 8)
            sum_f64x8 = _mm512_add_pd(sum_f64x8, high_f64x8),
            sumsq_f64x8 = _mm512_fmadd_pd(high_f64x8, high_f64x8, sumsq_f64x8);
    }
    *sum_ptr = nk_reduce_add_f64x8_skylake_(sum_f64x8);
    *sumsq_ptr = nk_reduce_add_f64x8_skylake_(sumsq_f64x8);
}

NK_INTERNAL void nk_reduce_moments_f32_skylake_gather_(                //
    nk_f32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f32_t));
    __m512i indices_i32x16 = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                                _mm512_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 gathered_f32x16 = _mm512_i32gather_ps(indices_i32x16, data_ptr + idx * stride_elements,
                                                     sizeof(nk_f32_t));
        __m256 low_f32x8 = _mm512_castps512_ps256(gathered_f32x16);
        __m256 high_f32x8 = _mm512_extractf32x8_ps(gathered_f32x16, 1);
        __m512d low_f64x8 = _mm512_cvtps_pd(low_f32x8);
        __m512d high_f64x8 = _mm512_cvtps_pd(high_f32x8);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, low_f64x8);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, high_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(low_f64x8, low_f64x8, sumsq_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(high_f64x8, high_f64x8, sumsq_f64x8);
    }
    nk_f64_t sum = nk_reduce_add_f64x8_skylake_(sum_f64x8);
    nk_f64_t sumsq = nk_reduce_add_f64x8_skylake_(sumsq_f64x8);
    unsigned char const *ptr = (unsigned char const *)(data_ptr + idx * stride_elements);
    for (; idx < count; ++idx, ptr += stride_bytes) {
        nk_f64_t val = (nk_f64_t)(*(nk_f32_t const *)ptr);
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_f32_skylake_strided_(                  //
    nk_f32_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    __mmask16 stride_mask = nk_stride_mask_b32x16_(stride_elements);
    __m512d sum_f64x8 = _mm512_setzero_pd(), sumsq_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0, total = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask) * stride_elements;
    for (; idx + step <= total; idx += step) {
        __m512 data_f32x16 = _mm512_maskz_loadu_ps(stride_mask, data_ptr + idx);
        __m512d low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(data_f32x16));
        __m512d high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(data_f32x16, 1));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, low_f64x8);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, high_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(low_f64x8, low_f64x8, sumsq_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(high_f64x8, high_f64x8, sumsq_f64x8);
    }
    nk_size_t remaining = total - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 data_f32x16 = _mm512_maskz_loadu_ps(stride_mask & tail_mask, data_ptr + idx);
        __m512d low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(data_f32x16));
        __m512d high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(data_f32x16, 1));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, low_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(low_f64x8, low_f64x8, sumsq_f64x8);
        if (remaining > 8)
            sum_f64x8 = _mm512_add_pd(sum_f64x8, high_f64x8),
            sumsq_f64x8 = _mm512_fmadd_pd(high_f64x8, high_f64x8, sumsq_f64x8);
    }
    *sum_ptr = nk_reduce_add_f64x8_skylake_(sum_f64x8);
    *sumsq_ptr = nk_reduce_add_f64x8_skylake_(sumsq_f64x8);
}

NK_PUBLIC void nk_reduce_moments_f32_skylake(                          //
    nk_f32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_f32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f32_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f32_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f32_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 8)
        nk_reduce_moments_f32_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_f32_skylake_gather_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_f32_skylake_contiguous_( //
    nk_f32_t const *data_ptr, nk_size_t count,             //
    nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_f32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512 min_f32x16 = _mm512_set1_ps(NK_F32_MAX);
    __m512 max_f32x16 = _mm512_set1_ps(NK_F32_MIN);
    __m512i min_loop_cycle_u32x16 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u32x16 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u32x16 = _mm512_setzero_si512();
    __m512i one_u32x16 = _mm512_set1_epi32(1);

    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data_ptr + idx);
        __mmask16 min_changed_mask = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        __mmask16 max_changed_mask = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, min_changed_mask, data_f32x16);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, max_changed_mask, data_f32x16);
        min_loop_cycle_u32x16 = _mm512_mask_mov_epi32(min_loop_cycle_u32x16, min_changed_mask,
                                                      current_loop_cycle_u32x16);
        max_loop_cycle_u32x16 = _mm512_mask_mov_epi32(max_loop_cycle_u32x16, max_changed_mask,
                                                      current_loop_cycle_u32x16);
        current_loop_cycle_u32x16 = _mm512_add_epi32(current_loop_cycle_u32x16, one_u32x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_load = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_load, data_ptr + idx);
        __mmask16 min_changed_mask = _mm512_mask_cmp_ps_mask(tail_load, tail_f32x16, min_f32x16, _CMP_LT_OQ);
        __mmask16 max_changed_mask = _mm512_mask_cmp_ps_mask(tail_load, tail_f32x16, max_f32x16, _CMP_GT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, min_changed_mask, tail_f32x16);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, max_changed_mask, tail_f32x16);
        min_loop_cycle_u32x16 = _mm512_mask_mov_epi32(min_loop_cycle_u32x16, min_changed_mask,
                                                      current_loop_cycle_u32x16);
        max_loop_cycle_u32x16 = _mm512_mask_mov_epi32(max_loop_cycle_u32x16, max_changed_mask,
                                                      current_loop_cycle_u32x16);
    }

    nk_f32_t min_value = nk_reduce_min_f32x16_skylake_(min_f32x16);
    nk_f32_t max_value = nk_reduce_max_f32x16_skylake_(max_f32x16);
    if (min_value == NK_F32_MAX && max_value == NK_F32_MIN) {
        *min_value_ptr = NK_F32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    unsigned int min_lane, max_lane;
    {
        __mmask16 value_match_mask = _mm512_cmp_ps_mask(min_f32x16, _mm512_set1_ps(min_value), _CMP_EQ_OQ);
        __m512i masked_cycle_u32x16 = _mm512_mask_blend_epi32(value_match_mask, _mm512_set1_epi32((int)NK_U32_MAX),
                                                              min_loop_cycle_u32x16);
        nk_u32_t earliest_loop_cycle = nk_reduce_min_u32x16_skylake_(masked_cycle_u32x16);
        __mmask16 cycle_match_mask = _mm512_cmpeq_epi32_mask(masked_cycle_u32x16,
                                                             _mm512_set1_epi32((int)earliest_loop_cycle));
        min_lane = _tzcnt_u32(cycle_match_mask);
    }
    {
        __mmask16 value_match_mask = _mm512_cmp_ps_mask(max_f32x16, _mm512_set1_ps(max_value), _CMP_EQ_OQ);
        __m512i masked_cycle_u32x16 = _mm512_mask_blend_epi32(value_match_mask, _mm512_set1_epi32((int)NK_U32_MAX),
                                                              max_loop_cycle_u32x16);
        nk_u32_t earliest_loop_cycle = nk_reduce_min_u32x16_skylake_(masked_cycle_u32x16);
        __mmask16 cycle_match_mask = _mm512_cmpeq_epi32_mask(masked_cycle_u32x16,
                                                             _mm512_set1_epi32((int)earliest_loop_cycle));
        max_lane = _tzcnt_u32(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u32x16;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u32s[min_lane] * 16 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u32x16;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u32s[max_lane] * 16 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_f32_skylake(                           //
    nk_f32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_F32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f32_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_f32_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_min, &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f32_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                 max_index_ptr);
    else
        nk_reduce_minmax_f32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_f64_skylake_contiguous_( //
    nk_f64_t const *data_ptr, nk_size_t count,              //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d sum_comp_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_comp_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512d val_f64x8 = _mm512_loadu_pd(data_ptr + idx);
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d val_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data_ptr + idx);
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    *sum_ptr = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sum_f64x8, sum_comp_f64x8));
    *sumsq_ptr = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sumsq_f64x8, sumsq_comp_f64x8));
}

NK_INTERNAL void nk_reduce_moments_f64_skylake_strided_(                  //
    nk_f64_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    __mmask8 stride_mask = nk_stride_mask_b64x8_(stride_elements);
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d sum_comp_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_comp_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0, total = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask) * stride_elements;
    for (; idx + step <= total; idx += step) {
        __m512d val_f64x8 = _mm512_maskz_loadu_pd(stride_mask, data_ptr + idx);
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    nk_size_t remaining = total - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = stride_mask & (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d val_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data_ptr + idx);
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    *sum_ptr = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sum_f64x8, sum_comp_f64x8));
    *sumsq_ptr = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sumsq_f64x8, sumsq_comp_f64x8));
}

NK_INTERNAL void nk_reduce_moments_f64_skylake_gather_(                //
    nk_f64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f64_t));
    __m256i indices_i32x8 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d sum_comp_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_comp_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512d val_f64x8 = _mm512_i32gather_pd(indices_i32x8, data_ptr + idx * stride_elements, sizeof(nk_f64_t));
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    nk_f64_t sum = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sum_f64x8, sum_comp_f64x8));
    nk_f64_t sumsq = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sumsq_f64x8, sumsq_comp_f64x8));
    unsigned char const *ptr = (unsigned char const *)(data_ptr + idx * stride_elements);
    for (; idx < count; ++idx, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_f64_skylake(                          //
    nk_f64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_f64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f64_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f64_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f64_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 8)
        nk_reduce_moments_f64_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_f64_skylake_gather_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_i8_skylake_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,              //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum: VPSADBW with XOR bias (same as nk_reduce_add_i8_skylake_contiguous_).
    // Sumsq: widen i8→i16, VPMADDWD(x,x) → i32 (pairs of squares), accumulate i32.
    // i32 overflow safe: max per lane = (128² + 128²) * 65536 iters ≈ 2.1B = safe limit.
    // The dispatch recurses at (NK_U16_MAX+1)*64 elements → at most 65536 iterations here.
    __m512i bias_i8x64 = _mm512_set1_epi8((char)0x80);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_low_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_high_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data_ptr + idx);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64, bias_i8x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
        __m512i low_i16x32 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(data_i8x64));
        __m512i high_i16x32 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_i8x64, 1));
        sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, _mm512_madd_epi16(low_i16x32, low_i16x32));
        sumsq_high_i32x16 = _mm512_add_epi32(sumsq_high_i32x16, _mm512_madd_epi16(high_i16x32, high_i16x32));
    }
    // Flush i32 → i64 once
    sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, sumsq_high_i32x16);
    __m512i sumsq_i64x8 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sumsq_low_i32x16));
    sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sumsq_low_i32x16, 1)));
    nk_i64_t sum = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8);
    sum -= (nk_i64_t)128 * (nk_i64_t)idx;
    nk_u64_t sumsq = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data_ptr[idx];
        sum += val, sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_i8_skylake_strided_(                  //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    nk_size_t elements_per_vector = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m64);
    __m512i masked_bias_i8x64 = _mm512_maskz_mov_epi8(stride_mask_m64, _mm512_set1_epi8((char)0x80));
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_low_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_high_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t vector_element_count = 0;
    nk_size_t step = elements_per_vector * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m512i data_i8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data_ptr + idx_scalars);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64, masked_bias_i8x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
        __m512i low_i16x32 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(data_i8x64));
        __m512i high_i16x32 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_i8x64, 1));
        sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, _mm512_madd_epi16(low_i16x32, low_i16x32));
        sumsq_high_i32x16 = _mm512_add_epi32(sumsq_high_i32x16, _mm512_madd_epi16(high_i16x32, high_i16x32));
        vector_element_count += elements_per_vector;
    }
    sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, sumsq_high_i32x16);
    __m512i sumsq_i64x8 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sumsq_low_i32x16));
    sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sumsq_low_i32x16, 1)));
    nk_i64_t sum = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8);
    sum -= (nk_i64_t)128 * (nk_i64_t)vector_element_count;
    nk_u64_t sumsq = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    nk_i8_t const *ptr = data_ptr + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_i64_t val = (nk_i64_t)*ptr;
        sum += val, sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i8_skylake(                          //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i8_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i8_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_sum, &right_sumsq);
        *sum_ptr = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 16)
        nk_reduce_moments_i8_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i8_skylake_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,             //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i min_i8x64 = _mm512_set1_epi8((char)NK_I8_MAX);
    __m512i max_i8x64 = _mm512_set1_epi8(NK_I8_MIN);
    __m512i min_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data_ptr + idx);
        __mmask64 min_changed_mask = _mm512_cmp_epi8_mask(data_i8x64, min_i8x64, _MM_CMPINT_LT);
        __mmask64 max_changed_mask = _mm512_cmp_epi8_mask(data_i8x64, max_i8x64, _MM_CMPINT_NLE);
        min_i8x64 = _mm512_mask_mov_epi8(min_i8x64, min_changed_mask, data_i8x64);
        max_i8x64 = _mm512_mask_mov_epi8(max_i8x64, max_changed_mask, data_i8x64);
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
        current_loop_cycle_u8x64 = _mm512_add_epi8(current_loop_cycle_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i tail_i8x64 = _mm512_maskz_loadu_epi8(tail_load, data_ptr + idx);
        __mmask64 min_changed_mask = _mm512_mask_cmp_epi8_mask(tail_load, tail_i8x64, min_i8x64, _MM_CMPINT_LT);
        __mmask64 max_changed_mask = _mm512_mask_cmp_epi8_mask(tail_load, tail_i8x64, max_i8x64, _MM_CMPINT_NLE);
        min_i8x64 = _mm512_mask_mov_epi8(min_i8x64, min_changed_mask, tail_i8x64);
        max_i8x64 = _mm512_mask_mov_epi8(max_i8x64, max_changed_mask, tail_i8x64);
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
    }

    nk_i8_t min_value = nk_reduce_min_i8x64_skylake_(min_i8x64);
    nk_i8_t max_value = nk_reduce_max_i8x64_skylake_(max_i8x64);
    unsigned int min_lane, max_lane;
    {
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(min_i8x64, _mm512_set1_epi8(min_value));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            min_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        min_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
    }
    {
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(max_i8x64, _mm512_set1_epi8(max_value));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            max_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        max_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u8x64;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u8s[min_lane] * 64 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u8x64;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u8s[max_lane] * 64 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i8_skylake(                           //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I8_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_i8_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i8_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                    &left_max_index);
        nk_reduce_minmax_i8_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                    &right_min, &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i8_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                max_index_ptr);
    else
        nk_reduce_minmax_i8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u8_skylake_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,              //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum: VPSADBW directly (same as nk_reduce_add_u8_skylake_contiguous_).
    // Sumsq: widen u8→i16, VPMADDWD(x,x) → i32 (pairs of squares), accumulate i32.
    // i32 overflow safe: max per lane = (255² + 255²) * 1024 iters ≈ 133M < 2.1B.
    __m512i zero_u8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_low_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_high_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data_ptr + idx);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
        __m512i low_i16x32 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(data_u8x64));
        __m512i high_i16x32 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(data_u8x64, 1));
        sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, _mm512_madd_epi16(low_i16x32, low_i16x32));
        sumsq_high_i32x16 = _mm512_add_epi32(sumsq_high_i32x16, _mm512_madd_epi16(high_i16x32, high_i16x32));
    }
    // Flush i32 → u64 once
    sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, sumsq_high_i32x16);
    __m512i sumsq_u64x8 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(sumsq_low_i32x16));
    sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(sumsq_low_i32x16, 1)));
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    nk_u64_t sumsq = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data_ptr[idx];
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_u8_skylake_strided_(                  //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i zero_u8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_low_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_high_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m64) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data_ptr + idx_scalars);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
        __m512i low_i16x32 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(data_u8x64));
        __m512i high_i16x32 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(data_u8x64, 1));
        sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, _mm512_madd_epi16(low_i16x32, low_i16x32));
        sumsq_high_i32x16 = _mm512_add_epi32(sumsq_high_i32x16, _mm512_madd_epi16(high_i16x32, high_i16x32));
    }
    sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, sumsq_high_i32x16);
    __m512i sumsq_u64x8 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(sumsq_low_i32x16));
    sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(sumsq_low_i32x16, 1)));
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    nk_u64_t sumsq = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    nk_u8_t const *ptr = data_ptr + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_u64_t val = (nk_u64_t)*ptr;
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u8_skylake(                          //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_sum, &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 16)
        nk_reduce_moments_u8_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u8_skylake_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,             //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i min_u8x64 = _mm512_set1_epi8((char)NK_U8_MAX);
    __m512i max_u8x64 = _mm512_setzero_si512();
    __m512i min_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data_ptr + idx);
        __mmask64 min_changed_mask = _mm512_cmp_epu8_mask(data_u8x64, min_u8x64, _MM_CMPINT_LT);
        __mmask64 max_changed_mask = _mm512_cmp_epu8_mask(data_u8x64, max_u8x64, _MM_CMPINT_NLE);
        min_u8x64 = _mm512_mask_mov_epi8(min_u8x64, min_changed_mask, data_u8x64);
        max_u8x64 = _mm512_mask_mov_epi8(max_u8x64, max_changed_mask, data_u8x64);
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
        current_loop_cycle_u8x64 = _mm512_add_epi8(current_loop_cycle_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i tail_u8x64 = _mm512_maskz_loadu_epi8(tail_load, data_ptr + idx);
        __mmask64 min_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, tail_u8x64, min_u8x64, _MM_CMPINT_LT);
        __mmask64 max_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, tail_u8x64, max_u8x64, _MM_CMPINT_NLE);
        min_u8x64 = _mm512_mask_mov_epi8(min_u8x64, min_changed_mask, tail_u8x64);
        max_u8x64 = _mm512_mask_mov_epi8(max_u8x64, max_changed_mask, tail_u8x64);
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
    }

    nk_u8_t min_value = nk_reduce_min_u8x64_skylake_(min_u8x64);
    nk_u8_t max_value = nk_reduce_max_u8x64_skylake_(max_u8x64);
    unsigned int min_lane, max_lane;
    {
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(min_u8x64, _mm512_set1_epi8((char)min_value));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            min_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        min_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
    }
    {
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(max_u8x64, _mm512_set1_epi8((char)max_value));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            max_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        max_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u8x64;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u8s[min_lane] * 64 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u8x64;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u8s[max_lane] * 64 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u8_skylake(                           //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_u8_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u8_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                    &left_max_index);
        nk_reduce_minmax_u8_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                    &right_min, &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u8_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                max_index_ptr);
    else
        nk_reduce_minmax_u8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i16_skylake_contiguous_( //
    nk_i16_t const *data_ptr, nk_size_t count,              //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum: VPMADDWD(data_ptr, ones) → i32 pairs, accumulate i32, single flush at end.
    // Within 65536-element block (2048 iters), max i32 = ±65536 * 2048 ≈ ±134M — safe.
    // Sumsq: VPMADDWD(data_ptr, data_ptr) → i32, each up to ~2.1B — must flush to i64 every iteration.
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data_ptr + idx);
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(data_i16x32, ones_i16x32));
        __m512i sq_i32x16 = _mm512_madd_epi16(data_i16x32, data_i16x32);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sq_i32x16)));
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sq_i32x16, 1)));
    }
    __m512i sum_i64x8 = _mm512_add_epi64(                                 //
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sum_i32x16)),        //
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sum_i32x16, 1))); //
    nk_i64_t sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_u64_t sumsq = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data_ptr[idx];
        sum += val, sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_i16_skylake_strided_(                  //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_(stride_elements);
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m32) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m512i data_i16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data_ptr + idx_scalars);
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(data_i16x32, ones_i16x32));
        __m512i sq_i32x16 = _mm512_madd_epi16(data_i16x32, data_i16x32);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sq_i32x16)));
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sq_i32x16, 1)));
    }
    __m512i sum_i64x8 = _mm512_add_epi64(                                 //
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sum_i32x16)),        //
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sum_i32x16, 1))); //
    nk_i64_t sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_u64_t sumsq = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    nk_i16_t const *ptr = data_ptr + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_i64_t val = (nk_i64_t)*ptr;
        sum += val, sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i16_skylake(                          //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_I16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i16_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i16_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_sum, &right_sumsq);
        *sum_ptr = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i16_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 16)
        nk_reduce_moments_i16_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i16_skylake_contiguous_( //
    nk_i16_t const *data_ptr, nk_size_t count,             //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i min_i16x32 = _mm512_set1_epi16((short)NK_I16_MAX);
    __m512i max_i16x32 = _mm512_set1_epi16(NK_I16_MIN);
    __m512i min_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i one_u16x32 = _mm512_set1_epi16(1);

    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data_ptr + idx);
        __mmask32 min_changed_mask = _mm512_cmp_epi16_mask(data_i16x32, min_i16x32, _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_cmp_epi16_mask(data_i16x32, max_i16x32, _MM_CMPINT_NLE);
        min_i16x32 = _mm512_mask_mov_epi16(min_i16x32, min_changed_mask, data_i16x32);
        max_i16x32 = _mm512_mask_mov_epi16(max_i16x32, max_changed_mask, data_i16x32);
        min_loop_cycle_u16x32 = _mm512_mask_mov_epi16(min_loop_cycle_u16x32, min_changed_mask,
                                                      current_loop_cycle_u16x32);
        max_loop_cycle_u16x32 = _mm512_mask_mov_epi16(max_loop_cycle_u16x32, max_changed_mask,
                                                      current_loop_cycle_u16x32);
        current_loop_cycle_u16x32 = _mm512_add_epi16(current_loop_cycle_u16x32, one_u16x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_load = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i tail_i16x32 = _mm512_maskz_loadu_epi16(tail_load, data_ptr + idx);
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(tail_load, tail_i16x32, min_i16x32, _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(tail_load, tail_i16x32, max_i16x32, _MM_CMPINT_NLE);
        min_i16x32 = _mm512_mask_mov_epi16(min_i16x32, min_changed_mask, tail_i16x32);
        max_i16x32 = _mm512_mask_mov_epi16(max_i16x32, max_changed_mask, tail_i16x32);
        min_loop_cycle_u16x32 = _mm512_mask_mov_epi16(min_loop_cycle_u16x32, min_changed_mask,
                                                      current_loop_cycle_u16x32);
        max_loop_cycle_u16x32 = _mm512_mask_mov_epi16(max_loop_cycle_u16x32, max_changed_mask,
                                                      current_loop_cycle_u16x32);
    }

    nk_i16_t min_value = nk_reduce_min_i16x32_skylake_(min_i16x32);
    nk_i16_t max_value = nk_reduce_max_i16x32_skylake_(max_i16x32);
    unsigned int min_lane, max_lane;
    {
        __mmask32 value_match_mask = _mm512_cmpeq_epi16_mask(min_i16x32, _mm512_set1_epi16(min_value));
        __m512i masked_cycle_u16x32 = _mm512_mask_blend_epi16(value_match_mask, _mm512_set1_epi16((short)NK_U16_MAX),
                                                              min_loop_cycle_u16x32);
        nk_u16_t earliest_loop_cycle = nk_reduce_min_u16x32_skylake_(masked_cycle_u16x32);
        __mmask32 cycle_match_mask = _mm512_cmpeq_epi16_mask(masked_cycle_u16x32,
                                                             _mm512_set1_epi16((short)earliest_loop_cycle));
        min_lane = _tzcnt_u32(cycle_match_mask);
    }
    {
        __mmask32 value_match_mask = _mm512_cmpeq_epi16_mask(max_i16x32, _mm512_set1_epi16(max_value));
        __m512i masked_cycle_u16x32 = _mm512_mask_blend_epi16(value_match_mask, _mm512_set1_epi16((short)NK_U16_MAX),
                                                              max_loop_cycle_u16x32);
        nk_u16_t earliest_loop_cycle = nk_reduce_min_u16x32_skylake_(masked_cycle_u16x32);
        __mmask32 cycle_match_mask = _mm512_cmpeq_epi16_mask(masked_cycle_u16x32,
                                                             _mm512_set1_epi16((short)earliest_loop_cycle));
        max_lane = _tzcnt_u32(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u16x32;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u16s[min_lane] * 32 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u16x32;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u16s[max_lane] * 32 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i16_skylake(                           //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_i16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i16_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_i16_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_min, &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i16_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                 max_index_ptr);
    else
        nk_reduce_minmax_i16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u16_skylake_contiguous_( //
    nk_u16_t const *data_ptr, nk_size_t count,              //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Widen u16→u32, square in u32, widen to u64. Avoids bias trick whose
    // VPMADDWD pair-of-squares overflows i32 when both lanes map to -32768.
    __m512i zero = _mm512_setzero_si512();
    __m512i sum_u32x16 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const *)(data_ptr + idx)));
        sum_u32x16 = _mm512_add_epi32(sum_u32x16, data_u32x16);
        __m512i sq_u32x16 = _mm512_mullo_epi32(data_u32x16, data_u32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpacklo_epi32(sq_u32x16, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpackhi_epi32(sq_u32x16, zero));
    }
    if (idx < count) {
        __mmask16 tail_mask = (__mmask16)((1u << (count - idx)) - 1);
        __m512i data_u32x16 = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, data_ptr + idx));
        sum_u32x16 = _mm512_add_epi32(sum_u32x16, data_u32x16);
        __m512i sq_u32x16 = _mm512_mullo_epi32(data_u32x16, data_u32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpacklo_epi32(sq_u32x16, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpackhi_epi32(sq_u32x16, zero));
    }
    __m512i sum_u64x8 = _mm512_add_epi64(         //
        _mm512_unpacklo_epi32(sum_u32x16, zero),  //
        _mm512_unpackhi_epi32(sum_u32x16, zero)); //
    *sum_ptr = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sum_u64x8);
    *sumsq_ptr = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_u64x8);
}

NK_INTERNAL void nk_reduce_moments_u16_skylake_strided_(                  //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_(stride_elements);
    __m512i zero = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m32) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m512i data_u16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data_ptr + idx_scalars);
        __m512i low_u32x16 = _mm512_unpacklo_epi16(data_u16x32, zero);
        __m512i high_u32x16 = _mm512_unpackhi_epi16(data_u16x32, zero);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_unpacklo_epi32(low_u32x16, zero));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_unpackhi_epi32(low_u32x16, zero));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_unpacklo_epi32(high_u32x16, zero));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_unpackhi_epi32(high_u32x16, zero));
        __m512i low_sq = _mm512_mullo_epi32(low_u32x16, low_u32x16);
        __m512i high_sq = _mm512_mullo_epi32(high_u32x16, high_u32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpacklo_epi32(low_sq, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpackhi_epi32(low_sq, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpacklo_epi32(high_sq, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpackhi_epi32(high_sq, zero));
    }
    nk_u64_t sum = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sum_u64x8);
    nk_u64_t sumsq = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_u64x8);
    nk_u16_t const *ptr = data_ptr + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_u64_t val = (nk_u64_t)*ptr;
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u16_skylake(                          //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u16_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u16_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_sum, &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u16_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 16)
        nk_reduce_moments_u16_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u16_skylake_contiguous_( //
    nk_u16_t const *data_ptr, nk_size_t count,             //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i min_u16x32 = _mm512_set1_epi16((short)NK_U16_MAX);
    __m512i max_u16x32 = _mm512_setzero_si512();
    __m512i min_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i one_u16x32 = _mm512_set1_epi16(1);

    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_u16x32 = _mm512_loadu_si512(data_ptr + idx);
        __mmask32 min_changed_mask = _mm512_cmp_epu16_mask(data_u16x32, min_u16x32, _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_cmp_epu16_mask(data_u16x32, max_u16x32, _MM_CMPINT_NLE);
        min_u16x32 = _mm512_mask_mov_epi16(min_u16x32, min_changed_mask, data_u16x32);
        max_u16x32 = _mm512_mask_mov_epi16(max_u16x32, max_changed_mask, data_u16x32);
        min_loop_cycle_u16x32 = _mm512_mask_mov_epi16(min_loop_cycle_u16x32, min_changed_mask,
                                                      current_loop_cycle_u16x32);
        max_loop_cycle_u16x32 = _mm512_mask_mov_epi16(max_loop_cycle_u16x32, max_changed_mask,
                                                      current_loop_cycle_u16x32);
        current_loop_cycle_u16x32 = _mm512_add_epi16(current_loop_cycle_u16x32, one_u16x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_load = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i tail_u16x32 = _mm512_maskz_loadu_epi16(tail_load, data_ptr + idx);
        __mmask32 min_changed_mask = _mm512_mask_cmp_epu16_mask(tail_load, tail_u16x32, min_u16x32, _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epu16_mask(tail_load, tail_u16x32, max_u16x32, _MM_CMPINT_NLE);
        min_u16x32 = _mm512_mask_mov_epi16(min_u16x32, min_changed_mask, tail_u16x32);
        max_u16x32 = _mm512_mask_mov_epi16(max_u16x32, max_changed_mask, tail_u16x32);
        min_loop_cycle_u16x32 = _mm512_mask_mov_epi16(min_loop_cycle_u16x32, min_changed_mask,
                                                      current_loop_cycle_u16x32);
        max_loop_cycle_u16x32 = _mm512_mask_mov_epi16(max_loop_cycle_u16x32, max_changed_mask,
                                                      current_loop_cycle_u16x32);
    }

    nk_u16_t min_value = nk_reduce_min_u16x32_skylake_(min_u16x32);
    nk_u16_t max_value = nk_reduce_max_u16x32_skylake_(max_u16x32);
    unsigned int min_lane, max_lane;
    {
        __mmask32 value_match_mask = _mm512_cmpeq_epi16_mask(min_u16x32, _mm512_set1_epi16((short)min_value));
        __m512i masked_cycle_u16x32 = _mm512_mask_blend_epi16(value_match_mask, _mm512_set1_epi16((short)NK_U16_MAX),
                                                              min_loop_cycle_u16x32);
        nk_u16_t earliest_loop_cycle = nk_reduce_min_u16x32_skylake_(masked_cycle_u16x32);
        __mmask32 cycle_match_mask = _mm512_cmpeq_epi16_mask(masked_cycle_u16x32,
                                                             _mm512_set1_epi16((short)earliest_loop_cycle));
        min_lane = _tzcnt_u32(cycle_match_mask);
    }
    {
        __mmask32 value_match_mask = _mm512_cmpeq_epi16_mask(max_u16x32, _mm512_set1_epi16((short)max_value));
        __m512i masked_cycle_u16x32 = _mm512_mask_blend_epi16(value_match_mask, _mm512_set1_epi16((short)NK_U16_MAX),
                                                              max_loop_cycle_u16x32);
        nk_u16_t earliest_loop_cycle = nk_reduce_min_u16x32_skylake_(masked_cycle_u16x32);
        __mmask32 cycle_match_mask = _mm512_cmpeq_epi16_mask(masked_cycle_u16x32,
                                                             _mm512_set1_epi16((short)earliest_loop_cycle));
        max_lane = _tzcnt_u32(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u16x32;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u16s[min_lane] * 32 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u16x32;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u16s[max_lane] * 32 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u16_skylake(                           //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_u16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u16_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_u16_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_min, &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u16_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                 max_index_ptr);
    else
        nk_reduce_minmax_u16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

/** @brief Unsigned saturating add of two i64x8 vectors (3 uops). */
NK_INTERNAL __m512i nk_u64_sadd_epi64_skylake_(__m512i a, __m512i b) {
    __m512i result = _mm512_add_epi64(a, b);
    __mmask8 ovf = _mm512_cmp_epu64_mask(result, a, _MM_CMPINT_LT);
    return _mm512_mask_mov_epi64(result, ovf, _mm512_set1_epi64((nk_i64_t)-1));
}

/** @brief Saturating i64 square: clamp when |val| > floor(sqrt(INT64_MAX)). */
NK_INTERNAL __m512i nk_i64_smul_sq_epi64_skylake_(__m512i val) {
    __m512i sq = _mm512_mullo_epi64(val, val);
    __m512i abs_val = _mm512_abs_epi64(val);
    __mmask8 ovf = _mm512_cmp_epu64_mask(abs_val, _mm512_set1_epi64(3037000499ll), _MM_CMPINT_NLE);
    return _mm512_mask_mov_epi64(sq, ovf, _mm512_set1_epi64(9223372036854775807ll));
}

/** @brief Saturating u64 square: clamp when val > floor(sqrt(UINT64_MAX)). */
NK_INTERNAL __m512i nk_u64_smul_sq_epi64_skylake_(__m512i val) {
    __m512i sq = _mm512_mullo_epi64(val, val);
    __mmask8 ovf = _mm512_cmp_epu64_mask(val, _mm512_set1_epi64(4294967295ll), _MM_CMPINT_NLE);
    return _mm512_mask_mov_epi64(sq, ovf, _mm512_set1_epi64((nk_i64_t)-1));
}

/** @brief Saturating horizontal sum of 8 unsigned u64 lanes.
 *  Tree reduction: unsigned saturating add is order-independent because the
 *  accumulator can only increase — once saturated to UINT64_MAX, it stays there.
 *  Result equals min(true_sum, UINT64_MAX) regardless of reduction order. */
NK_INTERNAL nk_u64_t nk_reduce_sadd_u64x8_skylake_(__m512i v) {
    // 8→4: fold high 256 bits into low 256 bits (VSHUFI64X2 + 3-uop sat-add)
    v = nk_u64_sadd_epi64_skylake_(v, _mm512_shuffle_i64x2(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
    // 4→2: fold lanes 2-3 into lanes 0-1
    v = nk_u64_sadd_epi64_skylake_(v, _mm512_shuffle_i64x2(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
    // 2→1: fold lane 1 into lane 0 (VALIGNQ + 3-uop sat-add)
    v = nk_u64_sadd_epi64_skylake_(v, _mm512_alignr_epi64(v, v, 1));
    return (nk_u64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(v));
}

NK_INTERNAL void nk_reduce_moments_i32_skylake_contiguous_( //
    nk_i32_t const *data_ptr, nk_size_t count,              //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum: 128-bit accumulation (lower + upper) — no block cap needed.
    // Sumsq: unsigned wrapping accumulation with carry-based overflow detection.
    __m512i sum_lower_i64x8 = _mm512_setzero_si512();
    __m512i sum_upper_i64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    __mmask8 sumsq_overflow_mask = 0;
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_i32x16 = _mm512_loadu_si512(data_ptr + idx);
        __m256i low_i32x8 = _mm512_castsi512_si256(data_i32x16);
        __m256i high_i32x8 = _mm512_extracti64x4_epi64(data_i32x16, 1);
        // 128-bit sum: lower half
        __m512i widened_low_i64x8 = _mm512_cvtepi32_epi64(low_i32x8);
        __m512i sum_before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, widened_low_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, sum_before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, _mm512_srai_epi64(widened_low_i64x8, 63));
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
        // 128-bit sum: upper half
        __m512i widened_high_i64x8 = _mm512_cvtepi32_epi64(high_i32x8);
        sum_before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, widened_high_i64x8);
        carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, sum_before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, _mm512_srai_epi64(widened_high_i64x8, 63));
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
        // Sumsq: unsigned accumulation with carry detection
        __m512i even_sq_u64x8 = _mm512_mul_epi32(data_i32x16, data_i32x16);
        __m512i odd_i32x16 = _mm512_srli_epi64(data_i32x16, 32);
        __m512i odd_sq_u64x8 = _mm512_mul_epi32(odd_i32x16, odd_i32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, even_sq_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, even_sq_u64x8, _MM_CMPINT_LT);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, odd_sq_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, odd_sq_u64x8, _MM_CMPINT_LT);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i data_i32x16 = _mm512_maskz_loadu_epi32(tail_mask, data_ptr + idx);
        __m256i low_i32x8 = _mm512_castsi512_si256(data_i32x16);
        __m256i high_i32x8 = _mm512_extracti64x4_epi64(data_i32x16, 1);
        __m512i widened_low_i64x8 = _mm512_cvtepi32_epi64(low_i32x8);
        __m512i sum_before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, widened_low_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, sum_before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, _mm512_srai_epi64(widened_low_i64x8, 63));
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
        if (remaining > 8) {
            __m512i widened_high_i64x8 = _mm512_cvtepi32_epi64(high_i32x8);
            sum_before_i64x8 = sum_lower_i64x8;
            sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, widened_high_i64x8);
            carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, sum_before_i64x8, _MM_CMPINT_LT);
            sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, _mm512_srai_epi64(widened_high_i64x8, 63));
            sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
        }
        __m512i even_sq_u64x8 = _mm512_mul_epi32(data_i32x16, data_i32x16);
        __m512i odd_i32x16 = _mm512_srli_epi64(data_i32x16, 32);
        __m512i odd_sq_u64x8 = _mm512_mul_epi32(odd_i32x16, odd_i32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, even_sq_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, even_sq_u64x8, _MM_CMPINT_LT);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, odd_sq_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, odd_sq_u64x8, _MM_CMPINT_LT);
    }
    // Sumsq: horizontal unsigned saturating reduction
    nk_u64_t sumsq;
    if (sumsq_overflow_mask) sumsq = NK_U64_MAX;
    else sumsq = nk_reduce_sadd_u64x8_skylake_(sumsq_u64x8);
    // Sum: horizontal 128-bit tree reduction, same as i64 skylake
    { // 8→4
        __m512i fold_lower_i64x8 = _mm512_shuffle_i64x2(sum_lower_i64x8, sum_lower_i64x8, _MM_SHUFFLE(1, 0, 3, 2));
        __m512i fold_upper_i64x8 = _mm512_shuffle_i64x2(sum_upper_i64x8, sum_upper_i64x8, _MM_SHUFFLE(1, 0, 3, 2));
        __m512i before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, fold_lower_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, fold_upper_i64x8);
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
    }
    { // 4→2
        __m512i fold_lower_i64x8 = _mm512_shuffle_i64x2(sum_lower_i64x8, sum_lower_i64x8, _MM_SHUFFLE(2, 3, 0, 1));
        __m512i fold_upper_i64x8 = _mm512_shuffle_i64x2(sum_upper_i64x8, sum_upper_i64x8, _MM_SHUFFLE(2, 3, 0, 1));
        __m512i before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, fold_lower_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, fold_upper_i64x8);
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
    }
    { // 2→1
        __m512i fold_lower_i64x8 = _mm512_alignr_epi64(sum_lower_i64x8, sum_lower_i64x8, 1);
        __m512i fold_upper_i64x8 = _mm512_alignr_epi64(sum_upper_i64x8, sum_upper_i64x8, 1);
        __m512i before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, fold_lower_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, fold_upper_i64x8);
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
    }
    nk_i64_t sum_lower = _mm_cvtsi128_si64(_mm512_castsi512_si128(sum_lower_i64x8));
    nk_i64_t sum_upper = _mm_cvtsi128_si64(_mm512_castsi512_si128(sum_upper_i64x8));
    if (sum_upper == (sum_lower >> 63)) *sum_ptr = sum_lower;
    else if (sum_upper >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i32_skylake(                          //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_i32_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i32_skylake_contiguous_( //
    nk_i32_t const *data_ptr, nk_size_t count,             //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i min_i32x16 = _mm512_set1_epi32(NK_I32_MAX);
    __m512i max_i32x16 = _mm512_set1_epi32(NK_I32_MIN);
    __m512i min_loop_cycle_u32x16 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u32x16 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u32x16 = _mm512_setzero_si512();
    __m512i one_u32x16 = _mm512_set1_epi32(1);

    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_i32x16 = _mm512_loadu_si512(data_ptr + idx);
        __mmask16 min_changed_mask = _mm512_cmp_epi32_mask(data_i32x16, min_i32x16, _MM_CMPINT_LT);
        __mmask16 max_changed_mask = _mm512_cmp_epi32_mask(data_i32x16, max_i32x16, _MM_CMPINT_NLE);
        min_i32x16 = _mm512_mask_mov_epi32(min_i32x16, min_changed_mask, data_i32x16);
        max_i32x16 = _mm512_mask_mov_epi32(max_i32x16, max_changed_mask, data_i32x16);
        min_loop_cycle_u32x16 = _mm512_mask_mov_epi32(min_loop_cycle_u32x16, min_changed_mask,
                                                      current_loop_cycle_u32x16);
        max_loop_cycle_u32x16 = _mm512_mask_mov_epi32(max_loop_cycle_u32x16, max_changed_mask,
                                                      current_loop_cycle_u32x16);
        current_loop_cycle_u32x16 = _mm512_add_epi32(current_loop_cycle_u32x16, one_u32x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_load = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i tail_i32x16 = _mm512_maskz_loadu_epi32(tail_load, data_ptr + idx);
        __mmask16 min_changed_mask = _mm512_mask_cmp_epi32_mask(tail_load, tail_i32x16, min_i32x16, _MM_CMPINT_LT);
        __mmask16 max_changed_mask = _mm512_mask_cmp_epi32_mask(tail_load, tail_i32x16, max_i32x16, _MM_CMPINT_NLE);
        min_i32x16 = _mm512_mask_mov_epi32(min_i32x16, min_changed_mask, tail_i32x16);
        max_i32x16 = _mm512_mask_mov_epi32(max_i32x16, max_changed_mask, tail_i32x16);
        min_loop_cycle_u32x16 = _mm512_mask_mov_epi32(min_loop_cycle_u32x16, min_changed_mask,
                                                      current_loop_cycle_u32x16);
        max_loop_cycle_u32x16 = _mm512_mask_mov_epi32(max_loop_cycle_u32x16, max_changed_mask,
                                                      current_loop_cycle_u32x16);
    }

    nk_i32_t min_value = nk_reduce_min_i32x16_skylake_(min_i32x16);
    nk_i32_t max_value = nk_reduce_max_i32x16_skylake_(max_i32x16);
    unsigned int min_lane, max_lane;
    {
        __mmask16 value_match_mask = _mm512_cmpeq_epi32_mask(min_i32x16, _mm512_set1_epi32(min_value));
        __m512i masked_cycle_u32x16 = _mm512_mask_blend_epi32(value_match_mask, _mm512_set1_epi32((int)NK_U32_MAX),
                                                              min_loop_cycle_u32x16);
        nk_u32_t earliest_loop_cycle = nk_reduce_min_u32x16_skylake_(masked_cycle_u32x16);
        __mmask16 cycle_match_mask = _mm512_cmpeq_epi32_mask(masked_cycle_u32x16,
                                                             _mm512_set1_epi32((int)earliest_loop_cycle));
        min_lane = _tzcnt_u32(cycle_match_mask);
    }
    {
        __mmask16 value_match_mask = _mm512_cmpeq_epi32_mask(max_i32x16, _mm512_set1_epi32(max_value));
        __m512i masked_cycle_u32x16 = _mm512_mask_blend_epi32(value_match_mask, _mm512_set1_epi32((int)NK_U32_MAX),
                                                              max_loop_cycle_u32x16);
        nk_u32_t earliest_loop_cycle = nk_reduce_min_u32x16_skylake_(masked_cycle_u32x16);
        __mmask16 cycle_match_mask = _mm512_cmpeq_epi32_mask(masked_cycle_u32x16,
                                                             _mm512_set1_epi32((int)earliest_loop_cycle));
        max_lane = _tzcnt_u32(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u32x16;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u32s[min_lane] * 16 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u32x16;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u32s[max_lane] * 16 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i32_skylake(                           //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 16) {
        nk_size_t left_count = count / 2;
        nk_i32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i32_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_i32_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_min, &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i32_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                 max_index_ptr);
    else
        nk_reduce_minmax_i32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u32_skylake_contiguous_( //
    nk_u32_t const *data_ptr, nk_size_t count,              //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum: widen u32→u64, accumulate. Sumsq: VPMULUDQ for even/odd lanes (5-cycle, 1 uop each).
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    __mmask8 sumsq_overflow_mask = 0;
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_loadu_si512(data_ptr + idx);
        __m256i low_u32x8 = _mm512_castsi512_si256(data_u32x16);
        __m256i high_u32x8 = _mm512_extracti64x4_epi64(data_u32x16, 1);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(low_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(high_u32x8));
        __m512i even_sq_u64x8 = _mm512_mul_epu32(data_u32x16, data_u32x16);
        __m512i odd_u32x16 = _mm512_srli_epi64(data_u32x16, 32);
        __m512i odd_sq_u64x8 = _mm512_mul_epu32(odd_u32x16, odd_u32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, even_sq_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, even_sq_u64x8, _MM_CMPINT_LT);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, odd_sq_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, odd_sq_u64x8, _MM_CMPINT_LT);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i data_u32x16 = _mm512_maskz_loadu_epi32(tail_mask, data_ptr + idx);
        __m256i low_u32x8 = _mm512_castsi512_si256(data_u32x16);
        __m256i high_u32x8 = _mm512_extracti64x4_epi64(data_u32x16, 1);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(low_u32x8));
        if (remaining > 8) sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(high_u32x8));
        __m512i even_sq_u64x8 = _mm512_mul_epu32(data_u32x16, data_u32x16);
        __m512i odd_u32x16 = _mm512_srli_epi64(data_u32x16, 32);
        __m512i odd_sq_u64x8 = _mm512_mul_epu32(odd_u32x16, odd_u32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, even_sq_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, even_sq_u64x8, _MM_CMPINT_LT);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, odd_sq_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, odd_sq_u64x8, _MM_CMPINT_LT);
    }
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    nk_u64_t sumsq;
    if (sumsq_overflow_mask) sumsq = NK_U64_MAX;
    else sumsq = nk_reduce_sadd_u64x8_skylake_(sumsq_u64x8);
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u32_skylake(                          //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u32_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u32_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_sum, &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u32_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u32_skylake_contiguous_( //
    nk_u32_t const *data_ptr, nk_size_t count,             //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i min_u32x16 = _mm512_set1_epi32((nk_i32_t)NK_U32_MAX);
    __m512i max_u32x16 = _mm512_setzero_si512();
    __m512i min_loop_cycle_u32x16 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u32x16 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u32x16 = _mm512_setzero_si512();
    __m512i one_u32x16 = _mm512_set1_epi32(1);

    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_loadu_si512(data_ptr + idx);
        __mmask16 min_changed_mask = _mm512_cmp_epu32_mask(data_u32x16, min_u32x16, _MM_CMPINT_LT);
        __mmask16 max_changed_mask = _mm512_cmp_epu32_mask(data_u32x16, max_u32x16, _MM_CMPINT_NLE);
        min_u32x16 = _mm512_mask_mov_epi32(min_u32x16, min_changed_mask, data_u32x16);
        max_u32x16 = _mm512_mask_mov_epi32(max_u32x16, max_changed_mask, data_u32x16);
        min_loop_cycle_u32x16 = _mm512_mask_mov_epi32(min_loop_cycle_u32x16, min_changed_mask,
                                                      current_loop_cycle_u32x16);
        max_loop_cycle_u32x16 = _mm512_mask_mov_epi32(max_loop_cycle_u32x16, max_changed_mask,
                                                      current_loop_cycle_u32x16);
        current_loop_cycle_u32x16 = _mm512_add_epi32(current_loop_cycle_u32x16, one_u32x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_load = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i tail_u32x16 = _mm512_maskz_loadu_epi32(tail_load, data_ptr + idx);
        __mmask16 min_changed_mask = _mm512_mask_cmp_epu32_mask(tail_load, tail_u32x16, min_u32x16, _MM_CMPINT_LT);
        __mmask16 max_changed_mask = _mm512_mask_cmp_epu32_mask(tail_load, tail_u32x16, max_u32x16, _MM_CMPINT_NLE);
        min_u32x16 = _mm512_mask_mov_epi32(min_u32x16, min_changed_mask, tail_u32x16);
        max_u32x16 = _mm512_mask_mov_epi32(max_u32x16, max_changed_mask, tail_u32x16);
        min_loop_cycle_u32x16 = _mm512_mask_mov_epi32(min_loop_cycle_u32x16, min_changed_mask,
                                                      current_loop_cycle_u32x16);
        max_loop_cycle_u32x16 = _mm512_mask_mov_epi32(max_loop_cycle_u32x16, max_changed_mask,
                                                      current_loop_cycle_u32x16);
    }

    nk_u32_t min_value = nk_reduce_min_u32x16_skylake_(min_u32x16);
    nk_u32_t max_value = nk_reduce_max_u32x16_skylake_(max_u32x16);
    unsigned int min_lane, max_lane;
    {
        __mmask16 value_match_mask = _mm512_cmpeq_epi32_mask(min_u32x16, _mm512_set1_epi32((nk_i32_t)min_value));
        __m512i masked_cycle_u32x16 = _mm512_mask_blend_epi32(value_match_mask, _mm512_set1_epi32((int)NK_U32_MAX),
                                                              min_loop_cycle_u32x16);
        nk_u32_t earliest_loop_cycle = nk_reduce_min_u32x16_skylake_(masked_cycle_u32x16);
        __mmask16 cycle_match_mask = _mm512_cmpeq_epi32_mask(masked_cycle_u32x16,
                                                             _mm512_set1_epi32((int)earliest_loop_cycle));
        min_lane = _tzcnt_u32(cycle_match_mask);
    }
    {
        __mmask16 value_match_mask = _mm512_cmpeq_epi32_mask(max_u32x16, _mm512_set1_epi32((nk_i32_t)max_value));
        __m512i masked_cycle_u32x16 = _mm512_mask_blend_epi32(value_match_mask, _mm512_set1_epi32((int)NK_U32_MAX),
                                                              max_loop_cycle_u32x16);
        nk_u32_t earliest_loop_cycle = nk_reduce_min_u32x16_skylake_(masked_cycle_u32x16);
        __mmask16 cycle_match_mask = _mm512_cmpeq_epi32_mask(masked_cycle_u32x16,
                                                             _mm512_set1_epi32((int)earliest_loop_cycle));
        max_lane = _tzcnt_u32(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u32x16;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u32s[min_lane] * 16 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u32x16;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u32s[max_lane] * 16 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u32_skylake(                           //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 16) {
        nk_size_t left_count = count / 2;
        nk_u32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u32_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_u32_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_min, &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u32_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                 max_index_ptr);
    else
        nk_reduce_minmax_u32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i64_skylake_contiguous_( //
    nk_i64_t const *data_ptr, nk_size_t count,              //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum: double-width 128-bit accumulation per lane.
    // Sumsq: unsigned wrapping accumulation with carry-based overflow detection.
    __m512i sum_lower_i64x8 = _mm512_setzero_si512();
    __m512i sum_upper_i64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    __mmask8 sumsq_overflow_mask = 0;
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_i64x8 = _mm512_loadu_si512(data_ptr + idx);
        __m512i squared_i64x8 = nk_i64_smul_sq_epi64_skylake_(data_i64x8);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, squared_i64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, squared_i64x8, _MM_CMPINT_LT);
        __m512i sum_before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, data_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, sum_before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, _mm512_srai_epi64(data_i64x8, 63));
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i data_i64x8 = _mm512_maskz_loadu_epi64(tail_mask, data_ptr + idx);
        __m512i squared_i64x8 = nk_i64_smul_sq_epi64_skylake_(data_i64x8);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, squared_i64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, squared_i64x8, _MM_CMPINT_LT);
        __m512i sum_before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, data_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, sum_before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, _mm512_srai_epi64(data_i64x8, 63));
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
    }
    // Sumsq: horizontal unsigned saturating reduction
    nk_u64_t sumsq;
    if (sumsq_overflow_mask) sumsq = NK_U64_MAX;
    else sumsq = nk_reduce_sadd_u64x8_skylake_(sumsq_u64x8);
    // Sum: horizontal 128-bit tree reduction (8→4→2→1), then clamp to i64
    { // 8→4: fold high 256 bits into low 256 bits
        __m512i fold_lower_i64x8 = _mm512_shuffle_i64x2(sum_lower_i64x8, sum_lower_i64x8, _MM_SHUFFLE(1, 0, 3, 2));
        __m512i fold_upper_i64x8 = _mm512_shuffle_i64x2(sum_upper_i64x8, sum_upper_i64x8, _MM_SHUFFLE(1, 0, 3, 2));
        __m512i before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, fold_lower_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, fold_upper_i64x8);
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
    }
    { // 4→2: fold lanes 2-3 into lanes 0-1
        __m512i fold_lower_i64x8 = _mm512_shuffle_i64x2(sum_lower_i64x8, sum_lower_i64x8, _MM_SHUFFLE(2, 3, 0, 1));
        __m512i fold_upper_i64x8 = _mm512_shuffle_i64x2(sum_upper_i64x8, sum_upper_i64x8, _MM_SHUFFLE(2, 3, 0, 1));
        __m512i before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, fold_lower_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, fold_upper_i64x8);
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
    }
    { // 2→1: fold lane 1 into lane 0
        __m512i fold_lower_i64x8 = _mm512_alignr_epi64(sum_lower_i64x8, sum_lower_i64x8, 1);
        __m512i fold_upper_i64x8 = _mm512_alignr_epi64(sum_upper_i64x8, sum_upper_i64x8, 1);
        __m512i before_i64x8 = sum_lower_i64x8;
        sum_lower_i64x8 = _mm512_add_epi64(sum_lower_i64x8, fold_lower_i64x8);
        __mmask8 carry = _mm512_cmp_epu64_mask(sum_lower_i64x8, before_i64x8, _MM_CMPINT_LT);
        sum_upper_i64x8 = _mm512_add_epi64(sum_upper_i64x8, fold_upper_i64x8);
        sum_upper_i64x8 = _mm512_mask_add_epi64(sum_upper_i64x8, carry, sum_upper_i64x8, one_i64x8);
    }
    // Clamp 128-bit result to [INT64_MIN, INT64_MAX]: fits iff upper == sign-extension of lower
    nk_i64_t sum_lower = _mm_cvtsi128_si64(_mm512_castsi512_si128(sum_lower_i64x8));
    nk_i64_t sum_upper = _mm_cvtsi128_si64(_mm512_castsi512_si128(sum_upper_i64x8));
    if (sum_upper == (sum_lower >> 63)) *sum_ptr = sum_lower;
    else if (sum_upper >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i64_skylake(                          //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_i64_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i64_skylake_contiguous_( //
    nk_i64_t const *data_ptr, nk_size_t count,             //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i min_i64x8 = _mm512_set1_epi64(NK_I64_MAX);
    __m512i max_i64x8 = _mm512_set1_epi64(NK_I64_MIN);
    __m512i min_loop_cycle_u64x8 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u64x8 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u64x8 = _mm512_setzero_si512();
    __m512i one_u64x8 = _mm512_set1_epi64(1);

    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_i64x8 = _mm512_loadu_si512(data_ptr + idx);
        __mmask8 min_changed_mask = _mm512_cmp_epi64_mask(data_i64x8, min_i64x8, _MM_CMPINT_LT);
        __mmask8 max_changed_mask = _mm512_cmp_epi64_mask(data_i64x8, max_i64x8, _MM_CMPINT_NLE);
        min_i64x8 = _mm512_mask_mov_epi64(min_i64x8, min_changed_mask, data_i64x8);
        max_i64x8 = _mm512_mask_mov_epi64(max_i64x8, max_changed_mask, data_i64x8);
        min_loop_cycle_u64x8 = _mm512_mask_mov_epi64(min_loop_cycle_u64x8, min_changed_mask, current_loop_cycle_u64x8);
        max_loop_cycle_u64x8 = _mm512_mask_mov_epi64(max_loop_cycle_u64x8, max_changed_mask, current_loop_cycle_u64x8);
        current_loop_cycle_u64x8 = _mm512_add_epi64(current_loop_cycle_u64x8, one_u64x8);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_load = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i tail_i64x8 = _mm512_maskz_loadu_epi64(tail_load, data_ptr + idx);
        __mmask8 min_changed_mask = _mm512_mask_cmp_epi64_mask(tail_load, tail_i64x8, min_i64x8, _MM_CMPINT_LT);
        __mmask8 max_changed_mask = _mm512_mask_cmp_epi64_mask(tail_load, tail_i64x8, max_i64x8, _MM_CMPINT_NLE);
        min_i64x8 = _mm512_mask_mov_epi64(min_i64x8, min_changed_mask, tail_i64x8);
        max_i64x8 = _mm512_mask_mov_epi64(max_i64x8, max_changed_mask, tail_i64x8);
        min_loop_cycle_u64x8 = _mm512_mask_mov_epi64(min_loop_cycle_u64x8, min_changed_mask, current_loop_cycle_u64x8);
        max_loop_cycle_u64x8 = _mm512_mask_mov_epi64(max_loop_cycle_u64x8, max_changed_mask, current_loop_cycle_u64x8);
    }

    nk_i64_t min_value = nk_reduce_min_i64x8_skylake_(min_i64x8);
    nk_i64_t max_value = nk_reduce_max_i64x8_skylake_(max_i64x8);
    unsigned int min_lane, max_lane;
    {
        __mmask8 value_match_mask = _mm512_cmpeq_epi64_mask(min_i64x8, _mm512_set1_epi64(min_value));
        __m512i masked_cycle_u64x8 = _mm512_mask_blend_epi64(value_match_mask, _mm512_set1_epi64((nk_i64_t)NK_U64_MAX),
                                                             min_loop_cycle_u64x8);
        nk_u64_t earliest_loop_cycle = nk_reduce_min_u64x8_skylake_(masked_cycle_u64x8);
        __mmask8 cycle_match_mask = _mm512_cmpeq_epi64_mask(masked_cycle_u64x8,
                                                            _mm512_set1_epi64((nk_i64_t)earliest_loop_cycle));
        min_lane = _tzcnt_u32((unsigned int)cycle_match_mask);
    }
    {
        __mmask8 value_match_mask = _mm512_cmpeq_epi64_mask(max_i64x8, _mm512_set1_epi64(max_value));
        __m512i masked_cycle_u64x8 = _mm512_mask_blend_epi64(value_match_mask, _mm512_set1_epi64((nk_i64_t)NK_U64_MAX),
                                                             max_loop_cycle_u64x8);
        nk_u64_t earliest_loop_cycle = nk_reduce_min_u64x8_skylake_(masked_cycle_u64x8);
        __mmask8 cycle_match_mask = _mm512_cmpeq_epi64_mask(masked_cycle_u64x8,
                                                            _mm512_set1_epi64((nk_i64_t)earliest_loop_cycle));
        max_lane = _tzcnt_u32((unsigned int)cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u64x8;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u64s[min_lane] * 8 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u64x8;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u64s[max_lane] * 8 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i64_skylake(                           //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i64_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                 max_index_ptr);
    else
        nk_reduce_minmax_i64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u64_skylake_contiguous_( //
    nk_u64_t const *data_ptr, nk_size_t count,              //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Unsigned saturating addition is order-independent: sat(sat(a+b)+c) == sat(a+b+c).
    // Once a lane saturates it stays saturated, so a running overflow mask is sufficient
    // for any count — no block cap or 128-bit accumulation needed.
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    __mmask8 sum_overflow_mask = 0, sumsq_overflow_mask = 0;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_u64x8 = _mm512_loadu_si512(data_ptr + idx);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, data_u64x8);
        sum_overflow_mask |= _mm512_cmp_epu64_mask(sum_u64x8, data_u64x8, _MM_CMPINT_LT);
        __m512i squared_u64x8 = nk_u64_smul_sq_epi64_skylake_(data_u64x8);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, squared_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, squared_u64x8, _MM_CMPINT_LT);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i data_u64x8 = _mm512_maskz_loadu_epi64(tail_mask, data_ptr + idx);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, data_u64x8);
        sum_overflow_mask |= _mm512_cmp_epu64_mask(sum_u64x8, data_u64x8, _MM_CMPINT_LT);
        __m512i squared_u64x8 = nk_u64_smul_sq_epi64_skylake_(data_u64x8);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, squared_u64x8);
        sumsq_overflow_mask |= _mm512_cmp_epu64_mask(sumsq_u64x8, squared_u64x8, _MM_CMPINT_LT);
    }
    nk_u64_t sum_scalar;
    if (sum_overflow_mask) sum_scalar = NK_U64_MAX;
    else sum_scalar = nk_reduce_sadd_u64x8_skylake_(sum_u64x8);
    nk_u64_t sumsq_scalar;
    if (sumsq_overflow_mask) sumsq_scalar = NK_U64_MAX;
    else sumsq_scalar = nk_reduce_sadd_u64x8_skylake_(sumsq_u64x8);
    *sum_ptr = sum_scalar, *sumsq_ptr = sumsq_scalar;
}

NK_PUBLIC void nk_reduce_moments_u64_skylake(                          //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_u64_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u64_skylake_contiguous_( //
    nk_u64_t const *data_ptr, nk_size_t count,             //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i min_u64x8 = _mm512_set1_epi64((nk_i64_t)NK_U64_MAX);
    __m512i max_u64x8 = _mm512_setzero_si512();
    __m512i min_loop_cycle_u64x8 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u64x8 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u64x8 = _mm512_setzero_si512();
    __m512i one_u64x8 = _mm512_set1_epi64(1);

    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_u64x8 = _mm512_loadu_si512(data_ptr + idx);
        __mmask8 min_changed_mask = _mm512_cmp_epu64_mask(data_u64x8, min_u64x8, _MM_CMPINT_LT);
        __mmask8 max_changed_mask = _mm512_cmp_epu64_mask(data_u64x8, max_u64x8, _MM_CMPINT_NLE);
        min_u64x8 = _mm512_mask_mov_epi64(min_u64x8, min_changed_mask, data_u64x8);
        max_u64x8 = _mm512_mask_mov_epi64(max_u64x8, max_changed_mask, data_u64x8);
        min_loop_cycle_u64x8 = _mm512_mask_mov_epi64(min_loop_cycle_u64x8, min_changed_mask, current_loop_cycle_u64x8);
        max_loop_cycle_u64x8 = _mm512_mask_mov_epi64(max_loop_cycle_u64x8, max_changed_mask, current_loop_cycle_u64x8);
        current_loop_cycle_u64x8 = _mm512_add_epi64(current_loop_cycle_u64x8, one_u64x8);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_load = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i tail_u64x8 = _mm512_maskz_loadu_epi64(tail_load, data_ptr + idx);
        __mmask8 min_changed_mask = _mm512_mask_cmp_epu64_mask(tail_load, tail_u64x8, min_u64x8, _MM_CMPINT_LT);
        __mmask8 max_changed_mask = _mm512_mask_cmp_epu64_mask(tail_load, tail_u64x8, max_u64x8, _MM_CMPINT_NLE);
        min_u64x8 = _mm512_mask_mov_epi64(min_u64x8, min_changed_mask, tail_u64x8);
        max_u64x8 = _mm512_mask_mov_epi64(max_u64x8, max_changed_mask, tail_u64x8);
        min_loop_cycle_u64x8 = _mm512_mask_mov_epi64(min_loop_cycle_u64x8, min_changed_mask, current_loop_cycle_u64x8);
        max_loop_cycle_u64x8 = _mm512_mask_mov_epi64(max_loop_cycle_u64x8, max_changed_mask, current_loop_cycle_u64x8);
    }

    nk_u64_t min_value = nk_reduce_min_u64x8_skylake_(min_u64x8);
    nk_u64_t max_value = nk_reduce_max_u64x8_skylake_(max_u64x8);
    unsigned int min_lane, max_lane;
    {
        __mmask8 value_match_mask = _mm512_cmpeq_epi64_mask(min_u64x8, _mm512_set1_epi64((nk_i64_t)min_value));
        __m512i masked_cycle_u64x8 = _mm512_mask_blend_epi64(value_match_mask, _mm512_set1_epi64((nk_i64_t)NK_U64_MAX),
                                                             min_loop_cycle_u64x8);
        nk_u64_t earliest_loop_cycle = nk_reduce_min_u64x8_skylake_(masked_cycle_u64x8);
        __mmask8 cycle_match_mask = _mm512_cmpeq_epi64_mask(masked_cycle_u64x8,
                                                            _mm512_set1_epi64((nk_i64_t)earliest_loop_cycle));
        min_lane = _tzcnt_u32((unsigned int)cycle_match_mask);
    }
    {
        __mmask8 value_match_mask = _mm512_cmpeq_epi64_mask(max_u64x8, _mm512_set1_epi64((nk_i64_t)max_value));
        __m512i masked_cycle_u64x8 = _mm512_mask_blend_epi64(value_match_mask, _mm512_set1_epi64((nk_i64_t)NK_U64_MAX),
                                                             max_loop_cycle_u64x8);
        nk_u64_t earliest_loop_cycle = nk_reduce_min_u64x8_skylake_(masked_cycle_u64x8);
        __mmask8 cycle_match_mask = _mm512_cmpeq_epi64_mask(masked_cycle_u64x8,
                                                            _mm512_set1_epi64((nk_i64_t)earliest_loop_cycle));
        max_lane = _tzcnt_u32((unsigned int)cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u64x8;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u64s[min_lane] * 8 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u64x8;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u64s[max_lane] * 8 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u64_skylake(                           //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u64_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                 max_index_ptr);
    else
        nk_reduce_minmax_u64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_f64_skylake_contiguous_( //
    nk_f64_t const *data_ptr, nk_size_t count,             //
    nk_f64_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_f64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512d min_f64x8 = _mm512_set1_pd(NK_F64_MAX);
    __m512d max_f64x8 = _mm512_set1_pd(NK_F64_MIN);
    __m512i min_loop_cycle_u64x8 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u64x8 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u64x8 = _mm512_setzero_si512();
    __m512i one_u64x8 = _mm512_set1_epi64(1);

    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data_ptr + idx);
        __mmask8 min_changed_mask = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        __mmask8 max_changed_mask = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, min_changed_mask, data_f64x8);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, max_changed_mask, data_f64x8);
        min_loop_cycle_u64x8 = _mm512_mask_mov_epi64(min_loop_cycle_u64x8, min_changed_mask, current_loop_cycle_u64x8);
        max_loop_cycle_u64x8 = _mm512_mask_mov_epi64(max_loop_cycle_u64x8, max_changed_mask, current_loop_cycle_u64x8);
        current_loop_cycle_u64x8 = _mm512_add_epi64(current_loop_cycle_u64x8, one_u64x8);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_load = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_load, data_ptr + idx);
        __mmask8 min_changed_mask = _mm512_mask_cmp_pd_mask(tail_load, tail_f64x8, min_f64x8, _CMP_LT_OQ);
        __mmask8 max_changed_mask = _mm512_mask_cmp_pd_mask(tail_load, tail_f64x8, max_f64x8, _CMP_GT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, min_changed_mask, tail_f64x8);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, max_changed_mask, tail_f64x8);
        min_loop_cycle_u64x8 = _mm512_mask_mov_epi64(min_loop_cycle_u64x8, min_changed_mask, current_loop_cycle_u64x8);
        max_loop_cycle_u64x8 = _mm512_mask_mov_epi64(max_loop_cycle_u64x8, max_changed_mask, current_loop_cycle_u64x8);
    }

    nk_f64_t min_value = nk_reduce_min_f64x8_skylake_(min_f64x8);
    nk_f64_t max_value = nk_reduce_max_f64x8_skylake_(max_f64x8);
    if (min_value == NK_F64_MAX && max_value == NK_F64_MIN) {
        *min_value_ptr = NK_F64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    unsigned int min_lane, max_lane;
    {
        __mmask8 value_match_mask = _mm512_cmp_pd_mask(min_f64x8, _mm512_set1_pd(min_value), _CMP_EQ_OQ);
        __m512i masked_cycle_u64x8 = _mm512_mask_blend_epi64(value_match_mask, _mm512_set1_epi64((nk_i64_t)NK_U64_MAX),
                                                             min_loop_cycle_u64x8);
        nk_u64_t earliest_loop_cycle = nk_reduce_min_u64x8_skylake_(masked_cycle_u64x8);
        __mmask8 cycle_match_mask = _mm512_cmpeq_epi64_mask(masked_cycle_u64x8,
                                                            _mm512_set1_epi64((nk_i64_t)earliest_loop_cycle));
        min_lane = _tzcnt_u32((unsigned int)cycle_match_mask);
    }
    {
        __mmask8 value_match_mask = _mm512_cmp_pd_mask(max_f64x8, _mm512_set1_pd(max_value), _CMP_EQ_OQ);
        __m512i masked_cycle_u64x8 = _mm512_mask_blend_epi64(value_match_mask, _mm512_set1_epi64((nk_i64_t)NK_U64_MAX),
                                                             max_loop_cycle_u64x8);
        nk_u64_t earliest_loop_cycle = nk_reduce_min_u64x8_skylake_(masked_cycle_u64x8);
        __mmask8 cycle_match_mask = _mm512_cmpeq_epi64_mask(masked_cycle_u64x8,
                                                            _mm512_set1_epi64((nk_i64_t)earliest_loop_cycle));
        max_lane = _tzcnt_u32((unsigned int)cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u64x8;
    *min_value_ptr = min_value, *min_index_ptr = (nk_size_t)loop_cycle_vec.u64s[min_lane] * 8 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u64x8;
    *max_value_ptr = max_value, *max_index_ptr = (nk_size_t)loop_cycle_vec.u64s[max_lane] * 8 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_f64_skylake(                           //
    nk_f64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_F64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_f64_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                 max_index_ptr);
    else
        nk_reduce_minmax_f64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e4m3_skylake_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data_ptr + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e4m3x16_to_f32x16_skylake_(data_ptr + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
        sumsq_f32x16 = _mm512_fmadd_ps(vec.zmm_ps, vec.zmm_ps, sumsq_f32x16);
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_INTERNAL void nk_reduce_moments_e4m3_skylake_strided_(                  //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m16) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m128i data_e4m3x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data_ptr + idx_scalars);
        __m512 data_f32x16 = nk_e4m3x16_to_f32x16_skylake_(data_e4m3x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e4m3x16 = _mm_maskz_loadu_epi8(tail_mask, data_ptr + idx_scalars);
        __m512 data_f32x16 = nk_e4m3x16_to_f32x16_skylake_(data_e4m3x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e4m3_skylake(                          //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e4m3_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e4m3_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e4m3_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_moments_e4m3_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e4m3_skylake_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,             //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    // E4M3 NaN: comparable 0x00 (neg NaN) and 0xFF (pos NaN). Replace with neutral values.
    nk_b512_vec_t min_vec, max_vec;
    min_vec.zmm = _mm512_set1_epi8((char)0xFF);
    max_vec.zmm = _mm512_setzero_si512();
    __m512i min_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data_ptr + idx);
        __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 is_nan_m64 = _mm512_cmpeq_epi8_mask(data_cmp_u8x64, _mm512_setzero_si512()) |
                               _mm512_cmpeq_epi8_mask(data_cmp_u8x64, _mm512_set1_epi8((char)0xFF));
        __m512i data_min_u8x64 = _mm512_mask_blend_epi8(is_nan_m64, data_cmp_u8x64, _mm512_set1_epi8((char)0xFF));
        __m512i data_max_u8x64 = _mm512_mask_blend_epi8(is_nan_m64, data_cmp_u8x64, _mm512_setzero_si512());
        __m512i new_min_u8x64 = _mm512_min_epu8(min_vec.zmm, data_min_u8x64);
        __mmask64 min_changed_mask = ~_mm512_cmpeq_epi8_mask(new_min_u8x64, min_vec.zmm);
        min_vec.zmm = new_min_u8x64;
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        __m512i new_max_u8x64 = _mm512_max_epu8(max_vec.zmm, data_max_u8x64);
        __mmask64 max_changed_mask = ~_mm512_cmpeq_epi8_mask(new_max_u8x64, max_vec.zmm);
        max_vec.zmm = new_max_u8x64;
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
        current_loop_cycle_u8x64 = _mm512_add_epi8(current_loop_cycle_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8((char)0xFF), tail_load, data_ptr + idx);
        __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 is_nan_m64 = _mm512_cmpeq_epi8_mask(data_cmp_u8x64, _mm512_setzero_si512()) |
                               _mm512_cmpeq_epi8_mask(data_cmp_u8x64, _mm512_set1_epi8((char)0xFF));
        __mmask64 valid_non_nan_m64 = tail_load & ~is_nan_m64;
        __m512i data_cmp_min = _mm512_mask_blend_epi8(valid_non_nan_m64, _mm512_set1_epi8((char)0xFF), data_cmp_u8x64);
        __m512i data_cmp_max = _mm512_mask_blend_epi8(valid_non_nan_m64, _mm512_setzero_si512(), data_cmp_u8x64);
        __m512i new_min_u8x64 = _mm512_min_epu8(min_vec.zmm, data_cmp_min);
        __mmask64 min_changed_mask = ~_mm512_cmpeq_epi8_mask(new_min_u8x64, min_vec.zmm);
        min_vec.zmm = new_min_u8x64;
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        __m512i new_max_u8x64 = _mm512_max_epu8(max_vec.zmm, data_cmp_max);
        __mmask64 max_changed_mask = ~_mm512_cmpeq_epi8_mask(new_max_u8x64, max_vec.zmm);
        max_vec.zmm = new_max_u8x64;
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
    }

    nk_u8_t min_value_comparable = nk_reduce_min_u8x64_skylake_(min_vec.zmm);
    nk_u8_t max_value_comparable = nk_reduce_max_u8x64_skylake_(max_vec.zmm);

    // All-NaN early return: both sentinels unchanged means no valid data was found
    if (min_value_comparable == 0xFF && max_value_comparable == 0x00) {
        *min_value_ptr = NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = NK_E4M3_MIN, *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    if (min_value_comparable == 0xFF) { *min_value_ptr = NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX; }
    else {
        unsigned int min_lane;
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(min_vec.zmm, _mm512_set1_epi8((char)min_value_comparable));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            min_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        min_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
        nk_b512_vec_t loop_cycle_vec;
        loop_cycle_vec.zmm = min_loop_cycle_u8x64;
        *min_index_ptr = (nk_size_t)loop_cycle_vec.u8s[min_lane] * 64 + min_lane;
        nk_b512_vec_t min_raw_vec;
        min_raw_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(min_vec.zmm);
        *min_value_ptr = min_raw_vec.e4m3s[min_lane];
    }
    if (max_value_comparable == 0x00) { *max_value_ptr = NK_E4M3_MIN, *max_index_ptr = NK_SIZE_MAX; }
    else {
        unsigned int max_lane;
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(max_vec.zmm, _mm512_set1_epi8((char)max_value_comparable));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            max_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        max_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
        nk_b512_vec_t loop_cycle_vec;
        loop_cycle_vec.zmm = max_loop_cycle_u8x64;
        *max_index_ptr = (nk_size_t)loop_cycle_vec.u8s[max_lane] * 64 + max_lane;
        nk_b512_vec_t max_raw_vec;
        max_raw_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(max_vec.zmm);
        *max_value_ptr = max_raw_vec.e4m3s[max_lane];
    }
}

NK_PUBLIC void nk_reduce_minmax_e4m3_skylake(                           //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    if (count == 0)
        *min_value_ptr = NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E4M3_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_e4m3_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e4m3_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e4m3_skylake(data_ptr + left_count, count - left_count, stride_bytes, &right_min,
                                      &right_min_index, &right_max, &right_max_index);
        if (left_min_index == NK_SIZE_MAX)
            *min_value_ptr = right_min,
            *min_index_ptr = right_min_index == NK_SIZE_MAX ? NK_SIZE_MAX : left_count + right_min_index;
        else if (right_min_index == NK_SIZE_MAX || nk_e4m3_order_serial(left_min, right_min) <= 0)
            *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        else *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        if (left_max_index == NK_SIZE_MAX)
            *max_value_ptr = right_max,
            *max_index_ptr = right_max_index == NK_SIZE_MAX ? NK_SIZE_MAX : left_count + right_max_index;
        else if (right_max_index == NK_SIZE_MAX || nk_e4m3_order_serial(left_max, right_max) >= 0)
            *max_value_ptr = left_max, *max_index_ptr = left_max_index;
        else *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e4m3_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                  max_index_ptr);
    else
        nk_reduce_minmax_e4m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e5m2_skylake_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e5m2x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data_ptr + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e5m2x16_to_f32x16_skylake_(data_ptr + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
        sumsq_f32x16 = _mm512_fmadd_ps(vec.zmm_ps, vec.zmm_ps, sumsq_f32x16);
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_INTERNAL void nk_reduce_moments_e5m2_skylake_strided_(                  //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m16) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m128i data_e5m2x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data_ptr + idx_scalars);
        __m512 data_f32x16 = nk_e5m2x16_to_f32x16_skylake_(data_e5m2x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e5m2x16 = _mm_maskz_loadu_epi8(tail_mask, data_ptr + idx_scalars);
        __m512 data_f32x16 = nk_e5m2x16_to_f32x16_skylake_(data_e5m2x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e5m2_skylake(                          //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e5m2_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e5m2_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e5m2_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_moments_e5m2_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e2m3_skylake_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e2m3x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data_ptr + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e2m3x16_to_f32x16_skylake_(data_ptr + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
        sumsq_f32x16 = _mm512_fmadd_ps(vec.zmm_ps, vec.zmm_ps, sumsq_f32x16);
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_INTERNAL void nk_reduce_moments_e2m3_skylake_strided_(                  //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m16) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m128i data_e2m3x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data_ptr + idx_scalars);
        __m512 data_f32x16 = nk_e2m3x16_to_f32x16_skylake_(data_e2m3x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e2m3x16 = _mm_maskz_loadu_epi8(tail_mask, data_ptr + idx_scalars);
        __m512 data_f32x16 = nk_e2m3x16_to_f32x16_skylake_(data_e2m3x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e2m3_skylake(                          //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e2m3_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e2m3_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_moments_e2m3_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e3m2_skylake_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e3m2x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data_ptr + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e3m2x16_to_f32x16_skylake_(data_ptr + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
        sumsq_f32x16 = _mm512_fmadd_ps(vec.zmm_ps, vec.zmm_ps, sumsq_f32x16);
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_INTERNAL void nk_reduce_moments_e3m2_skylake_strided_(                  //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m16) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m128i data_e3m2x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data_ptr + idx_scalars);
        __m512 data_f32x16 = nk_e3m2x16_to_f32x16_skylake_(data_e3m2x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e3m2x16 = _mm_maskz_loadu_epi8(tail_mask, data_ptr + idx_scalars);
        __m512 data_f32x16 = nk_e3m2x16_to_f32x16_skylake_(data_e3m2x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e3m2_skylake(                          //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e3m2_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e3m2_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e3m2_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_moments_e3m2_skylake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_skylake_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,             //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    // E5M2 NaN in comparable form: 0x00-0x02 (neg NaN) and 0xFD-0xFF (pos NaN).
    nk_b512_vec_t min_vec, max_vec;
    min_vec.zmm = _mm512_set1_epi8((char)0xFF);
    max_vec.zmm = _mm512_setzero_si512();
    __m512i min_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data_ptr + idx);
        __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 is_nan_m64 = _mm512_cmple_epu8_mask(data_cmp_u8x64, _mm512_set1_epi8(0x02)) |
                               _mm512_cmpge_epu8_mask(data_cmp_u8x64, _mm512_set1_epi8((char)0xFD));
        __m512i data_min_u8x64 = _mm512_mask_blend_epi8(is_nan_m64, data_cmp_u8x64, _mm512_set1_epi8((char)0xFF));
        __m512i data_max_u8x64 = _mm512_mask_blend_epi8(is_nan_m64, data_cmp_u8x64, _mm512_setzero_si512());
        __m512i new_min_u8x64 = _mm512_min_epu8(min_vec.zmm, data_min_u8x64);
        __mmask64 min_changed_mask = ~_mm512_cmpeq_epi8_mask(new_min_u8x64, min_vec.zmm);
        min_vec.zmm = new_min_u8x64;
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        __m512i new_max_u8x64 = _mm512_max_epu8(max_vec.zmm, data_max_u8x64);
        __mmask64 max_changed_mask = ~_mm512_cmpeq_epi8_mask(new_max_u8x64, max_vec.zmm);
        max_vec.zmm = new_max_u8x64;
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
        current_loop_cycle_u8x64 = _mm512_add_epi8(current_loop_cycle_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8((char)0xFF), tail_load, data_ptr + idx);
        __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 is_nan_m64 = _mm512_cmple_epu8_mask(data_cmp_u8x64, _mm512_set1_epi8(0x02)) |
                               _mm512_cmpge_epu8_mask(data_cmp_u8x64, _mm512_set1_epi8((char)0xFD));
        __mmask64 valid_non_nan_m64 = tail_load & ~is_nan_m64;
        __m512i data_cmp_min = _mm512_mask_blend_epi8(valid_non_nan_m64, _mm512_set1_epi8((char)0xFF), data_cmp_u8x64);
        __m512i data_cmp_max = _mm512_mask_blend_epi8(valid_non_nan_m64, _mm512_setzero_si512(), data_cmp_u8x64);
        __m512i new_min_u8x64 = _mm512_min_epu8(min_vec.zmm, data_cmp_min);
        __mmask64 min_changed_mask = ~_mm512_cmpeq_epi8_mask(new_min_u8x64, min_vec.zmm);
        min_vec.zmm = new_min_u8x64;
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        __m512i new_max_u8x64 = _mm512_max_epu8(max_vec.zmm, data_cmp_max);
        __mmask64 max_changed_mask = ~_mm512_cmpeq_epi8_mask(new_max_u8x64, max_vec.zmm);
        max_vec.zmm = new_max_u8x64;
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
    }

    nk_u8_t min_value_comparable = nk_reduce_min_u8x64_skylake_(min_vec.zmm);
    nk_u8_t max_value_comparable = nk_reduce_max_u8x64_skylake_(max_vec.zmm);

    // All-NaN early return: both sentinels unchanged means no valid data was found
    if (min_value_comparable == 0xFF && max_value_comparable == 0x00) {
        *min_value_ptr = NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = NK_E5M2_MIN, *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    if (min_value_comparable == 0xFF) { *min_value_ptr = NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX; }
    else {
        unsigned int min_lane;
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(min_vec.zmm, _mm512_set1_epi8((char)min_value_comparable));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            min_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        min_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
        nk_b512_vec_t loop_cycle_vec;
        loop_cycle_vec.zmm = min_loop_cycle_u8x64;
        *min_index_ptr = (nk_size_t)loop_cycle_vec.u8s[min_lane] * 64 + min_lane;
        nk_b512_vec_t min_raw_vec;
        min_raw_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(min_vec.zmm);
        *min_value_ptr = min_raw_vec.e5m2s[min_lane];
    }
    if (max_value_comparable == 0x00) { *max_value_ptr = NK_E5M2_MIN, *max_index_ptr = NK_SIZE_MAX; }
    else {
        unsigned int max_lane;
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(max_vec.zmm, _mm512_set1_epi8((char)max_value_comparable));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            max_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        max_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
        nk_b512_vec_t loop_cycle_vec;
        loop_cycle_vec.zmm = max_loop_cycle_u8x64;
        *max_index_ptr = (nk_size_t)loop_cycle_vec.u8s[max_lane] * 64 + max_lane;
        nk_b512_vec_t max_raw_vec;
        max_raw_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(max_vec.zmm);
        *max_value_ptr = max_raw_vec.e5m2s[max_lane];
    }
}

NK_PUBLIC void nk_reduce_minmax_e5m2_skylake(                           //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    if (count == 0)
        *min_value_ptr = NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E5M2_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_e5m2_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e5m2_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e5m2_skylake(data_ptr + left_count, count - left_count, stride_bytes, &right_min,
                                      &right_min_index, &right_max, &right_max_index);
        if (left_min_index == NK_SIZE_MAX)
            *min_value_ptr = right_min,
            *min_index_ptr = right_min_index == NK_SIZE_MAX ? NK_SIZE_MAX : left_count + right_min_index;
        else if (right_min_index == NK_SIZE_MAX || nk_e5m2_order_serial(left_min, right_min) <= 0)
            *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        else *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        if (left_max_index == NK_SIZE_MAX)
            *max_value_ptr = right_max,
            *max_index_ptr = right_max_index == NK_SIZE_MAX ? NK_SIZE_MAX : left_count + right_max_index;
        else if (right_max_index == NK_SIZE_MAX || nk_e5m2_order_serial(left_max, right_max) >= 0)
            *max_value_ptr = left_max, *max_index_ptr = left_max_index;
        else *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e5m2_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                  max_index_ptr);
    else
        nk_reduce_minmax_e5m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_skylake_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,             //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_b512_vec_t min_vec, max_vec;
    min_vec.zmm = _mm512_set1_epi8((char)0xFF);
    max_vec.zmm = _mm512_setzero_si512();
    __m512i min_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data_ptr + idx);
        __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_changed_mask = _mm512_cmp_epu8_mask(data_cmp_u8x64, min_vec.zmm, _MM_CMPINT_LT);
        min_vec.zmm = _mm512_min_epu8(min_vec.zmm, data_cmp_u8x64);
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        __mmask64 max_changed_mask = _mm512_cmp_epu8_mask(data_cmp_u8x64, max_vec.zmm, _MM_CMPINT_NLE);
        max_vec.zmm = _mm512_max_epu8(max_vec.zmm, data_cmp_u8x64);
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
        current_loop_cycle_u8x64 = _mm512_add_epi8(current_loop_cycle_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8(0x3F), tail_load, data_ptr + idx);
        __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, data_cmp_u8x64, min_vec.zmm, _MM_CMPINT_LT);
        min_vec.zmm = _mm512_mask_min_epu8(min_vec.zmm, tail_load, min_vec.zmm, data_cmp_u8x64);
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        __mmask64 max_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, data_cmp_u8x64, max_vec.zmm, _MM_CMPINT_NLE);
        max_vec.zmm = _mm512_mask_max_epu8(max_vec.zmm, tail_load, max_vec.zmm, data_cmp_u8x64);
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
    }

    nk_u8_t min_value_comparable = nk_reduce_min_u8x64_skylake_(min_vec.zmm);
    nk_u8_t max_value_comparable = nk_reduce_max_u8x64_skylake_(max_vec.zmm);
    unsigned int min_lane, max_lane;
    {
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(min_vec.zmm, _mm512_set1_epi8((char)min_value_comparable));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            min_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        min_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
    }
    {
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(max_vec.zmm, _mm512_set1_epi8((char)max_value_comparable));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            max_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        max_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u8x64;
    *min_index_ptr = (nk_size_t)loop_cycle_vec.u8s[min_lane] * 64 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u8x64;
    *max_index_ptr = (nk_size_t)loop_cycle_vec.u8s[max_lane] * 64 + max_lane;
    min_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(min_vec.zmm);
    max_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(max_vec.zmm);
    *min_value_ptr = min_vec.e2m3s[min_lane];
    *max_value_ptr = max_vec.e2m3s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e2m3_skylake(                           //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    if (count == 0)
        *min_value_ptr = NK_E2M3_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E2M3_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_e2m3_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e2m3_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e2m3_skylake(data_ptr + left_count, count - left_count, stride_bytes, &right_min,
                                      &right_min_index, &right_max, &right_max_index);
        if (nk_e2m3_order_serial(right_min, left_min) < 0)
            *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (nk_e2m3_order_serial(right_max, left_max) > 0)
            *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e2m3_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                  max_index_ptr);
    else
        nk_reduce_minmax_e2m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_skylake_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,             //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_b512_vec_t min_vec, max_vec;
    min_vec.zmm = _mm512_set1_epi8((char)0xFF);
    max_vec.zmm = _mm512_setzero_si512();
    __m512i min_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u8x64 = _mm512_setzero_si512();
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data_ptr + idx);
        __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_changed_mask = _mm512_cmp_epu8_mask(data_cmp_u8x64, min_vec.zmm, _MM_CMPINT_LT);
        min_vec.zmm = _mm512_min_epu8(min_vec.zmm, data_cmp_u8x64);
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        __mmask64 max_changed_mask = _mm512_cmp_epu8_mask(data_cmp_u8x64, max_vec.zmm, _MM_CMPINT_NLE);
        max_vec.zmm = _mm512_max_epu8(max_vec.zmm, data_cmp_u8x64);
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
        current_loop_cycle_u8x64 = _mm512_add_epi8(current_loop_cycle_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8(0x3F), tail_load, data_ptr + idx);
        __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, data_cmp_u8x64, min_vec.zmm, _MM_CMPINT_LT);
        min_vec.zmm = _mm512_mask_min_epu8(min_vec.zmm, tail_load, min_vec.zmm, data_cmp_u8x64);
        min_loop_cycle_u8x64 = _mm512_mask_mov_epi8(min_loop_cycle_u8x64, min_changed_mask, current_loop_cycle_u8x64);
        __mmask64 max_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, data_cmp_u8x64, max_vec.zmm, _MM_CMPINT_NLE);
        max_vec.zmm = _mm512_mask_max_epu8(max_vec.zmm, tail_load, max_vec.zmm, data_cmp_u8x64);
        max_loop_cycle_u8x64 = _mm512_mask_mov_epi8(max_loop_cycle_u8x64, max_changed_mask, current_loop_cycle_u8x64);
    }

    nk_u8_t min_value_comparable = nk_reduce_min_u8x64_skylake_(min_vec.zmm);
    nk_u8_t max_value_comparable = nk_reduce_max_u8x64_skylake_(max_vec.zmm);
    unsigned int min_lane, max_lane;
    {
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(min_vec.zmm, _mm512_set1_epi8((char)min_value_comparable));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            min_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        min_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
    }
    {
        __mmask64 value_match_mask = _mm512_cmpeq_epi8_mask(max_vec.zmm, _mm512_set1_epi8((char)max_value_comparable));
        __m512i masked_cycle_u8x64 = _mm512_mask_blend_epi8(value_match_mask, _mm512_set1_epi8((char)NK_U8_MAX),
                                                            max_loop_cycle_u8x64);
        nk_u8_t earliest_loop_cycle = nk_reduce_min_u8x64_skylake_(masked_cycle_u8x64);
        __mmask64 cycle_match_mask = _mm512_cmpeq_epi8_mask(masked_cycle_u8x64,
                                                            _mm512_set1_epi8((char)earliest_loop_cycle));
        max_lane = (unsigned int)_tzcnt_u64(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u8x64;
    *min_index_ptr = (nk_size_t)loop_cycle_vec.u8s[min_lane] * 64 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u8x64;
    *max_index_ptr = (nk_size_t)loop_cycle_vec.u8s[max_lane] * 64 + max_lane;
    min_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(min_vec.zmm);
    max_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(max_vec.zmm);
    *min_value_ptr = min_vec.e3m2s[min_lane];
    *max_value_ptr = max_vec.e3m2s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e3m2_skylake(                           //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    if (count == 0)
        *min_value_ptr = NK_E3M2_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E3M2_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_e3m2_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e3m2_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e3m2_skylake(data_ptr + left_count, count - left_count, stride_bytes, &right_min,
                                      &right_min_index, &right_max, &right_max_index);
        if (nk_e3m2_order_serial(right_min, left_min) < 0)
            *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (nk_e3m2_order_serial(right_max, left_max) > 0)
            *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e3m2_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                  max_index_ptr);
    else
        nk_reduce_minmax_e3m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i4_skylake_contiguous_( //
    nk_i4x2_t const *data_ptr, nk_size_t count,            //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum: XOR-bias nibbles to unsigned, vpsadbw, unbias at end.
    // Sumsq: squares are sign-independent; LUT maps nibble→square (max 225 fits u8), vpsadbw to u64.
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    __m512i eight_i8x64 = _mm512_set1_epi8(8);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    // Squares LUT: sq_lut[n] = n² for n in [0,15], all fit in u8 (max 225)
    __m512i sq_lut_u8x64 = _mm512_set_epi8(                                                       //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0);
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t count_bytes = nk_size_divide_round_up_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data_ptr;
    while (count_bytes > 0) {
        __m512i raw_i8x64;
        if (count_bytes < 64) {
            __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count_bytes);
            raw_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, ptr);
            count_bytes = 0;
        }
        else {
            raw_i8x64 = _mm512_loadu_si512(ptr);
            ptr += 64, count_bytes -= 64;
        }
        // Extract nibbles as unsigned [0,15]
        __m512i low_u4x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_u4x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        // Sum: XOR-bias nibbles to unsigned [0,15], add lo+hi per byte, vpsadbw
        __m512i low_biased_u4x64 = _mm512_xor_si512(low_u4x64, eight_i8x64);
        __m512i high_biased_u4x64 = _mm512_xor_si512(high_u4x64, eight_i8x64);
        __m512i pair_sum = _mm512_add_epi8(low_biased_u4x64, high_biased_u4x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(pair_sum, zero_i8x64));
        // Sumsq: squares are sign-independent, use LUT on unsigned nibbles
        __m512i low_sq_u8x64 = _mm512_shuffle_epi8(sq_lut_u8x64, low_u4x64);
        __m512i high_sq_u8x64 = _mm512_shuffle_epi8(sq_lut_u8x64, high_u4x64);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_sad_epu8(low_sq_u8x64, zero_i8x64));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_sad_epu8(high_sq_u8x64, zero_i8x64));
    }
    // The XOR-8 bias adds 8 per nibble to the SAD total. Subtract 8 × total nibbles processed
    // (including zero-masked register bytes, where 0 XOR 8 = 8, signed = 0).
    nk_size_t nibbles_processed = nk_size_round_up_to_multiple_(nk_size_divide_round_up_(count, 2), 64) * 2;
    nk_i64_t sum = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8) - (nk_i64_t)8 * (nk_i64_t)nibbles_processed;
    // sumsq uses sq_lut[0]=0 for zero-padded nibbles, so no register-padding correction needed.
    nk_u64_t sumsq = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i4_skylake(                            //
    nk_i4x2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    count = nk_size_round_up_to_multiple_(count, 2);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (stride_bytes == 1) nk_reduce_moments_i4_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i4_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_u4_skylake_contiguous_( //
    nk_u4x2_t const *data_ptr, nk_size_t count,            //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum: VPSADBW on extracted nibbles. Sumsq: LUT maps nibble→square (max 225 fits u8), vpsadbw to u64.
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    // Squares LUT: sq_lut[n] = n² for n in [0,15], all fit in u8 (max 225)
    __m512i sq_lut_u8x64 = _mm512_set_epi8(                                                       //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0);
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t count_bytes = count / 2;
    unsigned char const *ptr = (unsigned char const *)data_ptr;
    while (count_bytes > 0) {
        __m512i raw_i8x64;
        if (count_bytes < 64) {
            __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count_bytes);
            raw_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, ptr);
            count_bytes = 0;
        }
        else {
            raw_i8x64 = _mm512_loadu_si512(ptr);
            ptr += 64, count_bytes -= 64;
        }
        __m512i low_u4x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_u4x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        __m512i pair_sum = _mm512_add_epi8(low_u4x64, high_u4x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(pair_sum, zero_i8x64));
        // Sumsq: LUT maps nibble→square, vpsadbw accumulates into u64
        __m512i low_sq_u8x64 = _mm512_shuffle_epi8(sq_lut_u8x64, low_u4x64);
        __m512i high_sq_u8x64 = _mm512_shuffle_epi8(sq_lut_u8x64, high_u4x64);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_sad_epu8(low_sq_u8x64, zero_i8x64));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_sad_epu8(high_sq_u8x64, zero_i8x64));
    }
    nk_u64_t sum = _mm512_reduce_add_epi64(sum_u64x8);
    nk_u64_t sumsq = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u4_skylake(                            //
    nk_u4x2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    count = nk_size_round_up_to_multiple_(count, 2);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (stride_bytes == 1) nk_reduce_moments_u4_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u4_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_u1_skylake_contiguous_( //
    nk_u1x8_t const *data_ptr, nk_size_t count,            //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum = popcount via 4-bit LUT (same as nk_reduce_add_u1_skylake). Sumsq = sum for bits.
    __m512i lut_i8x64 = _mm512_set_epi8(                //
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, //
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, //
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, //
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t count_bytes = count / 8;
    unsigned char const *ptr = (unsigned char const *)data_ptr;
    while (count_bytes > 0) {
        __m512i raw_i8x64;
        if (count_bytes < 64) {
            __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count_bytes);
            raw_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, ptr);
            count_bytes = 0;
        }
        else {
            raw_i8x64 = _mm512_loadu_si512(ptr);
            ptr += 64, count_bytes -= 64;
        }
        __m512i low_nibble_u8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_nibble_u8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        __m512i popcnt_u8x64 = _mm512_add_epi8(_mm512_shuffle_epi8(lut_i8x64, low_nibble_u8x64),
                                               _mm512_shuffle_epi8(lut_i8x64, high_nibble_u8x64));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(popcnt_u8x64, zero_i8x64));
    }
    nk_u64_t sum = _mm512_reduce_add_epi64(sum_u64x8);
    *sum_ptr = sum;
    *sumsq_ptr = sum;
}

NK_PUBLIC void nk_reduce_moments_u1_skylake(                            //
    nk_u1x8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    count = nk_size_round_up_to_multiple_(count, 8);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (stride_bytes == 1) nk_reduce_moments_u1_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u1_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_bf16_skylake_contiguous_( //
    nk_bf16_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512 low_f32x16 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const *)(data_ptr + idx))), 16));
        __m512 high_f32x16 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const *)(data_ptr + idx + 16))), 16));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, low_f32x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, high_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(low_f32x16, low_f32x16, sumsq_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(high_f32x16, high_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 low_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(remaining > 16 ? 16 : remaining));
        __m512 low_f32x16 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(low_mask, data_ptr + idx)), 16));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, low_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(low_f32x16, low_f32x16, sumsq_f32x16);
        if (remaining > 16) {
            __mmask16 high_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(remaining - 16));
            __m512 high_f32x16 = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(high_mask, data_ptr + idx + 16)), 16));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, high_f32x16);
            sumsq_f32x16 = _mm512_fmadd_ps(high_f32x16, high_f32x16, sumsq_f32x16);
        }
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_bf16_skylake(                          //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_bf16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_bf16_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_bf16_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum;
        *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_bf16_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_bf16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL __m512i nk_bf16x32_to_comparable_i16x32_skylake_(__m512i raw_u16x32) {
    __m512i sign = _mm512_srai_epi16(raw_u16x32, 15);
    __m512i flip = _mm512_srli_epi16(sign, 1);
    return _mm512_xor_si512(raw_u16x32, flip);
}

NK_INTERNAL void nk_reduce_minmax_bf16_skylake_contiguous_( //
    nk_bf16_t const *data_ptr, nk_size_t count,             //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i abs_mask_u16x32 = _mm512_set1_epi16(0x7FFF);
    __m512i nan_threshold_u16x32 = _mm512_set1_epi16((short)0x7F80);
    __m512i min_cmp_i16x32 = _mm512_set1_epi16((short)0x7FFF);
    __m512i max_cmp_i16x32 = _mm512_set1_epi16((short)0x8000);
    __m512i min_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i one_u16x32 = _mm512_set1_epi16(1);

    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i raw_u16x32 = _mm512_loadu_si512(data_ptr + idx);
        __m512i data_cmp_i16x32 = nk_bf16x32_to_comparable_i16x32_skylake_(raw_u16x32);
        __m512i abs_u16x32 = _mm512_and_si512(raw_u16x32, abs_mask_u16x32);
        __mmask32 not_nan_m32 = _mm512_cmp_epu16_mask(abs_u16x32, nan_threshold_u16x32, _MM_CMPINT_LE);
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(not_nan_m32, data_cmp_i16x32, min_cmp_i16x32,
                                                                _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(not_nan_m32, data_cmp_i16x32, max_cmp_i16x32,
                                                                _MM_CMPINT_NLE);
        min_cmp_i16x32 = _mm512_mask_mov_epi16(min_cmp_i16x32, min_changed_mask, data_cmp_i16x32);
        max_cmp_i16x32 = _mm512_mask_mov_epi16(max_cmp_i16x32, max_changed_mask, data_cmp_i16x32);
        min_loop_cycle_u16x32 = _mm512_mask_mov_epi16(min_loop_cycle_u16x32, min_changed_mask,
                                                      current_loop_cycle_u16x32);
        max_loop_cycle_u16x32 = _mm512_mask_mov_epi16(max_loop_cycle_u16x32, max_changed_mask,
                                                      current_loop_cycle_u16x32);
        current_loop_cycle_u16x32 = _mm512_add_epi16(current_loop_cycle_u16x32, one_u16x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_load_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i raw_u16x32 = _mm512_maskz_loadu_epi16(tail_load_m32, data_ptr + idx);
        __m512i data_cmp_i16x32 = nk_bf16x32_to_comparable_i16x32_skylake_(raw_u16x32);
        __m512i abs_u16x32 = _mm512_and_si512(raw_u16x32, abs_mask_u16x32);
        __mmask32 not_nan_m32 = _mm512_cmp_epu16_mask(abs_u16x32, nan_threshold_u16x32, _MM_CMPINT_LE);
        __mmask32 valid_m32 = tail_load_m32 & not_nan_m32;
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(valid_m32, data_cmp_i16x32, min_cmp_i16x32,
                                                                _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(valid_m32, data_cmp_i16x32, max_cmp_i16x32,
                                                                _MM_CMPINT_NLE);
        min_cmp_i16x32 = _mm512_mask_mov_epi16(min_cmp_i16x32, min_changed_mask, data_cmp_i16x32);
        max_cmp_i16x32 = _mm512_mask_mov_epi16(max_cmp_i16x32, max_changed_mask, data_cmp_i16x32);
        min_loop_cycle_u16x32 = _mm512_mask_mov_epi16(min_loop_cycle_u16x32, min_changed_mask,
                                                      current_loop_cycle_u16x32);
        max_loop_cycle_u16x32 = _mm512_mask_mov_epi16(max_loop_cycle_u16x32, max_changed_mask,
                                                      current_loop_cycle_u16x32);
    }

    nk_i16_t min_value_comparable = nk_reduce_min_i16x32_skylake_(min_cmp_i16x32);
    nk_i16_t max_value_comparable = nk_reduce_max_i16x32_skylake_(max_cmp_i16x32);
    if (min_value_comparable == 0x7FFF && max_value_comparable == (nk_i16_t)0x8000) {
        *min_value_ptr = NK_BF16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_BF16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    unsigned int min_lane, max_lane;
    {
        __mmask32 value_match_mask = _mm512_cmpeq_epi16_mask(min_cmp_i16x32, _mm512_set1_epi16(min_value_comparable));
        __m512i masked_cycle_u16x32 = _mm512_mask_blend_epi16(value_match_mask, _mm512_set1_epi16((short)NK_U16_MAX),
                                                              min_loop_cycle_u16x32);
        nk_u16_t earliest_loop_cycle = nk_reduce_min_u16x32_skylake_(masked_cycle_u16x32);
        __mmask32 cycle_match_mask = _mm512_cmpeq_epi16_mask(masked_cycle_u16x32,
                                                             _mm512_set1_epi16((short)earliest_loop_cycle));
        min_lane = _tzcnt_u32(cycle_match_mask);
    }
    {
        __mmask32 value_match_mask = _mm512_cmpeq_epi16_mask(max_cmp_i16x32, _mm512_set1_epi16(max_value_comparable));
        __m512i masked_cycle_u16x32 = _mm512_mask_blend_epi16(value_match_mask, _mm512_set1_epi16((short)NK_U16_MAX),
                                                              max_loop_cycle_u16x32);
        nk_u16_t earliest_loop_cycle = nk_reduce_min_u16x32_skylake_(masked_cycle_u16x32);
        __mmask32 cycle_match_mask = _mm512_cmpeq_epi16_mask(masked_cycle_u16x32,
                                                             _mm512_set1_epi16((short)earliest_loop_cycle));
        max_lane = _tzcnt_u32(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u16x32;
    *min_index_ptr = (nk_size_t)loop_cycle_vec.u16s[min_lane] * 32 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u16x32;
    *max_index_ptr = (nk_size_t)loop_cycle_vec.u16s[max_lane] * 32 + max_lane;
    nk_i16_t min_sign = min_value_comparable >> 15;
    *min_value_ptr = (nk_bf16_t)((nk_u16_t)min_value_comparable ^ ((nk_u16_t)min_sign >> 1));
    nk_i16_t max_sign = max_value_comparable >> 15;
    *max_value_ptr = (nk_bf16_t)((nk_u16_t)max_value_comparable ^ ((nk_u16_t)max_sign >> 1));
}

NK_PUBLIC void nk_reduce_minmax_bf16_skylake(                           //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_BF16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_BF16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_bf16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_bf16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_bf16_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_bf16_skylake(data_ptr + left_count, count - left_count, stride_bytes, &right_min,
                                      &right_min_index, &right_max, &right_max_index);
        if (nk_bf16_order_serial(right_min, left_min) < 0)
            *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (nk_bf16_order_serial(right_max, left_max) > 0)
            *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_bf16_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                  max_index_ptr);
    else
        nk_reduce_minmax_bf16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_f16_skylake_contiguous_( //
    nk_f16_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512 low_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(data_ptr + idx)));
        __m512 high_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(data_ptr + idx + 16)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, low_f32x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, high_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(low_f32x16, low_f32x16, sumsq_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(high_f32x16, high_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 low_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(remaining > 16 ? 16 : remaining));
        __m512 low_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(low_mask_m16, data_ptr + idx));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, low_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(low_f32x16, low_f32x16, sumsq_f32x16);
        if (remaining > 16) {
            __mmask16 high_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(remaining - 16));
            __m512 high_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(high_mask_m16, data_ptr + idx + 16));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, high_f32x16);
            sumsq_f32x16 = _mm512_fmadd_ps(high_f32x16, high_f32x16, sumsq_f32x16);
        }
    }
    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_f16_skylake(                          //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_f16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f16_skylake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f16_skylake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum;
        *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f16_skylake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_f16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL __m512i nk_f16x32_to_comparable_i16x32_skylake_(__m512i raw_u16x32) {
    __m512i sign = _mm512_srai_epi16(raw_u16x32, 15);
    __m512i flip = _mm512_srli_epi16(sign, 1);
    return _mm512_xor_si512(raw_u16x32, flip);
}

NK_INTERNAL void nk_reduce_minmax_f16_skylake_contiguous_( //
    nk_f16_t const *data_ptr, nk_size_t count,             //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    __m512i abs_mask_u16x32 = _mm512_set1_epi16(0x7FFF);
    __m512i nan_threshold_u16x32 = _mm512_set1_epi16((short)0x7C00);
    __m512i min_cmp_i16x32 = _mm512_set1_epi16((short)0x7FFF);
    __m512i max_cmp_i16x32 = _mm512_set1_epi16((short)0x8000);
    __m512i min_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i max_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i current_loop_cycle_u16x32 = _mm512_setzero_si512();
    __m512i one_u16x32 = _mm512_set1_epi16(1);

    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i raw_u16x32 = _mm512_loadu_si512(data_ptr + idx);
        __m512i data_cmp_i16x32 = nk_f16x32_to_comparable_i16x32_skylake_(raw_u16x32);
        __m512i abs_u16x32 = _mm512_and_si512(raw_u16x32, abs_mask_u16x32);
        __mmask32 not_nan_m32 = _mm512_cmp_epu16_mask(abs_u16x32, nan_threshold_u16x32, _MM_CMPINT_LE);
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(not_nan_m32, data_cmp_i16x32, min_cmp_i16x32,
                                                                _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(not_nan_m32, data_cmp_i16x32, max_cmp_i16x32,
                                                                _MM_CMPINT_NLE);
        min_cmp_i16x32 = _mm512_mask_mov_epi16(min_cmp_i16x32, min_changed_mask, data_cmp_i16x32);
        max_cmp_i16x32 = _mm512_mask_mov_epi16(max_cmp_i16x32, max_changed_mask, data_cmp_i16x32);
        min_loop_cycle_u16x32 = _mm512_mask_mov_epi16(min_loop_cycle_u16x32, min_changed_mask,
                                                      current_loop_cycle_u16x32);
        max_loop_cycle_u16x32 = _mm512_mask_mov_epi16(max_loop_cycle_u16x32, max_changed_mask,
                                                      current_loop_cycle_u16x32);
        current_loop_cycle_u16x32 = _mm512_add_epi16(current_loop_cycle_u16x32, one_u16x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_load_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i raw_u16x32 = _mm512_maskz_loadu_epi16(tail_load_m32, data_ptr + idx);
        __m512i data_cmp_i16x32 = nk_f16x32_to_comparable_i16x32_skylake_(raw_u16x32);
        __m512i abs_u16x32 = _mm512_and_si512(raw_u16x32, abs_mask_u16x32);
        __mmask32 not_nan_m32 = _mm512_cmp_epu16_mask(abs_u16x32, nan_threshold_u16x32, _MM_CMPINT_LE);
        __mmask32 valid_m32 = tail_load_m32 & not_nan_m32;
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(valid_m32, data_cmp_i16x32, min_cmp_i16x32,
                                                                _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(valid_m32, data_cmp_i16x32, max_cmp_i16x32,
                                                                _MM_CMPINT_NLE);
        min_cmp_i16x32 = _mm512_mask_mov_epi16(min_cmp_i16x32, min_changed_mask, data_cmp_i16x32);
        max_cmp_i16x32 = _mm512_mask_mov_epi16(max_cmp_i16x32, max_changed_mask, data_cmp_i16x32);
        min_loop_cycle_u16x32 = _mm512_mask_mov_epi16(min_loop_cycle_u16x32, min_changed_mask,
                                                      current_loop_cycle_u16x32);
        max_loop_cycle_u16x32 = _mm512_mask_mov_epi16(max_loop_cycle_u16x32, max_changed_mask,
                                                      current_loop_cycle_u16x32);
    }

    nk_i16_t min_value_comparable = nk_reduce_min_i16x32_skylake_(min_cmp_i16x32);
    nk_i16_t max_value_comparable = nk_reduce_max_i16x32_skylake_(max_cmp_i16x32);
    if (min_value_comparable == 0x7FFF && max_value_comparable == (nk_i16_t)0x8000) {
        *min_value_ptr = NK_F16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    unsigned int min_lane, max_lane;
    {
        __mmask32 value_match_mask = _mm512_cmpeq_epi16_mask(min_cmp_i16x32, _mm512_set1_epi16(min_value_comparable));
        __m512i masked_cycle_u16x32 = _mm512_mask_blend_epi16(value_match_mask, _mm512_set1_epi16((short)NK_U16_MAX),
                                                              min_loop_cycle_u16x32);
        nk_u16_t earliest_loop_cycle = nk_reduce_min_u16x32_skylake_(masked_cycle_u16x32);
        __mmask32 cycle_match_mask = _mm512_cmpeq_epi16_mask(masked_cycle_u16x32,
                                                             _mm512_set1_epi16((short)earliest_loop_cycle));
        min_lane = _tzcnt_u32(cycle_match_mask);
    }
    {
        __mmask32 value_match_mask = _mm512_cmpeq_epi16_mask(max_cmp_i16x32, _mm512_set1_epi16(max_value_comparable));
        __m512i masked_cycle_u16x32 = _mm512_mask_blend_epi16(value_match_mask, _mm512_set1_epi16((short)NK_U16_MAX),
                                                              max_loop_cycle_u16x32);
        nk_u16_t earliest_loop_cycle = nk_reduce_min_u16x32_skylake_(masked_cycle_u16x32);
        __mmask32 cycle_match_mask = _mm512_cmpeq_epi16_mask(masked_cycle_u16x32,
                                                             _mm512_set1_epi16((short)earliest_loop_cycle));
        max_lane = _tzcnt_u32(cycle_match_mask);
    }
    nk_b512_vec_t loop_cycle_vec;
    loop_cycle_vec.zmm = min_loop_cycle_u16x32;
    *min_index_ptr = (nk_size_t)loop_cycle_vec.u16s[min_lane] * 32 + min_lane;
    loop_cycle_vec.zmm = max_loop_cycle_u16x32;
    *max_index_ptr = (nk_size_t)loop_cycle_vec.u16s[max_lane] * 32 + max_lane;
    nk_i16_t min_sign = min_value_comparable >> 15;
    *min_value_ptr = (nk_f16_t)((nk_u16_t)min_value_comparable ^ ((nk_u16_t)min_sign >> 1));
    nk_i16_t max_sign = max_value_comparable >> 15;
    *max_value_ptr = (nk_f16_t)((nk_u16_t)max_value_comparable ^ ((nk_u16_t)max_sign >> 1));
}

NK_PUBLIC void nk_reduce_minmax_f16_skylake(                           //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_F16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f16_skylake(data_ptr, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_f16_skylake(data_ptr + left_count, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (nk_f16_order_serial(right_min, left_min) < 0)
            *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_index;
        if (nk_f16_order_serial(right_max, left_max) > 0)
            *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f16_skylake_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                 max_index_ptr);
    else
        nk_reduce_minmax_f16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_REDUCE_SKYLAKE_H
