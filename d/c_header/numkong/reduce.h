/**
 *  @brief SIMD-accelerated Vector Reductions.
 *  @file include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2024
 *
 *  Provides horizontal reduction operations over vectors with:
 *  - `nk_reduce_moments_*` — sum + sum-of-squares in one pass
 *  - `nk_reduce_minmax_*` — min + max with argmin/argmax in one pass
 *  - Dynamic dispatch for runtime ISA selection
 *
 *  For dtypes:
 *
 *  - f64: 64-bit IEEE floating point numbers
 *  - f32: 32-bit IEEE floating point numbers
 *  - f16: 16-bit IEEE floating point numbers
 *  - bf16: 16-bit brain floating point numbers
 *  - e4m3: 8-bit e4m3 floating point numbers
 *  - e5m2: 8-bit e5m2 floating point numbers
 *  - e2m3: 8-bit e2m3 floating point numbers (MX)
 *  - e3m2: 8-bit e3m2 floating point numbers (MX)
 *  - i8: 8-bit signed integers
 *  - u8: 8-bit unsigned integers
 *  - i16: 16-bit signed integers
 *  - u16: 16-bit unsigned integers
 *  - i32: 32-bit signed integers
 *  - u32: 32-bit unsigned integers
 *  - i64: 64-bit signed integers
 *  - u64: 64-bit unsigned integers
 *  - i4: 4-bit signed integers (packed pairs)
 *  - u4: 4-bit unsigned integers (packed pairs)
 *  - u1: 1-bit binary (packed octets)
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, NEON+F16, NEON+FHM, NEON+BF16, NEON+SDOT
 *  - x86: Haswell, Skylake, Ice Lake, Genoa, Sierra Forest
 *  - RISC-V: RVV
 *  - WASM: V128Relaxed
 *
 *  @section numerical_stability Numerical stability
 *
 *  All accumulations are performed with stable techniques and @b saturation in mind.
 *  Single-precision inputs are aggregated in double-precision. Double-precision
 *  inputs are handled with @b Neumaier-like compensated summation schemes. Mini-floats
 *  are propagated to more hardware-friendly types. And integer are handled with
 *  proper saturation logic, as opposed to simple pairwise saturation, meaning that
 *  if several extremely large values are followed by equal negative values, the
 *  sum will be zero.
 *
 *  @code{.c}
 *
 *  @endcode{.c}
 *
 *
 *  All MinMax scans are performed with respect to NaN values beyond simple total ordering.
 *  All positive and negative NaN values are masked out on the fly and can never be included
 *  in the output. For empty or NaN-only inputs, the returned argmin/argmax positions will
 *  be set to sentinel value @b `NK_SIZE_MAX`.
 *
 *  @section reduction_strategy Reduction Strategy
 *
 *  The key insight is that `_mm512_reduce_add_ps()` and similar intrinsics are
 *  actually serial operations - they don't parallelize the reduction across lanes.
 *  The correct approach is:
 *
 *  1. Accumulate vertically in SIMD registers throughout the entire loop
 *  2. Perform a single horizontal reduction at the very end, reconstructing the lane positions
 *
 *  @code{.c}
 *  __m512 sum_f32x16 = _mm512_setzero_ps();
 *  for (...) {
 *      __m512 data_f32x16 = _mm512_loadu_ps(ptr);
 *      sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
 *  }
 *  // Single horizontal reduce at the END only
 *  nk_f32_t result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
 *  @endcode
 *
 *  @section stride_handling Stride Handling Strategies
 *
 *  - stride == sizeof(scalar): Contiguous SIMD loads with masked tail
 *  - Large stride with gather support: Use gather instructions (32/64-bit types)
 *  - Otherwise: Serial fallback
 *
 *  @section argminmax Argmin/Argmax Strategy
 *
 *  Single-pass algorithm tracking both value and index in SIMD registers:
 *  @code{.c}
 *  __m512 min_f32x16 = _mm512_set1_ps(FLT_MAX);
 *  __m512i min_idx_i32x16 = _mm512_setzero_si512();
 *  __m512i current_idx_i32x16 = _mm512_setr_epi32(0,1,2,3,...,15);
 *  __m512i step_i32x16 = _mm512_set1_epi32(16);
 *  for (...) {
 *      __m512 data_f32x16 = _mm512_loadu_ps(ptr);
 *      __mmask16 lt_mask = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
 *      min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, data_f32x16);
 *      min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask, current_idx_i32x16);
 *      current_idx_i32x16 = _mm512_add_epi32(current_idx_i32x16, step_i32x16);
 *  }
 *  @endcode
 */
#ifndef NK_REDUCE_H
#define NK_REDUCE_H

#include "numkong/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Horizontal moments reduction (sum + sum-of-squares) over a strided array.
 *  @param[in] data Pointer to the input data.
 *  @param[in] count Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes, equal to `sizeof(*data)` for contiguous arrays.
 *  @param[out] sum_ptr Output sum.
 *  @param[out] sumsq_ptr Output sum of squares.
 */
NK_DYNAMIC void nk_reduce_moments_f64(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *sum_ptr,
                                      nk_f64_t *sumsq_ptr);

/**
 *  @brief  Horizontal min+max reduction with argmin/argmax over a strided array.
 *  @param[in] data Pointer to the input data.
 *  @param[in] count Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes, equal to `sizeof(*data)` for contiguous arrays.
 *  @param[out] min_value_ptr Output minimum value.
 *  @param[out] min_index_ptr Output index of the minimum value.
 *  @param[out] max_value_ptr Output maximum value.
 *  @param[out] max_index_ptr Output index of the maximum value.
 */
NK_DYNAMIC void nk_reduce_minmax_f64(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                     nk_f64_t *min_value_ptr, nk_size_t *min_index_ptr, nk_f64_t *max_value_ptr,
                                     nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_f32(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *sum_ptr,
                                      nk_f64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_f32(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                     nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr, nk_f32_t *max_value_ptr,
                                     nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_i8(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *sum_ptr,
                                     nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_i8(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr, nk_i8_t *max_value_ptr,
                                    nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_u8(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *sum_ptr,
                                     nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_u8(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr, nk_u8_t *max_value_ptr,
                                    nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_i16(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *sum_ptr,
                                      nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_i16(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                     nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr, nk_i16_t *max_value_ptr,
                                     nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_u16(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *sum_ptr,
                                      nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_u16(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                     nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr, nk_u16_t *max_value_ptr,
                                     nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_i32(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *sum_ptr,
                                      nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_i32(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                     nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr, nk_i32_t *max_value_ptr,
                                     nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_u32(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *sum_ptr,
                                      nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_u32(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                     nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr, nk_u32_t *max_value_ptr,
                                     nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_i64(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *sum_ptr,
                                      nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_i64(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                     nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr, nk_i64_t *max_value_ptr,
                                     nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_u64(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *sum_ptr,
                                      nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_u64(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                     nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr, nk_u64_t *max_value_ptr,
                                     nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_f16(nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f32_t *sum_ptr,
                                      nk_f32_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_f16(nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                     nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr, nk_f16_t *max_value_ptr,
                                     nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_bf16(nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                       nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_bf16(nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr, nk_bf16_t *max_value_ptr,
                                      nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_e4m3(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                       nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_e4m3(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr, nk_e4m3_t *max_value_ptr,
                                      nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_e5m2(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                       nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_e5m2(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr, nk_e5m2_t *max_value_ptr,
                                      nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_e2m3(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                       nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_e2m3(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr, nk_e2m3_t *max_value_ptr,
                                      nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_e3m2(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                       nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_e3m2(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr, nk_e3m2_t *max_value_ptr,
                                      nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_i4(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *sum_ptr,
                                     nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_i4(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr, nk_i8_t *max_value_ptr,
                                    nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_u4(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *sum_ptr,
                                     nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_u4(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr, nk_u8_t *max_value_ptr,
                                    nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_DYNAMIC void nk_reduce_moments_u1(nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *sum_ptr,
                                     nk_u64_t *sumsq_ptr);
/** @copydoc nk_reduce_minmax_f64 */
NK_DYNAMIC void nk_reduce_minmax_u1(nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr, nk_u8_t *max_value_ptr,
                                    nk_size_t *max_index_ptr);

/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f32_serial(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f64_serial(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i8_serial(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_serial(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i16_serial(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u16_serial(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i32_serial(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u32_serial(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i64_serial(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u64_serial(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f16_serial(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_bf16_serial(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e4m3_serial(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e5m2_serial(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_serial(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e3m2_serial(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i4_serial(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u4_serial(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u1_serial(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);

/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f32_serial(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *, nk_f32_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f64_serial(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *, nk_f64_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i8_serial(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *, nk_i8_t *,
                                          nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u8_serial(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *, nk_u8_t *,
                                          nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i16_serial(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *, nk_i16_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u16_serial(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *, nk_u16_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i32_serial(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *, nk_i32_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u32_serial(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *, nk_u32_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i64_serial(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *, nk_i64_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u64_serial(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *, nk_u64_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f16_serial(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *, nk_f16_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_bf16_serial(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                            nk_bf16_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e4m3_serial(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                            nk_e4m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e5m2_serial(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                            nk_e5m2_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e2m3_serial(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                            nk_e2m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e3m2_serial(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                            nk_e3m2_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i4_serial(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *, nk_i8_t *,
                                          nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u4_serial(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *, nk_u8_t *,
                                          nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u1_serial(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *, nk_u8_t *,
                                          nk_size_t *);

#if NK_TARGET_NEON
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f32_neon(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f64_neon(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i8_neon(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_neon(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i16_neon(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u16_neon(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i32_neon(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u32_neon(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i64_neon(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u64_neon(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_neon(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e3m2_neon(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e4m3_neon(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e5m2_neon(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f32_neon(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *, nk_f32_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f64_neon(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *, nk_f64_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i8_neon(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *, nk_i8_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u8_neon(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *, nk_u8_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i16_neon(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *, nk_i16_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u16_neon(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *, nk_u16_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i32_neon(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *, nk_i32_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u32_neon(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *, nk_u32_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i64_neon(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *, nk_i64_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u64_neon(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *, nk_u64_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e2m3_neon(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                          nk_e2m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e3m2_neon(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                          nk_e3m2_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e4m3_neon(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                          nk_e4m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e5m2_neon(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                          nk_e5m2_t *, nk_size_t *);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f16_neonhalf(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_bf16_neonbfdot(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_bf16_neonbfdot(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                               nk_bf16_t *, nk_size_t *);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONSDOT
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i8_neonsdot(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_neonsdot(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_neonsdot(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONFHM
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e4m3_neonfhm(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e5m2_neonfhm(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e4m3_neonfhm(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                             nk_e4m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e5m2_neonfhm(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                             nk_e5m2_t *, nk_size_t *);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_HASWELL
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f32_haswell(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f64_haswell(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i8_haswell(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_haswell(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i16_haswell(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u16_haswell(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i32_haswell(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u32_haswell(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i64_haswell(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u64_haswell(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f16_haswell(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_bf16_haswell(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e4m3_haswell(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e5m2_haswell(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_haswell(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e3m2_haswell(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i4_haswell(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u4_haswell(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u1_haswell(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f32_haswell(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *, nk_f32_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f64_haswell(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *, nk_f64_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i8_haswell(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *, nk_i8_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u8_haswell(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *, nk_u8_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i16_haswell(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *, nk_i16_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u16_haswell(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *, nk_u16_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i32_haswell(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *, nk_i32_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u32_haswell(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *, nk_u32_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i64_haswell(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *, nk_i64_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u64_haswell(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *, nk_u64_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f16_haswell(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *, nk_f16_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_bf16_haswell(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                             nk_bf16_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e4m3_haswell(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                             nk_e4m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e5m2_haswell(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                             nk_e5m2_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e2m3_haswell(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                             nk_e2m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e3m2_haswell(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                             nk_e3m2_t *, nk_size_t *);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f32_skylake(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f64_skylake(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i8_skylake(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_skylake(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i16_skylake(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u16_skylake(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i32_skylake(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u32_skylake(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i64_skylake(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u64_skylake(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f16_skylake(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_bf16_skylake(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e4m3_skylake(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e5m2_skylake(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_skylake(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e3m2_skylake(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i4_skylake(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u4_skylake(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u1_skylake(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f32_skylake(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *, nk_f32_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f64_skylake(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *, nk_f64_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i8_skylake(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *, nk_i8_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u8_skylake(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *, nk_u8_t *,
                                           nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i16_skylake(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *, nk_i16_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u16_skylake(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *, nk_u16_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i32_skylake(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *, nk_i32_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u32_skylake(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *, nk_u32_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i64_skylake(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *, nk_i64_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u64_skylake(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *, nk_u64_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f16_skylake(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *, nk_f16_t *,
                                            nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_bf16_skylake(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                             nk_bf16_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e4m3_skylake(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                             nk_e4m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e5m2_skylake(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                             nk_e5m2_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e2m3_skylake(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                             nk_e2m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e3m2_skylake(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                             nk_e3m2_t *, nk_size_t *);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i8_icelake(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_icelake(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i16_icelake(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_icelake(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e3m2_icelake(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_GENOA
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_bf16_genoa(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e4m3_genoa(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e5m2_genoa(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
#endif // NK_TARGET_GENOA

#if NK_TARGET_ALDER
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_alder(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i16_alder(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u16_alder(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e3m2_alder(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_alder(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
#endif // NK_TARGET_ALDER
#if NK_TARGET_SIERRA
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i8_sierra(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_sierra(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_sierra(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
#endif // NK_TARGET_SIERRA

#if NK_TARGET_RVV
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f32_rvv(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f64_rvv(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i8_rvv(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_rvv(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i16_rvv(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u16_rvv(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i32_rvv(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u32_rvv(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i64_rvv(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u64_rvv(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f16_rvv(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_bf16_rvv(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e4m3_rvv(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e5m2_rvv(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_rvv(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e3m2_rvv(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f32_rvv(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *, nk_f32_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f64_rvv(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *, nk_f64_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i8_rvv(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *, nk_i8_t *,
                                       nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u8_rvv(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *, nk_u8_t *,
                                       nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i16_rvv(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *, nk_i16_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u16_rvv(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *, nk_u16_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i32_rvv(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *, nk_i32_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u32_rvv(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *, nk_u32_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i64_rvv(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *, nk_i64_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u64_rvv(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *, nk_u64_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f16_rvv(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *, nk_f16_t *,
                                        nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_bf16_rvv(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *, nk_bf16_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e4m3_rvv(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *, nk_e4m3_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e5m2_rvv(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *, nk_e5m2_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e2m3_rvv(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *, nk_e2m3_t *,
                                         nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e3m2_rvv(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *, nk_e3m2_t *,
                                         nk_size_t *);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f32_v128relaxed(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f64_v128relaxed(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i8_v128relaxed(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u8_v128relaxed(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i16_v128relaxed(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u16_v128relaxed(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i32_v128relaxed(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u32_v128relaxed(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_i64_v128relaxed(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_u64_v128relaxed(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_f16_v128relaxed(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_bf16_v128relaxed(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e4m3_v128relaxed(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e5m2_v128relaxed(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e2m3_v128relaxed(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_moments_f64 */
NK_PUBLIC void nk_reduce_moments_e3m2_v128relaxed(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f32_v128relaxed(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *,
                                                nk_f32_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f64_v128relaxed(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *,
                                                nk_f64_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i8_v128relaxed(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *, nk_i8_t *,
                                               nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u8_v128relaxed(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *, nk_u8_t *,
                                               nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i16_v128relaxed(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *,
                                                nk_i16_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u16_v128relaxed(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *,
                                                nk_u16_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i32_v128relaxed(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *,
                                                nk_i32_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u32_v128relaxed(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *,
                                                nk_u32_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_i64_v128relaxed(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *,
                                                nk_i64_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_u64_v128relaxed(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *,
                                                nk_u64_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_f16_v128relaxed(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *,
                                                nk_f16_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_bf16_v128relaxed(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                                 nk_bf16_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e4m3_v128relaxed(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                                 nk_e4m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e5m2_v128relaxed(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                                 nk_e5m2_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e2m3_v128relaxed(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                                 nk_e2m3_t *, nk_size_t *);
/** @copydoc nk_reduce_minmax_f64 */
NK_PUBLIC void nk_reduce_minmax_e3m2_v128relaxed(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                                 nk_e3m2_t *, nk_size_t *);
#endif // NK_TARGET_V128RELAXED

/**
 *  @brief  Returns the accumulator dtype for the `sum` output of reduce_moments.
 *
 *  Float types accumulate into wider floats; signed ints into i64; unsigned ints into u64.
 */
NK_INTERNAL nk_dtype_t nk_reduce_moments_sum_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_e2m3_k: return nk_f32_k;
    case nk_e3m2_k: return nk_f32_k;
    case nk_i8_k: return nk_i64_k;
    case nk_i16_k: return nk_i64_k;
    case nk_i32_k: return nk_i64_k;
    case nk_i64_k: return nk_i64_k;
    case nk_i4_k: return nk_i64_k;
    case nk_u8_k: return nk_u64_k;
    case nk_u16_k: return nk_u64_k;
    case nk_u32_k: return nk_u64_k;
    case nk_u64_k: return nk_u64_k;
    case nk_u4_k: return nk_u64_k;
    case nk_u1_k: return nk_u64_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Returns the accumulator dtype for the `sumsq` output of reduce_moments.
 *
 *  Same as sum except all integers (signed and unsigned) accumulate into u64.
 */
NK_INTERNAL nk_dtype_t nk_reduce_moments_sumsq_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_e2m3_k: return nk_f32_k;
    case nk_e3m2_k: return nk_f32_k;
    case nk_i8_k: return nk_u64_k;
    case nk_i16_k: return nk_u64_k;
    case nk_i32_k: return nk_u64_k;
    case nk_i64_k: return nk_u64_k;
    case nk_i4_k: return nk_u64_k;
    case nk_u8_k: return nk_u64_k;
    case nk_u16_k: return nk_u64_k;
    case nk_u32_k: return nk_u64_k;
    case nk_u64_k: return nk_u64_k;
    case nk_u4_k: return nk_u64_k;
    case nk_u1_k: return nk_u64_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Returns the value dtype for reduce_minmax outputs.
 *
 *  Standard types return themselves. Sub-byte types widen: i4->i8, u4->u8, u1->u8.
 */
NK_INTERNAL nk_dtype_t nk_reduce_minmax_value_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_i4_k: return nk_i8_k;
    case nk_u4_k: return nk_u8_k;
    case nk_u1_k: return nk_u8_k;
    default: return dtype;
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#include "numkong/reduce/serial.h"
#include "numkong/reduce/neon.h"
#include "numkong/reduce/neonhalf.h"
#include "numkong/reduce/neonbfdot.h"
#include "numkong/reduce/neonsdot.h"
#include "numkong/reduce/neonfhm.h"
#include "numkong/reduce/haswell.h"
#include "numkong/reduce/skylake.h"
#include "numkong/reduce/icelake.h"
#include "numkong/reduce/genoa.h"
#include "numkong/reduce/alder.h"
#include "numkong/reduce/sierra.h"
#include "numkong/reduce/rvv.h"
#include "numkong/reduce/v128relaxed.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_reduce_moments_f32(nk_f32_t const *d, nk_size_t n, nk_size_t s, nk_f64_t *sum, nk_f64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_f32_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_f32_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_f32_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_f32_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_f32_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_f32_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_f32(nk_f32_t const *d, nk_size_t n, nk_size_t s, nk_f32_t *mn, nk_size_t *mi,
                                    nk_f32_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_f32_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_f32_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_f32_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_f32_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_f32_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_f32_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_f64(nk_f64_t const *d, nk_size_t n, nk_size_t s, nk_f64_t *sum, nk_f64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_f64_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_f64_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_f64_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_f64_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_f64_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_f64_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_f64(nk_f64_t const *d, nk_size_t n, nk_size_t s, nk_f64_t *mn, nk_size_t *mi,
                                    nk_f64_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_f64_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_f64_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_f64_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_f64_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_f64_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_f64_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_i8(nk_i8_t const *d, nk_size_t n, nk_size_t s, nk_i64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_ICELAKE
    nk_reduce_moments_i8_icelake(d, n, s, sum, sumsq);
#elif NK_TARGET_SKYLAKE
    nk_reduce_moments_i8_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_SIERRA
    nk_reduce_moments_i8_sierra(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_i8_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEONSDOT
    nk_reduce_moments_i8_neonsdot(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_i8_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_i8_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_i8_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_i8_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_i8(nk_i8_t const *d, nk_size_t n, nk_size_t s, nk_i8_t *mn, nk_size_t *mi, nk_i8_t *mx,
                                   nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_i8_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_i8_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_i8_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_i8_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_i8_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_i8_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_u8(nk_u8_t const *d, nk_size_t n, nk_size_t s, nk_u64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_ICELAKE
    nk_reduce_moments_u8_icelake(d, n, s, sum, sumsq);
#elif NK_TARGET_SKYLAKE
    nk_reduce_moments_u8_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_SIERRA
    nk_reduce_moments_u8_sierra(d, n, s, sum, sumsq);
#elif NK_TARGET_ALDER
    nk_reduce_moments_u8_alder(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_u8_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEONSDOT
    nk_reduce_moments_u8_neonsdot(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_u8_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_u8_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_u8_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_u8_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_u8(nk_u8_t const *d, nk_size_t n, nk_size_t s, nk_u8_t *mn, nk_size_t *mi, nk_u8_t *mx,
                                   nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_u8_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_u8_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_u8_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_u8_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_u8_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_u8_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_i16(nk_i16_t const *d, nk_size_t n, nk_size_t s, nk_i64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_ICELAKE
    nk_reduce_moments_i16_icelake(d, n, s, sum, sumsq);
#elif NK_TARGET_SKYLAKE
    nk_reduce_moments_i16_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_ALDER
    nk_reduce_moments_i16_alder(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_i16_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_i16_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_i16_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_i16_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_i16_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_i16(nk_i16_t const *d, nk_size_t n, nk_size_t s, nk_i16_t *mn, nk_size_t *mi,
                                    nk_i16_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_i16_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_i16_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_i16_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_i16_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_i16_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_i16_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_u16(nk_u16_t const *d, nk_size_t n, nk_size_t s, nk_u64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_u16_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_ALDER
    nk_reduce_moments_u16_alder(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_u16_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_u16_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_u16_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_u16_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_u16_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_u16(nk_u16_t const *d, nk_size_t n, nk_size_t s, nk_u16_t *mn, nk_size_t *mi,
                                    nk_u16_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_u16_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_u16_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_u16_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_u16_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_u16_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_u16_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_i32(nk_i32_t const *d, nk_size_t n, nk_size_t s, nk_i64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_i32_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_i32_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_i32_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_i32_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_i32_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_i32_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_i32(nk_i32_t const *d, nk_size_t n, nk_size_t s, nk_i32_t *mn, nk_size_t *mi,
                                    nk_i32_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_i32_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_i32_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_i32_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_i32_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_i32_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_i32_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_u32(nk_u32_t const *d, nk_size_t n, nk_size_t s, nk_u64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_u32_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_u32_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_u32_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_u32_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_u32_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_u32_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_u32(nk_u32_t const *d, nk_size_t n, nk_size_t s, nk_u32_t *mn, nk_size_t *mi,
                                    nk_u32_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_u32_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_u32_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_u32_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_u32_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_u32_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_u32_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_i64(nk_i64_t const *d, nk_size_t n, nk_size_t s, nk_i64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_i64_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_i64_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_i64_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_i64_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_i64_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_i64_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_i64(nk_i64_t const *d, nk_size_t n, nk_size_t s, nk_i64_t *mn, nk_size_t *mi,
                                    nk_i64_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_i64_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_i64_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_i64_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_i64_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_i64_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_i64_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_u64(nk_u64_t const *d, nk_size_t n, nk_size_t s, nk_u64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_u64_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_u64_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_u64_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_u64_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_u64_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_u64_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_u64(nk_u64_t const *d, nk_size_t n, nk_size_t s, nk_u64_t *mn, nk_size_t *mi,
                                    nk_u64_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_u64_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_u64_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_u64_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_u64_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_u64_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_u64_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_f16(nk_f16_t const *d, nk_size_t n, nk_size_t s, nk_f32_t *sum, nk_f32_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_f16_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_f16_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEONHALF
    nk_reduce_moments_f16_neonhalf(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_f16_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_f16_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_f16_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_f16(nk_f16_t const *d, nk_size_t n, nk_size_t s, nk_f16_t *mn, nk_size_t *mi,
                                    nk_f16_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_f16_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_f16_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEONHALF
    nk_reduce_minmax_f16_serial(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_f16_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_f16_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_f16_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_bf16(nk_bf16_t const *d, nk_size_t n, nk_size_t s, nk_f32_t *sum, nk_f32_t *sumsq) {
#if NK_TARGET_GENOA
    nk_reduce_moments_bf16_genoa(d, n, s, sum, sumsq);
#elif NK_TARGET_SKYLAKE
    nk_reduce_moments_bf16_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_bf16_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEONBFDOT
    nk_reduce_moments_bf16_neonbfdot(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_bf16_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_bf16_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_bf16_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_bf16(nk_bf16_t const *d, nk_size_t n, nk_size_t s, nk_bf16_t *mn, nk_size_t *mi,
                                     nk_bf16_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_bf16_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_bf16_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEONBFDOT
    nk_reduce_minmax_bf16_neonbfdot(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_bf16_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_bf16_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_bf16_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_e4m3(nk_e4m3_t const *d, nk_size_t n, nk_size_t s, nk_f32_t *sum, nk_f32_t *sumsq) {
#if NK_TARGET_GENOA
    nk_reduce_moments_e4m3_genoa(d, n, s, sum, sumsq);
#elif NK_TARGET_SKYLAKE
    nk_reduce_moments_e4m3_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_e4m3_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEONFHM
    nk_reduce_moments_e4m3_neonfhm(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_e4m3_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_e4m3_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_e4m3_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_e4m3_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_e4m3(nk_e4m3_t const *d, nk_size_t n, nk_size_t s, nk_e4m3_t *mn, nk_size_t *mi,
                                     nk_e4m3_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_e4m3_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_e4m3_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEONFHM
    nk_reduce_minmax_e4m3_neonfhm(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_e4m3_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_e4m3_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_e4m3_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_e4m3_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_e5m2(nk_e5m2_t const *d, nk_size_t n, nk_size_t s, nk_f32_t *sum, nk_f32_t *sumsq) {
#if NK_TARGET_GENOA
    nk_reduce_moments_e5m2_genoa(d, n, s, sum, sumsq);
#elif NK_TARGET_SKYLAKE
    nk_reduce_moments_e5m2_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_e5m2_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEONFHM
    nk_reduce_moments_e5m2_neonfhm(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_e5m2_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_e5m2_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_e5m2_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_e5m2_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_e5m2(nk_e5m2_t const *d, nk_size_t n, nk_size_t s, nk_e5m2_t *mn, nk_size_t *mi,
                                     nk_e5m2_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_e5m2_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_e5m2_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEONFHM
    nk_reduce_minmax_e5m2_neonfhm(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_e5m2_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_e5m2_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_e5m2_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_e5m2_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_e2m3(nk_e2m3_t const *d, nk_size_t n, nk_size_t s, nk_f32_t *sum, nk_f32_t *sumsq) {
#if NK_TARGET_ICELAKE
    nk_reduce_moments_e2m3_icelake(d, n, s, sum, sumsq);
#elif NK_TARGET_SKYLAKE
    nk_reduce_moments_e2m3_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_SIERRA
    nk_reduce_moments_e2m3_sierra(d, n, s, sum, sumsq);
#elif NK_TARGET_ALDER
    nk_reduce_moments_e2m3_alder(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_e2m3_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEONSDOT
    nk_reduce_moments_e2m3_neonsdot(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_e2m3_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_e2m3_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_e2m3_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_e2m3_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_e2m3(nk_e2m3_t const *d, nk_size_t n, nk_size_t s, nk_e2m3_t *mn, nk_size_t *mi,
                                     nk_e2m3_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_e2m3_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_e2m3_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_e2m3_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_e2m3_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_e2m3_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_e2m3_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_e3m2(nk_e3m2_t const *d, nk_size_t n, nk_size_t s, nk_f32_t *sum, nk_f32_t *sumsq) {
#if NK_TARGET_ICELAKE
    nk_reduce_moments_e3m2_icelake(d, n, s, sum, sumsq);
#elif NK_TARGET_SKYLAKE
    nk_reduce_moments_e3m2_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_ALDER
    nk_reduce_moments_e3m2_alder(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_e3m2_haswell(d, n, s, sum, sumsq);
#elif NK_TARGET_NEON
    nk_reduce_moments_e3m2_neon(d, n, s, sum, sumsq);
#elif NK_TARGET_RVV
    nk_reduce_moments_e3m2_rvv(d, n, s, sum, sumsq);
#elif NK_TARGET_V128RELAXED
    nk_reduce_moments_e3m2_v128relaxed(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_e3m2_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_e3m2(nk_e3m2_t const *d, nk_size_t n, nk_size_t s, nk_e3m2_t *mn, nk_size_t *mi,
                                     nk_e3m2_t *mx, nk_size_t *xi) {
#if NK_TARGET_SKYLAKE
    nk_reduce_minmax_e3m2_skylake(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_HASWELL
    nk_reduce_minmax_e3m2_haswell(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_NEON
    nk_reduce_minmax_e3m2_neon(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_RVV
    nk_reduce_minmax_e3m2_rvv(d, n, s, mn, mi, mx, xi);
#elif NK_TARGET_V128RELAXED
    nk_reduce_minmax_e3m2_v128relaxed(d, n, s, mn, mi, mx, xi);
#else
    nk_reduce_minmax_e3m2_serial(d, n, s, mn, mi, mx, xi);
#endif
}

NK_PUBLIC void nk_reduce_moments_i4(nk_i4x2_t const *d, nk_size_t n, nk_size_t s, nk_i64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_i4_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_i4_haswell(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_i4_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_i4(nk_i4x2_t const *d, nk_size_t n, nk_size_t s, nk_i8_t *mn, nk_size_t *mi,
                                   nk_i8_t *mx, nk_size_t *xi) {
    nk_reduce_minmax_i4_serial(d, n, s, mn, mi, mx, xi);
}

NK_PUBLIC void nk_reduce_moments_u4(nk_u4x2_t const *d, nk_size_t n, nk_size_t s, nk_u64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_u4_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_u4_haswell(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_u4_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_u4(nk_u4x2_t const *d, nk_size_t n, nk_size_t s, nk_u8_t *mn, nk_size_t *mi,
                                   nk_u8_t *mx, nk_size_t *xi) {
    nk_reduce_minmax_u4_serial(d, n, s, mn, mi, mx, xi);
}

NK_PUBLIC void nk_reduce_moments_u1(nk_u1x8_t const *d, nk_size_t n, nk_size_t s, nk_u64_t *sum, nk_u64_t *sumsq) {
#if NK_TARGET_SKYLAKE
    nk_reduce_moments_u1_skylake(d, n, s, sum, sumsq);
#elif NK_TARGET_HASWELL
    nk_reduce_moments_u1_haswell(d, n, s, sum, sumsq);
#else
    nk_reduce_moments_u1_serial(d, n, s, sum, sumsq);
#endif
}

NK_PUBLIC void nk_reduce_minmax_u1(nk_u1x8_t const *d, nk_size_t n, nk_size_t s, nk_u8_t *mn, nk_size_t *mi,
                                   nk_u8_t *mx, nk_size_t *xi) {
    nk_reduce_minmax_u1_serial(d, n, s, mn, mi, mx, xi);
}

#endif // !NK_DYNAMIC_DISPATCH

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NK_REDUCE_H
