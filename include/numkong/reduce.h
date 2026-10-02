/**
 *  @file include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2024
 *  @brief SIMD-accelerated vector reductions.
 *
 *  Provides horizontal reduction operations over vectors with:
 *  - `nk_reduce_moments_*` — sum + sum-of-squares in one pass
 *  - `nk_reduce_minmax_*` — min + max with argmin/argmax in one pass
 *  - Runtime dispatch for runtime ISA selection
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
 *  - WASM: V128, V128Relaxed
 *
 *  @section reduce_numerical_stability Numerical stability
 *
 *  All accumulations are performed with stable techniques and @b saturation in mind.
 *  Single-precision inputs are aggregated in double-precision. Double-precision inputs are handled
 *  with @b Neumaier-like compensated summation schemes. Mini-floats are propagated to more
 *  hardware-friendly types. And integers are handled with proper saturation logic, as opposed to
 *  simple pairwise saturation, meaning that if several extremely large values are followed by equal
 *  negative values, the sum will be zero.
 *
 *  All MinMax scans are performed with respect to NaN values beyond simple total ordering. All
 *  positive and negative NaN values are masked out on the fly and can never be included in the
 *  output. For empty or NaN-only inputs, the returned argmin/argmax positions will be set to
 *  sentinel value @c NUMKONG_SIZE_MAX.
 *
 *  @section reduction_strategy Reduction Strategy
 *
 *  The key insight is that `_mm512_reduce_add_ps()` and similar intrinsics are actually serial
 *  operations, they don't parallelize the reduction across lanes. The correct approach is:
 *
 *  1. Accumulate vertically in SIMD registers throughout the entire loop
 *  2. Perform a single horizontal reduction at the very end, reconstructing the lane positions
 *
 *  @code{.c}
 *  __m512 sum_f32x16 = _mm512_setzero_ps();
 *  for (...) {
 *      __m512 data_f32x16 = _mm512_loadu_ps(data_pointer);
 *      sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
 *  }
 *  // Single horizontal reduce at the end only
 *  nk_f32_t result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
 *  @endcode
 *
 *  @section stride_handling Stride Handling Strategies
 *
 *  - stride == sizeof(scalar): Contiguous SIMD loads with masked tail
 *  - Large stride with gather support: Use gather instructions (32/64-bit types)
 *  - Otherwise: Serial fallback
 *
 *  @section reduce_argminmax Argmin/Argmax Strategy
 *
 *  Single-pass algorithm tracking both value and index in SIMD registers:
 *
 *  @code{.c}
 *  __m512 min_f32x16 = _mm512_set1_ps(NUMKONG_F32_INF);
 *  __m512i min_idx_i32x16 = _mm512_setzero_si512();
 *  __m512i current_idx_i32x16 = _mm512_setr_epi32(0,1,2,3,...,15);
 *  __m512i step_i32x16 = _mm512_set1_epi32(16);
 *  for (...) {
 *      __m512 data_f32x16 = _mm512_loadu_ps(data_pointer);
 *      __mmask16 lt_mask = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
 *      min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, data_f32x16);
 *      min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask, current_idx_i32x16);
 *      current_idx_i32x16 = _mm512_add_epi32(current_idx_i32x16, step_i32x16);
 *  }
 *  @endcode
 */
#ifndef NUMKONG_REDUCE_H
#define NUMKONG_REDUCE_H

#include "numkong/capabilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Horizontal moments reduction (sum + sum-of-squares) over a strided array.
 *  @param[in] data Pointer to the input data.
 *  @param[in] count Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride_bytes Byte stride between elements, `sizeof(*data)` for contiguous arrays.
 *  @param[out] sum_ptr Output sum.
 *  @param[out] sumsq_ptr Output sum of squares.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_reduce_moments_f64_best(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr, nk_capability_t capabilities,
                                                   void *stream);

/**
 *  @brief Horizontal min+max reduction with argmin/argmax over a strided array.
 *  @param[in] data Pointer to the input data.
 *  @param[in] count Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride_bytes Byte stride between elements, `sizeof(*data)` for contiguous arrays.
 *  @param[out] min_value_ptr Output minimum value.
 *  @param[out] min_index_ptr Output index of the minimum value.
 *  @param[out] max_value_ptr Output maximum value.
 *  @param[out] max_index_ptr Output index of the maximum value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_reduce_minmax_f64_best(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_f64_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                  nk_f64_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                  nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f32_best(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f32_best(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                  nk_f32_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                  nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_best(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                  void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i8_best(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                 nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                 nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_best(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                  void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u8_best(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                 nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                 nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_best(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i16_best(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                  nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                  nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u16_best(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u16_best(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                  nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                  nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i32_best(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i32_best(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                  nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                  nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u32_best(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u32_best(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                  nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                  nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i64_best(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i64_best(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                  nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                  nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u64_best(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u64_best(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                  nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                  nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f16_best(nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f16_best(nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                  nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                  nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_bf16_best(nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_bf16_best(nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                   nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                   nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_best(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_best(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                   nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                   nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_best(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_best(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                   nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                   nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_best(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_best(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                   nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                   nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_best(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_best(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                   nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                   nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m1_best(nk_e2m1x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr,
                                                    nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i4_best(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                  void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i4_best(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                 nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                 nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u4_best(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                  void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u4_best(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                 nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                 nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u1_best(nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr, nk_capability_t capabilities,
                                                  void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u1_best(nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,
                                                 nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr,
                                                 nk_capability_t capabilities, void *stream);

/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f32_serial(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f64_serial(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_serial(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_serial(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_serial(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u16_serial(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i32_serial(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u32_serial(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i64_serial(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u64_serial(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f16_serial(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_bf16_serial(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_serial(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_serial(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_serial(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m1_serial(nk_e2m1x2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_serial(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i4_serial(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u4_serial(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u1_serial(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                    void *stream);

/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f32_serial(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *,
                                                    nk_f32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f64_serial(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *,
                                                    nk_f64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i8_serial(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *,
                                                   nk_i8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u8_serial(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                   nk_u8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i16_serial(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *,
                                                    nk_i16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u16_serial(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *,
                                                    nk_u16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i32_serial(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *,
                                                    nk_i32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u32_serial(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *,
                                                    nk_u32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i64_serial(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *,
                                                    nk_i64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u64_serial(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *,
                                                    nk_u64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f16_serial(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *,
                                                    nk_f16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_bf16_serial(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                                     nk_bf16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_serial(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                                     nk_e4m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_serial(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                                     nk_e5m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_serial(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                                     nk_e2m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_serial(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                                     nk_e3m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i4_serial(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *,
                                                   nk_i8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u4_serial(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                   nk_u8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u1_serial(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                   nk_u8_t *, nk_size_t *, void *stream);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f32_neon(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f64_neon(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_neon(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_neon(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_neon(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u16_neon(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u1_neon(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i32_neon(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u32_neon(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i64_neon(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u64_neon(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_neon(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_neon(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_neon(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_neon(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f32_neon(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *,
                                                  nk_f32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f64_neon(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *,
                                                  nk_f64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i8_neon(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *,
                                                 nk_i8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u8_neon(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                 nk_u8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i16_neon(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *,
                                                  nk_i16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u16_neon(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *,
                                                  nk_u16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i32_neon(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *,
                                                  nk_i32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u32_neon(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *,
                                                  nk_u32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i64_neon(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *,
                                                  nk_i64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u64_neon(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *,
                                                  nk_u64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_neon(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                                   nk_e2m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_neon(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                                   nk_e3m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_neon(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                                   nk_e4m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_neon(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                                   nk_e5m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f16_neon(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                   void *stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_bf16_neonbfdot(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *,
                                                         nk_f32_t *, void *stream);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONSDOT
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_neonsdot(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_neonsdot(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_neonsdot(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                        void *stream);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_NEONFHM
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_neonfhm(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_neonfhm(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f32_haswell(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f64_haswell(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_haswell(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_haswell(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_haswell(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u16_haswell(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i32_haswell(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u32_haswell(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i64_haswell(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u64_haswell(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f16_haswell(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_bf16_haswell(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_haswell(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_haswell(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_haswell(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_haswell(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i4_haswell(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u4_haswell(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u1_haswell(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f32_haswell(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *,
                                                     nk_f32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f64_haswell(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *,
                                                     nk_f64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i8_haswell(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *,
                                                    nk_i8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u8_haswell(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                    nk_u8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i16_haswell(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *,
                                                     nk_i16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u16_haswell(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *,
                                                     nk_u16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i32_haswell(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *,
                                                     nk_i32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u32_haswell(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *,
                                                     nk_u32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i64_haswell(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *,
                                                     nk_i64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u64_haswell(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *,
                                                     nk_u64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f16_haswell(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *,
                                                     nk_f16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_bf16_haswell(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                                      nk_bf16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_haswell(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                                      nk_e4m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_haswell(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                                      nk_e5m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_haswell(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                                      nk_e2m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_haswell(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                                      nk_e3m2_t *, nk_size_t *, void *stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f32_skylake(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f64_skylake(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_skylake(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_skylake(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_skylake(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u16_skylake(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i32_skylake(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u32_skylake(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i64_skylake(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u64_skylake(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f16_skylake(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_bf16_skylake(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_skylake(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_skylake(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_skylake(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_skylake(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i4_skylake(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u4_skylake(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u1_skylake(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f32_skylake(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *,
                                                     nk_f32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f64_skylake(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *,
                                                     nk_f64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i8_skylake(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *,
                                                    nk_i8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u8_skylake(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                    nk_u8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i16_skylake(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *,
                                                     nk_i16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u16_skylake(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *,
                                                     nk_u16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i32_skylake(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *,
                                                     nk_i32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u32_skylake(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *,
                                                     nk_u32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i64_skylake(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *,
                                                     nk_i64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u64_skylake(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *,
                                                     nk_u64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f16_skylake(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *,
                                                     nk_f16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_bf16_skylake(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                                      nk_bf16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_skylake(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                                      nk_e4m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_skylake(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                                      nk_e5m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_skylake(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                                      nk_e2m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_skylake(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                                      nk_e3m2_t *, nk_size_t *, void *stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_icelake(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_icelake(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_icelake(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                      void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_icelake(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_icelake(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                       void *stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_bf16_genoa(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_genoa(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_genoa(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                     void *stream);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_ALDER
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_alder(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_alder(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u16_alder(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_alder(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                     void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_alder(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                     void *stream);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_SIERRA
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_sierra(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_sierra(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_sierra(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                      void *stream);
#endif // NUMKONG_TARGET_SIERRA

#if NUMKONG_TARGET_RVV
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f32_rvv(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f64_rvv(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_rvv(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                 void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_rvv(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                 void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_rvv(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u16_rvv(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i32_rvv(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u32_rvv(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i64_rvv(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u64_rvv(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f16_rvv(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_bf16_rvv(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_rvv(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_rvv(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_rvv(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_rvv(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                   void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f32_rvv(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *,
                                                 nk_f32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f64_rvv(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *,
                                                 nk_f64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i8_rvv(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *,
                                                nk_i8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u8_rvv(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                nk_u8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i16_rvv(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *,
                                                 nk_i16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u16_rvv(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *,
                                                 nk_u16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i32_rvv(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *,
                                                 nk_i32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u32_rvv(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *,
                                                 nk_u32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i64_rvv(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *,
                                                 nk_i64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u64_rvv(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *,
                                                 nk_u64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f16_rvv(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *,
                                                 nk_f16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_bf16_rvv(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                                  nk_bf16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_rvv(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                                  nk_e4m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_rvv(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                                  nk_e5m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_rvv(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                                  nk_e2m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_rvv(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                                  nk_e3m2_t *, nk_size_t *, void *stream);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f64_v128(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_bf16_v128(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_v128(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_v128(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_v128(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u16_v128(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i32_v128(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u32_v128(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                   void *stream);
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f32_v128relaxed(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *,
                                                          nk_f64_t *, void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i64_v128relaxed(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *,
                                                          nk_u64_t *, void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u64_v128relaxed(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *,
                                                          nk_u64_t *, void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f16_v128relaxed(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *,
                                                          nk_f32_t *, void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_v128relaxed(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *,
                                                           nk_f32_t *, void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_v128relaxed(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *,
                                                           nk_f32_t *, void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_v128relaxed(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *,
                                                           nk_f32_t *, void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_v128relaxed(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *,
                                                           nk_f32_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f32_v128relaxed(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *,
                                                         nk_size_t *, nk_f32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f64_v128relaxed(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *,
                                                         nk_size_t *, nk_f64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i8_v128relaxed(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *,
                                                        nk_i8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u8_v128relaxed(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                        nk_u8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i16_v128relaxed(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *,
                                                         nk_size_t *, nk_i16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u16_v128relaxed(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *,
                                                         nk_size_t *, nk_u16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i32_v128relaxed(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *,
                                                         nk_size_t *, nk_i32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u32_v128relaxed(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *,
                                                         nk_size_t *, nk_u32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i64_v128relaxed(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *,
                                                         nk_size_t *, nk_i64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u64_v128relaxed(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *,
                                                         nk_size_t *, nk_u64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f16_v128relaxed(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *,
                                                         nk_size_t *, nk_f16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_bf16_v128relaxed(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *,
                                                          nk_size_t *, nk_bf16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_v128relaxed(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *,
                                                          nk_size_t *, nk_e4m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_v128relaxed(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *,
                                                          nk_size_t *, nk_e5m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_v128relaxed(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *,
                                                          nk_size_t *, nk_e2m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_v128relaxed(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *,
                                                          nk_size_t *, nk_e3m2_t *, nk_size_t *, void *stream);
#endif // NUMKONG_TARGET_V128RELAXED

/*  GPU kernels take their CPU counterparts' arguments and return without waiting on the device;
 *  every operand is device memory of the vendor their capability names. */
#if NUMKONG_ARCH_CUDA_
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f32_cuda(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f64_cuda(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_f64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i8_cuda(nk_i8_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u8_cuda(nk_u8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i16_cuda(nk_i16_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u16_cuda(nk_u16_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i32_cuda(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u32_cuda(nk_u32_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i64_cuda(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u64_cuda(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_f16_cuda(nk_f16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                   void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_bf16_cuda(nk_bf16_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e4m3_cuda(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e5m2_cuda(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m3_cuda(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e3m2_cuda(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_e2m1_cuda(nk_e2m1x2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_f32_t *,
                                                    void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_i4_cuda(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u4_cuda(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_moments_f64_best */
NUMKONG_API nk_status_t nk_reduce_moments_u1_cuda(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_u64_t *,
                                                  void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f32_cuda(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *,
                                                  nk_f32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f64_cuda(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t *,
                                                  nk_f64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i8_cuda(nk_i8_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *,
                                                 nk_i8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u8_cuda(nk_u8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                 nk_u8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i16_cuda(nk_i16_t const *, nk_size_t, nk_size_t, nk_i16_t *, nk_size_t *,
                                                  nk_i16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u16_cuda(nk_u16_t const *, nk_size_t, nk_size_t, nk_u16_t *, nk_size_t *,
                                                  nk_u16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i32_cuda(nk_i32_t const *, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t *,
                                                  nk_i32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u32_cuda(nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t *,
                                                  nk_u32_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i64_cuda(nk_i64_t const *, nk_size_t, nk_size_t, nk_i64_t *, nk_size_t *,
                                                  nk_i64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u64_cuda(nk_u64_t const *, nk_size_t, nk_size_t, nk_u64_t *, nk_size_t *,
                                                  nk_u64_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_f16_cuda(nk_f16_t const *, nk_size_t, nk_size_t, nk_f16_t *, nk_size_t *,
                                                  nk_f16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_bf16_cuda(nk_bf16_t const *, nk_size_t, nk_size_t, nk_bf16_t *, nk_size_t *,
                                                   nk_bf16_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_cuda(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_e4m3_t *, nk_size_t *,
                                                   nk_e4m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_cuda(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_e5m2_t *, nk_size_t *,
                                                   nk_e5m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_cuda(nk_e2m3_t const *, nk_size_t, nk_size_t, nk_e2m3_t *, nk_size_t *,
                                                   nk_e2m3_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_cuda(nk_e3m2_t const *, nk_size_t, nk_size_t, nk_e3m2_t *, nk_size_t *,
                                                   nk_e3m2_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_i4_cuda(nk_i4x2_t const *, nk_size_t, nk_size_t, nk_i8_t *, nk_size_t *,
                                                 nk_i8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u4_cuda(nk_u4x2_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                 nk_u8_t *, nk_size_t *, void *stream);
/** @copydoc nk_reduce_minmax_f64_best */
NUMKONG_API nk_status_t nk_reduce_minmax_u1_cuda(nk_u1x8_t const *, nk_size_t, nk_size_t, nk_u8_t *, nk_size_t *,
                                                 nk_u8_t *, nk_size_t *, void *stream);
#endif // NUMKONG_ARCH_CUDA_

/**
 *  @brief Returns the accumulator dtype for the @c sum output of reduce_moments.
 *
 *  Float types accumulate into wider floats; signed ints into i64; unsigned ints into u64.
 */
NUMKONG_INLINE nk_dtype_t nk_reduce_moments_sum_dtype(nk_dtype_t dtype) {
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
 *  @brief Returns the accumulator dtype for the @c sumsq output of reduce_moments.
 *
 *  Same as sum except all integers (signed and unsigned) accumulate into u64.
 */
NUMKONG_INLINE nk_dtype_t nk_reduce_moments_sumsq_dtype(nk_dtype_t dtype) {
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

/** Returns the error bound of both reduce_moments outputs, per @c nk_accumulation_error_bound of
 *  their accumulators. */
NUMKONG_INLINE nk_f64_t nk_reduce_moments_error_bound(nk_dtype_t dtype) {
    return nk_accumulation_error_bound(nk_reduce_moments_sum_dtype(dtype));
}

/**
 *  @brief Returns the value dtype for reduce_minmax outputs.
 *
 *  Standard types return themselves. Sub-byte types widen: i4 → i8, u4 → u8, u1 → u8.
 */
NUMKONG_INLINE nk_dtype_t nk_reduce_minmax_value_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_i4_k: return nk_i8_k;
    case nk_u4_k: return nk_u8_k;
    case nk_u1_k: return nk_u8_k;
    default: return dtype;
    }
}

/**
 *  @brief Finds the reduce kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when no capability in @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_reduce_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability);

#ifdef __cplusplus
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/reduce/serial.h"
#include "numkong/reduce/neon.h"
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
#include "numkong/reduce/v128.h"
#include "numkong/reduce/v128relaxed.h"
#include "numkong/reduce/simt.cuh"

#ifdef __cplusplus
extern "C" {
#endif

NUMKONG_API nk_status_t nk_reduce_moments_f64_best(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_f64_t *sum, nk_f64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_f64_best(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_f64_t *min_value, nk_size_t *min_index, nk_f64_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_f32_best(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_f64_t *sum, nk_f64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_f32_best(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_f32_t *min_value, nk_size_t *min_index, nk_f32_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_bf16_best(nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_bf16_best(nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_bf16_t *min_value, nk_size_t *min_index, nk_bf16_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_f16_best(nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_f16_best(nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_f16_t *min_value, nk_size_t *min_index, nk_f16_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_e5m2_best(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_best(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_e5m2_t *min_value, nk_size_t *min_index, nk_e5m2_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_e4m3_best(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_best(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_e4m3_t *min_value, nk_size_t *min_index, nk_e4m3_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_e3m2_best(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_best(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_e3m2_t *min_value, nk_size_t *min_index, nk_e3m2_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_e2m3_best(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_best(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_e2m3_t *min_value, nk_size_t *min_index, nk_e2m3_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_e2m1_best(nk_e2m1x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_i64_best(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_i64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i64_best(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i64_t *min_value, nk_size_t *min_index, nk_i64_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_i32_best(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_i64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i32_best(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i32_t *min_value, nk_size_t *min_index, nk_i32_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_i16_best(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_i64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i16_best(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i16_t *min_value, nk_size_t *min_index, nk_i16_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_i8_best(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                  void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i8_best(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_i8_t *min_value, nk_size_t *min_index, nk_i8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_i4_best(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_i64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                  void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i4_best(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_i8_t *min_value, nk_size_t *min_index, nk_i8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_u64_best(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u64_best(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u64_t *min_value, nk_size_t *min_index, nk_u64_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_u32_best(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u32_best(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u32_t *min_value, nk_size_t *min_index, nk_u32_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_u16_best(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                   nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u16_best(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u16_t *min_value, nk_size_t *min_index, nk_u16_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_u8_best(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                  void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u8_best(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_u8_t *min_value, nk_size_t *min_index, nk_u8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_u4_best(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                  void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u4_best(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_u8_t *min_value, nk_size_t *min_index, nk_u8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_moments_u1_best(nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                  void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum), nk_unused_(sumsq),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u1_best(nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 nk_u8_t *min_value, nk_size_t *min_index, nk_u8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_reduce_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = (nk_kernel_punned_t)NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_REDUCE_H
