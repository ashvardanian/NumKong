/**
 *  @file include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief SIMD-accelerated spatial similarity measures.
 *
 *  Contains following similarity measures:
 *
 *  - L2 (Euclidean) regular and squared distance
 *  - Cosine (Angular) distance - @b not similarity!
 *
 *  For dtypes:
 *
 *  - f64: 64-bit IEEE floating point numbers → 64-bit floats
 *  - f32: 32-bit IEEE floating point numbers → 64-bit floats
 *  - f16: 16-bit IEEE floating point numbers → 32-bit floats
 *  - bf16: 16-bit brain floating point numbers → 32-bit floats
 *  - e4m3: 8-bit e4m3 floating point numbers → 32-bit floats
 *  - e5m2: 8-bit e5m2 floating point numbers → 32-bit floats
 *  - e2m3: 8-bit e2m3 floating point numbers (MX) → 32-bit floats
 *  - e3m2: 8-bit e3m2 floating point numbers (MX) → 32-bit floats
 *  - i8: 8-bit signed integers → 32-bit floats
 *  - u8: 8-bit unsigned integers → 32-bit floats
 *  - i4: 4-bit signed integers (packed pairs) → 32-bit floats
 *  - u4: 4-bit unsigned integers (packed pairs) → 32-bit floats
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, NEON+F16, NEON+BF16, NEON+SDOT, SVE, SVE+F16, SVE+BF16
 *  - x86: Haswell, Skylake, Ice Lake, Genoa, Sapphire Rapids, Sierra Forest
 *  - RISC-V: RVV, RVV+BF16, RVV+HALF
 *  - WASM: V128, V128Relaxed
 *
 *  @section spatial_numerical_stability Numerical Stability
 *
 *  Every capability follows the precision tiers of the dot products. @c f16, @c bf16 and the
 *  mini-floats multiply and sum in @c f32. @c f32 inputs widen first, so their products are exact
 *  in @c f64 and sum there, and their public outputs are @c f64. @c f64 inputs sum in Dot2, TwoProd
 *  and TwoSum, for the dot product, both norms and the squared differences alike, giving O(1) error
 *  growth regardless of vector dimension. Angular finalization uses rsqrt via magic constant plus 3
 *  Newton-Raphson iterations for ~34.9 correct bits in f32, or 4 iterations for ~69.3 bits in f64,
 *  then clamps the result to ≥ 0. L2 clamps @c dist_sq to zero before the square root, avoiding NaN
 *  from rounding error. Integer types, i8/u8/i4/u4, accumulate squared differences in i32,
 *  overflowing at n > 2^31/65,025 ≈ 33K for i8, max diff² = 255². Output is cast to f32.
 *
 *  @section spatial_streaming_api Streaming API
 *
 *  Angular and L2 distances can be computed from a single dot-product stream and precomputed
 *  magnitudes. The dot-product state helpers accumulate just a · b, four targets at a time, and the
 *  @c from_dot finalizers take the squared L2 norms of the full vectors and compute:
 *
 *  @verbatim
 *  a·b           = Σᵢ aᵢbᵢ
 *  ‖a‖           = √Σᵢ aᵢ²
 *  angular(a, b) = 1 − a·b / ‖a‖‖b‖
 *  l2(a, b)      = √(‖a‖² + ‖b‖² − 2a·b)
 *  @endverbatim
 *
 *  The angular @c from_dot finalizers follow one rule on every capability: NaN when a · b is
 *  NaN, 0 when both norms are zero, 1 when either norm or a · b is exactly zero, and otherwise
 *  1 − a · b · rsqrt(‖a‖²) · rsqrt(‖b‖²) clamped to ≥ 0, where an infinite norm has a
 *  reciprocal square root of 0. The L2 ones clamp their square-root argument at 0 to avoid
 *  negatives from rounding, and keep NaN.
 *
 *  @code{.c}
 *  nk_b128_vec_t query_block, target_blocks[4];
 *  nk_f64_t query_sumsq = ...;          // Precomputed squared L2 norm of the full query
 *  nk_b256_vec_t target_sumsqs = ...;   // Precomputed squared L2 norms of the full targets
 *  nk_b256_vec_t dots, distances;
 *  nk_dot_f32x4_state_haswell_t states[4];
 *  for (int i = 0; i != 4; ++i) nk_dot_f32x4_init_haswell(&states[i]);
 *  for (int i = 0; i != 4; ++i) nk_dot_f32x4_update_haswell(&states[i], query_block, target_blocks[i], 0, 4);
 *  nk_dot_f32x4_finalize_haswell(&states[0], &states[1], &states[2], &states[3], 4, &dots);
 *  nk_angular_through_f64_from_dot_haswell_(&dots, query_sumsq, &target_sumsqs, &distances);
 *  @endcode
 *
 *  @section rsqrt_notes Reciprocal Square Root and Newton-Raphson Notes
 *
 *  Angular distance normalization uses reciprocal square roots to avoid the latency of full
 *  sqrt/div pipelines. We refine the rsqrt estimate with Newton-Raphson iterations to reduce error:
 *  one on x86, two on Arm NEON.
 *
 *  Relevant instructions and caveats:
 *
 *  @verbatim
 *  Intrinsic                Instruction      Notes
 *  _mm_rsqrt_ps             VRSQRTPS         fast approx; refine with NR
 *  _mm_maskz_rsqrt14_pd     VRSQRT14PD       higher-precision approx; MSVC masked-only
 *  _mm_sqrt_ps/_mm_sqrt_pd  VSQRTPS/VSQRTPD  higher latency, sqrt/div unit
 *  @endverbatim
 *
 *  Latency/port notes, rule of thumb:
 *  - On Intel client cores, sqrt/rsqrt execute on the divide/sqrt unit, often port 0, and can
 *    bottleneck tight loops.
 *  - NR refinement uses mul/FMA ports and amortizes well when @c ab is reduced to a scalar and
 *    reused for finalization.
 *  - Arm NEON @c rsqrt is coarse; two refinement steps keep angular distance error bounded.
 *
 *  @section spatial_x86_instructions Relevant x86 Instructions
 *
 *  AVX2 lacks signed 8-bit dot products, so Haswell widens to i16 and uses VPMADDWD.
 *  AVX-512 VNNI replaces that with VPDPWSSD. BF16 uses VDPBF16PS where available to avoid
 *  convert+FMA sequences; if the ISA lacks it, we fall back to f32 FMA in the AVX2/serial:
 *
 *  @verbatim
 *  Intrinsic             Instruction                   Icelake    Genoa
 *  _mm256_fmadd_ps       VFMADD231PS (YMM, YMM, YMM)   4cy @ p01  4cy @ p01
 *  _mm256_fmadd_pd       VFMADD231PD (YMM, YMM, YMM)   4cy @ p01  4cy @ p01
 *  _mm256_madd_epi16     VPMADDWD (YMM, YMM, YMM)      5cy @ p01  3cy @ p01
 *  _mm512_dpwssd_epi32   VPDPWSSD (ZMM, K, ZMM, ZMM)   5cy @ p05  4cy @ p01
 *  _mm512_dpbf16_ps      VDPBF16PS (ZMM, K, ZMM, ZMM)  n/a        6cy @ p01
 *  _mm_rsqrt_ps          VRSQRTPS (XMM, XMM)           5cy @ p0   4cy @ p01
 *  _mm_maskz_rsqrt14_pd  VRSQRT14PD (XMM, K, XMM)      4cy @ p0   5cy @ p01
 *  _mm_sqrt_ps           VSQRTPS (XMM, XMM)            12cy @ p0  15cy @ p01
 *  @endverbatim
 *
 *  @section spatial_arm_instructions Relevant Arm Instructions
 *
 *  The NEON/SVE kernels in this header are structured around FMLA/SDOT/BFDOT loops, which is why we
 *  avoid mul+add splits and keep reductions to scalars before square roots. Dot-product kernels for
 *  i8/u8 are only built when the "dotprod+i8mm" target is enabled; otherwise we rely on the serial
 *  backends. BF16 kernels are enabled only with BF16 dot instructions, skipping @c vbfmlal and
 *  @c vbfmlalt to limit shuffle overhead and complexity.
 *
 *  @verbatim
 *  Intrinsic     Instruction      M1 Firestorm
 *  vfmaq_f32     FMLA.S (vec)     4c / 4c
 *  vfmaq_f64     FMLA.D (vec)     4c / 4c
 *  vdotq_s32     SDOT.B (vec)     3c / 4c
 *  vbfdotq_f32   BFDOT (vec)      n/a
 *  vrsqrteq_f32  FRSQRTE.S (vec)  3c / 1c
 *  vrsqrtsq_f32  FRSQRTS.S (vec)  4c / 4c
 *  vsqrtq_f32    FSQRT.S (vec)    10c / 0.5c
 *  @endverbatim
 *
 *  @section spatial_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NUMKONG_SPATIAL_H
#define NUMKONG_SPATIAL_H

#include "numkong/capabilities.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief L2 (Euclidean) distance between two vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n Counts dimensions, a multiple of the values per byte.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NUMKONG_API nk_status_t nk_euclidean_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Squared L2 (Euclidean) distance between two vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n Counts dimensions, a multiple of the values per byte.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Angular (cosine) distance between two vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n Counts dimensions, a multiple of the values per byte.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NUMKONG_API nk_status_t nk_angular_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream);

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats. */
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                   nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                   nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                   nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_serial(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_serial(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                   nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_serial(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_serial(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_serial(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                   nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_serial(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);

/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words. By far
 *  the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023. */
#if NUMKONG_TARGET_NEON
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONSDOT
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i4_neonsdot(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i4_neonsdot(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n,
                                                   nk_u32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i4_neonsdot(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u4_neonsdot(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u4_neonsdot(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n,
                                                   nk_u32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u4_neonsdot(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_SVESDOT
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_svesdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_svesdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_svesdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_svesdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_svesdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_svesdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
#endif // NUMKONG_TARGET_SVESDOT

#if NUMKONG_TARGET_NEONFP8
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_neonfp8(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_neonfp8(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_neonfp8(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_neonfp8(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e5m2_neonfp8(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e5m2_neonfp8(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_neonfp8(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_neonfp8(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_neonfp8(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_neonfp8(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_neonfp8(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_neonfp8(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONFP8

/*  SIMD-powered backends for Arm SVE, mostly using 32-bit arithmetic over variable-length
 *  platform-defined word sizes. Designed for Arm Graviton 3, Microsoft Cobalt, as well as NVIDIA
 *  Grace and newer Ampere Altra CPUs. */
#if NUMKONG_TARGET_SVE
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                           nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                           nk_stream_t stream);
#endif // NUMKONG_TARGET_SVE

#if NUMKONG_TARGET_SVEHALF
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
#endif // NUMKONG_TARGET_SVEHALF

#if NUMKONG_TARGET_SVEBFDOT
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                   nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                     nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
#endif // NUMKONG_TARGET_SVEBFDOT

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic
 *  over 256-bit words. First demonstrated in 2011, at least one Haswell-based processor was
 *  still being sold in 2022 — the Pentium G3420. Practically all modern x86 CPUs support AVX2,
 *  FMA, and F16C, making it a perfect baseline for SIMD algorithms. On other hand, there is no
 *  need to implement AVX2 versions of @c f32 and @c f64 functions, as those are properly
 *  vectorized by recent compilers. */
#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_haswell(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_haswell(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_haswell(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_haswell(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_haswell(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_haswell(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
#endif // NUMKONG_TARGET_HASWELL

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer, using 32-bit arithmetic
 *  over 512-bit words. Skylake was launched in 2015, and discontinued in 2019. Skylake had support
 *  for F, CD, VL, DQ, and BW extensions, as well as masked operations. This is enough to supersede
 *  auto-vectorization on @c f32 and @c f64 types.
 *
 *  Sadly, we can't effectively interleave different kinds of arithmetic instructions to utilize
 *  more ports, as Chips and Cheese explains:
 *
 *  "Like Intel server architectures since Skylake-X, SPR cores feature two 512-bit FMA units, and
 *  organize them in a similar fashion. One 512-bit FMA unit is created by fusing two 256-bit ones
 *  on port 0 and port 1. The other is added to port 5, as a server-specific core extension. The FMA
 *  units on port 0 and 1 are configured into 2×256-bit or 1×512-bit mode depending on whether
 *  512-bit FMA instructions are present in the scheduler. That means a mix of 256-bit and 512-bit
 *  FMA instructions will not achieve higher IPC than executing 512-bit instructions alone."
 *
 *  Source: https://chipsandcheese.com/p/a-peek-at-sapphire-rapids */
#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
#endif // NUMKONG_TARGET_SKYLAKE

/*  SIMD-powered backends for AVX512 CPUs of Ice Lake generation and newer, using mixed arithmetic
 *  over 512-bit words. Ice Lake added VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG,
 *  VPCLMULQDQ, and other extensions for integral operations. Sapphire Rapids added tiled matrix
 *  operations, but we are most interested in the new mixed-precision FMA instructions. */
#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i4_icelake(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i4_icelake(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i4_icelake(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u4_icelake(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u4_icelake(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u4_icelake(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_icelake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_icelake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_icelake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_icelake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_icelake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_icelake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_icelake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_icelake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_icelake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_icelake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_icelake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_icelake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_DIAMOND
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_diamond(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_diamond(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_diamond(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_diamond(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_diamond(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_diamond(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e5m2_diamond(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_diamond(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e5m2_diamond(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
#endif // NUMKONG_TARGET_DIAMOND

/*  SIMD-powered backends for AVX-INT8-VNNI extensions on Xeon 6 CPUs, including Sierra Forest and
 *  Granite Rapids. It packs many "efficiency" cores into a single socket, avoiding heavy 512-bit
 *  operations, and focusing on 256-bit ones. */
#if NUMKONG_TARGET_SIERRA
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_sierra(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_sierra(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_sierra(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_sierra(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_sierra(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_sierra(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                   nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_sierra(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_sierra(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                   nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_sierra(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
#endif // NUMKONG_TARGET_SIERRA

#if NUMKONG_TARGET_ALDER
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_alder(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_alder(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_alder(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_alder(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_alder(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_alder(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_alder(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_alder(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_alder(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_alder(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_alder(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_alder(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_V128
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_stream_t stream);
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                       nk_f64_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                       nk_f64_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f64_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                       nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                        nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                      nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                    nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                      nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                    nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_v128relaxed(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                        nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_v128relaxed(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                      nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_v128relaxed(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_v128relaxed(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                        nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e5m2_v128relaxed(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                      nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e5m2_v128relaxed(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_v128relaxed(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                        nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e2m3_v128relaxed(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                      nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e2m3_v128relaxed(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_v128relaxed(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                        nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e3m2_v128relaxed(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                      nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e3m2_v128relaxed(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
#endif // NUMKONG_TARGET_V128RELAXED

/*  SIMD-powered backends for RISC-V Vector extension, using scalable vector arithmetic.
 *  Designed for SiFive, T-Head, and other RISC-V processors with the V extension. */
#if NUMKONG_TARGET_RVV
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                           nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                           nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                          nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                          nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i4_rvv(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i4_rvv(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i4_rvv(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                          nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u4_rvv(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u4_rvv(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                              nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u4_rvv(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                          nk_stream_t stream);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_RVVHALF
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_rvvhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_rvvhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_rvvhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
#endif // NUMKONG_TARGET_RVVHALF

#if NUMKONG_TARGET_RVVBF16
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_rvvbf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_rvvbf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_rvvbf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
#endif // NUMKONG_TARGET_RVVBF16

#if NUMKONG_TARGET_POWERVSX
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                    nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                    nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                    nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                   nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                     nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream);
#endif // NUMKONG_TARGET_POWERVSX

#if NUMKONG_TARGET_LOONGSONASX
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f64_loongsonasx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f64_loongsonasx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                       nk_f64_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f64_loongsonasx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f32_loongsonasx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f64_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f32_loongsonasx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                       nk_f64_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f32_loongsonasx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_f16_loongsonasx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_f16_loongsonasx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                       nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_f16_loongsonasx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                   nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_bf16_loongsonasx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_loongsonasx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                        nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_bf16_loongsonasx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_i8_loongsonasx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                    nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_i8_loongsonasx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                      nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_i8_loongsonasx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
/** @copydoc nk_euclidean_f64_best */
NUMKONG_API nk_status_t nk_euclidean_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                    nk_stream_t stream);
/** @copydoc nk_sqeuclidean_f64_best */
NUMKONG_API nk_status_t nk_sqeuclidean_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                      nk_stream_t stream);
/** @copydoc nk_angular_f64_best */
NUMKONG_API nk_status_t nk_angular_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream);
#endif // NUMKONG_TARGET_LOONGSONASX

/** Returns the output dtype for L2 (Euclidean) distance. */
NUMKONG_INLINE nk_dtype_t nk_euclidean_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_e2m3_k: return nk_f32_k;
    case nk_e2m1_k: return nk_f32_k;
    case nk_nvfp4_k:
    case nk_mxfp4_k:
    case nk_mxfp6e2m3_k:
    case nk_mxfp6e3m2_k:
    case nk_mxfp8e4m3_k:
    case nk_mxfp8e5m2_k: return nk_f32_k;
    case nk_e3m2_k: return nk_f32_k;
    case nk_i8_k: return nk_f32_k;
    case nk_u8_k: return nk_f32_k;
    case nk_i4_k: return nk_f32_k;
    case nk_u4_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the output dtype for L2 squared distance. */
NUMKONG_INLINE nk_dtype_t nk_sqeuclidean_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_e2m3_k: return nk_f32_k;
    case nk_e3m2_k: return nk_f32_k;
    case nk_i8_k: return nk_u32_k;
    case nk_u8_k: return nk_u32_k;
    case nk_i4_k: return nk_u32_k;
    case nk_u4_k: return nk_u32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the output dtype for angular/cosine distance. */
NUMKONG_INLINE nk_dtype_t nk_angular_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_e2m3_k: return nk_f32_k;
    case nk_e2m1_k: return nk_f32_k;
    case nk_nvfp4_k:
    case nk_mxfp4_k:
    case nk_mxfp6e2m3_k:
    case nk_mxfp6e3m2_k:
    case nk_mxfp8e4m3_k:
    case nk_mxfp8e5m2_k: return nk_f32_k;
    case nk_e3m2_k: return nk_f32_k;
    case nk_i8_k: return nk_f32_k;
    case nk_u8_k: return nk_f32_k;
    case nk_i4_k: return nk_f32_k;
    case nk_u4_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the error bound of squared Euclidean distances, per @c nk_accumulation_error_bound of
 *  their output. */
NUMKONG_INLINE nk_f64_t nk_sqeuclidean_error_bound(nk_dtype_t dtype) {
    return nk_accumulation_error_bound(nk_sqeuclidean_output_dtype(dtype));
}

/** Returns the error bound of Euclidean distances, per @c nk_accumulation_error_bound of their
 *  output. */
NUMKONG_INLINE nk_f64_t nk_euclidean_error_bound(nk_dtype_t dtype) {
    return nk_accumulation_error_bound(nk_euclidean_output_dtype(dtype));
}

/** Returns the error bound of angular distances, per @c nk_accumulation_error_bound of their
 *  output. */
NUMKONG_INLINE nk_f64_t nk_angular_error_bound(nk_dtype_t dtype) {
    return nk_accumulation_error_bound(nk_angular_output_dtype(dtype));
}

/** Returns the error bound each term adds to the sums of a distance: none for F64 inputs, whose
 *  sums are compensated, and that of the terms themselves for the rest. */
NUMKONG_INLINE nk_f64_t nk_spatial_sum_error_bound(nk_dtype_t dtype) {
    return dtype == nk_f64_k ? 0 : nk_euclidean_error_bound(dtype);
}

/**
 *  @brief Finds the spatial kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_spatial_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                               nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/spatial/serial.h"
#include "numkong/spatial/neon.h"
#include "numkong/spatial/neonbfdot.h"
#include "numkong/spatial/neonsdot.h"
#include "numkong/spatial/sve.h"
#include "numkong/spatial/svehalf.h"
#include "numkong/spatial/svebfdot.h"
#include "numkong/spatial/svesdot.h"
#include "numkong/spatial/neonfp8.h"
#include "numkong/spatial/haswell.h"
#include "numkong/spatial/skylake.h"
#include "numkong/spatial/genoa.h"
#include "numkong/spatial/diamond.h"
#include "numkong/spatial/icelake.h"
#include "numkong/spatial/alder.h"
#include "numkong/spatial/sierra.h"
#include "numkong/spatial/rvv.h"
#include "numkong/spatial/rvvhalf.h"
#include "numkong/spatial/rvvbf16.h"
#include "numkong/spatial/v128.h"
#include "numkong/spatial/v128relaxed.h"
#include "numkong/spatial/powervsx.h"
#include "numkong/spatial/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_angular_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angular_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclidean_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_spatial_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                               nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif
