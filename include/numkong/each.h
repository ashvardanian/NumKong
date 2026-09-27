/**
 *  @file include/numkong/each.h
 *  @author Ash Vardanian
 *  @date October 16, 2024
 *  @brief SIMD-accelerated elementwise arithmetic.
 *
 *  Contains following element-wise operations:
 *
 *  - Scale (Multiply) with shift: result[i] = α · a[i] + β
 *  - Sum (Add): result[i] = a[i] + b[i]
 *  - Blend: result[i] = α · a[i] + β · b[i]
 *  - FMA (Fused Multiply-Add): result[i] = α · a[i] · b[i] + β · c[i]
 *
 *  Beyond their obvious usecases, those can be reused for vector-scalar math and other operations:
 *
 *  - Scale with β = 0 for a pure multiply.
 *  - Sum is equivalent to WSum with α = β = 1.
 *  - Average is WSum with α = β = 0.5.
 *  - Elementwise multiply is FMA with β = 0.
 *
 *  For dtypes:
 *
 *  - f64c: 64-bit complex × 64-bit complex scales
 *  - f32c: 32-bit complex × 32-bit complex scales
 *  - f64: 64-bit IEEE floating point × 64-bit scales
 *  - f32: 32-bit IEEE floating point × 32-bit scales
 *  - f16: 16-bit IEEE floating point × 32-bit scales
 *  - bf16: 16-bit brain floating point × 32-bit scales
 *  - e4m3: 8-bit e4m3 floating point × 32-bit scales
 *  - e5m2: 8-bit e5m2 floating point × 32-bit scales
 *  - e2m3: 8-bit e2m3 floating point (MX) × 32-bit scales
 *  - e3m2: 8-bit e3m2 floating point (MX) × 32-bit scales
 *  - i8/u8: 8-bit integers × 32-bit scales
 *  - i16/u16: 16-bit integers × 32-bit scales
 *  - i32/u32: 32-bit integers × 64-bit scales
 *  - i64/u64: 64-bit integers × 64-bit scales
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, NEON+F16, NEON+BF16
 *  - x86: Haswell, Skylake, Ice Lake, Sapphire Rapids
 *  - RISC-V: RVV
 *
 *  @section each_numerical_stability Numerical Stability
 *
 *  Integer sum is elementwise a[i]+b[i], clamped to the type's range. Serial widens to i64 and
 *  clamps on store, while NEON uses hardware saturating adds, SQADD/UQADD. An f16/bf16/FP8 sum is
 *  promoted to f32, added, and truncated back, so double rounding is possible. Scale/blend/fma use
 *  float alpha/beta arithmetic, round to nearest with ties to even, then clamp. f32/f64 operations
 *  are native precision with no widening.
 *
 *  @section each_x86_instructions Relevant x86 Instructions
 *
 *  FP16 conversions, VCVTPH2PS/VCVTPS2PH, serve f16 scale/sum/blend/fma operations, converting to
 *  f32 for arithmetic and back. The 6-7 cycle latency is amortized over vector-width elements.
 *  Saturating integer adds, VPADDSW/VPADDUSW, protect i16/u16 sums from overflow without branching.
 *  FMA, VFMADD231PS, is the workhorse for scale (alpha*x+beta) and blend (alpha*a+beta*b).
 *
 *  @verbatim
 *  Intrinsic               Instruction                  Icelake      Genoa
 *  _mm512_cvtph_ps         VCVTPH2PS (ZMM, YMM)         7cy @ p0+p5  6cy @ p12+p23
 *  _mm512_cvtps_ph         VCVTPS2PH (YMM, ZMM, I8)     7cy @ p0+p5  7cy @ p12+p23
 *  _mm256_adds_epi16       VPADDSW (YMM, YMM, YMM)      1cy @ p01    n/a
 *  _mm256_adds_epu16       VPADDUSW (YMM, YMM, YMM)     1cy @ p01    n/a
 *  _mm512_fpclass_ps_mask  VFPCLASSPS (K, ZMM, I8)      3cy @ p5     5cy @ p01
 *  _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)  4cy @ p01    4cy @ p01
 *  @endverbatim
 *
 *  @section each_arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  On ARM, i8/u8 and f16 scale/blend/fma widen to f32 intermediates, as f16 ones would round where
 *  the serial kernels do not. Saturating adds (SQADD/UQADD) handle integer overflow. FMLA provides
 *  fused multiply-add for floating-point scale/blend/fma.
 *
 *  @verbatim
 *  Intrinsic       Instruction   M1 Firestorm  Graviton 3   Graviton 4
 *  vfmaq_f32       FMLA.S (vec)  4cy @ V0123   4cy @ V0123  4cy @ V0123
 *  vqaddq_s16      SQADD (vec)   3cy @ V0123   2cy @ V0123  2cy @ V0123
 *  vqaddq_u16      UQADD (vec)   3cy @ V0123   2cy @ V0123  2cy @ V0123
 *  vcvtq_f32_s32   SCVTF (vec)   3cy @ V0123   3cy @ V01    3cy @ V01
 *  vcvtnq_s32_f32  FCVTNS (vec)  3cy @ V0123   3cy @ V01    3cy @ V01
 *  @endverbatim
 *
 *  @section each_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NUMKONG_EACH_H
#define NUMKONG_EACH_H

#include "numkong/capabilities.h" // `nk_capability_kernels_t`, `nk_kernel_pick_`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Element-wise scale with shift: result[i] = alpha * a[i] + beta.
 *
 *  @param[in] a The input vector.
 *  @param[in] n The number of elements in the vector.
 *  @param[in] alpha Pointer to the scaling factor (type depends on input precision).
 *  @param[in] beta Pointer to the shift (bias) value (type depends on input precision).
 *  @param[out] result The output vector.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_f64_best(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_f64_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_f32_best(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_f32_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_f16_best(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_f16_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_bf16_best(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_bf16_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_i8_best(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                      nk_f32_t const *beta, nk_i8_t *result,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_u8_best(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                      nk_f32_t const *beta, nk_u8_t *result,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_i16_best(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_i16_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_u16_best(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_u16_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_i32_best(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_i32_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_u32_best(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_u32_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_i64_best(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_i64_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_u64_best(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_u64_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_e4m3_best(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_e4m3_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_e5m2_best(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_e5m2_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_e2m3_best(nk_e2m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_e2m3_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_e3m2_best(nk_e3m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_e3m2_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_f32c_best(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                        nk_f32c_t const *beta, nk_f32c_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_scale_f64c_best(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                        nk_f64c_t const *beta, nk_f64c_t *result,
                                                        nk_capability_t capabilities, void *stream);

/**
 *  @brief Element-wise sum: result[i] = a[i] + b[i].
 *
 *  @param[in] a The first input vector.
 *  @param[in] b The second input vector.
 *  @param[in] n The number of elements in the vectors.
 *  @param[out] result The output vector.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f16_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_bf16_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_i16_best(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                     nk_i16_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                     nk_u16_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_i32_best(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                     nk_i32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                     nk_u32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_i64_best(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n,
                                                     nk_i64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n,
                                                     nk_u64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                      nk_e4m3_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                      nk_e5m2_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                      nk_e2m3_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                      nk_e3m2_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                      nk_f32c_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_sum_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                      nk_f64c_t *result, nk_capability_t capabilities, void *stream);

/**
 *  @brief Weighted sum: result[i] = alpha * a[i] + beta * b[i].
 *
 *  @param[in] a The first input vector.
 *  @param[in] b The second input vector.
 *  @param[in] n The number of elements in the vectors.
 *  @param[in] alpha Pointer to the first weight (type depends on input precision).
 *  @param[in] beta Pointer to the second weight (type depends on input precision).
 *  @param[out] result The output vector.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                       nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                      nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                      nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_i16_best(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_i32_best(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                       nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                       nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_i64_best(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n,
                                                       nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n,
                                                       nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e2m3_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e3m2_t *result,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                        nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                        nk_f32c_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_blend_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                        nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                        nk_f64c_t *result, nk_capability_t capabilities, void *stream);

/**
 *  @brief Fused multiply-add: result[i] = alpha * a[i] * b[i] + beta * c[i].
 *
 *  @param[in] a The first input vector.
 *  @param[in] b The second input vector.
 *  @param[in] c The third input vector.
 *  @param[in] n The number of elements in the vectors.
 *  @param[in] alpha Pointer to the scaling factor for a[i] * b[i] (type depends
 *      on input precision).
 *  @param[in] beta Pointer to the scaling factor for c[i] (type depends on input precision).
 *  @param[out] result The output vector.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_f64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                     nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                     nk_f32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                     nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                     nk_f16_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_bf16_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_i16_best(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c,
                                                     nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                     nk_i16_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c,
                                                     nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                     nk_u16_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_i32_best(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_i32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_u32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_i64_best(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_i64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_u64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_e4m3_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_e5m2_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_e2m3_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_e2m3_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_e3m2_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_e3m2_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c,
                                                      nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                      nk_f32c_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_fma_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c,
                                                      nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                      nk_f64c_t *result, nk_capability_t capabilities, void *stream);

/**
 *  @brief Fused SwiGLU: result[i] = silu(gate[i] × s) × up[i] × s, where s is @p input_scale.
 *
 *  @param[in] gate The gate input matrix of shape rows by cols.
 *  @param[in] up The up input matrix, same shape as gate; NULL collapses to plain SiLU.
 *  @param[out] y The output matrix, same shape and dtype as the inputs; may alias gate.
 *  @param[in] rows The number of rows in each matrix.
 *  @param[in] cols The number of columns in each matrix.
 *  @param[in] gate_row_stride The row stride of gate in bytes.
 *  @param[in] up_row_stride The row stride of up in bytes.
 *  @param[in] y_row_stride The row stride of y in bytes.
 *  @param[in] input_scale Scalar folded onto every loaded element (E4M3 descale; 1.0 for BF16/F32).
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API_RUNTIME nk_status_t nk_each_swiglu_f32_best(nk_f32_t const *gate, nk_f32_t const *up, nk_f32_t *y,
                                                        nk_size_t rows, nk_size_t cols, nk_size_t gate_row_stride,
                                                        nk_size_t up_row_stride, nk_size_t y_row_stride,
                                                        nk_f32_t input_scale, nk_capability_t capabilities,
                                                        void *stream);
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_swiglu_bf16_best(nk_bf16_t const *gate, nk_bf16_t const *up, nk_bf16_t *y,
                                                         nk_size_t rows, nk_size_t cols, nk_size_t gate_row_stride,
                                                         nk_size_t up_row_stride, nk_size_t y_row_stride,
                                                         nk_f32_t input_scale, nk_capability_t capabilities,
                                                         void *stream);
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_each_swiglu_e4m3_best(nk_e4m3_t const *gate, nk_e4m3_t const *up, nk_e4m3_t *y,
                                                         nk_size_t rows, nk_size_t cols, nk_size_t gate_row_stride,
                                                         nk_size_t up_row_stride, nk_size_t y_row_stride,
                                                         nk_f32_t input_scale, nk_capability_t capabilities,
                                                         void *stream);

/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64_serial(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                          nk_f64_t const *beta, nk_f64_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32_serial(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                          nk_f32_t const *beta, nk_f32_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f16_serial(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                          nk_f32_t const *beta, nk_f16_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_bf16_serial(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_bf16_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i8_serial(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                         nk_f32_t const *beta, nk_i8_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u8_serial(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                         nk_f32_t const *beta, nk_u8_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i16_serial(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                          nk_f32_t const *beta, nk_i16_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u16_serial(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                          nk_f32_t const *beta, nk_u16_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i32_serial(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                          nk_f64_t const *beta, nk_i32_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u32_serial(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                          nk_f64_t const *beta, nk_u32_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i64_serial(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                          nk_f64_t const *beta, nk_i64_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u64_serial(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                          nk_f64_t const *beta, nk_u64_t *result, void *stream);

/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                        nk_f64_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                        nk_f32_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                        nk_f16_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                         nk_bf16_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result,
                                                       void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result,
                                                       void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i16_serial(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                        nk_i16_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                        nk_u16_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i32_serial(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                        nk_i32_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                        nk_u32_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i64_serial(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n,
                                                        nk_i64_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n,
                                                        nk_u64_t *result, void *stream);

/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                          nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                          nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                          nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                           nk_f32_t const *alpha, nk_f32_t const *beta,
                                                           nk_bf16_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                         nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                         void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                         nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                         void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i16_serial(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                          nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                          nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i32_serial(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                          nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                          nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i64_serial(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n,
                                                          nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n,
                                                          nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result,
                                                          void *stream);

/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c,
                                                        nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                        nk_f64_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                        nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                        nk_f32_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                        nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                        nk_f16_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_bf16_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c,
                                                       nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                       nk_i8_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c,
                                                       nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                       nk_u8_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i16_serial(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c,
                                                        nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                        nk_i16_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c,
                                                        nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                        nk_u16_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i32_serial(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c,
                                                        nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                        nk_i32_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c,
                                                        nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                        nk_u32_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i64_serial(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c,
                                                        nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                        nk_i64_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c,
                                                        nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                        nk_u64_t *result, void *stream);

/** @copydoc nk_each_sum_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                         nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_sum_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                         nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_scale_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e4m3_serial(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_scale_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e5m2_serial(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_blend_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                           nk_f32_t const *alpha, nk_f32_t const *beta,
                                                           nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_blend_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                           nk_f32_t const *alpha, nk_f32_t const *beta,
                                                           nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_fma_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_fma_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_e5m2_t *result, void *stream);

/** @copydoc nk_each_sum_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e2m3_serial(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                         nk_e2m3_t *result, void *stream);
/** @copydoc nk_each_sum_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e3m2_serial(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                         nk_e3m2_t *result, void *stream);
/** @copydoc nk_each_scale_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e2m3_serial(nk_e2m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_e2m3_t *result, void *stream);
/** @copydoc nk_each_scale_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e3m2_serial(nk_e3m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_e3m2_t *result, void *stream);
/** @copydoc nk_each_blend_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e2m3_serial(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                           nk_f32_t const *alpha, nk_f32_t const *beta,
                                                           nk_e2m3_t *result, void *stream);
/** @copydoc nk_each_blend_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e3m2_serial(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                           nk_f32_t const *alpha, nk_f32_t const *beta,
                                                           nk_e3m2_t *result, void *stream);
/** @copydoc nk_each_fma_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e2m3_serial(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_e2m3_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_e2m3_t *result, void *stream);
/** @copydoc nk_each_fma_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e3m2_serial(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_e3m2_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_e3m2_t *result, void *stream);

/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                         nk_f32c_t *result, void *stream);
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                         nk_f64c_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32c_serial(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                           nk_f32c_t const *beta, nk_f32c_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64c_serial(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                           nk_f64c_t const *beta, nk_f64c_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                           nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                           nk_f32c_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                           nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                           nk_f64c_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c,
                                                         nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                         nk_f32c_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c,
                                                         nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                         nk_f64c_t *result, void *stream);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_each_scale_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32_neon(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_f32_t *result, void *stream);
/** @copydoc nk_each_scale_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i16_neon(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_i16_t *result, void *stream);
/** @copydoc nk_each_scale_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u16_neon(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_u16_t *result, void *stream);
/** @copydoc nk_each_scale_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i32_neon(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_i32_t *result, void *stream);
/** @copydoc nk_each_scale_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u32_neon(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_u32_t *result, void *stream);
/** @copydoc nk_each_scale_i64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i64_neon(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_i64_t *result, void *stream);
/** @copydoc nk_each_scale_u64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u64_neon(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_u64_t *result, void *stream);

/** @copydoc nk_each_sum_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                      nk_f32_t *result, void *stream);
/** @copydoc nk_each_sum_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i16_neon(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                      nk_i16_t *result, void *stream);
/** @copydoc nk_each_sum_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                      nk_u16_t *result, void *stream);
/** @copydoc nk_each_sum_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i32_neon(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                      nk_i32_t *result, void *stream);
/** @copydoc nk_each_sum_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                      nk_u32_t *result, void *stream);
/** @copydoc nk_each_sum_i64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i64_neon(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n,
                                                      nk_i64_t *result, void *stream);
/** @copydoc nk_each_sum_u64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n,
                                                      nk_u64_t *result, void *stream);

/** @copydoc nk_each_blend_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result,
                                                        void *stream);

/** @copydoc nk_each_fma_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_f32_t *result, void *stream);
/** @copydoc nk_each_fma_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i16_neon(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_i16_t *result, void *stream);
/** @copydoc nk_each_fma_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_u16_t *result, void *stream);
/** @copydoc nk_each_fma_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i32_neon(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_i32_t *result, void *stream);
/** @copydoc nk_each_fma_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_u32_t *result, void *stream);
/** @copydoc nk_each_fma_i64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i64_neon(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_i64_t *result, void *stream);
/** @copydoc nk_each_fma_u64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_u64_t *result, void *stream);

/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                      nk_f64_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64_neon(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_f64_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                        nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result,
                                                        void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_f64_t *result, void *stream);

/** @copydoc nk_each_sum_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                       nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_sum_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                       nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_scale_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e4m3_neon(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                         nk_f32_t const *beta, nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_scale_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e5m2_neon(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                         nk_f32_t const *beta, nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_blend_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                         nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result,
                                                         void *stream);
/** @copydoc nk_each_blend_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                         nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result,
                                                         void *stream);
/** @copydoc nk_each_fma_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c,
                                                       nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                       nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_fma_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c,
                                                       nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                       nk_e5m2_t *result, void *stream);

/** @copydoc nk_each_scale_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f16_neon(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_f16_t *result, void *stream);
/** @copydoc nk_each_blend_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result,
                                                        void *stream);
/** @copydoc nk_each_fma_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_f16_t *result, void *stream);

/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32c_neon(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                         nk_f32c_t const *beta, nk_f32c_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64c_neon(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                         nk_f64c_t const *beta, nk_f64c_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                         nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                         nk_f32c_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                         nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                         nk_f64c_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c,
                                                       nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                       nk_f32c_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c,
                                                       nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                       nk_f64c_t *result, void *stream);

/** @copydoc nk_each_sum_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i8_neon(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result,
                                                     void *stream);
/** @copydoc nk_each_sum_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result,
                                                     void *stream);
/** @copydoc nk_each_scale_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i8_neon(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_i8_t *result, void *stream);
/** @copydoc nk_each_scale_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u8_neon(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_u8_t *result, void *stream);
/** @copydoc nk_each_blend_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i8_neon(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                       void *stream);
/** @copydoc nk_each_blend_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                       void *stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_each_sum_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                            nk_bf16_t *result, void *stream);
/** @copydoc nk_each_scale_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_bf16_neonbfdot(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                              nk_f32_t const *beta, nk_bf16_t *result, void *stream);
/** @copydoc nk_each_blend_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                              nk_f32_t const *alpha, nk_f32_t const *beta,
                                                              nk_bf16_t *result, void *stream);
/** @copydoc nk_each_fma_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                            nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                            nk_bf16_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONHALF
/** @copydoc nk_each_sum_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                          nk_f16_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONHALF

#if NUMKONG_TARGET_V128
/** @copydoc nk_each_sum_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f32_v128(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                      nk_f32_t *result, void *stream);
/** @copydoc nk_each_sum_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                       nk_bf16_t *result, void *stream);
/** @copydoc nk_each_sum_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result,
                                                     void *stream);
/** @copydoc nk_each_sum_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result,
                                                     void *stream);
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_each_scale_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32_v128relaxed(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                               nk_f32_t const *beta, nk_f32_t *result, void *stream);
/** @copydoc nk_each_blend_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                               nk_f32_t const *alpha, nk_f32_t const *beta,
                                                               nk_f32_t *result, void *stream);
/** @copydoc nk_each_fma_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                             nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                             nk_f32_t *result, void *stream);
/** @copydoc nk_each_sum_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                             nk_f16_t *result, void *stream);
/** @copydoc nk_each_scale_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f16_v128relaxed(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                               nk_f32_t const *beta, nk_f16_t *result, void *stream);
/** @copydoc nk_each_blend_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                               nk_f32_t const *alpha, nk_f32_t const *beta,
                                                               nk_f16_t *result, void *stream);
/** @copydoc nk_each_fma_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                             nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                             nk_f16_t *result, void *stream);
/** @copydoc nk_each_scale_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_bf16_v128relaxed(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                                nk_f32_t const *beta, nk_bf16_t *result, void *stream);
/** @copydoc nk_each_blend_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                                nk_f32_t const *alpha, nk_f32_t const *beta,
                                                                nk_bf16_t *result, void *stream);
/** @copydoc nk_each_fma_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b,
                                                              nk_bf16_t const *c, nk_size_t n, nk_f32_t const *alpha,
                                                              nk_f32_t const *beta, nk_bf16_t *result, void *stream);
/** @copydoc nk_each_scale_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i8_v128relaxed(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                              nk_f32_t const *beta, nk_i8_t *result, void *stream);
/** @copydoc nk_each_blend_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                              nk_f32_t const *alpha, nk_f32_t const *beta,
                                                              nk_i8_t *result, void *stream);
/** @copydoc nk_each_fma_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c,
                                                            nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                            nk_i8_t *result, void *stream);
/** @copydoc nk_each_scale_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u8_v128relaxed(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                              nk_f32_t const *beta, nk_u8_t *result, void *stream);
/** @copydoc nk_each_blend_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                              nk_f32_t const *alpha, nk_f32_t const *beta,
                                                              nk_u8_t *result, void *stream);
/** @copydoc nk_each_fma_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c,
                                                            nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                            nk_u8_t *result, void *stream);
#endif // NUMKONG_TARGET_V128RELAXED

/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_f32_serial(nk_f32_t const *, nk_f32_t const *, nk_f32_t *, nk_size_t,
                                                           nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_f32_t,
                                                           void *stream);
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_bf16_serial(nk_bf16_t const *, nk_bf16_t const *, nk_bf16_t *,
                                                            nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                            nk_f32_t, void *stream);
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_e4m3_serial(nk_e4m3_t const *, nk_e4m3_t const *, nk_e4m3_t *,
                                                            nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                            nk_f32_t, void *stream);

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_f32_haswell(nk_f32_t const *, nk_f32_t const *, nk_f32_t *, nk_size_t,
                                                            nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_f32_t,
                                                            void *stream);
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_bf16_haswell(nk_bf16_t const *, nk_bf16_t const *, nk_bf16_t *,
                                                             nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                             nk_f32_t, void *stream);
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_e4m3_haswell(nk_e4m3_t const *, nk_e4m3_t const *, nk_e4m3_t *,
                                                             nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                             nk_f32_t, void *stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_f32_skylake(nk_f32_t const *, nk_f32_t const *, nk_f32_t *, nk_size_t,
                                                            nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_f32_t,
                                                            void *stream);
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_bf16_skylake(nk_bf16_t const *, nk_bf16_t const *, nk_bf16_t *,
                                                             nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                             nk_f32_t, void *stream);
/** @copydoc nk_each_swiglu_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_e4m3_skylake(nk_e4m3_t const *, nk_e4m3_t const *, nk_e4m3_t *,
                                                             nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                             nk_f32_t, void *stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64_haswell(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                           nk_f64_t const *beta, nk_f64_t *result, void *stream);
/** @copydoc nk_each_scale_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32_haswell(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_f32_t *result, void *stream);
/** @copydoc nk_each_scale_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f16_haswell(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_f16_t *result, void *stream);
/** @copydoc nk_each_scale_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_bf16_haswell(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                            nk_f32_t const *beta, nk_bf16_t *result, void *stream);
/** @copydoc nk_each_scale_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i8_haswell(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                          nk_f32_t const *beta, nk_i8_t *result, void *stream);
/** @copydoc nk_each_scale_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u8_haswell(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                          nk_f32_t const *beta, nk_u8_t *result, void *stream);
/** @copydoc nk_each_scale_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i16_haswell(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_i16_t *result, void *stream);
/** @copydoc nk_each_scale_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u16_haswell(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_u16_t *result, void *stream);
/** @copydoc nk_each_scale_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i32_haswell(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                           nk_f64_t const *beta, nk_i32_t *result, void *stream);
/** @copydoc nk_each_scale_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u32_haswell(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                           nk_f64_t const *beta, nk_u32_t *result, void *stream);

/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                         nk_f64_t *result, void *stream);
/** @copydoc nk_each_sum_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_each_sum_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                         nk_f16_t *result, void *stream);
/** @copydoc nk_each_sum_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                          nk_bf16_t *result, void *stream);
/** @copydoc nk_each_sum_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                        nk_i8_t *result, void *stream);
/** @copydoc nk_each_sum_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                        nk_u8_t *result, void *stream);
/** @copydoc nk_each_sum_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i16_haswell(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                         nk_i16_t *result, void *stream);
/** @copydoc nk_each_sum_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                         nk_u16_t *result, void *stream);
/** @copydoc nk_each_sum_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i32_haswell(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                         nk_i32_t *result, void *stream);
/** @copydoc nk_each_sum_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                         nk_u32_t *result, void *stream);

/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                           nk_f64_t const *alpha, nk_f64_t const *beta,
                                                           nk_f64_t *result, void *stream);
/** @copydoc nk_each_blend_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                           nk_f32_t const *alpha, nk_f32_t const *beta,
                                                           nk_f32_t *result, void *stream);
/** @copydoc nk_each_blend_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                           nk_f32_t const *alpha, nk_f32_t const *beta,
                                                           nk_f16_t *result, void *stream);
/** @copydoc nk_each_blend_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                            nk_f32_t const *alpha, nk_f32_t const *beta,
                                                            nk_bf16_t *result, void *stream);
/** @copydoc nk_each_blend_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                          nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                          nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                          void *stream);

/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c,
                                                         nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                         nk_f64_t *result, void *stream);
/** @copydoc nk_each_fma_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_each_fma_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_f16_t *result, void *stream);
/** @copydoc nk_each_fma_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                          nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                          nk_bf16_t *result, void *stream);
/** @copydoc nk_each_fma_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c,
                                                        nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                        nk_i8_t *result, void *stream);
/** @copydoc nk_each_fma_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c,
                                                        nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                        nk_u8_t *result, void *stream);
/** @copydoc nk_each_fma_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i16_haswell(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_i16_t *result, void *stream);
/** @copydoc nk_each_fma_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_u16_t *result, void *stream);
/** @copydoc nk_each_fma_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i32_haswell(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c,
                                                         nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                         nk_i32_t *result, void *stream);
/** @copydoc nk_each_fma_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c,
                                                         nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                         nk_u32_t *result, void *stream);

/** @copydoc nk_each_sum_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                          nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_sum_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                          nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_scale_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e4m3_haswell(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                            nk_f32_t const *beta, nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_scale_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e5m2_haswell(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                            nk_f32_t const *beta, nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_blend_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                            nk_f32_t const *alpha, nk_f32_t const *beta,
                                                            nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_blend_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                            nk_f32_t const *alpha, nk_f32_t const *beta,
                                                            nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_fma_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c,
                                                          nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                          nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_fma_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c,
                                                          nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                          nk_e5m2_t *result, void *stream);

/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32c_haswell(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                            nk_f32c_t const *beta, nk_f32c_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64c_haswell(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                            nk_f64c_t const *beta, nk_f64c_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                            nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                            nk_f32c_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64c_haswell(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                            nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                            nk_f64c_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c,
                                                          nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                          nk_f32c_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64c_haswell(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c,
                                                          nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                          nk_f64c_t *result, void *stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64_skylake(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                           nk_f64_t const *beta, nk_f64_t *result, void *stream);
/** @copydoc nk_each_scale_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32_skylake(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_f32_t *result, void *stream);
/** @copydoc nk_each_scale_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f16_skylake(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_f16_t *result, void *stream);
/** @copydoc nk_each_scale_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_bf16_skylake(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                            nk_f32_t const *beta, nk_bf16_t *result, void *stream);
/** @copydoc nk_each_scale_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i8_skylake(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                          nk_f32_t const *beta, nk_i8_t *result, void *stream);
/** @copydoc nk_each_scale_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u8_skylake(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                          nk_f32_t const *beta, nk_u8_t *result, void *stream);
/** @copydoc nk_each_scale_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i16_skylake(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_i16_t *result, void *stream);
/** @copydoc nk_each_scale_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u16_skylake(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                           nk_f32_t const *beta, nk_u16_t *result, void *stream);
/** @copydoc nk_each_scale_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i32_skylake(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                           nk_f64_t const *beta, nk_i32_t *result, void *stream);
/** @copydoc nk_each_scale_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u32_skylake(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                           nk_f64_t const *beta, nk_u32_t *result, void *stream);
/** @copydoc nk_each_scale_i64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i64_skylake(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                           nk_f64_t const *beta, nk_i64_t *result, void *stream);
/** @copydoc nk_each_scale_u64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u64_skylake(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                           nk_f64_t const *beta, nk_u64_t *result, void *stream);

/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                         nk_f64_t *result, void *stream);
/** @copydoc nk_each_sum_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_each_sum_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                          nk_bf16_t *result, void *stream);

/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                           nk_f64_t const *alpha, nk_f64_t const *beta,
                                                           nk_f64_t *result, void *stream);
/** @copydoc nk_each_blend_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                           nk_f32_t const *alpha, nk_f32_t const *beta,
                                                           nk_f32_t *result, void *stream);
/** @copydoc nk_each_blend_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                           nk_f32_t const *alpha, nk_f32_t const *beta,
                                                           nk_f16_t *result, void *stream);
/** @copydoc nk_each_blend_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                            nk_f32_t const *alpha, nk_f32_t const *beta,
                                                            nk_bf16_t *result, void *stream);
/** @copydoc nk_each_blend_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i8_skylake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                          nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                          void *stream);
/** @copydoc nk_each_blend_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u8_skylake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                          nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                          void *stream);

/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c,
                                                         nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                         nk_f64_t *result, void *stream);
/** @copydoc nk_each_fma_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_each_fma_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_f16_t *result, void *stream);
/** @copydoc nk_each_fma_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                          nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                          nk_bf16_t *result, void *stream);
/** @copydoc nk_each_fma_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i8_skylake(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c,
                                                        nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                        nk_i8_t *result, void *stream);
/** @copydoc nk_each_fma_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u8_skylake(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c,
                                                        nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                        nk_u8_t *result, void *stream);
/** @copydoc nk_each_fma_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i16_skylake(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_i16_t *result, void *stream);
/** @copydoc nk_each_fma_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u16_skylake(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c,
                                                         nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                         nk_u16_t *result, void *stream);
/** @copydoc nk_each_fma_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i32_skylake(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c,
                                                         nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                         nk_i32_t *result, void *stream);
/** @copydoc nk_each_fma_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u32_skylake(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c,
                                                         nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                         nk_u32_t *result, void *stream);
/** @copydoc nk_each_fma_i64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i64_skylake(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c,
                                                         nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                         nk_i64_t *result, void *stream);
/** @copydoc nk_each_fma_u64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u64_skylake(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c,
                                                         nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                         nk_u64_t *result, void *stream);
/** @copydoc nk_each_sum_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                          nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_sum_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                          nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_scale_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e4m3_skylake(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                            nk_f32_t const *beta, nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_scale_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e5m2_skylake(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                            nk_f32_t const *beta, nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_blend_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                            nk_f32_t const *alpha, nk_f32_t const *beta,
                                                            nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_blend_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                            nk_f32_t const *alpha, nk_f32_t const *beta,
                                                            nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_fma_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c,
                                                          nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                          nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_fma_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c,
                                                          nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                          nk_e5m2_t *result, void *stream);

/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32c_skylake(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                            nk_f32c_t const *beta, nk_f32c_t *result, void *stream);
/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64c_skylake(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                            nk_f64c_t const *beta, nk_f64c_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                            nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                            nk_f32c_t *result, void *stream);
/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                            nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                            nk_f64c_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c,
                                                          nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                          nk_f32c_t *result, void *stream);
/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c,
                                                          nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                          nk_f64c_t *result, void *stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_each_sum_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i8_icelake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                        nk_i8_t *result, void *stream);
/** @copydoc nk_each_sum_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                        nk_u8_t *result, void *stream);
/** @copydoc nk_each_sum_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i16_icelake(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                         nk_i16_t *result, void *stream);
/** @copydoc nk_each_sum_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u16_icelake(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                         nk_u16_t *result, void *stream);
/** @copydoc nk_each_sum_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i32_icelake(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                         nk_i32_t *result, void *stream);
/** @copydoc nk_each_sum_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u32_icelake(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                         nk_u32_t *result, void *stream);
/** @copydoc nk_each_sum_i64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i64_icelake(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n,
                                                         nk_i64_t *result, void *stream);
/** @copydoc nk_each_sum_u64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u64_icelake(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n,
                                                         nk_u64_t *result, void *stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_SAPPHIRE
/** @copydoc nk_each_sum_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                          nk_f16_t *result, void *stream);
/** @copydoc nk_each_sum_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e4m3_sapphire(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                           nk_e4m3_t *result, void *stream);
#endif // NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_RVV
/** @copydoc nk_each_sum_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t *result, void *stream);
/** @copydoc nk_each_sum_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_each_sum_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f16_t *result, void *stream);
/** @copydoc nk_each_sum_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_bf16_t *result, void *stream);
/** @copydoc nk_each_sum_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result,
                                                    void *stream);
/** @copydoc nk_each_sum_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result,
                                                    void *stream);
/** @copydoc nk_each_sum_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i16_rvv(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                     nk_i16_t *result, void *stream);
/** @copydoc nk_each_sum_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u16_rvv(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                     nk_u16_t *result, void *stream);
/** @copydoc nk_each_sum_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i32_rvv(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                     nk_i32_t *result, void *stream);
/** @copydoc nk_each_sum_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u32_rvv(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                     nk_u32_t *result, void *stream);
/** @copydoc nk_each_sum_i64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i64_rvv(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n,
                                                     nk_i64_t *result, void *stream);
/** @copydoc nk_each_sum_u64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u64_rvv(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n,
                                                     nk_u64_t *result, void *stream);
/** @copydoc nk_each_sum_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                      nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_sum_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                      nk_e5m2_t *result, void *stream);

/** @copydoc nk_each_scale_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64_rvv(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_f64_t *result, void *stream);
/** @copydoc nk_each_scale_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32_rvv(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_f32_t *result, void *stream);
/** @copydoc nk_each_scale_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f16_rvv(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_f16_t *result, void *stream);
/** @copydoc nk_each_scale_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_bf16_rvv(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_bf16_t *result, void *stream);
/** @copydoc nk_each_scale_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i8_rvv(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                      nk_f32_t const *beta, nk_i8_t *result, void *stream);
/** @copydoc nk_each_scale_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u8_rvv(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                      nk_f32_t const *beta, nk_u8_t *result, void *stream);
/** @copydoc nk_each_scale_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i16_rvv(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_i16_t *result, void *stream);
/** @copydoc nk_each_scale_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u16_rvv(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_u16_t *result, void *stream);
/** @copydoc nk_each_scale_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i32_rvv(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_i32_t *result, void *stream);
/** @copydoc nk_each_scale_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u32_rvv(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_u32_t *result, void *stream);
/** @copydoc nk_each_scale_i64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i64_rvv(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_i64_t *result, void *stream);
/** @copydoc nk_each_scale_u64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u64_rvv(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                       nk_f64_t const *beta, nk_u64_t *result, void *stream);
/** @copydoc nk_each_scale_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e4m3_rvv(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_scale_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e5m2_rvv(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_e5m2_t *result, void *stream);

/** @copydoc nk_each_blend_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                       nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result,
                                                       void *stream);
/** @copydoc nk_each_blend_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result,
                                                       void *stream);
/** @copydoc nk_each_blend_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result,
                                                       void *stream);
/** @copydoc nk_each_blend_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result,
                                                        void *stream);
/** @copydoc nk_each_blend_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                      nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                      void *stream);
/** @copydoc nk_each_blend_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                      nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                      void *stream);
/** @copydoc nk_each_blend_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result,
                                                        void *stream);
/** @copydoc nk_each_blend_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result,
                                                        void *stream);

/** @copydoc nk_each_fma_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_f64_t *result, void *stream);
/** @copydoc nk_each_fma_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                     nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_each_fma_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                     nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                     nk_f16_t *result, void *stream);
/** @copydoc nk_each_fma_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_bf16_t *result, void *stream);
/** @copydoc nk_each_fma_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                    void *stream);
/** @copydoc nk_each_fma_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                    void *stream);
/** @copydoc nk_each_fma_i16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i16_rvv(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c,
                                                     nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                     nk_i16_t *result, void *stream);
/** @copydoc nk_each_fma_u16_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u16_rvv(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c,
                                                     nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                     nk_u16_t *result, void *stream);
/** @copydoc nk_each_fma_i32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i32_rvv(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_i32_t *result, void *stream);
/** @copydoc nk_each_fma_u32_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u32_rvv(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_u32_t *result, void *stream);
/** @copydoc nk_each_fma_i64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i64_rvv(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_i64_t *result, void *stream);
/** @copydoc nk_each_fma_u64_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u64_rvv(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c,
                                                     nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                     nk_u64_t *result, void *stream);
/** @copydoc nk_each_fma_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_e4m3_t *result, void *stream);
/** @copydoc nk_each_fma_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_e5m2_t *result, void *stream);
/** @copydoc nk_each_scale_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32c_rvv(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                        nk_f32c_t const *beta, nk_f32c_t *result, void *stream);
/** @copydoc nk_each_scale_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64c_rvv(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                        nk_f64c_t const *beta, nk_f64c_t *result, void *stream);
/** @copydoc nk_each_blend_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32c_rvv(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                        nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                        nk_f32c_t *result, void *stream);
/** @copydoc nk_each_blend_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64c_rvv(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                        nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                        nk_f64c_t *result, void *stream);
/** @copydoc nk_each_fma_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32c_rvv(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c,
                                                      nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                      nk_f32c_t *result, void *stream);
/** @copydoc nk_each_fma_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64c_rvv(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c,
                                                      nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                      nk_f64c_t *result, void *stream);
#endif // NUMKONG_TARGET_RVV

/** Returns the scalar parameter dtype for elementwise scale/blend/fma operations. */
NUMKONG_HELPER_INLINE nk_dtype_t nk_each_scale_input_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64c_k: return nk_f64c_k;
    case nk_f32c_k: return nk_f32c_k;
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_i64_k: return nk_f64_k;
    case nk_u64_k: return nk_f64_k;
    case nk_i32_k: return nk_f64_k;
    case nk_u32_k: return nk_f64_k;
    case nk_i16_k: return nk_f32_k;
    case nk_u16_k: return nk_f32_k;
    case nk_i8_k: return nk_f32_k;
    case nk_u8_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_e2m3_k: return nk_f32_k;
    case nk_e3m2_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the error bound of elementwise scale, blend and FMA before their results round into
 *  @p dtype: zero for integers, which match exactly, and else per @c nk_accumulation_error_bound
 *  of their coefficients. */
NUMKONG_HELPER_INLINE nk_f64_t nk_each_error_bound(nk_dtype_t dtype) {
    nk_dtype_family_t const family = nk_dtype_family(dtype);
    if (family == nk_dtype_family_int_k || family == nk_dtype_family_uint_k) return 0;
    return nk_accumulation_error_bound(nk_each_scale_input_dtype(dtype));
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/each/serial.h"
#include "numkong/each/neon.h"
#include "numkong/each/neonhalf.h"
#include "numkong/each/neonbfdot.h"
#include "numkong/each/sme.h"
#include "numkong/each/haswell.h"
#include "numkong/each/skylake.h"
#include "numkong/each/icelake.h"
#include "numkong/each/sapphire.h"
#include "numkong/each/rvv.h"
#include "numkong/each/v128.h"
#include "numkong/each/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_f64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_f32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_f32_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_f16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_bf16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_each_scale_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_bf16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_bf16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_i8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_i8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_i8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_i8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_i8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_i8_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_u8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_u8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_u8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_u8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_u8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_u8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_u8_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_i16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_i16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_i16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_i16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_i16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_i16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_u16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_u16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_u16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_u16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_u16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_i32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_i32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_i32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_i32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_i32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_i32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_u32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_u32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_u32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_u32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_u32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_i64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_i64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_i64_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_i64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_i64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_u64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_u64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_u64_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_u64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_u64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_e4m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_e4m3_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_e4m3_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_e5m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_e5m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_e5m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_e5m2_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_e2m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_e2m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_e3m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_e3m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_f32c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_f32c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f32c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f32c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f32c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f32c_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_scale_f64c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_scale_f64c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f64c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f64c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f64c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f64c_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_f64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_f64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_f32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_f32_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_each_sum_f32_v128,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_f16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_f16_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_each_sum_f16_neonhalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_f16_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_each_sum_f16_sapphire,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_sum_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_bf16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_each_sum_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_bf16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_each_sum_bf16_v128,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_i8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_i8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_i8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_i8_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_i8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_each_sum_i8_v128,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_u8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_u8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_u8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_u8_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_u8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_each_sum_u8_v128,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_i16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_i16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_i16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_i16_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_i16_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_i16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_u16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_u16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_u16_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_u16_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_u16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_i32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_i32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_i32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_i32_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_i32_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_i32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_u32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_u32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_u32_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_u32_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_u32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_i64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_i64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_i64_neon,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_i64_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_i64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_u64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_u64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_u64_neon,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_u64_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_u64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_e4m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_e4m3_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_each_sum_e4m3_sapphire,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_e4m3_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_e5m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_e5m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_e5m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_e5m2_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_e2m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_e2m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_e3m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_e3m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_f32c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_f32c_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_sum_f64c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_sum_f64c_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_f64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_f32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_f32_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_f16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_bf16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_each_blend_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_bf16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_bf16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_i8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_i8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_i8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_i8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_i8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_i8_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_u8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_u8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_u8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_u8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_u8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_u8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_u8_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_i16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_i16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_u16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_u16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_i32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_i32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_u32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_u32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_i64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_i64_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_u64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_u64_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_e4m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_e4m3_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_e4m3_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_e5m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_e5m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_e5m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_e5m2_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_e2m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_e2m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_e3m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_e3m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_f32c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_f32c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f32c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f32c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f32c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f32c_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_blend_f64c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_blend_f64c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f64c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f64c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f64c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f64c_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_f64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_f32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_f32_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_f16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_bf16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_each_fma_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_bf16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_bf16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_i8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_i8_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_i8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_i8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_i8_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_u8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_u8_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_u8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_u8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_u8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_u8_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_i16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_i16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_i16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_i16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_i16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_i16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_u16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_u16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_u16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_u16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_u16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_i32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_i32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_i32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_i32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_i32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_i32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_u32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_u32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_u32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_u32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_u32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_i64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_i64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_i64_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_i64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_i64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_u64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_u64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_u64_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_u64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_u64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_e4m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_e4m3_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_e4m3_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_e5m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_e5m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_e5m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_e5m2_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_e2m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_e2m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_e3m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_e3m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_f32c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_f32c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f32c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f32c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f32c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f32c_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_fma_f64c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_fma_f64c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f64c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f64c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f64c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f64c_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_swiglu_f32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_swiglu_f32_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_swiglu_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_swiglu_f32_skylake,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_swiglu_bf16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_swiglu_bf16_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_swiglu_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_swiglu_bf16_skylake,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_each_swiglu_e4m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_each_swiglu_e4m3_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_swiglu_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_swiglu_e4m3_skylake,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

#if !NUMKONG_RUNTIME_DISPATCH

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64_best(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_f64_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f64_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32_best(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_f32_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f32_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f16_best(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_f16_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f16_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_bf16_best(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                         nk_f32_t const *beta, nk_bf16_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_bf16_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i8_best(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_i8_t *result,
                                                       nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_i8_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u8_best(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                       nk_f32_t const *beta, nk_u8_t *result,
                                                       nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_u8_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i16_best(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_i16_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_i16_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u16_best(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                        nk_f32_t const *beta, nk_u16_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_u16_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i32_best(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_i32_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_i32_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u32_best(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_u32_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_u32_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_i64_best(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_i64_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_i64_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_u64_best(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                                        nk_f64_t const *beta, nk_u64_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_u64_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e4m3_best(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                         nk_f32_t const *beta, nk_e4m3_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_e4m3_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e5m2_best(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                         nk_f32_t const *beta, nk_e5m2_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_e5m2_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e2m3_best(nk_e2m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                         nk_f32_t const *beta, nk_e2m3_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_e2m3_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_e3m2_best(nk_e3m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                         nk_f32_t const *beta, nk_e3m2_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_e3m2_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f32c_best(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                         nk_f32c_t const *beta, nk_f32c_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f32c_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_scale_f64c_best(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                         nk_f64c_t const *beta, nk_f64c_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f64c_capabilities_());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                      nk_f64_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f64_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                      nk_f32_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f32_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                      nk_f16_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f16_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                       nk_bf16_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_bf16_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result,
                                                     nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_i8_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result,
                                                     nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_u8_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i16_best(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                      nk_i16_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_i16_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                      nk_u16_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_u16_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i32_best(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                      nk_i32_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_i32_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                      nk_u32_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_u32_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_i64_best(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n,
                                                      nk_i64_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_i64_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n,
                                                      nk_u64_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_u64_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                       nk_e4m3_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_e4m3_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                       nk_e5m2_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_e5m2_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                       nk_e2m3_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_e2m3_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                       nk_e3m2_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_e3m2_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                       nk_f32c_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f32c_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_sum_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                       nk_f64c_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f64c_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                        nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f64_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f32_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f16_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                         nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_bf16_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                       nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_i8_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                       nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_u8_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i16_best(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_i16_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n,
                                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_u16_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i32_best(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n,
                                                        nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_i32_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n,
                                                        nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_u32_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_i64_best(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n,
                                                        nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_i64_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n,
                                                        nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result,
                                                        nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_u64_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                         nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_e4m3_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                         nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_e5m2_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                         nk_f32_t const *alpha, nk_f32_t const *beta, nk_e2m3_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_e2m3_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                         nk_f32_t const *alpha, nk_f32_t const *beta, nk_e3m2_t *result,
                                                         nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_e3m2_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                         nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                         nk_f32c_t *result, nk_capability_t capabilities,
                                                         void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f32c_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_blend_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                         nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                         nk_f64c_t *result, nk_capability_t capabilities,
                                                         void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f64c_capabilities_());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_f64_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f64_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_f32_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f32_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_f16_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f16_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                       nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                       nk_bf16_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_bf16_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                                     nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                                     nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_i8_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                                     nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                                     nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_u8_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i16_best(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_i16_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_i16_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c,
                                                      nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                      nk_u16_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_u16_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i32_best(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_i32_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_i32_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_u32_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_u32_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_i64_best(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_i64_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_i64_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c,
                                                      nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                                      nk_u64_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_u64_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c,
                                                       nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                       nk_e4m3_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_e4m3_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c,
                                                       nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                       nk_e5m2_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_e5m2_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_e2m3_t const *c,
                                                       nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                       nk_e2m3_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_e2m3_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_e3m2_t const *c,
                                                       nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                       nk_e3m2_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_e3m2_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c,
                                                       nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                                       nk_f32c_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f32c_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_fma_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c,
                                                       nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                                       nk_f64c_t *result, nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f64c_capabilities_());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_f32_best(nk_f32_t const *gate, nk_f32_t const *up, nk_f32_t *y,
                                                         nk_size_t rows, nk_size_t cols, nk_size_t gate_row_stride,
                                                         nk_size_t up_row_stride, nk_size_t y_row_stride,
                                                         nk_f32_t input_scale, nk_capability_t capabilities,
                                                         void *stream) {
    nk_each_swiglu_punned_t const kernel = (nk_each_swiglu_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_each_swiglu_f32_capabilities_());
    return kernel ? kernel(gate, up, y, rows, cols, gate_row_stride, up_row_stride, y_row_stride, input_scale, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_bf16_best(nk_bf16_t const *gate, nk_bf16_t const *up, nk_bf16_t *y,
                                                          nk_size_t rows, nk_size_t cols, nk_size_t gate_row_stride,
                                                          nk_size_t up_row_stride, nk_size_t y_row_stride,
                                                          nk_f32_t input_scale, nk_capability_t capabilities,
                                                          void *stream) {
    nk_each_swiglu_punned_t const kernel = (nk_each_swiglu_punned_t)nk_kernel_pick_(
        capabilities, nk_each_swiglu_bf16_capabilities_());
    return kernel ? kernel(gate, up, y, rows, cols, gate_row_stride, up_row_stride, y_row_stride, input_scale, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_each_swiglu_e4m3_best(nk_e4m3_t const *gate, nk_e4m3_t const *up, nk_e4m3_t *y,
                                                          nk_size_t rows, nk_size_t cols, nk_size_t gate_row_stride,
                                                          nk_size_t up_row_stride, nk_size_t y_row_stride,
                                                          nk_f32_t input_scale, nk_capability_t capabilities,
                                                          void *stream) {
    nk_each_swiglu_punned_t const kernel = (nk_each_swiglu_punned_t)nk_kernel_pick_(
        capabilities, nk_each_swiglu_e4m3_capabilities_());
    return kernel ? kernel(gate, up, y, rows, cols, gate_row_stride, up_row_stride, y_row_stride, input_scale, stream)
                  : nk_missing_kernel_k;
}

#endif // !NUMKONG_RUNTIME_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_EACH_H
