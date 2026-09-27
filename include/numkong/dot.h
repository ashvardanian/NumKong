/**
 *  @file include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief SIMD-accelerated dot products for real and complex numbers.
 *
 *  Contains:
 *
 *  - Dot Product for Real and Complex vectors
 *  - Conjugate Dot Product for Complex vectors
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
 *  - e2m1: 4-bit e2m1 floating point numbers (packed pairs) → 32-bit floats
 *  - i8: 8-bit signed integers → 32-bit signed integers
 *  - u8: 8-bit unsigned integers → 32-bit unsigned integers
 *  - i4: 4-bit signed integers (packed nibble pairs) → 32-bit signed integers
 *  - u4: 4-bit unsigned integers (packed nibble pairs) → 32-bit unsigned integers
 *  - u1: 1-bit binary (packed octets) → 32-bit unsigned integers
 *
 *  Complex dot product variants:
 *
 *  - f64c: 64-bit complex pairs → 64-bit complex
 *  - f32c: 32-bit complex pairs → 64-bit complex
 *  - f16c: 16-bit complex pairs → 32-bit complex
 *  - bf16c: 16-bit brain complex pairs → 32-bit complex
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, NEON+I8, NEON+F16, NEON+FHM, NEON+BF16, SVE, SVE+F16
 *  - x86: Haswell, Skylake, Ice Lake, Genoa, Sapphire Rapids, Sierra Forest
 *  - RISC-V: RVV, RVV+BF16, RVV+HALF, RVV+BB
 *  - WASM: V128, V128Relaxed
 *
 *  @section dot_numerical_stability Numerical Stability
 *
 *  - f64: Dot2/Ogita-Rump-Oishi style compensated summation across serial and SIMD stateful paths.
 *  - f32: public outputs widen to f64/f64c. Arithmetic widens before the first lossy reduction
 *    step.
 *  - f16/bf16: Promoted to f32 accumulator.
 *  - e4m3/e5m2: Promoted to f32.
 *  - i8: i32 accumulator. Max product |(-128)²| = 16,384. Overflows at n > 2^31/16,384 ≈ 131K.
 *  - u8: u32 accumulator. Max product 255² = 65,025. Overflows at n > 2^32/65,025 ≈ 66K.
 *  - i4: i32 accumulator. Max product 8² = 64. Safe for n ≤ ~33M.
 *  - u4: u32 accumulator. Max product 15² = 225. Safe for n ≤ ~19M.
 *  - u1: Popcount of AND into u32. Safe for n_bits ≤ 2^32.
 *  - Complex: Components accumulated independently; same guarantees as real counterpart.
 *
 *  @section dot_streaming_api Streaming API
 *
 *  For compile-time dispatch and vector-at-a-time accumulation, we provide streaming helpers that
 *  accept two @c nk_b512_vec_t blocks and update a running sum for non-complex dot products. The
 *  `<count>` suffix reflects how many scalars of that type fit in a 512-bit block. The helpers are
 *  exposed per scalar type as:
 *
 *  - nk_dot_<type>x<count>_state_<isa>_t
 *  - nk_dot_<type>x<count>_init_<isa>
 *  - nk_dot_<type>x<count>_update_<isa>
 *  - nk_dot_<type>x<count>_finalize_<isa>
 *
 *  @section dot_x86_instructions Relevant x86 Instructions
 *
 *  Floating-point dot products use FMA, VFMADD231PS/PD, for sum += aᵢ × bᵢ accumulation. Integer i8
 *  dot products use VPMADDUBSW, u8 × i8 → i16, plus VPMADDWD, i16 × 1 → i32, on Haswell, or the
 *  newer VNNI instructions VPDPBUSD/VPDPWSSD on Ice Lake+ for direct u8 × i8 → i32. BF16 dot
 *  products via VDPBF16PS are Genoa-only, accumulating bf16 pairs directly to f32. Genoa shows 40%
 *  faster integer multiply-add than Ice Lake, 3c vs 5c.
 *
 *  @verbatim
 *  Intrinsic             Instruction                  Haswell    Icelake    Genoa
 *  _mm256_fmadd_ps       VFMADD231PS (YMM, YMM, YMM)  5cy @ p01  4cy @ p01  4cy @ p01
 *  _mm256_fmadd_pd       VFMADD231PD (YMM, YMM, YMM)  5cy @ p01  4cy @ p01  4cy @ p01
 *  _mm256_maddubs_epi16  VPMADDUBSW (YMM, YMM, YMM)   5cy @ p0   5cy @ p01  3cy @ p01
 *  _mm256_madd_epi16     VPMADDWD (YMM, YMM, YMM)     5cy @ p0   5cy @ p01  3cy @ p01
 *  _mm256_dpbusd_epi32   VPDPBUSD (YMM, YMM, YMM)     n/a        5cy @ p01  4cy @ p01
 *  _mm512_dpwssd_epi32   VPDPWSSD (ZMM, ZMM, ZMM)     n/a        5cy @ p0   4cy @ p01
 *  _mm512_dpbf16_ps      VDPBF16PS (ZMM, ZMM, ZMM)    n/a        n/a        6cy @ p01
 *  @endverbatim
 *
 *  @section arm_neon_instructions Relevant ARM NEON Instructions
 *
 *  NEON integer dot products use SDOT/UDOT (ARMv8.2 dotprod) for direct i8 × i8 → i32 or u8 × u8 →
 *  u32 accumulation - 4x faster than the multiply-add sequence on older cores. BFDOT (ARMv8.6 bf16)
 *  provides native bf16 dot products on Graviton 3+. Complex dot products use LD2 for deinterleaved
 *  loads of real/imag pairs, though its L01+V throughput can bottleneck on memory-bound workloads.
 *
 *  @verbatim
 *  Intrinsic    Instruction   M1 Firestorm  Graviton 3   Graviton 4
 *  vfmaq_f32    FMLA.S (vec)  4cy @ V0123   4cy @ V0123  4cy @ V0123
 *  vfmaq_f64    FMLA.D (vec)  4cy @ V0123   4cy @ V0123  4cy @ V0123
 *  vdotq_s32    SDOT (vec)    3cy @ V0123   3cy @ V0123  3cy @ V0123
 *  vdotq_u32    UDOT (vec)    3cy @ V0123   3cy @ V0123  3cy @ V0123
 *  vbfdotq_f32  BFDOT (vec)   N/A           4cy @ V0123  5cy @ V0123
 *  vld2q_f32    LD2 (Q-form)  5cy @ L01+V   8cy @ L01+V  8cy @ L01+V
 *  @endverbatim
 *
 *  @section arm_sve_instructions Relevant ARM SVE Instructions
 *
 *  SVE implementations use predicated FMA (svmla_f32_x) with WHILELT for tail masking, avoiding
 *  scalar cleanup loops. FADDV performs horizontal reduction; notably 45% faster on Graviton 4 (6c)
 *  than Graviton 3 (11c). SVE complex dot products use svld2 for structure loads.
 *
 *  @verbatim
 *  Intrinsic      Instruction  Graviton 3    Graviton 4
 *  svmla_f32_x    FMLA (pred)  4cy @ V0123   4cy @ V0123
 *  svmls_f32_x    FMLS (pred)  4cy @ V0123   4cy @ V0123
 *  svwhilelt_b32  WHILELT      3cy @ M0      3cy @ M0
 *  svld2_f32      LD2 (SVE)    8cy @ L01+V   8cy @ L01+V
 *  svaddv_f32     FADDV        11cy @ V0123  6cy @ V0123
 *  @endverbatim
 *
 *  @section complex_instructions Complex Number Optimizations
 *
 *  Standard complex multiplication subtracts for the real part. Instead of subtracting FMA variants
 *  for every element, we accumulate real and imaginary products positively and flip the sign bits
 *  with a single bitwise XOR before the final horizontal reduction. Delaying the sign flip this way
 *  doubles the throughput on older x86 architectures like Haswell by maximizing FMA unit
 *  utilization and shortening execution dependency chains.
 *
 *  @section dot_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NUMKONG_DOT_H
#define NUMKONG_DOT_H

#include "numkong/capabilities.h" // `nk_capability_kernels_t`, `nk_kernel_pick_`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Dot product computing the sum of elementwise products between two vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n Counts dimensions, a multiple of the values per byte.
 *  @param[out] result The output dot product value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output value can be negative.
 *  @note The output value is zero if and only if the two vectors are orthogonal.
 *  @note Defined for floating-point, integer, and binary data types.
 */
NUMKONG_API_RUNTIME nk_status_t nk_dot_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                               nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result,
                                               nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_u1_best(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                               nk_u32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_e2m1_best(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                 nk_f32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, void *stream);

/**
 *  @brief Complex dot product computing the sum of elementwise products between
 *      two complex vectors.
 *
 *  @param[in] a_pairs The first complex vector.
 *  @param[in] b_pairs The second complex vector.
 *  @param[in] count_pairs The number of complex pairs in the vectors.
 *  @param[out] result The output complex value as {real, imag}.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API_RUNTIME nk_status_t nk_dot_f32c_best(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs,
                                                 nk_size_t count_pairs, nk_f64c_t *result, nk_capability_t capabilities,
                                                 void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_f64c_best(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs,
                                                 nk_size_t count_pairs, nk_f64c_t *result, nk_capability_t capabilities,
                                                 void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_f16c_best(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs,
                                                 nk_size_t count_pairs, nk_f32c_t *result, nk_capability_t capabilities,
                                                 void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_RUNTIME nk_status_t nk_dot_bf16c_best(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs,
                                                  nk_size_t count_pairs, nk_f32c_t *result,
                                                  nk_capability_t capabilities, void *stream);

/**
 *  @brief Complex conjugate dot product between two complex vectors.
 *
 *  @param[in] a_pairs The first complex vector.
 *  @param[in] b_pairs The second complex vector.
 *  @param[in] count_pairs The number of complex pairs in the vectors.
 *  @param[out] result The output complex value as {real, imag}.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API_RUNTIME nk_status_t nk_vdot_f32c_best(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs,
                                                  nk_size_t count_pairs, nk_f64c_t *result,
                                                  nk_capability_t capabilities, void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_RUNTIME nk_status_t nk_vdot_f64c_best(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs,
                                                  nk_size_t count_pairs, nk_f64c_t *result,
                                                  nk_capability_t capabilities, void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_RUNTIME nk_status_t nk_vdot_f16c_best(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs,
                                                  nk_size_t count_pairs, nk_f32c_t *result,
                                                  nk_capability_t capabilities, void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_RUNTIME nk_status_t nk_vdot_bf16c_best(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs,
                                                   nk_size_t count_pairs, nk_f32c_t *result,
                                                   nk_capability_t capabilities, void *stream);

/** @copydoc nk_dot_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                   void *stream);
/** @copydoc nk_dot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                    nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                     nk_f64c_t *result, void *stream);

/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                   void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                    nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                     nk_f64c_t *result, void *stream);

/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                   void *stream);
/** @copydoc nk_dot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16c_serial(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                    nk_f32c_t *result, void *stream);
/** @copydoc nk_vdot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f16c_serial(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                     nk_f32c_t *result, void *stream);

/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                    nk_f32_t *result, void *stream);
/** @copydoc nk_dot_bf16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16c_serial(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n,
                                                     nk_f32c_t *result, void *stream);
/** @copydoc nk_vdot_bf16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_bf16c_serial(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n,
                                                      nk_f32c_t *result, void *stream);

/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_i4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_u4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_u1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                                  nk_u32_t *result, void *stream);

/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_serial(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m1_serial(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_serial(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, void *stream);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                 void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                  nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                   nk_f64c_t *result, void *stream);

/** @copydoc nk_dot_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                 void *stream);
/** @copydoc nk_dot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                  nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                   nk_f64c_t *result, void *stream);

/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  void *stream);

/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  void *stream);

/** @copydoc nk_dot_u1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                                nk_u32_t *result, void *stream);

/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream);

/** @copydoc nk_dot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16c_neon(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                  nk_f32c_t *result, void *stream);
/** @copydoc nk_vdot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f16c_neon(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                   nk_f32c_t *result, void *stream);

#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONFHM
/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_neonfhm(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_neonfhm(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16c_neonfhm(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                     nk_f32c_t *result, void *stream);
/** @copydoc nk_vdot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f16c_neonfhm(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                      nk_f32c_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONSDOT
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_i4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i4_neonsdot(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n,
                                                    nk_i32_t *result, void *stream);
/** @copydoc nk_dot_u4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u4_neonsdot(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n,
                                                    nk_u32_t *result, void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_neonsdot(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                      nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m1_neonsdot(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                      nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_neonsdot(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                      nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_SVESDOT
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_svesdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                   void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_svesdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   void *stream);
#endif // NUMKONG_TARGET_SVESDOT

#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                       nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_neonbfdot(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                       nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_neonbfdot(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                       nk_f32_t *result, void *stream);
/** @copydoc nk_dot_bf16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n,
                                                        nk_f32c_t *result, void *stream);
/** @copydoc nk_vdot_bf16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n,
                                                         nk_f32c_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONFP8
/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_neonfp8(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_neonfp8(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_neonfp8(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_neonfp8(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_SVE
/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32c_sve(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result,
                                                 void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f32c_sve(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                  nk_f64c_t *result, void *stream);
/** @copydoc nk_dot_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                void *stream);
/** @copydoc nk_dot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64c_sve(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result,
                                                 void *stream);
/** @copydoc nk_vdot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f64c_sve(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                  nk_f64c_t *result, void *stream);
#endif // NUMKONG_TARGET_SVE

#if NUMKONG_TARGET_SVEHALF
/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16c_svehalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                     nk_f32c_t *result, void *stream);
/** @copydoc nk_vdot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f16c_svehalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                      nk_f32c_t *result, void *stream);
#endif // NUMKONG_TARGET_SVEHALF

#if NUMKONG_TARGET_SVEBFDOT
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_SVEBFDOT
#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                    void *stream);
/** @copydoc nk_dot_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                    void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                     nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                      nk_f64c_t *result, void *stream);
/** @copydoc nk_dot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64c_haswell(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                     nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f64c_haswell(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                      nk_f64c_t *result, void *stream);

/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16c_haswell(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                     nk_f32c_t *result, void *stream);
/** @copydoc nk_vdot_f16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f16c_haswell(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n,
                                                      nk_f32c_t *result, void *stream);

/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_bf16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16c_haswell(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n,
                                                      nk_f32c_t *result, void *stream);
/** @copydoc nk_vdot_bf16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_bf16c_haswell(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n,
                                                       nk_f32c_t *result, void *stream);

/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_haswell(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m1_haswell(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_haswell(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);

/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                   void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   void *stream);
/** @copydoc nk_dot_i4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i4_haswell(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n,
                                                   nk_i32_t *result, void *stream);
/** @copydoc nk_dot_u4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u4_haswell(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n,
                                                   nk_u32_t *result, void *stream);
/** @copydoc nk_dot_u1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                                   nk_u32_t *result, void *stream);

#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_dot_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                    void *stream);
/** @copydoc nk_dot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                     nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                      nk_f64c_t *result, void *stream);

/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                    void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                     nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                      nk_f64c_t *result, void *stream);

/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);

/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m1_skylake(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);

/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_skylake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                   void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_skylake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   void *stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_icelake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                   void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   void *stream);
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i4_icelake(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n,
                                                   nk_i32_t *result, void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u4_icelake(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n,
                                                   nk_u32_t *result, void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_icelake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_icelake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_u1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u1_icelake(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                                   nk_u32_t *result, void *stream);
/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_icelake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                   nk_f32_t *result, void *stream);
/** @copydoc nk_dot_bf16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n,
                                                    nk_f32c_t *result, void *stream);
/** @copydoc nk_vdot_bf16c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n,
                                                     nk_f32c_t *result, void *stream);

/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_genoa(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                   nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_DIAMOND
/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_diamond(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_diamond(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_diamond(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_DIAMOND

#if NUMKONG_TARGET_ALDER
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_alder(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                 void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_alder(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                 void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_alder(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                   nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m1_alder(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                   nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_SIERRA
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_sierra(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_sierra(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m1_sierra(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_SIERRA

#if NUMKONG_TARGET_RVV
/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                void *stream);
/** @copydoc nk_dot_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                void *stream);
/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                void *stream);
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream);
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                               void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                               void *stream);
/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_rvv(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream);
/** @copydoc nk_dot_e2m1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m1_rvv(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                 nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_rvv(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream);
/** @copydoc nk_dot_i4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i4_rvv(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result,
                                               void *stream);
/** @copydoc nk_dot_u4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u4_rvv(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                               void *stream);
/** @copydoc nk_dot_u1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u1_rvv(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                               nk_u32_t *result, void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32c_rvv(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result,
                                                 void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f32c_rvv(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                  nk_f64c_t *result, void *stream);
/** @copydoc nk_dot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64c_rvv(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result,
                                                 void *stream);
/** @copydoc nk_vdot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f64c_rvv(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                  nk_f64c_t *result, void *stream);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_RVVHALF
/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_rvvhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_rvvhalf(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_rvvhalf(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_RVVHALF

#if NUMKONG_TARGET_RVVBF16
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_rvvbf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e4m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_rvvbf16(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e5m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_rvvbf16(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_RVVBF16

#if NUMKONG_TARGET_RVVBB
/** @copydoc nk_dot_u1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                                 nk_u32_t *result, void *stream);
#endif // NUMKONG_TARGET_RVVBB

#if NUMKONG_TARGET_V128
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  void *stream);
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                void *stream);
/** @copydoc nk_dot_u1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u1_v128(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                                nk_u32_t *result, void *stream);
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                        nk_f64_t *result, void *stream);
/** @copydoc nk_dot_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                        nk_f64_t *result, void *stream);
/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                        nk_f32_t *result, void *stream);
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                       nk_i32_t *result, void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                       nk_u32_t *result, void *stream);
/** @copydoc nk_dot_e2m3_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_v128relaxed(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e2m1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m1_v128relaxed(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_dot_e3m2_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_v128relaxed(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_v128relaxed(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_v128relaxed(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_dot_i4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i4_v128relaxed(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n,
                                                       nk_i32_t *result, void *stream);
/** @copydoc nk_dot_u4_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u4_v128relaxed(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n,
                                                       nk_u32_t *result, void *stream);
/** @copydoc nk_dot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32c_v128relaxed(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                         nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f32c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f32c_v128relaxed(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                          nk_f64c_t *result, void *stream);
/** @copydoc nk_dot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64c_v128relaxed(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                         nk_f64c_t *result, void *stream);
/** @copydoc nk_vdot_f64c_best */
NUMKONG_API_COMPTIME nk_status_t nk_vdot_f64c_v128relaxed(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                          nk_f64c_t *result, void *stream);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_POWERVSX
/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f64_t *result, void *stream);
/** @copydoc nk_dot_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t *result, void *stream);
/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_f32_t *result, void *stream);
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                    void *stream);
/** @copydoc nk_dot_u1_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u1_powervsx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                                    nk_u32_t *result, void *stream);
#endif // NUMKONG_TARGET_POWERVSX

#if NUMKONG_TARGET_LOONGSONASX
/** @copydoc nk_dot_f32_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_loongsonasx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                        nk_f64_t *result, void *stream);
/** @copydoc nk_dot_f64_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_loongsonasx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                        nk_f64_t *result, void *stream);
/** @copydoc nk_dot_f16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_loongsonasx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                        nk_f32_t *result, void *stream);
/** @copydoc nk_dot_bf16_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_loongsonasx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                         nk_f32_t *result, void *stream);
/** @copydoc nk_dot_i8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_loongsonasx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                       nk_i32_t *result, void *stream);
/** @copydoc nk_dot_u8_best */
NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                       nk_u32_t *result, void *stream);
#endif // NUMKONG_TARGET_LOONGSONASX

/** Returns the output dtype for dot products. */
NUMKONG_HELPER_INLINE nk_dtype_t nk_dot_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_e2m3_k: return nk_f32_k;
    case nk_e2m1_k: return nk_f32_k;
    case nk_e3m2_k: return nk_f32_k;
    case nk_f64c_k: return nk_f64c_k;
    case nk_f32c_k: return nk_f64c_k;
    case nk_f16c_k: return nk_f32c_k;
    case nk_bf16c_k: return nk_f32c_k;
    case nk_i8_k: return nk_i32_k;
    case nk_u8_k: return nk_u32_k;
    case nk_i4_k: return nk_i32_k;
    case nk_u4_k: return nk_u32_k;
    case nk_u1_k: return nk_u32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the error bound of dot products, per @c nk_accumulation_error_bound of their output. */
NUMKONG_HELPER_INLINE nk_f64_t nk_dot_error_bound(nk_dtype_t dtype) {
    return nk_accumulation_error_bound(nk_dot_output_dtype(dtype));
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/dot/serial.h"
#include "numkong/dot/neon.h"
#include "numkong/dot/neonsdot.h"
#include "numkong/dot/neonfhm.h"
#include "numkong/dot/neonbfdot.h"
#include "numkong/dot/neonfp8.h"
#include "numkong/dot/sve.h"
#include "numkong/dot/svehalf.h"
#include "numkong/dot/svebfdot.h"
#include "numkong/dot/svesdot.h"
#include "numkong/dot/haswell.h"
#include "numkong/dot/skylake.h"
#include "numkong/dot/icelake.h"
#include "numkong/dot/genoa.h"
#include "numkong/dot/diamond.h"
#include "numkong/dot/alder.h"
#include "numkong/dot/sierra.h"
#include "numkong/dot/rvv.h"
#include "numkong/dot/rvvbb.h"
#include "numkong/dot/rvvhalf.h"
#include "numkong/dot/rvvbf16.h"
#include "numkong/dot/powervsx.h"
#include "numkong/dot/v128.h"
#include "numkong/dot/v128relaxed.h"
#include "numkong/dot/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_f32_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_f32_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_dot_f32_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_f32_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_dot_f32_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_dot_f32_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_f64_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_f64_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_dot_f64_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_f64_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_dot_f64_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_dot_f64_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_f16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_dot_f16_neonfhm,
#endif
#if NUMKONG_TARGET_SVEHALF
        (nk_kernel_punned_t)&nk_dot_f16_svehalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_f16_skylake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_dot_f16_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_f16_rvv,
#endif
#if NUMKONG_TARGET_RVVHALF
        (nk_kernel_punned_t)&nk_dot_f16_rvvhalf,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_f16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_dot_f16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_dot_f16_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_rvvhalf_k * NUMKONG_TARGET_RVVHALF |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_bf16_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_bf16_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_dot_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SVEBFDOT
        (nk_kernel_punned_t)&nk_dot_bf16_svebfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_dot_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_bf16_rvv,
#endif
#if NUMKONG_TARGET_RVVBF16
        (nk_kernel_punned_t)&nk_dot_bf16_rvvbf16,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_dot_bf16_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_bf16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_dot_bf16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_dot_bf16_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_svebfdot_k * NUMKONG_TARGET_SVEBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_rvvbf16_k * NUMKONG_TARGET_RVVBF16 |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_i8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_dot_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SVESDOT
        (nk_kernel_punned_t)&nk_dot_i8_svesdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_i8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_dot_i8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_dot_i8_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_i8_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_dot_i8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_dot_i8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_i8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_dot_i8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_dot_i8_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_svesdot_k * NUMKONG_TARGET_SVESDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_u8_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_u8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_dot_u8_neonsdot,
#endif
#if NUMKONG_TARGET_SVESDOT
        (nk_kernel_punned_t)&nk_dot_u8_svesdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_u8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_dot_u8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_dot_u8_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_u8_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_dot_u8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_dot_u8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_u8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_dot_u8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_dot_u8_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_svesdot_k * NUMKONG_TARGET_SVESDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_i4_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_i4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_dot_i4_neonsdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_i4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_dot_i4_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_i4_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_i4_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_u4_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_u4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_dot_u4_neonsdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_u4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_dot_u4_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_u4_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_u4_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_u1_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_u1_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_u1_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_u1_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_dot_u1_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_u1_rvv,
#endif
#if NUMKONG_TARGET_RVVBB
        (nk_kernel_punned_t)&nk_dot_u1_rvvbb,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_dot_u1_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_dot_u1_powervsx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_rvvbb_k * NUMKONG_TARGET_RVVBB | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_e4m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_e4m3_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_dot_e4m3_neonbfdot,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_dot_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_dot_e4m3_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_e4m3_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_dot_e4m3_icelake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_dot_e4m3_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_e4m3_rvv,
#endif
#if NUMKONG_TARGET_RVVBF16
        (nk_kernel_punned_t)&nk_dot_e4m3_rvvbf16,
#endif
#if NUMKONG_TARGET_RVVHALF
        (nk_kernel_punned_t)&nk_dot_e4m3_rvvhalf,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_e4m3_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_rvvbf16_k * NUMKONG_TARGET_RVVBF16 |
             nk_cap_rvvhalf_k * NUMKONG_TARGET_RVVHALF | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_e5m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_e5m2_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_dot_e5m2_neonbfdot,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_dot_e5m2_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_dot_e5m2_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_e5m2_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_dot_e5m2_genoa,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_dot_e5m2_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_e5m2_rvv,
#endif
#if NUMKONG_TARGET_RVVBF16
        (nk_kernel_punned_t)&nk_dot_e5m2_rvvbf16,
#endif
#if NUMKONG_TARGET_RVVHALF
        (nk_kernel_punned_t)&nk_dot_e5m2_rvvhalf,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_e5m2_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_rvvbf16_k * NUMKONG_TARGET_RVVBF16 |
             nk_cap_rvvhalf_k * NUMKONG_TARGET_RVVHALF | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_e2m3_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_e2m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_e2m3_neon,
#endif
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_dot_e2m3_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_dot_e2m3_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_e2m3_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_dot_e2m3_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_dot_e2m3_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_e2m3_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_dot_e2m3_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_e2m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_e2m3_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT |
             nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_e2m1_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_e2m1_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_dot_e2m1_neonsdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_e2m1_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_dot_e2m1_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_dot_e2m1_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_e2m1_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_e2m1_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_e2m1_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_e3m2_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_e3m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_e3m2_neon,
#endif
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_dot_e3m2_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_dot_e3m2_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_e3m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_e3m2_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_dot_e3m2_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_e3m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_e3m2_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT |
             nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_f32c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_f32c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_f32c_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_dot_f32c_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_f32c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_f32c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_f32c_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_f32c_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_f64c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_f64c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_f64c_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_dot_f64c_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_f64c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_dot_f64c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_dot_f64c_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_dot_f64c_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_f16c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_f16c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_dot_f16c_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_dot_f16c_neonfhm,
#endif
#if NUMKONG_TARGET_SVEHALF
        (nk_kernel_punned_t)&nk_dot_f16c_svehalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_f16c_haswell,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_dot_bf16c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_dot_bf16c_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_dot_bf16c_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_dot_bf16c_haswell,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_dot_bf16c_genoa,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_vdot_f32c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_vdot_f32c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_vdot_f32c_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_vdot_f32c_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_vdot_f32c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_vdot_f32c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_vdot_f32c_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_vdot_f32c_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_vdot_f64c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_vdot_f64c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_vdot_f64c_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_vdot_f64c_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_vdot_f64c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_vdot_f64c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_vdot_f64c_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_vdot_f64c_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_vdot_f16c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_vdot_f16c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_vdot_f16c_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_vdot_f16c_neonfhm,
#endif
#if NUMKONG_TARGET_SVEHALF
        (nk_kernel_punned_t)&nk_vdot_f16c_svehalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_vdot_f16c_haswell,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_vdot_bf16c_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_vdot_bf16c_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_vdot_bf16c_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_vdot_bf16c_haswell,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_vdot_bf16c_genoa,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

#if !NUMKONG_RUNTIME_DISPATCH

NUMKONG_API_COMPTIME nk_status_t nk_dot_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                 nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_f32_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                 nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_f64_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_f16_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_bf16_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_i8_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_u8_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_i4_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_u4_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_u1_best(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits,
                                                nk_u32_t *result, nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_u1_capabilities_());
    return kernel ? kernel(a, b, n_bits, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_e4m3_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_e5m2_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_e2m3_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_e2m1_best(nk_e2m1x2_t const *a, nk_e2m1x2_t const *b, nk_size_t n,
                                                  nk_f32_t *result, nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_e2m1_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_e3m2_capabilities_());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_f32c_best(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs,
                                                  nk_size_t count_pairs, nk_f64c_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_f32c_capabilities_());
    return kernel ? kernel(a_pairs, b_pairs, count_pairs, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_f64c_best(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs,
                                                  nk_size_t count_pairs, nk_f64c_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_f64c_capabilities_());
    return kernel ? kernel(a_pairs, b_pairs, count_pairs, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_f16c_best(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs,
                                                  nk_size_t count_pairs, nk_f32c_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_f16c_capabilities_());
    return kernel ? kernel(a_pairs, b_pairs, count_pairs, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_dot_bf16c_best(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs,
                                                   nk_size_t count_pairs, nk_f32c_t *result,
                                                   nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_dot_bf16c_capabilities_());
    return kernel ? kernel(a_pairs, b_pairs, count_pairs, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_vdot_f32c_best(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs,
                                                   nk_size_t count_pairs, nk_f64c_t *result,
                                                   nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_vdot_f32c_capabilities_());
    return kernel ? kernel(a_pairs, b_pairs, count_pairs, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_vdot_f64c_best(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs,
                                                   nk_size_t count_pairs, nk_f64c_t *result,
                                                   nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_vdot_f64c_capabilities_());
    return kernel ? kernel(a_pairs, b_pairs, count_pairs, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_vdot_f16c_best(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs,
                                                   nk_size_t count_pairs, nk_f32c_t *result,
                                                   nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_vdot_f16c_capabilities_());
    return kernel ? kernel(a_pairs, b_pairs, count_pairs, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API_COMPTIME nk_status_t nk_vdot_bf16c_best(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs,
                                                    nk_size_t count_pairs, nk_f32c_t *result,
                                                    nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_vdot_bf16c_capabilities_());
    return kernel ? kernel(a_pairs, b_pairs, count_pairs, result, stream) : nk_missing_kernel_k;
}

#endif // !NUMKONG_RUNTIME_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif
