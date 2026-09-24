/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers.
 *  @file include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date February 24, 2024
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
 *  - WASM: V128Relaxed
 *
 *  @section numerical_stability Numerical Stability
 *
 *  - f64: Dot2/Ogita-Rump-Oishi style compensated summation across serial and SIMD stateful paths.
 *  - f32: public outputs widen to f64/f64c. Arithmetic widens before the first lossy reduction step.
 *  - f16/bf16: Promoted to f32 accumulator.
 *  - e4m3/e5m2: Promoted to f32. On Sapphire, e2m3/e3m2 use f16 intermediate with periodic
 *    flush to f32 every 128 elements to avoid f16 overflow (max lane sum ~225 / ~3136).
 *  - i8: i32 accumulator. Max product |(-128)²| = 16,384. Overflows at n > 2^31/16,384 ≈ 131K.
 *  - u8: u32 accumulator. Max product 255² = 65,025. Overflows at n > 2^32/65,025 ≈ 66K.
 *  - i4: i32 accumulator. Max product 8² = 64. Safe for n ≤ ~33M.
 *  - u4: u32 accumulator. Max product 15² = 225. Safe for n ≤ ~19M.
 *  - u1: Popcount of AND into u32. Safe for n_bits ≤ 2^32.
 *  - Complex: Components accumulated independently; same guarantees as real counterpart.
 *
 *  @section streaming_api Streaming API
 *
 *  For compile-time dispatch and vector-at-a-time accumulation, we provide streaming helpers
 *  that accept two `nk_b512_vec_t` blocks and update a running sum for non-complex dot
 *  products. The `<count>` suffix reflects how many scalars of that type fit in a 512-bit block.
 *  The helpers are exposed per scalar type as:
 *
 *  - nk_dot_<type>x<count>_state_<isa>_t
 *  - nk_dot_<type>x<count>_init_<isa>
 *  - nk_dot_<type>x<count>_update_<isa>
 *  - nk_dot_<type>x<count>_finalize_<isa>
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  Floating-point dot products use FMA (VFMADD231PS/PD) for sum += a[i]*b[i] accumulation.
 *  Integer i8 dot products use VPMADDUBSW (u8 × i8 → i16) + VPMADDWD (i16 × 1 → i32) on Haswell,
 *  or the newer VNNI instructions VPDPBUSD/VPDPWSSD on Ice Lake+ for direct u8 × i8 → i32.
 *  BF16 dot products (VDPBF16PS) are Genoa-only, accumulating bf16 pairs directly to f32.
 *  Genoa shows 40% faster integer multiply-add (3c vs 5c) than Ice Lake.
 *
 *      Intrinsic               Instruction                     Haswell     Ice         Genoa
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)     5c @ p01    4c @ p01    4c @ p01
 *      _mm256_fmadd_pd         VFMADD231PD (YMM, YMM, YMM)     5c @ p01    4c @ p01    4c @ p01
 *      _mm256_maddubs_epi16    VPMADDUBSW (YMM, YMM, YMM)      5c @ p0     5c @ p01    3c @ p01
 *      _mm256_madd_epi16       VPMADDWD (YMM, YMM, YMM)        5c @ p0     5c @ p01    3c @ p01
 *      _mm256_dpbusd_epi32     VPDPBUSD (YMM, YMM, YMM)        N/A         5c @ p01    4c @ p01
 *      _mm512_dpwssd_epi32     VPDPWSSD (ZMM, ZMM, ZMM)        N/A         5c @ p0     4c @ p01
 *      _mm512_dpbf16_ps        VDPBF16PS (ZMM, ZMM, ZMM)       N/A         N/A         6c @ p01
 *
 *  @section arm_neon_instructions Relevant ARM NEON Instructions
 *
 *  NEON integer dot products use SDOT/UDOT (ARMv8.2 dotprod) for direct i8 × i8 → i32 or u8 × u8 → u32
 *  accumulation - 4x faster than the multiply-add sequence on older cores. BFDOT (ARMv8.6 bf16)
 *  provides native bf16 dot products on Graviton 3+. Complex dot products use LD2 for deinterleaved
 *  loads of real/imag pairs, though its L01+V throughput can bottleneck on memory-bound workloads.
 *
 *      Intrinsic               Instruction     M1 Firestorm    Graviton 3      Graviton 4
 *      vfmaq_f32               FMLA.S (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vfmaq_f64               FMLA.D (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vdotq_s32               SDOT (vec)      3c @ V0123      3c @ V0123      3c @ V0123
 *      vdotq_u32               UDOT (vec)      3c @ V0123      3c @ V0123      3c @ V0123
 *      vbfdotq_f32             BFDOT (vec)     N/A             4c @ V0123      5c @ V0123
 *      vld2q_f32               LD2 (Q-form)    5c @ L01+V      8c @ L01+V      8c @ L01+V
 *
 *  @section arm_sve_instructions Relevant ARM SVE Instructions
 *
 *  SVE implementations use predicated FMA (svmla_f32_x) with WHILELT for tail masking, avoiding
 *  scalar cleanup loops. FADDV performs horizontal reduction; notably 45% faster on Graviton 4
 *  (6c) than Graviton 3 (11c). SVE complex dot products use svld2 for structure loads.
 *
 *      Intrinsic               Instruction     Graviton 3      Graviton 4
 *      svmla_f32_x             FMLA (pred)     4c @ V0123      4c @ V0123
 *      svmls_f32_x             FMLS (pred)     4c @ V0123      4c @ V0123
 *      svwhilelt_b32           WHILELT         3c @ M0         3c @ M0
 *      svld2_f32               LD2 (SVE)       8c @ L01+V      8c @ L01+V
 *      svaddv_f32              FADDV           11c @ V0123     6c @ V0123
 *
 *  @section complex_instructions Complex Number Optimizations
 *
 *  Standard complex multiplication involves subtraction for the real part.
 *  Instead of using subtracting variants of FMA for every element, we accumulate real
 *  and imaginary products positively and apply a single bitwise XOR to flip the sign
 *  bits before the final horizontal reduction. This delayed application of the sign
 *  flip doubles the throughput on older x86 architectures like Haswell by maximizing
 *  FMA unit utilization and reducing execution dependency chains.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NK_DOT_H
#define NK_DOT_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Dot product computing the sum of elementwise products between two vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n The number of elements in the vectors.
 *  @param[out] result The output dot product value.
 *
 *  @note The output value can be negative.
 *  @note The output value is zero if and only if the two vectors are orthogonal.
 *  @note Defined for floating-point, integer, and binary data types.
 */
NK_DYNAMIC void nk_dot_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_u1(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_e2m3(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_e3m2(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);

/**
 *  @brief Complex dot product computing the sum of elementwise products between two complex vectors.
 *
 *  @param[in] a_pairs The first complex vector.
 *  @param[in] b_pairs The second complex vector.
 *  @param[in] count_pairs The number of complex pairs in the vectors.
 *  @param[out] result The output complex value as {real, imag}.
 */
NK_DYNAMIC void nk_dot_f32c(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                            nk_f64c_t *result);
/** @copydoc nk_dot_f32c */
NK_DYNAMIC void nk_dot_f64c(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                            nk_f64c_t *result);
/** @copydoc nk_dot_f32c */
NK_DYNAMIC void nk_dot_f16c(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                            nk_f32c_t *result);
/** @copydoc nk_dot_f32c */
NK_DYNAMIC void nk_dot_bf16c(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                             nk_f32c_t *result);

/**
 *  @brief Complex conjugate dot product between two complex vectors.
 *
 *  @param[in] a_pairs The first complex vector.
 *  @param[in] b_pairs The second complex vector.
 *  @param[in] count_pairs The number of complex pairs in the vectors.
 *  @param[out] result The output complex value as {real, imag}.
 */
NK_DYNAMIC void nk_vdot_f32c(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                             nk_f64c_t *result);
/** @copydoc nk_vdot_f32c */
NK_DYNAMIC void nk_vdot_f64c(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                             nk_f64c_t *result);
/** @copydoc nk_vdot_f32c */
NK_DYNAMIC void nk_vdot_f16c(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                             nk_f32c_t *result);
/** @copydoc nk_vdot_f32c */
NK_DYNAMIC void nk_vdot_bf16c(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                              nk_f32c_t *result);

/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_serial(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_serial(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16c */
NK_PUBLIC void nk_dot_bf16c_serial(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_bf16c */
NK_PUBLIC void nk_vdot_bf16c_serial(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_i4 */
NK_PUBLIC void nk_dot_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u4 */
NK_PUBLIC void nk_dot_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_u1 */
NK_PUBLIC void nk_dot_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_serial(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_serial(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);

#if NK_TARGET_NEON
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_dot_u1 */
NK_PUBLIC void nk_dot_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result);

/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);

#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_neonhalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_neonhalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONFHM
/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_neonfhm(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_neonfhm(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_neonfhm(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_neonfhm(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_NEONSDOT
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_i4 */
NK_PUBLIC void nk_dot_i4_neonsdot(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u4 */
NK_PUBLIC void nk_dot_u4_neonsdot(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_neonsdot(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_neonsdot(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONBFDOT
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_neonbfdot(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_neonbfdot(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16c */
NK_PUBLIC void nk_dot_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_bf16c */
NK_PUBLIC void nk_vdot_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_SVEBFDOT
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SVEBFDOT

#if NK_TARGET_SVE
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_sve(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_sve(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_sve(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_sve(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
#endif // NK_TARGET_SVE

#if NK_TARGET_SVEHALF
/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_svehalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_svehalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
#endif // NK_TARGET_SVEHALF

#if NK_TARGET_HASWELL
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_haswell(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_haswell(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_haswell(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_haswell(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16c */
NK_PUBLIC void nk_dot_bf16c_haswell(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_bf16c */
NK_PUBLIC void nk_vdot_bf16c_haswell(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_haswell(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_haswell(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_i4 */
NK_PUBLIC void nk_dot_i4_haswell(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u4 */
NK_PUBLIC void nk_dot_u4_haswell(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_u1 */
NK_PUBLIC void nk_dot_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result);

#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_skylake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_skylake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_icelake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i4_icelake(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u4_icelake(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_icelake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_icelake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_u1 */
NK_PUBLIC void nk_dot_u1_icelake(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_GENOA
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16c */
NK_PUBLIC void nk_dot_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_bf16c */
NK_PUBLIC void nk_vdot_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_genoa(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_genoa(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_GENOA

#if NK_TARGET_ALDER
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_alder(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_alder(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_alder(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_ALDER

#if NK_TARGET_SIERRA
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_sierra(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_sierra(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SIERRA

#if NK_TARGET_RVV
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_rvv(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_rvv(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_i4 */
NK_PUBLIC void nk_dot_i4_rvv(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u4 */
NK_PUBLIC void nk_dot_u4_rvv(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_u1 */
NK_PUBLIC void nk_dot_u1_rvv(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_rvv(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_rvv(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_rvv(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_rvv(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
#endif // NK_TARGET_RVV

#if NK_TARGET_RVVHALF
/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_rvvhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_rvvhalf(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_rvvhalf(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_RVVHALF

#if NK_TARGET_RVVBF16
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_rvvbf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_rvvbf16(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_rvvbf16(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_RVVBF16

#if NK_TARGET_RVVBB
/** @copydoc nk_dot_u1 */
NK_PUBLIC void nk_dot_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result);
#endif // NK_TARGET_RVVBB

#if NK_TARGET_V128RELAXED
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_v128relaxed(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_v128relaxed(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_u1 */
NK_PUBLIC void nk_dot_u1_v128relaxed(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result);
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_e4m3_v128relaxed(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_e5m2_v128relaxed(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_i4 */
NK_PUBLIC void nk_dot_i4_v128relaxed(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u4 */
NK_PUBLIC void nk_dot_u4_v128relaxed(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_v128relaxed(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_vdot_f32c_v128relaxed(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_v128relaxed(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_vdot_f64c_v128relaxed(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
#endif // NK_TARGET_V128RELAXED

/**
 *  @brief  Returns the output dtype for dot products.
 */
NK_INTERNAL nk_dtype_t nk_dot_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_e2m3_k: return nk_f32_k;
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

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/dot/serial.h"
#include "numkong/dot/neon.h"
#include "numkong/dot/neonsdot.h"
#include "numkong/dot/neonhalf.h"
#include "numkong/dot/neonfhm.h"
#include "numkong/dot/neonbfdot.h"
#include "numkong/dot/sve.h"
#include "numkong/dot/svehalf.h"
#include "numkong/dot/svebfdot.h"
#include "numkong/dot/haswell.h"
#include "numkong/dot/skylake.h"
#include "numkong/dot/icelake.h"
#include "numkong/dot/genoa.h"
#include "numkong/dot/sapphire.h"
#include "numkong/dot/alder.h"
#include "numkong/dot/sierra.h"
#include "numkong/dot/rvv.h"
#include "numkong/dot/rvvbb.h"
#include "numkong/dot/rvvhalf.h"
#include "numkong/dot/rvvbf16.h"
#include "numkong/dot/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_dot_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result) {
#if NK_TARGET_V128RELAXED
    nk_dot_i8_v128relaxed(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_i8_rvv(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_dot_i8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICELAKE
    nk_dot_i8_icelake(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_i8_skylake(a, b, n, result);
#elif NK_TARGET_SIERRA
    nk_dot_i8_sierra(a, b, n, result);
#elif NK_TARGET_ALDER
    nk_dot_i8_alder(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_i8_haswell(a, b, n, result);
#else
    nk_dot_i8_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_V128RELAXED
    nk_dot_u8_v128relaxed(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_u8_rvv(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_dot_u8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICELAKE
    nk_dot_u8_icelake(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_u8_skylake(a, b, n, result);
#elif NK_TARGET_SIERRA
    nk_dot_u8_sierra(a, b, n, result);
#elif NK_TARGET_ALDER
    nk_dot_u8_alder(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_u8_haswell(a, b, n, result);
#else
    nk_dot_u8_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
#if NK_TARGET_ICELAKE
    nk_dot_i4_icelake(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_dot_i4_neonsdot(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_i4_rvv(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_i4_haswell(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_dot_i4_v128relaxed(a, b, n, result);
#else
    nk_dot_i4_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_ICELAKE
    nk_dot_u4_icelake(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_dot_u4_neonsdot(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_u4_rvv(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_u4_haswell(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_dot_u4_v128relaxed(a, b, n, result);
#else
    nk_dot_u4_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_u1(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result) {
#if NK_TARGET_ICELAKE
    nk_dot_u1_icelake(a, b, n_bits, result);
#elif NK_TARGET_HASWELL
    nk_dot_u1_haswell(a, b, n_bits, result);
#elif NK_TARGET_V128RELAXED
    nk_dot_u1_v128relaxed(a, b, n_bits, result);
#elif NK_TARGET_RVVBB
    nk_dot_u1_rvvbb(a, b, n_bits, result);
#elif NK_TARGET_RVV
    nk_dot_u1_rvv(a, b, n_bits, result);
#elif NK_TARGET_NEON
    nk_dot_u1_neon(a, b, n_bits, result);
#else
    nk_dot_u1_serial(a, b, n_bits, result);
#endif
}

NK_PUBLIC void nk_dot_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_V128RELAXED
    nk_dot_f16_v128relaxed(a, b, n, result);
#elif NK_TARGET_RVVHALF
    nk_dot_f16_rvvhalf(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_f16_rvv(a, b, n, result);
#elif NK_TARGET_SVEHALF
    nk_dot_f16_svehalf(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_dot_f16_neonfhm(a, b, n, result);
#elif NK_TARGET_NEONHALF
    nk_dot_f16_neonhalf(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_f16_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f16_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f16_haswell(a, b, n, result);
#else
    nk_dot_f16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_V128RELAXED
    nk_dot_bf16_v128relaxed(a, b, n, result);
#elif NK_TARGET_GENOA
    nk_dot_bf16_genoa(a, b, n, result);
#elif NK_TARGET_RVVBF16
    nk_dot_bf16_rvvbf16(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_bf16_rvv(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_bf16_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_bf16_haswell(a, b, n, result);
#elif NK_TARGET_SVEBFDOT
    nk_dot_bf16_svebfdot(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_dot_bf16_neonbfdot(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_bf16_neon(a, b, n, result);
#else
    nk_dot_bf16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_dot_e4m3_genoa(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_dot_e4m3_neonbfdot(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_dot_e4m3_neonfhm(a, b, n, result);
#elif NK_TARGET_RVVHALF
    nk_dot_e4m3_rvvhalf(a, b, n, result);
#elif NK_TARGET_RVVBF16
    nk_dot_e4m3_rvvbf16(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_e4m3_rvv(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_dot_e4m3_v128relaxed(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_e4m3_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_e4m3_haswell(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_e4m3_neon(a, b, n, result);
#else
    nk_dot_e4m3_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_dot_e5m2_genoa(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_dot_e5m2_neonbfdot(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_dot_e5m2_neonfhm(a, b, n, result);
#elif NK_TARGET_RVVHALF
    nk_dot_e5m2_rvvhalf(a, b, n, result);
#elif NK_TARGET_RVVBF16
    nk_dot_e5m2_rvvbf16(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_e5m2_rvv(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_dot_e5m2_v128relaxed(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_e5m2_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_e5m2_haswell(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_e5m2_neon(a, b, n, result);
#else
    nk_dot_e5m2_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_e2m3(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_ICELAKE
    nk_dot_e2m3_icelake(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_e2m3_skylake(a, b, n, result);
#elif NK_TARGET_SIERRA
    nk_dot_e2m3_sierra(a, b, n, result);
#elif NK_TARGET_ALDER
    nk_dot_e2m3_alder(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_e2m3_rvv(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_e2m3_haswell(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_dot_e2m3_neonsdot(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_e2m3_neon(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_dot_e2m3_v128relaxed(a, b, n, result);
#else
    nk_dot_e2m3_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_e3m2(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_ICELAKE
    nk_dot_e3m2_icelake(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_dot_e3m2_neonsdot(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_dot_e3m2_v128relaxed(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_e3m2_rvv(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_e3m2_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_e3m2_haswell(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_e3m2_neon(a, b, n, result);
#else
    nk_dot_e3m2_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_V128RELAXED
    nk_dot_f32_v128relaxed(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_f32_rvv(a, b, n, result);
#elif NK_TARGET_SVE
    nk_dot_f32_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_f32_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f32_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f32_haswell(a, b, n, result);
#else
    nk_dot_f32_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_V128RELAXED
    nk_dot_f64_v128relaxed(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_f64_rvv(a, b, n, result);
#elif NK_TARGET_SVE
    nk_dot_f64_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_f64_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f64_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f64_haswell(a, b, n, result);
#else
    nk_dot_f64_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_f16c(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_SVEHALF
    nk_dot_f16c_svehalf(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_dot_f16c_neonfhm(a, b, n, result);
#elif NK_TARGET_NEONHALF
    nk_dot_f16c_neonhalf(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f16c_haswell(a, b, n, result);
#else
    nk_dot_f16c_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_bf16c(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_GENOA
    nk_dot_bf16c_genoa(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_dot_bf16c_neonbfdot(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_bf16c_haswell(a, b, n, result);
#else
    nk_dot_bf16c_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_f32c(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_TARGET_SVE
    nk_dot_f32c_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_f32c_neon(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_f32c_rvv(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f32c_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f32c_haswell(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_dot_f32c_v128relaxed(a, b, n, result);
#else
    nk_dot_f32c_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_dot_f64c(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_TARGET_SVE
    nk_dot_f64c_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_f64c_neon(a, b, n, result);
#elif NK_TARGET_RVV
    nk_dot_f64c_rvv(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f64c_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f64c_haswell(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_dot_f64c_v128relaxed(a, b, n, result);
#else
    nk_dot_f64c_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_vdot_f16c(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_SVEHALF
    nk_vdot_f16c_svehalf(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_vdot_f16c_neonfhm(a, b, n, result);
#elif NK_TARGET_NEONHALF
    nk_vdot_f16c_neonhalf(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_vdot_f16c_haswell(a, b, n, result);
#else
    nk_vdot_f16c_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_vdot_bf16c(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_GENOA
    nk_vdot_bf16c_genoa(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_vdot_bf16c_neonbfdot(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_vdot_bf16c_haswell(a, b, n, result);
#else
    nk_vdot_bf16c_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_vdot_f32c(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_TARGET_SVE
    nk_vdot_f32c_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_vdot_f32c_neon(a, b, n, result);
#elif NK_TARGET_RVV
    nk_vdot_f32c_rvv(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_vdot_f32c_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_vdot_f32c_haswell(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_vdot_f32c_v128relaxed(a, b, n, result);
#else
    nk_vdot_f32c_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_vdot_f64c(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_TARGET_SVE
    nk_vdot_f64c_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_vdot_f64c_neon(a, b, n, result);
#elif NK_TARGET_RVV
    nk_vdot_f64c_rvv(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_vdot_f64c_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_vdot_f64c_haswell(a, b, n, result);
#elif NK_TARGET_V128RELAXED
    nk_vdot_f64c_v128relaxed(a, b, n, result);
#else
    nk_vdot_f64c_serial(a, b, n, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif
