/**
 *  @brief SIMD-accelerated Trigonometric Functions.
 *  @file include/numkong/trigonometry.h
 *  @author Ash Vardanian
 *  @date July 1, 2023
 *  @see SLEEF: https://sleef.org/
 *
 *  Contains:
 *
 *  - Sine and Cosine approximations: fast for `f32` vs accurate for `f64`
 *  - Tangent and the 2-argument arctangent: fast for `f32` vs accurate for `f64`
 *
 *  For dtypes:
 *
 *  - 64-bit IEEE-754 floating point
 *  - 32-bit IEEE-754 floating point
 *  - 16-bit IEEE-754 floating point
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Skylake, Sapphire Rapids
 *
 *  Those functions partially complement the `each.h` module, and are necessary for
 *  the `geospatial.h` module, among others. Both Haversine and Vincenty's formulas require
 *  trigonometric functions, and those are the most expensive part of the computation.
 *
 *  @section glibc_math GLibC IEEE-754-compliant Math Functions
 *
 *  The GNU C Library (GLibC) provides a set of IEEE-754-compliant math functions, like `sinf`, `cosf`,
 *  and double-precision variants `sin`, `cos`. Those functions are accurate to ~0.55 ULP (units in the
 *  last place), but can be slow to evaluate. They use a combination of techniques, like:
 *
 *  - Taylor series expansions for small values.
 *  - Table lookups combined with corrections for moderate values.
 *  - Accurate modulo reduction for large values.
 *
 *  The precomputed tables may be the hardest part to accelerate with SIMD, as they contain 440x values,
 *  each 64-bit wide.
 *
 *  https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/branred.c#L54
 *  https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/s_sin.c#L84
 *
 *  @section approximation_algorithms Approximation Algorithms
 *
 *  There are several ways to approximate trigonometric functions, and the choice depends on the
 *  target hardware and the desired precision. Notably:
 *
 *  - Taylor Series approximation is a series expansion of a sum of its derivatives at a target point.
 *    It's easy to derive for differentiable functions, works well for functions smooth around the
 *    expsansion point, but can perform poorly for functions with singularities or high-frequency
 *    oscillations.
 *
 *  - Pade approximations are rational functions that approximate a function by a ratio of polynomials.
 *    It often converges faster than Taylor for functions with singularities or steep changes, provides
 *    good approximations for both smooth and rational functions, but can be more computationally
 *    intensive to evaluate, and can have holes (undefined points).
 *
 *  Moreover, most approximations can be combined with Horner's methods of evaluating polynomials
 *  to reduce the number of multiplications and additions, and to improve the numerical stability.
 *  In trigonometry, the Payne-Hanek Range Reduction is another technique used to reduce the argument
 *  to a smaller range, where the approximation is more accurate.
 *
 *  @section optimization_notes Optimization Notes
 *
 *  The following optimizations were evaluated but did not yield performance improvements:
 *
 *  - Estrin's scheme for polynomial evaluation: This tree-based approach reduces the dependency depth
 *    from N sequential FMAs to log2(N) by computing powers of x in parallel with partial sums.
 *    For an 8-term polynomial, Estrin reduces depth from 7 to 3. However, benchmarks showed ~20%
 *    regression because the extra MUL operations for computing x², x⁴, x⁸ hurt throughput more
 *    than the reduced dependency depth helps latency. For large arrays, out-of-order execution
 *    across loop iterations already hides FMA latency, making throughput the bottleneck.
 *
 *  - RCPPS with Newton-Raphson refinement: Fast reciprocal approximation (~4 cycles) with one
 *    refinement iteration for ~22-bit precision, tested as an alternative to VDIVPS (~11 cycles).
 *    Did not improve performance when combined with Estrin's scheme, likely because the division
 *    is not on the critical path when processing large arrays.
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  Polynomial evaluation (Horner's method) for sin/cos/tan uses chained FMAs - the 4-cycle latency
 *  is hidden by out-of-order execution across iterations. Range reduction uses VRNDSCALE for fast
 *  rounding (notably 3x faster on Genoa than Ice Lake). VFPCLASS detects NaN/Inf inputs for special
 *  case handling. Division appears in tangent's final step but isn't on the critical path.
 *
 *      Intrinsic               Instruction                  Icelake      Genoa
 *      _mm512_roundscale_ps    VRNDSCALEPS (ZMM, ZMM, I8)   8cy @ p0+p0  3cy @ p23
 *      _mm512_roundscale_pd    VRNDSCALEPD (ZMM, ZMM, I8)   8cy @ p0+p0  3cy @ p23
 *      _mm512_fpclass_ps_mask  VFPCLASSPS (K, ZMM, I8)      3cy @ p5     5cy @ p01
 *      _mm512_fmadd_ps         VFMADD231PS (ZMM, ZMM, ZMM)  4cy @ p0     4cy @ p01
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)  4cy @ p01    4cy @ p01
 *      _mm256_div_ps           VDIVPS (YMM, YMM, YMM)       ~11cy @ p0   ~11cy @ p01
 *      _mm256_div_pd           VDIVPD (YMM, YMM, YMM)       ~13cy @ p0   ~13cy @ p01
 *
 *  @section arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  ARM implementations use the same Horner polynomial approach with FMLA chains. FRINTA provides
 *  fast rounding for range reduction. The 4-cycle FMA latency with 4 inst/cycle throughput allows
 *  excellent pipelining when processing multiple elements.
 *
 *      Intrinsic   Instruction   M1 Firestorm  Graviton 3   Graviton 4
 *      vfmaq_f32   FMLA.S (vec)  4cy @ V0123   4cy @ V0123  4cy @ V0123
 *      vfmaq_f64   FMLA.D (vec)  4cy @ V0123   4cy @ V0123  4cy @ V0123
 *      vrndaq_f32  FRINTA.S      2cy @ V0123   2cy @ V01    2cy @ V01
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NK_TRIGONOMETRY_H
#define NK_TRIGONOMETRY_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief RoPE rotation-coefficient type for the cos/sin angle grids.
 */
typedef nk_f32_t nk_rope_angle_t;

/**
 *  @brief Element-wise sine over f64 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of sine values.
 */
NK_DYNAMIC void nk_each_sin_f64(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);

/**
 *  @brief Element-wise cosine over f64 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of cosine values.
 */
NK_DYNAMIC void nk_each_cos_f64(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);

/**
 *  @brief Element-wise arc-tangent over f64 inputs.
 *
 *  @param[in] ins Input array of input values.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of arc-tangent values.
 */
NK_DYNAMIC void nk_each_atan_f64(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);

/**
 *  @brief Element-wise sine over f32 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of sine values.
 */
NK_DYNAMIC void nk_each_sin_f32(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);

/**
 *  @brief Element-wise cosine over f32 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of cosine values.
 */
NK_DYNAMIC void nk_each_cos_f32(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);

/**
 *  @brief Element-wise arc-tangent over f32 inputs.
 *
 *  @param[in] ins Input array of input values.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of arc-tangent values.
 */
NK_DYNAMIC void nk_each_atan_f32(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);

/**
 *  @brief Element-wise sine over f16 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of sine values.
 */
NK_DYNAMIC void nk_each_sin_f16(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);

/**
 *  @brief Element-wise cosine over f16 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of cosine values.
 */
NK_DYNAMIC void nk_each_cos_f16(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);

/**
 *  @brief Element-wise arc-tangent over f16 inputs.
 *
 *  @param[in] ins Input array of input values.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of arc-tangent values.
 */
NK_DYNAMIC void nk_each_atan_f16(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);

/** @copydoc nk_each_sin_f64 */
NK_PUBLIC void nk_each_sin_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_cos_f64 */
NK_PUBLIC void nk_each_cos_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_atan_f64 */
NK_PUBLIC void nk_each_atan_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_sin_f32 */
NK_PUBLIC void nk_each_sin_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_cos_f32 */
NK_PUBLIC void nk_each_cos_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_atan_f32 */
NK_PUBLIC void nk_each_atan_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_sin_f16 */
NK_PUBLIC void nk_each_sin_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);
/** @copydoc nk_each_cos_f16 */
NK_PUBLIC void nk_each_cos_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);
/** @copydoc nk_each_atan_f16 */
NK_PUBLIC void nk_each_atan_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);

#if NK_TARGET_NEON
/** @copydoc nk_each_sin_f64 */
NK_PUBLIC void nk_each_sin_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_cos_f64 */
NK_PUBLIC void nk_each_cos_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_atan_f64 */
NK_PUBLIC void nk_each_atan_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_sin_f32 */
NK_PUBLIC void nk_each_sin_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_cos_f32 */
NK_PUBLIC void nk_each_cos_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_atan_f32 */
NK_PUBLIC void nk_each_atan_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
#endif // NK_TARGET_NEON

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */

/**
 *  @brief NeoX split-half rotary position embedding (RoPE): rotates channel pairs by per-token angles.
 *
 *  Rotates each pair `(i, i + half_dim)` of every head: `y[i] = x[i]·cos - x[i+half_dim]·sin`,
 *  `y[i+half_dim] = x[i]·sin + x[i+half_dim]·cos`, over the whole `[rows, heads · 2·half_dim]` tensor.
 *
 *  @param[in] x Input token matrix of shape rows by (heads * 2 * half_dim).
 *  @param[out] y Output matrix, same shape and dtype as x; may alias x for in-place rotation.
 *  @param[in] cos Per-token cosine angle grid of shape rows by half_dim, shared across heads.
 *  @param[in] sin Per-token sine angle grid of shape rows by half_dim, shared across heads.
 *  @param[in] rows The number of token rows.
 *  @param[in] heads The number of heads per token.
 *  @param[in] half_dim Half the head dimension; channel i pairs with channel i + half_dim.
 *  @param[in] x_row_stride Row (token) stride of x in bytes.
 *  @param[in] y_row_stride Row (token) stride of y in bytes.
 *  @param[in] input_scale Scalar folded onto every loaded element (E4M3 descale; 1.0 for BF16/F32).
 */
NK_DYNAMIC void nk_each_rope_f32(nk_f32_t const *x, nk_f32_t *y, nk_rope_angle_t const *cos, nk_rope_angle_t const *sin,
                                 nk_size_t rows, nk_size_t heads, nk_size_t half_dim, nk_size_t x_row_stride,
                                 nk_size_t y_row_stride, nk_f32_t input_scale);
/** @copydoc nk_each_rope_f32 */
NK_DYNAMIC void nk_each_rope_bf16(nk_bf16_t const *x, nk_bf16_t *y, nk_rope_angle_t const *cos,
                                  nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads, nk_size_t half_dim,
                                  nk_size_t x_row_stride, nk_size_t y_row_stride, nk_f32_t input_scale);
/** @copydoc nk_each_rope_f32 */
NK_DYNAMIC void nk_each_rope_e4m3(nk_e4m3_t const *x, nk_e4m3_t *y, nk_rope_angle_t const *cos,
                                  nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads, nk_size_t half_dim,
                                  nk_size_t x_row_stride, nk_size_t y_row_stride, nk_f32_t input_scale);
/** @copydoc nk_each_rope_f32 */
NK_PUBLIC void nk_each_rope_f32_serial(nk_f32_t const *, nk_f32_t *, nk_rope_angle_t const *, nk_rope_angle_t const *,
                                       nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_f32_t);
/** @copydoc nk_each_rope_f32 */
NK_PUBLIC void nk_each_rope_bf16_serial(nk_bf16_t const *, nk_bf16_t *, nk_rope_angle_t const *,
                                        nk_rope_angle_t const *, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                        nk_f32_t);
/** @copydoc nk_each_rope_f32 */
NK_PUBLIC void nk_each_rope_e4m3_serial(nk_e4m3_t const *, nk_e4m3_t *, nk_rope_angle_t const *,
                                        nk_rope_angle_t const *, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                        nk_f32_t);

#if NK_TARGET_HASWELL
/** @copydoc nk_each_rope_f32 */
NK_PUBLIC void nk_each_rope_f32_haswell(nk_f32_t const *, nk_f32_t *, nk_rope_angle_t const *, nk_rope_angle_t const *,
                                        nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_f32_t);
/** @copydoc nk_each_rope_f32 */
NK_PUBLIC void nk_each_rope_bf16_haswell(nk_bf16_t const *, nk_bf16_t *, nk_rope_angle_t const *,
                                         nk_rope_angle_t const *, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                         nk_f32_t);
/** @copydoc nk_each_rope_f32 */
NK_PUBLIC void nk_each_rope_e4m3_haswell(nk_e4m3_t const *, nk_e4m3_t *, nk_rope_angle_t const *,
                                         nk_rope_angle_t const *, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                         nk_f32_t);
#endif // NK_TARGET_HASWELL
#if NK_TARGET_SKYLAKE
/** @copydoc nk_each_rope_f32 */
NK_PUBLIC void nk_each_rope_f32_skylake(nk_f32_t const *, nk_f32_t *, nk_rope_angle_t const *, nk_rope_angle_t const *,
                                        nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_f32_t);
/** @copydoc nk_each_rope_f32 */
NK_PUBLIC void nk_each_rope_bf16_skylake(nk_bf16_t const *, nk_bf16_t *, nk_rope_angle_t const *,
                                         nk_rope_angle_t const *, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                         nk_f32_t);
/** @copydoc nk_each_rope_f32 */
NK_PUBLIC void nk_each_rope_e4m3_skylake(nk_e4m3_t const *, nk_e4m3_t *, nk_rope_angle_t const *,
                                         nk_rope_angle_t const *, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                         nk_f32_t);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_HASWELL
/** @copydoc nk_each_sin_f64 */
NK_PUBLIC void nk_each_sin_f64_haswell(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_cos_f64 */
NK_PUBLIC void nk_each_cos_f64_haswell(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_atan_f64 */
NK_PUBLIC void nk_each_atan_f64_haswell(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_sin_f32 */
NK_PUBLIC void nk_each_sin_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_cos_f32 */
NK_PUBLIC void nk_each_cos_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_atan_f32 */
NK_PUBLIC void nk_each_atan_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
#endif // NK_TARGET_HASWELL

/*  SIMD-powered backends for various generations of AVX512 CPUs.
 *  Skylake is handy, as it supports masked loads and other operations, avoiding the need for the tail loop.
 */
#if NK_TARGET_SKYLAKE
/** @copydoc nk_each_sin_f64 */
NK_PUBLIC void nk_each_sin_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_cos_f64 */
NK_PUBLIC void nk_each_cos_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_atan_f64 */
NK_PUBLIC void nk_each_atan_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_sin_f32 */
NK_PUBLIC void nk_each_sin_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_cos_f32 */
NK_PUBLIC void nk_each_cos_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_atan_f32 */
NK_PUBLIC void nk_each_atan_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_sin_f16 */
NK_PUBLIC void nk_each_sin_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);
/** @copydoc nk_each_cos_f16 */
NK_PUBLIC void nk_each_cos_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);
/** @copydoc nk_each_atan_f16 */
NK_PUBLIC void nk_each_atan_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_V128RELAXED
/** @copydoc nk_each_sin_f64 */
NK_PUBLIC void nk_each_sin_f64_v128relaxed(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_cos_f64 */
NK_PUBLIC void nk_each_cos_f64_v128relaxed(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_atan_f64 */
NK_PUBLIC void nk_each_atan_f64_v128relaxed(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_sin_f32 */
NK_PUBLIC void nk_each_sin_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_cos_f32 */
NK_PUBLIC void nk_each_cos_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_atan_f32 */
NK_PUBLIC void nk_each_atan_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_RVV
/** @copydoc nk_each_sin_f64 */
NK_PUBLIC void nk_each_sin_f64_rvv(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_cos_f64 */
NK_PUBLIC void nk_each_cos_f64_rvv(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_atan_f64 */
NK_PUBLIC void nk_each_atan_f64_rvv(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs);
/** @copydoc nk_each_sin_f32 */
NK_PUBLIC void nk_each_sin_f32_rvv(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_cos_f32 */
NK_PUBLIC void nk_each_cos_f32_rvv(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_atan_f32 */
NK_PUBLIC void nk_each_atan_f32_rvv(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs);
/** @copydoc nk_each_sin_f16 */
NK_PUBLIC void nk_each_sin_f16_rvv(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);
/** @copydoc nk_each_cos_f16 */
NK_PUBLIC void nk_each_cos_f16_rvv(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);
/** @copydoc nk_each_atan_f16 */
NK_PUBLIC void nk_each_atan_f16_rvv(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs);
#endif // NK_TARGET_RVV

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/trigonometry/serial.h"
#include "numkong/trigonometry/neon.h"
#include "numkong/trigonometry/haswell.h"
#include "numkong/trigonometry/skylake.h"
#include "numkong/trigonometry/v128relaxed.h"
#include "numkong/trigonometry/rvv.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_each_sin_f64(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
#if NK_TARGET_NEON
    nk_each_sin_f64_neon(ins, n, outs);
#elif NK_TARGET_SKYLAKE
    nk_each_sin_f64_skylake(ins, n, outs);
#elif NK_TARGET_HASWELL
    nk_each_sin_f64_haswell(ins, n, outs);
#elif NK_TARGET_V128RELAXED
    nk_each_sin_f64_v128relaxed(ins, n, outs);
#elif NK_TARGET_RVV
    nk_each_sin_f64_rvv(ins, n, outs);
#else
    nk_each_sin_f64_serial(ins, n, outs);
#endif
}

NK_PUBLIC void nk_each_cos_f64(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
#if NK_TARGET_NEON
    nk_each_cos_f64_neon(ins, n, outs);
#elif NK_TARGET_SKYLAKE
    nk_each_cos_f64_skylake(ins, n, outs);
#elif NK_TARGET_HASWELL
    nk_each_cos_f64_haswell(ins, n, outs);
#elif NK_TARGET_V128RELAXED
    nk_each_cos_f64_v128relaxed(ins, n, outs);
#elif NK_TARGET_RVV
    nk_each_cos_f64_rvv(ins, n, outs);
#else
    nk_each_cos_f64_serial(ins, n, outs);
#endif
}

NK_PUBLIC void nk_each_atan_f64(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
#if NK_TARGET_NEON
    nk_each_atan_f64_neon(ins, n, outs);
#elif NK_TARGET_SKYLAKE
    nk_each_atan_f64_skylake(ins, n, outs);
#elif NK_TARGET_HASWELL
    nk_each_atan_f64_haswell(ins, n, outs);
#elif NK_TARGET_V128RELAXED
    nk_each_atan_f64_v128relaxed(ins, n, outs);
#elif NK_TARGET_RVV
    nk_each_atan_f64_rvv(ins, n, outs);
#else
    nk_each_atan_f64_serial(ins, n, outs);
#endif
}

NK_PUBLIC void nk_each_sin_f32(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
#if NK_TARGET_NEON
    nk_each_sin_f32_neon(ins, n, outs);
#elif NK_TARGET_SKYLAKE
    nk_each_sin_f32_skylake(ins, n, outs);
#elif NK_TARGET_HASWELL
    nk_each_sin_f32_haswell(ins, n, outs);
#elif NK_TARGET_V128RELAXED
    nk_each_sin_f32_v128relaxed(ins, n, outs);
#elif NK_TARGET_RVV
    nk_each_sin_f32_rvv(ins, n, outs);
#else
    nk_each_sin_f32_serial(ins, n, outs);
#endif
}

NK_PUBLIC void nk_each_cos_f32(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
#if NK_TARGET_NEON
    nk_each_cos_f32_neon(ins, n, outs);
#elif NK_TARGET_SKYLAKE
    nk_each_cos_f32_skylake(ins, n, outs);
#elif NK_TARGET_HASWELL
    nk_each_cos_f32_haswell(ins, n, outs);
#elif NK_TARGET_V128RELAXED
    nk_each_cos_f32_v128relaxed(ins, n, outs);
#elif NK_TARGET_RVV
    nk_each_cos_f32_rvv(ins, n, outs);
#else
    nk_each_cos_f32_serial(ins, n, outs);
#endif
}

NK_PUBLIC void nk_each_atan_f32(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
#if NK_TARGET_NEON
    nk_each_atan_f32_neon(ins, n, outs);
#elif NK_TARGET_SKYLAKE
    nk_each_atan_f32_skylake(ins, n, outs);
#elif NK_TARGET_HASWELL
    nk_each_atan_f32_haswell(ins, n, outs);
#elif NK_TARGET_V128RELAXED
    nk_each_atan_f32_v128relaxed(ins, n, outs);
#elif NK_TARGET_RVV
    nk_each_atan_f32_rvv(ins, n, outs);
#else
    nk_each_atan_f32_serial(ins, n, outs);
#endif
}

NK_PUBLIC void nk_each_sin_f16(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
#if NK_TARGET_SKYLAKE
    nk_each_sin_f16_skylake(ins, n, outs);
#elif NK_TARGET_RVV
    nk_each_sin_f16_rvv(ins, n, outs);
#else
    nk_each_sin_f16_serial(ins, n, outs);
#endif
}

NK_PUBLIC void nk_each_cos_f16(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
#if NK_TARGET_SKYLAKE
    nk_each_cos_f16_skylake(ins, n, outs);
#elif NK_TARGET_RVV
    nk_each_cos_f16_rvv(ins, n, outs);
#else
    nk_each_cos_f16_serial(ins, n, outs);
#endif
}

NK_PUBLIC void nk_each_atan_f16(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
#if NK_TARGET_SKYLAKE
    nk_each_atan_f16_skylake(ins, n, outs);
#elif NK_TARGET_RVV
    nk_each_atan_f16_rvv(ins, n, outs);
#else
    nk_each_atan_f16_serial(ins, n, outs);
#endif
}

NK_PUBLIC void nk_each_rope_f32(nk_f32_t const *x, nk_f32_t *y, nk_rope_angle_t const *cos, nk_rope_angle_t const *sin,
                                nk_size_t rows, nk_size_t heads, nk_size_t half_dim, nk_size_t x_row_stride,
                                nk_size_t y_row_stride, nk_f32_t input_scale) {
#if NK_TARGET_SKYLAKE
    nk_each_rope_f32_skylake(x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);
#elif NK_TARGET_HASWELL
    nk_each_rope_f32_haswell(x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);
#else
    nk_each_rope_f32_serial(x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);
#endif
}

NK_PUBLIC void nk_each_rope_bf16(nk_bf16_t const *x, nk_bf16_t *y, nk_rope_angle_t const *cos,
                                 nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads, nk_size_t half_dim,
                                 nk_size_t x_row_stride, nk_size_t y_row_stride, nk_f32_t input_scale) {
#if NK_TARGET_SKYLAKE
    nk_each_rope_bf16_skylake(x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);
#elif NK_TARGET_HASWELL
    nk_each_rope_bf16_haswell(x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);
#else
    nk_each_rope_bf16_serial(x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);
#endif
}

NK_PUBLIC void nk_each_rope_e4m3(nk_e4m3_t const *x, nk_e4m3_t *y, nk_rope_angle_t const *cos,
                                 nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads, nk_size_t half_dim,
                                 nk_size_t x_row_stride, nk_size_t y_row_stride, nk_f32_t input_scale) {
#if NK_TARGET_SKYLAKE
    nk_each_rope_e4m3_skylake(x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);
#elif NK_TARGET_HASWELL
    nk_each_rope_e4m3_haswell(x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);
#else
    nk_each_rope_e4m3_serial(x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TRIGONOMETRY_H
