/**
 *  @file include/numkong/trigonometry.h
 *  @author Ash Vardanian
 *  @date July 1, 2023
 *  @brief SIMD-accelerated trigonometric functions.
 *
 *  Contains:
 *
 *  - Sine and Cosine approximations: fast for @c f32 vs accurate for @c f64
 *  - Tangent and the 2-argument arctangent: fast for @c f32 vs accurate for @c f64
 *
 *  For dtypes:
 *
 *  - 64-bit IEEE-754 floating point
 *  - 32-bit IEEE-754 floating point
 *  - 16-bit IEEE-754 floating point
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, NEON FP16, SVE FP16
 *  - x86: Haswell, Skylake, Sapphire Rapids
 *
 *  Those functions partially complement the `each.h` module, and are necessary for the
 *  `geospatial.h` module, among others. Both Haversine and Vincenty's formulas require
 *  trigonometric functions, and those are the most expensive part of the computation.
 *
 *  @see SLEEF: https://sleef.org/
 *
 *  @section trigonometry_accuracy Accuracy
 *
 *  - @c f64 sin, cos and atan: within 2 ULP of the correctly rounded result, so not faithful.
 *  - @c f32 sin and cos: within 2 ULP for |x| ≤ 10⁴, next to their zeros too; atan within 3 ULP.
 *  - @c f16: shorter polynomials evaluated in @c f32, within 1 ULP of the correctly rounded result.
 *  - @c f16 on NEON FP16, SVE FP16 and Sapphire Rapids: evaluated in @c f16, also within 1 ULP.
 *  - Their sine and cosine reduce in @c f16 up to |x| ≤ 256, and in @c f32 past it.
 *
 *  The measurements behind these bounds are in @c include/numkong/trigonometry/README.md.
 *
 *  @section glibc_math GLibC IEEE-754-compliant Math Functions
 *
 *  The GNU C Library, GLibC, provides IEEE-754-compliant math functions, like single-precision
 *  @c sinf and @c cosf or double-precision @c sin and @c cos. Those are accurate to ~0.55 ULP,
 *  units in the last place, but can be slow to evaluate. They combine techniques like:
 *
 *  - Taylor series expansions for small values.
 *  - Table lookups combined with corrections for moderate values.
 *  - Accurate modulo reduction for large values.
 *
 *  The precomputed tables may be the hardest part to accelerate with SIMD, as they contain 440x
 *  values, each 64-bit wide.
 *
 *  @see glibc argument reduction: https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/branred.c#L54
 *  @see glibc sine: https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/s_sin.c#L84
 *
 *  @section approximation_algorithms Approximation Algorithms
 *
 *  There are several ways to approximate trigonometric functions, and the choice depends on the
 *  target hardware and the desired precision.
 *
 *  Taylor Series approximation is a series expansion of a sum of its derivatives at a target point.
 *  It's easy to derive for differentiable functions and works well for functions smooth around the
 *  expansion point, but can perform poorly for functions with singularities or rapid,
 *  high-frequency oscillations.
 *
 *  Pade approximations are rational functions that approximate a function by a ratio of
 *  polynomials. They often converge faster than Taylor for functions with singularities or steep
 *  changes, and approximate both smooth and rational functions well, but can be more
 *  computationally intensive to evaluate, and can have holes, undefined points.
 *
 *  Moreover, most approximations can be combined with Horner's methods of evaluating polynomials to
 *  reduce the number of multiplications and additions, and to improve the numerical stability. In
 *  trigonometry, the Payne-Hanek Range Reduction is another technique used to reduce the argument
 *  to a smaller range, where the approximation is more accurate.
 *
 *  @section optimization_notes Optimization Notes
 *
 *  The following optimizations were evaluated but did not yield performance improvements.
 *
 *  Estrin's scheme for polynomial evaluation is a tree-based approach that reduces the dependency
 *  depth from N sequential FMAs to log2(N), by computing powers of x in parallel with partial sums.
 *  For an 8-term polynomial, it reduces depth from 7 to 3, but benchmarks showed a ~20% regression:
 *  the extra MUL operations for computing x², x⁴, x⁸ hurt throughput more than the reduced
 *  dependency depth helps latency. For large arrays, out-of-order execution across loop iterations
 *  already hides FMA latency, making throughput the bottleneck.
 *
 *  RCPPS with Newton-Raphson refinement, a fast reciprocal approximation at ~4 cycles with one
 *  refinement iteration for ~22-bit precision, was tested as an alternative to VDIVPS at ~11
 *  cycles. It did not improve performance when combined with Estrin's scheme, likely because the
 *  division is not on the critical path when processing large arrays.
 *
 *  @section trigonometry_x86_instructions Relevant x86 Instructions
 *
 *  Polynomial evaluation (Horner's method) for sin/cos/tan uses chained FMAs - the 4-cycle latency
 *  is hidden by out-of-order execution across iterations. Range reduction uses VRNDSCALE for fast
 *  rounding (notably 3x faster on Genoa than Ice Lake). VFPCLASS detects NaN/Inf inputs for special
 *  case handling. Division appears in tangent's final step but isn't on the critical path.
 *
 *  @verbatim
 *  Intrinsic               Instruction                  Icelake      Genoa
 *  _mm512_roundscale_ps    VRNDSCALEPS (ZMM, ZMM, I8)   8cy @ p0+p0  3cy @ p23
 *  _mm512_roundscale_pd    VRNDSCALEPD (ZMM, ZMM, I8)   8cy @ p0+p0  3cy @ p23
 *  _mm512_fpclass_ps_mask  VFPCLASSPS (K, ZMM, I8)      3cy @ p5     5cy @ p01
 *  _mm512_fmadd_ps         VFMADD231PS (ZMM, ZMM, ZMM)  4cy @ p0     4cy @ p01
 *  _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)  4cy @ p01    4cy @ p01
 *  _mm256_div_ps           VDIVPS (YMM, YMM, YMM)       ~11cy @ p0   ~11cy @ p01
 *  _mm256_div_pd           VDIVPD (YMM, YMM, YMM)       ~13cy @ p0   ~13cy @ p01
 *  @endverbatim
 *
 *  @section trigonometry_arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  ARM implementations use the same Horner polynomial approach with FMLA chains. FRINTA provides
 *  fast rounding for range reduction. The 4-cycle FMA latency with 4 inst/cycle throughput allows
 *  excellent pipelining when processing multiple elements.
 *
 *  @verbatim
 *  Intrinsic   Instruction   M1 Firestorm  Graviton 3   Graviton 4
 *  vfmaq_f32   FMLA.S (vec)  4cy @ V0123   4cy @ V0123  4cy @ V0123
 *  vfmaq_f64   FMLA.D (vec)  4cy @ V0123   4cy @ V0123  4cy @ V0123
 *  vrndaq_f32  FRINTA.S      2cy @ V0123   2cy @ V01    2cy @ V01
 *  @endverbatim
 *
 *  @section trigonometry_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NUMKONG_TRIGONOMETRY_H
#define NUMKONG_TRIGONOMETRY_H

#include "numkong/capabilities.h" // `nk_kernel_punned_t`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Element-wise sine over f64 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of sine values.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_trig_sin_f64_best(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs,
                                             nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Element-wise cosine over f64 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of cosine values.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_trig_cos_f64_best(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs,
                                             nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Element-wise arc-tangent over f64 inputs.
 *
 *  @param[in] ins Input array of input values.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of arc-tangent values.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_trig_atan_f64_best(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs,
                                              nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Element-wise sine over f32 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of sine values.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_trig_sin_f32_best(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs,
                                             nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Element-wise cosine over f32 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of cosine values.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_trig_cos_f32_best(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs,
                                             nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Element-wise arc-tangent over f32 inputs.
 *
 *  @param[in] ins Input array of input values.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of arc-tangent values.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_trig_atan_f32_best(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs,
                                              nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Element-wise sine over f16 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of sine values.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_trig_sin_f16_best(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs,
                                             nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Element-wise cosine over f16 inputs in radians.
 *
 *  @param[in] ins Input array of angles in radians.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of cosine values.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_trig_cos_f16_best(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs,
                                             nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Element-wise arc-tangent over f16 inputs.
 *
 *  @param[in] ins Input array of input values.
 *  @param[in] n Number of elements in the input/output arrays.
 *  @param[out] outs Output array of arc-tangent values.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_trig_atan_f16_best(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs,
                                              nk_capability_t capabilities, nk_stream_t stream);

/** @copydoc nk_trig_sin_f64_best */
NUMKONG_API nk_status_t nk_trig_sin_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f64_best */
NUMKONG_API nk_status_t nk_trig_cos_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f64_best */
NUMKONG_API nk_status_t nk_trig_atan_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_sin_f32_best */
NUMKONG_API nk_status_t nk_trig_sin_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f32_best */
NUMKONG_API nk_status_t nk_trig_cos_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f32_best */
NUMKONG_API nk_status_t nk_trig_atan_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_sin_f16_best */
NUMKONG_API nk_status_t nk_trig_sin_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f16_best */
NUMKONG_API nk_status_t nk_trig_cos_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f16_best */
NUMKONG_API nk_status_t nk_trig_atan_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_trig_sin_f64_best */
NUMKONG_API nk_status_t nk_trig_sin_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f64_best */
NUMKONG_API nk_status_t nk_trig_cos_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f64_best */
NUMKONG_API nk_status_t nk_trig_atan_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_sin_f32_best */
NUMKONG_API nk_status_t nk_trig_sin_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f32_best */
NUMKONG_API nk_status_t nk_trig_cos_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f32_best */
NUMKONG_API nk_status_t nk_trig_atan_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONHALF
/** @copydoc nk_trig_sin_f16_best */
NUMKONG_API nk_status_t nk_trig_sin_f16_neonhalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f16_best */
NUMKONG_API nk_status_t nk_trig_cos_f16_neonhalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f16_best */
NUMKONG_API nk_status_t nk_trig_atan_f16_neonhalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONHALF

#if NUMKONG_TARGET_SVEHALF
/** @copydoc nk_trig_sin_f16_best */
NUMKONG_API nk_status_t nk_trig_sin_f16_svehalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f16_best */
NUMKONG_API nk_status_t nk_trig_cos_f16_svehalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f16_best */
NUMKONG_API nk_status_t nk_trig_atan_f16_svehalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
#endif // NUMKONG_TARGET_SVEHALF

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic
 *  over 256-bit words. First demonstrated in 2011, at least one Haswell-based processor was
 *  still being sold in 2022 — the Pentium G3420. Practically all modern x86 CPUs support AVX2,
 *  FMA, and F16C, making it a perfect baseline for SIMD algorithms. On other hand, there is no
 *  need to implement AVX2 versions of @c f32 and @c f64 functions, as those are properly
 *  vectorized by recent compilers. */

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_trig_sin_f64_best */
NUMKONG_API nk_status_t nk_trig_sin_f64_haswell(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f64_best */
NUMKONG_API nk_status_t nk_trig_cos_f64_haswell(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f64_best */
NUMKONG_API nk_status_t nk_trig_atan_f64_haswell(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_sin_f32_best */
NUMKONG_API nk_status_t nk_trig_sin_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f32_best */
NUMKONG_API nk_status_t nk_trig_cos_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f32_best */
NUMKONG_API nk_status_t nk_trig_atan_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
#endif // NUMKONG_TARGET_HASWELL

/*  SIMD-powered backends for various generations of AVX512 CPUs. Skylake is handy, as it supports
 *  masked loads and other operations, avoiding the need for the tail loop. */
#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_trig_sin_f64_best */
NUMKONG_API nk_status_t nk_trig_sin_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f64_best */
NUMKONG_API nk_status_t nk_trig_cos_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f64_best */
NUMKONG_API nk_status_t nk_trig_atan_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_sin_f32_best */
NUMKONG_API nk_status_t nk_trig_sin_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f32_best */
NUMKONG_API nk_status_t nk_trig_cos_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f32_best */
NUMKONG_API nk_status_t nk_trig_atan_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_sin_f16_best */
NUMKONG_API nk_status_t nk_trig_sin_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f16_best */
NUMKONG_API nk_status_t nk_trig_cos_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f16_best */
NUMKONG_API nk_status_t nk_trig_atan_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_SAPPHIRE
/** @copydoc nk_trig_sin_f16_best */
NUMKONG_API nk_status_t nk_trig_sin_f16_sapphire(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f16_best */
NUMKONG_API nk_status_t nk_trig_cos_f16_sapphire(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f16_best */
NUMKONG_API nk_status_t nk_trig_atan_f16_sapphire(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
#endif // NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_trig_sin_f64_best */
NUMKONG_API nk_status_t nk_trig_sin_f64_v128relaxed(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs,
                                                    nk_stream_t stream);
/** @copydoc nk_trig_cos_f64_best */
NUMKONG_API nk_status_t nk_trig_cos_f64_v128relaxed(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs,
                                                    nk_stream_t stream);
/** @copydoc nk_trig_atan_f64_best */
NUMKONG_API nk_status_t nk_trig_atan_f64_v128relaxed(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs,
                                                     nk_stream_t stream);
/** @copydoc nk_trig_sin_f32_best */
NUMKONG_API nk_status_t nk_trig_sin_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs,
                                                    nk_stream_t stream);
/** @copydoc nk_trig_cos_f32_best */
NUMKONG_API nk_status_t nk_trig_cos_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs,
                                                    nk_stream_t stream);
/** @copydoc nk_trig_atan_f32_best */
NUMKONG_API nk_status_t nk_trig_atan_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs,
                                                     nk_stream_t stream);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_RVV
/** @copydoc nk_trig_sin_f64_best */
NUMKONG_API nk_status_t nk_trig_sin_f64_rvv(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f64_best */
NUMKONG_API nk_status_t nk_trig_cos_f64_rvv(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f64_best */
NUMKONG_API nk_status_t nk_trig_atan_f64_rvv(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_sin_f32_best */
NUMKONG_API nk_status_t nk_trig_sin_f32_rvv(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f32_best */
NUMKONG_API nk_status_t nk_trig_cos_f32_rvv(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f32_best */
NUMKONG_API nk_status_t nk_trig_atan_f32_rvv(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_sin_f16_best */
NUMKONG_API nk_status_t nk_trig_sin_f16_rvv(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_cos_f16_best */
NUMKONG_API nk_status_t nk_trig_cos_f16_rvv(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
/** @copydoc nk_trig_atan_f16_best */
NUMKONG_API nk_status_t nk_trig_atan_f16_rvv(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream);
#endif // NUMKONG_TARGET_RVV

/**
 *  @brief Finds the trigonometry kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when no capability in @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_trigonometry_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype,
                                                    nk_capability_t capabilities, nk_kernel_punned_t *kernel,
                                                    nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/trigonometry/serial.h"
#include "numkong/trigonometry/neon.h"
#include "numkong/trigonometry/neonhalf.h"
#include "numkong/trigonometry/svehalf.h"
#include "numkong/trigonometry/haswell.h"
#include "numkong/trigonometry/skylake.h"
#include "numkong/trigonometry/sapphire.h"
#include "numkong/trigonometry/v128relaxed.h"
#include "numkong/trigonometry/rvv.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_trig_sin_f64_best(nk_f64_t const *inputs, nk_size_t n, nk_f64_t *outputs,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(inputs), nk_unused_(n), nk_unused_(outputs), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_trig_cos_f64_best(nk_f64_t const *inputs, nk_size_t n, nk_f64_t *outputs,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(inputs), nk_unused_(n), nk_unused_(outputs), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_trig_atan_f64_best(nk_f64_t const *inputs, nk_size_t n, nk_f64_t *outputs,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(inputs), nk_unused_(n), nk_unused_(outputs), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_trig_sin_f32_best(nk_f32_t const *inputs, nk_size_t n, nk_f32_t *outputs,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(inputs), nk_unused_(n), nk_unused_(outputs), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_trig_cos_f32_best(nk_f32_t const *inputs, nk_size_t n, nk_f32_t *outputs,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(inputs), nk_unused_(n), nk_unused_(outputs), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_trig_atan_f32_best(nk_f32_t const *inputs, nk_size_t n, nk_f32_t *outputs,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(inputs), nk_unused_(n), nk_unused_(outputs), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_trig_sin_f16_best(nk_f16_t const *inputs, nk_size_t n, nk_f16_t *outputs,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(inputs), nk_unused_(n), nk_unused_(outputs), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_trig_cos_f16_best(nk_f16_t const *inputs, nk_size_t n, nk_f16_t *outputs,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(inputs), nk_unused_(n), nk_unused_(outputs), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_trig_atan_f16_best(nk_f16_t const *inputs, nk_size_t n, nk_f16_t *outputs,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(inputs), nk_unused_(n), nk_unused_(outputs), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_trigonometry_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype,
                                                    nk_capability_t capabilities, nk_kernel_punned_t *kernel,
                                                    nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = (nk_kernel_punned_t)NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_TRIGONOMETRY_H
