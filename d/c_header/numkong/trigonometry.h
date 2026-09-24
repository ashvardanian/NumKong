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
 *      Intrinsic               Instruction                     Ice         Genoa
 *      _mm512_roundscale_ps    VRNDSCALEPS (ZMM, ZMM, I8)      8c @ p0     3c @ p23
 *      _mm512_roundscale_pd    VRNDSCALEPD (ZMM, ZMM, I8)      8c @ p0     3c @ p23
 *      _mm512_fpclass_ps_mask  VFPCLASSPS (K, ZMM, I8)         3c @ p5     5c @ p01
 *      _mm512_fmadd_ps         VFMADD231PS (ZMM, ZMM, ZMM)     4c @ p0     4c @ p01
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *      _mm256_div_ps           VDIVPS (YMM, YMM, YMM)          ~14c @ p0   ~11c @ p01
 *      _mm256_div_pd           VDIVPD (YMM, YMM, YMM)          ~23c @ p0   ~13c @ p01
 *
 *  @section arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  ARM implementations use the same Horner polynomial approach with FMLA chains. FRINTA provides
 *  fast rounding for range reduction. The 4-cycle FMA latency with 4 inst/cycle throughput allows
 *  excellent pipelining when processing multiple elements.
 *
 *      Intrinsic               Instruction     M1 Firestorm    Graviton 3      Graviton 4
 *      vfmaq_f32               FMLA.S (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vfmaq_f64               FMLA.D (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vrndaq_f32              FRINTA.S        2c @ V0123      2c @ V01        2c @ V01
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

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TRIGONOMETRY_H
