/**
 *  @brief SIMD-accelerated Geospatial Distances.
 *  @file include/numkong/geospatial.h
 *  @author Ash Vardanian
 *  @date July 1, 2023
 *
 *  Contains following distance functions:
 *
 *  - Haversine (Great Circle) distance for 2 points
 *  - Haversine (Great Circle) distance for 2 arrays of points
 *  - Vincenty's distance function for Oblate Spheroid Geodesics
 *
 *  All outputs are in meters, and the input coordinates are in radians.
 *
 *  For dtypes:
 *
 *  - 64-bit IEEE-754 floating point → 64-bit
 *  - 32-bit IEEE-754 floating point → 32-bit
 *
 *  Precision policy:
 *
 *  - `f32` remains the throughput-oriented lane and intentionally stays narrow end-to-end.
 *  - `f64` is the higher-accuracy lane for the same formulas.
 *  - We do not widen `f32` outputs here because the dominant error comes from the geodesic model
 *    and transcendental approximations, not from long horizontal reductions.
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Skylake
 *
 *  @section haversine_similarity Low-Accuracy High-Performance Haversine Similarity
 *
 *  In most cases, for distance computations, we don't need the exact Haversine formula.
 *  The very last part of the computation applies `asin(√x)` non-linear transformation.
 *  Both `asin` and `sqrt` are monotonically increasing functions, so their product is also
 *  monotonically increasing. This means, for relative similarity/closeness computation we
 *  can avoid that expensive last step.
 *
 *  @section trig_approximations Trigonometric Approximations & SIMD Vectorization
 *
 *  The trigonometric functions (sin, cos, atan2) use polynomial approximations with SLEEF-level
 *  error bounds (~3.5 ULP). For f64, this translates to ~1e-15 absolute error; for f32, ~1e-7.
 *
 *  @section accuracy_comparison Accuracy Comparison: Haversine vs Vincenty
 *
 *  Both algorithms compute geodesic distances, but with different Earth models:
 *
 *  - Haversine: Sphere (R=6335km), 0.3% - 0.6% vs WGS-84, fast approximation, ranking
 *  - Vincenty: WGS-84 Ellipsoid, 0.01% - 0.2% vs WGS-84, high-precision navigation
 *
 *  Vincenty is ~3-20x more accurate than Haversine for most routes. The improvement is most
 *  significant for long-distance routes and near-polar paths where Earth's oblateness matters.
 *
 *  @note   SIMD implementations may have slightly different results than serial due to
 *          floating-point ordering in iterative algorithms. For Vincenty, expect <0.001%
 *          difference between SIMD and serial implementations.
 *
 *  @section vincenty_precision High-Precision Vincenty's Formulae & Earth Ellipsoid
 *
 *  Several approximations of the Earth Ellipsoid exist, each defined by the Equatorial radius (m),
 *  Polar radius (m), and Inverse flattening. The earliest ones date back to 1738, when Pierre Louis
 *  Maupertuis in France suggested a shape, that is only 0.3% different from the most accurate modern
 *  estimates by the International Earth Rotation and Reference Systems Service (IERS).
 *  The Global Positioning System (GPS) uses the World Geodetic Systems's (WGS) WGS-84 standard.
 *  NumKong uses the newer & more accurate @b IERS-2003 standard, but allows overriding default parameters:
 *
 *      #define NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS (6378136.6)
 *      #define NK_EARTH_ELLIPSOID_POLAR_RADIUS (6356751.9)
 *      #define NK_EARTH_ELLIPSOID_INVERSE_FLATTENING (298.25642)
 *
 *  To revert from oblate spheroids to spheres, use `NK_EARTH_MEDIATORIAL_RADIUS`.
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  Haversine and Vincenty formulas require sqrt for the final distance calculation and division
 *  for Vincenty's iterative convergence. These are the most expensive operations (12-23 cycles)
 *  but only execute once per point-pair. The polynomial trig approximations use FMA chains.
 *  Note: ZMM sqrt is faster on Genoa (15c) than Ice Lake (19c) due to better 512-bit support.
 *
 *      Intrinsic               Instruction                     Ice         Genoa
 *      _mm256_sqrt_ps          VSQRTPS (YMM, YMM)              12c @ p0    15c @ p01
 *      _mm256_sqrt_pd          VSQRTPD (YMM, YMM)              13c @ p0    21c @ p01
 *      _mm512_sqrt_ps          VSQRTPS (ZMM, ZMM)              19c @ p05   15c @ p01
 *      _mm512_sqrt_pd          VSQRTPD (ZMM, ZMM)              23c @ p05   21c @ p01
 *      _mm256_div_ps           VDIVPS (YMM, YMM, YMM)          11c @ p0    11c @ p01
 *      _mm256_div_pd           VDIVPD (YMM, YMM, YMM)          13c @ p0    13c @ p01
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *      _mm256_fmadd_pd         VFMADD231PD (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *
 *  @section arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  ARM sqrt (FSQRT) has low throughput as it uses a dedicated V02 execution unit. This is
 *  acceptable since sqrt only appears once per distance calculation. FMA chains for trig
 *  polynomial evaluation pipeline well across all 4 V-units.
 *
 *      Intrinsic               Instruction     M1 Firestorm    Graviton 3      Graviton 4
 *      vfmaq_f32               FMLA.S (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vfmaq_f64               FMLA.D (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vsqrtq_f32              FSQRT.S (vec)   10c @ V02       10c @ V02       9c @ V02
 *      vsqrtq_f64              FSQRT.D (vec)   13c @ V02       16c @ V02       16c @ V02
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  - Earth Ellipsoid: https://en.wikipedia.org/wiki/Earth_ellipsoid
 *  - Oblate Spheroid Geodesic: https://mathworld.wolfram.com/OblateSpheroidGeodesic.html
 *  - Speeding up atan2f by 50x: https://mazzo.li/posts/vectorized-atan2.html
 *  - Simplifying the GNU C Sine Function:
 *    https://web.archive.org/web/20230605051610/https://www.awelm.com/posts/simplifying-the-gnu-c-sine-function/
 *
 */
#ifndef NK_GEOSPATIAL_H
#define NK_GEOSPATIAL_H

#include "numkong/types.h"
#include "numkong/trigonometry.h"

/*  Earth Ellipsoid Constants
 *  The default values use the IERS-2003 standard, but can be overridden before including this header.
 */
#ifndef NK_EARTH_MEDIATORIAL_RADIUS
#define NK_EARTH_MEDIATORIAL_RADIUS (6335439.0)
#endif
#ifndef NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS
#define NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS (6378136.6)
#endif
#ifndef NK_EARTH_ELLIPSOID_POLAR_RADIUS
#define NK_EARTH_ELLIPSOID_POLAR_RADIUS (6356751.9)
#endif
#ifndef NK_EARTH_ELLIPSOID_INVERSE_FLATTENING
#define NK_EARTH_ELLIPSOID_INVERSE_FLATTENING (298.25642)
#endif
#ifndef NK_VINCENTY_MAX_ITERATIONS
#define NK_VINCENTY_MAX_ITERATIONS 100
#endif
#ifndef NK_VINCENTY_CONVERGENCE_THRESHOLD_F64
#define NK_VINCENTY_CONVERGENCE_THRESHOLD_F64 1e-12
#endif
#ifndef NK_VINCENTY_CONVERGENCE_THRESHOLD_F32
#define NK_VINCENTY_CONVERGENCE_THRESHOLD_F32 1e-7f
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Haversine distance between two arrays of points on a sphere.
 *
 *  @param[in] a_lats Latitudes of the first points, in radians.
 *  @param[in] a_lons Longitudes of the first points, in radians.
 *  @param[in] b_lats Latitudes of the second points, in radians.
 *  @param[in] b_lons Longitudes of the second points, in radians.
 *  @param[in] n The number of point pairs.
 *  @param[out] results Output distances in meters, length `n`.
 *
 *  @note Inputs are in radians and outputs are in meters.
 */
NK_DYNAMIC void nk_haversine_f64(                   //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);

/** @copydoc nk_haversine_f64 */
NK_DYNAMIC void nk_haversine_f32(                   //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);

/**
 *  @brief Vincenty distance between two arrays of points on an oblate spheroid.
 *
 *  @param[in] a_lats Latitudes of the first points, in radians.
 *  @param[in] a_lons Longitudes of the first points, in radians.
 *  @param[in] b_lats Latitudes of the second points, in radians.
 *  @param[in] b_lons Longitudes of the second points, in radians.
 *  @param[in] n The number of point pairs.
 *  @param[out] results Output distances in meters, length `n`.
 *
 *  @note Inputs are in radians and outputs are in meters.
 *  @note Uses the Earth ellipsoid parameters configured via `NK_EARTH_ELLIPSOID_*`.
 */
NK_DYNAMIC void nk_vincenty_f64(                    //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);

/** @copydoc nk_vincenty_f64 */
NK_DYNAMIC void nk_vincenty_f32(                    //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);

/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_serial(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_serial(              //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_serial(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_serial(              //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);

#if NK_TARGET_NEON
/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_neon(               //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_neon(                //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_neon(               //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_neon(                //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_haswell(            //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_haswell(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_haswell(            //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_haswell(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_skylake(            //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_skylake(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_skylake(            //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_skylake(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_V128RELAXED
/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_v128relaxed(        //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_v128relaxed(         //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_v128relaxed(        //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_v128relaxed(         //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_RVV
/** @copydoc nk_haversine_f64 */
NK_PUBLIC void nk_haversine_f64_rvv(                //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_vincenty_f64 */
NK_PUBLIC void nk_vincenty_f64_rvv(                 //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results);
/** @copydoc nk_haversine_f32 */
NK_PUBLIC void nk_haversine_f32_rvv(                //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
/** @copydoc nk_vincenty_f32 */
NK_PUBLIC void nk_vincenty_f32_rvv(                 //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results);
#endif // NK_TARGET_RVV

/**
 *  @brief  Returns the output dtype for Haversine distance.
 */
NK_INTERNAL nk_dtype_t nk_haversine_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Returns the output dtype for Vincenty distance.
 */
NK_INTERNAL nk_dtype_t nk_vincenty_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/geospatial/serial.h"
#include "numkong/geospatial/neon.h"
#include "numkong/geospatial/haswell.h"
#include "numkong/geospatial/skylake.h"
#include "numkong/geospatial/v128relaxed.h"
#include "numkong/geospatial/rvv.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_haversine_f64(                    //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {
#if NK_TARGET_SKYLAKE
    nk_haversine_f64_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_HASWELL
    nk_haversine_f64_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_NEON
    nk_haversine_f64_neon(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_V128RELAXED
    nk_haversine_f64_v128relaxed(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_RVV
    nk_haversine_f64_rvv(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    nk_haversine_f64_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

NK_PUBLIC void nk_haversine_f32(                    //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {
#if NK_TARGET_SKYLAKE
    nk_haversine_f32_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_HASWELL
    nk_haversine_f32_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_NEON
    nk_haversine_f32_neon(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_V128RELAXED
    nk_haversine_f32_v128relaxed(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_RVV
    nk_haversine_f32_rvv(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    nk_haversine_f32_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

NK_PUBLIC void nk_vincenty_f64(                     //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {
#if NK_TARGET_SKYLAKE
    nk_vincenty_f64_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_HASWELL
    nk_vincenty_f64_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_NEON
    nk_vincenty_f64_neon(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_V128RELAXED
    nk_vincenty_f64_v128relaxed(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_RVV
    nk_vincenty_f64_rvv(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    nk_vincenty_f64_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

NK_PUBLIC void nk_vincenty_f32(                     //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {
#if NK_TARGET_SKYLAKE
    nk_vincenty_f32_skylake(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_HASWELL
    nk_vincenty_f32_haswell(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_NEON
    nk_vincenty_f32_neon(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_V128RELAXED
    nk_vincenty_f32_v128relaxed(a_lats, a_lons, b_lats, b_lons, n, results);
#elif NK_TARGET_RVV
    nk_vincenty_f32_rvv(a_lats, a_lons, b_lats, b_lons, n, results);
#else
    nk_vincenty_f32_serial(a_lats, a_lons, b_lats, b_lons, n, results);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif
