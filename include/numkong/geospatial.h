/**
 *  @file include/numkong/geospatial.h
 *  @author Ash Vardanian
 *  @date July 1, 2023
 *  @brief SIMD-accelerated many-to-many geospatial distances and terrain visibility.
 *
 *  Contains following kernel families, each scoring every row against every column:
 *
 *  - @c sqeuclideans3d: squared Euclidean distances between 3D points, exact on @c i32 coordinates
 *  - @c haversines: great-circle metres on the mean sphere, over packed unit vectors
 *  - @c vincentys: geodesic metres on the WGS-84 ellipsoid, over packed reduced latitudes
 *  - @c clearances: the margin by which every sight line clears a terrain grid, over packed maxima
 *
 *  Angles are radians, distances and heights are metres. Outputs land in a @b [rows,columns] matrix
 *  with a byte stride between its rows.
 *
 *  For dtypes:
 *
 *  - @c sqeuclideans3d: 32-bit integers → 64-bit unsigned; 64-bit IEEE-754 → 64-bit
 *  - @c haversines and @c vincentys: 64-bit IEEE-754 → 64-bit
 *  - terrain: 32-bit IEEE-754 heights and positions → 32-bit clearances, computed in 64-bit
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Skylake
 *  - RISC-V: RVV
 *  - WebAssembly: Relaxed SIMD
 *
 *  The terrain kernels are serial for now.
 *
 *  @section geospatial_packing Packing
 *
 *  @c haversines and @c vincentys take both sides as packs, built once per point set by one serial
 *  routine for every capability, so a pack is byte-identical on every machine. A Haversine pack
 *  holds unit vectors on the sphere, a Vincenty pack the longitude and the sine and cosine of the
 *  reduced latitude, so no many-to-many kernel evaluates trigonometry before its work per pair.
 *  @c sqeuclideans3d needs no pack: at depth 3 the difference form needs no norms, and the SIMD
 *  bodies de-interleave plain @b [points,3] rows on the fly.
 *
 *  A terrain pack holds only the maxima of every 2ᵏ × 2ᵏ block of a grid and the grid's
 *  shape, never the heights themselves, which stay in the caller's buffer and are named again
 *  by every call. A raster reaches tens of gigabytes, so it is borrowed rather than copied.
 *
 *  @section geospatial_precision Precision
 *
 *  - @c sqeuclideans3d on @c i32 is exact whenever every axis difference fits 32 bits, which
 *    Earth-fixed centimetres below 4,359 km of altitude always satisfy; on @c f64 the three squares
 *    are summed in Dot2.
 *  - @c haversines computes 2R · atan2(‖a − b‖, ‖a + b‖) over unit vectors, summing
 *    both norms in Dot2. Unlike 2 · asin(√h) or 2 · atan2(√h, √(1 − h)), this stays
 *    conditioned at every separation, from a millimetre to the antipode.
 *  - @c vincentys iterates to @c NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64; a pair still
 *    unsettled after @c NUMKONG_VINCENTY_MAX_ITERATIONS, which happens only near the antipode,
 *    reports its last iterate.
 *
 *  The trigonometric functions use polynomial approximations with SLEEF-level error bounds,
 *  about 3.5 ULP, so about 1e-15 absolute error in @c f64.
 *
 *  @section geospatial_earth Earth Models
 *
 *  @c haversines uses the mean radius R₁ of the International Union of Geodesy and Geophysics,
 *  6,371,008.8 m, which reads within 0.5% of the ellipsoidal geodesic. @c vincentys uses WGS-84,
 *  the ellipsoid GPS reports on. Both can be overridden before this header is included, and the
 *  polar radius always derives from the other two:
 *
 *  @code{.c}
 *  #define NUMKONG_EARTH_MEAN_RADIUS (6371008.8)
 *  #define NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS (6378137.0)
 *  #define NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING (298.257223563)
 *  @endcode
 *
 *  @section geospatial_clearance Clearance
 *
 *  A viewpoint and a target are @b [column,row,height] triples: fractional cell coordinates of a
 *  north-up grid, with row 0 at its north edge, and heights in metres above the grid's datum. Their
 *  ground distance D is their offset in cells, scaled by @c column_metres and @c row_metres along
 *  each axis. The curvature lowers the ground at a distance s by drop(s), the square of s over
 *  twice the mean radius, scaled by one minus the refraction; pass 1/7 as the refraction of light,
 *  0.25 for radio's 4/3-radius model, and 0 to ignore it. The sight line runs straight from the
 *  viewpoint's height to the target's, lowered by drop(D).
 *
 *  The terrain is sampled wherever the open segment from viewpoint to target crosses a column line
 *  or a row line through cell centres: each sample interpolates linearly between the two heights on
 *  that line, which evaluates the bilinear surface exactly on its edges, and is lowered by drop(s).
 *  Missing cells, which are NaN, never block. The clearance is the least gap between the sight line
 *  and the sampled terrain over all samples:
 *
 *  - non-negative when the target is visible, by that many metres
 *  - negative when it is blocked, by the deepest obstruction, exactly
 *  - +∞ when the segment crosses no line, as between two points of one cell
 *  - −∞ when the target falls outside the viewpoint's view or beyond its range
 *  - NaN when either end lies off the grid
 *
 *  Being a minimum over a fixed set of samples, the result does not depend on the order samples are
 *  visited in, so every capability returns the same clearance. A view is a circular cone of a full
 *  @c field angle around an @c azimuth, clockwise from the grid's north, and an @c elevation above
 *  the horizon, reaching @c range metres along the slant; a NaN azimuth looks everywhere, a NaN
 *  elevation tests the azimuth alone, and an infinite or NaN range reaches without limit. The
 *  pack's maxima only let a traversal skip blocks that cannot lower the running minimum, so they
 *  change the cost and never the result.
 *
 *  @section geospatial_x86_instructions Relevant x86 Instructions
 *
 *  The 3D kernels square 32-bit differences with @c VPMULDQ, which multiplies the even 32-bit lanes
 *  into 64-bit products; scalar x86 has one 64-bit multiplier, so the vector form wins about 3×.
 *  The spherical and ellipsoidal kernels spend most of their time in square roots, divisions and
 *  the FMA chains of the trigonometric polynomials.
 *
 *  @verbatim
 *  Intrinsic        Instruction                  Icelake           Genoa
 *  _mm512_mul_epi32 VPMULDQ (ZMM, ZMM, ZMM)      5cy @ p0          3cy @ p01
 *  _mm256_sqrt_pd   VSQRTPD (YMM, YMM)           13cy @ p0         21cy @ p01
 *  _mm512_sqrt_pd   VSQRTPD (ZMM, ZMM)           23cy @ p0+p0+p05  21cy @ p01
 *  _mm256_div_pd    VDIVPD (YMM, YMM, YMM)       13cy @ p0         13cy @ p01
 *  _mm256_fmadd_pd  VFMADD231PD (YMM, YMM, YMM)  4cy @ p01         4cy @ p01
 *  @endverbatim
 *
 *  @section geospatial_arm_instructions Relevant Arm NEON Instructions
 *
 *  @c LD3 de-interleaves four 3D points into three registers, and @c SMULL/SMLAL square 32-bit
 *  differences into 64-bit sums. Square roots run on a dedicated unit with low throughput, which is
 *  acceptable as they appear once per pair.
 *
 *  @verbatim
 *  Intrinsic        Instruction    M1 Firestorm  Graviton 3   Graviton 4
 *  vmlal_s32        SMLAL (vec)    3cy @ V0123   4cy @ V02    4cy @ V02
 *  vfmaq_f64        FMLA.D (vec)   4cy @ V0123   4cy @ V0123  4cy @ V0123
 *  vsqrtq_f64       FSQRT.D (vec)  13cy @ V02    16cy @ V02   16cy @ V02
 *  @endverbatim
 *
 *  @section geospatial_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  @see Earth Ellipsoid: https://en.wikipedia.org/wiki/Earth_ellipsoid
 *  @see Oblate Spheroid Geodesic: https://mathworld.wolfram.com/OblateSpheroidGeodesic.html
 *  @see Vincenty's inverse solution on the ellipsoid: https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf
 *  @see Kahan's needle-like triangles, for the atan2 form: https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf
 *  @see Maximum mipmaps for height fields, for the maxima pyramid: https://doi.org/10.1145/1342250.1342279
 *  @see Speeding up atan2f by 50x: https://mazzo.li/posts/vectorized-atan2.html
 *  @see Simplifying the GNU C Sine Function: https://web.archive.org/web/20230605051610/https://www.awelm.com/posts/simplifying-the-gnu-c-sine-function/
 */
#ifndef NUMKONG_GEOSPATIAL_H
#define NUMKONG_GEOSPATIAL_H

#include "numkong/capabilities.h"

/** The IUGG mean radius R₁, in metres, of the Haversine sphere and the terrain curvature. */
#ifndef NUMKONG_EARTH_MEAN_RADIUS
#define NUMKONG_EARTH_MEAN_RADIUS (6371008.8)
#endif

/** The WGS-84 semi-major axis, in metres; the polar radius derives as a × (1 − f). */
#ifndef NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS
#define NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS (6378137.0)
#endif

/** The WGS-84 inverse flattening. */
#ifndef NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING
#define NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING (298.257223563)
#endif
#ifndef NUMKONG_VINCENTY_MAX_ITERATIONS
#define NUMKONG_VINCENTY_MAX_ITERATIONS 100
#endif
#ifndef NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64
#define NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64 1e-12
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Squared Euclidean distances between every row point and every column point in 3D.
 *
 *  Integer results are exact whenever every axis difference fits 32 bits, which Earth-fixed
 *  centimetres below 4,359 km of altitude always satisfy. F64 inputs sum their squares in Dot2.
 *
 *  @param[in] a The rows, @p rows × 3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] rows The number of row points.
 *  @param[in] b The columns, @p columns × 3 interleaved.
 *  @param[in] columns The number of column points.
 *  @param[out] results The @b [rows,columns] squared distances.
 *  @param[in] results_stride The bytes between consecutive rows of @p results.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_best(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                   nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                   nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_best(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                   nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                   nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Returns the bytes a Haversine or Vincenty pack of @p points points takes.
 *  @param[in] points The number of points.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[out] bytes The size of the pack.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_haversines_pack_size_f64_best(nk_size_t points, nk_capability_t capabilities,
                                                         nk_size_t *bytes);
/** @copydoc nk_haversines_pack_size_f64_best */
NUMKONG_API nk_status_t nk_vincentys_pack_size_f64_best(nk_size_t points, nk_capability_t capabilities,
                                                        nk_size_t *bytes);

/**
 *  @brief Packs points as unit vectors on the mean sphere, for @c nk_haversines_packed_f64_best.
 *  @param[in] latitudes Latitudes, in radians.
 *  @param[in] longitudes Longitudes, in radians.
 *  @param[in] count The number of points.
 *  @param[out] packed The pack, on an @c nk_default_alignment_k boundary.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_haversines_pack_f64_best(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                    nk_size_t count, void *packed, nk_capability_t capabilities,
                                                    nk_stream_t stream);

/** Packs points as their longitudes and the sines and cosines of their reduced latitudes on the
 *  ellipsoid, for @c nk_vincentys_packed_f64_best; parameters as @c nk_haversines_pack_f64_best. */
NUMKONG_API nk_status_t nk_vincentys_pack_f64_best(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                   nk_size_t count, void *packed, nk_capability_t capabilities,
                                                   nk_stream_t stream);

/**
 *  @brief Great-circle metres between every row and every column point on the mean sphere.
 *  @param[in] a_packed The row points, as @c nk_haversines_pack_f64_best packed them.
 *  @param[in] b_packed The column points, packed the same way.
 *  @param[out] results The @b [rows,columns] distances.
 *  @param[in] results_stride The bytes between consecutive rows of @p results.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_haversines_packed_f64_best(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_capability_t capabilities,
                                                      nk_stream_t stream);

/** Geodesic metres on the WGS-84 ellipsoid between every row and every column point, from packs
 *  of @c nk_vincentys_pack_f64_best; parameters as @c nk_haversines_packed_f64_best. */
NUMKONG_API nk_status_t nk_vincentys_packed_f64_best(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_capability_t capabilities,
                                                     nk_stream_t stream);

/**
 *  @brief Returns the bytes the terrain pack of a @p rows × @p columns grid takes.
 *  @param[in] rows The grid rows.
 *  @param[in] columns The grid columns.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[out] bytes The size of the pack.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_terrain_pack_size_f32_best(nk_size_t rows, nk_size_t columns, nk_capability_t capabilities,
                                                      nk_size_t *bytes);

/**
 *  @brief Packs the maxima of every 2ᵏ × 2ᵏ block of a north-up grid, its shape and scale.
 *  @param[in] heights The @b [rows,columns] heights in metres, row 0 at the north edge; NaN marks a
 *      missing cell. Every later call names the same buffer again.
 *  @param[in] rows The grid rows.
 *  @param[in] columns The grid columns.
 *  @param[in] heights_stride The bytes between consecutive rows of @p heights.
 *  @param[in] row_metres The ground length of one row step, constant over the grid.
 *  @param[in] column_metres The ground length of one column step, constant over the grid.
 *  @param[out] packed The pack, on an @c nk_default_alignment_k boundary.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_terrain_pack_f32_best(nk_f32_t const *heights, nk_size_t rows, nk_size_t columns,
                                                 nk_size_t heights_stride, nk_f64_t row_metres, nk_f64_t column_metres,
                                                 void *packed, nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Bilinear terrain heights under @p count points.
 *  @param[in] heights The grid @p packed was built from.
 *  @param[in] packed The grid's terrain pack.
 *  @param[in] points The @p count × 2 interleaved [column,row] cell coordinates of the points.
 *  @param[in] count The number of points.
 *  @param[out] results The @p count heights in metres, NaN off the grid or beside a missing cell.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_terrain_heights_packed_f32_best(nk_f32_t const *heights, void const *packed,
                                                           nk_f32_t const *points, nk_size_t count, nk_f32_t *results,
                                                           nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Clearances in metres from every row viewpoint to every column target over one terrain, as
 *      @ref geospatial_clearance defines them.
 *  @param[in] heights The grid @p packed was built from.
 *  @param[in] packed The grid's terrain pack.
 *  @param[in] viewpoints The rows, @p rows × 3 interleaved [column, row, height].
 *  @param[in] views Null, seeing everywhere without limit, or @p rows × 4 interleaved [azimuth,
 *      elevation, field, range]: radians clockwise from the grid's north and above the horizon,
 *      the full cone angle in radians, and metres.
 *  @param[in] rows The number of viewpoints.
 *  @param[in] targets The columns, @p columns × 3 interleaved [column, row, height].
 *  @param[in] columns The number of targets.
 *  @param[in] refraction The refraction coefficient: 1/7 for light, 0.25 for radio, 0 for none.
 *  @param[out] results The @b [rows,columns] clearances.
 *  @param[in] results_stride The bytes between consecutive rows of @p results.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_clearances_packed_f32_best(nk_f32_t const *heights, void const *packed,
                                                      nk_f32_t const *viewpoints, nk_f32_t const *views, nk_size_t rows,
                                                      nk_f32_t const *targets, nk_size_t columns, nk_f64_t refraction,
                                                      nk_f32_t *results, nk_size_t results_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream);

/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_serial(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                     nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                     nk_stream_t stream);
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_serial(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                     nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                     nk_stream_t stream);
/** @copydoc nk_haversines_pack_size_f64_best */
NUMKONG_API nk_status_t nk_haversines_pack_size_f64_serial(nk_size_t points, nk_size_t *bytes);
/** @copydoc nk_haversines_pack_size_f64_best */
NUMKONG_API nk_status_t nk_vincentys_pack_size_f64_serial(nk_size_t points, nk_size_t *bytes);
/** @copydoc nk_haversines_pack_f64_best */
NUMKONG_API nk_status_t nk_haversines_pack_f64_serial(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                      nk_size_t count, void *packed, nk_stream_t stream);
/** @copydoc nk_vincentys_pack_f64_best */
NUMKONG_API nk_status_t nk_vincentys_pack_f64_serial(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                     nk_size_t count, void *packed, nk_stream_t stream);
/** @copydoc nk_haversines_packed_f64_best */
NUMKONG_API nk_status_t nk_haversines_packed_f64_serial(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                        nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_vincentys_packed_f64_best */
NUMKONG_API nk_status_t nk_vincentys_packed_f64_serial(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                       nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_terrain_pack_size_f32_best */
NUMKONG_API nk_status_t nk_terrain_pack_size_f32_serial(nk_size_t rows, nk_size_t columns, nk_size_t *bytes);
/** @copydoc nk_terrain_pack_f32_best */
NUMKONG_API nk_status_t nk_terrain_pack_f32_serial(nk_f32_t const *heights, nk_size_t rows, nk_size_t columns,
                                                   nk_size_t heights_stride, nk_f64_t row_metres,
                                                   nk_f64_t column_metres, void *packed, nk_stream_t stream);
/** @copydoc nk_terrain_heights_packed_f32_best */
NUMKONG_API nk_status_t nk_terrain_heights_packed_f32_serial(nk_f32_t const *heights, void const *packed,
                                                             nk_f32_t const *points, nk_size_t count, nk_f32_t *results,
                                                             nk_stream_t stream);
/** @copydoc nk_clearances_packed_f32_best */
NUMKONG_API nk_status_t nk_clearances_packed_f32_serial(nk_f32_t const *heights, void const *packed,
                                                        nk_f32_t const *viewpoints, nk_f32_t const *views,
                                                        nk_size_t rows, nk_f32_t const *targets, nk_size_t columns,
                                                        nk_f64_t refraction, nk_f32_t *results,
                                                        nk_size_t results_stride, nk_stream_t stream);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_neon(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                   nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                   nk_stream_t stream);
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_neon(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                   nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                   nk_stream_t stream);
/** @copydoc nk_haversines_packed_f64_best */
NUMKONG_API nk_status_t nk_haversines_packed_f64_neon(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_vincentys_packed_f64_best */
NUMKONG_API nk_status_t nk_vincentys_packed_f64_neon(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_haswell(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                      nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                      nk_stream_t stream);
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_haswell(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                      nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                      nk_stream_t stream);
/** @copydoc nk_haversines_packed_f64_best */
NUMKONG_API nk_status_t nk_haversines_packed_f64_haswell(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                         nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_vincentys_packed_f64_best */
NUMKONG_API nk_status_t nk_vincentys_packed_f64_haswell(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                        nk_size_t results_stride, nk_stream_t stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_skylake(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                      nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                      nk_stream_t stream);
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_skylake(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                      nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                      nk_stream_t stream);
/** @copydoc nk_haversines_packed_f64_best */
NUMKONG_API nk_status_t nk_haversines_packed_f64_skylake(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                         nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_vincentys_packed_f64_best */
NUMKONG_API nk_status_t nk_vincentys_packed_f64_skylake(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                        nk_size_t results_stride, nk_stream_t stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_v128relaxed(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                          nk_size_t columns, nk_u64_t *results,
                                                          nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_v128relaxed(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                          nk_size_t columns, nk_f64_t *results,
                                                          nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_haversines_packed_f64_best */
NUMKONG_API nk_status_t nk_haversines_packed_f64_v128relaxed(void const *a_packed, void const *b_packed,
                                                             nk_f64_t *results, nk_size_t results_stride,
                                                             nk_stream_t stream);
/** @copydoc nk_vincentys_packed_f64_best */
NUMKONG_API nk_status_t nk_vincentys_packed_f64_v128relaxed(void const *a_packed, void const *b_packed,
                                                            nk_f64_t *results, nk_size_t results_stride,
                                                            nk_stream_t stream);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_RVV
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_rvv(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                  nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                  nk_stream_t stream);
/** @copydoc nk_sqeuclideans3d_i32_best */
NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_rvv(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                  nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                  nk_stream_t stream);
/** @copydoc nk_haversines_packed_f64_best */
NUMKONG_API nk_status_t nk_haversines_packed_f64_rvv(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_vincentys_packed_f64_best */
NUMKONG_API nk_status_t nk_vincentys_packed_f64_rvv(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_stream_t stream);
#endif // NUMKONG_TARGET_RVV

/** Returns the output dtype of @c sqeuclideans3d: exact @c u64 for @c i32, @c f64 for @c f64. */
NUMKONG_INLINE nk_dtype_t nk_sqeuclideans3d_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_i32_k: return nk_u64_k;
    case nk_f64_k: return nk_f64_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the output dtype of @c haversines and @c vincentys. */
NUMKONG_INLINE nk_dtype_t nk_geodesics_output_dtype(nk_dtype_t dtype) {
    return dtype == nk_f64_k ? nk_f64_k : nk_dtype_unknown_k;
}

/** Returns the output dtype of @c clearances. */
NUMKONG_INLINE nk_dtype_t nk_clearances_output_dtype(nk_dtype_t dtype) {
    return dtype == nk_f32_k ? nk_f32_k : nk_dtype_unknown_k;
}

/** Returns the error bound of a Haversine central angle in radians, relative to one plus the angle:
 *  3 sines and cosines per unit vector at 3.5 ULP, the Dot2 norms, 2 roots and the arctangent. */
NUMKONG_INLINE nk_f64_t nk_haversines_error_bound(nk_dtype_t dtype) { return 16 * nk_accumulation_error_bound(dtype); }

/** Returns the error bound of a Vincenty angular distance, in radians: the longitude the
 *  iteration stops within, and 20 roundings on the longest path from the reduced latitudes
 *  through one iteration and the series. */
NUMKONG_INLINE nk_f64_t nk_vincentys_error_bound(nk_dtype_t dtype) {
    return NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64 + 20 * nk_accumulation_error_bound(dtype);
}

/**
 *  @brief Finds the geospatial kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when no capability in @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_geospatial_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                                  nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/geospatial/serial.h"
#include "numkong/geospatial/neon.h"
#include "numkong/geospatial/haswell.h"
#include "numkong/geospatial/skylake.h"
#include "numkong/geospatial/v128relaxed.h"
#include "numkong/geospatial/rvv.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_best(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                   nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                   nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(rows), nk_unused_(b), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_best(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                   nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                   nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(rows), nk_unused_(b), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_haversines_pack_size_f64_best(nk_size_t points, nk_capability_t capabilities,
                                                         nk_size_t *bytes) {
    nk_unused_(points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_vincentys_pack_size_f64_best(nk_size_t points, nk_capability_t capabilities,
                                                        nk_size_t *bytes) {
    nk_unused_(points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_haversines_pack_f64_best(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                    nk_size_t count, void *packed, nk_capability_t capabilities,
                                                    nk_stream_t stream) {
    nk_unused_(latitudes), nk_unused_(longitudes), nk_unused_(count), nk_unused_(packed), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_vincentys_pack_f64_best(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                   nk_size_t count, void *packed, nk_capability_t capabilities,
                                                   nk_stream_t stream) {
    nk_unused_(latitudes), nk_unused_(longitudes), nk_unused_(count), nk_unused_(packed), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_haversines_packed_f64_best(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_capability_t capabilities,
                                                      nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(b_packed), nk_unused_(results), nk_unused_(results_stride),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_vincentys_packed_f64_best(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_capability_t capabilities,
                                                     nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(b_packed), nk_unused_(results), nk_unused_(results_stride),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_terrain_pack_size_f32_best(nk_size_t rows, nk_size_t columns, nk_capability_t capabilities,
                                                      nk_size_t *bytes) {
    nk_unused_(rows), nk_unused_(columns), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_terrain_pack_f32_best(nk_f32_t const *heights, nk_size_t rows, nk_size_t columns,
                                                 nk_size_t heights_stride, nk_f64_t row_metres, nk_f64_t column_metres,
                                                 void *packed, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(heights), nk_unused_(rows), nk_unused_(columns), nk_unused_(heights_stride), nk_unused_(row_metres),
        nk_unused_(column_metres), nk_unused_(packed), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_terrain_heights_packed_f32_best(nk_f32_t const *heights, void const *packed,
                                                           nk_f32_t const *points, nk_size_t count, nk_f32_t *results,
                                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(heights), nk_unused_(packed), nk_unused_(points), nk_unused_(count), nk_unused_(results),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_clearances_packed_f32_best(nk_f32_t const *heights, void const *packed,
                                                      nk_f32_t const *viewpoints, nk_f32_t const *views, nk_size_t rows,
                                                      nk_f32_t const *targets, nk_size_t columns, nk_f64_t refraction,
                                                      nk_f32_t *results, nk_size_t results_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(heights), nk_unused_(packed), nk_unused_(viewpoints), nk_unused_(views), nk_unused_(rows),
        nk_unused_(targets), nk_unused_(columns), nk_unused_(refraction), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_geospatial_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                                  nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = (nk_kernel_punned_t)NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif
