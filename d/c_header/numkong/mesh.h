/**
 *  @brief SIMD-accelerated Point Cloud Alignment.
 *  @file include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date June 19, 2024
 *
 *  Contains:
 *
 *  - Root Mean Square Deviation (RMSD) for rigid body superposition
 *  - Kabsch algorithm for optimal rigid body alignment (rotation only)
 *  - Umeyama algorithm for similarity transform (rotation + uniform scaling)
 *
 *  Precision policy is intentionally mixed across algorithm phases:
 *
 *  - `f64` inputs keep both the geometric transform and the scalar fit metric in `f64`
 *  - `f32` inputs keep transform outputs narrow (`a_centroid`, `b_centroid`, `rotation`, `scale`) but widen
 *    the scalar fit metric to `f64`
 *  - `f16` and `bf16` inputs keep transform and metric outputs in `f32`
 *
 *  This keeps `f32` mesh kernels materially faster than `f64` kernels by preserving narrower input bandwidth,
 *  while still widening the numerically sensitive stages that dominate alignment quality.
 *
 *  For hardware architectures:
 *
 *  - x86 (AVX2, AVX512)
 *  - Arm (NEON, SVE)
 *
 *  @section applications Applications
 *
 *  These routines are the core of point-cloud alignment pipelines:
 *
 *  - Structural biology: protein backbone or ligand alignment (RMSD, Kabsch)
 *  - Computer graphics: mesh registration and deformation transfer
 *  - Robotics/SLAM: point-cloud registration and tracking
 *
 *  @section transformation_convention Transformation Convention
 *
 *  All functions compute a transformation that aligns the FIRST point cloud (a) to the SECOND (b).
 *  The transformation to apply is:
 *
 *      a′ᵢ = scale × R × (aᵢ - ā) + b̄
 *
 *  Where:
 *
 *  - R is a 3×3 rotation matrix (row-major, 9 values)
 *  - scale is a uniform scaling factor (1.0 for RMSD and Kabsch)
 *  - ā, b̄ are the centroids of the respective point clouds
 *
 *  @section algorithm_overview Algorithm Overview
 *
 *  - RMSD: Simple root mean square deviation without alignment. R = identity, scale = 1.0
 *  - Kabsch: Finds optimal rotation R minimizing ‖R × (a - ā) - (b - b̄)‖. scale = 1.0
 *  - Umeyama: Finds optimal rotation R and scale c minimizing ‖c × R × (a - ā) - (b - b̄)‖
 *
 *  Kabsch and Umeyama compute a 3×3 cross-covariance matrix H = Σ(aᵢ - ā)(bᵢ - b̄)ᵀ
 *  and recover R from the SVD of H. Umeyama additionally estimates a uniform scale from the
 *  singular values and the variance of the centered source points.
 *
 *  The 3×3 SVD implementation is based on the McAdams et al. paper:
 *  "Computing the Singular Value Decomposition of 3×3 matrices with minimal branching
 *  and elementary floating point operations", University of Wisconsin - Madison TR1690, 2011.
 *
 *  @section numerical_notes Numerical Notes
 *
 *  Let `n` be the number of 3D points:
 *
 *  - `O(n)` stages are the point-cloud passes for centroids, cross-covariance, source variance, and transformed SSD.
 *  - `O(1)` stages are the fixed-size 3×3 SVD/eigensolve, determinant/reflection fix, and scale construction.
 *
 *  Kernel policy:
 *
 *  - `f64`: both `O(n)` reductions and the `O(1)` 3×3 solve run in `f64`.
 *  - `f32`: point coordinates load as `f32`, widen before arithmetic, keep `O(n)` reductions in `f64`,
 *    keep the `O(1)` 3×3 solve in `f64`, and only narrow public transform outputs on store.
 *  - `f16`/`bf16`: keep both `O(n)` and `O(1)` stages in `f32`.
 *
 *  - `f32` transform outputs stay narrow because they are typically applied back onto `f32` point clouds.
 *  - Reflections are handled by flipping the last singular vector when det(R) < 0.
 *  - For very small point sets, the loops are scalar-heavy and dominate over SIMD setup costs.
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  The SIMD kernels are dominated by FMA, permutes, and gathers:
 *
 *      Intrinsic                     Instruction        Notes
 *      _mm256_fmadd_ps/pd            VFMADD*            FMA on FP ports (Haswell/Skylake: ports 0/1)
 *      _mm256_i32gather_ps           VGATHERDPS         High-latency; memory-bound
 *      _mm512_permutex2var_ps/pd     VPERMT2*           Shuffle-heavy; can bottleneck on shuffle ports
 *      _mm512_reduce_add_ps/pd       (sequence)         Implemented via shuffles + adds
 *
 *  Gather-heavy tails are intentionally isolated to keep the steady-state loop on contiguous loads.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NK_MESH_H
#define NK_MESH_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief RMSD mesh superposition function.
 *
 *  The transformation aligns a to b: a′ᵢ = scale × R × (aᵢ - ā) + b̄
 *
 *  @param[in] a First point cloud (source), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] b Second point cloud (target), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] n Number of 3D points in each cloud.
 *  @param[out] a_centroid Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation Row-major 3×3 rotation matrix (9 values), always identity. Can be NULL.
 *  @param[out] scale Scale factor applied, always 1. Can be NULL.
 *  @param[out] result RMSD after applying the transformation.
 */
NK_DYNAMIC void nk_rmsd_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_rmsd_f64 */
NK_DYNAMIC void nk_rmsd_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_rmsd_f64 */
NK_DYNAMIC void nk_rmsd_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_rmsd_f64 */
NK_DYNAMIC void nk_rmsd_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/**
 *  @brief Kabsch mesh superposition function.
 *
 *  The transformation aligns a to b: a′ᵢ = scale × R × (aᵢ - ā) + b̄
 *
 *  @param[in] a First point cloud (source), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] b Second point cloud (target), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] n Number of 3D points in each cloud.
 *  @param[out] a_centroid Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation Row-major 3×3 rotation matrix (9 values). Can be NULL.
 *  @param[out] scale Scale factor applied, always 1. Can be NULL.
 *  @param[out] result RMSD after applying the transformation.
 */
NK_DYNAMIC void nk_kabsch_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                              nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_DYNAMIC void nk_kabsch_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_DYNAMIC void nk_kabsch_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f64 */
NK_DYNAMIC void nk_kabsch_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/**
 *  @brief Umeyama mesh superposition function.
 *
 *  The transformation aligns a to b: a′ᵢ = scale × R × (aᵢ - ā) + b̄
 *
 *  @param[in] a First point cloud (source), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] b Second point cloud (target), n×3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] n Number of 3D points in each cloud.
 *  @param[out] a_centroid Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation Row-major 3×3 rotation matrix (9 values). Can be NULL.
 *  @param[out] scale Scale factor applied. Can be NULL.
 *  @param[out] result RMSD after applying the transformation.
 */
NK_DYNAMIC void nk_umeyama_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                               nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_DYNAMIC void nk_umeyama_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_DYNAMIC void nk_umeyama_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f64 */
NK_DYNAMIC void nk_umeyama_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                    nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_f16 */
NK_PUBLIC void nk_rmsd_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f16 */
NK_PUBLIC void nk_kabsch_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f16 */
NK_PUBLIC void nk_umeyama_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_bf16 */
NK_PUBLIC void nk_rmsd_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_bf16 */
NK_PUBLIC void nk_kabsch_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_bf16 */
NK_PUBLIC void nk_umeyama_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer.
 */
#if NK_TARGET_SKYLAKE
/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_f16 */
NK_PUBLIC void nk_rmsd_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f16 */
NK_PUBLIC void nk_kabsch_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f16 */
NK_PUBLIC void nk_umeyama_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_rmsd_bf16 */
NK_PUBLIC void nk_rmsd_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_bf16 */
NK_PUBLIC void nk_kabsch_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_bf16 */
NK_PUBLIC void nk_umeyama_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
#endif // NK_TARGET_SKYLAKE

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer.
 */
#if NK_TARGET_HASWELL
/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_f16 */
NK_PUBLIC void nk_rmsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f16 */
NK_PUBLIC void nk_kabsch_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f16 */
NK_PUBLIC void nk_umeyama_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);

/** @copydoc nk_rmsd_bf16 */
NK_PUBLIC void nk_rmsd_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_bf16 */
NK_PUBLIC void nk_kabsch_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_bf16 */
NK_PUBLIC void nk_umeyama_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
#endif // NK_TARGET_HASWELL

/*  SIMD-powered backends for Arm NEON CPUs.
 */
#if NK_TARGET_NEON
/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);

/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
#endif // NK_TARGET_NEON

/*  SIMD-powered backends for Arm NEON FP16 CPUs.
 */
#if NK_TARGET_NEONHALF
/** @copydoc nk_rmsd_f16 */
NK_PUBLIC void nk_rmsd_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f16 */
NK_PUBLIC void nk_kabsch_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f16 */
NK_PUBLIC void nk_umeyama_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
#endif // NK_TARGET_NEONHALF

/*  SIMD-powered backends for Arm NEON BF16 CPUs.
 */
#if NK_TARGET_NEONBFDOT
/** @copydoc nk_rmsd_bf16 */
NK_PUBLIC void nk_rmsd_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_bf16 */
NK_PUBLIC void nk_kabsch_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                        nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_bf16 */
NK_PUBLIC void nk_umeyama_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_RVV
/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                               nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_rmsd_f16 */
NK_PUBLIC void nk_rmsd_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_rmsd_bf16 */
NK_PUBLIC void nk_rmsd_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                 nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                 nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f16 */
NK_PUBLIC void nk_kabsch_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                 nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_kabsch_bf16 */
NK_PUBLIC void nk_kabsch_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f16 */
NK_PUBLIC void nk_umeyama_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
/** @copydoc nk_umeyama_bf16 */
NK_PUBLIC void nk_umeyama_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result);
#endif // NK_TARGET_RVV

/*  WASM Relaxed SIMD backends using wasm_f32x4_relaxed_madd for FMA.
 */
#if NK_TARGET_V128RELAXED
/** @copydoc nk_rmsd_f32 */
NK_PUBLIC void nk_rmsd_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f32 */
NK_PUBLIC void nk_kabsch_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f32 */
NK_PUBLIC void nk_umeyama_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                          nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result);
/** @copydoc nk_rmsd_f64 */
NK_PUBLIC void nk_rmsd_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                       nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_kabsch_f64 */
NK_PUBLIC void nk_kabsch_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                         nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
/** @copydoc nk_umeyama_f64 */
NK_PUBLIC void nk_umeyama_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                          nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result);
#endif // NK_TARGET_V128RELAXED

/**
 *  @brief  Returns the metric output dtype for mesh alignment operations.
 *  Matches the C++ `mesh_metric_t` alias in types.hpp.
 */
NK_INTERNAL nk_dtype_t nk_mesh_metric_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Returns the transform output dtype for mesh alignment operations.
 *  Matches the C++ `mesh_transform_t` alias in types.hpp.
 */
NK_INTERNAL nk_dtype_t nk_mesh_transform_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/mesh/serial.h"
#include "numkong/mesh/neon.h"
#include "numkong/mesh/neonhalf.h"
#include "numkong/mesh/neonbfdot.h"
#include "numkong/mesh/haswell.h"
#include "numkong/mesh/skylake.h"
#include "numkong/mesh/rvv.h"
#include "numkong/mesh/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_rmsd_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                           nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_rmsd_f64_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_rmsd_f64_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_rmsd_f64_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_rmsd_f64_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_V128RELAXED
    nk_rmsd_f64_v128relaxed(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_rmsd_f64_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_rmsd_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_rmsd_f32_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_rmsd_f32_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_rmsd_f32_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_rmsd_f32_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_V128RELAXED
    nk_rmsd_f32_v128relaxed(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_rmsd_f32_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_rmsd_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_rmsd_f16_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_rmsd_f16_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEONHALF
    nk_rmsd_f16_neonhalf(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_rmsd_f16_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_rmsd_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_rmsd_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_rmsd_bf16_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_rmsd_bf16_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEONBFDOT
    nk_rmsd_bf16_neonbfdot(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_rmsd_bf16_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_rmsd_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_kabsch_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                             nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_kabsch_f64_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_kabsch_f64_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_kabsch_f64_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_kabsch_f64_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_V128RELAXED
    nk_kabsch_f64_v128relaxed(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_kabsch_f64_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_kabsch_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_kabsch_f32_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_kabsch_f32_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_kabsch_f32_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_kabsch_f32_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_V128RELAXED
    nk_kabsch_f32_v128relaxed(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_kabsch_f32_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_kabsch_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_kabsch_f16_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_kabsch_f16_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEONHALF
    nk_kabsch_f16_neonhalf(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_kabsch_f16_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_kabsch_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_kabsch_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_kabsch_bf16_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_kabsch_bf16_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEONBFDOT
    nk_kabsch_bf16_neonbfdot(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_kabsch_bf16_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_kabsch_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_umeyama_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                              nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_umeyama_f64_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_umeyama_f64_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_umeyama_f64_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_umeyama_f64_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_V128RELAXED
    nk_umeyama_f64_v128relaxed(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_umeyama_f64_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_umeyama_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_umeyama_f32_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_umeyama_f32_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEON
    nk_umeyama_f32_neon(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_umeyama_f32_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_V128RELAXED
    nk_umeyama_f32_v128relaxed(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_umeyama_f32_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_umeyama_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_umeyama_f16_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_umeyama_f16_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEONHALF
    nk_umeyama_f16_neonhalf(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_umeyama_f16_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_umeyama_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

NK_PUBLIC void nk_umeyama_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
#if NK_TARGET_SKYLAKE
    nk_umeyama_bf16_skylake(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_HASWELL
    nk_umeyama_bf16_haswell(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_NEONBFDOT
    nk_umeyama_bf16_neonbfdot(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#elif NK_TARGET_RVV
    nk_umeyama_bf16_rvv(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#else
    nk_umeyama_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_MESH_H
