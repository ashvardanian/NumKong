/**
 *  @file include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date June 19, 2024
 *  @brief SIMD-accelerated point cloud alignment.
 *
 *  Contains:
 *
 *  - Root Mean Square Deviation (RMSD) of raw point differences
 *  - Kabsch algorithm for optimal rigid body alignment (rotation only)
 *  - Umeyama algorithm for similarity transform (rotation + uniform scaling)
 *
 *  Precision policy is intentionally mixed across algorithm phases:
 *
 *  - @c f64 inputs keep both the geometric transform and the scalar fit metric in @c f64
 *  - @c f32 inputs keep the @c a_centroid, @c b_centroid, @c rotation, and @c scale outputs narrow,
 *    but widen the scalar fit metric to @c f64
 *  - @c f16 and @c bf16 inputs keep transform and metric outputs in @c f32
 *
 *  This keeps @c f32 mesh kernels much faster than @c f64 ones by preserving narrower input
 *  bandwidth, while still widening the numerically sensitive stages that govern alignment quality.
 *
 *  For hardware architectures:
 *
 *  - x86 (AVX2, AVX512)
 *  - Arm (NEON, SVE)
 *
 *  @section mesh_applications Applications
 *
 *  These routines are the core of point-cloud alignment pipelines:
 *
 *  - Structural biology: protein backbone or ligand alignment (RMSD, Kabsch)
 *  - Computer graphics: mesh registration and deformation transfer
 *  - Robotics/SLAM: point-cloud registration and tracking
 *
 *  @section transformation_convention Transformation Convention
 *
 *  All functions compute a transformation that aligns the first point cloud (a) to the second (b).
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
 *  - RMSD: raw √(Σ‖aᵢ − bᵢ‖² / n) without centering or alignment; R = identity, scale = 1.0,
 *    centroids zeroed
 *  - Kabsch: Finds optimal rotation R minimizing ‖R × (a - ā) - (b - b̄)‖. scale = 1.0
 *  - Umeyama: Finds optimal rotation R and scale c minimizing ‖c × R × (a - ā) - (b - b̄)‖
 *
 *  Kabsch and Umeyama compute a 3×3 cross-covariance matrix H = Σ(aᵢ - ā)(bᵢ - b̄)ᵀ and recover R
 *  from the SVD of H. Umeyama additionally estimates a uniform scale from the singular values and
 *  the variance of the centered source points.
 *
 *  The 3×3 SVD implementation is based on the McAdams et al. paper:
 *  "Computing the Singular Value Decomposition of 3×3 matrices with minimal branching and
 *  elementary floating point operations", University of Wisconsin - Madison TR1690, 2011.
 *
 *  @section numerical_notes Numerical Notes
 *
 *  Let @c n be the number of 3D points:
 *
 *  - `O(n)` stages are the point-cloud passes for centroids, cross-covariance, source variance, and
 *    transformed SSD.
 *  - `O(1)` stages are the fixed-size @b [3,3] SVD/eigensolve, scale construction, and
 *    determinant/reflection fix.
 *
 *  Kernel policy:
 *
 *  - @c f64: both `O(n)` reductions and the `O(1)` @b [3,3] solve run in @c f64.
 *  - @c f32: point coordinates load as @c f32, widen before arithmetic, keep `O(n)` reductions in
 *    @c f64, keep the `O(1)` @b [3,3] solve in @c f64, and only narrow public transform outputs on
 *    store.
 *  - @c f16 and @c bf16: keep both `O(n)` and `O(1)` stages in @c f32.
 *
 *  - @c f32 transform outputs stay narrow because they are typically applied back onto @c f32 point
 *    clouds.
 *  - Reflections are handled by flipping the last singular vector when det(R) < 0.
 *  - For very small point sets, the loops are scalar-heavy and dominate over SIMD setup costs.
 *
 *  @section mesh_x86_instructions Relevant x86 Instructions
 *
 *  The SIMD kernels are dominated by FMA, permutes, and gathers:
 *
 *  @verbatim
 *  Intrinsic                  Instruction  Notes
 *  _mm256_fmadd_ps/pd         VFMADD*      FMA on FP ports (Haswell/Skylake: ports 0/1)
 *  _mm256_i32gather_ps        VGATHERDPS   High-latency; memory-bound
 *  _mm512_permutex2var_ps/pd  VPERMT2*     Shuffle-heavy; can bottleneck on shuffle ports
 *  _mm512_reduce_add_ps/pd    (sequence)   Implemented via shuffles + adds
 *  @endverbatim
 *
 *  Gather-heavy tails are intentionally isolated to keep the steady-state loop on contiguous loads.
 *
 *  @section mesh_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NUMKONG_MESH_H
#define NUMKONG_MESH_H

#include "numkong/capabilities.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief RMSD mesh superposition function.
 *
 *  The transformation aligns a to b: a′ᵢ = scale × R × (aᵢ - ā) + b̄
 *
 *  @param[in] a First point cloud (source), n × 3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] b Second point cloud (target), n × 3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] n Number of 3D points in each cloud.
 *  @param[out] a_centroid Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation Row-major 3×3 rotation matrix (9 values), always identity. Can be NULL.
 *  @param[out] scale Scale factor applied, always 1. Can be NULL.
 *  @param[out] result RMSD after applying the transformation.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_rmsd_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                         nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                         nk_capability_t capabilities, void *stream);
/** @copydoc nk_rmsd_f64_best */
NUMKONG_API nk_status_t nk_rmsd_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                         nk_capability_t capabilities, void *stream);
/** @copydoc nk_rmsd_f64_best */
NUMKONG_API nk_status_t nk_rmsd_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                         nk_capability_t capabilities, void *stream);
/** @copydoc nk_rmsd_f64_best */
NUMKONG_API nk_status_t nk_rmsd_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                          nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                          nk_capability_t capabilities, void *stream);

/**
 *  @brief Kabsch mesh superposition function.
 *
 *  The transformation aligns a to b: a′ᵢ = scale × R × (aᵢ - ā) + b̄
 *
 *  @param[in] a First point cloud (source), n × 3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] b Second point cloud (target), n × 3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] n Number of 3D points in each cloud.
 *  @param[out] a_centroid Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation Row-major 3×3 rotation matrix (9 values). Can be NULL.
 *  @param[out] scale Scale factor applied, always 1. Can be NULL.
 *  @param[out] result RMSD after applying the transformation.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_kabsch_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                           nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                           nk_capability_t capabilities, void *stream);
/** @copydoc nk_kabsch_f64_best */
NUMKONG_API nk_status_t nk_kabsch_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                           nk_capability_t capabilities, void *stream);
/** @copydoc nk_kabsch_f64_best */
NUMKONG_API nk_status_t nk_kabsch_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           nk_capability_t capabilities, void *stream);
/** @copydoc nk_kabsch_f64_best */
NUMKONG_API nk_status_t nk_kabsch_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream);

/**
 *  @brief Umeyama mesh superposition function.
 *
 *  The transformation aligns a to b: a′ᵢ = scale × R × (aᵢ - ā) + b̄
 *
 *  @param[in] a First point cloud (source), n × 3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] b Second point cloud (target), n × 3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] n Number of 3D points in each cloud.
 *  @param[out] a_centroid Centroid of first cloud (3 values). Can be NULL.
 *  @param[out] b_centroid Centroid of second cloud (3 values). Can be NULL.
 *  @param[out] rotation Row-major 3×3 rotation matrix (9 values). Can be NULL.
 *  @param[out] scale Scale factor applied. Can be NULL.
 *  @param[out] result RMSD after applying the transformation.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_umeyama_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                            nk_capability_t capabilities, void *stream);
/** @copydoc nk_umeyama_f64_best */
NUMKONG_API nk_status_t nk_umeyama_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                            nk_capability_t capabilities, void *stream);
/** @copydoc nk_umeyama_f64_best */
NUMKONG_API nk_status_t nk_umeyama_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream);
/** @copydoc nk_umeyama_f64_best */
NUMKONG_API nk_status_t nk_umeyama_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                             nk_f32_t *result, nk_capability_t capabilities, void *stream);

/** @copydoc nk_rmsd_f64_best */
NUMKONG_API nk_status_t nk_rmsd_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                           nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_kabsch_f64_best */
NUMKONG_API nk_status_t nk_kabsch_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                             nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                             nk_f64_t *result, void *stream);
/** @copydoc nk_umeyama_f64_best */
NUMKONG_API nk_status_t nk_umeyama_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                              nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                              nk_f64_t *result, void *stream);

/** @copydoc nk_rmsd_f32_best */
NUMKONG_API nk_status_t nk_rmsd_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_kabsch_f32_best */
NUMKONG_API nk_status_t nk_kabsch_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                             nk_f64_t *result, void *stream);
/** @copydoc nk_umeyama_f32_best */
NUMKONG_API nk_status_t nk_umeyama_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f64_t *result, void *stream);

/** @copydoc nk_rmsd_f16_best */
NUMKONG_API nk_status_t nk_rmsd_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_kabsch_f16_best */
NUMKONG_API nk_status_t nk_kabsch_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                             nk_f32_t *result, void *stream);
/** @copydoc nk_umeyama_f16_best */
NUMKONG_API nk_status_t nk_umeyama_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f32_t *result, void *stream);

/** @copydoc nk_rmsd_bf16_best */
NUMKONG_API nk_status_t nk_rmsd_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            void *stream);
/** @copydoc nk_kabsch_bf16_best */
NUMKONG_API nk_status_t nk_kabsch_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f32_t *result, void *stream);
/** @copydoc nk_umeyama_bf16_best */
NUMKONG_API nk_status_t nk_umeyama_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                               nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                               nk_f32_t *scale, nk_f32_t *result, void *stream);

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer. */
#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_rmsd_f32_best */
NUMKONG_API nk_status_t nk_rmsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                            void *stream);
/** @copydoc nk_kabsch_f32_best */
NUMKONG_API nk_status_t nk_kabsch_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f64_t *result, void *stream);
/** @copydoc nk_umeyama_f32_best */
NUMKONG_API nk_status_t nk_umeyama_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                               nk_f64_t *result, void *stream);

/** @copydoc nk_rmsd_f64_best */
NUMKONG_API nk_status_t nk_rmsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                            void *stream);
/** @copydoc nk_kabsch_f64_best */
NUMKONG_API nk_status_t nk_kabsch_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                              nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                              nk_f64_t *result, void *stream);
/** @copydoc nk_umeyama_f64_best */
NUMKONG_API nk_status_t nk_umeyama_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                               nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                               nk_f64_t *result, void *stream);

/** @copydoc nk_rmsd_f16_best */
NUMKONG_API nk_status_t nk_rmsd_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            void *stream);
/** @copydoc nk_kabsch_f16_best */
NUMKONG_API nk_status_t nk_kabsch_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f32_t *result, void *stream);
/** @copydoc nk_umeyama_f16_best */
NUMKONG_API nk_status_t nk_umeyama_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                               nk_f32_t *result, void *stream);
/** @copydoc nk_rmsd_bf16_best */
NUMKONG_API nk_status_t nk_rmsd_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                             nk_f32_t *result, void *stream);
/** @copydoc nk_kabsch_bf16_best */
NUMKONG_API nk_status_t nk_kabsch_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                               nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                               nk_f32_t *scale, nk_f32_t *result, void *stream);
/** @copydoc nk_umeyama_bf16_best */
NUMKONG_API nk_status_t nk_umeyama_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                                nk_f32_t *scale, nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_SKYLAKE

/*  SIMD-powered backends for AVX512-BF16 CPUs of AMD Genoa / Intel Sapphire Rapids
 *  generation and newer. */
#if NUMKONG_TARGET_GENOA
/** @copydoc nk_rmsd_bf16_best */
NUMKONG_API nk_status_t nk_rmsd_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_kabsch_bf16_best */
NUMKONG_API nk_status_t nk_kabsch_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                             nk_f32_t *result, void *stream);
/** @copydoc nk_umeyama_bf16_best */
NUMKONG_API nk_status_t nk_umeyama_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_GENOA

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer. */
#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_rmsd_f32_best */
NUMKONG_API nk_status_t nk_rmsd_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                            void *stream);
/** @copydoc nk_kabsch_f32_best */
NUMKONG_API nk_status_t nk_kabsch_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f64_t *result, void *stream);
/** @copydoc nk_umeyama_f32_best */
NUMKONG_API nk_status_t nk_umeyama_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                               nk_f64_t *result, void *stream);

/** @copydoc nk_rmsd_f64_best */
NUMKONG_API nk_status_t nk_rmsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                            void *stream);
/** @copydoc nk_kabsch_f64_best */
NUMKONG_API nk_status_t nk_kabsch_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                              nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                              nk_f64_t *result, void *stream);
/** @copydoc nk_umeyama_f64_best */
NUMKONG_API nk_status_t nk_umeyama_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                               nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                               nk_f64_t *result, void *stream);

/** @copydoc nk_rmsd_f16_best */
NUMKONG_API nk_status_t nk_rmsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            void *stream);
/** @copydoc nk_kabsch_f16_best */
NUMKONG_API nk_status_t nk_kabsch_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f32_t *result, void *stream);
/** @copydoc nk_umeyama_f16_best */
NUMKONG_API nk_status_t nk_umeyama_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                               nk_f32_t *result, void *stream);

/** @copydoc nk_rmsd_bf16_best */
NUMKONG_API nk_status_t nk_rmsd_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                             nk_f32_t *result, void *stream);
/** @copydoc nk_kabsch_bf16_best */
NUMKONG_API nk_status_t nk_kabsch_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                               nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                               nk_f32_t *scale, nk_f32_t *result, void *stream);
/** @copydoc nk_umeyama_bf16_best */
NUMKONG_API nk_status_t nk_umeyama_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                                nk_f32_t *scale, nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_HASWELL

/*  SIMD-powered backends for Arm NEON CPUs. */
#if NUMKONG_TARGET_NEON
/** @copydoc nk_rmsd_f32_best */
NUMKONG_API nk_status_t nk_rmsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                         void *stream);
/** @copydoc nk_kabsch_f32_best */
NUMKONG_API nk_status_t nk_kabsch_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_umeyama_f32_best */
NUMKONG_API nk_status_t nk_umeyama_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                            void *stream);

/** @copydoc nk_rmsd_f64_best */
NUMKONG_API nk_status_t nk_rmsd_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                         nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                         void *stream);
/** @copydoc nk_kabsch_f64_best */
NUMKONG_API nk_status_t nk_kabsch_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                           nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_umeyama_f64_best */
NUMKONG_API nk_status_t nk_umeyama_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                            void *stream);

/**
 *  @copydoc nk_rmsd_f16_best
 *  @note Widens FP16 to FP32 before accumulating.
 */
NUMKONG_API nk_status_t nk_rmsd_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                         void *stream);

/**
 *  @copydoc nk_kabsch_f16_best
 *  @note Widens FP16 to FP32 before accumulating.
 */
NUMKONG_API nk_status_t nk_kabsch_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_umeyama_f16_best */
NUMKONG_API nk_status_t nk_umeyama_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            void *stream);
#endif // NUMKONG_TARGET_NEON

/*  SIMD-powered backends for Arm NEON BF16 CPUs. */
#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_rmsd_bf16_best */
NUMKONG_API nk_status_t nk_rmsd_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                               nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                               nk_f32_t *scale, nk_f32_t *result, void *stream);
/** @copydoc nk_kabsch_bf16_best */
NUMKONG_API nk_status_t nk_kabsch_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                 nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                                 nk_f32_t *scale, nk_f32_t *result, void *stream);
/** @copydoc nk_umeyama_bf16_best */
NUMKONG_API nk_status_t nk_umeyama_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                  nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                                  nk_f32_t *scale, nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONBFDOT

/*  SIMD-powered backends for Arm NEON FHM (FP16 widening FMA) CPUs. */
#if NUMKONG_TARGET_NEONFHM
/**
 *  @copydoc nk_rmsd_f16_best
 *  @note Accumulates FP16 products into FP32 with the FHM widening FMA.
 */
NUMKONG_API nk_status_t nk_rmsd_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            void *stream);

/**
 *  @copydoc nk_kabsch_f16_best
 *  @note Accumulates FP16 products into FP32 with the FHM widening FMA.
 */
NUMKONG_API nk_status_t nk_kabsch_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                              nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                              nk_f32_t *result, void *stream);

/**
 *  @copydoc nk_umeyama_f16_best
 *  @note Accumulates FP16 products into FP32 with the FHM widening FMA.
 */
NUMKONG_API nk_status_t nk_umeyama_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                               nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_RVV
/** @copydoc nk_rmsd_f32_best */
NUMKONG_API nk_status_t nk_rmsd_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                        nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                        void *stream);
/** @copydoc nk_rmsd_f64_best */
NUMKONG_API nk_status_t nk_rmsd_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                        nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                        void *stream);
/** @copydoc nk_rmsd_f16_best */
NUMKONG_API nk_status_t nk_rmsd_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                        nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                        void *stream);
/** @copydoc nk_rmsd_bf16_best */
NUMKONG_API nk_status_t nk_rmsd_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                         void *stream);
/** @copydoc nk_kabsch_f32_best */
NUMKONG_API nk_status_t nk_kabsch_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                          nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                          void *stream);
/** @copydoc nk_kabsch_f64_best */
NUMKONG_API nk_status_t nk_kabsch_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                          nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                          void *stream);
/** @copydoc nk_kabsch_f16_best */
NUMKONG_API nk_status_t nk_kabsch_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                          nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                          void *stream);
/** @copydoc nk_kabsch_bf16_best */
NUMKONG_API nk_status_t nk_kabsch_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_umeyama_f32_best */
NUMKONG_API nk_status_t nk_umeyama_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_umeyama_f64_best */
NUMKONG_API nk_status_t nk_umeyama_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                           nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_umeyama_f16_best */
NUMKONG_API nk_status_t nk_umeyama_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_umeyama_bf16_best */
NUMKONG_API nk_status_t nk_umeyama_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            void *stream);
#endif // NUMKONG_TARGET_RVV

/*  WASM Relaxed SIMD backends using wasm_f32x4_relaxed_madd for FMA. */
#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_rmsd_f32_best */
NUMKONG_API nk_status_t nk_rmsd_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                                nk_f64_t *result, void *stream);
/** @copydoc nk_kabsch_f32_best */
NUMKONG_API nk_status_t nk_kabsch_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                  nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                                  nk_f32_t *scale, nk_f64_t *result, void *stream);
/** @copydoc nk_umeyama_f32_best */
NUMKONG_API nk_status_t nk_umeyama_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                   nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                                   nk_f32_t *scale, nk_f64_t *result, void *stream);
/** @copydoc nk_rmsd_f64_best */
NUMKONG_API nk_status_t nk_rmsd_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                                nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                                nk_f64_t *result, void *stream);
/** @copydoc nk_kabsch_f64_best */
NUMKONG_API nk_status_t nk_kabsch_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                  nk_f64_t *a_centroid, nk_f64_t *b_centroid, nk_f64_t *rotation,
                                                  nk_f64_t *scale, nk_f64_t *result, void *stream);
/** @copydoc nk_umeyama_f64_best */
NUMKONG_API nk_status_t nk_umeyama_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                   nk_f64_t *a_centroid, nk_f64_t *b_centroid, nk_f64_t *rotation,
                                                   nk_f64_t *scale, nk_f64_t *result, void *stream);
#endif // NUMKONG_TARGET_V128RELAXED

/** Returns the metric output dtype for mesh alignment operations. Matches the C++ @c mesh_metric_t
 *  alias in types.hpp. */
NUMKONG_INLINE nk_dtype_t nk_mesh_metric_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the transform output dtype for mesh alignment operations. Matches the C++
 *  @c mesh_transform_t alias in types.hpp. */
NUMKONG_INLINE nk_dtype_t nk_mesh_transform_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief Finds the mesh kernel of @p kind for @p dtype from the best capability in @p capabilities.
 *  @param[out] kernel The kernel, or null when no capability in @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k in header-only builds.
 */
NUMKONG_API nk_status_t nk_mesh_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/mesh/serial.h"
#include "numkong/mesh/neon.h"
#include "numkong/mesh/neonbfdot.h"
#include "numkong/mesh/neonfhm.h"
#include "numkong/mesh/haswell.h"
#include "numkong/mesh/skylake.h"
#include "numkong/mesh/genoa.h"
#include "numkong/mesh/rvv.h"
#include "numkong/mesh/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_rmsd_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                         nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                         nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_kabsch_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                           nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                           nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_umeyama_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_rmsd_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                         nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_kabsch_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                           nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_umeyama_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_rmsd_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                          nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                          nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_kabsch_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_umeyama_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                             nk_f32_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_rmsd_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                         nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_kabsch_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_umeyama_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(a_centroid), nk_unused_(b_centroid), nk_unused_(rotation),
        nk_unused_(scale), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_mesh_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = (nk_kernel_punned_t)NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_MESH_H
