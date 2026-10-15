/**
 *  @file include/numkong/shapes.h
 *  @author Ash Vardanian
 *  @date June 19, 2024
 *  @brief Many-to-many shape distances between 3D point clouds of any size.
 *
 *  Contains:
 *
 *  - Chamfer: the mean distance from every point to its nearest partner, both ways, halved
 *  - Hausdorff: the worst of those nearest-partner distances
 *  - Wasserstein-Procrustes: the transport distance after a rotation or reflection fit to the plan
 *  - Gromov-Wasserstein: the transport distance between the two clouds' own distance matrices
 *  - RMSD: the root mean square deviation of matched frames after the best rotation
 *
 *  None of the first four needs a correspondence between points, and the two sides may differ in
 *  size: a 20-atom molecule scores against a 60-atom one without padding or resampling. Every kind
 *  is invariant to rotation, reflection and translation, and between normalized packs to scale as
 *  well, so mirror images score as equals and the returned rotation's determinant tells them apart.
 *
 *  @section shapes_packing Packing
 *
 *  Each cloud is packed once, on its own, by @c nk_shapes_pack_*: centered on its centroid and
 *  rotated onto its principal axes, the eigenvectors of its covariance in descending eigenvalue
 *  order. @c nk_shapes_pack_normalized_* also divides it by its RMS distance from the centroid, the
 *  root of its covariance's trace, and keeps that as the pack's @c scale, which plain packs and
 *  clouds without spread leave at 1. A pack is opaque and starts on an @c nk_default_alignment_k
 *  boundary, like every NumKong buffer. @c nk_shapes_packed_shape_* reads its points and constants,
 *  and @c nk_shapes_unpack_* restores each input point from its packed x′, rounded once into the
 *  input type, as
 *
 *      x = scale × 𝑄ᵀ × x′ + centroid
 *
 *  Packing is one serial routine for every capability today, so a pack is byte-identical on every
 *  machine and every capability reads it. The many-to-many kernels take arrays of pointers to packs
 *  on both sides, gathered however the caller stores them. Every kind compares the packed
 *  coordinates, so mixing plain and normalized packs in one call is the caller's choice.
 *
 *  The spreads and largest radius that @c nk_shapes_packed_shape_* reads work like the norms of
 *  dense vectors: they bound the distances for free.
 *
 *  - Wasserstein-Procrustes ≥ √Σₖ (√λa,ₖ − √λb,ₖ)² over the descending spreads
 *  - Hausdorff ≥ |max_radius_a − max_radius_b| over the largest radii
 *
 *  @section shapes_kinds Kinds
 *
 *  Chamfer and Hausdorff compare the two packed clouds under all 8 sign flips of the principal
 *  axes, 4 rotations and 4 reflections, and keep the best. Each flip costs a pass over n × m pairs.
 *
 *  Wasserstein-Procrustes alternates an exact optimal transport plan P between uniform weights with
 *  the rotation or reflection R that best superposes it. From each of the 8 flips it descends to a
 *  local optimum of the cost below, where the cost stops falling, and reports the best of the 8.
 *  On equal sizes the least cost over all P and R is at most the Kabsch RMSD of a known
 *  correspondence, and equals it when that correspondence is an optimal plan.
 *
 *      cost(P, R) = √( Σᵢⱼ Pᵢⱼ × ‖aᵢ − R × bⱼ‖² )
 *
 *  Gromov-Wasserstein starts from the best proper and the best mirrored Wasserstein-Procrustes
 *  plans and from the product plan, then takes a fixed budget of Frank-Wolfe steps, each with an
 *  exact transport plan of its own, and reports the least loss they reach.
 *
 *      loss(P) = ½ × √( Σᵢⱼₖₗ (‖aᵢ − aₖ‖ − ‖bⱼ − bₗ‖)² × Pᵢⱼ × Pₖₗ )
 *
 *  Both transport kinds solve each pair in a canonical order, so swapping the sides returns the
 *  same distance bit for bit and the transposed rotation.
 *
 *  @section shapes_rmsds Superposed RMSD
 *
 *  RMSD is the one kind that needs a correspondence: every frame holds the same points in the same
 *  order, like the frames of one molecular dynamics trajectory. It reads the packs of
 *  @c nk_shapes_pack_*, which keep the point order, as no proper rotation of either frame changes
 *  it. Between frames a and b, with G = Σ‖xᵢ′‖² and M = Σ aᵢ × bᵢᵀ over the packed points,
 *
 *      RMSD² = min_R Σ‖aᵢ − R × bᵢ‖² / n = (Ga + Gb − 2λ) / n
 *
 *  over proper rotations R, where λ is the largest eigenvalue of Theobald's 4 × 4 quaternion key
 *  matrix of M, or σ₁ + σ₂ ± σ₃ over its singular values. Like the spreads, the norms bound it:
 *
 *  - RMSD ≥ |√(Ga / n) − √(Gb / n)|
 *
 *  Packs of different sizes cannot pair, and the kernels refuse them. Between normalized packs the
 *  RMSD is scale-free.
 *
 *  @section shapes_outputs Outputs
 *
 *  Distances land in a @b [rows,columns] matrix with a byte stride between rows. The optional
 *  rotations land in a contiguous @b [rows,columns,3,3] tensor of row-major matrices, each taking a
 *  column cloud onto a row cloud through the centroids and scales of the packs:
 *
 *      aᵢ ≈ (scale_a / scale_b) × R × (bⱼ − b̄) + ā
 *
 *  @section shapes_precision Precision
 *
 *  - @c f64 inputs pack with @c f64 constants, compute in @c f64 with the moments, Chamfer's roots,
 *    each plan's cost and every Gromov-Wasserstein reduction summed by TwoProd and TwoSum, and
 *    return @c f64 distances
 *  - @c f32 inputs pack with @c f64 constants, compute in @c f64, return @c f64 distances
 *  - @c f16 inputs pack with @c f32 constants, compute in @c f32, return @c f32 distances
 *  - Rotations are @c f32 for every input type
 *
 *  RMSD keeps the rules of the one-to-one Kabsch kernels it replaces:
 *
 *  - @c f64 packs sum M with TwoProd and TwoSum, and otherwise follow @c f32 packs
 *  - @c f32 packs sum M in @c f64, fit the Kabsch rotation in @c f64, then sum ‖aᵢ − R × bᵢ‖²
 *    directly in a second, compensated pass, as folding it into Ga + Gb − 2λ cancels to √ε once
 *    the frames align, and return @c f64
 *  - @c f16 packs sum M and G in @c f32 and return @c f32, folding into Ga + Gb − 2λ clamped at
 *    zero, as the √ε of @c f32 stays below the rounding of their points
 *  - packing rounds each rotated point once into the input type, which moves an RMSD by at most
 *    √3 × u × (r_a + r_b), with u the unit roundoff and r the largest radius of each pack
 *
 *  @section shapes_workspace Workspace
 *
 *  A transport plan depends on both clouds, so it cannot live in either pack. The transport kinds
 *  take a caller's workspace, sized by their @c *_workspace_size_* query for the largest cloud on
 *  each side and aligned to @c nk_default_alignment_k: Θ(n + m) bytes for Wasserstein-Procrustes
 *  and Θ(n² + m² + n × m) for Gromov-Wasserstein. Chamfer and Hausdorff need none.
 */
#ifndef NUMKONG_SHAPES_H
#define NUMKONG_SHAPES_H

#include "numkong/capabilities.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns the bytes a pack of @p points takes.
 *  @param[in] points The number of 3D points.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[out] bytes The size of the pack.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_shapes_pack_size_f16_best(nk_size_t points, nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_shapes_pack_size_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_size_f32_best(nk_size_t points, nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_shapes_pack_size_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_size_f64_best(nk_size_t points, nk_capability_t capabilities, nk_size_t *bytes);

/**
 *  @brief Packs one cloud: centers it, rotates it onto its principal axes, and keeps its constants.
 *  @param[in] points The cloud, @p count × 3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] count The number of 3D points.
 *  @param[out] packed The pack, as many bytes as @c nk_shapes_pack_size_f16_best reports, starting
 *      on an @c nk_default_alignment_k boundary.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_shapes_pack_f16_best(nk_f16_t const *points, nk_size_t count, void *packed,
                                                nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_shapes_pack_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_f32_best(nk_f32_t const *points, nk_size_t count, void *packed,
                                                nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_shapes_pack_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_f64_best(nk_f64_t const *points, nk_size_t count, void *packed,
                                                nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Packs one cloud like @c nk_shapes_pack_f16_best, also dividing it by its RMS distance
 *      from the centroid, kept as the pack's @c scale, or by 1 when the cloud has no spread.
 *
 *  Every kind then compares shapes regardless of size, and a rotation between two normalized packs
 *  takes the column cloud onto the row cloud as aᵢ ≈ (scale_a / scale_b) × R × (bⱼ − b̄) + ā.
 */
NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_best(nk_f16_t const *points, nk_size_t count, void *packed,
                                                           nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_shapes_pack_normalized_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_best(nk_f32_t const *points, nk_size_t count, void *packed,
                                                           nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_shapes_pack_normalized_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_normalized_f64_best(nk_f64_t const *points, nk_size_t count, void *packed,
                                                           nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Reads a pack's points and constants, skipping every null output.
 *  @param[in] packed A pack of @c f16 points, plain or normalized.
 *  @param[out] points Null, or the number of 3D points.
 *  @param[out] centroid Null, or the 3 coordinates of the mean input point.
 *  @param[out] frame Null, or the unit quaternion (x, y, z, w) of 𝑄, rotating centered inputs onto
 *      the principal axes.
 *  @param[out] scale Null, or what the rotated inputs were divided by: their RMS distance from the
 *      centroid, or 1.
 *  @param[out] spreads Null, or the 3 descending eigenvalues of the packed points' second moments.
 *  @param[out] max_radius Null, or the largest packed point norm.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_shapes_packed_shape_f16_best(void const *packed, nk_size_t *points, nk_f32_t *centroid,
                                                        nk_f32_t *frame, nk_f32_t *scale, nk_f32_t *spreads,
                                                        nk_f32_t *max_radius, nk_capability_t capabilities,
                                                        nk_stream_t stream);
/** @copydoc nk_shapes_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_shapes_packed_shape_f32_best(void const *packed, nk_size_t *points, nk_f64_t *centroid,
                                                        nk_f64_t *frame, nk_f64_t *scale, nk_f64_t *spreads,
                                                        nk_f64_t *max_radius, nk_capability_t capabilities,
                                                        nk_stream_t stream);
/** @copydoc nk_shapes_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_shapes_packed_shape_f64_best(void const *packed, nk_size_t *points, nk_f64_t *centroid,
                                                        nk_f64_t *frame, nk_f64_t *scale, nk_f64_t *spreads,
                                                        nk_f64_t *max_radius, nk_capability_t capabilities,
                                                        nk_stream_t stream);

/**
 *  @brief Restores a pack's cloud as x = scale × 𝑄ᵀ × x′ + centroid, computed in the math type and
 *      rounded once into the input type.
 *
 *  Unpacking an absolute pack returns its inputs within one storage rounding, and a normalized pack
 *  its cloud at the original scale.
 *
 *  @param[in] packed A pack of @c f16 points, plain or normalized.
 *  @param[out] points The cloud, its points × 3 interleaved [x0,y0,z0, x1,y1,z1, ...].
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_shapes_unpack_f16_best(void const *packed, nk_f16_t *points, nk_capability_t capabilities,
                                                  nk_stream_t stream);
/** @copydoc nk_shapes_unpack_f16_best */
NUMKONG_API nk_status_t nk_shapes_unpack_f32_best(void const *packed, nk_f32_t *points, nk_capability_t capabilities,
                                                  nk_stream_t stream);
/** @copydoc nk_shapes_unpack_f16_best */
NUMKONG_API nk_status_t nk_shapes_unpack_f64_best(void const *packed, nk_f64_t *points, nk_capability_t capabilities,
                                                  nk_stream_t stream);

/**
 *  @brief Chamfer distances between every row and every column cloud, the best of 8 axis flips.
 *
 *  Each is ½ × (meanᵢ minⱼ ‖aᵢ − S × bⱼ‖ + meanⱼ minᵢ ‖aᵢ − S × bⱼ‖), zero between two empty
 *  clouds and infinite between an empty and a non-empty one.
 *
 *  @param[in] a_packed The row packs, @p rows pointers.
 *  @param[in] rows The number of row clouds.
 *  @param[in] b_packed The column packs, @p columns pointers.
 *  @param[in] columns The number of column clouds.
 *  @param[out] results The @b [rows,columns] distances.
 *  @param[in] results_stride The bytes between consecutive rows of @p results.
 *  @param[out] rotations Null, or the @b [rows,columns,3,3] rotations taking each column cloud onto
 *      each row cloud.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_chamfers_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Hausdorff distances between every row and every column cloud, the best of 8 axis flips.
 *
 *  Each is max(maxᵢ minⱼ ‖aᵢ − S × bⱼ‖, maxⱼ minᵢ ‖aᵢ − S × bⱼ‖), with the same empty-cloud
 *  convention and parameters as @c nk_chamfers_packed_f64_best.
 */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Returns the workspace bytes the transport kinds need between clouds of up to
 *      @p row_points and @p column_points points.
 *  @param[in] row_points The largest row cloud.
 *  @param[in] column_points The largest column cloud.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[out] bytes The size of the workspace.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f64_best(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_capability_t capabilities,
                                                                         nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f32_best(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_capability_t capabilities,
                                                                         nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f16_best(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_capability_t capabilities,
                                                                         nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f64_best(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f32_best(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f16_best(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_capability_t capabilities, nk_size_t *bytes);

/**
 *  @brief Wasserstein-Procrustes distances between every row cloud and every column cloud.
 *
 *  Each is the root mean squared distance of an exact optimal transport plan between uniform
 *  weights under a rotation or reflection, the best of the local optima that alternating the two
 *  reaches from the 8 axis flips, zero between two empty clouds and infinite between an empty and a
 *  non-empty one. Other parameters match the Chamfer kernel's.
 *
 *  @param[in] workspace The scratch memory, aligned to @c nk_default_alignment_k.
 *  @param[in] workspace_bytes At least the bytes the matching workspace-size query reports for the
 *      largest clouds on each side.
 *  @return @c nk_success_k, @c nk_misaligned_k for a misaligned @p workspace,
 *      @c nk_unexpected_dimensions_k when @p workspace_bytes falls short, or
 *      @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes,
                                                                 nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes,
                                                                 nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f32_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes,
                                                                 nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Gromov-Wasserstein distances between every row cloud and every column cloud.
 *
 *  Each compares the two clouds' plain distance matrices under the best transport plan reached from
 *  the Wasserstein-Procrustes plans and the product plan. The rotation is the orthogonal Procrustes
 *  fit of that plan. Parameters and returns match the Wasserstein-Procrustes kernel's, with the
 *  workspace sized by @c nk_gromovwassersteins_workspace_size_f64_best.
 */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f64_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_capability_t capabilities,
                                                              nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f64_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_capability_t capabilities,
                                                              nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f32_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_capability_t capabilities,
                                                              nk_stream_t stream);

/**
 *  @brief Superposed RMSD between every row and every column frame, matched point by point.
 *
 *  Each is √(min_R Σ‖aᵢ − R × bᵢ‖² / n) over proper rotations R, and zero between empty frames.
 *
 *  @param[in] a_packed The row shape packs, plain or normalized, @p rows pointers.
 *  @param[in] rows The number of row frames.
 *  @param[in] b_packed The column packs, @p columns pointers.
 *  @param[in] columns The number of column frames.
 *  @param[out] results The @b [rows,columns] distances.
 *  @param[in] results_stride The bytes between consecutive rows of @p results.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, @c nk_unexpected_dimensions_k when the packs differ in points, or
 *      @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_rmsds_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                 nk_size_t results_stride, nk_capability_t capabilities,
                                                 nk_stream_t stream);
/** @copydoc nk_rmsds_packed_f64_best */
NUMKONG_API nk_status_t nk_rmsds_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                 nk_size_t results_stride, nk_capability_t capabilities,
                                                 nk_stream_t stream);
/** @copydoc nk_rmsds_packed_f64_best */
NUMKONG_API nk_status_t nk_rmsds_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                 nk_size_t results_stride, nk_capability_t capabilities,
                                                 nk_stream_t stream);

/** @copydoc nk_shapes_pack_size_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_size_f16_serial(nk_size_t points, nk_size_t *bytes);
/** @copydoc nk_shapes_pack_size_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_size_f32_serial(nk_size_t points, nk_size_t *bytes);
/** @copydoc nk_shapes_pack_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_f16_serial(nk_f16_t const *points, nk_size_t count, void *packed,
                                                  nk_stream_t stream);
/** @copydoc nk_shapes_pack_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_f32_serial(nk_f32_t const *points, nk_size_t count, void *packed,
                                                  nk_stream_t stream);
/** @copydoc nk_shapes_pack_normalized_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_serial(nk_f16_t const *points, nk_size_t count, void *packed,
                                                             nk_stream_t stream);
/** @copydoc nk_shapes_pack_normalized_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_serial(nk_f32_t const *points, nk_size_t count, void *packed,
                                                             nk_stream_t stream);
/** @copydoc nk_shapes_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_shapes_packed_shape_f16_serial(void const *packed, nk_size_t *points, nk_f32_t *centroid,
                                                          nk_f32_t *frame, nk_f32_t *scale, nk_f32_t *spreads,
                                                          nk_f32_t *max_radius, nk_stream_t stream);
/** @copydoc nk_shapes_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_shapes_packed_shape_f32_serial(void const *packed, nk_size_t *points, nk_f64_t *centroid,
                                                          nk_f64_t *frame, nk_f64_t *scale, nk_f64_t *spreads,
                                                          nk_f64_t *max_radius, nk_stream_t stream);
/** @copydoc nk_shapes_unpack_f16_best */
NUMKONG_API nk_status_t nk_shapes_unpack_f16_serial(void const *packed, nk_f16_t *points, nk_stream_t stream);
/** @copydoc nk_shapes_unpack_f16_best */
NUMKONG_API nk_status_t nk_shapes_unpack_f32_serial(void const *packed, nk_f32_t *points, nk_stream_t stream);
/** @copydoc nk_shapes_pack_size_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_size_f64_serial(nk_size_t points, nk_size_t *bytes);
/** @copydoc nk_shapes_pack_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_f64_serial(nk_f64_t const *points, nk_size_t count, void *packed,
                                                  nk_stream_t stream);
/** @copydoc nk_shapes_pack_normalized_f16_best */
NUMKONG_API nk_status_t nk_shapes_pack_normalized_f64_serial(nk_f64_t const *points, nk_size_t count, void *packed,
                                                             nk_stream_t stream);
/** @copydoc nk_shapes_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_shapes_packed_shape_f64_serial(void const *packed, nk_size_t *points, nk_f64_t *centroid,
                                                          nk_f64_t *frame, nk_f64_t *scale, nk_f64_t *spreads,
                                                          nk_f64_t *max_radius, nk_stream_t stream);
/** @copydoc nk_shapes_unpack_f16_best */
NUMKONG_API nk_status_t nk_shapes_unpack_f64_serial(void const *packed, nk_f64_t *points, nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f64_serial(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f64_serial(void const *const *a_packed, nk_size_t rows,
                                                        void const *const *b_packed, nk_size_t columns,
                                                        nk_f64_t *results, nk_size_t results_stride,
                                                        nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f64_serial(nk_size_t row_points,
                                                                           nk_size_t column_points, nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f64_serial(nk_size_t row_points, nk_size_t column_points,
                                                                        nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f64_serial(void const *const *a_packed, nk_size_t rows,
                                                                   void const *const *b_packed, nk_size_t columns,
                                                                   nk_f64_t *results, nk_size_t results_stride,
                                                                   nk_f32_t *rotations, void *workspace,
                                                                   nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f64_serial(void const *const *a_packed, nk_size_t rows,
                                                                void const *const *b_packed, nk_size_t columns,
                                                                nk_f64_t *results, nk_size_t results_stride,
                                                                nk_f32_t *rotations, void *workspace,
                                                                nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f16_serial(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f32_serial(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_serial(void const *const *a_packed, nk_size_t rows,
                                                        void const *const *b_packed, nk_size_t columns,
                                                        nk_f32_t *results, nk_size_t results_stride,
                                                        nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_serial(void const *const *a_packed, nk_size_t rows,
                                                        void const *const *b_packed, nk_size_t columns,
                                                        nk_f64_t *results, nk_size_t results_stride,
                                                        nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f16_serial(nk_size_t row_points,
                                                                           nk_size_t column_points, nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f32_serial(nk_size_t row_points,
                                                                           nk_size_t column_points, nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f16_serial(nk_size_t row_points, nk_size_t column_points,
                                                                        nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f32_serial(nk_size_t row_points, nk_size_t column_points,
                                                                        nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_serial(void const *const *a_packed, nk_size_t rows,
                                                                   void const *const *b_packed, nk_size_t columns,
                                                                   nk_f32_t *results, nk_size_t results_stride,
                                                                   nk_f32_t *rotations, void *workspace,
                                                                   nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_serial(void const *const *a_packed, nk_size_t rows,
                                                                   void const *const *b_packed, nk_size_t columns,
                                                                   nk_f64_t *results, nk_size_t results_stride,
                                                                   nk_f32_t *rotations, void *workspace,
                                                                   nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_serial(void const *const *a_packed, nk_size_t rows,
                                                                void const *const *b_packed, nk_size_t columns,
                                                                nk_f32_t *results, nk_size_t results_stride,
                                                                nk_f32_t *rotations, void *workspace,
                                                                nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_serial(void const *const *a_packed, nk_size_t rows,
                                                                void const *const *b_packed, nk_size_t columns,
                                                                nk_f64_t *results, nk_size_t results_stride,
                                                                nk_f32_t *rotations, void *workspace,
                                                                nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_rmsds_packed_f64_best */
NUMKONG_API nk_status_t nk_rmsds_packed_f64_serial(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                   nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_rmsds_packed_f64_best */
NUMKONG_API nk_status_t nk_rmsds_packed_f32_serial(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                   nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_rmsds_packed_f64_best */
NUMKONG_API nk_status_t nk_rmsds_packed_f16_serial(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                   nk_size_t results_stride, nk_stream_t stream);

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f16_haswell(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f32_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f32_haswell(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f64_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_haswell(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f32_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_haswell(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f64_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_haswell(void const *const *a_packed, nk_size_t rows,
                                                                    void const *const *b_packed, nk_size_t columns,
                                                                    nk_f32_t *results, nk_size_t results_stride,
                                                                    nk_f32_t *rotations, void *workspace,
                                                                    nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_haswell(void const *const *a_packed, nk_size_t rows,
                                                                    void const *const *b_packed, nk_size_t columns,
                                                                    nk_f64_t *results, nk_size_t results_stride,
                                                                    nk_f32_t *rotations, void *workspace,
                                                                    nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_haswell(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f32_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_haswell(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f64_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f64_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
                                                                    void const *const *b_packed, nk_size_t columns,
                                                                    nk_f64_t *results, nk_size_t results_stride,
                                                                    nk_f32_t *rotations, void *workspace,
                                                                    nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f32_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f64_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f32_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f64_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
                                                                    void const *const *b_packed, nk_size_t columns,
                                                                    nk_f32_t *results, nk_size_t results_stride,
                                                                    nk_f32_t *rotations, void *workspace,
                                                                    nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
                                                                    void const *const *b_packed, nk_size_t columns,
                                                                    nk_f64_t *results, nk_size_t results_stride,
                                                                    nk_f32_t *rotations, void *workspace,
                                                                    nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f32_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_rmsds_packed_f64_best */
NUMKONG_API nk_status_t nk_rmsds_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_rmsds_packed_f64_best */
NUMKONG_API nk_status_t nk_rmsds_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_stream_t stream);
/** @copydoc nk_rmsds_packed_f64_best */
NUMKONG_API nk_status_t nk_rmsds_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                    nk_size_t results_stride, nk_stream_t stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f16_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                           void const *const *b_packed, nk_size_t columns,
                                                           nk_f32_t *results, nk_size_t results_stride,
                                                           nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f32_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                           void const *const *b_packed, nk_size_t columns,
                                                           nk_f64_t *results, nk_size_t results_stride,
                                                           nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                             void const *const *b_packed, nk_size_t columns,
                                                             nk_f32_t *results, nk_size_t results_stride,
                                                             nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                             void const *const *b_packed, nk_size_t columns,
                                                             nk_f64_t *results, nk_size_t results_stride,
                                                             nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                                        void const *const *b_packed, nk_size_t columns,
                                                                        nk_f32_t *results, nk_size_t results_stride,
                                                                        nk_f32_t *rotations, void *workspace,
                                                                        nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                                        void const *const *b_packed, nk_size_t columns,
                                                                        nk_f64_t *results, nk_size_t results_stride,
                                                                        nk_f32_t *rotations, void *workspace,
                                                                        nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                                     void const *const *b_packed, nk_size_t columns,
                                                                     nk_f32_t *results, nk_size_t results_stride,
                                                                     nk_f32_t *rotations, void *workspace,
                                                                     nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                                     void const *const *b_packed, nk_size_t columns,
                                                                     nk_f64_t *results, nk_size_t results_stride,
                                                                     nk_f32_t *rotations, void *workspace,
                                                                     nk_size_t workspace_bytes, nk_stream_t stream);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_NEON
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f16_neon(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f32_neon(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_neon(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_neon(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_neon(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f32_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_neon(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_neon(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f32_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_neon(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f64_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONFHM
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f16_neonfhm(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f32_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_neonfhm(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f32_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_RVV
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f16_rvv(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                   nk_size_t results_stride, nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f32_rvv(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                   nk_size_t results_stride, nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_rvv(void const *const *a_packed, nk_size_t rows,
                                                     void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                     nk_size_t results_stride, nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_rvv(void const *const *a_packed, nk_size_t rows,
                                                     void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_rvv(void const *const *a_packed, nk_size_t rows,
                                                                void const *const *b_packed, nk_size_t columns,
                                                                nk_f32_t *results, nk_size_t results_stride,
                                                                nk_f32_t *rotations, void *workspace,
                                                                nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_rvv(void const *const *a_packed, nk_size_t rows,
                                                                void const *const *b_packed, nk_size_t columns,
                                                                nk_f64_t *results, nk_size_t results_stride,
                                                                nk_f32_t *rotations, void *workspace,
                                                                nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_rvv(void const *const *a_packed, nk_size_t rows,
                                                             void const *const *b_packed, nk_size_t columns,
                                                             nk_f32_t *results, nk_size_t results_stride,
                                                             nk_f32_t *rotations, void *workspace,
                                                             nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_rvv(void const *const *a_packed, nk_size_t rows,
                                                             void const *const *b_packed, nk_size_t columns,
                                                             nk_f64_t *results, nk_size_t results_stride,
                                                             nk_f32_t *rotations, void *workspace,
                                                             nk_size_t workspace_bytes, nk_stream_t stream);
#endif // NUMKONG_TARGET_RVV

/*  NVIDIA backends for the compute capability 10.x family, with F16 tensor-core products. */
#if NUMKONG_TARGET_BLACKWELL
/** @copydoc nk_chamfers_packed_f64_best */
NUMKONG_API nk_status_t nk_chamfers_packed_f16_blackwell(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f32_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_hausdorffs_packed_f64_best */
NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_blackwell(void const *const *a_packed, nk_size_t rows,
                                                           void const *const *b_packed, nk_size_t columns,
                                                           nk_f32_t *results, nk_size_t results_stride,
                                                           nk_f32_t *rotations, nk_stream_t stream);
/** @copydoc nk_rmsds_packed_f64_best */
NUMKONG_API nk_status_t nk_rmsds_packed_f16_blackwell(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                      nk_size_t results_stride, nk_stream_t stream);
#endif // NUMKONG_TARGET_BLACKWELL

/*  CUDA baseline on every NVIDIA device: exact transport by successive shortest paths, one block
 *  per flip of each pair, with the serial math types. Packs keep the serial layout, and the pack
 *  tables, packs, outputs and workspace are device memory. */
#if NUMKONG_TARGET_CUDA
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f16_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f32_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_cuda(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f32_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_packed_f64_best */
NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_cuda(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f16_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_size_t *bytes);
/** @copydoc nk_wassersteinprocrustes_workspace_size_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f32_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_size_t *bytes);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_cuda(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f32_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream);
/** @copydoc nk_gromovwassersteins_packed_f64_best */
NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_cuda(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f64_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream);
#endif // NUMKONG_TARGET_CUDA

/** Returns the distance dtype of the many-to-many shape kernels, as the C++ @c shapes_distance_t
 *  alias in types.hpp names it. */
NUMKONG_INLINE nk_dtype_t nk_shapes_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the distance dtype of the superposed RMSD kernels. */
NUMKONG_INLINE nk_dtype_t nk_rmsds_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief Finds the shape kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when no capability in @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_shapes_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/shapes/serial.h"
#include "numkong/shapes/neon.h"
#include "numkong/shapes/neonfhm.h"
#include "numkong/shapes/haswell.h"
#include "numkong/shapes/skylake.h"
#include "numkong/shapes/rvv.h"
#include "numkong/shapes/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_shapes_pack_size_f16_best(nk_size_t points, nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_size_f32_best(nk_size_t points, nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_f16_best(nk_f16_t const *points, nk_size_t count, void *packed,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(points), nk_unused_(count), nk_unused_(packed), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_f32_best(nk_f32_t const *points, nk_size_t count, void *packed,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(points), nk_unused_(count), nk_unused_(packed), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_best(nk_f16_t const *points, nk_size_t count, void *packed,
                                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(points), nk_unused_(count), nk_unused_(packed), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_best(nk_f32_t const *points, nk_size_t count, void *packed,
                                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(points), nk_unused_(count), nk_unused_(packed), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_packed_shape_f16_best(void const *packed, nk_size_t *points, nk_f32_t *centroid,
                                                        nk_f32_t *frame, nk_f32_t *scale, nk_f32_t *spreads,
                                                        nk_f32_t *max_radius, nk_capability_t capabilities,
                                                        nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(points), nk_unused_(centroid), nk_unused_(frame), nk_unused_(scale),
        nk_unused_(spreads), nk_unused_(max_radius), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_packed_shape_f32_best(void const *packed, nk_size_t *points, nk_f64_t *centroid,
                                                        nk_f64_t *frame, nk_f64_t *scale, nk_f64_t *spreads,
                                                        nk_f64_t *max_radius, nk_capability_t capabilities,
                                                        nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(points), nk_unused_(centroid), nk_unused_(frame), nk_unused_(scale),
        nk_unused_(spreads), nk_unused_(max_radius), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_unpack_f16_best(void const *packed, nk_f16_t *points, nk_capability_t capabilities,
                                                  nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(points), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_unpack_f32_best(void const *packed, nk_f32_t *points, nk_capability_t capabilities,
                                                  nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(points), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f16_best(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_capability_t capabilities,
                                                                         nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f32_best(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_capability_t capabilities,
                                                                         nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f16_best(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f32_best(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f32_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes,
                                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(workspace), nk_unused_(workspace_bytes),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes,
                                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(workspace), nk_unused_(workspace_bytes),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f32_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_capability_t capabilities,
                                                              nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(workspace), nk_unused_(workspace_bytes),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f64_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_capability_t capabilities,
                                                              nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(workspace), nk_unused_(workspace_bytes),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_size_f64_best(nk_size_t points, nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_f64_best(nk_f64_t const *points, nk_size_t count, void *packed,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(points), nk_unused_(count), nk_unused_(packed), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f64_best(nk_f64_t const *points, nk_size_t count, void *packed,
                                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(points), nk_unused_(count), nk_unused_(packed), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_packed_shape_f64_best(void const *packed, nk_size_t *points, nk_f64_t *centroid,
                                                        nk_f64_t *frame, nk_f64_t *scale, nk_f64_t *spreads,
                                                        nk_f64_t *max_radius, nk_capability_t capabilities,
                                                        nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(points), nk_unused_(centroid), nk_unused_(frame), nk_unused_(scale),
        nk_unused_(spreads), nk_unused_(max_radius), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_unpack_f64_best(void const *packed, nk_f64_t *points, nk_capability_t capabilities,
                                                  nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(points), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f64_best(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_capability_t capabilities,
                                                                         nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f64_best(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes,
                                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(workspace), nk_unused_(workspace_bytes),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f64_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_capability_t capabilities,
                                                              nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(rotations), nk_unused_(workspace), nk_unused_(workspace_bytes),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f64_best(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                 nk_size_t results_stride, nk_capability_t capabilities,
                                                 nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                 nk_size_t results_stride, nk_capability_t capabilities,
                                                 nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                 nk_size_t results_stride, nk_capability_t capabilities,
                                                 nk_stream_t stream) {
    nk_unused_(a_packed), nk_unused_(rows), nk_unused_(b_packed), nk_unused_(columns), nk_unused_(results),
        nk_unused_(results_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_shapes_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = (nk_kernel_punned_t)NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_SHAPES_H
