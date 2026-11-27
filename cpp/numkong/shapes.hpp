/**
 *  @file cpp/numkong/shapes.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for many-to-many shape distances.
 *
 *  Every wrapper dispatches to the best C kernel of the capabilities it is given.
 *
 *  @code{.cpp}
 *  std::span<nk::f32_t const> clouds[] = {first_cloud, second_cloud};
 *  auto [packs, status] = nk::shapes_pack<nk::f32_t>(clouds);
 *  nk::f64_t distances[4];
 *  status = nk::hausdorffs_packed<nk::f32_t>(packs.data(), 2, packs.data(), 2, distances, 2 * sizeof(nk::f64_t),
 *                                            nullptr, nk::f64_t::positive_infinity());
 *  @endcode
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_SHAPES_HPP
#define NUMKONG_SHAPES_HPP

#include <cstddef> // `std::byte`

#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span`
#include <type_traits> // `std::conditional_t`
#include <utility>     // `std::exchange`

#include "numkong/shapes.h"
#include "numkong/packed.hpp"  // `shapes_packed`, `a_shapes_dtype`
#include "numkong/devices.hpp" // `default_capabilities`

namespace ashvardanian::numkong {

/**
 *  @brief Returns the bytes a pack of @p points takes.
 *  @param[in] points Number of points in the cloud.
 *  @param[in] capabilities Capabilities to pick from.
 *  @tparam in_type_ Point type, @c f64_t, @c f32_t or @c f16_t.
 */
template <a_shapes_dtype in_type_>
expected<std::size_t> shapes_pack_size(std::size_t points,
                                       nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>) status = nk_shapes_pack_size_f64_best(points, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_shapes_pack_size_f32_best(points, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_shapes_pack_size_f16_best(points, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_shapes_pack_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs one cloud: centers it, rotates it onto its principal axes, and keeps its constants.
 *  @param[in] points The cloud, @p count × 3 interleaved values.
 *  @param[in] count Number of points in the cloud.
 *  @param[out] packed At least @c shapes_pack_size bytes, aligned to @c nk_default_alignment_k.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
template <a_shapes_dtype in_type_>
status_t shapes_pack(typename in_type_::raw_t const *points, std::size_t count, void *packed,
                     nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_shapes_pack_f64_best(points, count, packed, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_shapes_pack_f32_best(points, count, packed, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_shapes_pack_f16_best(points, count, packed, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_shapes_pack has no kernel for this dtype");
}

/** Packs one cloud like @c shapes_pack, also dividing it by its RMS distance from the centroid,
 *  which the pack keeps as its scale, or by 1 when the cloud has no spread. */
template <a_shapes_dtype in_type_>
status_t shapes_pack_normalized(typename in_type_::raw_t const *points, std::size_t count, void *packed,
                                nk_capability_t capabilities = default_capabilities(),
                                nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_shapes_pack_normalized_f64_best(points, count, packed, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_shapes_pack_normalized_f32_best(points, count, packed, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_shapes_pack_normalized_f16_best(points, count, packed, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_shapes_pack_normalized has no kernel for this dtype");
}

/**
 *  @brief Reads a pack's points and constants, skipping every null output.
 *  @param[in] packed The pack to read.
 *  @param[out] points Null, or the number of 3D points.
 *  @param[out] centroid Null, or the 3 coordinates of the mean input point.
 *  @param[out] frame Null, or the unit quaternion (x, y, z, w) of 𝑄.
 *  @param[out] scale Null, or what the rotated inputs were divided by: their RMS distance from the
 *      centroid, or 1.
 *  @param[out] spreads Null, or the 3 descending eigenvalues of the packed points' second moments.
 *  @param[out] max_radius Null, or the largest packed point norm.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
template <a_shapes_dtype in_type_>
status_t shapes_packed_shape(void const *packed, std::size_t *points, typename in_type_::shapes_distance_t *centroid,
                             typename in_type_::shapes_distance_t *frame, typename in_type_::shapes_distance_t *scale,
                             typename in_type_::shapes_distance_t *spreads,
                             typename in_type_::shapes_distance_t *max_radius,
                             nk_capability_t capabilities = default_capabilities(),
                             nk_stream_t stream = nullptr) noexcept {
    using constant_t = typename in_type_::shapes_distance_t;
    auto const raw = [](constant_t *values) noexcept { return reinterpret_cast<typename constant_t::raw_t *>(values); };
    nk_size_t count = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_shapes_packed_shape_f64_best(packed, &count, raw(centroid), raw(frame), raw(scale), raw(spreads),
                                                 raw(max_radius), capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_shapes_packed_shape_f32_best(packed, &count, raw(centroid), raw(frame), raw(scale), raw(spreads),
                                                 raw(max_radius), capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_shapes_packed_shape_f16_best(packed, &count, raw(centroid), raw(frame), raw(scale), raw(spreads),
                                                 raw(max_radius), capabilities, stream);
    else static_assert(sizeof(in_type_) == 0, "nk_shapes_packed_shape has no kernel for this dtype");
    if (points) *points = static_cast<std::size_t>(count);
    return static_cast<status_t>(status);
}

/**
 *  @brief Restores a pack's cloud as x = scale × 𝑄ᵀ × x′ + centroid, rounded once into @p in_type_.
 *  @param[in] packed The pack to restore the cloud from.
 *  @param[out] points The cloud, as many × 3 interleaved values as the pack holds points.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
template <a_shapes_dtype in_type_>
status_t shapes_unpack(void const *packed, typename in_type_::raw_t *points,
                       nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_shapes_unpack_f64_best(packed, points, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_shapes_unpack_f32_best(packed, points, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_shapes_unpack_f16_best(packed, points, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_shapes_unpack has no kernel for this dtype");
}

/**
 *  @brief Superposed RMSDs between every row pack and every column pack of equally many points.
 *  @param[in] a_packed The row packs, one per row cloud.
 *  @param[in] rows Number of row packs.
 *  @param[in] b_packed The column packs, one per column cloud.
 *  @param[in] columns Number of column packs.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 *  @param[in] results_stride Bytes between the rows of @p results.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @par Statuses:
 *  - @c misaligned_k
 *  - @c unexpected_dimensions_k The packs differ in points.
 *  - @c missing_kernel_k
 */
template <a_shapes_dtype in_type_>
status_t rmsds_packed(void const *const *a_packed, std::size_t rows, void const *const *b_packed, std::size_t columns,
                      typename in_type_::shapes_distance_t *results, std::size_t results_stride,
                      nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_rmsds_packed_f64_best(a_packed, rows, b_packed, columns, &results->raw_,
                                                              results_stride, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_rmsds_packed_f32_best(a_packed, rows, b_packed, columns, &results->raw_,
                                                              results_stride, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_rmsds_packed_f16_best(a_packed, rows, b_packed, columns, &results->raw_,
                                                              results_stride, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_rmsds_packed has no kernel for this dtype");
}

/**
 *  @brief Hausdorff distances between every row and every column pack, the best of 8 axis flips.
 *  @param[in] a_packed The row packs, one per row cloud.
 *  @param[in] rows Number of row packs.
 *  @param[in] b_packed The column packs, one per column cloud.
 *  @param[in] columns Number of column packs.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 *  @param[in] results_stride Bytes between the rows of @p results.
 *  @param[out] rotations Null, or the @b [rows,columns,3,3] rotations taking each column cloud onto
 *      each row cloud through the centroids.
 *  @param[in] limit Distances up to it are exact; past it, a pair may stop early and report a lower
 *      bound that also lies past it, with that bound's flip in @p rotations. Infinity scores all.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
template <a_shapes_dtype in_type_>
status_t hausdorffs_packed(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                           std::size_t columns, typename in_type_::shapes_distance_t *results,
                           std::size_t results_stride, f32_t *rotations, typename in_type_::shapes_distance_t limit,
                           nk_capability_t capabilities = default_capabilities(),
                           nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(
            nk_hausdorffs_packed_f64_best(a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                                          reinterpret_cast<nk_f32_t *>(rotations), &limit.raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(
            nk_hausdorffs_packed_f32_best(a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                                          reinterpret_cast<nk_f32_t *>(rotations), &limit.raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(
            nk_hausdorffs_packed_f16_best(a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                                          reinterpret_cast<nk_f32_t *>(rotations), &limit.raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_hausdorffs_packed has no kernel for this dtype");
}

/**
 *  @brief Chamfer distances between every row pack and every column pack, the best of 8 axis flips.
 *  @param[in] a_packed The row packs, one per row cloud.
 *  @param[in] rows Number of row packs.
 *  @param[in] b_packed The column packs, one per column cloud.
 *  @param[in] columns Number of column packs.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 *  @param[in] results_stride Bytes between the rows of @p results.
 *  @param[out] rotations Null, or the @b [rows,columns,3,3] rotations taking each column cloud onto
 *      each row cloud through the centroids.
 *  @param[in] limit Distances up to it are exact; past it, a pair may stop early and report a lower
 *      bound that also lies past it, with that bound's flip in @p rotations. Infinity scores all.
 *  @param[in,out] workspace Scratch of what @c chamfers_workspace_size reports, or more.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
template <a_shapes_dtype in_type_>
status_t chamfers_packed(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                         std::size_t columns, typename in_type_::shapes_distance_t *results, std::size_t results_stride,
                         f32_t *rotations, typename in_type_::shapes_distance_t limit, std::span<std::byte> workspace,
                         nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_chamfers_packed_f64_best(
            a_packed, rows, b_packed, columns, &results->raw_, results_stride, reinterpret_cast<nk_f32_t *>(rotations),
            &limit.raw_, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_chamfers_packed_f32_best(
            a_packed, rows, b_packed, columns, &results->raw_, results_stride, reinterpret_cast<nk_f32_t *>(rotations),
            &limit.raw_, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_chamfers_packed_f16_best(
            a_packed, rows, b_packed, columns, &results->raw_, results_stride, reinterpret_cast<nk_f32_t *>(rotations),
            &limit.raw_, workspace.data(), workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_chamfers_packed has no kernel for this dtype");
}

/**
 *  @brief Returns the workspace bytes of @c chamfers_packed between clouds of up to @p row_points
 *      and @p column_points points.
 *  @param[in] row_points Most points in any row cloud.
 *  @param[in] column_points Most points in any column cloud.
 *  @param[in] capabilities Capabilities to pick from.
 */
template <a_shapes_dtype in_type_>
expected<std::size_t> chamfers_workspace_size(std::size_t row_points, std::size_t column_points,
                                              nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_chamfers_workspace_size_f64_best(row_points, column_points, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_chamfers_workspace_size_f32_best(row_points, column_points, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_chamfers_workspace_size_f16_best(row_points, column_points, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_chamfers_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Returns the workspace bytes of @c wassersteinprocrustes_packed between clouds of up to
 *      @p row_points and @p column_points points.
 *  @param[in] row_points Most points in any row cloud.
 *  @param[in] column_points Most points in any column cloud.
 *  @param[in] capabilities Capabilities to pick from.
 */
template <a_shapes_dtype in_type_>
expected<std::size_t> wassersteinprocrustes_workspace_size(
    std::size_t row_points, std::size_t column_points, nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_wassersteinprocrustes_workspace_size_f64_best(row_points, column_points, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_wassersteinprocrustes_workspace_size_f32_best(row_points, column_points, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_wassersteinprocrustes_workspace_size_f16_best(row_points, column_points, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_wassersteinprocrustes_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Wasserstein-Procrustes distances between every row pack and every column pack.
 *  @param[in] a_packed The row packs, one per row cloud.
 *  @param[in] rows Number of row packs.
 *  @param[in] b_packed The column packs, one per column cloud.
 *  @param[in] columns Number of column packs.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 *  @param[in] results_stride Bytes between the rows of @p results.
 *  @param[out] rotations Null, or the @b [rows,columns,3,3] rotations taking each column cloud onto
 *      each row cloud through the centroids.
 *  @param[in] workspace Scratch aligned to @c nk_default_alignment_k, of what
 *      @c wassersteinprocrustes_workspace_size reports for the largest clouds, with the same
 *      capabilities, or more.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  The other parameters match @c chamfers_packed.
 *
 *  @par Statuses:
 *  - @c misaligned_k
 *  - @c unexpected_dimensions_k A short workspace.
 *  - @c missing_kernel_k
 */
template <a_shapes_dtype in_type_>
status_t wassersteinprocrustes_packed(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                                      std::size_t columns, typename in_type_::shapes_distance_t *results,
                                      std::size_t results_stride, f32_t *rotations, std::span<std::byte> workspace,
                                      nk_capability_t capabilities = default_capabilities(),
                                      nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_wassersteinprocrustes_packed_f64_best(
            a_packed, rows, b_packed, columns, &results->raw_, results_stride, reinterpret_cast<nk_f32_t *>(rotations),
            workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_wassersteinprocrustes_packed_f32_best(
            a_packed, rows, b_packed, columns, &results->raw_, results_stride, reinterpret_cast<nk_f32_t *>(rotations),
            workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_wassersteinprocrustes_packed_f16_best(
            a_packed, rows, b_packed, columns, &results->raw_, results_stride, reinterpret_cast<nk_f32_t *>(rotations),
            workspace.data(), workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_wassersteinprocrustes_packed has no kernel for this dtype");
}

/** The workspace bytes of @c gromovwassersteins_packed, queried like the Procrustes workspace. */
template <a_shapes_dtype in_type_>
expected<std::size_t> gromovwassersteins_workspace_size(
    std::size_t row_points, std::size_t column_points, nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_gromovwassersteins_workspace_size_f64_best(row_points, column_points, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_gromovwassersteins_workspace_size_f32_best(row_points, column_points, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_gromovwassersteins_workspace_size_f16_best(row_points, column_points, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_gromovwassersteins_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** Gromov-Wasserstein distances between every row and column pack, with the parameters of
 *  @c wassersteinprocrustes_packed and the workspace of @c gromovwassersteins_workspace_size. */
template <a_shapes_dtype in_type_>
status_t gromovwassersteins_packed(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                                   std::size_t columns, typename in_type_::shapes_distance_t *results,
                                   std::size_t results_stride, f32_t *rotations, std::span<std::byte> workspace,
                                   nk_capability_t capabilities = default_capabilities(),
                                   nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_gromovwassersteins_packed_f64_best(
            a_packed, rows, b_packed, columns, &results->raw_, results_stride, reinterpret_cast<nk_f32_t *>(rotations),
            workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_gromovwassersteins_packed_f32_best(
            a_packed, rows, b_packed, columns, &results->raw_, results_stride, reinterpret_cast<nk_f32_t *>(rotations),
            workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_gromovwassersteins_packed_f16_best(
            a_packed, rows, b_packed, columns, &results->raw_, results_stride, reinterpret_cast<nk_f32_t *>(rotations),
            workspace.data(), workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_gromovwassersteins_packed has no kernel for this dtype");
}

/**
 *  @brief Packs every cloud of @p clouds, each @b [points,3] interleaved, into one owner.
 *  @param[in] clouds The point clouds to pack, one pack per cloud.
 *  @param[in] alloc Source of the packs' blocks.
 *  @param[in] capabilities Capabilities to pack with; measure with the same ones.
 *  @return The packs.
 *
 *  @par Statuses:
 *  - @c unexpected_dimensions_k A cloud's length is not a multiple of 3.
 *  - @c missing_kernel_k
 */
template <a_shapes_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
expected<shapes_packed<value_type_, allocator_type_>> shapes_pack(
    std::span<std::span<value_type_ const> const> clouds, allocator_type_ alloc = {},
    nk_capability_t capabilities = default_capabilities()) noexcept {
    auto const pack_size = [capabilities](nk_size_t points, nk_size_t *bytes) noexcept {
        auto const [size, status] = shapes_pack_size<value_type_>(points, capabilities);
        *bytes = size;
        return status;
    };
    auto const pack = [capabilities](typename value_type_::raw_t const *cloud, nk_size_t points, void *packed,
                                     nk_stream_t) noexcept {
        return shapes_pack<value_type_>(cloud, points, packed, capabilities);
    };
    return shapes_packed<value_type_, allocator_type_>::make(clouds, alloc, pack_size, pack);
}

/** Packs every cloud like @c shapes_pack, each divided by its RMS distance from the centroid. */
template <a_shapes_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
expected<shapes_packed<value_type_, allocator_type_>> shapes_pack_normalized(
    std::span<std::span<value_type_ const> const> clouds, allocator_type_ alloc = {},
    nk_capability_t capabilities = default_capabilities()) noexcept {
    auto const pack_size = [capabilities](nk_size_t points, nk_size_t *bytes) noexcept {
        auto const [size, status] = shapes_pack_size<value_type_>(points, capabilities);
        *bytes = size;
        return status;
    };
    auto const pack = [capabilities](typename value_type_::raw_t const *cloud, nk_size_t points, void *packed,
                                     nk_stream_t) noexcept {
        return shapes_pack_normalized<value_type_>(cloud, points, packed, capabilities);
    };
    return shapes_packed<value_type_, allocator_type_>::make(clouds, alloc, pack_size, pack);
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_SHAPES_HPP
