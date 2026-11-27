/**
 *  @file cpp/numkong/geospatial.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ wrappers for the geospatial kernels.
 *
 *  Every wrapper dispatches to the best kernel of the capabilities it is given.
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_GEOSPATIAL_HPP
#define NUMKONG_GEOSPATIAL_HPP

#include <cstddef> // `std::size_t`

#include <type_traits> // `std::conditional_t`, `std::is_same_v`

#include "numkong/geospatial.h"

#include "numkong/types.hpp"
#include "numkong/devices.hpp" // `default_capabilities`

namespace ashvardanian::numkong {

/**
 *  @brief Squared Euclidean distances between every row point and every column point in 3D.
 *  @param[in] a The @p rows points, 3 interleaved coordinates each.
 *  @param[in] b The @p columns points, 3 interleaved coordinates each.
 *  @param[in] rows Number of row points.
 *  @param[in] columns Number of column points.
 *  @param[out] results The @b [rows,columns] squares, @p results_stride bytes apart, exact @c u64_t
 *      ones for @c i32_t points and @c f64_t ones for @c f64_t points.
 *  @param[in] results_stride Bytes between the rows of @p results.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
template <a_numeric_dtype in_type_>
status_t sqeuclideans3d(in_type_ const *a, std::size_t rows, in_type_ const *b, std::size_t columns,
                        std::conditional_t<std::is_same_v<in_type_, i32_t>, u64_t, f64_t> *results,
                        std::size_t results_stride, nk_capability_t capabilities = default_capabilities(),
                        nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, i32_t>)
        return static_cast<status_t>(nk_sqeuclideans3d_i32_best(&a->raw_, rows, &b->raw_, columns, &results->raw_,
                                                                results_stride, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_sqeuclideans3d_f64_best(&a->raw_, rows, &b->raw_, columns, &results->raw_,
                                                                results_stride, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_sqeuclideans3d has no kernel for this dtype");
}

/**
 *  @brief Returns the bytes a Haversine pack of @p points takes.
 *  @param[in] points Number of points in the pack.
 *  @param[in] capabilities Capabilities to pick from.
 */
inline expected<std::size_t> haversines_pack_size(std::size_t points,
                                                  nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t const status = nk_haversines_pack_size_f64_best(points, capabilities, &bytes);
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs points as unit vectors on the mean sphere, for @c haversines_packed.
 *  @param[in] latitudes The @p count latitudes, in radians.
 *  @param[in] longitudes The @p count longitudes, in radians.
 *  @param[in] count Number of points.
 *  @param[out] packed At least @c haversines_pack_size bytes, aligned to @c nk_default_alignment_k.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
inline status_t haversines_pack(f64_t const *latitudes, f64_t const *longitudes, std::size_t count, void *packed,
                                nk_capability_t capabilities = default_capabilities(),
                                nk_stream_t stream = nullptr) noexcept {
    return static_cast<status_t>(
        nk_haversines_pack_f64_best(&latitudes->raw_, &longitudes->raw_, count, packed, capabilities, stream));
}

/**
 *  @brief Great-circle metres between every row and every column point of two Haversine packs.
 *  @param[in] a_packed The row points' Haversine pack.
 *  @param[in] b_packed The column points' Haversine pack.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 *  @param[in] results_stride Bytes between the rows of @p results.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
inline status_t haversines_packed(void const *a_packed, void const *b_packed, f64_t *results,
                                  std::size_t results_stride, nk_capability_t capabilities = default_capabilities(),
                                  nk_stream_t stream = nullptr) noexcept {
    return static_cast<status_t>(
        nk_haversines_packed_f64_best(a_packed, b_packed, &results->raw_, results_stride, capabilities, stream));
}

/** Returns the bytes a Vincenty pack of @p points takes, as @c haversines_pack_size does. */
inline expected<std::size_t> vincentys_pack_size(std::size_t points,
                                                 nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t const status = nk_vincentys_pack_size_f64_best(points, capabilities, &bytes);
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** Packs points as their longitudes and the sines and cosines of their reduced latitudes, for
 *  @c vincentys_packed, taking what @c haversines_pack takes. */
inline status_t vincentys_pack(f64_t const *latitudes, f64_t const *longitudes, std::size_t count, void *packed,
                               nk_capability_t capabilities = default_capabilities(),
                               nk_stream_t stream = nullptr) noexcept {
    return static_cast<status_t>(
        nk_vincentys_pack_f64_best(&latitudes->raw_, &longitudes->raw_, count, packed, capabilities, stream));
}

/** Geodesic metres on the WGS-84 ellipsoid between every row and every column point of two Vincenty
 *  packs, taking what @c haversines_packed takes. */
inline status_t vincentys_packed(void const *a_packed, void const *b_packed, f64_t *results, std::size_t results_stride,
                                 nk_capability_t capabilities = default_capabilities(),
                                 nk_stream_t stream = nullptr) noexcept {
    return static_cast<status_t>(
        nk_vincentys_packed_f64_best(a_packed, b_packed, &results->raw_, results_stride, capabilities, stream));
}

/**
 *  @brief Returns the bytes the terrain pack of a @p rows × @p columns grid takes.
 *  @param[in] rows Number of grid rows.
 *  @param[in] columns Number of grid columns.
 *  @param[in] capabilities Capabilities to pick from.
 */
inline expected<std::size_t> terrain_pack_size(std::size_t rows, std::size_t columns,
                                               nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t const status = nk_terrain_pack_size_f32_best(rows, columns, capabilities, &bytes);
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs the maxima of every 2ᵏ × 2ᵏ block of a north-up grid, its shape and scale.
 *  @param[in] heights The @b [rows,columns] heights in metres, @p heights_stride bytes apart.
 *  @param[in] rows Number of grid rows.
 *  @param[in] columns Number of grid columns.
 *  @param[in] heights_stride Bytes between the rows of @p heights.
 *  @param[in] row_metres Metres between adjacent rows.
 *  @param[in] column_metres Metres between adjacent columns.
 *  @param[out] packed At least @c terrain_pack_size bytes, aligned to @c nk_default_alignment_k.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
inline status_t terrain_pack(f32_t const *heights, std::size_t rows, std::size_t columns, std::size_t heights_stride,
                             f64_t row_metres, f64_t column_metres, void *packed,
                             nk_capability_t capabilities = default_capabilities(),
                             nk_stream_t stream = nullptr) noexcept {
    return static_cast<status_t>(nk_terrain_pack_f32_best(&heights->raw_, rows, columns, heights_stride,
                                                          row_metres.raw_, column_metres.raw_, packed, capabilities,
                                                          stream));
}

/** Bilinear terrain heights under @p count interleaved [column, row] points, NaN off the grid,
 *  dispatched as @c terrain_pack is. */
template <typename in_type_ = f32_t>
status_t terrain_heights_packed(std::type_identity_t<in_type_> const *heights, void const *packed,
                                std::type_identity_t<in_type_> const *points, std::size_t count,
                                std::type_identity_t<in_type_> *results,
                                nk_capability_t capabilities = default_capabilities(),
                                nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_terrain_heights_packed_f32_best(&heights->raw_, packed, &points->raw_, count,
                                                                        &results->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_terrain_heights_packed has no kernel for this dtype");
}

/** Clearances in metres from every row viewpoint to every column target over one terrain, as
 *  @c nk_clearances_packed_f32_best defines them, dispatched as @c terrain_pack is. */
template <typename in_type_ = f32_t>
status_t clearances_packed(std::type_identity_t<in_type_> const *heights, void const *packed,
                           std::type_identity_t<in_type_> const *viewpoints,
                           std::type_identity_t<in_type_> const *views, std::size_t rows,
                           std::type_identity_t<in_type_> const *targets, std::size_t columns, f64_t refraction,
                           std::type_identity_t<in_type_> *results, std::size_t results_stride,
                           nk_capability_t capabilities = default_capabilities(),
                           nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_clearances_packed_f32_best(
            &heights->raw_, packed, &viewpoints->raw_, views ? &views->raw_ : nullptr, rows, &targets->raw_, columns,
            refraction.raw_, &results->raw_, results_stride, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_clearances_packed has no kernel for this dtype");
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_GEOSPATIAL_HPP
