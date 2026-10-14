/**
 *  @file include/numkong/geospatial.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ wrappers for the geospatial kernels, and reference geodesics in any precision.
 *
 *  Every wrapper dispatches to the best kernel of the capabilities it is given and falls back to
 *  its serial kernel, or to its reference, when they are zero. The Haversine and Vincenty
 *  references evaluate the textbook formulas in a type of the caller's choosing, such as the
 *  double-double @c f118_t, to serve as the oracle of the packed kernels.
 */
#ifndef NUMKONG_GEOSPATIAL_HPP
#define NUMKONG_GEOSPATIAL_HPP

#include <cstddef> // `std::size_t`

#include <type_traits> // `std::conditional_t`, `std::is_same_v`

#include "numkong/geospatial.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Great-circle metres between every row and every column point on the mean sphere,
 *      through the haversine 2R · atan2(√h, √(1 − h)) evaluated in @p precision_type_,
 *      where h = sin²(Δφ / 2) + cos φ₁ · cos φ₂ · sin²(Δλ / 2).
 *  @param[in] a_latitudes, @p a_longitudes The @p rows row points, in radians.
 *  @param[in] b_latitudes, @p b_longitudes The @p columns column points, in radians.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 */
template <numeric_dtype precision_type_>
void haversines(f64_t const *a_latitudes, f64_t const *a_longitudes, std::size_t rows, f64_t const *b_latitudes,
                f64_t const *b_longitudes, std::size_t columns, f64_t *results, std::size_t results_stride) noexcept {
    precision_type_ const earth_radius = precision_type_(NUMKONG_EARTH_MEAN_RADIUS);
    precision_type_ const half = precision_type_(0.5), one = precision_type_(1.0), two = precision_type_(2.0);
    for (std::size_t row = 0; row != rows; ++row) {
        f64_t *output = reinterpret_cast<f64_t *>(reinterpret_cast<char *>(results) + row * results_stride);
        precision_type_ const first_latitude = precision_type_(a_latitudes[row]);
        precision_type_ const first_longitude = precision_type_(a_longitudes[row]);
        for (std::size_t column = 0; column != columns; ++column) {
            precision_type_ const second_latitude = precision_type_(b_latitudes[column]);
            precision_type_ const second_longitude = precision_type_(b_longitudes[column]);
            precision_type_ const sin_latitude_half = ((second_latitude - first_latitude) * half).sin();
            precision_type_ const sin_longitude_half = ((second_longitude - first_longitude) * half).sin();
            precision_type_ const haversine = sin_latitude_half * sin_latitude_half +
                                              first_latitude.cos() * second_latitude.cos() * sin_longitude_half *
                                                  sin_longitude_half;
            precision_type_ const angle = two * haversine.sqrt().atan2((one - haversine).sqrt());
            output[column] = f64_t(static_cast<double>(earth_radius * angle));
        }
    }
}

/** Geodesic metres on the WGS-84 ellipsoid between every row and every column point, through
 *  Vincenty's inverse formula in @p precision_type_, with the reduced latitudes taken through
 *  atan2 to stay finite at the poles; parameters as @c haversines. */
template <numeric_dtype precision_type_>
void vincentys(f64_t const *a_latitudes, f64_t const *a_longitudes, std::size_t rows, f64_t const *b_latitudes,
               f64_t const *b_longitudes, std::size_t columns, f64_t *results, std::size_t results_stride) noexcept {
    using precision_t = precision_type_;
    precision_t const equatorial_radius = precision_t(NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    precision_t const flattening = precision_t(1.0) / precision_t(NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    precision_t const polar_radius = equatorial_radius * (precision_t(1.0) - flattening);
    precision_t const convergence_threshold = precision_t(NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    auto reduced = [&](f64_t latitude) noexcept {
        precision_t const angle = precision_t(latitude);
        return ((precision_t(1.0) - flattening) * angle.sin()).atan2(angle.cos());
    };

    for (std::size_t row = 0; row != rows; ++row) {
        f64_t *output = reinterpret_cast<f64_t *>(reinterpret_cast<char *>(results) + row * results_stride);
        precision_t const reduced_first = reduced(a_latitudes[row]);
        precision_t const sin_reduced_first = reduced_first.sin(), cos_reduced_first = reduced_first.cos();
        for (std::size_t column = 0; column != columns; ++column) {
            precision_t const reduced_second = reduced(b_latitudes[column]);
            precision_t const sin_reduced_second = reduced_second.sin(), cos_reduced_second = reduced_second.cos();
            precision_t const longitude_difference = precision_t(b_longitudes[column]) - precision_t(a_longitudes[row]);

            precision_t lambda = longitude_difference;
            precision_t sin_angular_distance, cos_angular_distance, angular_distance;
            precision_t cos_squared_azimuth, cos_double_angular_midpoint;
            bool coincident = false;
            for (int iteration = 0; iteration != NUMKONG_VINCENTY_MAX_ITERATIONS; ++iteration) {
                precision_t const sin_lambda = lambda.sin(), cos_lambda = lambda.cos();
                precision_t const cross_term = cos_reduced_second * sin_lambda;
                precision_t const mixed_term = cos_reduced_first * sin_reduced_second -
                                               sin_reduced_first * cos_reduced_second * cos_lambda;
                sin_angular_distance = (cross_term * cross_term + mixed_term * mixed_term).sqrt();
                if (sin_angular_distance == precision_t(0.0)) {
                    coincident = true;
                    break;
                }
                cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                       cos_reduced_first * cos_reduced_second * cos_lambda;
                angular_distance = sin_angular_distance.atan2(cos_angular_distance);
                precision_t const sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda /
                                                sin_angular_distance;
                cos_squared_azimuth = precision_t(1.0) - sin_azimuth * sin_azimuth;
                cos_double_angular_midpoint = cos_squared_azimuth != precision_t(0.0)
                                                  ? cos_angular_distance - precision_t(2.0) * sin_reduced_first *
                                                                               sin_reduced_second / cos_squared_azimuth
                                                  : precision_t(0.0);
                precision_t const correction_factor =
                    flattening / precision_t(16.0) * cos_squared_azimuth *
                    (precision_t(4.0) + flattening * (precision_t(4.0) - precision_t(3.0) * cos_squared_azimuth));
                precision_t const lambda_previous = lambda;
                lambda = longitude_difference +
                         (precision_t(1.0) - correction_factor) * flattening * sin_azimuth *
                             (angular_distance +
                              correction_factor * sin_angular_distance *
                                  (cos_double_angular_midpoint +
                                   correction_factor * cos_angular_distance *
                                       (precision_t(-1.0) +
                                        precision_t(2.0) * cos_double_angular_midpoint * cos_double_angular_midpoint)));
                if (!((lambda - lambda_previous).abs() > convergence_threshold)) break;
            }
            if (coincident) {
                output[column] = f64_t(0.0);
                continue;
            }
            precision_t const u_squared = cos_squared_azimuth *
                                          (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                          (polar_radius * polar_radius);
            precision_t const series_a =
                precision_t(1.0) +
                u_squared / precision_t(16384.0) *
                    (precision_t(4096.0) +
                     u_squared *
                         (precision_t(-768.0) + u_squared * (precision_t(320.0) - precision_t(175.0) * u_squared)));
            precision_t const series_b =
                u_squared / precision_t(1024.0) *
                (precision_t(256.0) +
                 u_squared * (precision_t(-128.0) + u_squared * (precision_t(74.0) - precision_t(47.0) * u_squared)));
            precision_t const angular_correction =
                series_b * sin_angular_distance *
                (cos_double_angular_midpoint +
                 series_b / precision_t(4.0) *
                     (cos_angular_distance * (precision_t(-1.0) + precision_t(2.0) * cos_double_angular_midpoint *
                                                                      cos_double_angular_midpoint) -
                      series_b / precision_t(6.0) * cos_double_angular_midpoint *
                          (precision_t(-3.0) + precision_t(4.0) * sin_angular_distance * sin_angular_distance) *
                          (precision_t(-3.0) +
                           precision_t(4.0) * cos_double_angular_midpoint * cos_double_angular_midpoint)));
            output[column] = f64_t(
                static_cast<double>(polar_radius * series_a * (angular_distance - angular_correction)));
        }
    }
}

/**
 *  @brief Squared Euclidean distances between every row point and every column point in 3D.
 *  @param[in] a, b The @p rows and @p columns points, 3 interleaved coordinates each.
 *  @param[out] results The @b [rows,columns] squares, @p results_stride bytes apart.
 *  @param[in] capabilities Capabilities to pick from, or zero for the reference in @p result_type_.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *  @tparam result_type_ The reference's arithmetic; the kernels output exact @c u64_t squares for
 *      @c i32_t and @c f64_t ones for @c f64_t.
 */
template <numeric_dtype in_type_,
          numeric_dtype result_type_ = std::conditional_t<std::is_same_v<in_type_, i32_t>, u64_t, f64_t>>
    requires(std::is_same_v<in_type_, i32_t> || std::is_same_v<in_type_, f64_t>)
status_t sqeuclideans3d(in_type_ const *a, std::size_t rows, in_type_ const *b, std::size_t columns,
                        result_type_ *results, std::size_t results_stride,
                        nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, i32_t> && std::is_same_v<result_type_, u64_t>)
            return static_cast<status_t>(nk_sqeuclideans3d_i32_best(&a->raw_, rows, &b->raw_, columns, &results->raw_,
                                                                    results_stride, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f64_t> && std::is_same_v<result_type_, f64_t>)
            return static_cast<status_t>(nk_sqeuclideans3d_f64_best(&a->raw_, rows, &b->raw_, columns, &results->raw_,
                                                                    results_stride, capabilities, stream));
    }
    for (std::size_t row = 0; row != rows; ++row) {
        result_type_ *output = reinterpret_cast<result_type_ *>(reinterpret_cast<char *>(results) +
                                                                row * results_stride);
        for (std::size_t column = 0; column != columns; ++column) {
            result_type_ sum {};
            for (std::size_t axis = 0; axis != 3; ++axis) {
                if constexpr (std::is_same_v<in_type_, i32_t>) {
                    std::int64_t const difference = std::int64_t(a[row * 3 + axis].raw_) -
                                                    std::int64_t(b[column * 3 + axis].raw_);
                    std::uint64_t const magnitude = std::uint64_t(difference < 0 ? -difference : difference);
                    sum = result_type_(sum + result_type_(magnitude * magnitude));
                }
                else {
                    result_type_ const difference = result_type_(a[row * 3 + axis].raw_) -
                                                    result_type_(b[column * 3 + axis].raw_);
                    sum = sum + difference * difference;
                }
            }
            output[column] = sum;
        }
    }
    return status_t::success_k;
}

/**
 *  @brief Returns the bytes a Haversine pack of @p points takes.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial layout.
 */
inline expected<std::size_t> haversines_pack_size(std::size_t points,
                                                  nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t const status = capabilities ? nk_haversines_pack_size_f64_best(points, capabilities, &bytes)
                                            : nk_haversines_pack_size_f64_serial(points, &bytes);
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs points as unit vectors on the mean sphere, for @c haversines_packed.
 *  @param[in] latitudes, longitudes The @p count points, in radians.
 *  @param[out] packed At least @c haversines_pack_size bytes, aligned to @c nk_default_alignment_k.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial kernel.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
inline status_t haversines_pack(f64_t const *latitudes, f64_t const *longitudes, std::size_t count, void *packed,
                                nk_capability_t capabilities = default_capabilities(),
                                nk_stream_t stream = nullptr) noexcept {
    if (capabilities)
        return static_cast<status_t>(
            nk_haversines_pack_f64_best(&latitudes->raw_, &longitudes->raw_, count, packed, capabilities, stream));
    return static_cast<status_t>(
        nk_haversines_pack_f64_serial(&latitudes->raw_, &longitudes->raw_, count, packed, stream));
}

/**
 *  @brief Great-circle metres between every row and every column point of two Haversine packs.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial kernel.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
inline status_t haversines_packed(void const *a_packed, void const *b_packed, f64_t *results,
                                  std::size_t results_stride, nk_capability_t capabilities = default_capabilities(),
                                  nk_stream_t stream = nullptr) noexcept {
    if (capabilities)
        return static_cast<status_t>(
            nk_haversines_packed_f64_best(a_packed, b_packed, &results->raw_, results_stride, capabilities, stream));
    return static_cast<status_t>(
        nk_haversines_packed_f64_serial(a_packed, b_packed, &results->raw_, results_stride, stream));
}

/** Returns the bytes a Vincenty pack of @p points takes, as @c haversines_pack_size does. */
inline expected<std::size_t> vincentys_pack_size(std::size_t points,
                                                 nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t const status = capabilities ? nk_vincentys_pack_size_f64_best(points, capabilities, &bytes)
                                            : nk_vincentys_pack_size_f64_serial(points, &bytes);
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** Packs points as their longitudes and the sines and cosines of their reduced latitudes, for
 *  @c vincentys_packed, taking what @c haversines_pack takes. */
inline status_t vincentys_pack(f64_t const *latitudes, f64_t const *longitudes, std::size_t count, void *packed,
                               nk_capability_t capabilities = default_capabilities(),
                               nk_stream_t stream = nullptr) noexcept {
    if (capabilities)
        return static_cast<status_t>(
            nk_vincentys_pack_f64_best(&latitudes->raw_, &longitudes->raw_, count, packed, capabilities, stream));
    return static_cast<status_t>(
        nk_vincentys_pack_f64_serial(&latitudes->raw_, &longitudes->raw_, count, packed, stream));
}

/** Geodesic metres on the WGS-84 ellipsoid between every row and every column point of two Vincenty
 *  packs, taking what @c haversines_packed takes. */
inline status_t vincentys_packed(void const *a_packed, void const *b_packed, f64_t *results, std::size_t results_stride,
                                 nk_capability_t capabilities = default_capabilities(),
                                 nk_stream_t stream = nullptr) noexcept {
    if (capabilities)
        return static_cast<status_t>(
            nk_vincentys_packed_f64_best(a_packed, b_packed, &results->raw_, results_stride, capabilities, stream));
    return static_cast<status_t>(
        nk_vincentys_packed_f64_serial(a_packed, b_packed, &results->raw_, results_stride, stream));
}

/**
 *  @brief Returns the bytes the terrain pack of a @p rows × @p columns grid takes.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial layout.
 */
inline expected<std::size_t> terrain_pack_size(std::size_t rows, std::size_t columns,
                                               nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t const status = capabilities ? nk_terrain_pack_size_f32_best(rows, columns, capabilities, &bytes)
                                            : nk_terrain_pack_size_f32_serial(rows, columns, &bytes);
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs the maxima of every 2ᵏ × 2ᵏ block of a north-up grid, its shape and scale.
 *  @param[in] heights The @b [rows,columns] heights in metres, @p heights_stride bytes apart.
 *  @param[out] packed At least @c terrain_pack_size bytes, aligned to @c nk_default_alignment_k.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial kernel.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
inline status_t terrain_pack(f32_t const *heights, std::size_t rows, std::size_t columns, std::size_t heights_stride,
                             f64_t row_metres, f64_t column_metres, void *packed,
                             nk_capability_t capabilities = default_capabilities(),
                             nk_stream_t stream = nullptr) noexcept {
    if (capabilities)
        return static_cast<status_t>(nk_terrain_pack_f32_best(&heights->raw_, rows, columns, heights_stride,
                                                              row_metres.raw_, column_metres.raw_, packed, capabilities,
                                                              stream));
    return static_cast<status_t>(nk_terrain_pack_f32_serial(&heights->raw_, rows, columns, heights_stride,
                                                            row_metres.raw_, column_metres.raw_, packed, stream));
}

/** Bilinear terrain heights under @p count interleaved [column, row] points, NaN off the grid,
 *  dispatched as @c terrain_pack is. */
inline status_t terrain_heights_packed(f32_t const *heights, void const *packed, f32_t const *points, std::size_t count,
                                       f32_t *results, nk_capability_t capabilities = default_capabilities(),
                                       nk_stream_t stream = nullptr) noexcept {
    if (capabilities)
        return static_cast<status_t>(nk_terrain_heights_packed_f32_best(&heights->raw_, packed, &points->raw_, count,
                                                                        &results->raw_, capabilities, stream));
    return static_cast<status_t>(
        nk_terrain_heights_packed_f32_serial(&heights->raw_, packed, &points->raw_, count, &results->raw_, stream));
}

/** Clearances in metres from every row viewpoint to every column target over one terrain, as
 *  @c nk_clearances_packed_f32_best defines them, dispatched as @c terrain_pack is. */
inline status_t clearances_packed(f32_t const *heights, void const *packed, f32_t const *viewpoints, f32_t const *views,
                                  std::size_t rows, f32_t const *targets, std::size_t columns, f64_t refraction,
                                  f32_t *results, std::size_t results_stride,
                                  nk_capability_t capabilities = default_capabilities(),
                                  nk_stream_t stream = nullptr) noexcept {
    nk_f32_t const *raw_views = views ? &views->raw_ : nullptr;
    if (capabilities)
        return static_cast<status_t>(nk_clearances_packed_f32_best(
            &heights->raw_, packed, &viewpoints->raw_, raw_views, rows, &targets->raw_, columns, refraction.raw_,
            &results->raw_, results_stride, capabilities, stream));
    return static_cast<status_t>(nk_clearances_packed_f32_serial(&heights->raw_, packed, &viewpoints->raw_, raw_views,
                                                                 rows, &targets->raw_, columns, refraction.raw_,
                                                                 &results->raw_, results_stride, stream));
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_GEOSPATIAL_HPP
