/**
 *  @file cpp/numkong/geospatial.hpp
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

#include <limits>      // `std::numeric_limits`
#include <type_traits> // `std::conditional_t`, `std::is_same_v`

#include "numkong/geospatial.h"
#include "numkong/geospatial/serial.h" // `nk_terrain_packed_header_t`

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Great-circle metres between every row and every column point on the mean sphere,
 *      through the haversine 2R · atan2(√h, √(1 − h)) evaluated in @p precision_type_,
 *      where h = sin²(Δφ / 2) + cos φ₁ · cos φ₂ · sin²(Δλ / 2).
 *  @param[in] a_latitudes The @p rows row latitudes, in radians.
 *  @param[in] a_longitudes The @p rows row longitudes, in radians.
 *  @param[in] rows Number of row points.
 *  @param[in] b_latitudes The @p columns column latitudes, in radians.
 *  @param[in] b_longitudes The @p columns column longitudes, in radians.
 *  @param[in] columns Number of column points.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 *  @param[in] results_stride Bytes between the rows of @p results.
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

/** Geodesic metres on the WGS-84 ellipsoid between two points, given the sines and cosines of
 *  their reduced latitudes and the @p longitude_difference from the first to the second, through
 *  Vincenty's inverse formula in @p precision_type_. */
template <numeric_dtype precision_type_>
precision_type_ vincenty_reference_(precision_type_ sin_reduced_first, precision_type_ cos_reduced_first,
                                    precision_type_ sin_reduced_second, precision_type_ cos_reduced_second,
                                    precision_type_ longitude_difference) noexcept {
    using precision_t = precision_type_;
    precision_t const equatorial_radius = precision_t(NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    precision_t const flattening = precision_t(1.0) / precision_t(NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    precision_t const polar_radius = equatorial_radius * (precision_t(1.0) - flattening);
    precision_t const convergence_threshold = precision_t(NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64);
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
        precision_t const sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
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
                     (angular_distance + correction_factor * sin_angular_distance *
                                             (cos_double_angular_midpoint +
                                              correction_factor * cos_angular_distance *
                                                  (precision_t(-1.0) + precision_t(2.0) * cos_double_angular_midpoint *
                                                                           cos_double_angular_midpoint)));
        if (!((lambda - lambda_previous).abs() > convergence_threshold)) break;
    }
    if (coincident) return precision_t(0.0);
    precision_t const u_squared = cos_squared_azimuth *
                                  (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                  (polar_radius * polar_radius);
    precision_t const series_a =
        precision_t(1.0) +
        u_squared / precision_t(16384.0) *
            (precision_t(4096.0) +
             u_squared * (precision_t(-768.0) + u_squared * (precision_t(320.0) - precision_t(175.0) * u_squared)));
    precision_t const series_b =
        u_squared / precision_t(1024.0) *
        (precision_t(256.0) +
         u_squared * (precision_t(-128.0) + u_squared * (precision_t(74.0) - precision_t(47.0) * u_squared)));
    precision_t const angular_correction =
        series_b * sin_angular_distance *
        (cos_double_angular_midpoint +
         series_b / precision_t(4.0) *
             (cos_angular_distance *
                  (precision_t(-1.0) + precision_t(2.0) * cos_double_angular_midpoint * cos_double_angular_midpoint) -
              series_b / precision_t(6.0) * cos_double_angular_midpoint *
                  (precision_t(-3.0) + precision_t(4.0) * sin_angular_distance * sin_angular_distance) *
                  (precision_t(-3.0) + precision_t(4.0) * cos_double_angular_midpoint * cos_double_angular_midpoint)));
    return polar_radius * series_a * (angular_distance - angular_correction);
}

/** Geodesic metres on the WGS-84 ellipsoid between every row and every column point, through
 *  Vincenty's inverse formula in @p precision_type_, with the reduced latitudes taken through
 *  atan2 to stay finite at the poles; parameters as @c haversines. */
template <numeric_dtype precision_type_>
void vincentys(f64_t const *a_latitudes, f64_t const *a_longitudes, std::size_t rows, f64_t const *b_latitudes,
               f64_t const *b_longitudes, std::size_t columns, f64_t *results, std::size_t results_stride) noexcept {
    using precision_t = precision_type_;
    precision_t const flattening = precision_t(1.0) / precision_t(NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    auto reduced = [&](f64_t latitude) noexcept {
        precision_t const angle = precision_t(latitude);
        return ((precision_t(1.0) - flattening) * angle.sin()).atan2(angle.cos());
    };

    for (std::size_t row = 0; row != rows; ++row) {
        f64_t *output = reinterpret_cast<f64_t *>(reinterpret_cast<char *>(results) + row * results_stride);
        precision_t const reduced_first = reduced(a_latitudes[row]);
        for (std::size_t column = 0; column != columns; ++column) {
            precision_t const reduced_second = reduced(b_latitudes[column]);
            precision_t const longitude_difference = precision_t(b_longitudes[column]) - precision_t(a_longitudes[row]);
            output[column] = f64_t(static_cast<double>(
                vincenty_reference_<precision_t>(reduced_first.sin(), reduced_first.cos(), reduced_second.sin(),
                                                 reduced_second.cos(), longitude_difference)));
        }
    }
}

/**
 *  @brief Squared Euclidean distances between every row point and every column point in 3D.
 *  @param[in] a The @p rows points, 3 interleaved coordinates each.
 *  @param[in] b The @p columns points, 3 interleaved coordinates each.
 *  @param[in] rows Number of row points.
 *  @param[in] columns Number of column points.
 *  @param[out] results The @b [rows,columns] squares, @p results_stride bytes apart.
 *  @param[in] results_stride Bytes between the rows of @p results.
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

/** Bytes of a Haversine or Vincenty pack of @p points, as @c geospatial/serial.h lays it out. */
constexpr std::size_t geodesics_pack_bytes_(std::size_t points) noexcept {
    return nk_default_alignment_k +
           3 * nk_size_round_up_to_multiple_(points, nk_default_alignment_k / sizeof(f64_t)) * sizeof(f64_t);
}

/** Packs @p count points as unit vectors on the mean sphere, evaluated in @p precision_type_ and
 *  rounded once, by the layout of @c geospatial/serial.h. */
template <numeric_dtype precision_type_>
void haversines_pack_reference_(f64_t const *latitudes, f64_t const *longitudes, std::size_t count,
                                void *packed) noexcept {
    std::size_t const axis_stride = nk_size_round_up_to_multiple_(count, nk_default_alignment_k / sizeof(f64_t));
    nk_geospatial_packed_header_t *header = static_cast<nk_geospatial_packed_header_t *>(packed);
    header->points = count, header->axis_stride = axis_stride;
    f64_t *packed_x = reinterpret_cast<f64_t *>(static_cast<char *>(packed) + nk_default_alignment_k);
    f64_t *packed_y = packed_x + axis_stride, *packed_z = packed_y + axis_stride;
    for (std::size_t point = 0; point != count; ++point) {
        precision_type_ const latitude = precision_type_(latitudes[point]);
        precision_type_ const longitude = precision_type_(longitudes[point]);
        precision_type_ const cos_latitude = latitude.cos();
        packed_x[point] = f64_t(static_cast<double>(cos_latitude * longitude.cos()));
        packed_y[point] = f64_t(static_cast<double>(cos_latitude * longitude.sin()));
        packed_z[point] = f64_t(static_cast<double>(latitude.sin()));
    }
}

/** Packs @p count points as their longitudes and the sines and cosines of their reduced latitudes,
 *  evaluated in @p precision_type_ through atan2 as @c vincentys does, by the layout of
 *  @c geospatial/serial.h. */
template <numeric_dtype precision_type_>
void vincentys_pack_reference_(f64_t const *latitudes, f64_t const *longitudes, std::size_t count,
                               void *packed) noexcept {
    using precision_t = precision_type_;
    precision_t const flattening = precision_t(1.0) / precision_t(NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    std::size_t const axis_stride = nk_size_round_up_to_multiple_(count, nk_default_alignment_k / sizeof(f64_t));
    nk_geospatial_packed_header_t *header = static_cast<nk_geospatial_packed_header_t *>(packed);
    header->points = count, header->axis_stride = axis_stride;
    f64_t *packed_longitudes = reinterpret_cast<f64_t *>(static_cast<char *>(packed) + nk_default_alignment_k);
    f64_t *packed_sines = packed_longitudes + axis_stride, *packed_cosines = packed_sines + axis_stride;
    for (std::size_t point = 0; point != count; ++point) {
        precision_t const latitude = precision_t(latitudes[point]);
        precision_t const reduced = ((precision_t(1.0) - flattening) * latitude.sin()).atan2(latitude.cos());
        packed_longitudes[point] = longitudes[point];
        packed_sines[point] = f64_t(static_cast<double>(reduced.sin()));
        packed_cosines[point] = f64_t(static_cast<double>(reduced.cos()));
    }
}

/** Great-circle metres between every row and every column point of two Haversine packs, through
 *  the chord formula of @c geospatial/serial.h in @p precision_type_. */
template <numeric_dtype precision_type_>
void haversines_packed_reference_(void const *a_packed, void const *b_packed, f64_t *results,
                                  std::size_t results_stride) noexcept {
    using precision_t = precision_type_;
    nk_geospatial_packed_header_t const *a_header = static_cast<nk_geospatial_packed_header_t const *>(a_packed);
    nk_geospatial_packed_header_t const *b_header = static_cast<nk_geospatial_packed_header_t const *>(b_packed);
    f64_t const *a_x = reinterpret_cast<f64_t const *>(static_cast<char const *>(a_packed) + nk_default_alignment_k);
    f64_t const *b_x = reinterpret_cast<f64_t const *>(static_cast<char const *>(b_packed) + nk_default_alignment_k);
    std::size_t const a_stride = static_cast<std::size_t>(a_header->axis_stride);
    std::size_t const b_stride = static_cast<std::size_t>(b_header->axis_stride);
    precision_t const diameter = precision_t(2.0 * NUMKONG_EARTH_MEAN_RADIUS);
    for (std::size_t row = 0; row != a_header->points; ++row) {
        f64_t *output = reinterpret_cast<f64_t *>(reinterpret_cast<char *>(results) + row * results_stride);
        for (std::size_t column = 0; column != b_header->points; ++column) {
            precision_t chord(0.0), span(0.0);
            for (std::size_t axis = 0; axis != 3; ++axis) {
                precision_t const first = precision_t(a_x[axis * a_stride + row]);
                precision_t const second = precision_t(b_x[axis * b_stride + column]);
                chord = chord + (first - second) * (first - second);
                span = span + (first + second) * (first + second);
            }
            output[column] = f64_t(static_cast<double>(diameter * chord.sqrt().atan2(span.sqrt())));
        }
    }
}

/** Geodesic metres on the WGS-84 ellipsoid between every row and every column point of two
 *  Vincenty packs, through @c vincenty_reference_ in @p precision_type_. */
template <numeric_dtype precision_type_>
void vincentys_packed_reference_(void const *a_packed, void const *b_packed, f64_t *results,
                                 std::size_t results_stride) noexcept {
    using precision_t = precision_type_;
    nk_geospatial_packed_header_t const *a_header = static_cast<nk_geospatial_packed_header_t const *>(a_packed);
    nk_geospatial_packed_header_t const *b_header = static_cast<nk_geospatial_packed_header_t const *>(b_packed);
    f64_t const *a_longitudes = reinterpret_cast<f64_t const *>(static_cast<char const *>(a_packed) +
                                                                nk_default_alignment_k);
    f64_t const *b_longitudes = reinterpret_cast<f64_t const *>(static_cast<char const *>(b_packed) +
                                                                nk_default_alignment_k);
    f64_t const *a_sines = a_longitudes + a_header->axis_stride, *a_cosines = a_sines + a_header->axis_stride;
    f64_t const *b_sines = b_longitudes + b_header->axis_stride, *b_cosines = b_sines + b_header->axis_stride;
    for (std::size_t row = 0; row != a_header->points; ++row) {
        f64_t *output = reinterpret_cast<f64_t *>(reinterpret_cast<char *>(results) + row * results_stride);
        for (std::size_t column = 0; column != b_header->points; ++column)
            output[column] = f64_t(static_cast<double>(vincenty_reference_<precision_t>(
                precision_t(a_sines[row]), precision_t(a_cosines[row]), precision_t(b_sines[column]),
                precision_t(b_cosines[column]), precision_t(b_longitudes[column]) - precision_t(a_longitudes[row]))));
    }
}

/** Blocks along one axis of @p cells cells at terrain level @p level. */
constexpr std::size_t terrain_level_blocks_(std::size_t cells, std::size_t level) noexcept {
    return nk_size_divide_round_up_(cells > 1 ? cells - 1 : 1, std::size_t(1) << level);
}

/** Levels of maxima a @p rows × @p columns grid needs, the coarsest holding a single block. */
constexpr std::size_t terrain_levels_(std::size_t rows, std::size_t columns) noexcept {
    std::size_t levels = 0;
    while (terrain_level_blocks_(max_of(rows, columns), levels) > 1) ++levels;
    return levels;
}

/** Bytes the terrain pack of a @p rows × @p columns grid takes, by the layout of
 *  @c geospatial/serial.h. */
constexpr std::size_t terrain_pack_bytes_(std::size_t rows, std::size_t columns) noexcept {
    std::size_t bytes = nk_default_alignment_k;
    for (std::size_t level = 1, levels = terrain_levels_(rows, columns); level <= levels; ++level)
        bytes += terrain_level_blocks_(rows, level) * terrain_level_blocks_(columns, level) * sizeof(f32_t);
    return bytes;
}

/** The largest height in rows @p first_row to @p last_row and columns @p first_column to
 *  @p last_column , both inclusive, skipping NaNs, or negative infinity when all are NaN. */
inline f32_t terrain_window_maximum_(f32_t const *heights, std::size_t heights_stride, std::size_t first_row,
                                     std::size_t last_row, std::size_t first_column, std::size_t last_column) noexcept {
    // NaN never wins a comparison
    f32_t maximum = f32_t::negative_infinity();
    for (std::size_t row = first_row; row <= last_row; ++row) {
        f32_t const *line = reinterpret_cast<f32_t const *>(reinterpret_cast<char const *>(heights) +
                                                            row * heights_stride);
        for (std::size_t column = first_column; column <= last_column; ++column)
            if (maximum < line[column]) maximum = line[column];
    }
    return maximum;
}

/** Packs a grid as @c nk_terrain_packed_header_t lays it out, taking every block's maximum straight
 *  from the heights under it and its ring; parameters as @c terrain_pack. */
inline status_t terrain_pack_reference_(f32_t const *heights, std::size_t rows, std::size_t columns,
                                        std::size_t heights_stride, f64_t row_metres, f64_t column_metres,
                                        void *packed) noexcept {
    if (!rows || !columns) return status_t::unexpected_dimensions_k;
    std::size_t const levels = terrain_levels_(rows, columns);
    nk_terrain_packed_header_t *header = static_cast<nk_terrain_packed_header_t *>(packed);
    header->rows = rows, header->columns = columns, header->heights_stride = heights_stride;
    header->levels = levels, header->row_metres = row_metres.raw_, header->column_metres = column_metres.raw_;
    f32_t *maxima = reinterpret_cast<f32_t *>(static_cast<char *>(packed) + nk_default_alignment_k);
    for (std::size_t level = 1; level <= levels; ++level) {
        std::size_t const span = std::size_t(1) << level;
        std::size_t const block_rows = terrain_level_blocks_(rows, level);
        std::size_t const block_columns = terrain_level_blocks_(columns, level);
        for (std::size_t block_row = 0; block_row != block_rows; ++block_row)
            for (std::size_t block_column = 0; block_column != block_columns; ++block_column) {
                std::size_t const first_row = block_row ? block_row * span - 1 : 0;
                std::size_t const first_column = block_column ? block_column * span - 1 : 0;
                std::size_t const last_row = min_of(block_row * span + span + 1, rows - 1);
                std::size_t const last_column = min_of(block_column * span + span + 1, columns - 1);
                *maxima++ = terrain_window_maximum_(heights, heights_stride, first_row, last_row, first_column,
                                                    last_column);
            }
    }
    return status_t::success_k;
}

/**
 *  @brief Returns the bytes a Haversine pack of @p points takes.
 *  @param[in] points Number of points in the pack.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial layout.
 */
inline expected<std::size_t> haversines_pack_size(std::size_t points,
                                                  nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = geodesics_pack_bytes_(points);
    nk_status_t const status = capabilities ? nk_haversines_pack_size_f64_best(points, capabilities, &bytes)
                                            : nk_success_k;
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs points as unit vectors on the mean sphere, for @c haversines_packed.
 *  @param[in] latitudes The @p count latitudes, in radians.
 *  @param[in] longitudes The @p count longitudes, in radians.
 *  @param[in] count Number of points.
 *  @param[out] packed At least @c haversines_pack_size bytes, aligned to @c nk_default_alignment_k.
 *  @param[in] capabilities Capabilities to pick from, or zero for the @p precision_type_ template.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *  @tparam precision_type_ The template's arithmetic; the kernels compute in @c f64_t.
 */
template <numeric_dtype precision_type_ = f64_t>
status_t haversines_pack(f64_t const *latitudes, f64_t const *longitudes, std::size_t count, void *packed,
                         nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<precision_type_, f64_t>)
        if (capabilities)
            return static_cast<status_t>(
                nk_haversines_pack_f64_best(&latitudes->raw_, &longitudes->raw_, count, packed, capabilities, stream));
    haversines_pack_reference_<precision_type_>(latitudes, longitudes, count, packed);
    return status_t::success_k;
}

/**
 *  @brief Great-circle metres between every row and every column point of two Haversine packs.
 *  @param[in] a_packed The row points' Haversine pack.
 *  @param[in] b_packed The column points' Haversine pack.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 *  @param[in] results_stride Bytes between the rows of @p results.
 *  @param[in] capabilities Capabilities to pick from, or zero for the @p precision_type_ template.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *  @tparam precision_type_ The template's arithmetic; the kernels compute in @c f64_t.
 */
template <numeric_dtype precision_type_ = f64_t>
status_t haversines_packed(void const *a_packed, void const *b_packed, f64_t *results, std::size_t results_stride,
                           nk_capability_t capabilities = default_capabilities(),
                           nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<precision_type_, f64_t>)
        if (capabilities)
            return static_cast<status_t>(nk_haversines_packed_f64_best(a_packed, b_packed, &results->raw_,
                                                                       results_stride, capabilities, stream));
    haversines_packed_reference_<precision_type_>(a_packed, b_packed, results, results_stride);
    return status_t::success_k;
}

/** Returns the bytes a Vincenty pack of @p points takes, as @c haversines_pack_size does. */
inline expected<std::size_t> vincentys_pack_size(std::size_t points,
                                                 nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = geodesics_pack_bytes_(points);
    nk_status_t const status = capabilities ? nk_vincentys_pack_size_f64_best(points, capabilities, &bytes)
                                            : nk_success_k;
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** Packs points as their longitudes and the sines and cosines of their reduced latitudes, for
 *  @c vincentys_packed, taking what @c haversines_pack takes. */
template <numeric_dtype precision_type_ = f64_t>
status_t vincentys_pack(f64_t const *latitudes, f64_t const *longitudes, std::size_t count, void *packed,
                        nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<precision_type_, f64_t>)
        if (capabilities)
            return static_cast<status_t>(
                nk_vincentys_pack_f64_best(&latitudes->raw_, &longitudes->raw_, count, packed, capabilities, stream));
    vincentys_pack_reference_<precision_type_>(latitudes, longitudes, count, packed);
    return status_t::success_k;
}

/** Geodesic metres on the WGS-84 ellipsoid between every row and every column point of two Vincenty
 *  packs, taking what @c haversines_packed takes. */
template <numeric_dtype precision_type_ = f64_t>
status_t vincentys_packed(void const *a_packed, void const *b_packed, f64_t *results, std::size_t results_stride,
                          nk_capability_t capabilities = default_capabilities(),
                          nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<precision_type_, f64_t>)
        if (capabilities)
            return static_cast<status_t>(
                nk_vincentys_packed_f64_best(a_packed, b_packed, &results->raw_, results_stride, capabilities, stream));
    vincentys_packed_reference_<precision_type_>(a_packed, b_packed, results, results_stride);
    return status_t::success_k;
}

/**
 *  @brief Returns the bytes the terrain pack of a @p rows × @p columns grid takes.
 *  @param[in] rows Number of grid rows.
 *  @param[in] columns Number of grid columns.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial layout.
 */
inline expected<std::size_t> terrain_pack_size(std::size_t rows, std::size_t columns,
                                               nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = terrain_pack_bytes_(rows, columns);
    nk_status_t const status = capabilities ? nk_terrain_pack_size_f32_best(rows, columns, capabilities, &bytes)
                                            : nk_success_k;
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
 *  @param[in] capabilities Capabilities to pick from, or zero for the template.
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
    return terrain_pack_reference_(heights, rows, columns, heights_stride, row_metres, column_metres, packed);
}

/** Bilinear heights under @p count interleaved [column, row] points, NaN off the grid, rounding
 *  once into @p result_type_ from at least @c f64_t, as the kernels do. */
template <typename result_type_>
void terrain_heights_reference_(f32_t const *heights, void const *packed, f32_t const *points, std::size_t count,
                                result_type_ *results) noexcept {
    using math_t = std::conditional_t<(sizeof(result_type_) < sizeof(f64_t)), f64_t, result_type_>;
    nk_terrain_packed_header_t const *header = static_cast<nk_terrain_packed_header_t const *>(packed);
    std::size_t const rows = static_cast<std::size_t>(header->rows);
    std::size_t const columns = static_cast<std::size_t>(header->columns);
    std::size_t const stride = static_cast<std::size_t>(header->heights_stride);
    math_t const zero(0.0), last_row(static_cast<double>(rows - 1)), last_column(static_cast<double>(columns - 1));
    auto const height = [&](std::size_t row, std::size_t column) noexcept {
        f32_t const *line = reinterpret_cast<f32_t const *>(reinterpret_cast<char const *>(heights) + row * stride);
        return math_t(static_cast<double>(line[column].raw_));
    };
    // The index of the cell a coordinate on the grid falls in, its fraction left to interpolate
    auto const cell = [](math_t coordinate) noexcept {
        std::size_t index = static_cast<std::size_t>(static_cast<double>(coordinate));
        if (coordinate < math_t(static_cast<double>(index))) --index;
        return index;
    };
    for (std::size_t point = 0; point != count; ++point) {
        math_t const column(static_cast<double>(points[point * 2].raw_));
        math_t const row(static_cast<double>(points[point * 2 + 1].raw_));
        if (!(zero <= column && column <= last_column && zero <= row && row <= last_row)) {
            results[point] = result_type_(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        std::size_t column_first = cell(column), row_first = cell(row);
        if (column_first + 1 == columns && columns > 1) --column_first;
        if (row_first + 1 == rows && rows > 1) --row_first;
        std::size_t const column_second = columns > 1 ? column_first + 1 : column_first;
        std::size_t const row_second = rows > 1 ? row_first + 1 : row_first;
        math_t const column_fraction = column - math_t(static_cast<double>(column_first));
        math_t const row_fraction = row - math_t(static_cast<double>(row_first));
        math_t const north_west = height(row_first, column_first), north_east = height(row_first, column_second);
        math_t const south_west = height(row_second, column_first), south_east = height(row_second, column_second);
        math_t const north = north_west + column_fraction * (north_east - north_west);
        math_t const south = south_west + column_fraction * (south_east - south_west);
        results[point] = result_type_(static_cast<double>(north + row_fraction * (south - north)));
    }
}

/** Clearances from every row viewpoint to every column target, rounding once into @p result_type_
 *  from at least @c f64_t: the lowest gap over every crossing of the sight line with the grid
 *  lines, without the pyramid, NaN when either end leaves the grid, and minus infinity when the
 *  target falls outside the viewpoint's view. */
template <typename result_type_>
void clearances_reference_(f32_t const *heights, void const *packed, f32_t const *viewpoints, f32_t const *views,
                           std::size_t rows, f32_t const *targets, std::size_t columns, f64_t refraction,
                           result_type_ *results, std::size_t results_stride) noexcept {
    using math_t = std::conditional_t<(sizeof(result_type_) < sizeof(f64_t)), f64_t, result_type_>;
    nk_terrain_packed_header_t const *header = static_cast<nk_terrain_packed_header_t const *>(packed);
    std::size_t const grid_rows = static_cast<std::size_t>(header->rows);
    std::size_t const grid_columns = static_cast<std::size_t>(header->columns);
    std::size_t const stride = static_cast<std::size_t>(header->heights_stride);
    math_t const zero(0.0), one(1.0), two(2.0);
    math_t const last_row(static_cast<double>(grid_rows - 1)), last_column(static_cast<double>(grid_columns - 1));
    math_t const row_metres(header->row_metres), column_metres(header->column_metres);
    math_t const drop_scale = (one - math_t(refraction.raw_)) / (two * math_t(NUMKONG_EARTH_MEAN_RADIUS));
    math_t const full_turn(2 * 3.14159265358979323846);
    auto const value = [](f32_t const *values, std::size_t index) noexcept {
        return math_t(static_cast<double>(values[index].raw_));
    };
    auto const height = [&](std::size_t row, std::size_t column) noexcept {
        f32_t const *line = reinterpret_cast<f32_t const *>(reinterpret_cast<char const *>(heights) + row * stride);
        return math_t(static_cast<double>(line[column].raw_));
    };
    auto const cell = [](math_t coordinate) noexcept {
        std::size_t index = static_cast<std::size_t>(static_cast<double>(coordinate));
        if (coordinate < math_t(static_cast<double>(index))) --index;
        return index;
    };
    // Both ends lie on the grid, so only a rounding near an edge can step a crossing outside it
    auto const clamp = [&](math_t coordinate, math_t last) noexcept {
        return coordinate < zero ? zero : (last < coordinate ? last : coordinate);
    };
    // A view of azimuth, elevation, field and range, all optional, hides what falls outside it
    auto const outside = [&](math_t east, math_t north, math_t up, f32_t const *view) noexcept {
        math_t const azimuth = value(view, 0), elevation = value(view, 1), field = value(view, 2);
        math_t const range = value(view, 3);
        math_t const slant = (east * east + north * north + up * up).sqrt();
        if (range == range && range < slant) return true;
        if (azimuth != azimuth || !(field < full_turn)) return false;
        math_t const half_field = field / two;
        if (elevation != elevation) {
            math_t const ground = (east * east + north * north).sqrt();
            if (ground == zero) return false;
            return (east * azimuth.sin() + north * azimuth.cos()) / ground < half_field.cos();
        }
        if (slant == zero) return false;
        math_t const cos_elevation = elevation.cos();
        math_t const along = east * azimuth.sin() * cos_elevation + north * azimuth.cos() * cos_elevation +
                             up * elevation.sin();
        return along / slant < half_field.cos();
    };
    // One pass per sight line, its viewpoint and target read off the line's index
    for (std::size_t sight = 0; sight != rows * columns; ++sight) {
        std::size_t const row = sight / columns, column = sight % columns;
        result_type_ *output = reinterpret_cast<result_type_ *>(reinterpret_cast<char *>(results) +
                                                                row * results_stride);
        math_t const column_start = value(viewpoints, row * 3), row_start = value(viewpoints, row * 3 + 1);
        math_t const height_start = value(viewpoints, row * 3 + 2);
        bool const viewpoint_on_grid = zero <= column_start && column_start <= last_column && zero <= row_start &&
                                       row_start <= last_row;
        math_t const column_end = value(targets, column * 3), row_end = value(targets, column * 3 + 1);
        if (!viewpoint_on_grid ||
            !(zero <= column_end && column_end <= last_column && zero <= row_end && row_end <= last_row)) {
            output[column] = result_type_(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        math_t const column_delta = column_end - column_start, row_delta = row_end - row_start;
        math_t const east = column_delta * column_metres, south = row_delta * row_metres;
        math_t const distance = (east * east + south * south).sqrt();
        math_t const height_delta = value(targets, column * 3 + 2) - drop_scale * distance * distance - height_start;
        if (views && outside(east, -south, height_delta, views + row * 4)) {
            output[column] = result_type_(-std::numeric_limits<double>::infinity());
            continue;
        }
        math_t clearance(std::numeric_limits<double>::infinity());
        auto const fold = [&](math_t u, math_t first, math_t second, math_t fraction) noexcept {
            math_t const ground = fraction == zero ? first : first + fraction * (second - first);
            if (ground != ground) return;
            math_t const along = u * distance;
            math_t const gap = height_start + u * height_delta - (ground - drop_scale * along * along);
            if (gap < clearance) clearance = gap;
        };
        for (std::size_t line = 0; line != grid_columns && column_delta != zero; ++line) {
            math_t const u = (math_t(static_cast<double>(line)) - column_start) / column_delta;
            if (!(zero < u && u < one)) continue;
            math_t const crossing = clamp(row_start + u * row_delta, last_row);
            std::size_t const first = cell(crossing), second = first + 1 < grid_rows ? first + 1 : first;
            fold(u, height(first, line), height(second, line), crossing - math_t(static_cast<double>(first)));
        }
        for (std::size_t line = 0; line != grid_rows && row_delta != zero; ++line) {
            math_t const u = (math_t(static_cast<double>(line)) - row_start) / row_delta;
            if (!(zero < u && u < one)) continue;
            math_t const crossing = clamp(column_start + u * column_delta, last_column);
            std::size_t const first = cell(crossing), second = first + 1 < grid_columns ? first + 1 : first;
            fold(u, height(line, first), height(line, second), crossing - math_t(static_cast<double>(first)));
        }
        output[column] = result_type_(static_cast<double>(clearance));
    }
}

/** Bilinear terrain heights under @p count interleaved [column, row] points, NaN off the grid,
 *  dispatched as @c terrain_pack is, or with zero capabilities from the template rounding once
 *  into @p result_type_. */
template <typename in_type_ = f32_t, typename result_type_ = in_type_>
status_t terrain_heights_packed(std::type_identity_t<in_type_> const *heights, void const *packed,
                                std::type_identity_t<in_type_> const *points, std::size_t count, result_type_ *results,
                                nk_capability_t capabilities = default_capabilities(),
                                nk_stream_t stream = nullptr) noexcept {
    static_assert(std::is_same_v<in_type_, f32_t>, "terrain packs hold f32 heights");
    if constexpr (std::is_same_v<result_type_, f32_t>)
        if (capabilities)
            return static_cast<status_t>(nk_terrain_heights_packed_f32_best(
                &heights->raw_, packed, &points->raw_, count, &results->raw_, capabilities, stream));
    terrain_heights_reference_(heights, packed, points, count, results);
    return status_t::success_k;
}

/** Clearances in metres from every row viewpoint to every column target over one terrain, as
 *  @c nk_clearances_packed_f32_best defines them, dispatched as @c terrain_pack is, or with zero
 *  capabilities from the template in @p result_type_. */
template <typename in_type_ = f32_t, typename result_type_ = in_type_>
status_t clearances_packed(std::type_identity_t<in_type_> const *heights, void const *packed,
                           std::type_identity_t<in_type_> const *viewpoints,
                           std::type_identity_t<in_type_> const *views, std::size_t rows,
                           std::type_identity_t<in_type_> const *targets, std::size_t columns, f64_t refraction,
                           result_type_ *results, std::size_t results_stride,
                           nk_capability_t capabilities = default_capabilities(),
                           nk_stream_t stream = nullptr) noexcept {
    static_assert(std::is_same_v<in_type_, f32_t>, "terrain packs hold f32 heights");
    if constexpr (std::is_same_v<result_type_, f32_t>)
        if (capabilities)
            return static_cast<status_t>(nk_clearances_packed_f32_best(
                &heights->raw_, packed, &viewpoints->raw_, views ? &views->raw_ : nullptr, rows, &targets->raw_,
                columns, refraction.raw_, &results->raw_, results_stride, capabilities, stream));
    clearances_reference_(heights, packed, viewpoints, views, rows, targets, columns, refraction, results,
                          results_stride);
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_GEOSPATIAL_HPP
