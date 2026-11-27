/**
 *  @file test/geospatial.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Backend-neutral geodesic and terrain scenarios, run on the host and on devices alike.
 *
 *  Every scenario is a template over its kernels and a backend owning where the operands live and
 *  when results become readable, @c host_backend_t by default. Each judges a backend against the
 *  same references: geodesics against their formulas in double-double, and heights and clearances
 *  against their templates, visiting every crossing without the pyramid in the kernels' arithmetic.
 */
#pragma once
#ifndef NUMKONG_TEST_GEOSPATIAL_HPP
#define NUMKONG_TEST_GEOSPATIAL_HPP

#include <cmath>   // `std::sqrt`, `std::floor`
#include <cstring> // `std::memcmp`, `std::memcpy`

#include <limits>      // `std::numeric_limits`
#include <random>      // `std::normal_distribution`, `std::uniform_real_distribution`
#include <type_traits> // `std::conditional_t`

#include "numkong/geospatial.h" // `nk_haversines_error_bound`, `nk_vincentys_error_bound`
#include "numkong/geospatial.hpp"
#include "numkong/random.hpp" // `nk::fill_coordinates`

#include "harness.hpp"

namespace ashvardanian::numkong::test {

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
template <a_numeric_dtype precision_type_>
void haversines_reference(f64_t const *a_latitudes, f64_t const *a_longitudes, std::size_t rows,
                          f64_t const *b_latitudes, f64_t const *b_longitudes, std::size_t columns, f64_t *results,
                          std::size_t results_stride) noexcept {
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
            precision_type_ const angle = two * square_root_reference(haversine).atan2(
                                                    square_root_reference(one - haversine));
            output[column] = f64_t(to_double(earth_radius * angle));
        }
    }
}

/** Geodesic metres on the WGS-84 ellipsoid between two points, given the sines and cosines of
 *  their reduced latitudes and the @p longitude_difference from the first to the second, through
 *  Vincenty's inverse formula in @p precision_type_. */
template <a_numeric_dtype precision_type_>
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
        sin_angular_distance = square_root_reference(cross_term * cross_term + mixed_term * mixed_term);
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
 *  atan2 to stay finite at the poles; parameters as @c haversines_reference. */
template <a_numeric_dtype precision_type_>
void vincentys_reference(f64_t const *a_latitudes, f64_t const *a_longitudes, std::size_t rows,
                         f64_t const *b_latitudes, f64_t const *b_longitudes, std::size_t columns, f64_t *results,
                         std::size_t results_stride) noexcept {
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
            output[column] = f64_t(to_double(
                vincenty_reference_<precision_t>(reduced_first.sin(), reduced_first.cos(), reduced_second.sin(),
                                                 reduced_second.cos(), longitude_difference)));
        }
    }
}

/** Bilinear heights under @p count interleaved [column, row] points of a @p rows × @p columns grid,
 *  its rows @p stride bytes apart, NaN off the grid, rounding once into @p result_type_ from at
 *  least @c f64_t, as the kernels do. */
template <typename result_type_>
void terrain_heights_reference(f32_t const *heights, std::size_t rows, std::size_t columns, std::size_t stride,
                               f32_t const *points, std::size_t count, result_type_ *results) noexcept {
    using math_t = std::conditional_t<(sizeof(result_type_) < sizeof(f64_t)), f64_t, result_type_>;
    math_t const zero(0.0), last_row(to_double(rows - 1)), last_column(to_double(columns - 1));
    auto const height = [&](std::size_t row, std::size_t column) noexcept {
        f32_t const *line = reinterpret_cast<f32_t const *>(reinterpret_cast<char const *>(heights) + row * stride);
        return math_t(to_double(line[column].raw_));
    };
    // The index of the cell a coordinate on the grid falls in, its fraction left to interpolate
    auto const cell = [](math_t coordinate) noexcept {
        std::size_t index = static_cast<std::size_t>(to_double(coordinate));
        if (coordinate < math_t(to_double(index))) --index;
        return index;
    };
    for (std::size_t point = 0; point != count; ++point) {
        math_t const column(to_double(points[point * 2].raw_));
        math_t const row(to_double(points[point * 2 + 1].raw_));
        if (!(zero <= column && column <= last_column && zero <= row && row <= last_row)) {
            results[point] = result_type_(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        std::size_t column_first = cell(column), row_first = cell(row);
        if (column_first + 1 == columns && columns > 1) --column_first;
        if (row_first + 1 == rows && rows > 1) --row_first;
        std::size_t const column_second = columns > 1 ? column_first + 1 : column_first;
        std::size_t const row_second = rows > 1 ? row_first + 1 : row_first;
        math_t const column_fraction = column - math_t(to_double(column_first));
        math_t const row_fraction = row - math_t(to_double(row_first));
        math_t const north_west = height(row_first, column_first), north_east = height(row_first, column_second);
        math_t const south_west = height(row_second, column_first), south_east = height(row_second, column_second);
        math_t const north = north_west + column_fraction * (north_east - north_west);
        math_t const south = south_west + column_fraction * (south_east - south_west);
        results[point] = result_type_(to_double(north + row_fraction * (south - north)));
    }
}

/** Clearances from every row viewpoint to every column target, rounding once into @p result_type_
 *  from at least @c f64_t: the lowest gap over every crossing of the sight line with the grid
 *  lines, without the pyramid, NaN when either end leaves the grid, and minus infinity when the
 *  target falls outside the viewpoint's view. The grid takes the parameters of @c terrain_pack. */
template <typename result_type_>
void clearances_reference(f32_t const *heights, std::size_t grid_rows, std::size_t grid_columns, std::size_t stride,
                          f64_t grid_row_metres, f64_t grid_column_metres, f32_t const *viewpoints, f32_t const *views,
                          std::size_t rows, f32_t const *targets, std::size_t columns, f64_t refraction,
                          result_type_ *results, std::size_t results_stride) noexcept {
    using math_t = std::conditional_t<(sizeof(result_type_) < sizeof(f64_t)), f64_t, result_type_>;
    math_t const zero(0.0), one(1.0), two(2.0);
    math_t const last_row(to_double(grid_rows - 1)), last_column(to_double(grid_columns - 1));
    math_t const row_metres(grid_row_metres.raw_), column_metres(grid_column_metres.raw_);
    math_t const drop_scale = (one - math_t(refraction.raw_)) / (two * math_t(NUMKONG_EARTH_MEAN_RADIUS));
    math_t const full_turn(2 * 3.14159265358979323846);
    auto const value = [](f32_t const *values, std::size_t index) noexcept {
        return math_t(to_double(values[index].raw_));
    };
    auto const height = [&](std::size_t row, std::size_t column) noexcept {
        f32_t const *line = reinterpret_cast<f32_t const *>(reinterpret_cast<char const *>(heights) + row * stride);
        return math_t(to_double(line[column].raw_));
    };
    auto const cell = [](math_t coordinate) noexcept {
        std::size_t index = static_cast<std::size_t>(to_double(coordinate));
        if (coordinate < math_t(to_double(index))) --index;
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
        math_t const slant = square_root_reference(east * east + north * north + up * up);
        if (range == range && range < slant) return true;
        if (azimuth != azimuth || !(field < full_turn)) return false;
        math_t const half_field = field / two;
        if (elevation != elevation) {
            math_t const ground = square_root_reference(east * east + north * north);
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
        math_t const distance = square_root_reference(east * east + south * south);
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
            math_t const u = (math_t(to_double(line)) - column_start) / column_delta;
            if (!(zero < u && u < one)) continue;
            math_t const crossing = clamp(row_start + u * row_delta, last_row);
            std::size_t const first = cell(crossing), second = first + 1 < grid_rows ? first + 1 : first;
            fold(u, height(first, line), height(second, line), crossing - math_t(to_double(first)));
        }
        for (std::size_t line = 0; line != grid_rows && row_delta != zero; ++line) {
            math_t const u = (math_t(to_double(line)) - row_start) / row_delta;
            if (!(zero < u && u < one)) continue;
            math_t const crossing = clamp(column_start + u * column_delta, last_column);
            std::size_t const first = cell(crossing), second = first + 1 < grid_columns ? first + 1 : first;
            fold(u, height(line, first), height(line, second), crossing - math_t(to_double(first)));
        }
        output[column] = result_type_(to_double(clearance));
    }
}

/** Geodesic inputs: columns anywhere within a cap, near their rows, on them, opposite them, or on
 *  the poles, the equator and the antimeridian. */
enum class geodesics_case_t { anywhere_k, nearby_k, coincident_k, antipodal_k, landmarks_k };

/**
 *  @brief A geodesic family's @p pack and @p kernel against its formula in double-double.
 *
 *  Shapes are ragged, and columns lie anywhere within a cap of a common centre, near their rows,
 *  on them, opposite them, or on landmarks. Vincenty's points stay within a hemisphere, as its
 *  iteration need not converge near the antipode, but its poles, half a meridian apart, must not
 *  come out as coincident.
 */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename backend_type_ = host_backend_t,
          typename pack_size_type_, typename pack_type_, typename kernel_type_>
error_stats_t test_geodesics_packed(pack_size_type_ pack_size, pack_type_ pack, kernel_type_ kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    constexpr bool spherical = kind_ == nk_kernel_haversines_packed_k;
    error_stats_t stats(spherical ? nk_haversines_error_bound(scalar_t::dtype())
                                  : nk_vincentys_error_bound(scalar_t::dtype()));
    auto backend = make_backend<backend_type_>();
    split_mix_generator_t generator(case_stream_key());
    double const pi = 3.14159265358979323846, angle = double(test_run.settings.max_coord_angle_degrees) * pi / 180;
    double const cap = spherical ? pi : nk::min_of(angle, pi / 2) / 2;
    std::pair<geodesics_case_t, double> const cases[] = {{geodesics_case_t::anywhere_k, spherical ? angle : cap},
                                                         {geodesics_case_t::nearby_k, 1e-9},
                                                         {geodesics_case_t::coincident_k, 0},
                                                         {geodesics_case_t::antipodal_k, 0},
                                                         {geodesics_case_t::landmarks_k, 0}};
    std::pair<std::size_t, std::size_t> const shapes[] =
        {{test_run.settings.matrix_height, test_run.settings.matrix_width}, {1, 1}, {2, 3}, {3, 2}, {17, 33}};
    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();)
        for (auto const [kind, separation] : cases) {
            // Opposite points and landmarks reach the antipode, where Vincenty may not settle.
            if (!spherical && (kind == geodesics_case_t::antipodal_k || kind == geodesics_case_t::landmarks_k))
                continue;
            for (auto const [rows, columns] : shapes) {
                auto a_latitudes = make_vector<scalar_t>(backend, rows),
                     a_longitudes = make_vector<scalar_t>(backend, rows);
                auto b_latitudes = make_vector<scalar_t>(backend, columns);
                auto b_longitudes = make_vector<scalar_t>(backend, columns);
                scalar_t *a_lat = a_latitudes.values_data(), *a_lon = a_longitudes.values_data();
                scalar_t *b_lat = b_latitudes.values_data(), *b_lon = b_longitudes.values_data();
                std::size_t const paired = nk::min_of(rows, columns);
                if (cap < pi) {
                    scalar_t centre_latitude, centre_longitude;
                    nk::fill_coordinates(generator, &centre_latitude, &centre_longitude, 1);
                    std::vector<scalar_t> centre_latitudes(nk::max_of(rows, columns), centre_latitude),
                        centre_longitudes(centre_latitudes.size(), centre_longitude);
                    nk::fill_nearby_coordinates(generator, centre_latitudes.data(), centre_longitudes.data(), a_lat,
                                                a_lon, rows, cap);
                    nk::fill_nearby_coordinates(generator, centre_latitudes.data(), centre_longitudes.data(), b_lat,
                                                b_lon, columns, cap);
                }
                else {
                    nk::fill_coordinates(generator, a_lat, a_lon, rows);
                    nk::fill_coordinates(generator, b_lat, b_lon, columns);
                }
                if (kind == geodesics_case_t::anywhere_k || kind == geodesics_case_t::nearby_k)
                    nk::fill_nearby_coordinates(generator, a_lat, a_lon, b_lat, b_lon, paired, separation);
                for (std::size_t index = 0; index != paired && kind == geodesics_case_t::coincident_k; ++index)
                    b_lat[index] = a_lat[index], b_lon[index] = a_lon[index];
                for (std::size_t index = 0; index != paired && kind == geodesics_case_t::antipodal_k; ++index) {
                    double const longitude = double(a_lon[index]) + pi;
                    b_lat[index] = -double(a_lat[index]);
                    b_lon[index] = longitude > pi ? longitude - 2 * pi : longitude;
                }
                double const latitudes[] = {pi / 2, -pi / 2, 0.0, -0.0, pi / 4, -pi / 3};
                double const longitudes[] = {0.0, pi, -pi, pi / 2, -0.0, 3.0};
                for (std::size_t index = 0; index != rows && kind == geodesics_case_t::landmarks_k; ++index)
                    a_lat[index] = latitudes[index % 6], a_lon[index] = longitudes[index / 6 % 6];
                for (std::size_t index = 0; index != columns && kind == geodesics_case_t::landmarks_k; ++index)
                    b_lat[index] = latitudes[index * 5 % 6], b_lon[index] = longitudes[(index + index / 6) % 6];

                nk_size_t a_bytes = 0, b_bytes = 0;
                stats.expect_status(pack_size(rows, &a_bytes)), stats.expect_status(pack_size(columns, &b_bytes));
                auto a_packed = make_vector<std::byte>(backend, a_bytes);
                auto b_packed = make_vector<std::byte>(backend, b_bytes);
                auto results = make_vector<scalar_t>(backend, rows * columns);
                backend.call(pack, a_latitudes.raw_values_data(), a_longitudes.raw_values_data(), rows,
                             static_cast<void *>(a_packed.raw_values_data()));
                backend.call(pack, b_latitudes.raw_values_data(), b_longitudes.raw_values_data(), columns,
                             static_cast<void *>(b_packed.raw_values_data()));
                backend.call(kernel, static_cast<void const *>(a_packed.raw_values_data()),
                             static_cast<void const *>(b_packed.raw_values_data()), results.raw_values_data(),
                             columns * sizeof(raw_t));
                if (!stats.expect_status(backend.synchronize())) return stats;
                std::vector<scalar_t> reference(rows * columns);
                if constexpr (spherical)
                    haversines_reference<f118_t>(a_lat, a_lon, rows, b_lat, b_lon, columns, reference.data(),
                                                 columns * sizeof(scalar_t));
                else
                    vincentys_reference<f118_t>(a_lat, a_lon, rows, b_lat, b_lon, columns, reference.data(),
                                                columns * sizeof(scalar_t));
                for (std::size_t cell = 0; cell != reference.size(); ++cell) {
                    double const expected = to_double(reference[cell]);
                    double const bound = spherical ? stats.term_error_bound * NUMKONG_EARTH_MEAN_RADIUS *
                                                         (1 + expected / NUMKONG_EARTH_MEAN_RADIUS)
                                                   : stats.term_error_bound * NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
                    stats.accumulate_bounded(scalar_t::from_raw(results.raw_values_data()[cell]), expected, bound);
                }
                // Poles are antipodes, half a meridian apart, which no kernel may zero.
                if (spherical || rows != 1 || columns != 1) continue;
                a_lat[0] = pi / 2, b_lat[0] = -pi / 2, a_lon[0] = b_lon[0] = 0.0;
                backend.call(pack, a_latitudes.raw_values_data(), a_longitudes.raw_values_data(), nk_size_t(1),
                             static_cast<void *>(a_packed.raw_values_data()));
                backend.call(pack, b_latitudes.raw_values_data(), b_longitudes.raw_values_data(), nk_size_t(1),
                             static_cast<void *>(b_packed.raw_values_data()));
                backend.call(kernel, static_cast<void const *>(a_packed.raw_values_data()),
                             static_cast<void const *>(b_packed.raw_values_data()), results.raw_values_data(),
                             sizeof(raw_t));
                if (!stats.expect_status(backend.synchronize())) return stats;
                vincentys_reference<f118_t>(a_lat, a_lon, 1, b_lat, b_lon, 1, reference.data(), sizeof(scalar_t));
                stats.accumulate_bounded(scalar_t(results[0]), to_double(reference[0]),
                                         stats.term_error_bound * NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
            }
        }
    return stats;
}

/**
 *  @brief A terrain @p pack_kernel, @p heights_kernel and @p clearances_kernel over random walks
 *      with a missing cell, a flat sea of signed zeros, and single cells, rows and columns.
 *
 *  The heights must match their template byte for byte, and every clearance without a view must
 *  equal a visit of every crossing without the pyramid, in the arithmetic the kernels use, as
 *  skipping blocks may change nothing. A view only ever hides a target as minus infinity, matching
 *  the template. Points fall off the grid, onto its edges, onto their viewpoints and into their
 *  cells, under refractions of light, none and full. A flat sea 50 km wide closes it: two 10 m
 *  masts see each other across twice the distance to the horizon from 10 m, 22.58 km without
 *  refraction and 24.38 km with the 1/7 of light, a view facing east sees east and not west and
 *  reaches no farther than its range, and heights interpolate bilinearly.
 */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename pack_size_type_, typename pack_type_,
          typename heights_type_, typename clearances_type_>
error_stats_t test_terrain_packed(pack_size_type_ pack_size, pack_type_ pack_kernel, heights_type_ heights_kernel,
                                  clearances_type_ clearances_kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    error_stats_t stats(0.0);
    auto backend = make_backend<backend_type_>();
    split_mix_generator_t generator(case_stream_key());
    std::normal_distribution<raw_t> step(0.0f, 4.0f);
    std::uniform_real_distribution<raw_t> unit(0.0f, 1.0f);
    raw_t const not_a_number = std::numeric_limits<raw_t>::quiet_NaN();
    raw_t const infinity = std::numeric_limits<raw_t>::infinity();
    auto const same = [](raw_t actual, raw_t expected) {
        return !std::memcmp(&actual, &expected, sizeof(raw_t)) || (actual != actual && expected != expected);
    };
    struct grid_t {
        std::size_t rows, columns;
        bool flat;
    };
    grid_t const grids[] = {{97, 83, false}, {257, 129, false}, {5, 300, false}, {3, 3, false},
                            {2, 1, false},   {1, 2, false},     {1, 1, false},   {1, 501, true}};
    std::size_t const viewpoints_count = 13, targets_count = 29, points_count = 257;
    nk_f64_t const refractions[] = {1.0 / 7, 1.0 / 7, 0.0, 1.0};
    std::size_t const passes = sizeof(refractions) / sizeof(refractions[0]), cells = viewpoints_count * targets_count;
    nk_f64_t const row_metres = 30.0, column_metres = 21.0;
    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();)
        for (grid_t const grid : grids) {
            std::size_t const rows = grid.rows, columns = grid.columns, padded = columns + 3;
            auto heights = make_vector<scalar_t>(backend, rows * padded);
            raw_t *terrain = heights.raw_values_data();
            for (std::size_t row = 0; row != rows; ++row)
                for (std::size_t column = 0; column != columns; ++column) {
                    raw_t const north = row ? terrain[(row - 1) * padded + column] : 100.0f;
                    raw_t const west = column ? terrain[row * padded + column - 1] : north;
                    terrain[row * padded + column] = grid.flat ? ((row + column) % 2 ? -0.0f : 0.0f)
                                                               : (north + west) / 2 + step(generator);
                }
            if (!grid.flat) terrain[rows / 2 * padded + columns / 3] = not_a_number;
            auto place = [&](raw_t *point, std::size_t index) {
                // Some points fall off the grid, some onto its edges, the rest 2-60 m above it.
                point[0] = unit(generator) * raw_t(columns + 1) - 1.0f;
                point[1] = unit(generator) * raw_t(rows + 1) - 1.0f;
                if (index % 5 == 1) point[0] = raw_t(columns - 1);
                if (index % 7 == 2) point[1] = 0.0f;
                std::size_t const row = nk::min_of<std::size_t>(rows - 1, std::size_t(nk::max_of(point[1], raw_t(0))));
                std::size_t const column = nk::min_of<std::size_t>(columns - 1,
                                                                   std::size_t(nk::max_of(point[0], raw_t(0))));
                raw_t const ground = terrain[row * padded + column];
                point[2] = (ground == ground ? ground : 100.0f) + 2.0f + unit(generator) * 58.0f;
            };
            auto points = make_vector<scalar_t>(backend, points_count * 2);
            auto viewpoints = make_vector<scalar_t>(backend, viewpoints_count * 3);
            auto views = make_vector<scalar_t>(backend, viewpoints_count * 4);
            auto targets = make_vector<scalar_t>(backend, targets_count * 3);
            raw_t *point_values = points.raw_values_data(), *viewpoint_values = viewpoints.raw_values_data();
            raw_t *view_values = views.raw_values_data(), *target_values = targets.raw_values_data();
            for (std::size_t point = 0; point != points_count; ++point) {
                raw_t *coordinates = point_values + point * 2;
                coordinates[0] = unit(generator) * raw_t(columns + 1) - 1.0f;
                coordinates[1] = unit(generator) * raw_t(rows + 1) - 1.0f;
                if (point % 5 == 1) coordinates[0] = raw_t(columns - 1);
                if (point % 7 == 2) coordinates[1] = raw_t(rows - 1);
                if (point % 11 == 3) coordinates[0] = std::floor(nk::max_of(coordinates[0], raw_t(0)));
                if (point % 13 == 4) coordinates[1] = not_a_number;
            }
            for (std::size_t viewpoint = 0; viewpoint != viewpoints_count; ++viewpoint) {
                place(viewpoint_values + viewpoint * 3, viewpoint);
                view_values[viewpoint * 4 + 0] = viewpoint % 6 == 5 ? not_a_number : unit(generator) * 6.2831853f;
                view_values[viewpoint * 4 + 1] = viewpoint % 3 ? unit(generator) - 0.5f : not_a_number;
                view_values[viewpoint * 4 + 2] = viewpoint % 5 == 4 ? 7.0f : 0.3f + unit(generator) * 3.0f;
                view_values[viewpoint * 4 + 3] = viewpoint % 4 ? 3000.0f * unit(generator)
                                                               : (viewpoint % 8 ? infinity : not_a_number);
            }
            for (std::size_t target = 0; target != targets_count; ++target) place(target_values + target * 3, target);
            // Some targets coincide with their viewpoint, and some share its cell.
            for (std::size_t target = 0; target < viewpoints_count; target += 4)
                std::memcpy(target_values + target * 3, viewpoint_values + target * 3, 3 * sizeof(raw_t));
            for (std::size_t target = 1; target < viewpoints_count; target += 4) {
                target_values[target * 3] = viewpoint_values[target * 3] + 0.25f;
                target_values[target * 3 + 1] = viewpoint_values[target * 3 + 1];
            }

            nk_size_t bytes = 0;
            stats.expect_status(pack_size(rows, columns, &bytes));
            auto packed = make_vector<std::byte>(backend, bytes);
            auto point_heights = make_vector<scalar_t>(backend, points_count);
            auto clearances = make_vector<scalar_t>(backend, passes * cells);
            void const *pack = packed.raw_values_data();
            backend.call(pack_kernel, static_cast<raw_t const *>(terrain), rows, columns, padded * sizeof(raw_t),
                         row_metres, column_metres, static_cast<void *>(packed.raw_values_data()));
            backend.call(heights_kernel, static_cast<raw_t const *>(terrain), pack,
                         static_cast<raw_t const *>(point_values), nk_size_t(points_count),
                         point_heights.raw_values_data());
            for (std::size_t pass = 0; pass != passes; ++pass)
                backend.call(clearances_kernel, static_cast<raw_t const *>(terrain), pack,
                             static_cast<raw_t const *>(viewpoint_values),
                             static_cast<raw_t const *>(pass == 1 ? view_values : nullptr), nk_size_t(viewpoints_count),
                             static_cast<raw_t const *>(target_values), nk_size_t(targets_count), refractions[pass],
                             clearances.raw_values_data() + pass * cells, targets_count * sizeof(raw_t));
            if (!stats.expect_status(backend.synchronize())) return stats;

            // Heights and viewed clearances against their templates in f64, as the kernels compute
            std::vector<f64_t> expected(nk::max_of(points_count, cells));
            terrain_heights_reference<f64_t>(heights.values_data(), rows, columns, padded * sizeof(raw_t),
                                             points.values_data(), points_count, expected.data());
            for (std::size_t point = 0; point != points_count; ++point)
                stats.expect_true(
                    same(point_heights.raw_values_data()[point], static_cast<raw_t>(expected[point].raw_)),
                    "a terrain height differs from its template");
            clearances_reference<f64_t>(heights.values_data(), rows, columns, padded * sizeof(raw_t), row_metres,
                                        column_metres, viewpoints.values_data(), views.values_data(), viewpoints_count,
                                        targets.values_data(), targets_count, refractions[1], expected.data(),
                                        targets_count * sizeof(f64_t));
            raw_t const *viewed = clearances.raw_values_data() + cells;
            for (std::size_t cell = 0; cell != cells; ++cell)
                stats.expect_true(
                    same(viewed[cell], static_cast<raw_t>(expected[cell].raw_)) &&
                        (viewed[cell] == -infinity || same(viewed[cell], clearances.raw_values_data()[cell])),
                    "a view changes a clearance other than by hiding it as its template does");

            // Every crossing in turn, without the pyramid, in the arithmetic the kernels use.
            double const last_row = double(rows - 1), last_column = double(columns - 1);
            auto const clamp = [](nk_f64_t value, nk_f64_t last) {
                return value < 0 ? 0 : (value > last ? last : value);
            };
            for (std::size_t pass = 0; pass != passes; ++pass) {
                if (pass == 1) continue;
                nk_f64_t const drop_scale = (1.0 - refractions[pass]) / (2.0 * NUMKONG_EARTH_MEAN_RADIUS);
                for (std::size_t cell = 0; cell != cells; ++cell) {
                    raw_t const *viewpoint = viewpoint_values + cell / targets_count * 3;
                    raw_t const *target = target_values + cell % targets_count * 3;
                    bool const on_grid = viewpoint[0] >= 0 && viewpoint[0] <= last_column && viewpoint[1] >= 0 &&
                                         viewpoint[1] <= last_row && target[0] >= 0 && target[0] <= last_column &&
                                         target[1] >= 0 && target[1] <= last_row;
                    nk_f64_t clearance = on_grid ? std::numeric_limits<nk_f64_t>::infinity() : not_a_number;
                    nk_f64_t const column_start = viewpoint[0], column_delta = (nk_f64_t)target[0] - viewpoint[0];
                    nk_f64_t const row_start = viewpoint[1], row_delta = (nk_f64_t)target[1] - viewpoint[1];
                    nk_f64_t const east = column_delta * column_metres, south = row_delta * row_metres;
                    nk_f64_t const distance = std::sqrt(east * east + south * south);
                    nk_f64_t const height_delta = (nk_f64_t)target[2] - drop_scale * distance * distance - viewpoint[2];
                    auto const fold = [&](nk_f64_t u, nk_f64_t first, nk_f64_t second, nk_f64_t fraction) {
                        nk_f64_t const ground = fraction == 0 ? first : first + fraction * (second - first);
                        if (ground != ground) return;
                        nk_f64_t const along = u * distance;
                        nk_f64_t const gap = viewpoint[2] + u * height_delta - (ground - drop_scale * along * along);
                        clearance = nk::min_of(clearance, gap);
                    };
                    for (std::size_t column = 0; on_grid && column != columns && column_delta != 0; ++column) {
                        nk_f64_t const u = ((nk_f64_t)column - column_start) / column_delta;
                        if (!(u > 0 && u < 1)) continue;
                        nk_f64_t const row = clamp(row_start + u * row_delta, last_row);
                        std::size_t const row_first = std::size_t(row),
                                          row_second = nk::min_of(row_first + 1, rows - 1);
                        fold(u, terrain[row_first * padded + column], terrain[row_second * padded + column],
                             row - (nk_f64_t)row_first);
                    }
                    for (std::size_t row = 0; on_grid && row != rows && row_delta != 0; ++row) {
                        nk_f64_t const u = ((nk_f64_t)row - row_start) / row_delta;
                        if (!(u > 0 && u < 1)) continue;
                        nk_f64_t const column = clamp(column_start + u * column_delta, last_column);
                        std::size_t const column_first = std::size_t(column);
                        std::size_t const column_second = nk::min_of(column_first + 1, columns - 1);
                        fold(u, terrain[row * padded + column_first], terrain[row * padded + column_second],
                             column - (nk_f64_t)column_first);
                    }
                    stats.expect_true(same(clearances.raw_values_data()[pass * cells + cell], (raw_t)clearance),
                                      "a pyramided clearance differs from the visit of every crossing");
                }
            }
        }

    // A flat sea 50 km wide, of 100 m cells, with a viewpoint, two targets and a view in 13 values.
    std::size_t const sea_columns = 501;
    auto sea = make_vector<scalar_t>(backend, sea_columns);
    nk_size_t sea_bytes = 0;
    stats.expect_status(pack_size(1, sea_columns, &sea_bytes));
    auto sea_packed = make_vector<std::byte>(backend, sea_bytes);
    auto sea_points = make_vector<scalar_t>(backend, 13);
    auto sea_results = make_vector<scalar_t>(backend, 2);
    raw_t *sea_heights = sea.raw_values_data(), *sea_values = sea_points.raw_values_data();
    raw_t *seen = sea_results.raw_values_data();
    std::fill_n(sea_heights, sea_columns, 0.0f);
    auto const pack_sea = [&] {
        backend.call(pack_kernel, static_cast<raw_t const *>(sea_heights), nk_size_t(1), sea_columns,
                     sea_columns * sizeof(raw_t), 100.0, 100.0, static_cast<void *>(sea_packed.raw_values_data()));
    };
    auto const clear = [&](nk_size_t targets, raw_t const *view, double refraction) {
        backend.call(clearances_kernel, static_cast<raw_t const *>(sea_heights),
                     static_cast<void const *>(sea_packed.raw_values_data()), static_cast<raw_t const *>(sea_values),
                     view, nk_size_t(1), static_cast<raw_t const *>(sea_values + 3), targets, refraction, seen,
                     targets * sizeof(raw_t));
        return stats.expect_status(backend.synchronize());
    };
    pack_sea();
    for (auto const [column, refraction, visible] :
         {std::tuple {224.0f, 0.0, true}, std::tuple {228.0f, 0.0, false}, std::tuple {228.0f, 1.0 / 7, true}}) {
        raw_t const masts[6] = {0.0f, 0.0f, 10.0f, column, 0.0f, 10.0f};
        std::memcpy(sea_values, masts, sizeof(masts));
        if (clear(1, nullptr, refraction))
            stats.expect_true((seen[0] > 0) == visible, "masts over a flat sea break the horizon at their distance");
    }
    sea_values[3] = 490.0f;
    if (clear(1, nullptr, 1.0))
        stats.expect_true(seen[0] == 10.0f, "without curvature the clearance is the masts' height");
    raw_t const facing_east[13] = {250.0f, 0.0f,  10.0f,      300.0f, 0.0f, 10.0f,   200.0f,
                                   0.0f,   10.0f, 1.5707964f, 0.0f,   0.5f, infinity};
    std::memcpy(sea_values, facing_east, sizeof(facing_east));
    if (clear(2, sea_values + 9, 1.0))
        stats.expect_true(seen[0] == 10.0f && seen[1] == -infinity, "a view facing east sees east and not west");
    sea_values[12] = 1000.0f;
    if (clear(2, sea_values + 9, 1.0))
        stats.expect_true(seen[0] == -infinity, "a view reaches no farther than its range");
    sea_heights[10] = 4.0f, sea_heights[11] = 8.0f;
    pack_sea();
    raw_t const probes[4] = {10.25f, 0.0f, 600.0f, 0.0f};
    std::memcpy(sea_values, probes, sizeof(probes));
    backend.call(heights_kernel, static_cast<raw_t const *>(sea_heights),
                 static_cast<void const *>(sea_packed.raw_values_data()), static_cast<raw_t const *>(sea_values),
                 nk_size_t(2), seen);
    if (stats.expect_status(backend.synchronize()))
        stats.expect_true(seen[0] == 5.0f && seen[1] != seen[1], "heights interpolate and leave the grid as NaN");
    return stats;
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_GEOSPATIAL_HPP
