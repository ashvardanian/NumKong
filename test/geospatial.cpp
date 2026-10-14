/**
 *  @file test/geospatial.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief Many-to-many 3D squares, Haversine, Vincenty and terrain clearance tests.
 */

#include "harness.hpp"
#include "numkong/geospatial.hpp" // `nk::haversines`, `nk::vincentys`

namespace ashvardanian::numkong::test {

using sqeuclideans3d_i32_kernel_t = nk_status_t (*)(nk_i32_t const *, nk_size_t, nk_i32_t const *, nk_size_t,
                                                    nk_u64_t *, nk_size_t, nk_stream_t);
using sqeuclideans3d_f64_kernel_t = nk_status_t (*)(nk_f64_t const *, nk_size_t, nk_f64_t const *, nk_size_t,
                                                    nk_f64_t *, nk_size_t, nk_stream_t);
using geodesics_packed_kernel_t = nk_status_t (*)(void const *, void const *, nk_f64_t *, nk_size_t, nk_stream_t);
using clearances_packed_kernel_t = nk_status_t (*)(nk_f32_t const *, void const *, nk_f32_t const *, nk_f32_t const *,
                                                   nk_size_t, nk_f32_t const *, nk_size_t, nk_f64_t, nk_f32_t *,
                                                   nk_size_t, nk_stream_t);

/** One rows × columns case of a many-to-many family. */
struct geospatial_shape_t {
    std::size_t rows, columns;
};

/** The configured shape, then single cells and odd edges off every vector width, as
 *  @c matrix_shapes lists them. */
static std::vector<geospatial_shape_t> geospatial_shapes(settings_t const &settings) {
    return {{settings.matrix_height, settings.matrix_width}, {1, 1}, {2, 3}, {3, 2}, {17, 33}};
}

/** Exact squares of Earth-fixed centimetres against brute force in 64-bit integers. */
error_stats_t test_sqeuclideans3d_i32(settings_t const &settings, sqeuclideans3d_i32_kernel_t kernel) {
    error_stats_t stats(0.0);
    std::mt19937 generator(settings.seed.value);
    // Earth-fixed centimetres reach ±6.4e8 on the surface, so their differences fit 32 bits.
    std::uniform_int_distribution<nk_i32_t> coordinate(-680'000'000, 680'000'000);
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();)
        for (geospatial_shape_t const &shape : geospatial_shapes(settings)) {
            std::vector<nk_i32_t> a(shape.rows * 3), b(shape.columns * 3);
            std::vector<nk_u64_t> results(shape.rows * shape.columns);
            for (nk_i32_t &value : a) value = coordinate(generator);
            for (nk_i32_t &value : b) value = coordinate(generator);
            // A coincident pair pins the zero.
            for (std::size_t axis = 0; axis != 3; ++axis) b[axis] = a[axis];
            if (!stats.expect(kernel(a.data(), shape.rows, b.data(), shape.columns, results.data(),
                                     shape.columns * sizeof(nk_u64_t), nullptr)))
                return stats;
            for (std::size_t row = 0; row != shape.rows; ++row)
                for (std::size_t column = 0; column != shape.columns; ++column) {
                    nk_u64_t expected = 0;
                    for (std::size_t axis = 0; axis != 3; ++axis) {
                        nk_i64_t const difference = (nk_i64_t)a[row * 3 + axis] - (nk_i64_t)b[column * 3 + axis];
                        expected += (nk_u64_t)(difference * difference);
                    }
                    stats.expect(results[row * shape.columns + column] == expected,
                                 "squared distance differs from brute force");
                }
        }
    return stats;
}

/** Dot2 squares of doubles against the double-double sum of exact differences. */
error_stats_t test_sqeuclideans3d_f64(settings_t const &settings, sqeuclideans3d_f64_kernel_t kernel) {
    error_stats_t stats(nk_accumulation_error_bound(nk_f64_k));
    std::mt19937 generator(settings.seed.value);
    std::uniform_real_distribution<double> coordinate(-1e7, 1e7);
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();)
        for (geospatial_shape_t const &shape : geospatial_shapes(settings)) {
            std::vector<nk_f64_t> a(shape.rows * 3), b(shape.columns * 3), results(shape.rows * shape.columns);
            for (nk_f64_t &value : a) value = coordinate(generator);
            for (nk_f64_t &value : b) value = coordinate(generator);
            if (!stats.expect(kernel(a.data(), shape.rows, b.data(), shape.columns, results.data(),
                                     shape.columns * sizeof(nk_f64_t), nullptr)))
                return stats;
            for (std::size_t row = 0; row != shape.rows; ++row)
                for (std::size_t column = 0; column != shape.columns; ++column) {
                    f118_t expected = f118_t(0.0);
                    for (std::size_t axis = 0; axis != 3; ++axis) {
                        f118_t const difference = f118_t(a[row * 3 + axis]) - f118_t(b[column * 3 + axis]);
                        expected = expected + difference * difference;
                    }
                    // A difference rounds once, its square doubles that: 2 roundings a term.
                    stats.accumulate_bounded(f64_t(results[row * shape.columns + column]), expected,
                                             3 * stats.term_error_bound * static_cast<double>(expected));
                }
        }
    return stats;
}

/** Points for a geodesic family: rows anywhere, and columns as near neighbors of the rows
 *  within @p max_separation_radians, followed by the rest drawn anywhere, all within
 *  @p cap_radians of a common centre when it is below π. */
static void fill_geodesic_points(std::mt19937 &generator, double cap_radians, double max_separation_radians,
                                 std::vector<f64_t> &a_latitudes, std::vector<f64_t> &a_longitudes,
                                 std::vector<f64_t> &b_latitudes, std::vector<f64_t> &b_longitudes) {
    f64_t centre_latitude, centre_longitude;
    nk::fill_coordinates(generator, &centre_latitude, &centre_longitude, 1);
    std::vector<f64_t> centre_latitudes(std::max(a_latitudes.size(), b_latitudes.size()), centre_latitude),
        centre_longitudes(centre_latitudes.size(), centre_longitude);
    if (cap_radians < 3.14159265358979323846) {
        nk::fill_nearby_coordinates(generator, centre_latitudes.data(), centre_longitudes.data(), a_latitudes.data(),
                                    a_longitudes.data(), a_latitudes.size(), cap_radians);
        nk::fill_nearby_coordinates(generator, centre_latitudes.data(), centre_longitudes.data(), b_latitudes.data(),
                                    b_longitudes.data(), b_latitudes.size(), cap_radians);
    }
    else {
        nk::fill_coordinates(generator, a_latitudes.data(), a_longitudes.data(), a_latitudes.size());
        nk::fill_coordinates(generator, b_latitudes.data(), b_longitudes.data(), b_latitudes.size());
    }
    nk::fill_nearby_coordinates(generator, a_latitudes.data(), a_longitudes.data(), b_latitudes.data(),
                                b_longitudes.data(), std::min(a_latitudes.size(), b_latitudes.size()),
                                max_separation_radians);
}

/** Packs @p latitudes and @p longitudes through the serial @p pack_size and @p pack routines
 *  into a fresh aligned buffer, as every capability reads it. */
template <typename pack_size_type_, typename pack_type_>
static nk::vector<f64_t> geodesic_pack(error_stats_t &stats, pack_size_type_ pack_size, pack_type_ pack,
                                       std::vector<f64_t> const &latitudes, std::vector<f64_t> const &longitudes) {
    nk_size_t bytes = 0;
    stats.expect(pack_size(latitudes.size(), &bytes));
    nk::vector<f64_t> packed = make_vector<f64_t>(bytes / sizeof(nk_f64_t));
    stats.expect(
        pack(&latitudes.data()->raw_, &longitudes.data()->raw_, latitudes.size(), packed.raw_values_data(), nullptr));
    return packed;
}

/** Many-to-many Haversine against the textbook haversine in double-double, over the whole sphere:
 *  the atan2 form of the kernels stays conditioned from coincident to antipodal points. */
error_stats_t test_haversines(settings_t const &settings, geodesics_packed_kernel_t kernel) {
    error_stats_t stats(nk_haversines_error_bound(nk_f64_k));
    std::mt19937 generator(settings.seed.value);
    double const max_separation_radians = double(settings.max_coord_angle_degrees) * 3.14159265358979323846 / 180.0;
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();)
        for (geospatial_shape_t const &shape : geospatial_shapes(settings)) {
            std::vector<f64_t> a_latitudes(shape.rows), a_longitudes(shape.rows);
            std::vector<f64_t> b_latitudes(shape.columns), b_longitudes(shape.columns);
            std::vector<nk_f64_t> results(shape.rows * shape.columns);
            std::vector<f64_t> reference(shape.rows * shape.columns);
            fill_geodesic_points(generator, 3.14159265358979323846, max_separation_radians, a_latitudes, a_longitudes,
                                 b_latitudes, b_longitudes);
            nk::vector<f64_t> const a_packed = geodesic_pack(stats, nk_haversines_pack_size_f64_serial,
                                                             nk_haversines_pack_f64_serial, a_latitudes, a_longitudes);
            nk::vector<f64_t> const b_packed = geodesic_pack(stats, nk_haversines_pack_size_f64_serial,
                                                             nk_haversines_pack_f64_serial, b_latitudes, b_longitudes);
            if (!stats.expect(kernel(a_packed.raw_values_data(), b_packed.raw_values_data(), results.data(),
                                     shape.columns * sizeof(nk_f64_t), nullptr)))
                return stats;
            nk::haversines<f118_t>(a_latitudes.data(), a_longitudes.data(), shape.rows, b_latitudes.data(),
                                   b_longitudes.data(), shape.columns, reference.data(), shape.columns * sizeof(f64_t));
            for (std::size_t cell = 0; cell != results.size(); ++cell) {
                double const expected = static_cast<double>(reference[cell]);
                double const angle = expected / NUMKONG_EARTH_MEAN_RADIUS;
                stats.accumulate_bounded(f64_t(results[cell]), expected,
                                         stats.term_error_bound * NUMKONG_EARTH_MEAN_RADIUS * (1 + angle));
            }
        }
    return stats;
}

/** Many-to-many Vincenty against its own formula in double-double, within a hemisphere, as
 *  the iteration need not converge near the antipode. */
error_stats_t test_vincentys(settings_t const &settings, geodesics_packed_kernel_t kernel) {
    error_stats_t stats(nk_vincentys_error_bound(nk_f64_k));
    std::mt19937 generator(settings.seed.value);
    double const cap_radians = std::min(double(settings.max_coord_angle_degrees), 90.0) * 3.14159265358979323846 /
                               360.0;
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();)
        for (geospatial_shape_t const &shape : geospatial_shapes(settings)) {
            std::vector<f64_t> a_latitudes(shape.rows), a_longitudes(shape.rows);
            std::vector<f64_t> b_latitudes(shape.columns), b_longitudes(shape.columns);
            std::vector<nk_f64_t> results(shape.rows * shape.columns);
            std::vector<f64_t> reference(shape.rows * shape.columns);
            fill_geodesic_points(generator, cap_radians, cap_radians, a_latitudes, a_longitudes, b_latitudes,
                                 b_longitudes);
            nk::vector<f64_t> const a_packed = geodesic_pack(stats, nk_vincentys_pack_size_f64_serial,
                                                             nk_vincentys_pack_f64_serial, a_latitudes, a_longitudes);
            nk::vector<f64_t> const b_packed = geodesic_pack(stats, nk_vincentys_pack_size_f64_serial,
                                                             nk_vincentys_pack_f64_serial, b_latitudes, b_longitudes);
            if (!stats.expect(kernel(a_packed.raw_values_data(), b_packed.raw_values_data(), results.data(),
                                     shape.columns * sizeof(nk_f64_t), nullptr)))
                return stats;
            nk::vincentys<f118_t>(a_latitudes.data(), a_longitudes.data(), shape.rows, b_latitudes.data(),
                                  b_longitudes.data(), shape.columns, reference.data(), shape.columns * sizeof(f64_t));
            for (std::size_t cell = 0; cell != results.size(); ++cell)
                stats.accumulate_bounded(f64_t(results[cell]), static_cast<double>(reference[cell]),
                                         stats.term_error_bound * NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
        }
    return stats;
}

/** Bytes of the terrain pack of a @p rows × @p columns grid. */
static std::size_t terrain_pack_bytes(std::size_t rows, std::size_t columns) {
    nk_size_t bytes = 0;
    return nk_terrain_pack_size_f32_serial(rows, columns, &bytes) == nk_success_k ? bytes : 0;
}

/** A terrain grid, padded past its last column, with the pack the serial routine builds and
 *  every capability reads. */
struct terrain_fixture_t {
    std::size_t rows, columns, stride;
    nk_f64_t row_metres, column_metres;
    std::vector<nk_f32_t> heights;
    nk::vector<f64_t> packed;

    terrain_fixture_t(std::size_t rows, std::size_t columns, nk_f64_t row_metres, nk_f64_t column_metres)
        : rows(rows), columns(columns), stride((columns + 3) * sizeof(nk_f32_t)), row_metres(row_metres),
          column_metres(column_metres), heights(rows * (columns + 3)),
          packed(make_vector<f64_t>(terrain_pack_bytes(rows, columns) / sizeof(nk_f64_t) + 1)) {}

    nk_f32_t &at(std::size_t row, std::size_t column) noexcept { return heights[row * (columns + 3) + column]; }
    nk_f32_t at(std::size_t row, std::size_t column) const noexcept { return heights[row * (columns + 3) + column]; }

    nk_status_t pack() noexcept {
        return nk_terrain_pack_f32_serial(heights.data(), rows, columns, stride, row_metres, column_metres,
                                          packed.raw_values_data(), nullptr);
    }
};

/** The clearance of one pair by visiting every crossing in turn, without the pyramid and
 *  without a view, in the arithmetic the kernels use: what skipping must reproduce exactly. */
static nk_f32_t clearance_unpyramided(terrain_fixture_t const &terrain, nk_f32_t const *viewpoint,
                                      nk_f32_t const *target, nk_f64_t refraction) {
    double const last_row = double(terrain.rows - 1), last_column = double(terrain.columns - 1);
    auto on_grid = [&](nk_f32_t const *point) {
        return point[0] >= 0 && point[0] <= last_column && point[1] >= 0 && point[1] <= last_row;
    };
    if (!on_grid(viewpoint) || !on_grid(target)) return std::numeric_limits<nk_f32_t>::quiet_NaN();
    nk_f64_t const drop_scale = (1.0 - refraction) / (2.0 * NUMKONG_EARTH_MEAN_RADIUS);
    nk_f64_t const column_start = viewpoint[0], column_delta = (nk_f64_t)target[0] - viewpoint[0];
    nk_f64_t const row_start = viewpoint[1], row_delta = (nk_f64_t)target[1] - viewpoint[1];
    nk_f64_t const east = column_delta * terrain.column_metres, south = row_delta * terrain.row_metres;
    nk_f64_t const distance = std::sqrt(east * east + south * south);
    nk_f64_t const height_start = viewpoint[2];
    nk_f64_t const height_delta = (nk_f64_t)target[2] - drop_scale * distance * distance - viewpoint[2];
    nk_f64_t clearance = std::numeric_limits<nk_f64_t>::infinity();
    auto fold = [&](nk_f64_t u, nk_f64_t first, nk_f64_t second, nk_f64_t fraction) {
        nk_f64_t const ground = fraction == 0 ? first : first + fraction * (second - first);
        if (ground != ground) return;
        nk_f64_t const along = u * distance;
        nk_f64_t const gap = height_start + u * height_delta - (ground - drop_scale * along * along);
        clearance = std::min(clearance, gap);
    };
    auto clamp = [](nk_f64_t value, nk_f64_t last) { return value < 0 ? 0 : (value > last ? last : value); };
    for (std::size_t column = 0; column != terrain.columns && column_delta != 0; ++column) {
        nk_f64_t const u = ((nk_f64_t)column - column_start) / column_delta;
        if (!(u > 0 && u < 1)) continue;
        nk_f64_t const row = clamp(row_start + u * row_delta, last_row);
        std::size_t const row_first = std::size_t(row);
        std::size_t const row_second = std::min(row_first + 1, terrain.rows - 1);
        fold(u, terrain.at(row_first, column), terrain.at(row_second, column), row - (nk_f64_t)row_first);
    }
    for (std::size_t row = 0; row != terrain.rows && row_delta != 0; ++row) {
        nk_f64_t const u = ((nk_f64_t)row - row_start) / row_delta;
        if (!(u > 0 && u < 1)) continue;
        nk_f64_t const column = clamp(column_start + u * column_delta, last_column);
        std::size_t const column_first = std::size_t(column);
        std::size_t const column_second = std::min(column_first + 1, terrain.columns - 1);
        fold(u, terrain.at(row, column_first), terrain.at(row, column_second), column - (nk_f64_t)column_first);
    }
    return (nk_f32_t)clearance;
}

/** Clearances over rough random terrain against the unpyramided visit, exactly, then the closed
 *  forms a flat sea gives for curvature and refraction, for the edges of a view and of the grid. */
error_stats_t test_clearances(settings_t const &settings, clearances_packed_kernel_t kernel) {
    error_stats_t stats(0.0);
    std::mt19937 generator(settings.seed.value);

    // Rough terrain: a random walk along rows and columns, with a few missing cells.
    terrain_fixture_t rough(97, 83, 30.0, 21.0);
    std::normal_distribution<float> step(0.0f, 4.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::size_t const viewpoints_count = 13, targets_count = 29;
    std::vector<nk_f32_t> viewpoints(viewpoints_count * 3), views(viewpoints_count * 4), targets(targets_count * 3);
    std::vector<nk_f32_t> results(viewpoints_count * targets_count);
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        for (std::size_t row = 0; row != rough.rows; ++row)
            for (std::size_t column = 0; column != rough.columns; ++column) {
                float const north = row ? rough.at(row - 1, column) : 100.0f;
                float const west = column ? rough.at(row, column - 1) : north;
                rough.at(row, column) = (north + west) / 2 + step(generator);
            }
        rough.at(rough.rows / 2, rough.columns / 3) = std::numeric_limits<nk_f32_t>::quiet_NaN();
        stats.expect(rough.pack());
        auto place = [&](nk_f32_t *point) {
            // A few points fall off the grid, the rest land on it, 2 to 60 metres above the ground.
            point[0] = unit(generator) * float(rough.columns + 1) - 1.0f;
            point[1] = unit(generator) * float(rough.rows + 1) - 1.0f;
            std::size_t const row = std::min<std::size_t>(rough.rows - 1, std::size_t(std::max(point[1], 0.0f)));
            std::size_t const column = std::min<std::size_t>(rough.columns - 1, std::size_t(std::max(point[0], 0.0f)));
            float const ground = rough.at(row, column);
            point[2] = (ground == ground ? ground : 100.0f) + 2.0f + unit(generator) * 58.0f;
        };
        for (std::size_t viewpoint = 0; viewpoint != viewpoints_count; ++viewpoint) {
            place(&viewpoints[viewpoint * 3]);
            views[viewpoint * 4 + 0] = unit(generator) * 6.2831853f;
            views[viewpoint * 4 + 1] = viewpoint % 3 ? (unit(generator) - 0.5f)
                                                     : std::numeric_limits<float>::quiet_NaN();
            views[viewpoint * 4 + 2] = 0.3f + unit(generator) * 3.0f;
            views[viewpoint * 4 + 3] = viewpoint % 4 ? 3000.0f * unit(generator)
                                                     : std::numeric_limits<float>::infinity();
        }
        for (std::size_t target = 0; target != targets_count; ++target) place(&targets[target * 3]);

        if (!stats.expect(kernel(rough.heights.data(), rough.packed.raw_values_data(), viewpoints.data(), nullptr,
                                 viewpoints_count, targets.data(), targets_count, 1.0 / 7, results.data(),
                                 targets_count * sizeof(nk_f32_t), nullptr)))
            return stats;
        for (std::size_t viewpoint = 0; viewpoint != viewpoints_count; ++viewpoint)
            for (std::size_t target = 0; target != targets_count; ++target) {
                nk_f32_t const expected = clearance_unpyramided(rough, &viewpoints[viewpoint * 3], &targets[target * 3],
                                                                1.0 / 7);
                nk_f32_t const actual = results[viewpoint * targets_count + target];
                bool const same = actual == expected || (actual != actual && expected != expected);
                stats.expect(same, "pyramided clearance differs from the unpyramided visit");
            }
        // A view only ever replaces a clearance by -∞, so other cells keep their unviewed values.
        std::vector<nk_f32_t> viewed(results.size());
        if (!stats.expect(kernel(rough.heights.data(), rough.packed.raw_values_data(), viewpoints.data(), views.data(),
                                 viewpoints_count, targets.data(), targets_count, 1.0 / 7, viewed.data(),
                                 targets_count * sizeof(nk_f32_t), nullptr)))
            return stats;
        for (std::size_t cell = 0; cell != results.size(); ++cell)
            stats.expect(viewed[cell] == -std::numeric_limits<nk_f32_t>::infinity() || viewed[cell] == results[cell] ||
                             (viewed[cell] != viewed[cell] && results[cell] != results[cell]),
                         "a view changes a clearance other than by hiding it");
    }

    // A flat sea 50 km wide: two 10 m masts see each other across twice the distance to the horizon
    // from 10 m, 22.58 km without refraction and 24.38 km with the 1/7 of light.
    terrain_fixture_t sea(1, 501, 100.0, 100.0);
    stats.expect(sea.pack());
    auto clearance_at = [&](float column, double refraction) {
        nk_f32_t const viewpoint[3] = {0.0f, 0.0f, 10.0f}, target[3] = {column, 0.0f, 10.0f};
        nk_f32_t clearance = 0;
        stats.expect(kernel(sea.heights.data(), sea.packed.raw_values_data(), viewpoint, nullptr, 1, target, 1,
                            refraction, &clearance, sizeof(nk_f32_t), nullptr));
        return clearance;
    };
    stats.expect(clearance_at(224, 0.0) > 0, "masts 22.4 km apart see each other over a flat sea");
    stats.expect(clearance_at(228, 0.0) < 0, "masts 22.8 km apart hide behind the bulge of a flat sea");
    stats.expect(clearance_at(228, 1.0 / 7) > 0, "refraction of light lifts masts 22.8 km apart into view");
    stats.expect(clearance_at(490, 1.0) == 10.0f, "without curvature the clearance over a flat sea is the height");

    // A view facing east sees an eastern target and not a western one; a short range hides both.
    nk_f32_t const centre[3] = {250.0f, 0.0f, 10.0f}, east_west[6] = {300.0f, 0.0f, 10.0f, 200.0f, 0.0f, 10.0f};
    nk_f32_t const facing_east[4] = {1.5707964f, 0.0f, 0.5f, std::numeric_limits<float>::infinity()};
    nk_f32_t const short_range[4] = {1.5707964f, 0.0f, 0.5f, 1000.0f};
    nk_f32_t seen[2];
    stats.expect(kernel(sea.heights.data(), sea.packed.raw_values_data(), centre, facing_east, 1, east_west, 2, 1.0,
                        seen, 2 * sizeof(nk_f32_t), nullptr));
    stats.expect(seen[0] == 10.0f && seen[1] == -std::numeric_limits<float>::infinity(),
                 "a view facing east sees east and not west");
    stats.expect(kernel(sea.heights.data(), sea.packed.raw_values_data(), centre, short_range, 1, east_west, 2, 1.0,
                        seen, 2 * sizeof(nk_f32_t), nullptr));
    stats.expect(seen[0] == -std::numeric_limits<float>::infinity(), "a view reaches no farther than its range");

    // Heights interpolate bilinearly and are NaN off the grid.
    sea.at(0, 10) = 4.0f, sea.at(0, 11) = 8.0f;
    stats.expect(sea.pack());
    nk_f32_t const points[4] = {10.25f, 0.0f, 600.0f, 0.0f};
    nk_f32_t heights[2];
    stats.expect(nk_terrain_heights_packed_f32_serial(sea.heights.data(), sea.packed.raw_values_data(), points, 2,
                                                      heights, nullptr));
    stats.expect(heights[0] == 5.0f && heights[1] != heights[1], "heights interpolate and leave the grid as NaN");
    return stats;
}

void test_geospatial(error_stats_section_t &check) {

    check.section("Geospatial Functions Serial", nk_cap_serial_k);
    check("sqeuclideans3d_i32_serial", test_sqeuclideans3d_i32, nk_sqeuclideans3d_i32_serial);
    check("sqeuclideans3d_f64_serial", test_sqeuclideans3d_f64, nk_sqeuclideans3d_f64_serial);
    check("haversines_packed_f64_serial", test_haversines, nk_haversines_packed_f64_serial);
    check("vincentys_packed_f64_serial", test_vincentys, nk_vincentys_packed_f64_serial);
    check("clearances_packed_f32_serial", test_clearances, nk_clearances_packed_f32_serial);

#if !NUMKONG_HEADER_ONLY
    check.section("Geospatial Functions Runtime Dispatch", nk_cap_serial_k);
    check("sqeuclideans3d_i32", test_sqeuclideans3d_i32, cpu_best<nk_sqeuclideans3d_i32_best>);
    check("sqeuclideans3d_f64", test_sqeuclideans3d_f64, cpu_best<nk_sqeuclideans3d_f64_best>);
    check("haversines_packed_f64", test_haversines, cpu_best<nk_haversines_packed_f64_best>);
    check("vincentys_packed_f64", test_vincentys, cpu_best<nk_vincentys_packed_f64_best>);
    check("clearances_packed_f32", test_clearances, cpu_best<nk_clearances_packed_f32_best>);
#endif

#if NUMKONG_TARGET_NEON
    check.section("Geospatial Functions NEON", nk_cap_neon_k);
    check("sqeuclideans3d_i32_neon", test_sqeuclideans3d_i32, nk_sqeuclideans3d_i32_neon);
    check("sqeuclideans3d_f64_neon", test_sqeuclideans3d_f64, nk_sqeuclideans3d_f64_neon);
    check("haversines_packed_f64_neon", test_haversines, nk_haversines_packed_f64_neon);
    check("vincentys_packed_f64_neon", test_vincentys, nk_vincentys_packed_f64_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_HASWELL
    check.section("Geospatial Functions Haswell", nk_cap_haswell_k);
    check("sqeuclideans3d_i32_haswell", test_sqeuclideans3d_i32, nk_sqeuclideans3d_i32_haswell);
    check("sqeuclideans3d_f64_haswell", test_sqeuclideans3d_f64, nk_sqeuclideans3d_f64_haswell);
    check("haversines_packed_f64_haswell", test_haversines, nk_haversines_packed_f64_haswell);
    check("vincentys_packed_f64_haswell", test_vincentys, nk_vincentys_packed_f64_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    check.section("Geospatial Functions Skylake", nk_cap_skylake_k);
    check("sqeuclideans3d_i32_skylake", test_sqeuclideans3d_i32, nk_sqeuclideans3d_i32_skylake);
    check("sqeuclideans3d_f64_skylake", test_sqeuclideans3d_f64, nk_sqeuclideans3d_f64_skylake);
    check("haversines_packed_f64_skylake", test_haversines, nk_haversines_packed_f64_skylake);
    check("vincentys_packed_f64_skylake", test_vincentys, nk_vincentys_packed_f64_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_RVV
    check.section("Geospatial Functions RVV", nk_cap_rvv_k);
    check("sqeuclideans3d_i32_rvv", test_sqeuclideans3d_i32, nk_sqeuclideans3d_i32_rvv);
    check("sqeuclideans3d_f64_rvv", test_sqeuclideans3d_f64, nk_sqeuclideans3d_f64_rvv);
    check("haversines_packed_f64_rvv", test_haversines, nk_haversines_packed_f64_rvv);
    check("vincentys_packed_f64_rvv", test_vincentys, nk_vincentys_packed_f64_rvv);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
    check.section("Geospatial Functions V128 Relaxed", nk_cap_v128relaxed_k);
    check("sqeuclideans3d_i32_v128relaxed", test_sqeuclideans3d_i32, nk_sqeuclideans3d_i32_v128relaxed);
    check("sqeuclideans3d_f64_v128relaxed", test_sqeuclideans3d_f64, nk_sqeuclideans3d_f64_v128relaxed);
    check("haversines_packed_f64_v128relaxed", test_haversines, nk_haversines_packed_f64_v128relaxed);
    check("vincentys_packed_f64_v128relaxed", test_vincentys, nk_vincentys_packed_f64_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED
}

} // namespace ashvardanian::numkong::test
