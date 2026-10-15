/**
 *  @file test/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batch operation tests - CUDA capabilities.
 *
 *  Compiled with every capability some listed architecture runs: the CUDA baseline, Ampere, Hopper,
 *  Blackwell and Blackwell RTX.
 */
#include "cross_device.hpp"
#include "numkong/geospatial.h"        // `nk_haversines_packed_f64_cuda`, `nk_haversines_error_bound`
#include "numkong/geospatial/serial.h" // `nk_geospatial_packed_header_t`, `nk_terrain_packed_header_t`
#include "numkong/shapes/serial.h"     // `nk_shapes_packed_header_f32_t`
#include "numkong/shapes.hpp"          // `nk::shapes_transport_epsilon_k`

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_

#if NUMKONG_TARGET_BLACKWELL

/** Packs @p count random clouds of @p points each with @p pack into device memory, back to back,
 *  and the table of their addresses after them. */
template <typename pack_size_type_, typename pack_type_>
auto device_shape_packs(settings_t const &settings, cuda_backend_t &backend, std::mt19937 &generator, std::size_t count,
                        std::size_t points, pack_size_type_ pack_size, pack_type_ pack, void const ***pointers) {
    nk_size_t bytes = 0;
    pack_size(points, &bytes);
    std::size_t const stride = nk_size_round_up_to_multiple_(bytes, nk_default_alignment_k);
    auto packs = make_vector<char>(backend, count * stride + count * sizeof(void const *));
    auto cloud = make_vector<f16_t>(points * 3);
    *pointers = reinterpret_cast<void const **>(packs.raw_values_data() + count * stride);
    for (std::size_t index = 0; index != count; ++index) {
        fill_random(settings, generator, cloud);
        pack(cloud.raw_values_data(), points, packs.raw_values_data() + index * stride, nullptr);
        (*pointers)[index] = packs.raw_values_data() + index * stride;
    }
    return packs;
}

/** Superposed RMSD of F16 frames on the device against the serial kernel, whose fold it repeats
 *  over products the tensor cores sum in another order, within the bound of the one-to-one F16
 *  kernels: n roundings of the F32 moments, n × ε × spread², through the square root. */
error_stats_t test_rmsds_packed_f16_blackwell(settings_t const &settings) {
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<cuda_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 37, columns = 45;
    void const **row_pointers, **column_pointers;
    auto row_packs = device_shape_packs(settings, backend, generator, rows, points, nk_shapes_pack_size_f16_serial,
                                        nk_shapes_pack_f16_serial, &row_pointers);
    auto column_packs = device_shape_packs(settings, backend, generator, columns, points,
                                           nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, &column_pointers);
    auto results = make_vector<f32_t>(backend, rows * columns);
    backend.call(nk_rmsds_packed_f16_blackwell, row_pointers, rows, column_pointers, columns, results.raw_values_data(),
                 columns * sizeof(nk_f32_t));
    if (!stats.expect(backend.synchronize())) return stats;
    std::vector<nk_f32_t> expected(rows * columns);
    stats.expect(nk_rmsds_packed_f16_serial(row_pointers, rows, column_pointers, columns, expected.data(),
                                            columns * sizeof(nk_f32_t), nullptr));
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            auto const radius = [](void const *packed) {
                nk_f32_t const *spreads = ((nk_shapes_packed_header_f32_t const *)packed)->spreads;
                return std::sqrt(static_cast<nk_f64_t>(spreads[0]) + spreads[1] + spreads[2]);
            };
            nk_f64_t const spread = radius(row_pointers[row]) + radius(column_pointers[column]);
            nk_f64_t const reference = expected[row * columns + column];
            nk_f64_t const squares = points * nk_accumulation_error_bound(nk_f32_k) * spread * spread;
            nk_f64_t const bound = reference > 0 ? std::min(std::sqrt(squares), squares / reference)
                                                 : std::sqrt(squares);
            stats.accumulate_bounded(f32_t::from_raw(results.raw_values_data()[row * columns + column]),
                                     f64_t(reference), bound);
        }
    return stats;
}

/** Chamfer or Hausdorff of F16 clouds on the device against the serial kernel. The products expand
 *  ‖a − b‖² into ‖a‖² + ‖b‖² − 2 × a · b, each norm split into two F16 halves, which costs F32
 *  roundings of the norms rather than of the distance, so a nearest distance moves by the root of
 *  a few of them. */
template <nk_kernel_kind_t kind_>
error_stats_t test_shapes_flips_f16_blackwell(settings_t const &settings) {
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<cuda_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 7, columns = 9;
    void const **row_pointers, **column_pointers;
    auto row_packs = device_shape_packs(settings, backend, generator, rows, points, nk_shapes_pack_size_f16_serial,
                                        nk_shapes_pack_f16_serial, &row_pointers);
    auto column_packs = device_shape_packs(settings, backend, generator, columns, points + 3,
                                           nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, &column_pointers);
    auto results = make_vector<f32_t>(backend, rows * columns);
    auto const kernel = kind_ == nk_kernel_chamfers_packed_k ? nk_chamfers_packed_f16_blackwell
                                                             : nk_hausdorffs_packed_f16_blackwell;
    auto const serial = kind_ == nk_kernel_chamfers_packed_k ? nk_chamfers_packed_f16_serial
                                                             : nk_hausdorffs_packed_f16_serial;
    backend.call(kernel, row_pointers, rows, column_pointers, columns, results.raw_values_data(),
                 columns * sizeof(nk_f32_t), nullptr);
    if (!stats.expect(backend.synchronize())) return stats;
    std::vector<nk_f32_t> expected(rows * columns);
    stats.expect(serial(row_pointers, rows, column_pointers, columns, expected.data(), columns * sizeof(nk_f32_t),
                        nullptr, nullptr));
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            auto const radius = [](void const *packed) {
                return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
            };
            nk_f64_t const reach = radius(row_pointers[row]) + radius(column_pointers[column]);
            nk_f64_t const squares = 8 * nk_accumulation_error_bound(nk_f32_k) * reach * reach;
            stats.accumulate_bounded(f32_t::from_raw(results.raw_values_data()[row * columns + column]),
                                     f64_t(expected[row * columns + column]), std::sqrt(squares));
        }
    return stats;
}
#endif // NUMKONG_TARGET_BLACKWELL

/** Packs @p count random clouds of @p points each of @p scalar_type_ with @p pack into device
 *  memory, back to back, and the table of their addresses after them. */
template <typename scalar_type_, typename pack_size_type_, typename pack_type_>
auto device_transport_packs(settings_t const &settings, cuda_backend_t &backend, std::mt19937 &generator,
                            std::size_t count, std::size_t points, pack_size_type_ pack_size, pack_type_ pack,
                            void const ***pointers) {
    nk_size_t bytes = 0;
    nk_unused_(pack_size(points, &bytes));
    std::size_t const stride = nk_size_round_up_to_multiple_(bytes, nk_default_alignment_k);
    auto packs = make_vector<char>(backend, count * stride + count * sizeof(void const *));
    auto cloud = make_vector<scalar_type_>(points * 3);
    *pointers = reinterpret_cast<void const **>(packs.raw_values_data() + count * stride);
    for (std::size_t index = 0; index != count; ++index) {
        fill_random(settings, generator, cloud);
        nk_unused_(pack(cloud.raw_values_data(), points, packs.raw_values_data() + index * stride, nullptr));
        (*pointers)[index] = packs.raw_values_data() + index * stride;
    }
    return packs;
}

/** The CUDA transport kernels against the serial ones on clouds of unequal sizes, within the bound
 *  the CPU capabilities meet: the roundings of a plan of up to n + m − 1 arcs, each costing at most
 *  the squared sum of both radii, and the simplex tolerance, through the square root. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename pack_size_type_, typename pack_type_,
          typename workspace_size_type_, typename kernel_type_, typename serial_size_type_, typename serial_type_>
error_stats_t test_transport_packed_cuda(settings_t const &settings, pack_size_type_ pack_size, pack_type_ pack,
                                         workspace_size_type_ workspace_size, kernel_type_ kernel,
                                         serial_size_type_ serial_workspace_size, serial_type_ serial) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<cuda_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 5, columns = 7, largest = points + 3;
    void const **row_pointers, **column_pointers;
    auto row_packs = device_transport_packs<scalar_type_>(settings, backend, generator, rows, points, pack_size, pack,
                                                          &row_pointers);
    auto column_packs = device_transport_packs<scalar_type_>(settings, backend, generator, columns, largest, pack_size,
                                                             pack, &column_pointers);
    nk_size_t bytes = 0, serial_bytes = 0;
    nk_unused_(workspace_size(largest, largest, &bytes));
    nk_unused_(serial_workspace_size(largest, largest, &serial_bytes));
    auto workspace = make_vector<char>(backend, bytes);
    auto results = make_vector<distance_t>(backend, rows * columns);
    auto rotations = make_vector<f32_t>(backend, rows * columns * 9);
    backend.call(kernel, row_pointers, rows, column_pointers, columns, results.raw_values_data(),
                 columns * sizeof(raw_distance_t), rotations.raw_values_data(), workspace.raw_values_data(), bytes);
    if (!stats.expect(backend.synchronize())) return stats;
    auto serial_workspace = make_vector<char>(backend, serial_bytes);
    std::vector<raw_distance_t> expected(rows * columns);
    std::vector<nk_f32_t> expected_rotations(rows * columns * 9);
    stats.expect(serial(row_pointers, rows, column_pointers, columns, expected.data(), columns * sizeof(raw_distance_t),
                        expected_rotations.data(), serial_workspace.raw_values_data(), serial_bytes, nullptr));
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0), terms = static_cast<nk_f64_t>(points + largest);
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            auto const radius = [](void const *packed) {
                if constexpr (std::is_same_v<scalar_type_, f32_t>)
                    return static_cast<nk_f64_t>(((nk_shapes_packed_header_f64_t const *)packed)->max_radius);
                else return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
            };
            nk_f64_t const reach = radius(row_pointers[row]) + radius(column_pointers[column]);
            nk_f64_t const scale = reach * reach, tolerance = nk::shapes_transport_epsilon_k * scale;
            nk_f64_t const reference = expected[row * columns + column];
            nk_f64_t bound;
            if constexpr (kind_ == nk_kernel_wassersteinprocrustes_packed_k) {
                nk_f64_t const squared = (terms + 32) * roundoff * scale + tolerance;
                bound = std::min(std::sqrt(squared), squared / reference);
            }
            else {
                nk_f64_t const loss = 4 * ((terms + 64) * roundoff * scale + tolerance);
                bound = std::min(0.5 * std::sqrt(loss), loss / (8 * reference));
            }
            stats.accumulate_bounded(distance_t::from_raw(results.raw_values_data()[row * columns + column]),
                                     f64_t(reference), bound);
        }
    return stats;
}

/** Geodesic inputs: columns anywhere within a cap, near their rows, on them, opposite them, or on
 *  the poles, the equator and the antimeridian. */
enum class geodesics_case_t { anywhere_k, nearby_k, coincident_k, antipodal_k, landmarks_k };

/** Fills @p rows row points and @p columns column points of @p kind in radians: within @p cap of
 *  a common centre unless it is π, the first of them paired by index within @p separation. */
static void fill_geodesics_case(std::mt19937 &generator, geodesics_case_t kind, double cap, double separation,
                                f64_t *a_latitudes, f64_t *a_longitudes, std::size_t rows, f64_t *b_latitudes,
                                f64_t *b_longitudes, std::size_t columns) {
    double const pi = 3.14159265358979323846;
    std::size_t const paired = std::min(rows, columns);
    if (cap < pi) {
        f64_t centre_latitude, centre_longitude;
        nk::fill_coordinates(generator, &centre_latitude, &centre_longitude, 1);
        std::vector<f64_t> centre_latitudes(std::max(rows, columns), centre_latitude),
            centre_longitudes(centre_latitudes.size(), centre_longitude);
        nk::fill_nearby_coordinates(generator, centre_latitudes.data(), centre_longitudes.data(), a_latitudes,
                                    a_longitudes, rows, cap);
        nk::fill_nearby_coordinates(generator, centre_latitudes.data(), centre_longitudes.data(), b_latitudes,
                                    b_longitudes, columns, cap);
    }
    else {
        nk::fill_coordinates(generator, a_latitudes, a_longitudes, rows);
        nk::fill_coordinates(generator, b_latitudes, b_longitudes, columns);
    }
    if (kind == geodesics_case_t::anywhere_k || kind == geodesics_case_t::nearby_k)
        nk::fill_nearby_coordinates(generator, a_latitudes, a_longitudes, b_latitudes, b_longitudes, paired,
                                    separation);
    for (std::size_t index = 0; index != paired && kind == geodesics_case_t::coincident_k; ++index)
        b_latitudes[index] = a_latitudes[index], b_longitudes[index] = a_longitudes[index];
    for (std::size_t index = 0; index != paired && kind == geodesics_case_t::antipodal_k; ++index) {
        double const longitude = double(a_longitudes[index]) + pi;
        b_latitudes[index] = -double(a_latitudes[index]);
        b_longitudes[index] = longitude > pi ? longitude - 2 * pi : longitude;
    }
    if (kind != geodesics_case_t::landmarks_k) return;
    double const latitudes[] = {pi / 2, -pi / 2, 0.0, -0.0, pi / 4, -pi / 3};
    double const longitudes[] = {0.0, pi, -pi, pi / 2, -0.0, 3.0};
    for (std::size_t index = 0; index != rows; ++index)
        a_latitudes[index] = latitudes[index % 6], a_longitudes[index] = longitudes[index / 6 % 6];
    for (std::size_t index = 0; index != columns; ++index)
        b_latitudes[index] = latitudes[index * 5 % 6], b_longitudes[index] = longitudes[(index + index / 6) % 6];
}

/** Whether two geodesic packs of @p count points hold the same header and coordinates, past the
 *  padding neither writes. */
static bool same_geodesic_packs(char const *first, char const *second, std::size_t count) {
    if (std::memcmp(first, second, sizeof(nk_geospatial_packed_header_t))) return false;
    std::size_t const axis_bytes = ((nk_geospatial_packed_header_t const *)first)->axis_stride * sizeof(nk_f64_t);
    for (std::size_t axis = 0; axis != 3; ++axis)
        if (std::memcmp(first + nk_default_alignment_k + axis * axis_bytes,
                        second + nk_default_alignment_k + axis * axis_bytes, count * sizeof(nk_f64_t)))
            return false;
    return true;
}

/** A geodesic family's CUDA packs and kernel against the serial ones: the packs byte for byte, then
 *  every distance within the bound `test/geospatial.cpp` holds the serial kernels to, over ragged
 *  shapes and coincident, near, opposite and landmark points. Vincenty's random points stay within
 *  a hemisphere, as its iteration need not converge near the antipode. */
template <nk_kernel_kind_t kind_>
error_stats_t test_geodesics_cuda(settings_t const &settings) {
    bool const spherical = kind_ == nk_kernel_haversines_packed_k;
    auto const pack_size = spherical ? nk_haversines_pack_size_f64_cuda : nk_vincentys_pack_size_f64_cuda;
    auto const pack = spherical ? nk_haversines_pack_f64_cuda : nk_vincentys_pack_f64_cuda;
    auto const kernel = spherical ? nk_haversines_packed_f64_cuda : nk_vincentys_packed_f64_cuda;
    auto const serial_pack = spherical ? nk_haversines_pack_f64_serial : nk_vincentys_pack_f64_serial;
    auto const serial = spherical ? nk_haversines_packed_f64_serial : nk_vincentys_packed_f64_serial;
    error_stats_t stats(spherical ? nk_haversines_error_bound(nk_f64_k) : nk_vincentys_error_bound(nk_f64_k));
    auto backend = make_backend<cuda_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    double const pi = 3.14159265358979323846, angle = double(settings.max_coord_angle_degrees) * pi / 180;
    double const cap = spherical ? pi : std::min(angle, pi / 2) / 2;
    struct case_t {
        geodesics_case_t kind;
        double separation;
    };
    std::vector<case_t> cases = {{geodesics_case_t::anywhere_k, spherical ? angle : cap},
                                 {geodesics_case_t::nearby_k, 1e-9},
                                 {geodesics_case_t::coincident_k, 0},
                                 {geodesics_case_t::landmarks_k, 0}};
    if (spherical) cases.push_back({geodesics_case_t::antipodal_k, 0});
    std::pair<std::size_t, std::size_t> const shapes[] = {{settings.matrix_height, settings.matrix_width},
                                                          {1, 1},
                                                          {2, 3},
                                                          {3, 2},
                                                          {17, 33}};
    for (case_t const test_case : cases)
        for (auto const [rows, columns] : shapes) {
            auto a_latitudes = make_vector<f64_t>(backend, rows), a_longitudes = make_vector<f64_t>(backend, rows);
            auto b_latitudes = make_vector<f64_t>(backend, columns);
            auto b_longitudes = make_vector<f64_t>(backend, columns);
            fill_geodesics_case(generator, test_case.kind, cap, test_case.separation, a_latitudes.values_data(),
                                a_longitudes.values_data(), rows, b_latitudes.values_data(), b_longitudes.values_data(),
                                columns);
            nk_size_t a_bytes = 0, b_bytes = 0;
            stats.expect(pack_size(rows, &a_bytes)), stats.expect(pack_size(columns, &b_bytes));
            auto a_packed = make_vector<char>(backend, a_bytes), b_packed = make_vector<char>(backend, b_bytes);
            auto results = make_vector<f64_t>(backend, rows * columns);
            backend.call(pack, a_latitudes.raw_values_data(), a_longitudes.raw_values_data(), rows,
                         static_cast<void *>(a_packed.raw_values_data()));
            backend.call(pack, b_latitudes.raw_values_data(), b_longitudes.raw_values_data(), columns,
                         static_cast<void *>(b_packed.raw_values_data()));
            backend.call(kernel, static_cast<void const *>(a_packed.raw_values_data()),
                         static_cast<void const *>(b_packed.raw_values_data()), results.raw_values_data(),
                         columns * sizeof(nk_f64_t));
            if (!stats.expect(backend.synchronize())) return stats;
            auto a_expected = make_vector<char>(a_bytes), b_expected = make_vector<char>(b_bytes);
            stats.expect(serial_pack(a_latitudes.raw_values_data(), a_longitudes.raw_values_data(), rows,
                                     a_expected.raw_values_data(), nullptr));
            stats.expect(serial_pack(b_latitudes.raw_values_data(), b_longitudes.raw_values_data(), columns,
                                     b_expected.raw_values_data(), nullptr));
            stats.expect(same_geodesic_packs(a_packed.raw_values_data(), a_expected.raw_values_data(), rows) &&
                             same_geodesic_packs(b_packed.raw_values_data(), b_expected.raw_values_data(), columns),
                         "a CUDA pack differs from the serial one");
            std::vector<nk_f64_t> expected(rows * columns);
            stats.expect(serial(a_expected.raw_values_data(), b_expected.raw_values_data(), expected.data(),
                                columns * sizeof(nk_f64_t), nullptr));
            for (std::size_t cell = 0; cell != expected.size(); ++cell) {
                double const reference = expected[cell];
                double const bound = spherical ? stats.term_error_bound * NUMKONG_EARTH_MEAN_RADIUS *
                                                     (1 + reference / NUMKONG_EARTH_MEAN_RADIUS)
                                               : stats.term_error_bound * NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
                stats.accumulate_bounded(f64_t::from_raw(results.raw_values_data()[cell]), f64_t(reference), bound);
            }
        }
    return stats;
}

/** The CUDA terrain pack, heights and clearances against the serial ones, which they repeat
 *  operation for operation, so every byte and every result must match: random walks with a missing
 *  cell, a flat sea of signed zeros, single cells, rows and columns, points off the grid and on its
 *  edges, with refractions of light, none and full, with and without views. */
error_stats_t test_terrain_cuda(settings_t const &settings) {
    error_stats_t stats(0.0);
    auto backend = make_backend<cuda_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    std::normal_distribution<float> step(0.0f, 4.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    float const not_a_number = std::numeric_limits<float>::quiet_NaN();
    float const infinity = std::numeric_limits<float>::infinity();
    auto const same = [](nk_f32_t actual, nk_f32_t expected) {
        return !std::memcmp(&actual, &expected, sizeof(nk_f32_t)) || (actual != actual && expected != expected);
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
    for (grid_t const grid : grids) {
        std::size_t const rows = grid.rows, columns = grid.columns, padded = columns + 3;
        auto heights = make_vector<f32_t>(backend, rows * padded);
        nk_f32_t *terrain = heights.raw_values_data();
        for (std::size_t row = 0; row != rows; ++row)
            for (std::size_t column = 0; column != columns; ++column) {
                float const north = row ? terrain[(row - 1) * padded + column] : 100.0f;
                float const west = column ? terrain[row * padded + column - 1] : north;
                terrain[row * padded + column] = grid.flat ? ((row + column) % 2 ? -0.0f : 0.0f)
                                                           : (north + west) / 2 + step(generator);
            }
        if (!grid.flat) terrain[rows / 2 * padded + columns / 3] = not_a_number;
        auto place = [&](nk_f32_t *point, std::size_t index) {
            // Some points fall off the grid, some onto its edges, the rest 2 to 60 metres above it.
            point[0] = unit(generator) * float(columns + 1) - 1.0f;
            point[1] = unit(generator) * float(rows + 1) - 1.0f;
            if (index % 5 == 1) point[0] = float(columns - 1);
            if (index % 7 == 2) point[1] = 0.0f;
            std::size_t const row = std::min<std::size_t>(rows - 1, std::size_t(std::max(point[1], 0.0f)));
            std::size_t const column = std::min<std::size_t>(columns - 1, std::size_t(std::max(point[0], 0.0f)));
            float const ground = terrain[row * padded + column];
            point[2] = (ground == ground ? ground : 100.0f) + 2.0f + unit(generator) * 58.0f;
        };
        auto points = make_vector<f32_t>(backend, points_count * 2);
        auto viewpoints = make_vector<f32_t>(backend, viewpoints_count * 3);
        auto views = make_vector<f32_t>(backend, viewpoints_count * 4);
        auto targets = make_vector<f32_t>(backend, targets_count * 3);
        nk_f32_t *point_values = points.raw_values_data(), *viewpoint_values = viewpoints.raw_values_data();
        nk_f32_t *view_values = views.raw_values_data(), *target_values = targets.raw_values_data();
        for (std::size_t point = 0; point != points_count; ++point) {
            nk_f32_t *coordinates = point_values + point * 2;
            coordinates[0] = unit(generator) * float(columns + 1) - 1.0f;
            coordinates[1] = unit(generator) * float(rows + 1) - 1.0f;
            if (point % 5 == 1) coordinates[0] = float(columns - 1);
            if (point % 7 == 2) coordinates[1] = float(rows - 1);
            if (point % 11 == 3) coordinates[0] = std::floor(std::max(coordinates[0], 0.0f));
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
            std::memcpy(target_values + target * 3, viewpoint_values + target * 3, 3 * sizeof(nk_f32_t));
        for (std::size_t target = 1; target < viewpoints_count; target += 4) {
            target_values[target * 3] = viewpoint_values[target * 3] + 0.25f;
            target_values[target * 3 + 1] = viewpoint_values[target * 3 + 1];
        }

        nk_size_t bytes = 0;
        stats.expect(nk_terrain_pack_size_f32_cuda(rows, columns, &bytes));
        auto packed = make_vector<char>(backend, bytes);
        auto point_heights = make_vector<f32_t>(backend, points_count);
        auto clearances = make_vector<f32_t>(backend, passes * cells);
        void const *pack = packed.raw_values_data();
        backend.call(nk_terrain_pack_f32_cuda, static_cast<nk_f32_t const *>(terrain), rows, columns,
                     padded * sizeof(nk_f32_t), 30.0, 21.0, static_cast<void *>(packed.raw_values_data()));
        backend.call(nk_terrain_heights_packed_f32_cuda, static_cast<nk_f32_t const *>(terrain), pack,
                     static_cast<nk_f32_t const *>(point_values), points_count, point_heights.raw_values_data());
        for (std::size_t pass = 0; pass != passes; ++pass)
            backend.call(nk_clearances_packed_f32_cuda, static_cast<nk_f32_t const *>(terrain), pack,
                         static_cast<nk_f32_t const *>(viewpoint_values),
                         static_cast<nk_f32_t const *>(pass == 1 ? view_values : nullptr), viewpoints_count,
                         static_cast<nk_f32_t const *>(target_values), targets_count, refractions[pass],
                         clearances.raw_values_data() + pass * cells, targets_count * sizeof(nk_f32_t));
        if (!stats.expect(backend.synchronize())) return stats;

        auto expected_packed = make_vector<char>(bytes);
        stats.expect(nk_terrain_pack_f32_serial(terrain, rows, columns, padded * sizeof(nk_f32_t), 30.0, 21.0,
                                                expected_packed.raw_values_data(), nullptr));
        stats.expect(!std::memcmp(packed.raw_values_data(), expected_packed.raw_values_data(),
                                  sizeof(nk_terrain_packed_header_t)) &&
                         !std::memcmp(packed.raw_values_data() + nk_default_alignment_k,
                                      expected_packed.raw_values_data() + nk_default_alignment_k,
                                      bytes - nk_default_alignment_k),
                     "the CUDA terrain pack differs from the serial one");
        std::vector<nk_f32_t> expected(passes * cells);
        stats.expect(nk_terrain_heights_packed_f32_serial(terrain, expected_packed.raw_values_data(), point_values,
                                                          points_count, expected.data(), nullptr));
        for (std::size_t point = 0; point != points_count; ++point)
            stats.expect(same(point_heights.raw_values_data()[point], expected[point]),
                         "a CUDA terrain height differs from the serial one");
        for (std::size_t pass = 0; pass != passes; ++pass)
            stats.expect(nk_clearances_packed_f32_serial(
                terrain, expected_packed.raw_values_data(), viewpoint_values, pass == 1 ? view_values : nullptr,
                viewpoints_count, target_values, targets_count, refractions[pass], expected.data() + pass * cells,
                targets_count * sizeof(nk_f32_t), nullptr));
        for (std::size_t cell = 0; cell != passes * cells; ++cell)
            stats.expect(same(clearances.raw_values_data()[cell], expected[cell]),
                         "a CUDA clearance differs from the serial one");
    }
    return stats;
}

void test_cross_cuda(error_stats_section_t &check) {
    check.section("Cross CUDA", nk_cap_cuda_k);
    check("dots_packed_f64_cuda", test_dots_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_cuda,
          nk_dots_pack_f64_cuda, nk_dots_packed_f64_cuda);
    check("dots_pack_f64_cuda", test_dots_pack_layout<f64_t, cuda_backend_t, nk_dots_pack_size_f64_cuda,
                                                      nk_dots_packed_shape_f64_cuda, nk_dots_pack_f64_cuda>);
    check("dots_contract_f64_cuda", test_dots_launch_contract<f64_t, cuda_backend_t, nk_dots_pack_size_f64_cuda,
                                                              nk_dots_packed_f64_cuda, nk_dots_symmetric_f64_cuda>);
    check("dots_symmetric_f64_cuda", test_dots_symmetric<f64_t, cuda_backend_t>, nk_dots_symmetric_f64_cuda);
    check("dots_packed_f32_cuda", test_dots_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_cuda,
          nk_dots_pack_f32_cuda, nk_dots_packed_f32_cuda);
    check("dots_pack_f32_cuda", test_dots_pack_layout<f32_t, cuda_backend_t, nk_dots_pack_size_f32_cuda,
                                                      nk_dots_packed_shape_f32_cuda, nk_dots_pack_f32_cuda>);
    check("dots_contract_f32_cuda", test_dots_launch_contract<f32_t, cuda_backend_t, nk_dots_pack_size_f32_cuda,
                                                              nk_dots_packed_f32_cuda, nk_dots_symmetric_f32_cuda>);
    check("dots_symmetric_f32_cuda", test_dots_symmetric<f32_t, cuda_backend_t>, nk_dots_symmetric_f32_cuda);
    check("dots_packed_bf16_cuda", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_cuda,
          nk_dots_pack_bf16_cuda, nk_dots_packed_bf16_cuda);
    check("dots_pack_bf16_cuda", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_cuda,
                                                       nk_dots_packed_shape_bf16_cuda, nk_dots_pack_bf16_cuda>);
    check("dots_contract_bf16_cuda", test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_cuda,
                                                               nk_dots_packed_bf16_cuda, nk_dots_symmetric_bf16_cuda>);
    check("dots_symmetric_bf16_cuda", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_cuda);
    check("dots_packed_f16_cuda", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_cuda,
          nk_dots_pack_f16_cuda, nk_dots_packed_f16_cuda);
    check("dots_pack_f16_cuda", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_cuda,
                                                      nk_dots_packed_shape_f16_cuda, nk_dots_pack_f16_cuda>);
    check("dots_contract_f16_cuda", test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_cuda,
                                                              nk_dots_packed_f16_cuda, nk_dots_symmetric_f16_cuda>);
    check("dots_symmetric_f16_cuda", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_cuda);
    check("dots_packed_e5m2_cuda", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_cuda,
          nk_dots_pack_e5m2_cuda, nk_dots_packed_e5m2_cuda);
    check("dots_pack_e5m2_cuda", test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_cuda,
                                                       nk_dots_packed_shape_e5m2_cuda, nk_dots_pack_e5m2_cuda>);
    check("dots_contract_e5m2_cuda", test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_cuda,
                                                               nk_dots_packed_e5m2_cuda, nk_dots_symmetric_e5m2_cuda>);
    check("dots_symmetric_e5m2_cuda", test_dots_symmetric<e5m2_t, cuda_backend_t>, nk_dots_symmetric_e5m2_cuda);
    check("dots_packed_e4m3_cuda", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_cuda,
          nk_dots_pack_e4m3_cuda, nk_dots_packed_e4m3_cuda);
    check("dots_pack_e4m3_cuda", test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_cuda,
                                                       nk_dots_packed_shape_e4m3_cuda, nk_dots_pack_e4m3_cuda>);
    check("dots_contract_e4m3_cuda", test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_cuda,
                                                               nk_dots_packed_e4m3_cuda, nk_dots_symmetric_e4m3_cuda>);
    check("dots_symmetric_e4m3_cuda", test_dots_symmetric<e4m3_t, cuda_backend_t>, nk_dots_symmetric_e4m3_cuda);
    check("dots_packed_e3m2_cuda", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_cuda,
          nk_dots_pack_e3m2_cuda, nk_dots_packed_e3m2_cuda);
    check("dots_pack_e3m2_cuda", test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_cuda,
                                                       nk_dots_packed_shape_e3m2_cuda, nk_dots_pack_e3m2_cuda>);
    check("dots_contract_e3m2_cuda", test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_cuda,
                                                               nk_dots_packed_e3m2_cuda, nk_dots_symmetric_e3m2_cuda>);
    check("dots_symmetric_e3m2_cuda", test_dots_symmetric<e3m2_t, cuda_backend_t>, nk_dots_symmetric_e3m2_cuda);
    check("dots_packed_e2m3_cuda", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_cuda,
          nk_dots_pack_e2m3_cuda, nk_dots_packed_e2m3_cuda);
    check("dots_pack_e2m3_cuda", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_cuda,
                                                       nk_dots_packed_shape_e2m3_cuda, nk_dots_pack_e2m3_cuda>);
    check("dots_contract_e2m3_cuda", test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_cuda,
                                                               nk_dots_packed_e2m3_cuda, nk_dots_symmetric_e2m3_cuda>);
    check("dots_symmetric_e2m3_cuda", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_cuda);
    check("dots_packed_e2m1_cuda", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_cuda,
          nk_dots_pack_e2m1_cuda, nk_dots_packed_e2m1_cuda);
    check("dots_pack_e2m1_cuda", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_cuda,
                                                       nk_dots_packed_shape_e2m1_cuda, nk_dots_pack_e2m1_cuda>);
    check("dots_contract_e2m1_cuda", test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_cuda,
                                                               nk_dots_packed_e2m1_cuda, nk_dots_symmetric_e2m1_cuda>);
    check("dots_symmetric_e2m1_cuda", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_cuda);
    check("dots_packed_nvfp4_cuda", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_cuda,
          nk_dots_pack_nvfp4_cuda, nk_dots_packed_nvfp4_cuda);
    check("dots_pack_nvfp4_cuda", test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_cuda,
                                                        nk_dots_packed_shape_nvfp4_cuda, nk_dots_pack_nvfp4_cuda>);
    check("dots_contract_nvfp4_cuda",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_cuda, nk_dots_packed_nvfp4_cuda,
                                    nk_dots_symmetric_nvfp4_cuda>);
    check("dots_symmetric_nvfp4_cuda", test_dots_symmetric<nvfp4_t, cuda_backend_t>, nk_dots_symmetric_nvfp4_cuda);
    check("dots_packed_mxfp4_cuda", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_cuda,
          nk_dots_pack_mxfp4_cuda, nk_dots_packed_mxfp4_cuda);
    check("dots_pack_mxfp4_cuda", test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_cuda,
                                                        nk_dots_packed_shape_mxfp4_cuda, nk_dots_pack_mxfp4_cuda>);
    check("dots_contract_mxfp4_cuda",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_cuda, nk_dots_packed_mxfp4_cuda,
                                    nk_dots_symmetric_mxfp4_cuda>);
    check("dots_symmetric_mxfp4_cuda", test_dots_symmetric<mxfp4_t, cuda_backend_t>, nk_dots_symmetric_mxfp4_cuda);
    check("dots_packed_mxfp8e4m3_cuda", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>, nk_dots_pack_size_mxfp8e4m3_cuda,
          nk_dots_pack_mxfp8e4m3_cuda, nk_dots_packed_mxfp8e4m3_cuda);
    check("dots_pack_mxfp8e4m3_cuda",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_cuda,
                                nk_dots_packed_shape_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda>);
    check("dots_contract_mxfp8e4m3_cuda",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_cuda,
                                    nk_dots_packed_mxfp8e4m3_cuda, nk_dots_symmetric_mxfp8e4m3_cuda>);
    check("dots_symmetric_mxfp8e4m3_cuda", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_cuda);
    check("dots_packed_mxfp8e5m2_cuda", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>, nk_dots_pack_size_mxfp8e5m2_cuda,
          nk_dots_pack_mxfp8e5m2_cuda, nk_dots_packed_mxfp8e5m2_cuda);
    check("dots_pack_mxfp8e5m2_cuda",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_cuda,
                                nk_dots_packed_shape_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda>);
    check("dots_contract_mxfp8e5m2_cuda",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_cuda,
                                    nk_dots_packed_mxfp8e5m2_cuda, nk_dots_symmetric_mxfp8e5m2_cuda>);
    check("dots_symmetric_mxfp8e5m2_cuda", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_cuda);
    check("dots_packed_i8_cuda", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_cuda,
          nk_dots_pack_i8_cuda, nk_dots_packed_i8_cuda);
    check("dots_pack_i8_cuda", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_cuda,
                                                     nk_dots_packed_shape_i8_cuda, nk_dots_pack_i8_cuda>);
    check("dots_contract_i8_cuda", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_cuda,
                                                             nk_dots_packed_i8_cuda, nk_dots_symmetric_i8_cuda>);
    check("dots_symmetric_i8_cuda", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_cuda);
    check("dots_packed_u8_cuda", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_cuda,
          nk_dots_pack_u8_cuda, nk_dots_packed_u8_cuda);
    check("dots_pack_u8_cuda", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_cuda,
                                                     nk_dots_packed_shape_u8_cuda, nk_dots_pack_u8_cuda>);
    check("dots_contract_u8_cuda", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_cuda,
                                                             nk_dots_packed_u8_cuda, nk_dots_symmetric_u8_cuda>);
    check("dots_symmetric_u8_cuda", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_cuda);
    check("dots_packed_i4_cuda", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_cuda,
          nk_dots_pack_i4_cuda, nk_dots_packed_i4_cuda);
    check("dots_pack_i4_cuda", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_cuda,
                                                     nk_dots_packed_shape_i4_cuda, nk_dots_pack_i4_cuda>);
    check("dots_contract_i4_cuda", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_cuda,
                                                             nk_dots_packed_i4_cuda, nk_dots_symmetric_i4_cuda>);
    check("dots_symmetric_i4_cuda", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_cuda);
    check("dots_packed_u4_cuda", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_cuda,
          nk_dots_pack_u4_cuda, nk_dots_packed_u4_cuda);
    check("dots_pack_u4_cuda", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_cuda,
                                                     nk_dots_packed_shape_u4_cuda, nk_dots_pack_u4_cuda>);
    check("dots_contract_u4_cuda", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_cuda,
                                                             nk_dots_packed_u4_cuda, nk_dots_symmetric_u4_cuda>);
    check("dots_symmetric_u4_cuda", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_cuda);
    check("angulars_packed_f64_cuda", test_angulars_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_cuda,
          nk_dots_pack_f64_cuda, nk_angulars_packed_f64_cuda);
    check("angulars_symmetric_f64_cuda", test_angulars_symmetric<f64_t, cuda_backend_t>,
          nk_angulars_symmetric_f64_cuda);
    check("euclideans_packed_f64_cuda", test_euclideans_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_cuda,
          nk_dots_pack_f64_cuda, nk_euclideans_packed_f64_cuda);
    check("euclideans_symmetric_f64_cuda", test_euclideans_symmetric<f64_t, cuda_backend_t>,
          nk_euclideans_symmetric_f64_cuda);
    check("angulars_packed_f32_cuda", test_angulars_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_cuda,
          nk_dots_pack_f32_cuda, nk_angulars_packed_f32_cuda);
    check("angulars_symmetric_f32_cuda", test_angulars_symmetric<f32_t, cuda_backend_t>,
          nk_angulars_symmetric_f32_cuda);
    check("euclideans_packed_f32_cuda", test_euclideans_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_cuda,
          nk_dots_pack_f32_cuda, nk_euclideans_packed_f32_cuda);
    check("euclideans_symmetric_f32_cuda", test_euclideans_symmetric<f32_t, cuda_backend_t>,
          nk_euclideans_symmetric_f32_cuda);
    check("angulars_packed_bf16_cuda", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_cuda,
          nk_dots_pack_bf16_cuda, nk_angulars_packed_bf16_cuda);
    check("angulars_symmetric_bf16_cuda", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_cuda);
    check("euclideans_packed_bf16_cuda", test_euclideans_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_cuda,
          nk_dots_pack_bf16_cuda, nk_euclideans_packed_bf16_cuda);
    check("euclideans_symmetric_bf16_cuda", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_cuda);
    check("angulars_packed_f16_cuda", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_cuda,
          nk_dots_pack_f16_cuda, nk_angulars_packed_f16_cuda);
    check("angulars_symmetric_f16_cuda", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_cuda);
    check("euclideans_packed_f16_cuda", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_cuda,
          nk_dots_pack_f16_cuda, nk_euclideans_packed_f16_cuda);
    check("euclideans_symmetric_f16_cuda", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_cuda);
    check("angulars_packed_e5m2_cuda", test_angulars_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_cuda,
          nk_dots_pack_e5m2_cuda, nk_angulars_packed_e5m2_cuda);
    check("angulars_symmetric_e5m2_cuda", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_cuda);
    check("euclideans_packed_e5m2_cuda", test_euclideans_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_cuda,
          nk_dots_pack_e5m2_cuda, nk_euclideans_packed_e5m2_cuda);
    check("euclideans_symmetric_e5m2_cuda", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_cuda);
    check("angulars_packed_e4m3_cuda", test_angulars_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_cuda,
          nk_dots_pack_e4m3_cuda, nk_angulars_packed_e4m3_cuda);
    check("angulars_symmetric_e4m3_cuda", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_cuda);
    check("euclideans_packed_e4m3_cuda", test_euclideans_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_cuda,
          nk_dots_pack_e4m3_cuda, nk_euclideans_packed_e4m3_cuda);
    check("euclideans_symmetric_e4m3_cuda", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_cuda);
    check("angulars_packed_e3m2_cuda", test_angulars_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_cuda,
          nk_dots_pack_e3m2_cuda, nk_angulars_packed_e3m2_cuda);
    check("angulars_symmetric_e3m2_cuda", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_cuda);
    check("euclideans_packed_e3m2_cuda", test_euclideans_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_cuda,
          nk_dots_pack_e3m2_cuda, nk_euclideans_packed_e3m2_cuda);
    check("euclideans_symmetric_e3m2_cuda", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_cuda);
    check("angulars_packed_e2m3_cuda", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_cuda,
          nk_dots_pack_e2m3_cuda, nk_angulars_packed_e2m3_cuda);
    check("angulars_symmetric_e2m3_cuda", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_cuda);
    check("euclideans_packed_e2m3_cuda", test_euclideans_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_cuda,
          nk_dots_pack_e2m3_cuda, nk_euclideans_packed_e2m3_cuda);
    check("euclideans_symmetric_e2m3_cuda", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_cuda);
    check("angulars_packed_e2m1_cuda", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_cuda,
          nk_dots_pack_e2m1_cuda, nk_angulars_packed_e2m1_cuda);
    check("angulars_symmetric_e2m1_cuda", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_cuda);
    check("euclideans_packed_e2m1_cuda", test_euclideans_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_cuda,
          nk_dots_pack_e2m1_cuda, nk_euclideans_packed_e2m1_cuda);
    check("euclideans_symmetric_e2m1_cuda", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_cuda);
    check("angulars_packed_nvfp4_cuda", test_angulars_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_cuda,
          nk_dots_pack_nvfp4_cuda, nk_angulars_packed_nvfp4_cuda);
    check("angulars_symmetric_nvfp4_cuda", test_angulars_symmetric<nvfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_nvfp4_cuda);
    check("euclideans_packed_nvfp4_cuda", test_euclideans_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_cuda,
          nk_dots_pack_nvfp4_cuda, nk_euclideans_packed_nvfp4_cuda);
    check("euclideans_symmetric_nvfp4_cuda", test_euclideans_symmetric<nvfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_nvfp4_cuda);
    check("angulars_packed_mxfp4_cuda", test_angulars_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_cuda,
          nk_dots_pack_mxfp4_cuda, nk_angulars_packed_mxfp4_cuda);
    check("angulars_symmetric_mxfp4_cuda", test_angulars_symmetric<mxfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp4_cuda);
    check("euclideans_packed_mxfp4_cuda", test_euclideans_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_cuda,
          nk_dots_pack_mxfp4_cuda, nk_euclideans_packed_mxfp4_cuda);
    check("euclideans_symmetric_mxfp4_cuda", test_euclideans_symmetric<mxfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp4_cuda);
    check("angulars_packed_mxfp8e4m3_cuda", test_angulars_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda, nk_angulars_packed_mxfp8e4m3_cuda);
    check("angulars_symmetric_mxfp8e4m3_cuda", test_angulars_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e4m3_cuda);
    check("euclideans_packed_mxfp8e4m3_cuda", test_euclideans_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda, nk_euclideans_packed_mxfp8e4m3_cuda);
    check("euclideans_symmetric_mxfp8e4m3_cuda", test_euclideans_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e4m3_cuda);
    check("angulars_packed_mxfp8e5m2_cuda", test_angulars_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda, nk_angulars_packed_mxfp8e5m2_cuda);
    check("angulars_symmetric_mxfp8e5m2_cuda", test_angulars_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e5m2_cuda);
    check("euclideans_packed_mxfp8e5m2_cuda", test_euclideans_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda, nk_euclideans_packed_mxfp8e5m2_cuda);
    check("euclideans_symmetric_mxfp8e5m2_cuda", test_euclideans_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e5m2_cuda);
    check("angulars_packed_i8_cuda", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_cuda,
          nk_dots_pack_i8_cuda, nk_angulars_packed_i8_cuda);
    check("angulars_symmetric_i8_cuda", test_angulars_symmetric<i8_t, cuda_backend_t>, nk_angulars_symmetric_i8_cuda);
    check("euclideans_packed_i8_cuda", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_cuda,
          nk_dots_pack_i8_cuda, nk_euclideans_packed_i8_cuda);
    check("euclideans_symmetric_i8_cuda", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_cuda);
    check("angulars_packed_u8_cuda", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_cuda,
          nk_dots_pack_u8_cuda, nk_angulars_packed_u8_cuda);
    check("angulars_symmetric_u8_cuda", test_angulars_symmetric<u8_t, cuda_backend_t>, nk_angulars_symmetric_u8_cuda);
    check("euclideans_packed_u8_cuda", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_cuda,
          nk_dots_pack_u8_cuda, nk_euclideans_packed_u8_cuda);
    check("euclideans_symmetric_u8_cuda", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_cuda);
    check("angulars_packed_i4_cuda", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_cuda,
          nk_dots_pack_i4_cuda, nk_angulars_packed_i4_cuda);
    check("angulars_symmetric_i4_cuda", test_angulars_symmetric<i4x2_t, cuda_backend_t>, nk_angulars_symmetric_i4_cuda);
    check("euclideans_packed_i4_cuda", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_cuda,
          nk_dots_pack_i4_cuda, nk_euclideans_packed_i4_cuda);
    check("euclideans_symmetric_i4_cuda", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_cuda);
    check("angulars_packed_u4_cuda", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_cuda,
          nk_dots_pack_u4_cuda, nk_angulars_packed_u4_cuda);
    check("angulars_symmetric_u4_cuda", test_angulars_symmetric<u4x2_t, cuda_backend_t>, nk_angulars_symmetric_u4_cuda);
    check("euclideans_packed_u4_cuda", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_cuda,
          nk_dots_pack_u4_cuda, nk_euclideans_packed_u4_cuda);
    check("euclideans_symmetric_u4_cuda", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_cuda);
    check("attention_packed_bf16_cuda", test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_cuda, nk_attention_pack_bf16_cuda, nk_attention_packed_bf16_cuda);
    check("attention_packed_f16_cuda", test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_cuda, nk_attention_pack_f16_cuda, nk_attention_packed_f16_cuda);
    check("attention_packed_e4m3_cuda", test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_cuda, nk_attention_pack_e4m3_cuda, nk_attention_packed_e4m3_cuda);
    check("attention_packed_i8_cuda", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_cuda, nk_attention_pack_i8_cuda, nk_attention_packed_i8_cuda);
    check("attention_packed_gradients_bf16_cuda", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_cuda, nk_attention_pack_bf16_cuda, nk_attention_packed_gradients_bf16_cuda);
    check("attention_rope_f32_cuda", test_attention_rope<f32_t, cuda_backend_t>, nk_attention_rope_f32_cuda);
    check("attention_rope_bf16_cuda", test_attention_rope<bf16_t, cuda_backend_t>, nk_attention_rope_bf16_cuda);
    check("attention_rope_e4m3_cuda", test_attention_rope<e4m3_t, cuda_backend_t>, nk_attention_rope_e4m3_cuda);
    check("wassersteinprocrustes_packed_f32_cuda",
          test_transport_packed_cuda<f32_t, nk_kernel_wassersteinprocrustes_packed_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_cuda,
          nk_wassersteinprocrustes_packed_f32_cuda, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_serial);
    check("wassersteinprocrustes_packed_f16_cuda",
          test_transport_packed_cuda<f16_t, nk_kernel_wassersteinprocrustes_packed_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_cuda,
          nk_wassersteinprocrustes_packed_f16_cuda, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_serial);
    check("gromovwassersteins_packed_f32_cuda",
          test_transport_packed_cuda<f32_t, nk_kernel_gromovwassersteins_packed_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_cuda,
          nk_gromovwassersteins_packed_f32_cuda, nk_gromovwassersteins_workspace_size_f32_serial,
          nk_gromovwassersteins_packed_f32_serial);
    check("gromovwassersteins_packed_f16_cuda",
          test_transport_packed_cuda<f16_t, nk_kernel_gromovwassersteins_packed_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_cuda,
          nk_gromovwassersteins_packed_f16_cuda, nk_gromovwassersteins_workspace_size_f16_serial,
          nk_gromovwassersteins_packed_f16_serial);
    check("haversines_packed_f64_cuda", test_geodesics_cuda<nk_kernel_haversines_packed_k>);
    check("vincentys_packed_f64_cuda", test_geodesics_cuda<nk_kernel_vincentys_packed_k>);
    check("clearances_packed_f32_cuda", test_terrain_cuda);

#if NUMKONG_TARGET_AMPERE
    check.section("Cross Ampere", nk_cap_ampere_k);
    check("dots_packed_f64_ampere", test_dots_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, nk_dots_packed_f64_ampere);
    check("dots_pack_f64_ampere", test_dots_pack_layout<f64_t, cuda_backend_t, nk_dots_pack_size_f64_ampere,
                                                        nk_dots_packed_shape_f64_ampere, nk_dots_pack_f64_ampere>);
    check("dots_contract_f64_ampere",
          test_dots_launch_contract<f64_t, cuda_backend_t, nk_dots_pack_size_f64_ampere, nk_dots_packed_f64_ampere,
                                    nk_dots_symmetric_f64_ampere>);
    check("dots_symmetric_f64_ampere", test_dots_symmetric<f64_t, cuda_backend_t>, nk_dots_symmetric_f64_ampere);
    check("dots_packed_f32_ampere", test_dots_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_ampere,
          nk_dots_pack_f32_ampere, nk_dots_packed_f32_ampere);
    check("dots_pack_f32_ampere", test_dots_pack_layout<f32_t, cuda_backend_t, nk_dots_pack_size_f32_ampere,
                                                        nk_dots_packed_shape_f32_ampere, nk_dots_pack_f32_ampere>);
    check("dots_contract_f32_ampere",
          test_dots_launch_contract<f32_t, cuda_backend_t, nk_dots_pack_size_f32_ampere, nk_dots_packed_f32_ampere,
                                    nk_dots_symmetric_f32_ampere>);
    check("dots_symmetric_f32_ampere", test_dots_symmetric<f32_t, cuda_backend_t>, nk_dots_symmetric_f32_ampere);
    check("dots_packed_bf16_ampere", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_ampere,
          nk_dots_pack_bf16_ampere, nk_dots_packed_bf16_ampere);
    check("dots_pack_bf16_ampere", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_ampere,
                                                         nk_dots_packed_shape_bf16_ampere, nk_dots_pack_bf16_ampere>);
    check("dots_contract_bf16_ampere",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_ampere, nk_dots_packed_bf16_ampere,
                                    nk_dots_symmetric_bf16_ampere>);
    check("dots_symmetric_bf16_ampere", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_ampere);
    check("dots_packed_f16_ampere", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_dots_packed_f16_ampere);
    check("dots_pack_f16_ampere", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_ampere,
                                                        nk_dots_packed_shape_f16_ampere, nk_dots_pack_f16_ampere>);
    check("dots_contract_f16_ampere",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_ampere, nk_dots_packed_f16_ampere,
                                    nk_dots_symmetric_f16_ampere>);
    check("dots_symmetric_f16_ampere", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_ampere);
    check("dots_packed_e5m2_ampere", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_ampere,
          nk_dots_pack_e5m2_ampere, nk_dots_packed_e5m2_ampere);
    check("dots_pack_e5m2_ampere", test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_ampere,
                                                         nk_dots_packed_shape_e5m2_ampere, nk_dots_pack_e5m2_ampere>);
    check("dots_contract_e5m2_ampere",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_ampere, nk_dots_packed_e5m2_ampere,
                                    nk_dots_symmetric_e5m2_ampere>);
    check("dots_symmetric_e5m2_ampere", test_dots_symmetric<e5m2_t, cuda_backend_t>, nk_dots_symmetric_e5m2_ampere);
    check("dots_packed_e4m3_ampere", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_ampere,
          nk_dots_pack_e4m3_ampere, nk_dots_packed_e4m3_ampere);
    check("dots_pack_e4m3_ampere", test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_ampere,
                                                         nk_dots_packed_shape_e4m3_ampere, nk_dots_pack_e4m3_ampere>);
    check("dots_contract_e4m3_ampere",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_ampere, nk_dots_packed_e4m3_ampere,
                                    nk_dots_symmetric_e4m3_ampere>);
    check("dots_symmetric_e4m3_ampere", test_dots_symmetric<e4m3_t, cuda_backend_t>, nk_dots_symmetric_e4m3_ampere);
    check("dots_packed_e3m2_ampere", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_ampere,
          nk_dots_pack_e3m2_ampere, nk_dots_packed_e3m2_ampere);
    check("dots_pack_e3m2_ampere", test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_ampere,
                                                         nk_dots_packed_shape_e3m2_ampere, nk_dots_pack_e3m2_ampere>);
    check("dots_contract_e3m2_ampere",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_ampere, nk_dots_packed_e3m2_ampere,
                                    nk_dots_symmetric_e3m2_ampere>);
    check("dots_symmetric_e3m2_ampere", test_dots_symmetric<e3m2_t, cuda_backend_t>, nk_dots_symmetric_e3m2_ampere);
    check("dots_packed_e2m3_ampere", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_ampere,
          nk_dots_pack_e2m3_ampere, nk_dots_packed_e2m3_ampere);
    check("dots_pack_e2m3_ampere", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_ampere,
                                                         nk_dots_packed_shape_e2m3_ampere, nk_dots_pack_e2m3_ampere>);
    check("dots_contract_e2m3_ampere",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_ampere, nk_dots_packed_e2m3_ampere,
                                    nk_dots_symmetric_e2m3_ampere>);
    check("dots_symmetric_e2m3_ampere", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_ampere);
    check("dots_packed_e2m1_ampere", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_ampere,
          nk_dots_pack_e2m1_ampere, nk_dots_packed_e2m1_ampere);
    check("dots_pack_e2m1_ampere", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_ampere,
                                                         nk_dots_packed_shape_e2m1_ampere, nk_dots_pack_e2m1_ampere>);
    check("dots_contract_e2m1_ampere",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_ampere, nk_dots_packed_e2m1_ampere,
                                    nk_dots_symmetric_e2m1_ampere>);
    check("dots_symmetric_e2m1_ampere", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_ampere);
    check("dots_packed_nvfp4_ampere", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_ampere,
          nk_dots_pack_nvfp4_ampere, nk_dots_packed_nvfp4_ampere);
    check("dots_pack_nvfp4_ampere",
          test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_ampere,
                                nk_dots_packed_shape_nvfp4_ampere, nk_dots_pack_nvfp4_ampere>);
    check("dots_contract_nvfp4_ampere",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_ampere,
                                    nk_dots_packed_nvfp4_ampere, nk_dots_symmetric_nvfp4_ampere>);
    check("dots_symmetric_nvfp4_ampere", test_dots_symmetric<nvfp4_t, cuda_backend_t>, nk_dots_symmetric_nvfp4_ampere);
    check("dots_packed_mxfp4_ampere", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_ampere,
          nk_dots_pack_mxfp4_ampere, nk_dots_packed_mxfp4_ampere);
    check("dots_pack_mxfp4_ampere",
          test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_ampere,
                                nk_dots_packed_shape_mxfp4_ampere, nk_dots_pack_mxfp4_ampere>);
    check("dots_contract_mxfp4_ampere",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_ampere,
                                    nk_dots_packed_mxfp4_ampere, nk_dots_symmetric_mxfp4_ampere>);
    check("dots_symmetric_mxfp4_ampere", test_dots_symmetric<mxfp4_t, cuda_backend_t>, nk_dots_symmetric_mxfp4_ampere);
    check("dots_packed_mxfp8e4m3_ampere", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_ampere, nk_dots_pack_mxfp8e4m3_ampere, nk_dots_packed_mxfp8e4m3_ampere);
    check("dots_pack_mxfp8e4m3_ampere",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_ampere,
                                nk_dots_packed_shape_mxfp8e4m3_ampere, nk_dots_pack_mxfp8e4m3_ampere>);
    check("dots_contract_mxfp8e4m3_ampere",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_ampere,
                                    nk_dots_packed_mxfp8e4m3_ampere, nk_dots_symmetric_mxfp8e4m3_ampere>);
    check("dots_symmetric_mxfp8e4m3_ampere", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_ampere);
    check("dots_packed_mxfp8e5m2_ampere", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_ampere, nk_dots_pack_mxfp8e5m2_ampere, nk_dots_packed_mxfp8e5m2_ampere);
    check("dots_pack_mxfp8e5m2_ampere",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_ampere,
                                nk_dots_packed_shape_mxfp8e5m2_ampere, nk_dots_pack_mxfp8e5m2_ampere>);
    check("dots_contract_mxfp8e5m2_ampere",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_ampere,
                                    nk_dots_packed_mxfp8e5m2_ampere, nk_dots_symmetric_mxfp8e5m2_ampere>);
    check("dots_symmetric_mxfp8e5m2_ampere", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_ampere);
    check("dots_packed_i8_ampere", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_dots_packed_i8_ampere);
    check("dots_pack_i8_ampere", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_ampere,
                                                       nk_dots_packed_shape_i8_ampere, nk_dots_pack_i8_ampere>);
    check("dots_contract_i8_ampere", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_ampere,
                                                               nk_dots_packed_i8_ampere, nk_dots_symmetric_i8_ampere>);
    check("dots_symmetric_i8_ampere", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_ampere);
    check("dots_packed_u8_ampere", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_dots_packed_u8_ampere);
    check("dots_pack_u8_ampere", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_ampere,
                                                       nk_dots_packed_shape_u8_ampere, nk_dots_pack_u8_ampere>);
    check("dots_contract_u8_ampere", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_ampere,
                                                               nk_dots_packed_u8_ampere, nk_dots_symmetric_u8_ampere>);
    check("dots_symmetric_u8_ampere", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_ampere);
    check("dots_packed_i4_ampere", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_dots_packed_i4_ampere);
    check("dots_pack_i4_ampere", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_ampere,
                                                       nk_dots_packed_shape_i4_ampere, nk_dots_pack_i4_ampere>);
    check("dots_contract_i4_ampere", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_ampere,
                                                               nk_dots_packed_i4_ampere, nk_dots_symmetric_i4_ampere>);
    check("dots_symmetric_i4_ampere", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_ampere);
    check("dots_packed_u4_ampere", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_dots_packed_u4_ampere);
    check("dots_pack_u4_ampere", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_ampere,
                                                       nk_dots_packed_shape_u4_ampere, nk_dots_pack_u4_ampere>);
    check("dots_contract_u4_ampere", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_ampere,
                                                               nk_dots_packed_u4_ampere, nk_dots_symmetric_u4_ampere>);
    check("dots_symmetric_u4_ampere", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_ampere);
    check("angulars_packed_f64_ampere", test_angulars_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, nk_angulars_packed_f64_ampere);
    check("angulars_symmetric_f64_ampere", test_angulars_symmetric<f64_t, cuda_backend_t>,
          nk_angulars_symmetric_f64_ampere);
    check("euclideans_packed_f64_ampere", test_euclideans_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, nk_euclideans_packed_f64_ampere);
    check("euclideans_symmetric_f64_ampere", test_euclideans_symmetric<f64_t, cuda_backend_t>,
          nk_euclideans_symmetric_f64_ampere);
    check("angulars_packed_f32_ampere", test_angulars_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_ampere,
          nk_dots_pack_f32_ampere, nk_angulars_packed_f32_ampere);
    check("angulars_symmetric_f32_ampere", test_angulars_symmetric<f32_t, cuda_backend_t>,
          nk_angulars_symmetric_f32_ampere);
    check("euclideans_packed_f32_ampere", test_euclideans_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_ampere,
          nk_dots_pack_f32_ampere, nk_euclideans_packed_f32_ampere);
    check("euclideans_symmetric_f32_ampere", test_euclideans_symmetric<f32_t, cuda_backend_t>,
          nk_euclideans_symmetric_f32_ampere);
    check("angulars_packed_bf16_ampere", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_ampere,
          nk_dots_pack_bf16_ampere, nk_angulars_packed_bf16_ampere);
    check("angulars_symmetric_bf16_ampere", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_ampere);
    check("euclideans_packed_bf16_ampere", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_ampere, nk_dots_pack_bf16_ampere, nk_euclideans_packed_bf16_ampere);
    check("euclideans_symmetric_bf16_ampere", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_ampere);
    check("angulars_packed_f16_ampere", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_angulars_packed_f16_ampere);
    check("angulars_symmetric_f16_ampere", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_ampere);
    check("euclideans_packed_f16_ampere", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_euclideans_packed_f16_ampere);
    check("euclideans_symmetric_f16_ampere", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_ampere);
    check("angulars_packed_e5m2_ampere", test_angulars_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_ampere,
          nk_dots_pack_e5m2_ampere, nk_angulars_packed_e5m2_ampere);
    check("angulars_symmetric_e5m2_ampere", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_ampere);
    check("euclideans_packed_e5m2_ampere", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_ampere, nk_dots_pack_e5m2_ampere, nk_euclideans_packed_e5m2_ampere);
    check("euclideans_symmetric_e5m2_ampere", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_ampere);
    check("angulars_packed_e4m3_ampere", test_angulars_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_ampere,
          nk_dots_pack_e4m3_ampere, nk_angulars_packed_e4m3_ampere);
    check("angulars_symmetric_e4m3_ampere", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_ampere);
    check("euclideans_packed_e4m3_ampere", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_ampere, nk_dots_pack_e4m3_ampere, nk_euclideans_packed_e4m3_ampere);
    check("euclideans_symmetric_e4m3_ampere", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_ampere);
    check("angulars_packed_e3m2_ampere", test_angulars_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_ampere,
          nk_dots_pack_e3m2_ampere, nk_angulars_packed_e3m2_ampere);
    check("angulars_symmetric_e3m2_ampere", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_ampere);
    check("euclideans_packed_e3m2_ampere", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_ampere, nk_dots_pack_e3m2_ampere, nk_euclideans_packed_e3m2_ampere);
    check("euclideans_symmetric_e3m2_ampere", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_ampere);
    check("angulars_packed_e2m3_ampere", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_ampere,
          nk_dots_pack_e2m3_ampere, nk_angulars_packed_e2m3_ampere);
    check("angulars_symmetric_e2m3_ampere", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_ampere);
    check("euclideans_packed_e2m3_ampere", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_ampere, nk_dots_pack_e2m3_ampere, nk_euclideans_packed_e2m3_ampere);
    check("euclideans_symmetric_e2m3_ampere", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_ampere);
    check("angulars_packed_e2m1_ampere", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_ampere,
          nk_dots_pack_e2m1_ampere, nk_angulars_packed_e2m1_ampere);
    check("angulars_symmetric_e2m1_ampere", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_ampere);
    check("euclideans_packed_e2m1_ampere", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_ampere, nk_dots_pack_e2m1_ampere, nk_euclideans_packed_e2m1_ampere);
    check("euclideans_symmetric_e2m1_ampere", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_ampere);
    check("angulars_packed_i8_ampere", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_angulars_packed_i8_ampere);
    check("angulars_symmetric_i8_ampere", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_ampere);
    check("euclideans_packed_i8_ampere", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_euclideans_packed_i8_ampere);
    check("euclideans_symmetric_i8_ampere", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_ampere);
    check("angulars_packed_u8_ampere", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_angulars_packed_u8_ampere);
    check("angulars_symmetric_u8_ampere", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_ampere);
    check("euclideans_packed_u8_ampere", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_euclideans_packed_u8_ampere);
    check("euclideans_symmetric_u8_ampere", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_ampere);
    check("angulars_packed_i4_ampere", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_angulars_packed_i4_ampere);
    check("angulars_symmetric_i4_ampere", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_ampere);
    check("euclideans_packed_i4_ampere", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_euclideans_packed_i4_ampere);
    check("euclideans_symmetric_i4_ampere", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_ampere);
    check("angulars_packed_u4_ampere", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_angulars_packed_u4_ampere);
    check("angulars_symmetric_u4_ampere", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_ampere);
    check("euclideans_packed_u4_ampere", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_euclideans_packed_u4_ampere);
    check("euclideans_symmetric_u4_ampere", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_ampere);
    check("attention_packed_bf16_ampere", test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere, nk_attention_packed_bf16_ampere);
    check("attention_packed_f16_ampere", test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_ampere, nk_attention_pack_f16_ampere, nk_attention_packed_f16_ampere);
    check("attention_packed_e4m3_ampere", test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_ampere, nk_attention_pack_e4m3_ampere, nk_attention_packed_e4m3_ampere);
    check("attention_packed_i8_ampere", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_ampere, nk_attention_pack_i8_ampere, nk_attention_packed_i8_ampere);
    check("attention_packed_gradients_bf16_ampere", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere, nk_attention_packed_gradients_bf16_ampere);
#endif // NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_HOPPER
    check.section("Cross Hopper", nk_cap_hopper_k);
    check("dots_packed_bf16_hopper", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_hopper,
          nk_dots_pack_bf16_hopper, nk_dots_packed_bf16_hopper);
    check("dots_pack_bf16_hopper", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_hopper,
                                                         nk_dots_packed_shape_bf16_hopper, nk_dots_pack_bf16_hopper>);
    check("dots_contract_bf16_hopper",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_hopper, nk_dots_packed_bf16_hopper,
                                    nk_dots_symmetric_bf16_hopper>);
    check("dots_symmetric_bf16_hopper", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_hopper);
    check("dots_packed_f16_hopper", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_hopper,
          nk_dots_pack_f16_hopper, nk_dots_packed_f16_hopper);
    check("dots_pack_f16_hopper", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_hopper,
                                                        nk_dots_packed_shape_f16_hopper, nk_dots_pack_f16_hopper>);
    check("dots_contract_f16_hopper",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_hopper, nk_dots_packed_f16_hopper,
                                    nk_dots_symmetric_f16_hopper>);
    check("dots_symmetric_f16_hopper", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_hopper);
    check("dots_packed_e2m3_hopper", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_hopper,
          nk_dots_pack_e2m3_hopper, nk_dots_packed_e2m3_hopper);
    check("dots_pack_e2m3_hopper", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_hopper,
                                                         nk_dots_packed_shape_e2m3_hopper, nk_dots_pack_e2m3_hopper>);
    check("dots_contract_e2m3_hopper",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_hopper, nk_dots_packed_e2m3_hopper,
                                    nk_dots_symmetric_e2m3_hopper>);
    check("dots_symmetric_e2m3_hopper", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_hopper);
    check("dots_packed_e2m1_hopper", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_hopper,
          nk_dots_pack_e2m1_hopper, nk_dots_packed_e2m1_hopper);
    check("dots_pack_e2m1_hopper", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_hopper,
                                                         nk_dots_packed_shape_e2m1_hopper, nk_dots_pack_e2m1_hopper>);
    check("dots_contract_e2m1_hopper",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_hopper, nk_dots_packed_e2m1_hopper,
                                    nk_dots_symmetric_e2m1_hopper>);
    check("dots_symmetric_e2m1_hopper", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_hopper);
    check("dots_packed_nvfp4_hopper", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_hopper,
          nk_dots_pack_nvfp4_hopper, nk_dots_packed_nvfp4_hopper);
    check("dots_pack_nvfp4_hopper",
          test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_hopper,
                                nk_dots_packed_shape_nvfp4_hopper, nk_dots_pack_nvfp4_hopper>);
    check("dots_contract_nvfp4_hopper",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_hopper,
                                    nk_dots_packed_nvfp4_hopper, nk_dots_symmetric_nvfp4_hopper>);
    check("dots_symmetric_nvfp4_hopper", test_dots_symmetric<nvfp4_t, cuda_backend_t>, nk_dots_symmetric_nvfp4_hopper);
    check("dots_packed_mxfp4_hopper", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_hopper,
          nk_dots_pack_mxfp4_hopper, nk_dots_packed_mxfp4_hopper);
    check("dots_pack_mxfp4_hopper",
          test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_hopper,
                                nk_dots_packed_shape_mxfp4_hopper, nk_dots_pack_mxfp4_hopper>);
    check("dots_contract_mxfp4_hopper",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_hopper,
                                    nk_dots_packed_mxfp4_hopper, nk_dots_symmetric_mxfp4_hopper>);
    check("dots_symmetric_mxfp4_hopper", test_dots_symmetric<mxfp4_t, cuda_backend_t>, nk_dots_symmetric_mxfp4_hopper);
    check("dots_packed_mxfp8e4m3_hopper", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_hopper, nk_dots_pack_mxfp8e4m3_hopper, nk_dots_packed_mxfp8e4m3_hopper);
    check("dots_pack_mxfp8e4m3_hopper",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_hopper,
                                nk_dots_packed_shape_mxfp8e4m3_hopper, nk_dots_pack_mxfp8e4m3_hopper>);
    check("dots_contract_mxfp8e4m3_hopper",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_hopper,
                                    nk_dots_packed_mxfp8e4m3_hopper, nk_dots_symmetric_mxfp8e4m3_hopper>);
    check("dots_symmetric_mxfp8e4m3_hopper", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_hopper);
    check("dots_packed_mxfp8e5m2_hopper", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_hopper, nk_dots_pack_mxfp8e5m2_hopper, nk_dots_packed_mxfp8e5m2_hopper);
    check("dots_pack_mxfp8e5m2_hopper",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_hopper,
                                nk_dots_packed_shape_mxfp8e5m2_hopper, nk_dots_pack_mxfp8e5m2_hopper>);
    check("dots_contract_mxfp8e5m2_hopper",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_hopper,
                                    nk_dots_packed_mxfp8e5m2_hopper, nk_dots_symmetric_mxfp8e5m2_hopper>);
    check("dots_symmetric_mxfp8e5m2_hopper", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_hopper);
    check("dots_packed_i8_hopper", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_hopper,
          nk_dots_pack_i8_hopper, nk_dots_packed_i8_hopper);
    check("dots_pack_i8_hopper", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_hopper,
                                                       nk_dots_packed_shape_i8_hopper, nk_dots_pack_i8_hopper>);
    check("dots_contract_i8_hopper", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_hopper,
                                                               nk_dots_packed_i8_hopper, nk_dots_symmetric_i8_hopper>);
    check("dots_symmetric_i8_hopper", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_hopper);
    check("dots_packed_u8_hopper", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_hopper,
          nk_dots_pack_u8_hopper, nk_dots_packed_u8_hopper);
    check("dots_pack_u8_hopper", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_hopper,
                                                       nk_dots_packed_shape_u8_hopper, nk_dots_pack_u8_hopper>);
    check("dots_contract_u8_hopper", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_hopper,
                                                               nk_dots_packed_u8_hopper, nk_dots_symmetric_u8_hopper>);
    check("dots_symmetric_u8_hopper", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_hopper);
    check("dots_packed_i4_hopper", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_hopper,
          nk_dots_pack_i4_hopper, nk_dots_packed_i4_hopper);
    check("dots_pack_i4_hopper", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_hopper,
                                                       nk_dots_packed_shape_i4_hopper, nk_dots_pack_i4_hopper>);
    check("dots_contract_i4_hopper", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_hopper,
                                                               nk_dots_packed_i4_hopper, nk_dots_symmetric_i4_hopper>);
    check("dots_symmetric_i4_hopper", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_hopper);
    check("dots_packed_u4_hopper", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_hopper,
          nk_dots_pack_u4_hopper, nk_dots_packed_u4_hopper);
    check("dots_pack_u4_hopper", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_hopper,
                                                       nk_dots_packed_shape_u4_hopper, nk_dots_pack_u4_hopper>);
    check("dots_contract_u4_hopper", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_hopper,
                                                               nk_dots_packed_u4_hopper, nk_dots_symmetric_u4_hopper>);
    check("dots_symmetric_u4_hopper", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_hopper);
    check("angulars_packed_bf16_hopper", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_hopper,
          nk_dots_pack_bf16_hopper, nk_angulars_packed_bf16_hopper);
    check("angulars_symmetric_bf16_hopper", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_hopper);
    check("euclideans_packed_bf16_hopper", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_hopper, nk_dots_pack_bf16_hopper, nk_euclideans_packed_bf16_hopper);
    check("euclideans_symmetric_bf16_hopper", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_hopper);
    check("angulars_packed_f16_hopper", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_hopper,
          nk_dots_pack_f16_hopper, nk_angulars_packed_f16_hopper);
    check("angulars_symmetric_f16_hopper", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_hopper);
    check("euclideans_packed_f16_hopper", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_hopper,
          nk_dots_pack_f16_hopper, nk_euclideans_packed_f16_hopper);
    check("euclideans_symmetric_f16_hopper", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_hopper);
    check("angulars_packed_e2m3_hopper", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_hopper,
          nk_dots_pack_e2m3_hopper, nk_angulars_packed_e2m3_hopper);
    check("angulars_symmetric_e2m3_hopper", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_hopper);
    check("euclideans_packed_e2m3_hopper", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_hopper, nk_dots_pack_e2m3_hopper, nk_euclideans_packed_e2m3_hopper);
    check("euclideans_symmetric_e2m3_hopper", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_hopper);
    check("angulars_packed_e2m1_hopper", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_hopper,
          nk_dots_pack_e2m1_hopper, nk_angulars_packed_e2m1_hopper);
    check("angulars_symmetric_e2m1_hopper", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_hopper);
    check("euclideans_packed_e2m1_hopper", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_hopper, nk_dots_pack_e2m1_hopper, nk_euclideans_packed_e2m1_hopper);
    check("euclideans_symmetric_e2m1_hopper", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_hopper);
    check("angulars_packed_i8_hopper", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_hopper,
          nk_dots_pack_i8_hopper, nk_angulars_packed_i8_hopper);
    check("angulars_symmetric_i8_hopper", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_hopper);
    check("euclideans_packed_i8_hopper", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_hopper,
          nk_dots_pack_i8_hopper, nk_euclideans_packed_i8_hopper);
    check("euclideans_symmetric_i8_hopper", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_hopper);
    check("angulars_packed_u8_hopper", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_hopper,
          nk_dots_pack_u8_hopper, nk_angulars_packed_u8_hopper);
    check("angulars_symmetric_u8_hopper", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_hopper);
    check("euclideans_packed_u8_hopper", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_hopper,
          nk_dots_pack_u8_hopper, nk_euclideans_packed_u8_hopper);
    check("euclideans_symmetric_u8_hopper", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_hopper);
    check("angulars_packed_i4_hopper", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_hopper,
          nk_dots_pack_i4_hopper, nk_angulars_packed_i4_hopper);
    check("angulars_symmetric_i4_hopper", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_hopper);
    check("euclideans_packed_i4_hopper", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_hopper,
          nk_dots_pack_i4_hopper, nk_euclideans_packed_i4_hopper);
    check("euclideans_symmetric_i4_hopper", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_hopper);
    check("angulars_packed_u4_hopper", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_hopper,
          nk_dots_pack_u4_hopper, nk_angulars_packed_u4_hopper);
    check("angulars_symmetric_u4_hopper", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_hopper);
    check("euclideans_packed_u4_hopper", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_hopper,
          nk_dots_pack_u4_hopper, nk_euclideans_packed_u4_hopper);
    check("euclideans_symmetric_u4_hopper", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_hopper);
    check("attention_packed_bf16_hopper", test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_hopper, nk_attention_pack_bf16_hopper, nk_attention_packed_bf16_hopper);
    check("attention_packed_e4m3_hopper", test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_hopper, nk_attention_pack_e4m3_hopper, nk_attention_packed_e4m3_hopper);
    check("attention_packed_i8_hopper", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_hopper, nk_attention_pack_i8_hopper, nk_attention_packed_i8_hopper);
    check("attention_packed_gradients_bf16_hopper", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_hopper, nk_attention_pack_bf16_hopper, nk_attention_packed_gradients_bf16_hopper);
#endif // NUMKONG_TARGET_HOPPER

#if NUMKONG_TARGET_BLACKWELL
    check.section("Cross Blackwell", nk_cap_blackwell_k);
    check("dots_packed_bf16_blackwell", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_blackwell,
          nk_dots_pack_bf16_blackwell, nk_dots_packed_bf16_blackwell);
    check("dots_pack_bf16_blackwell",
          test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_blackwell,
                                nk_dots_packed_shape_bf16_blackwell, nk_dots_pack_bf16_blackwell>);
    check("dots_contract_bf16_blackwell",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_blackwell,
                                    nk_dots_packed_bf16_blackwell, nk_dots_symmetric_bf16_blackwell>);
    check("dots_symmetric_bf16_blackwell", test_dots_symmetric<bf16_t, cuda_backend_t>,
          nk_dots_symmetric_bf16_blackwell);
    check("dots_packed_f16_blackwell", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_blackwell,
          nk_dots_pack_f16_blackwell, nk_dots_packed_f16_blackwell);
    check("dots_pack_f16_blackwell",
          test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_blackwell,
                                nk_dots_packed_shape_f16_blackwell, nk_dots_pack_f16_blackwell>);
    check("dots_contract_f16_blackwell",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_blackwell,
                                    nk_dots_packed_f16_blackwell, nk_dots_symmetric_f16_blackwell>);
    check("dots_symmetric_f16_blackwell", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_blackwell);
    check("dots_packed_e5m2_blackwell", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_blackwell,
          nk_dots_pack_e5m2_blackwell, nk_dots_packed_e5m2_blackwell);
    check("dots_pack_e5m2_blackwell",
          test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwell,
                                nk_dots_packed_shape_e5m2_blackwell, nk_dots_pack_e5m2_blackwell>);
    check("dots_contract_e5m2_blackwell",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwell,
                                    nk_dots_packed_e5m2_blackwell, nk_dots_symmetric_e5m2_blackwell>);
    check("dots_symmetric_e5m2_blackwell", test_dots_symmetric<e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_e5m2_blackwell);
    check("dots_packed_e4m3_blackwell", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_blackwell,
          nk_dots_pack_e4m3_blackwell, nk_dots_packed_e4m3_blackwell);
    check("dots_pack_e4m3_blackwell",
          test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwell,
                                nk_dots_packed_shape_e4m3_blackwell, nk_dots_pack_e4m3_blackwell>);
    check("dots_contract_e4m3_blackwell",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwell,
                                    nk_dots_packed_e4m3_blackwell, nk_dots_symmetric_e4m3_blackwell>);
    check("dots_symmetric_e4m3_blackwell", test_dots_symmetric<e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_e4m3_blackwell);
    check("dots_packed_e3m2_blackwell", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_blackwell,
          nk_dots_pack_e3m2_blackwell, nk_dots_packed_e3m2_blackwell);
    check("dots_pack_e3m2_blackwell",
          test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwell,
                                nk_dots_packed_shape_e3m2_blackwell, nk_dots_pack_e3m2_blackwell>);
    check("dots_contract_e3m2_blackwell",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwell,
                                    nk_dots_packed_e3m2_blackwell, nk_dots_symmetric_e3m2_blackwell>);
    check("dots_symmetric_e3m2_blackwell", test_dots_symmetric<e3m2_t, cuda_backend_t>,
          nk_dots_symmetric_e3m2_blackwell);
    check("dots_packed_e2m3_blackwell", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_blackwell,
          nk_dots_pack_e2m3_blackwell, nk_dots_packed_e2m3_blackwell);
    check("dots_pack_e2m3_blackwell",
          test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwell,
                                nk_dots_packed_shape_e2m3_blackwell, nk_dots_pack_e2m3_blackwell>);
    check("dots_contract_e2m3_blackwell",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwell,
                                    nk_dots_packed_e2m3_blackwell, nk_dots_symmetric_e2m3_blackwell>);
    check("dots_symmetric_e2m3_blackwell", test_dots_symmetric<e2m3_t, cuda_backend_t>,
          nk_dots_symmetric_e2m3_blackwell);
    check("dots_packed_e2m1_blackwell", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_blackwell,
          nk_dots_pack_e2m1_blackwell, nk_dots_packed_e2m1_blackwell);
    check("dots_pack_e2m1_blackwell",
          test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwell,
                                nk_dots_packed_shape_e2m1_blackwell, nk_dots_pack_e2m1_blackwell>);
    check("dots_contract_e2m1_blackwell",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwell,
                                    nk_dots_packed_e2m1_blackwell, nk_dots_symmetric_e2m1_blackwell>);
    check("dots_symmetric_e2m1_blackwell", test_dots_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_dots_symmetric_e2m1_blackwell);
    check("dots_packed_nvfp4_blackwell", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_blackwell,
          nk_dots_pack_nvfp4_blackwell, nk_dots_packed_nvfp4_blackwell);
    check("dots_pack_nvfp4_blackwell",
          test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_blackwell,
                                nk_dots_packed_shape_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell>);
    check("dots_contract_nvfp4_blackwell",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_blackwell,
                                    nk_dots_packed_nvfp4_blackwell, nk_dots_symmetric_nvfp4_blackwell>);
    check("dots_symmetric_nvfp4_blackwell", test_dots_symmetric<nvfp4_t, cuda_backend_t>,
          nk_dots_symmetric_nvfp4_blackwell);
    check("dots_packed_mxfp4_blackwell", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_blackwell,
          nk_dots_pack_mxfp4_blackwell, nk_dots_packed_mxfp4_blackwell);
    check("dots_pack_mxfp4_blackwell",
          test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_blackwell,
                                nk_dots_packed_shape_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell>);
    check("dots_contract_mxfp4_blackwell",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_blackwell,
                                    nk_dots_packed_mxfp4_blackwell, nk_dots_symmetric_mxfp4_blackwell>);
    check("dots_symmetric_mxfp4_blackwell", test_dots_symmetric<mxfp4_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp4_blackwell);
    check("dots_packed_mxfp8e4m3_blackwell", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell, nk_dots_packed_mxfp8e4m3_blackwell);
    check("dots_pack_mxfp8e4m3_blackwell",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_blackwell,
                                nk_dots_packed_shape_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell>);
    check("dots_contract_mxfp8e4m3_blackwell",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_blackwell,
                                    nk_dots_packed_mxfp8e4m3_blackwell, nk_dots_symmetric_mxfp8e4m3_blackwell>);
    check("dots_symmetric_mxfp8e4m3_blackwell", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_blackwell);
    check("dots_packed_mxfp8e5m2_blackwell", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell, nk_dots_packed_mxfp8e5m2_blackwell);
    check("dots_pack_mxfp8e5m2_blackwell",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_blackwell,
                                nk_dots_packed_shape_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell>);
    check("dots_contract_mxfp8e5m2_blackwell",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_blackwell,
                                    nk_dots_packed_mxfp8e5m2_blackwell, nk_dots_symmetric_mxfp8e5m2_blackwell>);
    check("dots_symmetric_mxfp8e5m2_blackwell", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_blackwell);
    check("angulars_packed_bf16_blackwell", test_angulars_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_blackwell, nk_dots_pack_bf16_blackwell, nk_angulars_packed_bf16_blackwell);
    check("angulars_symmetric_bf16_blackwell", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_blackwell);
    check("euclideans_packed_bf16_blackwell", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_blackwell, nk_dots_pack_bf16_blackwell, nk_euclideans_packed_bf16_blackwell);
    check("euclideans_symmetric_bf16_blackwell", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_blackwell);
    check("angulars_packed_f16_blackwell", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_blackwell,
          nk_dots_pack_f16_blackwell, nk_angulars_packed_f16_blackwell);
    check("angulars_symmetric_f16_blackwell", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_blackwell);
    check("euclideans_packed_f16_blackwell", test_euclideans_packed<f16_t, cuda_backend_t>,
          nk_dots_pack_size_f16_blackwell, nk_dots_pack_f16_blackwell, nk_euclideans_packed_f16_blackwell);
    check("euclideans_symmetric_f16_blackwell", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_blackwell);
    check("angulars_packed_e5m2_blackwell", test_angulars_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwell, nk_dots_pack_e5m2_blackwell, nk_angulars_packed_e5m2_blackwell);
    check("angulars_symmetric_e5m2_blackwell", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_blackwell);
    check("euclideans_packed_e5m2_blackwell", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwell, nk_dots_pack_e5m2_blackwell, nk_euclideans_packed_e5m2_blackwell);
    check("euclideans_symmetric_e5m2_blackwell", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_blackwell);
    check("angulars_packed_e4m3_blackwell", test_angulars_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell, nk_angulars_packed_e4m3_blackwell);
    check("angulars_symmetric_e4m3_blackwell", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_blackwell);
    check("euclideans_packed_e4m3_blackwell", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell, nk_euclideans_packed_e4m3_blackwell);
    check("euclideans_symmetric_e4m3_blackwell", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_blackwell);
    check("angulars_packed_e3m2_blackwell", test_angulars_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwell, nk_dots_pack_e3m2_blackwell, nk_angulars_packed_e3m2_blackwell);
    check("angulars_symmetric_e3m2_blackwell", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_blackwell);
    check("euclideans_packed_e3m2_blackwell", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwell, nk_dots_pack_e3m2_blackwell, nk_euclideans_packed_e3m2_blackwell);
    check("euclideans_symmetric_e3m2_blackwell", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_blackwell);
    check("angulars_packed_e2m3_blackwell", test_angulars_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwell, nk_dots_pack_e2m3_blackwell, nk_angulars_packed_e2m3_blackwell);
    check("angulars_symmetric_e2m3_blackwell", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_blackwell);
    check("euclideans_packed_e2m3_blackwell", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwell, nk_dots_pack_e2m3_blackwell, nk_euclideans_packed_e2m3_blackwell);
    check("euclideans_symmetric_e2m3_blackwell", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_blackwell);
    check("angulars_packed_e2m1_blackwell", test_angulars_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwell, nk_dots_pack_e2m1_blackwell, nk_angulars_packed_e2m1_blackwell);
    check("angulars_symmetric_e2m1_blackwell", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_blackwell);
    check("euclideans_packed_e2m1_blackwell", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwell, nk_dots_pack_e2m1_blackwell, nk_euclideans_packed_e2m1_blackwell);
    check("euclideans_symmetric_e2m1_blackwell", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_blackwell);
    check("angulars_packed_nvfp4_blackwell", test_angulars_packed<nvfp4_t, cuda_backend_t>,
          nk_dots_pack_size_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell, nk_angulars_packed_nvfp4_blackwell);
    check("angulars_symmetric_nvfp4_blackwell", test_angulars_symmetric<nvfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_nvfp4_blackwell);
    check("euclideans_packed_nvfp4_blackwell", test_euclideans_packed<nvfp4_t, cuda_backend_t>,
          nk_dots_pack_size_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell, nk_euclideans_packed_nvfp4_blackwell);
    check("euclideans_symmetric_nvfp4_blackwell", test_euclideans_symmetric<nvfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_nvfp4_blackwell);
    check("angulars_packed_mxfp4_blackwell", test_angulars_packed<mxfp4_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell, nk_angulars_packed_mxfp4_blackwell);
    check("angulars_symmetric_mxfp4_blackwell", test_angulars_symmetric<mxfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp4_blackwell);
    check("euclideans_packed_mxfp4_blackwell", test_euclideans_packed<mxfp4_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell, nk_euclideans_packed_mxfp4_blackwell);
    check("euclideans_symmetric_mxfp4_blackwell", test_euclideans_symmetric<mxfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp4_blackwell);
    check("angulars_packed_mxfp8e4m3_blackwell", test_angulars_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell,
          nk_angulars_packed_mxfp8e4m3_blackwell);
    check("angulars_symmetric_mxfp8e4m3_blackwell", test_angulars_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e4m3_blackwell);
    check("euclideans_packed_mxfp8e4m3_blackwell", test_euclideans_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell,
          nk_euclideans_packed_mxfp8e4m3_blackwell);
    check("euclideans_symmetric_mxfp8e4m3_blackwell", test_euclideans_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e4m3_blackwell);
    check("angulars_packed_mxfp8e5m2_blackwell", test_angulars_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell,
          nk_angulars_packed_mxfp8e5m2_blackwell);
    check("angulars_symmetric_mxfp8e5m2_blackwell", test_angulars_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e5m2_blackwell);
    check("euclideans_packed_mxfp8e5m2_blackwell", test_euclideans_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell,
          nk_euclideans_packed_mxfp8e5m2_blackwell);
    check("euclideans_symmetric_mxfp8e5m2_blackwell", test_euclideans_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e5m2_blackwell);
    check("dots_packed_i8_blackwell", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_blackwell,
          nk_dots_pack_i8_blackwell, nk_dots_packed_i8_blackwell);
    check("dots_pack_i8_blackwell",
          test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_blackwell, nk_dots_packed_shape_i8_blackwell,
                                nk_dots_pack_i8_blackwell>);
    check("dots_contract_i8_blackwell",
          test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_blackwell, nk_dots_packed_i8_blackwell,
                                    nk_dots_symmetric_i8_blackwell>);
    check("dots_symmetric_i8_blackwell", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_blackwell);
    check("dots_packed_u8_blackwell", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_blackwell,
          nk_dots_pack_u8_blackwell, nk_dots_packed_u8_blackwell);
    check("dots_pack_u8_blackwell",
          test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_blackwell, nk_dots_packed_shape_u8_blackwell,
                                nk_dots_pack_u8_blackwell>);
    check("dots_contract_u8_blackwell",
          test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_blackwell, nk_dots_packed_u8_blackwell,
                                    nk_dots_symmetric_u8_blackwell>);
    check("dots_symmetric_u8_blackwell", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_blackwell);
    check("dots_packed_i4_blackwell", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_blackwell,
          nk_dots_pack_i4_blackwell, nk_dots_packed_i4_blackwell);
    check("dots_pack_i4_blackwell",
          test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_blackwell,
                                nk_dots_packed_shape_i4_blackwell, nk_dots_pack_i4_blackwell>);
    check("dots_contract_i4_blackwell",
          test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_blackwell, nk_dots_packed_i4_blackwell,
                                    nk_dots_symmetric_i4_blackwell>);
    check("dots_symmetric_i4_blackwell", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_blackwell);
    check("dots_packed_u4_blackwell", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_blackwell,
          nk_dots_pack_u4_blackwell, nk_dots_packed_u4_blackwell);
    check("dots_pack_u4_blackwell",
          test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_blackwell,
                                nk_dots_packed_shape_u4_blackwell, nk_dots_pack_u4_blackwell>);
    check("dots_contract_u4_blackwell",
          test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_blackwell, nk_dots_packed_u4_blackwell,
                                    nk_dots_symmetric_u4_blackwell>);
    check("dots_symmetric_u4_blackwell", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_blackwell);
    check("angulars_packed_i8_blackwell", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_blackwell,
          nk_dots_pack_i8_blackwell, nk_angulars_packed_i8_blackwell);
    check("angulars_symmetric_i8_blackwell", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_blackwell);
    check("euclideans_packed_i8_blackwell", test_euclideans_packed<i8_t, cuda_backend_t>,
          nk_dots_pack_size_i8_blackwell, nk_dots_pack_i8_blackwell, nk_euclideans_packed_i8_blackwell);
    check("euclideans_symmetric_i8_blackwell", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_blackwell);
    check("angulars_packed_u8_blackwell", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_blackwell,
          nk_dots_pack_u8_blackwell, nk_angulars_packed_u8_blackwell);
    check("angulars_symmetric_u8_blackwell", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_blackwell);
    check("euclideans_packed_u8_blackwell", test_euclideans_packed<u8_t, cuda_backend_t>,
          nk_dots_pack_size_u8_blackwell, nk_dots_pack_u8_blackwell, nk_euclideans_packed_u8_blackwell);
    check("euclideans_symmetric_u8_blackwell", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_blackwell);
    check("angulars_packed_i4_blackwell", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_blackwell,
          nk_dots_pack_i4_blackwell, nk_angulars_packed_i4_blackwell);
    check("angulars_symmetric_i4_blackwell", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_blackwell);
    check("euclideans_packed_i4_blackwell", test_euclideans_packed<i4x2_t, cuda_backend_t>,
          nk_dots_pack_size_i4_blackwell, nk_dots_pack_i4_blackwell, nk_euclideans_packed_i4_blackwell);
    check("euclideans_symmetric_i4_blackwell", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_blackwell);
    check("angulars_packed_u4_blackwell", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_blackwell,
          nk_dots_pack_u4_blackwell, nk_angulars_packed_u4_blackwell);
    check("angulars_symmetric_u4_blackwell", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_blackwell);
    check("euclideans_packed_u4_blackwell", test_euclideans_packed<u4x2_t, cuda_backend_t>,
          nk_dots_pack_size_u4_blackwell, nk_dots_pack_u4_blackwell, nk_euclideans_packed_u4_blackwell);
    check("euclideans_symmetric_u4_blackwell", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_blackwell);
    check("attention_packed_bf16_blackwell",
          test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_blackwell, nk_attention_pack_bf16_blackwell, nk_attention_packed_bf16_blackwell);
    check("attention_packed_f16_blackwell",
          test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_blackwell, nk_attention_pack_f16_blackwell, nk_attention_packed_f16_blackwell);
    check("attention_packed_e4m3_blackwell",
          test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwell, nk_attention_pack_e4m3_blackwell, nk_attention_packed_e4m3_blackwell);
    check("attention_packed_i8_blackwell", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_blackwell, nk_attention_pack_i8_blackwell, nk_attention_packed_i8_blackwell);
    check("attention_packed_gradients_bf16_blackwell", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_blackwell, nk_attention_pack_bf16_blackwell,
          nk_attention_packed_gradients_bf16_blackwell);
    check("rmsds_packed_f16_blackwell", test_rmsds_packed_f16_blackwell);
    check("chamfers_packed_f16_blackwell", test_shapes_flips_f16_blackwell<nk_kernel_chamfers_packed_k>);
    check("hausdorffs_packed_f16_blackwell", test_shapes_flips_f16_blackwell<nk_kernel_hausdorffs_packed_k>);
#endif // NUMKONG_TARGET_BLACKWELL

#if NUMKONG_TARGET_BLACKWELLRTX
    check.section("Cross Blackwell RTX", nk_cap_blackwellrtx_k);
    check("dots_packed_e5m2_blackwellrtx", test_dots_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_dots_packed_e5m2_blackwellrtx);
    check("dots_pack_e5m2_blackwellrtx",
          test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwellrtx,
                                nk_dots_packed_shape_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx>);
    check("dots_contract_e5m2_blackwellrtx",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwellrtx,
                                    nk_dots_packed_e5m2_blackwellrtx, nk_dots_symmetric_e5m2_blackwellrtx>);
    check("dots_symmetric_e5m2_blackwellrtx", test_dots_symmetric<e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_e5m2_blackwellrtx);
    check("dots_packed_e4m3_blackwellrtx", test_dots_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_dots_packed_e4m3_blackwellrtx);
    check("dots_pack_e4m3_blackwellrtx",
          test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwellrtx,
                                nk_dots_packed_shape_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx>);
    check("dots_contract_e4m3_blackwellrtx",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwellrtx,
                                    nk_dots_packed_e4m3_blackwellrtx, nk_dots_symmetric_e4m3_blackwellrtx>);
    check("dots_symmetric_e4m3_blackwellrtx", test_dots_symmetric<e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_e4m3_blackwellrtx);
    check("dots_packed_e3m2_blackwellrtx", test_dots_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_dots_packed_e3m2_blackwellrtx);
    check("dots_pack_e3m2_blackwellrtx",
          test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwellrtx,
                                nk_dots_packed_shape_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx>);
    check("dots_contract_e3m2_blackwellrtx",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwellrtx,
                                    nk_dots_packed_e3m2_blackwellrtx, nk_dots_symmetric_e3m2_blackwellrtx>);
    check("dots_symmetric_e3m2_blackwellrtx", test_dots_symmetric<e3m2_t, cuda_backend_t>,
          nk_dots_symmetric_e3m2_blackwellrtx);
    check("dots_packed_e2m3_blackwellrtx", test_dots_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_dots_packed_e2m3_blackwellrtx);
    check("dots_pack_e2m3_blackwellrtx",
          test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwellrtx,
                                nk_dots_packed_shape_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx>);
    check("dots_contract_e2m3_blackwellrtx",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwellrtx,
                                    nk_dots_packed_e2m3_blackwellrtx, nk_dots_symmetric_e2m3_blackwellrtx>);
    check("dots_symmetric_e2m3_blackwellrtx", test_dots_symmetric<e2m3_t, cuda_backend_t>,
          nk_dots_symmetric_e2m3_blackwellrtx);
    check("dots_packed_e2m1_blackwellrtx", test_dots_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_dots_packed_e2m1_blackwellrtx);
    check("dots_pack_e2m1_blackwellrtx",
          test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwellrtx,
                                nk_dots_packed_shape_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx>);
    check("dots_contract_e2m1_blackwellrtx",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwellrtx,
                                    nk_dots_packed_e2m1_blackwellrtx, nk_dots_symmetric_e2m1_blackwellrtx>);
    check("dots_symmetric_e2m1_blackwellrtx", test_dots_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_dots_symmetric_e2m1_blackwellrtx);
    check("angulars_packed_e5m2_blackwellrtx", test_angulars_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_angulars_packed_e5m2_blackwellrtx);
    check("angulars_symmetric_e5m2_blackwellrtx", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_blackwellrtx);
    check("euclideans_packed_e5m2_blackwellrtx", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_euclideans_packed_e5m2_blackwellrtx);
    check("euclideans_symmetric_e5m2_blackwellrtx", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_blackwellrtx);
    check("angulars_packed_e4m3_blackwellrtx", test_angulars_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_angulars_packed_e4m3_blackwellrtx);
    check("angulars_symmetric_e4m3_blackwellrtx", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_blackwellrtx);
    check("euclideans_packed_e4m3_blackwellrtx", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_euclideans_packed_e4m3_blackwellrtx);
    check("euclideans_symmetric_e4m3_blackwellrtx", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_blackwellrtx);
    check("angulars_packed_e3m2_blackwellrtx", test_angulars_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_angulars_packed_e3m2_blackwellrtx);
    check("angulars_symmetric_e3m2_blackwellrtx", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_blackwellrtx);
    check("euclideans_packed_e3m2_blackwellrtx", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_euclideans_packed_e3m2_blackwellrtx);
    check("euclideans_symmetric_e3m2_blackwellrtx", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_blackwellrtx);
    check("angulars_packed_e2m3_blackwellrtx", test_angulars_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_angulars_packed_e2m3_blackwellrtx);
    check("angulars_symmetric_e2m3_blackwellrtx", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_blackwellrtx);
    check("euclideans_packed_e2m3_blackwellrtx", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_euclideans_packed_e2m3_blackwellrtx);
    check("euclideans_symmetric_e2m3_blackwellrtx", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_blackwellrtx);
    check("angulars_packed_e2m1_blackwellrtx", test_angulars_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_angulars_packed_e2m1_blackwellrtx);
    check("angulars_symmetric_e2m1_blackwellrtx", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_blackwellrtx);
    check("euclideans_packed_e2m1_blackwellrtx", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_euclideans_packed_e2m1_blackwellrtx);
    check("euclideans_symmetric_e2m1_blackwellrtx", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_blackwellrtx);
    check("attention_packed_e4m3_blackwellrtx",
          test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwellrtx, nk_attention_pack_e4m3_blackwellrtx,
          nk_attention_packed_e4m3_blackwellrtx);
#endif // NUMKONG_TARGET_BLACKWELLRTX

#if NUMKONG_TARGET_BLACKWELLULTRA
    check.section("Cross Blackwell Ultra", nk_cap_blackwellultra_k);
    check("attention_packed_bf16_blackwellultra",
          test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_blackwellultra, nk_attention_pack_bf16_blackwellultra,
          nk_attention_packed_bf16_blackwellultra);
    check("attention_packed_f16_blackwellultra",
          test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_blackwellultra, nk_attention_pack_f16_blackwellultra,
          nk_attention_packed_f16_blackwellultra);
    check("attention_packed_e4m3_blackwellultra",
          test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwellultra, nk_attention_pack_e4m3_blackwellultra,
          nk_attention_packed_e4m3_blackwellultra);
    check("attention_packed_i8_blackwellultra",
          test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_blackwellultra, nk_attention_pack_i8_blackwellultra,
          nk_attention_packed_i8_blackwellultra);
    check("attention_packed_gradients_bf16_blackwellultra", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_blackwellultra, nk_attention_pack_bf16_blackwellultra,
          nk_attention_packed_gradients_bf16_blackwellultra);
#endif // NUMKONG_TARGET_BLACKWELLULTRA

    test_cross_dispatch<cuda_backend_t>(check);
}

#else  // !NUMKONG_ARCH_CUDA_
void test_cross_cuda(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::test
