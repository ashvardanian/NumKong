/**
 *  @file test/cross_device.hpp
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Shared dispatch checks for CUDA, ROCm, and Metal backends.
 */
#pragma once
#ifndef NUMKONG_TEST_CROSS_DEVICE_HPP
#define NUMKONG_TEST_CROSS_DEVICE_HPP

#include <numeric> // `std::accumulate`
#include <span>    // `std::span`

#include "harness.hpp"             // `device_backend`
#include "cross.hpp"               // `test_dots_packed`, `attention_weights_t`
#include "numkong/shapes/serial.h" // `nk_shapes_packed_header_f32_t`
#include "numkong/shapes.hpp"      // `nk::shapes_packed`, `nk::shapes_transport_epsilon_k`

namespace ashvardanian::numkong::test {

/** Clouds over every branch of the device shape bodies: no points, one, fewer and more than a warp
 *  of them, each random, coincident, collinear, coplanar or with every point doubled, plus a
 *  mirrored copy of every random cloud, which lies a rounding away from it under one flip.
 *  Coordinates follow the settings' distribution, clamped to a quarter of the range so that every
 *  pack stays finite. */
template <typename scalar_type_, typename backend_type_>
auto device_shapes_packed(settings_t const &settings, backend_type_ const &backend, std::mt19937 &generator,
                          error_stats_t &stats) {
    std::size_t const warp = 32, points = settings.shape_points;
    std::size_t const sizes[] = {0, 1, 2, 3, warp - 1, warp, warp + 1, points, 2 * points + 3};
    nk_f64_t const limit = static_cast<nk_f64_t>(scalar_type_::finite_max()) / 4;
    constexpr std::size_t shapes = 5, clouds_count = std::size(sizes) * (shapes + 1);
    auto coordinates = make_vector<scalar_type_>(std::accumulate(std::begin(sizes), std::end(sizes), std::size_t(0)) *
                                                 3 * (shapes + 1));
    fill_random(settings, generator, coordinates);
    std::array<std::span<scalar_type_ const>, clouds_count> clouds;
    scalar_type_ *values = coordinates.values_data();
    std::size_t next_cloud = 0;
    for (std::size_t size : sizes)
        for (std::size_t shape = 0; shape != shapes; ++shape) {
            for (std::size_t index = 0; index != size * 3; ++index)
                values[index] = scalar_type_(std::clamp(static_cast<nk_f64_t>(values[index]), -limit, limit));
            for (std::size_t point = 0; point != size; ++point) {
                scalar_type_ *point_coordinates = values + point * 3;
                if (shape == 1)
                    point_coordinates[0] = values[0], point_coordinates[1] = values[1],
                    point_coordinates[2] = values[2];
                if (shape == 2)
                    point_coordinates[1] = scalar_type_(static_cast<nk_f64_t>(point_coordinates[0]) / 2),
                    point_coordinates[2] = scalar_type_(static_cast<nk_f64_t>(point_coordinates[0]) / -4);
                if (shape == 3) point_coordinates[2] = scalar_type_(0.0);
                if (shape == 4 && point % 2)
                    point_coordinates[0] = point_coordinates[-3], point_coordinates[1] = point_coordinates[-2],
                    point_coordinates[2] = point_coordinates[-1];
            }
            clouds[next_cloud++] = {values, size * 3};
            scalar_type_ const *original = values;
            values += size * 3;
            if (shape == 0) {
                for (std::size_t point = 0; point != size; ++point) {
                    scalar_type_ const *point_coordinates = original + point * 3;
                    scalar_type_ *image = values + point * 3;
                    image[0] = point_coordinates[1], image[1] = point_coordinates[0], image[2] = point_coordinates[2];
                }
                clouds[next_cloud++] = {values, size * 3};
                values += size * 3;
            }
        }
    using packs_t = nk::shapes_packed<scalar_type_, typename backend_type_::template allocator<char>>;
    auto packs = packs_t::make(clouds, allocator_of<char>(backend), nk_cap_serial_k);
    stats.expect(packs.status);
    return std::move(packs.value);
}

/** @p count random clouds of @p scalar_type_ of @p points each, packed serially into @p backend
 *  memory. */
template <typename scalar_type_, typename backend_type_>
auto device_random_packed(settings_t const &settings, backend_type_ const &backend, std::mt19937 &generator,
                          std::size_t count, std::size_t points, error_stats_t &stats) {
    auto coordinates = make_vector<scalar_type_>(count * points * 3);
    fill_random(settings, generator, coordinates);
    std::vector<std::span<scalar_type_ const>> clouds(count);
    for (std::size_t index = 0; index != count; ++index)
        clouds[index] = {coordinates.values_data() + index * points * 3, points * 3};
    using packs_t = nk::shapes_packed<scalar_type_, typename backend_type_::template allocator<char>>;
    auto packs = packs_t::make(clouds, allocator_of<char>(backend), nk_cap_serial_k);
    stats.expect(packs.status);
    return std::move(packs.value);
}

/** Superposed RMSD on the device against the serial kernel, whose fold it repeats over products the
 *  matrix units sum in another order: n roundings of the moments in the distance type, n × ε ×
 *  spread², through the square root. */
template <typename scalar_type_, typename backend_type_, typename kernel_type_>
error_stats_t test_rmsds_packed(settings_t const &settings, kernel_type_ kernel) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<backend_type_>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 37, columns = 45;
    auto const row_packs = device_random_packed<scalar_type_>(settings, backend, generator, rows, points, stats);
    auto const column_packs = device_random_packed<scalar_type_>(settings, backend, generator, columns, points, stats);
    auto results = make_vector<distance_t>(backend, rows * columns);
    backend.call(kernel, row_packs.data(), rows, column_packs.data(), columns, results.raw_values_data(),
                 columns * sizeof(raw_distance_t));
    if (!stats.expect(backend.synchronize())) return stats;
    std::vector<raw_distance_t> expected(rows * columns);
    auto const serial = [] {
        if constexpr (std::is_same_v<scalar_type_, f16_t>) return nk_rmsds_packed_f16_serial;
        else if constexpr (std::is_same_v<scalar_type_, f32_t>) return nk_rmsds_packed_f32_serial;
        else return nk_rmsds_packed_f64_serial;
    }();
    stats.expect(serial(row_packs.data(), rows, column_packs.data(), columns, expected.data(),
                        columns * sizeof(raw_distance_t), nullptr));
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            auto const radius = [](void const *packed) {
                if constexpr (std::is_same_v<scalar_type_, f16_t>) {
                    nk_f32_t const *spreads = ((nk_shapes_packed_header_f32_t const *)packed)->spreads;
                    return std::sqrt(static_cast<nk_f64_t>(spreads[0]) + spreads[1] + spreads[2]);
                }
                else {
                    nk_f64_t const *spreads = ((nk_shapes_packed_header_f64_t const *)packed)->spreads;
                    return std::sqrt(spreads[0] + spreads[1] + spreads[2]);
                }
            };
            nk_f64_t const spread = radius(row_packs[row]) + radius(column_packs[column]);
            nk_f64_t const reference = expected[row * columns + column];
            nk_f64_t const squares = points * nk_accumulation_error_bound(distance_t::dtype()) * spread * spread;
            nk_f64_t const bound = reference > 0 ? nk::min_of(std::sqrt(squares), squares / reference)
                                                 : std::sqrt(squares);
            stats.accumulate_bounded(distance_t::from_raw(results.raw_values_data()[row * columns + column]),
                                     f64_t(reference), bound);
        }
    return stats;
}

/**
 *  @brief Every pair of the clouds above, scored by a backend's @p pairs_kernel over the device
 *      bodies with @p lanes_ lanes each, against the serial kernel.
 *
 *  Unbounded calls must match within the CPU capabilities' bounds: n + m + 8 roundings of a Chamfer
 *  distance, whose sums the lanes reduce in another order, and 8 of a Hausdorff distance, which
 *  differs only by the hardware square root. Bounded calls draw each pair's limit up to twice its
 *  distance, and a result past the limit must still be a lower bound of the distance, up to the 3
 *  roundings of each pack radius the Hausdorff header bound subtracts. Every lane of a pair must
 *  return the same value as its first.
 */
template <typename scalar_type_, nk_kernel_kind_t kind_, std::size_t lanes_, typename backend_type_,
          typename pairs_kernel_type_>
error_stats_t test_shapes_device(settings_t const &settings, pairs_kernel_type_ pairs_kernel) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    constexpr bool is_f32 = std::is_same_v<scalar_type_, f32_t>;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<backend_type_>(settings);
    std::mt19937 generator(settings.seed.value);
    auto const packs = device_shapes_packed<scalar_type_>(settings, backend, generator, stats);
    std::size_t const count = packs.size(), pairs = count * count;

    auto pair_packs = make_vector<void const *>(backend, 2 * pairs);
    void const **a_packed = pair_packs.values_data();
    void const **b_packed = a_packed + pairs;
    for (std::size_t pair = 0; pair != pairs; ++pair)
        a_packed[pair] = packs[pair / count], b_packed[pair] = packs[pair % count];

    std::vector<raw_distance_t> references(pairs);
    raw_distance_t const infinity = distance_t::positive_infinity().raw_;
    auto const serial = [&](auto chamfers, auto hausdorffs) {
        if constexpr (kind_ == nk_kernel_chamfers_packed_k)
            return chamfers(packs.data(), count, packs.data(), count, references.data(), count * sizeof(raw_distance_t),
                            nullptr, &infinity, nullptr, 0, nullptr);
        else
            return hausdorffs(packs.data(), count, packs.data(), count, references.data(),
                              count * sizeof(raw_distance_t), nullptr, &infinity, nullptr);
    };
    if constexpr (is_f32) stats.expect(serial(nk_chamfers_packed_f32_serial, nk_hausdorffs_packed_f32_serial));
    else stats.expect(serial(nk_chamfers_packed_f16_serial, nk_hausdorffs_packed_f16_serial));

    auto limits = make_vector<distance_t>(backend, pairs);
    auto results = make_vector<distance_t>(backend, pairs * lanes_);
    std::uniform_real_distribution<nk_f64_t> fraction(0.0, 2.0);
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0);
    auto const radius = [](void const *packed) {
        if constexpr (is_f32) return static_cast<nk_f64_t>(((nk_shapes_packed_header_f64_t const *)packed)->max_radius);
        else return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
    };
    for (int bounded = 0; bounded != 2; ++bounded) {
        for (std::size_t pair = 0; pair != pairs; ++pair)
            limits.raw_values_data()[pair] = bounded
                                                 ? static_cast<raw_distance_t>(references[pair] * fraction(generator))
                                                 : std::numeric_limits<raw_distance_t>::infinity();
        backend.call(pairs_kernel, static_cast<void const *const *>(a_packed),
                     static_cast<void const *const *>(b_packed), nk_size_t(pairs),
                     static_cast<raw_distance_t const *>(limits.raw_values_data()), results.raw_values_data());
        if (!stats.expect(backend.synchronize())) return stats;
        for (std::size_t pair = 0; pair != pairs; ++pair) {
            raw_distance_t const *lane_results = results.raw_values_data() + pair * lanes_;
            for (std::size_t lane = 1; lane != lanes_; ++lane)
                stats.expect(std::memcmp(lane_results + lane, lane_results, sizeof(raw_distance_t)) == 0,
                             "the lanes of a pair disagree");
            std::size_t const a_points = packs.points(pair / count);
            std::size_t const b_points = packs.points(pair % count);
            nk_f64_t const reference = references[pair], result = lane_results[0];
            nk_f64_t const roundings = kind_ == nk_kernel_chamfers_packed_k ? a_points + b_points + 8 : 8;
            nk_f64_t const bound = roundings * roundoff * reference;
            if (result <= limits.raw_values_data()[pair] || !std::isfinite(reference))
                stats.accumulate_bounded(distance_t::from_raw(lane_results[0]), f64_t(reference), bound);
            else
                stats.expect(result <= reference + bound + half_ulp<distance_t>(reference) +
                                           3 * roundoff * (radius(a_packed[pair]) + radius(b_packed[pair])),
                             "a result past its limit exceeds the distance");
        }
    }
    return stats;
}

/** Whether @p rotation maps the frame of @p a_packed onto that of @p b_packed, packs of
 *  @p scalar_type_ clouds, up to one sign per axis: one of the pair's own flips. */
template <typename scalar_type_>
bool shapes_rotation_is_flip(nk_f32_t const *rotation, void const *a_packed, void const *b_packed) {
    nk_f64_t a_axes[9], b_axes[9];
    if constexpr (std::is_same_v<scalar_type_, f16_t>) {
        nk_f32_t a_frame[9], b_frame[9];
        nk_quaternion_to_mat3x3_f32_serial_(((nk_shapes_packed_header_f32_t const *)a_packed)->frame, a_frame);
        nk_quaternion_to_mat3x3_f32_serial_(((nk_shapes_packed_header_f32_t const *)b_packed)->frame, b_frame);
        std::copy_n(a_frame, 9, a_axes), std::copy_n(b_frame, 9, b_axes);
    }
    else {
        nk_quaternion_to_mat3x3_f64_serial_(((nk_shapes_packed_header_f64_t const *)a_packed)->frame, a_axes);
        nk_quaternion_to_mat3x3_f64_serial_(((nk_shapes_packed_header_f64_t const *)b_packed)->frame, b_axes);
    }
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t column = 0; column != 3; ++column) {
            nk_f64_t sign = 0;
            for (std::size_t left = 0; left != 3; ++left)
                for (std::size_t right = 0; right != 3; ++right)
                    sign += a_axes[row * 3 + left] * rotation[left * 3 + right] * b_axes[column * 3 + right];
            if (std::fabs(std::fabs(sign) - (row == column)) > 1e-4) return false;
        }
    return true;
}

/** Chamfer or Hausdorff and their rotations on the device against the serial kernel, then at
 *  limits below, at and above each exact distance: within it a limit keeps the bits and the
 *  rotation, past it a pair reports a lower bound past the limit with one of its own flips.
 *  Matrix-unit kernels expand ‖a − b‖² into ‖a‖² + ‖b‖² − 2 × a · b only to search, each norm
 *  split into two halves, and recompute every nearest square in the serial order, so a distance
 *  moves by the root of a few roundings of the norms in the distance type, and a rotation, built
 *  from the same frames, by a few F32 roundings of each of its entries. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename backend_type_, typename kernel_type_>
error_stats_t test_shapes_flips(settings_t const &settings, kernel_type_ kernel) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    constexpr bool narrow_k = std::is_same_v<scalar_type_, f16_t>;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<backend_type_>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 7, columns = 9;
    auto results = make_vector<distance_t>(backend, rows * columns);
    auto rotations = make_vector<f32_t>(backend, rows * columns * 9);
    raw_distance_t *score = results.raw_values_data();
    nk_f32_t *rotation = rotations.raw_values_data();
    // Calls a Chamfer or a Hausdorff kernel, of the kind under test, on one block of pairs.
    auto const call = [&](auto chamfers, auto hausdorffs, void const *const *a, std::size_t row_count,
                          void const *const *b, std::size_t column_count, raw_distance_t *distances, nk_f32_t *turns,
                          raw_distance_t const *limit) {
        if constexpr (kind_ == nk_kernel_chamfers_packed_k)
            return chamfers(a, nk_size_t(row_count), b, nk_size_t(column_count), distances,
                            column_count * sizeof(raw_distance_t), turns, limit, static_cast<void *>(nullptr),
                            nk_size_t(0));
        else
            return hausdorffs(a, nk_size_t(row_count), b, nk_size_t(column_count), distances,
                              column_count * sizeof(raw_distance_t), turns, limit);
    };
    auto const device = [&](void const *const *a, std::size_t row_count, void const *const *b, std::size_t column_count,
                            raw_distance_t const *limit) {
        auto const launch = [&](auto... arguments) { backend.call(kernel, arguments...); };
        call(launch, launch, a, row_count, b, column_count, score, rotation, limit);
        return stats.expect(backend.synchronize());
    };
    auto const serial = [&](void const *const *a, std::size_t row_count, void const *const *b, std::size_t column_count,
                            raw_distance_t *distances, nk_f32_t *turns) {
        raw_distance_t const unbounded = distance_t::positive_infinity().raw_;
        auto const bind = [](auto function) {
            return [function](auto... arguments) { return function(arguments..., nullptr); };
        };
        if constexpr (narrow_k)
            return call(bind(nk_chamfers_packed_f16_serial), bind(nk_hausdorffs_packed_f16_serial), a, row_count, b,
                        column_count, distances, turns, &unbounded);
        else
            return call(bind(nk_chamfers_packed_f32_serial), bind(nk_hausdorffs_packed_f32_serial), a, row_count, b,
                        column_count, distances, turns, &unbounded);
    };
    auto const radius = [](void const *packed) {
        if constexpr (narrow_k)
            return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
        else return static_cast<nk_f64_t>(((nk_shapes_packed_header_f64_t const *)packed)->max_radius);
    };

    auto const row_packs = device_random_packed<scalar_type_>(settings, backend, generator, rows, points, stats);
    auto const column_packs = device_random_packed<scalar_type_>(settings, backend, generator, columns, points + 3,
                                                                 stats);
    raw_distance_t const unbounded = distance_t::positive_infinity().raw_;
    if (!device(row_packs.data(), rows, column_packs.data(), columns, &unbounded)) return stats;
    std::vector<raw_distance_t> expected(rows * columns);
    std::vector<nk_f32_t> expected_rotations(rows * columns * 9);
    stats.expect(
        serial(row_packs.data(), rows, column_packs.data(), columns, expected.data(), expected_rotations.data()));
    for (std::size_t cell = 0; cell != rows * columns; ++cell) {
        nk_f64_t const reach = radius(row_packs[cell / columns]) + radius(column_packs[cell % columns]);
        nk_f64_t const squares = 8 * nk_accumulation_error_bound(distance_t::dtype()) * reach * reach;
        stats.accumulate_bounded(distance_t::from_raw(score[cell]), f64_t(expected[cell]), std::sqrt(squares));
    }
    for (std::size_t cell = 0; cell != rows * columns * 9; ++cell)
        stats.accumulate_bounded(f32_t::from_raw(rotation[cell]), f64_t(expected_rotations[cell]),
                                 8 * nk_accumulation_error_bound(nk_f32_k));

    // Limits below, at and above each exact distance, on clouds spanning several slabs.
    std::size_t const slabbed = nk::max_of<std::size_t>(points, 97);
    auto const slab_rows = device_random_packed<scalar_type_>(settings, backend, generator, 3, slabbed, stats);
    auto const slab_columns = device_random_packed<scalar_type_>(settings, backend, generator, 3, slabbed + 3, stats);
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0);
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t column = 0; column != 3; ++column) {
            if (!device(slab_rows.data() + row, 1, slab_columns.data() + column, 1, &unbounded)) return stats;
            raw_distance_t const exact = score[0];
            nk_f32_t exact_rotation[9];
            std::memcpy(exact_rotation, rotation, sizeof(exact_rotation));
            nk_f64_t const reach = 3 * roundoff * (radius(slab_rows[row]) + radius(slab_columns[column]));
            for (raw_distance_t limit : {raw_distance_t(0), exact / 4, exact / 2, exact, exact * 2}) {
                if (!device(slab_rows.data() + row, 1, slab_columns.data() + column, 1, &limit)) return stats;
                if (exact <= limit) {
                    stats.expect(score[0] == exact && !std::memcmp(rotation, exact_rotation, sizeof(exact_rotation)),
                                 "a limit changed a distance within it or its rotation");
                    continue;
                }
                stats.expect(score[0] > limit && (limit != 0 || score[0] < exact) && score[0] <= exact + reach,
                             "a pair past its limit reported no lower bound past the limit");
                stats.expect(shapes_rotation_is_flip<scalar_type_>(rotation, slab_rows[row], slab_columns[column]),
                             "a stopped pair reported no flip of its frames");
            }
        }
    return stats;
}

/** Transport clouds: random, or exact ties of a lattice with power-of-two steps, points repeated
 *  4 times, or a regular polygon around a column of points. */
enum class transport_layout_t { random_k, lattice_k, repeated_k, ring_k };

/** Packs @p count clouds of @p points each of @p scalar_type_ in @p layout serially into @p backend
 *  memory, back to back, and the table of their addresses after them. */
template <typename scalar_type_, typename backend_type_>
auto device_transport_packs(settings_t const &settings, backend_type_ const &backend, std::mt19937 &generator,
                            transport_layout_t layout, std::size_t count, std::size_t points, error_stats_t &stats) {
    auto coordinates = make_vector<scalar_type_>(count * points * 3);
    fill_random(settings, generator, coordinates);
    std::vector<std::span<scalar_type_ const>> clouds(count);
    std::size_t const corners = points > 8 ? points - points / 8 : points;
    for (std::size_t index = 0; index != count; ++index) {
        scalar_type_ *values = coordinates.values_data() + index * points * 3;
        for (std::size_t point = 0; point != points; ++point) {
            float const shift = float(index % 3);
            if (layout == transport_layout_t::lattice_k)
                values[point * 3] = float(point % 4) + shift, values[point * 3 + 1] = 0.5f * float(point / 4 % 4),
                               values[point * 3 + 2] = 0.25f * float(point / 16);
            else if (layout == transport_layout_t::repeated_k)
                for (std::size_t axis = 0; axis != 3; ++axis)
                    values[point * 3 + axis] = values[(point - point % 4) * 3 + axis];
            else if (layout == transport_layout_t::ring_k) {
                double const angle = 6.283185307179586 * double(point) / double(corners);
                bool const corner = point < corners;
                values[point * 3] = float(corner ? std::cos(angle) : 0.0) + shift;
                values[point * 3 + 1] = float(corner ? std::sin(angle) : 0.0);
                values[point * 3 + 2] = corner ? 0.0f : 0.5f * float(point - corners);
            }
        }
        clouds[index] = {values, points * 3};
    }
    using packs_t = nk::shapes_packed<scalar_type_, typename backend_type_::template allocator<char>>;
    auto packs = packs_t::make(clouds, allocator_of<char>(backend), nk_cap_serial_k);
    stats.expect(packs.status);
    return std::move(packs.value);
}

/** A backend's transport @p kernel against the serial one on random and tied clouds of unequal
 *  sizes, within the bound the CPU capabilities meet: under a root, the roundings of a plan of up
 *  to n + m − 1 arcs, each at most the squared sum of both radii, and the simplex tolerance. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename backend_type_, typename workspace_size_type_,
          typename kernel_type_>
error_stats_t test_transport_packed(settings_t const &settings, workspace_size_type_ workspace_size,
                                    kernel_type_ kernel) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    constexpr bool procrustes_k = kind_ == nk_kernel_wassersteinprocrustes_packed_k;
    constexpr bool wide_k = std::is_same_v<scalar_type_, f32_t>;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<backend_type_>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 5, columns = 7, largest = points + 3;
    auto const serial_workspace_size = procrustes_k ? (wide_k ? nk_wassersteinprocrustes_workspace_size_f32_serial
                                                              : nk_wassersteinprocrustes_workspace_size_f16_serial)
                                                    : (wide_k ? nk_gromovwassersteins_workspace_size_f32_serial
                                                              : nk_gromovwassersteins_workspace_size_f16_serial);
    auto const serial = [] {
        if constexpr (procrustes_k && wide_k) return nk_wassersteinprocrustes_packed_f32_serial;
        else if constexpr (procrustes_k) return nk_wassersteinprocrustes_packed_f16_serial;
        else if constexpr (wide_k) return nk_gromovwassersteins_packed_f32_serial;
        else return nk_gromovwassersteins_packed_f16_serial;
    }();
    nk_size_t bytes = 0, serial_bytes = 0;
    stats.expect(workspace_size(largest, largest, &bytes));
    stats.expect(serial_workspace_size(largest, largest, &serial_bytes));
    auto workspace = make_vector<std::byte>(backend, bytes);
    auto serial_workspace = make_vector<std::byte>(serial_bytes);
    auto results = make_vector<distance_t>(backend, rows * columns);
    auto rotations = make_vector<f32_t>(backend, rows * columns * 9);
    std::vector<raw_distance_t> expected(rows * columns);
    std::vector<nk_f32_t> expected_rotations(rows * columns * 9);
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0), terms = static_cast<nk_f64_t>(points + largest);
    for (transport_layout_t const layout : {transport_layout_t::random_k, transport_layout_t::lattice_k,
                                            transport_layout_t::repeated_k, transport_layout_t::ring_k}) {
        auto const row_packs = device_transport_packs<scalar_type_>(settings, backend, generator, layout, rows, points,
                                                                    stats);
        auto const column_packs = device_transport_packs<scalar_type_>(settings, backend, generator, layout, columns,
                                                                       largest, stats);
        backend.call(kernel, row_packs.data(), nk_size_t(rows), column_packs.data(), nk_size_t(columns),
                     results.raw_values_data(), columns * sizeof(raw_distance_t), rotations.raw_values_data(),
                     static_cast<void *>(workspace.raw_values_data()), bytes);
        if (!stats.expect(backend.synchronize())) return stats;
        stats.expect(serial(row_packs.data(), rows, column_packs.data(), columns, expected.data(),
                            columns * sizeof(raw_distance_t), expected_rotations.data(),
                            serial_workspace.raw_values_data(), serial_bytes, nullptr));
        for (std::size_t row = 0; row != rows; ++row)
            for (std::size_t column = 0; column != columns; ++column) {
                auto const radius = [](void const *packed) {
                    if constexpr (wide_k)
                        return static_cast<nk_f64_t>(((nk_shapes_packed_header_f64_t const *)packed)->max_radius);
                    else return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
                };
                nk_f64_t const reach = radius(row_packs[row]) + radius(column_packs[column]);
                nk_f64_t const scale = reach * reach, tolerance = nk::shapes_transport_epsilon_k * scale;
                nk_f64_t const reference = expected[row * columns + column];
                nk_f64_t bound;
                if constexpr (procrustes_k) {
                    nk_f64_t const squared = (terms + 32) * roundoff * scale + tolerance;
                    bound = nk::min_of(std::sqrt(squared), squared / reference);
                }
                else {
                    nk_f64_t const loss = 4 * ((terms + 64) * roundoff * scale + tolerance);
                    bound = nk::min_of(0.5 * std::sqrt(loss), loss / (8 * reference));
                }
                stats.accumulate_bounded(distance_t::from_raw(results.raw_values_data()[row * columns + column]),
                                         f64_t(reference), bound);
            }
    }
    return stats;
}

/** The dispatching entry points, over the capabilities of the device @p backend_type_ runs on. */
template <typename backend_type_>
void test_cross_dispatch(error_stats_section_t &check) {
    check.section("Cross Dispatch", nk_cap_cuda_k | nk_cap_rocm_k | nk_cap_metal_k);
    check(
        "dots_packed_i8_dispatch", test_dots_packed<i8_t, backend_type_>,
        [capabilities = check.available](auto... arguments) noexcept {
            return call_best<nk_dots_pack_size_i8_best>(capabilities, arguments...);
        },
        [capabilities = check.available](auto... arguments) noexcept {
            return call_best<nk_dots_pack_i8_best>(capabilities, arguments...);
        },
        [capabilities = check.available](auto... arguments) noexcept {
            return call_best<nk_dots_packed_i8_best>(capabilities, arguments...);
        });
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_CROSS_DEVICE_HPP
