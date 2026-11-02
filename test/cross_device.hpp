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
#include "numkong/shapes.hpp"      // `nk::shapes_packed`

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

/** @p count random F16 clouds of @p points each, packed serially into @p backend memory. */
template <typename backend_type_>
auto device_random_packed(settings_t const &settings, backend_type_ const &backend, std::mt19937 &generator,
                          std::size_t count, std::size_t points, error_stats_t &stats) {
    auto coordinates = make_vector<f16_t>(count * points * 3);
    fill_random(settings, generator, coordinates);
    std::vector<std::span<f16_t const>> clouds(count);
    for (std::size_t index = 0; index != count; ++index)
        clouds[index] = {coordinates.values_data() + index * points * 3, points * 3};
    using packs_t = nk::shapes_packed<f16_t, typename backend_type_::template allocator<char>>;
    auto packs = packs_t::make(clouds, allocator_of<char>(backend), nk_cap_serial_k);
    stats.expect(packs.status);
    return std::move(packs.value);
}

/** Superposed RMSD of F16 frames on the device against the serial kernel, whose fold it repeats
 *  over products the matrix units sum in another order, within the bound of the one-to-one F16
 *  kernels: n roundings of the F32 moments, n × ε × spread², through the square root. */
template <typename backend_type_, typename kernel_type_>
error_stats_t test_rmsds_packed_f16(settings_t const &settings, kernel_type_ kernel) {
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<backend_type_>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 37, columns = 45;
    auto const row_packs = device_random_packed(settings, backend, generator, rows, points, stats);
    auto const column_packs = device_random_packed(settings, backend, generator, columns, points, stats);
    auto results = make_vector<f32_t>(backend, rows * columns);
    backend.call(kernel, row_packs.data(), rows, column_packs.data(), columns, results.raw_values_data(),
                 columns * sizeof(nk_f32_t));
    if (!stats.expect(backend.synchronize())) return stats;
    std::vector<nk_f32_t> expected(rows * columns);
    stats.expect(nk_rmsds_packed_f16_serial(row_packs.data(), rows, column_packs.data(), columns, expected.data(),
                                            columns * sizeof(nk_f32_t), nullptr));
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            auto const radius = [](void const *packed) {
                nk_f32_t const *spreads = ((nk_shapes_packed_header_f32_t const *)packed)->spreads;
                return std::sqrt(static_cast<nk_f64_t>(spreads[0]) + spreads[1] + spreads[2]);
            };
            nk_f64_t const spread = radius(row_packs[row]) + radius(column_packs[column]);
            nk_f64_t const reference = expected[row * columns + column];
            nk_f64_t const squares = points * nk_accumulation_error_bound(nk_f32_k) * spread * spread;
            nk_f64_t const bound = reference > 0 ? std::min(std::sqrt(squares), squares / reference)
                                                 : std::sqrt(squares);
            stats.accumulate_bounded(f32_t::from_raw(results.raw_values_data()[row * columns + column]),
                                     f64_t(reference), bound);
        }
    return stats;
}

/** Chamfer or Hausdorff of F16 clouds and their rotations on the device against the serial kernel.
 *  Matrix-unit kernels expand ‖a − b‖² into ‖a‖² + ‖b‖² − 2 × a · b only to search, each norm
 *  split into two F16 halves, and recompute every nearest square in the serial order, so a
 *  distance moves by the root of a few F32 roundings of the norms, and a rotation, built from the
 *  same frames, by a few roundings of each of its entries. */
template <typename backend_type_, nk_kernel_kind_t kind_, typename kernel_type_>
error_stats_t test_shapes_flips_f16(settings_t const &settings, kernel_type_ kernel) {
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<backend_type_>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 7, columns = 9;
    auto const row_packs = device_random_packed(settings, backend, generator, rows, points, stats);
    auto const column_packs = device_random_packed(settings, backend, generator, columns, points + 3, stats);
    auto results = make_vector<f32_t>(backend, rows * columns);
    auto rotations = make_vector<f32_t>(backend, rows * columns * 9);
    nk_f32_t const unbounded = NUMKONG_F32_INF;
    std::vector<nk_f32_t> expected(rows * columns), expected_rotations(rows * columns * 9);
    if constexpr (kind_ == nk_kernel_chamfers_packed_k) {
        backend.call(kernel, row_packs.data(), rows, column_packs.data(), columns, results.raw_values_data(),
                     columns * sizeof(nk_f32_t), rotations.raw_values_data(), &unbounded, nullptr, nk_size_t(0));
        stats.expect(nk_chamfers_packed_f16_serial(row_packs.data(), rows, column_packs.data(), columns,
                                                   expected.data(), columns * sizeof(nk_f32_t),
                                                   expected_rotations.data(), &unbounded, nullptr, 0, nullptr));
    }
    else {
        backend.call(kernel, row_packs.data(), rows, column_packs.data(), columns, results.raw_values_data(),
                     columns * sizeof(nk_f32_t), rotations.raw_values_data(), &unbounded);
        stats.expect(nk_hausdorffs_packed_f16_serial(row_packs.data(), rows, column_packs.data(), columns,
                                                     expected.data(), columns * sizeof(nk_f32_t),
                                                     expected_rotations.data(), &unbounded, nullptr));
    }
    if (!stats.expect(backend.synchronize())) return stats;
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            auto const radius = [](void const *packed) {
                return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
            };
            nk_f64_t const reach = radius(row_packs[row]) + radius(column_packs[column]);
            nk_f64_t const squares = 8 * nk_accumulation_error_bound(nk_f32_k) * reach * reach;
            stats.accumulate_bounded(f32_t::from_raw(results.raw_values_data()[row * columns + column]),
                                     f64_t(expected[row * columns + column]), std::sqrt(squares));
        }
    for (std::size_t cell = 0; cell != rows * columns * 9; ++cell)
        stats.accumulate_bounded(f32_t::from_raw(rotations.raw_values_data()[cell]), f64_t(expected_rotations[cell]),
                                 8 * nk_accumulation_error_bound(nk_f32_k));
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
