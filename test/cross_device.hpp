/**
 *  @file test/cross_device.hpp
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Shared dispatch checks for CUDA, ROCm, and Metal backends.
 */
#pragma once
#ifndef NUMKONG_TEST_CROSS_DEVICE_HPP
#define NUMKONG_TEST_CROSS_DEVICE_HPP

#include <span> // `std::span`

#include "backend_device.hpp" // `device_backend`
#include "cross.hpp"          // `test_dots_packed`, `attention_weights_t`
#include "numkong/shapes.hpp" // `nk::shapes_packed`
#include "shapes.hpp"         // `shapes_nearest_reference`, `shapes_storage_bound`

namespace ashvardanian::numkong::test {

/** The @p count clouds of @p points each that @p coordinates holds back to back. */
template <typename scalar_type_>
std::vector<std::span<scalar_type_ const>> back_to_back_clouds(nk::vector<scalar_type_> const &coordinates,
                                                               std::size_t count, std::size_t points) {
    std::vector<std::span<scalar_type_ const>> clouds(count);
    for (std::size_t index = 0; index != count; ++index)
        clouds[index] = {coordinates.values_data() + index * points * 3, points * 3};
    return clouds;
}

/** @p count random clouds of @p scalar_type_ of @p points each, kept in @p coordinates and packed
 *  on the host into @p backend memory. */
template <typename scalar_type_, typename backend_type_>
auto device_random_packed(backend_type_ const &backend, split_mix_generator_t &generator, std::size_t count,
                          std::size_t points, nk::vector<scalar_type_> &coordinates, error_stats_t &stats) {
    coordinates = make_vector<scalar_type_>(count * points * 3);
    fill_random(generator, coordinates);
    std::vector<std::span<scalar_type_ const>> const clouds = back_to_back_clouds(coordinates, count, points);
    using packs_t = nk::shapes_packed<scalar_type_, typename backend_type_::template allocator<char>>;
    auto packs = packs_t::make(clouds, allocator_of<char>(backend));
    stats.expect_status(packs.status);
    return std::move(packs.value);
}

/** Superposed RMSD on the device against the serial kernel on the same packs, within twice the fold
 *  the matrix units sum in their own order, n × ε × spread² through the square root, then against
 *  the reference in the scalar's reference type on the raw clouds, within that fold plus the
 *  displacement both clouds' packing allows. */
template <typename scalar_type_, typename backend_type_, typename kernel_type_>
error_stats_t test_rmsds_packed(kernel_type_ kernel) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    using reference_t = reference_for<scalar_type_>;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<backend_type_>();
    split_mix_generator_t generator(case_stream_key());
    std::size_t const points = test_run.settings.shape_points, rows = 37, columns = 45;
    nk::vector<scalar_type_> row_coordinates, column_coordinates;
    auto const row_packs = device_random_packed<scalar_type_>(backend, generator, rows, points, row_coordinates, stats);
    auto const column_packs = device_random_packed<scalar_type_>(backend, generator, columns, points,
                                                                 column_coordinates, stats);
    auto results = make_vector<distance_t>(backend, rows * columns);
    backend.call(kernel, row_packs.data(), rows, column_packs.data(), columns, results.raw_values_data(),
                 columns * sizeof(raw_distance_t));
    if (!stats.expect_status(backend.synchronize())) return stats;
    std::vector<std::span<scalar_type_ const>> const row_clouds = back_to_back_clouds(row_coordinates, rows, points);
    std::vector<std::span<scalar_type_ const>> const column_clouds = back_to_back_clouds(column_coordinates, columns,
                                                                                         points);
    std::vector<shapes_reference_cloud_<reference_t>> const row_references =
        shapes_pack_references_<scalar_type_, reference_t, nk_kernel_shapes_pack_k>(row_clouds);
    std::vector<shapes_reference_cloud_<reference_t>> const column_references =
        shapes_pack_references_<scalar_type_, reference_t, nk_kernel_shapes_pack_k>(column_clouds);
    std::vector<reference_t> expected(rows * columns);
    stats.expect_status(shapes_rmsds_reference<scalar_type_>(row_clouds, column_clouds, expected.data(),
                                                             columns * sizeof(reference_t)));
    auto const spread = [](shapes_reference_cloud_<reference_t> const &cloud) {
        return std::sqrt(to_double(cloud.spreads[0] + cloud.spreads[1] + cloud.spreads[2]));
    };
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            std::size_t const cell = row * columns + column;
            nk_f64_t const reach = spread(row_references[row]) + spread(column_references[column]);
            nk_f64_t const reference = to_double(expected[cell]);
            nk_f64_t const squares = points * nk_accumulation_error_bound(distance_t::dtype()) * reach * reach;
            nk_f64_t const fold = reference > 0 ? nk::min_of(std::sqrt(squares), squares / reference)
                                                : std::sqrt(squares);
            nk_f64_t const displacement =
                shapes_storage_bound<scalar_type_, nk_kernel_rmsds_packed_k>(row_references[row]) +
                shapes_storage_bound<scalar_type_, nk_kernel_rmsds_packed_k>(column_references[column]);
            distance_t const actual = results[cell];
            stats.accumulate_bounded(actual, expected[cell], fold + displacement);
        }
    return stats;
}

/** Whether @p rotation maps the frame of @p a_packed onto that of @p b_packed, packs of
 *  @p scalar_type_ clouds, up to one sign per axis: one of the pair's own flips. */
template <typename scalar_type_>
bool shapes_rotation_is_flip(nk_f32_t const *rotation, void const *a_packed, void const *b_packed) {
    using constant_t = typename scalar_type_::shapes_distance_t;
    constant_t a_frame[4], b_frame[4];
    if (nk::failed(nk::shapes_packed_shape<scalar_type_>(a_packed, nullptr, nullptr, a_frame, nullptr, nullptr, nullptr,
                                                         nk::default_capabilities())) ||
        nk::failed(nk::shapes_packed_shape<scalar_type_>(b_packed, nullptr, nullptr, b_frame, nullptr, nullptr, nullptr,
                                                         nk::default_capabilities())))
        return false;
    constant_t a_axes[9], b_axes[9];
    quaternion_to_mat3x3_(a_frame, a_axes);
    quaternion_to_mat3x3_(b_frame, b_axes);
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t column = 0; column != 3; ++column) {
            nk_f64_t sign = 0;
            for (std::size_t left = 0; left != 3; ++left)
                for (std::size_t right = 0; right != 3; ++right)
                    sign += to_double(a_axes[row * 3 + left]) * rotation[left * 3 + right] * b_axes[column * 3 + right];
            if (std::fabs(std::fabs(sign) - (row == column)) > 1e-4) return false;
        }
    return true;
}

/** Chamfer or Hausdorff and their rotations on the device against their template in the scalar's
 *  reference type, then at limits below, at and above each exact distance: within it a limit keeps
 *  the bits and the rotation, past it a pair reports a lower bound past the limit with one of its
 *  own flips. Matrix-unit kernels expand ‖a − b‖² into ‖a‖² + ‖b‖² − 2 × a · b only to search, each
 *  norm split into two halves, and recompute every nearest square from the coordinate differences,
 *  so a distance moves by the root of a few roundings of the norms in the distance type, and a
 *  rotation, built from the same frames, by a few F32 roundings of each entry. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename backend_type_, typename kernel_type_>
error_stats_t test_shapes_flips(kernel_type_ kernel) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    using reference_t = reference_for<scalar_type_>;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<backend_type_>();
    split_mix_generator_t generator(case_stream_key());
    std::size_t const points = test_run.settings.shape_points, rows = 7, columns = 9;
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
        return stats.expect_status(backend.synchronize());
    };
    auto const radius = [](void const *packed) { return shapes_packed_radius<scalar_type_>(packed); };

    nk::vector<scalar_type_> row_coordinates, column_coordinates;
    auto const row_packs = device_random_packed<scalar_type_>(backend, generator, rows, points, row_coordinates, stats);
    auto const column_packs = device_random_packed<scalar_type_>(backend, generator, columns, points + 3,
                                                                 column_coordinates, stats);
    raw_distance_t const unbounded = distance_t::positive_infinity().raw_;
    if (!device(row_packs.data(), rows, column_packs.data(), columns, &unbounded)) return stats;
    std::vector<std::span<scalar_type_ const>> const row_clouds = back_to_back_clouds(row_coordinates, rows, points);
    std::vector<std::span<scalar_type_ const>> const column_clouds = back_to_back_clouds(column_coordinates, columns,
                                                                                         points + 3);
    std::vector<shapes_reference_cloud_<reference_t>> const row_references =
        shapes_pack_references_<scalar_type_, reference_t, nk_kernel_shapes_pack_k>(row_clouds);
    std::vector<shapes_reference_cloud_<reference_t>> const column_references =
        shapes_pack_references_<scalar_type_, reference_t, nk_kernel_shapes_pack_k>(column_clouds);
    std::vector<reference_t> expected(rows * columns), expected_rotations(rows * columns * 9);
    shapes_nearest_reference<scalar_type_, kind_>(row_clouds, column_clouds, expected.data(),
                                                  columns * sizeof(reference_t), expected_rotations.data());
    for (std::size_t cell = 0; cell != rows * columns; ++cell) {
        shapes_reference_cloud_<reference_t> const &row_reference = row_references[cell / columns];
        shapes_reference_cloud_<reference_t> const &column_reference = column_references[cell % columns];
        nk_f64_t const reach = radius(row_packs[cell / columns]) + radius(column_packs[cell % columns]);
        nk_f64_t const squares = 8 * nk_accumulation_error_bound(distance_t::dtype()) * reach * reach;
        nk_f64_t const row_displacement = shapes_storage_bound<scalar_type_, kind_>(row_reference);
        nk_f64_t const column_displacement = shapes_storage_bound<scalar_type_, kind_>(column_reference);
        distance_t const actual = distance_t::from_raw(score[cell]);
        stats.accumulate_bounded(actual, expected[cell], std::sqrt(squares) + row_displacement + column_displacement);
        nk_f64_t const turn = row_displacement / to_double(row_reference.max_radius) +
                              column_displacement / to_double(column_reference.max_radius);
        nk_f64_t const entry_bound = 8 * nk_accumulation_error_bound(nk_f32_k) +
                                     shapes_rotation_bound<scalar_type_, kind_>(turn);
        for (std::size_t entry = 0; entry != 9; ++entry)
            stats.accumulate_bounded(f32_t::from_raw(rotation[cell * 9 + entry]), expected_rotations[cell * 9 + entry],
                                     entry_bound);
    }

    // Limits below, at and above each exact distance, on clouds spanning several slabs.
    std::size_t const slabbed = nk::max_of<std::size_t>(points, 97);
    nk::vector<scalar_type_> slab_row_coordinates, slab_column_coordinates;
    auto const slab_rows = device_random_packed<scalar_type_>(backend, generator, 3, slabbed, slab_row_coordinates,
                                                              stats);
    auto const slab_columns = device_random_packed<scalar_type_>(backend, generator, 3, slabbed + 3,
                                                                 slab_column_coordinates, stats);
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0);
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t column = 0; column != 4; ++column) {
            // Last, the row's own cloud: at a limit of 0 only its identity flip runs to the end
            void const *const *column_pack = column != 3 ? slab_columns.data() + column : slab_rows.data() + row;
            if (!device(slab_rows.data() + row, 1, column_pack, 1, &unbounded)) return stats;
            raw_distance_t const exact = score[0];
            nk_f32_t exact_rotation[9];
            std::memcpy(exact_rotation, rotation, sizeof(exact_rotation));
            nk_f64_t const reach = 3 * roundoff * (radius(slab_rows[row]) + radius(*column_pack));
            for (raw_distance_t limit : {raw_distance_t(0), exact / 4, exact / 2, exact, exact * 2}) {
                if (!device(slab_rows.data() + row, 1, column_pack, 1, &limit)) return stats;
                if (exact <= limit) {
                    stats.expect_true(
                        score[0] == exact && !std::memcmp(rotation, exact_rotation, sizeof(exact_rotation)),
                        "a limit changed a distance within it or its rotation");
                    continue;
                }
                stats.expect_true(score[0] > limit && (limit != 0 || score[0] < exact) && score[0] <= exact + reach,
                                  "a pair past its limit reported no lower bound past the limit");
                stats.expect_true(shapes_rotation_is_flip<scalar_type_>(rotation, slab_rows[row], *column_pack),
                                  "a stopped pair reported no flip of its frames");
            }
        }
    return stats;
}

/** Transport clouds: random, or exact ties of a lattice with power-of-two steps, points repeated
 *  4 times, or a regular polygon around a column of points. */
enum class transport_layout_t { random_k, lattice_k, repeated_k, ring_k };

/** Packs @p count clouds of @p points each of @p scalar_type_ in @p layout on the host into
 *  @p backend memory, back to back, and the table of their addresses after them, keeping the
 *  coordinates in @p coordinates. */
template <typename scalar_type_, typename backend_type_>
auto device_transport_packs(backend_type_ const &backend, split_mix_generator_t &generator, transport_layout_t layout,
                            std::size_t count, std::size_t points, nk::vector<scalar_type_> &coordinates,
                            error_stats_t &stats) {
    coordinates = make_vector<scalar_type_>(count * points * 3);
    fill_random(generator, coordinates);
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
    }
    std::vector<std::span<scalar_type_ const>> const clouds = back_to_back_clouds(coordinates, count, points);
    using packs_t = nk::shapes_packed<scalar_type_, typename backend_type_::template allocator<char>>;
    auto packs = packs_t::make(clouds, allocator_of<char>(backend));
    stats.expect_status(packs.status);
    return std::move(packs.value);
}

/** A backend's transport @p kernel against its template in the scalar's reference type on random
 *  and tied clouds of unequal sizes, within the bound the CPU capabilities meet: under a root, the
 *  roundings of a plan of up to n + m − 1 arcs, each at most the squared sum of both radii, and the
 *  simplex tolerance. Tied clouds keep wider types in the template's local optimum, as every solver
 *  enters the first arc within the tolerance of a block's least, but f16 costs round in f32, far
 *  above that tolerance, so f16 rings may settle in another local optimum: Wasserstein-Procrustes
 *  ones within the root, Gromov-Wasserstein ones within the f32 rounding of their costs. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename backend_type_, typename workspace_size_type_,
          typename kernel_type_>
error_stats_t test_transport_packed(workspace_size_type_ workspace_size, kernel_type_ kernel) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    using reference_t = reference_for<scalar_type_>;
    constexpr bool procrustes_k = kind_ == nk_kernel_wassersteinprocrustes_packed_k;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<backend_type_>();
    split_mix_generator_t generator(case_stream_key());
    std::size_t const points = test_run.settings.shape_points, rows = 5, columns = 7, largest = points + 3;
    nk_size_t bytes = 0;
    stats.expect_status(workspace_size(largest, largest, &bytes));
    auto workspace = make_vector<std::byte>(backend, bytes);
    auto results = make_vector<distance_t>(backend, rows * columns);
    auto rotations = make_vector<f32_t>(backend, rows * columns * 9);
    std::vector<reference_t> expected(rows * columns);
    nk::vector<scalar_type_> row_coordinates, column_coordinates;
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0), terms = to_double(points + largest);
    for (transport_layout_t const layout : {transport_layout_t::random_k, transport_layout_t::lattice_k,
                                            transport_layout_t::repeated_k, transport_layout_t::ring_k}) {
        auto const row_packs = device_transport_packs<scalar_type_>(backend, generator, layout, rows, points,
                                                                    row_coordinates, stats);
        auto const column_packs = device_transport_packs<scalar_type_>(backend, generator, layout, columns, largest,
                                                                       column_coordinates, stats);
        backend.call(kernel, row_packs.data(), nk_size_t(rows), column_packs.data(), nk_size_t(columns),
                     results.raw_values_data(), columns * sizeof(raw_distance_t), rotations.raw_values_data(),
                     static_cast<void *>(workspace.raw_values_data()), bytes);
        if (!stats.expect_status(backend.synchronize())) return stats;
        std::vector<std::span<scalar_type_ const>> const row_clouds = back_to_back_clouds(row_coordinates, rows,
                                                                                          points);
        std::vector<std::span<scalar_type_ const>> const column_clouds = back_to_back_clouds(column_coordinates,
                                                                                             columns, largest);
        std::vector<shapes_reference_cloud_<reference_t>> const row_references =
            shapes_pack_references_<scalar_type_, reference_t, nk_kernel_shapes_pack_k>(row_clouds);
        std::vector<shapes_reference_cloud_<reference_t>> const column_references =
            shapes_pack_references_<scalar_type_, reference_t, nk_kernel_shapes_pack_k>(column_clouds);
        // Tied layouts leave the rotations free, so only the distances are compared
        stats.expect_status(
            shapes_transport_reference<scalar_type_, kind_, nk_kernel_shapes_pack_k, reference_t, reference_t>(
                row_clouds, column_clouds, expected.data(), columns * sizeof(reference_t), nullptr));
        for (std::size_t row = 0; row != rows; ++row)
            for (std::size_t column = 0; column != columns; ++column) {
                auto const radius = [](void const *packed) { return shapes_packed_radius<scalar_type_>(packed); };
                nk_f64_t const reach = radius(row_packs[row]) + radius(column_packs[column]);
                nk_f64_t const scale = reach * reach, tolerance = shapes_transport_epsilon_k * scale;
                nk_f64_t const reference = to_double(expected[row * columns + column]);
                bool const rooted = std::is_same_v<scalar_type_, f16_t> && layout == transport_layout_t::ring_k;
                nk_f64_t bound;
                if constexpr (procrustes_k) {
                    nk_f64_t const squared = (terms + 32) * roundoff * scale + tolerance;
                    bound = rooted ? std::sqrt(squared) : nk::min_of(std::sqrt(squared), squared / reference);
                }
                else {
                    nk_f64_t const tie_roundoff = rooted ? nk::max_of(roundoff, half_ulp<f32_t>(1.0)) : roundoff;
                    nk_f64_t const loss = 4 * ((terms + 64) * tie_roundoff * scale + tolerance);
                    bound = nk::min_of(0.5 * std::sqrt(loss), loss / (8 * reference));
                }
                nk_f64_t const displacement = shapes_storage_bound<scalar_type_, kind_>(row_references[row]) +
                                              shapes_storage_bound<scalar_type_, kind_>(column_references[column]);
                distance_t const actual = results[row * columns + column];
                stats.accumulate_bounded(actual, expected[row * columns + column],
                                         bound + (procrustes_k ? 1 : 2) * displacement);
            }
    }
    return stats;
}

/** The dispatching entry points, over the capabilities of the device @p backend_type_ runs on. */
template <typename backend_type_>
void run_cross_dispatch_cases() {
    run_kernel_case(
        "test_dots_packed<i8_t, backend_type_>(nk_dots_packed_i8_best)", test_dots_packed<i8_t, backend_type_>,
        nk_cap_cuda_k | nk_cap_rocm_k | nk_cap_metal_k,
        [capabilities = test_run.settings.capabilities](auto... arguments) noexcept {
            return call_best<nk_dots_pack_size_i8_best>(capabilities, arguments...);
        },
        [capabilities = test_run.settings.capabilities](auto... arguments) noexcept {
            return call_best<nk_dots_pack_i8_best>(capabilities, arguments...);
        },
        [capabilities = test_run.settings.capabilities](auto... arguments) noexcept {
            return call_best<nk_dots_packed_i8_best>(capabilities, arguments...);
        });
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_CROSS_DEVICE_HPP
