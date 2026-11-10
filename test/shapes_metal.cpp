/**
 *  @file test/shapes_metal.cpp
 *  @author Ash Vardanian
 *  @date October 24, 2026
 *  @brief Chamfer and Hausdorff simdgroup bodies against the serial kernels, limits of the public
 *      kernels, and transport on random and tied clouds.
 *
 *  The bodies are device helpers, so this unit compiles its own kernel over them, after the source
 *  the library embeds, and keeps the @c metal exports of their header switched off, as the library
 *  already defines them.
 */
#undef NUMKONG_TARGET_METAL
#define NUMKONG_TARGET_METAL 0
#include "cross_device.hpp" // `device_shapes_packed`

#if NUMKONG_ARCH_METAL_
#include "numkong/shapes/metal.h" // `nk_shapes_source_metal_`, `nk_shapes_resolve_packs_metal_`
#endif

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_METAL_

/** The library's shapes source and a kernel scoring each pair of the two pack tables on its own
 *  simdgroup, each lane keeping its own result. Kept for the program's life, as compiled pipelines
 *  remember their source by address. */
static char const *shapes_pairs_source_metal() {
    static std::string const source = std::string(nk_shapes_source_metal_) + R"(
kernel void shapes_pairs_kernel_(device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
                                 device float const *limits [[buffer(2)]], device float *results [[buffer(3)]],
                                 constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]],
                                 constant uint &hausdorff [[buffer(5)]], uint thread_index [[thread_position_in_grid]]) {
    ulong const pair = thread_index / 32;
    if (pair >= arguments.rows) return;
    device uchar const *a = nk_shapes_pack_metal_(a_table[pair], arguments);
    device uchar const *b = nk_shapes_pack_metal_(b_table[pair], arguments);
    results[thread_index] = hausdorff ? nk_hausdorff_simdgroup_packed_f16_metal_(a, b, limits[pair], thread_index % 32)
                                      : nk_chamfer_simdgroup_packed_f16_metal_(a, b, limits[pair], thread_index % 32);
}
)";
    return source.c_str();
}

/** Encodes the pairs kernel over @p pairs pairs of the two tables. */
static nk_status_t shapes_pairs_metal(void const *const *a_packed, void const *const *b_packed, std::size_t pairs,
                                      nk_f32_t const *limits, nk_f32_t *results, nk_kernel_kind_t kind,
                                      nk_stream_t stream) {
    nk_metal_call_t call;
    nk_status_t status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_shapes_arguments_metal_t arguments {};
    arguments.rows = pairs;
    status = nk_shapes_resolve_packs_metal_(&call, a_packed, pairs, NUMKONG_SIZE_MAX, &arguments);
    if (status == nk_success_k)
        status = nk_shapes_resolve_packs_metal_(&call, b_packed, pairs, NUMKONG_SIZE_MAX, &arguments);
    if (status != nk_success_k) return nk_abort_metal_(&call, status);
    nk_size_t offsets[4] = {0, 0, 0, 0};
    void *const buffers[4] = {nk_resolve_metal_(&call, a_packed, pairs * sizeof(void const *), offsets),
                              nk_resolve_metal_(&call, b_packed, pairs * sizeof(void const *), offsets + 1),
                              nk_resolve_metal_(&call, limits, pairs * sizeof(nk_f32_t), offsets + 2),
                              nk_resolve_metal_(&call, results, pairs * 32 * sizeof(nk_f32_t), offsets + 3)};
    for (void *buffer : buffers)
        if (!buffer) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    void *const pipeline = nk_pipeline_metal_(call.context, shapes_pairs_source_metal(), "shapes_pairs_kernel_",
                                              NUMKONG_METAL_LANGUAGE_3_1_);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);
    for (std::size_t slot = 0; slot != 4; ++slot) nk_bind_metal_(call.encoder, buffers[slot], offsets[slot], slot);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 4);
    nk_u32_t const hausdorff = kind == nk_kernel_hausdorffs_packed_k;
    nk_bind_bytes_metal_(call.encoder, &hausdorff, sizeof(hausdorff), 5);
    nk_metal_size_t const grid = {nk_size_divide_round_up_(pairs * 32, nk_shapes_threads_metal_k), 1, 1};
    nk_metal_size_t const threads = {nk_shapes_threads_metal_k, 1, 1};
    return nk_dispatch_metal_(&call, grid, threads);
}

/**
 *  @brief Every pair of the device clouds on a simdgroup against the serial kernel.
 *
 *  Unbounded calls must match within the CPU capabilities' bounds: n + m + 8 roundings of a Chamfer
 *  distance, whose sums the simdgroup reduces in another order, and 8 of a Hausdorff distance.
 *  Bounded calls draw each pair's limit up to twice its distance, and a result past the limit must
 *  still be a lower bound of the distance, up to the 3 roundings of each pack radius the Hausdorff
 *  header bound subtracts. Every lane of a simdgroup must return the same value.
 */
template <nk_kernel_kind_t kind_>
error_stats_t test_shapes_simdgroup_metal(settings_t const &settings) {
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<metal_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    auto const packs = device_shapes_packed<f16_t>(settings, backend, generator, stats);
    std::size_t const count = packs.size(), pairs = count * count, lanes = 32;

    auto pair_packs = make_vector<void const *>(backend, 2 * pairs);
    void const **a_packed = pair_packs.values_data();
    void const **b_packed = a_packed + pairs;
    for (std::size_t pair = 0; pair != pairs; ++pair)
        a_packed[pair] = packs[pair / count], b_packed[pair] = packs[pair % count];

    std::vector<nk_f32_t> references(pairs);
    nk_f32_t const unbounded = NUMKONG_F32_INF;
    if constexpr (kind_ == nk_kernel_chamfers_packed_k)
        stats.expect(nk_chamfers_packed_f16_serial(packs.data(), count, packs.data(), count, references.data(),
                                                   count * sizeof(nk_f32_t), nullptr, &unbounded, nullptr, 0, nullptr));
    else
        stats.expect(nk_hausdorffs_packed_f16_serial(packs.data(), count, packs.data(), count, references.data(),
                                                     count * sizeof(nk_f32_t), nullptr, &unbounded, nullptr));

    auto limits = make_vector<f32_t>(backend, pairs);
    auto results = make_vector<f32_t>(backend, pairs * lanes);
    std::uniform_real_distribution<nk_f64_t> fraction(0.0, 2.0);
    nk_f64_t const roundoff = half_ulp<f32_t>(1.0);
    auto const radius = [](void const *packed) {
        return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
    };
    for (int bounded = 0; bounded != 2; ++bounded) {
        for (std::size_t pair = 0; pair != pairs; ++pair)
            limits.raw_values_data()[pair] = bounded ? static_cast<nk_f32_t>(references[pair] * fraction(generator))
                                                     : std::numeric_limits<nk_f32_t>::infinity();
        backend.call(shapes_pairs_metal, a_packed, b_packed, pairs, limits.raw_values_data(), results.raw_values_data(),
                     kind_);
        if (!stats.expect(backend.synchronize())) return stats;
        for (std::size_t pair = 0; pair != pairs; ++pair) {
            nk_f32_t const *lane_results = results.raw_values_data() + pair * lanes;
            for (std::size_t lane = 1; lane != lanes; ++lane)
                stats.expect(std::memcmp(lane_results + lane, lane_results, sizeof(nk_f32_t)) == 0,
                             "the lanes of a simdgroup disagree");
            std::size_t const a_points = packs.points(pair / count);
            std::size_t const b_points = packs.points(pair % count);
            nk_f64_t const reference = references[pair], result = lane_results[0];
            nk_f64_t const roundings = kind_ == nk_kernel_chamfers_packed_k ? a_points + b_points + 8 : 8;
            nk_f64_t const bound = roundings * roundoff * reference;
            if (result <= limits.raw_values_data()[pair] || !std::isfinite(reference))
                stats.accumulate_bounded(f32_t::from_raw(lane_results[0]), f64_t(reference), bound);
            else
                stats.expect(result <= reference + bound + half_ulp<f32_t>(reference) +
                                           3 * roundoff * (radius(a_packed[pair]) + radius(b_packed[pair])),
                             "a result past its limit exceeds the distance");
        }
    }
    return stats;
}

/** Whether @p rotation is 𝑄ₐᵀ × S × 𝑄ᵦ for one of the 8 sign diagonals S of two packs' frames. */
static bool shapes_rotation_is_flip_metal(nk_f32_t const *rotation, void const *a_packed, void const *b_packed) {
    nk_f32_t a_axes[9], b_axes[9];
    nk_quaternion_to_mat3x3_f32_serial_(((nk_shapes_packed_header_f32_t const *)a_packed)->frame, a_axes);
    nk_quaternion_to_mat3x3_f32_serial_(((nk_shapes_packed_header_f32_t const *)b_packed)->frame, b_axes);
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t column = 0; column != 3; ++column) {
            nk_f64_t sign = 0;
            for (std::size_t left = 0; left != 3; ++left)
                for (std::size_t right = 0; right != 3; ++right)
                    sign += static_cast<nk_f64_t>(a_axes[row * 3 + left]) * rotation[left * 3 + right] *
                            b_axes[column * 3 + right];
            if (std::fabs(std::fabs(sign) - (row == column)) > 1e-4) return false;
        }
    return true;
}

/**
 *  @brief One capability's public Chamfer or Hausdorff kernel at limits below, at and above each
 *      pair's exact distance, with rotations, over clouds spanning several slabs of its grain.
 *
 *  At or above the distance a limit keeps the bits of the unbounded call, and its rotation. Below
 *  it, a pair reports a lower bound past the limit, up to the 3 roundings of each pack radius the
 *  Hausdorff header bound subtracts, with one of the pair's own flips.
 */
template <nk_kernel_kind_t kind_>
error_stats_t test_shapes_limits_metal(settings_t const &settings, nk_capability_t capability) {
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<metal_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const rows = 3, columns = 3, points = nk::max_of<std::size_t>(settings.shape_points, 97);
    auto const row_packs = device_random_packed(settings, backend, generator, rows, points, stats);
    auto const column_packs = device_random_packed(settings, backend, generator, columns, points + 3, stats);
    auto score = make_vector<f32_t>(backend, 1);
    auto rotation = make_vector<f32_t>(backend, 9);
    auto const call = [&](std::size_t row, std::size_t column, nk_f32_t limit) {
        if constexpr (kind_ == nk_kernel_chamfers_packed_k) {
            nk_size_t bytes = 1;
            stats.expect(nk_chamfers_workspace_size_f16_best(points, points + 3, capability, &bytes));
            stats.expect(bytes == 0, "GPU Chamfer asked for a workspace");
            backend.call(
                [capability](auto... arguments) noexcept {
                    return call_best<nk_chamfers_packed_f16_best>(capability, arguments...);
                },
                row_packs.data() + row, nk_size_t(1), column_packs.data() + column, nk_size_t(1),
                score.raw_values_data(), sizeof(nk_f32_t), rotation.raw_values_data(), &limit,
                static_cast<void *>(nullptr), bytes);
        }
        else
            backend.call(
                [capability](auto... arguments) noexcept {
                    return call_best<nk_hausdorffs_packed_f16_best>(capability, arguments...);
                },
                row_packs.data() + row, nk_size_t(1), column_packs.data() + column, nk_size_t(1),
                score.raw_values_data(), sizeof(nk_f32_t), rotation.raw_values_data(), &limit);
        return stats.expect(backend.synchronize());
    };
    nk_f64_t const roundoff = half_ulp<f32_t>(1.0);
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            if (!call(row, column, NUMKONG_F32_INF)) return stats;
            nk_f32_t const exact = score.raw_values_data()[0];
            std::array<nk_f32_t, 9> exact_rotation;
            std::memcpy(exact_rotation.data(), rotation.raw_values_data(), sizeof(exact_rotation));
            nk_f64_t const reach =
                3 * roundoff *
                (static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)row_packs[row])->max_radius) +
                 ((nk_shapes_packed_header_f32_t const *)column_packs[column])->max_radius);
            for (nk_f32_t limit : {nk_f32_t(0), exact / 4, exact / 2, exact, exact * 2}) {
                if (!call(row, column, limit)) return stats;
                nk_f32_t const bounded = score.raw_values_data()[0];
                nk_f32_t const *bounded_rotation = rotation.raw_values_data();
                if (exact <= limit) {
                    stats.expect(bounded == exact, "a limit changed a distance within it");
                    stats.expect(std::memcmp(bounded_rotation, exact_rotation.data(), sizeof(exact_rotation)) == 0,
                                 "a limit changed the rotation of a distance within it");
                    continue;
                }
                stats.expect(bounded > limit, "a limit admitted a distance past it");
                stats.expect(limit != 0 || bounded < exact, "a limit of zero never stopped a pair");
                stats.expect(bounded <= exact + reach, "a result past its limit exceeds the distance");
                stats.expect(shapes_rotation_is_flip_metal(bounded_rotation, row_packs[row], column_packs[column]),
                             "a stopped pair reported no flip of its frames");
            }
        }
    return stats;
}

/** Transport clouds: random, or exact ties of a lattice with power-of-two steps, points repeated
 *  4 times, or a regular polygon around a column of points. */
enum class transport_layout_metal_t { random_k, lattice_k, repeated_k, ring_k };

/** @p count F16 clouds of @p points each in @p layout , packed serially into @p backend memory. */
static auto transport_packs_metal(settings_t const &settings, metal_backend_t &backend, std::mt19937 &generator,
                                  transport_layout_metal_t layout, std::size_t count, std::size_t points,
                                  error_stats_t &stats) {
    auto coordinates = make_vector<f16_t>(count * points * 3);
    fill_random(settings, generator, coordinates);
    std::vector<std::span<f16_t const>> clouds(count);
    std::size_t const corners = points > 8 ? points - points / 8 : points;
    for (std::size_t index = 0; index != count; ++index) {
        f16_t *values = coordinates.values_data() + index * points * 3;
        for (std::size_t point = 0; point != points; ++point) {
            float const shift = float(index % 3);
            if (layout == transport_layout_metal_t::lattice_k)
                values[point * 3] = float(point % 4) + shift, values[point * 3 + 1] = 0.5f * float(point / 4 % 4),
                               values[point * 3 + 2] = 0.25f * float(point / 16);
            else if (layout == transport_layout_metal_t::repeated_k)
                for (std::size_t axis = 0; axis != 3; ++axis)
                    values[point * 3 + axis] = values[(point - point % 4) * 3 + axis];
            else if (layout == transport_layout_metal_t::ring_k) {
                double const angle = 6.283185307179586 * double(point) / double(corners);
                bool const corner = point < corners;
                values[point * 3] = float(corner ? std::cos(angle) : 0.0) + shift;
                values[point * 3 + 1] = float(corner ? std::sin(angle) : 0.0);
                values[point * 3 + 2] = corner ? 0.0f : 0.5f * float(point - corners);
            }
        }
        clouds[index] = {values, points * 3};
    }
    using packs_t = nk::shapes_packed<f16_t, metal_backend_t::allocator<char>>;
    auto packs = packs_t::make(clouds, allocator_of<char>(backend), nk_cap_serial_k);
    stats.expect(packs.status);
    return std::move(packs.value);
}

/**
 *  @brief The Metal transport kernels against the serial ones on random and tied clouds of unequal
 *      sizes, launched in slots fewer than the pairs.
 *
 *  Distances stay within the bound the CPU capabilities meet: the roundings of a plan of up to n +
 *  m − 1 arcs, each costing at most the squared sum of both radii, and the simplex tolerance, under
 *  a root. Swapping the sides keeps the bits and transposes the rotation. A workspace below one
 *  slot and a misaligned one are refused, and @c apple10 in the mask falls to @c metal .
 */
template <nk_kernel_kind_t kind_>
error_stats_t test_transport_packed_metal(settings_t const &settings) {
    constexpr bool procrustes_k = kind_ == nk_kernel_wassersteinprocrustes_packed_k;
    constexpr nk_capability_t capabilities = nk_cap_metal_k | nk_cap_apple10_k;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<metal_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 5, columns = 7, largest = points + 3;
    nk_size_t bytes = 0, serial_bytes = 0;
    if constexpr (procrustes_k) {
        stats.expect(nk_wassersteinprocrustes_workspace_size_f16_best(largest, largest, capabilities, &bytes));
        stats.expect(nk_wassersteinprocrustes_workspace_size_f16_serial(largest, largest, &serial_bytes));
    }
    else {
        stats.expect(nk_gromovwassersteins_workspace_size_f16_best(largest, largest, capabilities, &bytes));
        stats.expect(nk_gromovwassersteins_workspace_size_f16_serial(largest, largest, &serial_bytes));
    }
    auto workspace = make_vector<std::byte>(backend, bytes);
    auto serial_workspace = make_vector<std::byte>(serial_bytes);
    auto results = make_vector<f32_t>(backend, rows * columns);
    auto rotations = make_vector<f32_t>(backend, rows * columns * 9);
    auto swapped_results = make_vector<f32_t>(backend, rows * columns);
    auto swapped_rotations = make_vector<f32_t>(backend, rows * columns * 9);
    auto const launch = [&](auto const &row_packs, std::size_t row_count, auto const &column_packs,
                            std::size_t column_count, nk_f32_t *distances, nk_f32_t *turns, std::byte *memory,
                            nk_size_t memory_bytes) {
        auto const kernel = [](auto... arguments) noexcept {
            if constexpr (procrustes_k)
                return call_best<nk_wassersteinprocrustes_packed_f16_best>(capabilities, arguments...);
            else return call_best<nk_gromovwassersteins_packed_f16_best>(capabilities, arguments...);
        };
        return kernel(row_packs.data(), row_count, column_packs.data(), column_count, distances,
                      column_count * sizeof(nk_f32_t), turns, static_cast<void *>(memory), memory_bytes,
                      backend.stream);
    };
    std::vector<nk_f32_t> expected(rows * columns), expected_rotations(rows * columns * 9);
    nk_f64_t const roundoff = half_ulp<f32_t>(1.0), terms = static_cast<nk_f64_t>(points + largest);
    for (transport_layout_metal_t const layout :
         {transport_layout_metal_t::random_k, transport_layout_metal_t::lattice_k, transport_layout_metal_t::repeated_k,
          transport_layout_metal_t::ring_k}) {
        auto const row_packs = transport_packs_metal(settings, backend, generator, layout, rows, points, stats);
        auto const column_packs = transport_packs_metal(settings, backend, generator, layout, columns, largest, stats);
        nk_f32_t *distances = results.raw_values_data(), *turns = rotations.raw_values_data();
        nk_f32_t *swapped_distances = swapped_results.raw_values_data();
        nk_f32_t *swapped_turns = swapped_rotations.raw_values_data();
        std::byte *memory = workspace.raw_values_data();
        stats.expect(
            launch(row_packs, rows, column_packs, columns, distances, turns, memory + 1, bytes) == nk_misaligned_k,
            "a misaligned workspace passed");
        stats.expect(
            launch(row_packs, rows, column_packs, columns, distances, turns, memory, 1) == nk_unexpected_dimensions_k,
            "a workspace below one slot passed");
        // A workspace of 6 slots walks the 35 pairs in launches that reuse them
        nk_size_t const slot_bytes = procrustes_k ? nk_wassersteinprocrustes_slot_bytes_metal_(largest)
                                                  : nk_gromovwassersteins_slot_bytes_metal_(largest);
        nk_size_t const memory_bytes = nk::min_of(bytes, 6 * slot_bytes);
        stats.expect(launch(row_packs, rows, column_packs, columns, distances, turns, memory, memory_bytes));
        if (!stats.expect(backend.synchronize())) return stats;
        stats.expect(
            launch(column_packs, columns, row_packs, rows, swapped_distances, swapped_turns, memory, memory_bytes));
        if (!stats.expect(backend.synchronize())) return stats;
        if constexpr (procrustes_k)
            stats.expect(nk_wassersteinprocrustes_packed_f16_serial(
                row_packs.data(), rows, column_packs.data(), columns, expected.data(), columns * sizeof(nk_f32_t),
                expected_rotations.data(), serial_workspace.raw_values_data(), serial_bytes, nullptr));
        else
            stats.expect(nk_gromovwassersteins_packed_f16_serial(
                row_packs.data(), rows, column_packs.data(), columns, expected.data(), columns * sizeof(nk_f32_t),
                expected_rotations.data(), serial_workspace.raw_values_data(), serial_bytes, nullptr));
        for (std::size_t row = 0; row != rows; ++row)
            for (std::size_t column = 0; column != columns; ++column) {
                auto const radius = [](void const *packed) {
                    return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
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
                nk_f32_t const distance = distances[row * columns + column];
                stats.accumulate_bounded(f32_t::from_raw(distance), f64_t(reference), bound);
                stats.expect(std::memcmp(&distance, swapped_distances + column * rows + row, sizeof(distance)) == 0,
                             "swapping the sides changed a distance");
                for (std::size_t cell = 0; cell != 9; ++cell)
                    stats.expect(std::memcmp(turns + (row * columns + column) * 9 + cell,
                                             swapped_turns + (column * rows + row) * 9 + cell % 3 * 3 + cell / 3,
                                             sizeof(nk_f32_t)) == 0,
                                 "swapping the sides left the rotation untransposed");
            }
    }
    return stats;
}

void test_shapes_metal(error_stats_section_t &check) {
    check.section("Shapes Metal", nk_cap_metal_k);
    check("rmsds_packed_f16_metal", test_rmsds_packed_f16<metal_backend_t>,
          [](auto... arguments) noexcept { return call_best<nk_rmsds_packed_f16_best>(nk_cap_metal_k, arguments...); });
    check("hausdorff_simdgroup_packed_f16_metal", test_shapes_simdgroup_metal<nk_kernel_hausdorffs_packed_k>);
    check("hausdorffs_limits_f16_metal", test_shapes_limits_metal<nk_kernel_hausdorffs_packed_k>, nk_cap_metal_k);
    check("chamfer_simdgroup_packed_f16_metal", test_shapes_simdgroup_metal<nk_kernel_chamfers_packed_k>);
    check("chamfers_limits_f16_metal", test_shapes_limits_metal<nk_kernel_chamfers_packed_k>, nk_cap_metal_k);
    check("wassersteinprocrustes_packed_f16_metal",
          test_transport_packed_metal<nk_kernel_wassersteinprocrustes_packed_k>);
    check("gromovwassersteins_packed_f16_metal", test_transport_packed_metal<nk_kernel_gromovwassersteins_packed_k>);
    check.section("Shapes Apple10", nk_cap_apple10_k);
    check("hausdorffs_limits_f16_apple10", test_shapes_limits_metal<nk_kernel_hausdorffs_packed_k>, nk_cap_apple10_k);
    check("chamfers_limits_f16_apple10", test_shapes_limits_metal<nk_kernel_chamfers_packed_k>, nk_cap_apple10_k);
}

#else  // !NUMKONG_ARCH_METAL_
void test_shapes_metal(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_METAL_

} // namespace ashvardanian::numkong::test
