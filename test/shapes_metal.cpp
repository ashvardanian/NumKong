/**
 *  @file test/shapes_metal.cpp
 *  @author Ash Vardanian
 *  @date October 24, 2026
 *  @brief RMSD and transport exports through the shared device scenarios, and the transport
 *      workspace's slots, refusals and side swaps.
 *
 *  The slots come from the Metal header's helpers, so this unit keeps the @c metal exports of that
 *  header switched off, as the library already defines them.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#undef NUMKONG_TARGET_METAL
#define NUMKONG_TARGET_METAL 0
#include "cross_device.hpp" // `test_rmsds_packed`, `device_transport_packs`

#if NUMKONG_ARCH_METAL_
#include "numkong/shapes/metal.h" // `nk_wassersteinprocrustes_slot_bytes_metal_`
#endif

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_METAL_

/** The Metal transport workspace: a misaligned one and one below a slot are refused, and one of 6
 *  slots walks the 35 pairs in launches that reuse them, with sides swapped keeping the distances
 *  and transposing the rotations. */
template <nk_kernel_kind_t kind_>
error_stats_t test_transport_packed_metal() {
    constexpr bool procrustes_k = kind_ == nk_kernel_wassersteinprocrustes_packed_k;
    constexpr nk_capability_t capabilities = nk_cap_metal_k | nk_cap_apple10_k;
    error_stats_t stats(comparison_family_t::exact_k);
    auto backend = make_backend<metal_backend_t>();
    split_mix_generator_t generator(case_stream_key());
    std::size_t const points = test_run.settings.shape_points, rows = 5, columns = 7, largest = points + 3;
    nk_size_t bytes = 0;
    stats.expect_status(procrustes_k
                            ? nk_wassersteinprocrustes_workspace_size_f16_best(largest, largest, capabilities, &bytes)
                            : nk_gromovwassersteins_workspace_size_f16_best(largest, largest, capabilities, &bytes));
    auto workspace = make_vector<std::byte>(backend, bytes);
    auto results = make_vector<f32_t>(backend, 2 * rows * columns);
    auto rotations = make_vector<f32_t>(backend, 2 * rows * columns * 9);
    nk_f32_t *distances = results.raw_values_data(), *swapped_distances = distances + rows * columns;
    nk_f32_t *turns = rotations.raw_values_data(), *swapped_turns = turns + rows * columns * 9;
    auto const launch = [&](auto const &row_packs, std::size_t row_count, auto const &column_packs,
                            std::size_t column_count, nk_f32_t *scores, nk_f32_t *flips, std::byte *memory,
                            nk_size_t memory_bytes) {
        auto const kernel = procrustes_k ? nk_wassersteinprocrustes_packed_f16_best
                                         : nk_gromovwassersteins_packed_f16_best;
        return kernel(row_packs.data(), row_count, column_packs.data(), column_count, scores,
                      column_count * sizeof(nk_f32_t), flips, static_cast<void *>(memory), memory_bytes, capabilities,
                      backend.stream);
    };
    auto const row_packs = device_transport_packs<f16_t>(backend, generator, transport_layout_t::random_k, rows, points,
                                                         stats);
    auto const column_packs = device_transport_packs<f16_t>(backend, generator, transport_layout_t::random_k, columns,
                                                            largest, stats);
    std::byte *memory = workspace.raw_values_data();
    stats.expect_true(
        launch(row_packs, rows, column_packs, columns, distances, turns, memory + 1, bytes) == nk_misaligned_k,
        "a misaligned workspace passed");
    stats.expect_true(
        launch(row_packs, rows, column_packs, columns, distances, turns, memory, 1) == nk_unexpected_dimensions_k,
        "a workspace below one slot passed");
    nk_size_t const slot_bytes = procrustes_k ? nk_wassersteinprocrustes_slot_bytes_metal_(largest)
                                              : nk_gromovwassersteins_slot_bytes_metal_(largest);
    nk_size_t const memory_bytes = nk::min_of(bytes, 6 * slot_bytes);
    stats.expect_status(launch(row_packs, rows, column_packs, columns, distances, turns, memory, memory_bytes));
    stats.expect_status(
        launch(column_packs, columns, row_packs, rows, swapped_distances, swapped_turns, memory, memory_bytes));
    if (!stats.expect_status(backend.synchronize())) return stats;
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            stats.expect_true(std::memcmp(distances + row * columns + column, swapped_distances + column * rows + row,
                                          sizeof(nk_f32_t)) == 0,
                              "swapping the sides changed a distance");
            for (std::size_t cell = 0; cell != 9; ++cell)
                stats.expect_true(std::memcmp(turns + (row * columns + column) * 9 + cell,
                                              swapped_turns + (column * rows + row) * 9 + cell % 3 * 3 + cell / 3,
                                              sizeof(nk_f32_t)) == 0,
                                  "swapping the sides left the rotation untransposed");
        }
    return stats;
}

/** The dispatch point @p best_ in the shape of its capability kernels, over the Metal baseline and
 *  its Apple10 kits, so a transport's workspace query and kernel pick the same kit. */
template <auto best_>
static constexpr auto shapes_best =
    [](auto... arguments) noexcept { return call_best<best_>(nk_cap_metal_k | nk_cap_apple10_k, arguments...); };

void run_shapes_metal_cases() {
    auto const metal = nk_cap_metal_k;
    run_kernel_case("test_rmsds_packed<f16_t, metal_backend_t>(nk_rmsds_packed_f16_best)",
                    test_rmsds_packed<f16_t, metal_backend_t>, nk_cap_metal_k, [](auto... arguments) noexcept {
                        return call_best<nk_rmsds_packed_f16_best>(nk_cap_metal_k, arguments...);
                    });
    run_kernel_case("test_transport_packed<f16_t, nk_kernel_wassersteinprocrustes_packed_k, metal_backend_t>",
                    test_transport_packed<f16_t, nk_kernel_wassersteinprocrustes_packed_k, metal_backend_t>, metal,
                    shapes_best<nk_wassersteinprocrustes_workspace_size_f16_best>,
                    shapes_best<nk_wassersteinprocrustes_packed_f16_best>);
    run_kernel_case("test_transport_packed<f16_t, nk_kernel_gromovwassersteins_packed_k, metal_backend_t>",
                    test_transport_packed<f16_t, nk_kernel_gromovwassersteins_packed_k, metal_backend_t>, metal,
                    shapes_best<nk_gromovwassersteins_workspace_size_f16_best>,
                    shapes_best<nk_gromovwassersteins_packed_f16_best>);
    run_kernel_case("test_transport_packed_metal<nk_kernel_wassersteinprocrustes_packed_k>",
                    test_transport_packed_metal<nk_kernel_wassersteinprocrustes_packed_k>, metal);
    run_kernel_case("test_transport_packed_metal<nk_kernel_gromovwassersteins_packed_k>",
                    test_transport_packed_metal<nk_kernel_gromovwassersteins_packed_k>, metal);
}

#else  // !NUMKONG_ARCH_METAL_
void run_shapes_metal_cases() {}
#endif // NUMKONG_ARCH_METAL_

} // namespace ashvardanian::numkong::test
