/**
 *  @file bench/shapes.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Many-to-many Chamfer, Hausdorff, Wasserstein-Procrustes, Gromov-Wasserstein benchmarks.
 */

#include "numkong/shapes.h"

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

/**
 *  @brief Measures a many-to-many shape kernel over @p rows × @p columns clouds of @p points each.
 *  @param[inout] loop The timed loop, which takes the counters.
 *  @param[in] workspace_size The transport kinds' workspace query, null for Chamfer and Hausdorff.
 */
template <nk_dtype_t input_dtype_, typename kernel_type_>
void measure_shapes(loop_t &loop, environment_t const &env,
                    typename nk::type_for<input_dtype_>::type::shapes_pack_size_kernel_t pack_size,
                    typename nk::type_for<input_dtype_>::type::shapes_pack_kernel_t pack,
                    typename nk::type_for<input_dtype_>::type::shapes_workspace_size_kernel_t workspace_size,
                    kernel_type_ kernel, std::size_t rows, std::size_t columns, std::size_t points) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using distance_t = typename input_t::shapes_distance_t;

    nk_size_t pack_bytes = 0, workspace_bytes = 0;
    if (!succeeded(loop, pack_size(points, &pack_bytes))) return;
    if (workspace_size && !succeeded(loop, workspace_size(points, points, &workspace_bytes))) return;

    // Each set holds the row packs, then the column packs, back to back in one buffer
    std::size_t const clouds = rows + columns;
    std::size_t const pack_stride = nk_size_round_up_to_multiple_(pack_bytes, nk_default_alignment_k);
    std::size_t const sets_count = input_sets_count(bytes_t {clouds * pack_stride});
    std::vector<nk::vector<char>> sets(sets_count);
    std::vector<void const *> pointers(sets_count * clouds);
    std::mt19937 generator(env.settings.seed.value);
    auto cloud = make_vector<input_t>(points * 3);
    for (std::size_t set = 0; set != sets_count; ++set) {
        sets[set] = make_vector<char>(clouds * pack_stride);
        for (std::size_t index = 0; index != clouds; ++index) {
            nk::fill_uniform(generator, cloud.values_data(), cloud.size_values());
            char *packed = sets[set].raw_values_data() + index * pack_stride;
            if (!succeeded(loop, pack(cloud.raw_values_data(), points, packed, nullptr))) return;
            pointers[set * clouds + index] = packed;
        }
    }
    auto workspace = make_vector<char>(std::max<nk_size_t>(workspace_bytes, 1));
    auto results = make_vector<distance_t>(rows * columns);
    std::size_t const results_stride = columns * sizeof(typename distance_t::raw_t);

    for (std::size_t call : loop) {
        void const *const *set = pointers.data() + (call & (sets_count - 1)) * clouds;
        nk_status_t status;
        if constexpr (std::is_same_v<kernel_type_, typename input_t::shapes_packed_workspace_kernel_t>)
            status = kernel(set, rows, set + rows, columns, results.raw_values_data(), results_stride, nullptr,
                            workspace.raw_values_data(), workspace_bytes, nullptr);
        else
            status = kernel(set, rows, set + rows, columns, results.raw_values_data(), results_stride, nullptr,
                            nullptr);
        if (!succeeded(loop, status)) break;
        do_not_optimize(results.values_data()[0]);
    }

    loop.rate("distances", static_cast<double>(rows * columns));
}

/** Runs @p kernel on one row cloud against many, and on a square tile of clouds. */
template <nk_dtype_t input_dtype_, typename kernel_type_>
void run_shapes(environment_t const &env, std::string const &name,
                typename nk::type_for<input_dtype_>::type::shapes_pack_size_kernel_t pack_size,
                typename nk::type_for<input_dtype_>::type::shapes_pack_kernel_t pack,
                typename nk::type_for<input_dtype_>::type::shapes_workspace_size_kernel_t workspace_size,
                kernel_type_ kernel) {
    std::size_t const points = env.settings.shape_points;
    std::size_t const tiles[][2] = {{1, 16}, {8, 8}};
    for (auto const &tile : tiles) {
        std::string const bench_name = name + "<" + std::to_string(tile[0]) + "x" + std::to_string(tile[1]) + "x" +
                                       std::to_string(points) + ">";
        run_benchmark(env, bench_name, measure_shapes<input_dtype_, kernel_type_>, pack_size, pack, workspace_size,
                      kernel, tile[0], tile[1], points);
    }
}

void bench_shapes(environment_t const &env) {
    constexpr nk_dtype_t f32_k = nk_f32_k, f16_k = nk_f16_k;

    section(env, "Shapes Serial", nk_cap_serial_k);
    run_shapes<f32_k>(env, "chamfers_packed_f32_serial", nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial,
                      nullptr, nk_chamfers_packed_f32_serial);
    run_shapes<f16_k>(env, "chamfers_packed_f16_serial", nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
                      nullptr, nk_chamfers_packed_f16_serial);
    run_shapes<f32_k>(env, "hausdorffs_packed_f32_serial", nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial,
                      nullptr, nk_hausdorffs_packed_f32_serial);
    run_shapes<f16_k>(env, "hausdorffs_packed_f16_serial", nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
                      nullptr, nk_hausdorffs_packed_f16_serial);
    run_shapes<f32_k>(env, "wassersteinprocrustes_packed_f32_serial", nk_shapes_pack_size_f32_serial,
                      nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
                      nk_wassersteinprocrustes_packed_f32_serial);
    run_shapes<f16_k>(env, "wassersteinprocrustes_packed_f16_serial", nk_shapes_pack_size_f16_serial,
                      nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
                      nk_wassersteinprocrustes_packed_f16_serial);
    run_shapes<f32_k>(env, "gromovwassersteins_packed_f32_serial", nk_shapes_pack_size_f32_serial,
                      nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
                      nk_gromovwassersteins_packed_f32_serial);
    run_shapes<f16_k>(env, "gromovwassersteins_packed_f16_serial", nk_shapes_pack_size_f16_serial,
                      nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
                      nk_gromovwassersteins_packed_f16_serial);

#if NUMKONG_TARGET_NEON
    if (section(env, "Shapes NEON", nk_cap_neon_k)) {
        run_shapes<f32_k>(env, "chamfers_packed_f32_neon", nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial,
                          nullptr, nk_chamfers_packed_f32_neon);
        run_shapes<f16_k>(env, "chamfers_packed_f16_neon", nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
                          nullptr, nk_chamfers_packed_f16_neon);
        run_shapes<f32_k>(env, "hausdorffs_packed_f32_neon", nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial,
                          nullptr, nk_hausdorffs_packed_f32_neon);
        run_shapes<f16_k>(env, "hausdorffs_packed_f16_neon", nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
                          nullptr, nk_hausdorffs_packed_f16_neon);
        run_shapes<f32_k>(env, "wassersteinprocrustes_packed_f32_neon", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
                          nk_wassersteinprocrustes_packed_f32_neon);
        run_shapes<f16_k>(env, "wassersteinprocrustes_packed_f16_neon", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
                          nk_wassersteinprocrustes_packed_f16_neon);
        run_shapes<f32_k>(env, "gromovwassersteins_packed_f32_neon", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
                          nk_gromovwassersteins_packed_f32_neon);
        run_shapes<f16_k>(env, "gromovwassersteins_packed_f16_neon", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
                          nk_gromovwassersteins_packed_f16_neon);
    }
#endif

#if NUMKONG_TARGET_NEONFHM
    if (section(env, "Shapes NEON FHM", nk_cap_neonfhm_k)) {
        run_shapes<f16_k>(env, "chamfers_packed_f16_neonfhm", nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
                          nullptr, nk_chamfers_packed_f16_neonfhm);
        run_shapes<f16_k>(env, "hausdorffs_packed_f16_neonfhm", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nullptr, nk_hausdorffs_packed_f16_neonfhm);
    }
#endif

#if NUMKONG_TARGET_HASWELL
    if (section(env, "Shapes Haswell", nk_cap_haswell_k)) {
        run_shapes<f32_k>(env, "chamfers_packed_f32_haswell", nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial,
                          nullptr, nk_chamfers_packed_f32_haswell);
        run_shapes<f16_k>(env, "chamfers_packed_f16_haswell", nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
                          nullptr, nk_chamfers_packed_f16_haswell);
        run_shapes<f32_k>(env, "hausdorffs_packed_f32_haswell", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nullptr, nk_hausdorffs_packed_f32_haswell);
        run_shapes<f16_k>(env, "hausdorffs_packed_f16_haswell", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nullptr, nk_hausdorffs_packed_f16_haswell);
        run_shapes<f32_k>(env, "wassersteinprocrustes_packed_f32_haswell", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
                          nk_wassersteinprocrustes_packed_f32_haswell);
        run_shapes<f16_k>(env, "wassersteinprocrustes_packed_f16_haswell", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
                          nk_wassersteinprocrustes_packed_f16_haswell);
        run_shapes<f32_k>(env, "gromovwassersteins_packed_f32_haswell", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
                          nk_gromovwassersteins_packed_f32_haswell);
        run_shapes<f16_k>(env, "gromovwassersteins_packed_f16_haswell", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
                          nk_gromovwassersteins_packed_f16_haswell);
    }
#endif

#if NUMKONG_TARGET_SKYLAKE
    if (section(env, "Shapes Skylake", nk_cap_skylake_k)) {
        run_shapes<f32_k>(env, "chamfers_packed_f32_skylake", nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial,
                          nullptr, nk_chamfers_packed_f32_skylake);
        run_shapes<f16_k>(env, "chamfers_packed_f16_skylake", nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
                          nullptr, nk_chamfers_packed_f16_skylake);
        run_shapes<f32_k>(env, "hausdorffs_packed_f32_skylake", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nullptr, nk_hausdorffs_packed_f32_skylake);
        run_shapes<f16_k>(env, "hausdorffs_packed_f16_skylake", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nullptr, nk_hausdorffs_packed_f16_skylake);
        run_shapes<f32_k>(env, "wassersteinprocrustes_packed_f32_skylake", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
                          nk_wassersteinprocrustes_packed_f32_skylake);
        run_shapes<f16_k>(env, "wassersteinprocrustes_packed_f16_skylake", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
                          nk_wassersteinprocrustes_packed_f16_skylake);
        run_shapes<f32_k>(env, "gromovwassersteins_packed_f32_skylake", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
                          nk_gromovwassersteins_packed_f32_skylake);
        run_shapes<f16_k>(env, "gromovwassersteins_packed_f16_skylake", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
                          nk_gromovwassersteins_packed_f16_skylake);
    }
#endif

#if NUMKONG_TARGET_RVV
    if (section(env, "Shapes RVV", nk_cap_rvv_k)) {
        run_shapes<f32_k>(env, "chamfers_packed_f32_rvv", nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial,
                          nullptr, nk_chamfers_packed_f32_rvv);
        run_shapes<f16_k>(env, "chamfers_packed_f16_rvv", nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
                          nullptr, nk_chamfers_packed_f16_rvv);
        run_shapes<f32_k>(env, "hausdorffs_packed_f32_rvv", nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial,
                          nullptr, nk_hausdorffs_packed_f32_rvv);
        run_shapes<f16_k>(env, "hausdorffs_packed_f16_rvv", nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
                          nullptr, nk_hausdorffs_packed_f16_rvv);
        run_shapes<f32_k>(env, "wassersteinprocrustes_packed_f32_rvv", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
                          nk_wassersteinprocrustes_packed_f32_rvv);
        run_shapes<f16_k>(env, "wassersteinprocrustes_packed_f16_rvv", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
                          nk_wassersteinprocrustes_packed_f16_rvv);
        run_shapes<f32_k>(env, "gromovwassersteins_packed_f32_rvv", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
                          nk_gromovwassersteins_packed_f32_rvv);
        run_shapes<f16_k>(env, "gromovwassersteins_packed_f16_rvv", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
                          nk_gromovwassersteins_packed_f16_rvv);
    }
#endif

#if NUMKONG_TARGET_V128RELAXED
    if (section(env, "Shapes V128 Relaxed", nk_cap_v128relaxed_k)) {
        run_shapes<f32_k>(env, "chamfers_packed_f32_v128relaxed", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nullptr, nk_chamfers_packed_f32_v128relaxed);
        run_shapes<f16_k>(env, "chamfers_packed_f16_v128relaxed", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nullptr, nk_chamfers_packed_f16_v128relaxed);
        run_shapes<f32_k>(env, "hausdorffs_packed_f32_v128relaxed", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nullptr, nk_hausdorffs_packed_f32_v128relaxed);
        run_shapes<f16_k>(env, "hausdorffs_packed_f16_v128relaxed", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nullptr, nk_hausdorffs_packed_f16_v128relaxed);
        run_shapes<f32_k>(env, "wassersteinprocrustes_packed_f32_v128relaxed", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
                          nk_wassersteinprocrustes_packed_f32_v128relaxed);
        run_shapes<f16_k>(env, "wassersteinprocrustes_packed_f16_v128relaxed", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
                          nk_wassersteinprocrustes_packed_f16_v128relaxed);
        run_shapes<f32_k>(env, "gromovwassersteins_packed_f32_v128relaxed", nk_shapes_pack_size_f32_serial,
                          nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
                          nk_gromovwassersteins_packed_f32_v128relaxed);
        run_shapes<f16_k>(env, "gromovwassersteins_packed_f16_v128relaxed", nk_shapes_pack_size_f16_serial,
                          nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
                          nk_gromovwassersteins_packed_f16_v128relaxed);
    }
#endif
}

} // namespace ashvardanian::numkong::bench
