/**
 *  @file bench/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batch operation benchmarks for the Metal kernels, the twin of `cross_cuda.cu`.
 *
 *  Runs the drivers of `cross.hpp` through a @c metal_backend_t, on an owned queue on the selected
 *  device, over its unified memory. Metal has no C-level events, so every window of launches is
 *  timed by wall clock through the synchronization after it. Input sets rotate until their
 *  footprint is at least twice the system-level cache.
 */

#include "cross.hpp"
#include "harness_metal.hpp"

namespace ashvardanian::numkong::bench {

#if NUMKONG_ARCH_METAL_

using rmsds_kernel_metal_t = nk_status_t (*)(void const *const *, nk_size_t, void const *const *, nk_size_t, nk_f32_t *,
                                             nk_size_t, nk_stream_t);

/** Times an F16 shape @p kernel over a @p side × @p side tile of clouds of @p points each, with
 *  each set's packs, their pointer tables and its outputs in the queue's shared memory. */
template <typename kernel_type_>
void measure_shapes_metal(loop_t &loop, environment_t const &env, metal_backend_t backend, kernel_type_ kernel,
                          std::size_t side, std::size_t points) {
    using packs_t = nk::vector<std::byte, metal_backend_t::allocator<std::byte>>;
    using outputs_t = nk::vector<f32_t, metal_backend_t::allocator<f32_t>>;
    struct input_set {
        packs_t packs;
        outputs_t outputs;
    };
    nk_size_t pack_bytes = 0;
    if (!succeeded(loop, nk_shapes_pack_size_f16_serial(points, &pack_bytes))) return;
    std::size_t const clouds = 2 * side,
                      pack_stride = nk_size_round_up_to_multiple_(pack_bytes, nk_default_alignment_k);
    std::size_t const packs_bytes = clouds * pack_stride, table_bytes = clouds * sizeof(void const *);
    std::vector<input_set> sets(
        backend.input_sets(bytes_t {packs_bytes + table_bytes + side * side * sizeof(nk_f32_t)}));
    std::mt19937 generator(env.settings.seed.value);
    auto cloud = make_vector<f16_t>(points * 3);
    for (input_set &set : sets) {
        set.packs = packs_t::uninitialized(packs_bytes + table_bytes, allocator_of<std::byte>(backend)).value;
        set.outputs = outputs_t::uninitialized(side * side, allocator_of<f32_t>(backend)).value;
        if (set.packs.empty() || set.outputs.empty()) return loop.skip("set allocation failed");
        void const **table = reinterpret_cast<void const **>(set.packs.raw_values_data() + packs_bytes);
        for (std::size_t index = 0; index != clouds; ++index) {
            nk::fill_uniform(generator, cloud.values_data(), cloud.size_values());
            std::byte *packed = set.packs.raw_values_data() + index * pack_stride;
            if (!succeeded(loop, nk_shapes_pack_f16_serial(cloud.raw_values_data(), points, packed, nullptr))) return;
            table[index] = packed;
        }
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        input_set &set = sets[index];
        void const **table = reinterpret_cast<void const **>(set.packs.raw_values_data() + packs_bytes);
        std::size_t const results_stride = side * sizeof(nk_f32_t);
        if constexpr (std::is_same_v<kernel_type_, rmsds_kernel_metal_t>)
            return backend.call(kernel, table, side, table + side, side, set.outputs.raw_values_data(), results_stride);
        else
            return backend.call(kernel, table, side, table + side, side, set.outputs.raw_values_data(), results_stride,
                                static_cast<nk_f32_t *>(nullptr));
    });
    if (timed) loop.rate("distances", static_cast<double>(side * side));
}

/** Runs @p kernel on the 256 × 256 tile that keeps a GPU busy. */
template <typename kernel_type_>
void run_shapes_metal(environment_t const &env, std::string const &name, metal_backend_t const &backend,
                      kernel_type_ kernel) {
    std::size_t const side = 256, points = env.settings.shape_points;
    std::string const bench_name = name + "<" + std::to_string(side) + "x" + std::to_string(side) + "x" +
                                   std::to_string(points) + ">";
    run_benchmark(env, bench_name, measure_shapes_metal<kernel_type_>, backend, kernel, side, points);
}

#endif // NUMKONG_ARCH_METAL_

/** Every Metal baseline entry point beside every Apple9 and Apple10 one, so the matrix units'
 *  speedup shows, on devices whose families include each. */
nk::status_t bench_cross_metal([[maybe_unused]] environment_t const &env,
                               [[maybe_unused]] device_backend_t const &runtime) {
#if NUMKONG_ARCH_METAL_
    nk_capability_t const enabled = runtime.capabilities;
    metal_backend_t const backend {runtime};
    if (enabled & nk_cap_metal_k) {
        run_dots_packed<nk_nvfp4_k>(env, "dots_packed_nvfp4_metal", nk_dots_pack_size_nvfp4_metal,
                                    nk_dots_pack_nvfp4_metal, nk_dots_packed_nvfp4_metal, backend);
        run_dots_symmetric<nk_nvfp4_k>(env, "dots_symmetric_nvfp4_metal", nk_dots_symmetric_nvfp4_metal, backend);
        run_angulars_packed<nk_nvfp4_k>(env, "angulars_packed_nvfp4_metal", nk_dots_pack_size_nvfp4_metal,
                                        nk_dots_pack_nvfp4_metal, nk_angulars_packed_nvfp4_metal, backend);
        run_angulars_symmetric<nk_nvfp4_k>(env, "angulars_symmetric_nvfp4_metal", nk_angulars_symmetric_nvfp4_metal,
                                           backend);
        run_euclideans_packed<nk_nvfp4_k>(env, "euclideans_packed_nvfp4_metal", nk_dots_pack_size_nvfp4_metal,
                                          nk_dots_pack_nvfp4_metal, nk_euclideans_packed_nvfp4_metal, backend);
        run_euclideans_symmetric<nk_nvfp4_k>(env, "euclideans_symmetric_nvfp4_metal",
                                             nk_euclideans_symmetric_nvfp4_metal, backend);
        run_dots_packed<nk_mxfp4_k>(env, "dots_packed_mxfp4_metal", nk_dots_pack_size_mxfp4_metal,
                                    nk_dots_pack_mxfp4_metal, nk_dots_packed_mxfp4_metal, backend);
        run_dots_symmetric<nk_mxfp4_k>(env, "dots_symmetric_mxfp4_metal", nk_dots_symmetric_mxfp4_metal, backend);
        run_angulars_packed<nk_mxfp4_k>(env, "angulars_packed_mxfp4_metal", nk_dots_pack_size_mxfp4_metal,
                                        nk_dots_pack_mxfp4_metal, nk_angulars_packed_mxfp4_metal, backend);
        run_angulars_symmetric<nk_mxfp4_k>(env, "angulars_symmetric_mxfp4_metal", nk_angulars_symmetric_mxfp4_metal,
                                           backend);
        run_euclideans_packed<nk_mxfp4_k>(env, "euclideans_packed_mxfp4_metal", nk_dots_pack_size_mxfp4_metal,
                                          nk_dots_pack_mxfp4_metal, nk_euclideans_packed_mxfp4_metal, backend);
        run_euclideans_symmetric<nk_mxfp4_k>(env, "euclideans_symmetric_mxfp4_metal",
                                             nk_euclideans_symmetric_mxfp4_metal, backend);
        run_dots_packed<nk_mxfp6e2m3_k>(env, "dots_packed_mxfp6e2m3_metal", nk_dots_pack_size_mxfp6e2m3_metal,
                                        nk_dots_pack_mxfp6e2m3_metal, nk_dots_packed_mxfp6e2m3_metal, backend);
        run_dots_symmetric<nk_mxfp6e2m3_k>(env, "dots_symmetric_mxfp6e2m3_metal", nk_dots_symmetric_mxfp6e2m3_metal,
                                           backend);
        run_angulars_packed<nk_mxfp6e2m3_k>(env, "angulars_packed_mxfp6e2m3_metal", nk_dots_pack_size_mxfp6e2m3_metal,
                                            nk_dots_pack_mxfp6e2m3_metal, nk_angulars_packed_mxfp6e2m3_metal, backend);
        run_angulars_symmetric<nk_mxfp6e2m3_k>(env, "angulars_symmetric_mxfp6e2m3_metal",
                                               nk_angulars_symmetric_mxfp6e2m3_metal, backend);
        run_euclideans_packed<nk_mxfp6e2m3_k>(env, "euclideans_packed_mxfp6e2m3_metal",
                                              nk_dots_pack_size_mxfp6e2m3_metal, nk_dots_pack_mxfp6e2m3_metal,
                                              nk_euclideans_packed_mxfp6e2m3_metal, backend);
        run_euclideans_symmetric<nk_mxfp6e2m3_k>(env, "euclideans_symmetric_mxfp6e2m3_metal",
                                                 nk_euclideans_symmetric_mxfp6e2m3_metal, backend);
        run_dots_packed<nk_mxfp6e3m2_k>(env, "dots_packed_mxfp6e3m2_metal", nk_dots_pack_size_mxfp6e3m2_metal,
                                        nk_dots_pack_mxfp6e3m2_metal, nk_dots_packed_mxfp6e3m2_metal, backend);
        run_dots_symmetric<nk_mxfp6e3m2_k>(env, "dots_symmetric_mxfp6e3m2_metal", nk_dots_symmetric_mxfp6e3m2_metal,
                                           backend);
        run_angulars_packed<nk_mxfp6e3m2_k>(env, "angulars_packed_mxfp6e3m2_metal", nk_dots_pack_size_mxfp6e3m2_metal,
                                            nk_dots_pack_mxfp6e3m2_metal, nk_angulars_packed_mxfp6e3m2_metal, backend);
        run_angulars_symmetric<nk_mxfp6e3m2_k>(env, "angulars_symmetric_mxfp6e3m2_metal",
                                               nk_angulars_symmetric_mxfp6e3m2_metal, backend);
        run_euclideans_packed<nk_mxfp6e3m2_k>(env, "euclideans_packed_mxfp6e3m2_metal",
                                              nk_dots_pack_size_mxfp6e3m2_metal, nk_dots_pack_mxfp6e3m2_metal,
                                              nk_euclideans_packed_mxfp6e3m2_metal, backend);
        run_euclideans_symmetric<nk_mxfp6e3m2_k>(env, "euclideans_symmetric_mxfp6e3m2_metal",
                                                 nk_euclideans_symmetric_mxfp6e3m2_metal, backend);
        run_dots_packed<nk_mxfp8e4m3_k>(env, "dots_packed_mxfp8e4m3_metal", nk_dots_pack_size_mxfp8e4m3_metal,
                                        nk_dots_pack_mxfp8e4m3_metal, nk_dots_packed_mxfp8e4m3_metal, backend);
        run_dots_symmetric<nk_mxfp8e4m3_k>(env, "dots_symmetric_mxfp8e4m3_metal", nk_dots_symmetric_mxfp8e4m3_metal,
                                           backend);
        run_angulars_packed<nk_mxfp8e4m3_k>(env, "angulars_packed_mxfp8e4m3_metal", nk_dots_pack_size_mxfp8e4m3_metal,
                                            nk_dots_pack_mxfp8e4m3_metal, nk_angulars_packed_mxfp8e4m3_metal, backend);
        run_angulars_symmetric<nk_mxfp8e4m3_k>(env, "angulars_symmetric_mxfp8e4m3_metal",
                                               nk_angulars_symmetric_mxfp8e4m3_metal, backend);
        run_euclideans_packed<nk_mxfp8e4m3_k>(env, "euclideans_packed_mxfp8e4m3_metal",
                                              nk_dots_pack_size_mxfp8e4m3_metal, nk_dots_pack_mxfp8e4m3_metal,
                                              nk_euclideans_packed_mxfp8e4m3_metal, backend);
        run_euclideans_symmetric<nk_mxfp8e4m3_k>(env, "euclideans_symmetric_mxfp8e4m3_metal",
                                                 nk_euclideans_symmetric_mxfp8e4m3_metal, backend);
        run_dots_packed<nk_mxfp8e5m2_k>(env, "dots_packed_mxfp8e5m2_metal", nk_dots_pack_size_mxfp8e5m2_metal,
                                        nk_dots_pack_mxfp8e5m2_metal, nk_dots_packed_mxfp8e5m2_metal, backend);
        run_dots_symmetric<nk_mxfp8e5m2_k>(env, "dots_symmetric_mxfp8e5m2_metal", nk_dots_symmetric_mxfp8e5m2_metal,
                                           backend);
        run_angulars_packed<nk_mxfp8e5m2_k>(env, "angulars_packed_mxfp8e5m2_metal", nk_dots_pack_size_mxfp8e5m2_metal,
                                            nk_dots_pack_mxfp8e5m2_metal, nk_angulars_packed_mxfp8e5m2_metal, backend);
        run_angulars_symmetric<nk_mxfp8e5m2_k>(env, "angulars_symmetric_mxfp8e5m2_metal",
                                               nk_angulars_symmetric_mxfp8e5m2_metal, backend);
        run_euclideans_packed<nk_mxfp8e5m2_k>(env, "euclideans_packed_mxfp8e5m2_metal",
                                              nk_dots_pack_size_mxfp8e5m2_metal, nk_dots_pack_mxfp8e5m2_metal,
                                              nk_euclideans_packed_mxfp8e5m2_metal, backend);
        run_euclideans_symmetric<nk_mxfp8e5m2_k>(env, "euclideans_symmetric_mxfp8e5m2_metal",
                                                 nk_euclideans_symmetric_mxfp8e5m2_metal, backend);
        run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_metal", nk_dots_pack_size_bf16_metal, nk_dots_pack_bf16_metal,
                                   nk_dots_packed_bf16_metal, backend);
        run_dots_packed<nk_f16_k>(env, "dots_packed_f16_metal", nk_dots_pack_size_f16_metal, nk_dots_pack_f16_metal,
                                  nk_dots_packed_f16_metal, backend);
        run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_metal", nk_dots_pack_size_e5m2_metal, nk_dots_pack_e5m2_metal,
                                   nk_dots_packed_e5m2_metal, backend);
        run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_metal", nk_dots_pack_size_e4m3_metal, nk_dots_pack_e4m3_metal,
                                   nk_dots_packed_e4m3_metal, backend);
        run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_metal", nk_dots_pack_size_e3m2_metal, nk_dots_pack_e3m2_metal,
                                   nk_dots_packed_e3m2_metal, backend);
        run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_metal", nk_dots_pack_size_e2m3_metal, nk_dots_pack_e2m3_metal,
                                   nk_dots_packed_e2m3_metal, backend);
        run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_metal", nk_dots_pack_size_e2m1_metal, nk_dots_pack_e2m1_metal,
                                   nk_dots_packed_e2m1_metal, backend);
        run_dots_packed<nk_i8_k>(env, "dots_packed_i8_metal", nk_dots_pack_size_i8_metal, nk_dots_pack_i8_metal,
                                 nk_dots_packed_i8_metal, backend);
        run_dots_packed<nk_u8_k>(env, "dots_packed_u8_metal", nk_dots_pack_size_u8_metal, nk_dots_pack_u8_metal,
                                 nk_dots_packed_u8_metal, backend);

        run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_metal", nk_dots_symmetric_bf16_metal, backend);
        run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_metal", nk_dots_symmetric_f16_metal, backend);
        run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_metal", nk_dots_symmetric_e4m3_metal, backend);
        run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_metal", nk_dots_symmetric_i8_metal, backend);
        run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_metal", nk_dots_pack_size_i8_metal, nk_dots_pack_i8_metal,
                                     nk_angulars_packed_i8_metal, backend);
        run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_metal", nk_angulars_symmetric_i8_metal, backend);
        run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_metal", nk_dots_pack_size_i8_metal,
                                       nk_dots_pack_i8_metal, nk_euclideans_packed_i8_metal, backend);
        run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_metal", nk_euclideans_symmetric_i8_metal,
                                          backend);
        run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_metal", nk_dots_pack_size_u8_metal, nk_dots_pack_u8_metal,
                                     nk_angulars_packed_u8_metal, backend);
        run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_metal", nk_angulars_symmetric_u8_metal, backend);
        run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_metal", nk_dots_pack_size_u8_metal,
                                       nk_dots_pack_u8_metal, nk_euclideans_packed_u8_metal, backend);
        run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_metal", nk_euclideans_symmetric_u8_metal,
                                          backend);
        run_dots_packed<nk_i4_k>(env, "dots_packed_i4_metal", nk_dots_pack_size_i4_metal, nk_dots_pack_i4_metal,
                                 nk_dots_packed_i4_metal, backend);
        run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_metal", nk_dots_symmetric_i4_metal, backend);
        run_dots_packed<nk_u4_k>(env, "dots_packed_u4_metal", nk_dots_pack_size_u4_metal, nk_dots_pack_u4_metal,
                                 nk_dots_packed_u4_metal, backend);
        run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_metal", nk_dots_symmetric_u4_metal, backend);
        run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_metal", nk_dots_pack_size_i4_metal, nk_dots_pack_i4_metal,
                                     nk_angulars_packed_i4_metal, backend);
        run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_metal", nk_angulars_symmetric_i4_metal, backend);
        run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_metal", nk_dots_pack_size_i4_metal,
                                       nk_dots_pack_i4_metal, nk_euclideans_packed_i4_metal, backend);
        run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_metal", nk_euclideans_symmetric_i4_metal,
                                          backend);
        run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_metal", nk_dots_pack_size_u4_metal, nk_dots_pack_u4_metal,
                                     nk_angulars_packed_u4_metal, backend);
        run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_metal", nk_angulars_symmetric_u4_metal, backend);
        run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_metal", nk_dots_pack_size_u4_metal,
                                       nk_dots_pack_u4_metal, nk_euclideans_packed_u4_metal, backend);
        run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_metal", nk_euclideans_symmetric_u4_metal,
                                          backend);
        run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_metal", nk_dots_pack_size_f16_metal,
                                      nk_dots_pack_f16_metal, nk_angulars_packed_f16_metal, backend);
        run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_metal", nk_angulars_symmetric_f16_metal, backend);
        run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_metal", nk_dots_pack_size_f16_metal,
                                        nk_dots_pack_f16_metal, nk_euclideans_packed_f16_metal, backend);
        run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_metal", nk_euclideans_symmetric_f16_metal,
                                           backend);
        run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_metal", nk_dots_pack_size_bf16_metal,
                                       nk_dots_pack_bf16_metal, nk_angulars_packed_bf16_metal, backend);
        run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_metal", nk_angulars_symmetric_bf16_metal,
                                          backend);
        run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_metal", nk_dots_pack_size_bf16_metal,
                                         nk_dots_pack_bf16_metal, nk_euclideans_packed_bf16_metal, backend);
        run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_metal", nk_euclideans_symmetric_bf16_metal,
                                            backend);
        run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_metal", nk_dots_pack_size_e4m3_metal,
                                       nk_dots_pack_e4m3_metal, nk_angulars_packed_e4m3_metal, backend);
        run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_metal", nk_angulars_symmetric_e4m3_metal,
                                          backend);
        run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_metal", nk_dots_pack_size_e4m3_metal,
                                         nk_dots_pack_e4m3_metal, nk_euclideans_packed_e4m3_metal, backend);
        run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_metal", nk_euclideans_symmetric_e4m3_metal,
                                            backend);
        run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_metal", nk_dots_pack_size_e5m2_metal,
                                       nk_dots_pack_e5m2_metal, nk_angulars_packed_e5m2_metal, backend);
        run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_metal", nk_angulars_symmetric_e5m2_metal,
                                          backend);
        run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_metal", nk_dots_pack_size_e5m2_metal,
                                         nk_dots_pack_e5m2_metal, nk_euclideans_packed_e5m2_metal, backend);
        run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_metal", nk_euclideans_symmetric_e5m2_metal,
                                            backend);
        run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_metal", nk_dots_pack_size_e3m2_metal,
                                       nk_dots_pack_e3m2_metal, nk_angulars_packed_e3m2_metal, backend);
        run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_metal", nk_angulars_symmetric_e3m2_metal,
                                          backend);
        run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_metal", nk_dots_pack_size_e3m2_metal,
                                         nk_dots_pack_e3m2_metal, nk_euclideans_packed_e3m2_metal, backend);
        run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_metal", nk_euclideans_symmetric_e3m2_metal,
                                            backend);
        run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_metal", nk_dots_pack_size_e2m3_metal,
                                       nk_dots_pack_e2m3_metal, nk_angulars_packed_e2m3_metal, backend);
        run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_metal", nk_angulars_symmetric_e2m3_metal,
                                          backend);
        run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_metal", nk_dots_pack_size_e2m3_metal,
                                         nk_dots_pack_e2m3_metal, nk_euclideans_packed_e2m3_metal, backend);
        run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_metal", nk_euclideans_symmetric_e2m3_metal,
                                            backend);
        run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_metal", nk_dots_pack_size_e2m1_metal,
                                       nk_dots_pack_e2m1_metal, nk_angulars_packed_e2m1_metal, backend);
        run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_metal", nk_angulars_symmetric_e2m1_metal,
                                          backend);
        run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_metal", nk_dots_pack_size_e2m1_metal,
                                         nk_dots_pack_e2m1_metal, nk_euclideans_packed_e2m1_metal, backend);
        run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_metal", nk_euclideans_symmetric_e2m1_metal,
                                            backend);
        run_attention<nk_bf16_k>(env, "attention_packed_bf16_metal", nk_attention_pack_size_bf16_metal,
                                 nk_attention_pack_bf16_metal, nk_attention_packed_bf16_metal, backend);
        run_attention<nk_f16_k>(env, "attention_packed_f16_metal", nk_attention_pack_size_f16_metal,
                                nk_attention_pack_f16_metal, nk_attention_packed_f16_metal, backend);
        run_attention<nk_e4m3_k>(env, "attention_packed_e4m3_metal", nk_attention_pack_size_e4m3_metal,
                                 nk_attention_pack_e4m3_metal, nk_attention_packed_e4m3_metal, backend);
        run_attention<nk_i8_k>(env, "attention_packed_i8_metal", nk_attention_pack_size_i8_metal,
                               nk_attention_pack_i8_metal, nk_attention_packed_i8_metal, backend);
        run_attention<nk_nvfp4_k>(env, "attention_packed_nvfp4_metal", nk_attention_pack_size_nvfp4_metal,
                                  nk_attention_pack_nvfp4_metal, nk_attention_packed_nvfp4_metal, backend);
        run_attention<nk_mxfp4_k>(env, "attention_packed_mxfp4_metal", nk_attention_pack_size_mxfp4_metal,
                                  nk_attention_pack_mxfp4_metal, nk_attention_packed_mxfp4_metal, backend);
        run_attention<nk_mxfp6e2m3_k>(env, "attention_packed_mxfp6e2m3_metal", nk_attention_pack_size_mxfp6e2m3_metal,
                                      nk_attention_pack_mxfp6e2m3_metal, nk_attention_packed_mxfp6e2m3_metal, backend);
        run_attention<nk_mxfp6e3m2_k>(env, "attention_packed_mxfp6e3m2_metal", nk_attention_pack_size_mxfp6e3m2_metal,
                                      nk_attention_pack_mxfp6e3m2_metal, nk_attention_packed_mxfp6e3m2_metal, backend);
        run_attention<nk_mxfp8e4m3_k>(env, "attention_packed_mxfp8e4m3_metal", nk_attention_pack_size_mxfp8e4m3_metal,
                                      nk_attention_pack_mxfp8e4m3_metal, nk_attention_packed_mxfp8e4m3_metal, backend);
        run_attention<nk_mxfp8e5m2_k>(env, "attention_packed_mxfp8e5m2_metal", nk_attention_pack_size_mxfp8e5m2_metal,
                                      nk_attention_pack_mxfp8e5m2_metal, nk_attention_packed_mxfp8e5m2_metal, backend);
        run_attention_gradients<nk_bf16_k>(env, "attention_packed_gradients_bf16_metal",
                                           nk_attention_pack_size_bf16_metal, nk_attention_pack_bf16_metal,
                                           nk_attention_packed_bf16_metal, nk_attention_packed_gradients_bf16_metal,
                                           backend);
        run_attention_rope<nk_f32_k>(env, "attention_rope_f32_metal", nk_attention_rope_f32_metal, backend);
        run_attention_rope<nk_bf16_k>(env, "attention_rope_bf16_metal", nk_attention_rope_bf16_metal, backend);
        run_attention_rope<nk_e4m3_k>(env, "attention_rope_e4m3_metal", nk_attention_rope_e4m3_metal, backend);
        run_shapes_metal(env, "chamfers_packed_f16_metal", backend, nk_chamfers_packed_f16_metal);
        run_shapes_metal(env, "hausdorffs_packed_f16_metal", backend, nk_hausdorffs_packed_f16_metal);
    }
#if NUMKONG_TARGET_APPLE9
    if (enabled & nk_cap_apple9_k) {
        run_dots_packed<nk_nvfp4_k>(env, "dots_packed_nvfp4_apple9", nk_dots_pack_size_nvfp4_apple9,
                                    nk_dots_pack_nvfp4_apple9, nk_dots_packed_nvfp4_apple9, backend);
        run_dots_symmetric<nk_nvfp4_k>(env, "dots_symmetric_nvfp4_apple9", nk_dots_symmetric_nvfp4_apple9, backend);
        run_angulars_packed<nk_nvfp4_k>(env, "angulars_packed_nvfp4_apple9", nk_dots_pack_size_nvfp4_apple9,
                                        nk_dots_pack_nvfp4_apple9, nk_angulars_packed_nvfp4_apple9, backend);
        run_angulars_symmetric<nk_nvfp4_k>(env, "angulars_symmetric_nvfp4_apple9", nk_angulars_symmetric_nvfp4_apple9,
                                           backend);
        run_euclideans_packed<nk_nvfp4_k>(env, "euclideans_packed_nvfp4_apple9", nk_dots_pack_size_nvfp4_apple9,
                                          nk_dots_pack_nvfp4_apple9, nk_euclideans_packed_nvfp4_apple9, backend);
        run_euclideans_symmetric<nk_nvfp4_k>(env, "euclideans_symmetric_nvfp4_apple9",
                                             nk_euclideans_symmetric_nvfp4_apple9, backend);
        run_dots_packed<nk_mxfp4_k>(env, "dots_packed_mxfp4_apple9", nk_dots_pack_size_mxfp4_apple9,
                                    nk_dots_pack_mxfp4_apple9, nk_dots_packed_mxfp4_apple9, backend);
        run_dots_symmetric<nk_mxfp4_k>(env, "dots_symmetric_mxfp4_apple9", nk_dots_symmetric_mxfp4_apple9, backend);
        run_angulars_packed<nk_mxfp4_k>(env, "angulars_packed_mxfp4_apple9", nk_dots_pack_size_mxfp4_apple9,
                                        nk_dots_pack_mxfp4_apple9, nk_angulars_packed_mxfp4_apple9, backend);
        run_angulars_symmetric<nk_mxfp4_k>(env, "angulars_symmetric_mxfp4_apple9", nk_angulars_symmetric_mxfp4_apple9,
                                           backend);
        run_euclideans_packed<nk_mxfp4_k>(env, "euclideans_packed_mxfp4_apple9", nk_dots_pack_size_mxfp4_apple9,
                                          nk_dots_pack_mxfp4_apple9, nk_euclideans_packed_mxfp4_apple9, backend);
        run_euclideans_symmetric<nk_mxfp4_k>(env, "euclideans_symmetric_mxfp4_apple9",
                                             nk_euclideans_symmetric_mxfp4_apple9, backend);
        run_dots_packed<nk_mxfp6e2m3_k>(env, "dots_packed_mxfp6e2m3_apple9", nk_dots_pack_size_mxfp6e2m3_apple9,
                                        nk_dots_pack_mxfp6e2m3_apple9, nk_dots_packed_mxfp6e2m3_apple9, backend);
        run_dots_symmetric<nk_mxfp6e2m3_k>(env, "dots_symmetric_mxfp6e2m3_apple9", nk_dots_symmetric_mxfp6e2m3_apple9,
                                           backend);
        run_angulars_packed<nk_mxfp6e2m3_k>(env, "angulars_packed_mxfp6e2m3_apple9", nk_dots_pack_size_mxfp6e2m3_apple9,
                                            nk_dots_pack_mxfp6e2m3_apple9, nk_angulars_packed_mxfp6e2m3_apple9,
                                            backend);
        run_angulars_symmetric<nk_mxfp6e2m3_k>(env, "angulars_symmetric_mxfp6e2m3_apple9",
                                               nk_angulars_symmetric_mxfp6e2m3_apple9, backend);
        run_euclideans_packed<nk_mxfp6e2m3_k>(env, "euclideans_packed_mxfp6e2m3_apple9",
                                              nk_dots_pack_size_mxfp6e2m3_apple9, nk_dots_pack_mxfp6e2m3_apple9,
                                              nk_euclideans_packed_mxfp6e2m3_apple9, backend);
        run_euclideans_symmetric<nk_mxfp6e2m3_k>(env, "euclideans_symmetric_mxfp6e2m3_apple9",
                                                 nk_euclideans_symmetric_mxfp6e2m3_apple9, backend);
        run_dots_packed<nk_mxfp6e3m2_k>(env, "dots_packed_mxfp6e3m2_apple9", nk_dots_pack_size_mxfp6e3m2_apple9,
                                        nk_dots_pack_mxfp6e3m2_apple9, nk_dots_packed_mxfp6e3m2_apple9, backend);
        run_dots_symmetric<nk_mxfp6e3m2_k>(env, "dots_symmetric_mxfp6e3m2_apple9", nk_dots_symmetric_mxfp6e3m2_apple9,
                                           backend);
        run_angulars_packed<nk_mxfp6e3m2_k>(env, "angulars_packed_mxfp6e3m2_apple9", nk_dots_pack_size_mxfp6e3m2_apple9,
                                            nk_dots_pack_mxfp6e3m2_apple9, nk_angulars_packed_mxfp6e3m2_apple9,
                                            backend);
        run_angulars_symmetric<nk_mxfp6e3m2_k>(env, "angulars_symmetric_mxfp6e3m2_apple9",
                                               nk_angulars_symmetric_mxfp6e3m2_apple9, backend);
        run_euclideans_packed<nk_mxfp6e3m2_k>(env, "euclideans_packed_mxfp6e3m2_apple9",
                                              nk_dots_pack_size_mxfp6e3m2_apple9, nk_dots_pack_mxfp6e3m2_apple9,
                                              nk_euclideans_packed_mxfp6e3m2_apple9, backend);
        run_euclideans_symmetric<nk_mxfp6e3m2_k>(env, "euclideans_symmetric_mxfp6e3m2_apple9",
                                                 nk_euclideans_symmetric_mxfp6e3m2_apple9, backend);
        run_dots_packed<nk_mxfp8e4m3_k>(env, "dots_packed_mxfp8e4m3_apple9", nk_dots_pack_size_mxfp8e4m3_apple9,
                                        nk_dots_pack_mxfp8e4m3_apple9, nk_dots_packed_mxfp8e4m3_apple9, backend);
        run_dots_symmetric<nk_mxfp8e4m3_k>(env, "dots_symmetric_mxfp8e4m3_apple9", nk_dots_symmetric_mxfp8e4m3_apple9,
                                           backend);
        run_angulars_packed<nk_mxfp8e4m3_k>(env, "angulars_packed_mxfp8e4m3_apple9", nk_dots_pack_size_mxfp8e4m3_apple9,
                                            nk_dots_pack_mxfp8e4m3_apple9, nk_angulars_packed_mxfp8e4m3_apple9,
                                            backend);
        run_angulars_symmetric<nk_mxfp8e4m3_k>(env, "angulars_symmetric_mxfp8e4m3_apple9",
                                               nk_angulars_symmetric_mxfp8e4m3_apple9, backend);
        run_euclideans_packed<nk_mxfp8e4m3_k>(env, "euclideans_packed_mxfp8e4m3_apple9",
                                              nk_dots_pack_size_mxfp8e4m3_apple9, nk_dots_pack_mxfp8e4m3_apple9,
                                              nk_euclideans_packed_mxfp8e4m3_apple9, backend);
        run_euclideans_symmetric<nk_mxfp8e4m3_k>(env, "euclideans_symmetric_mxfp8e4m3_apple9",
                                                 nk_euclideans_symmetric_mxfp8e4m3_apple9, backend);
        run_dots_packed<nk_mxfp8e5m2_k>(env, "dots_packed_mxfp8e5m2_apple9", nk_dots_pack_size_mxfp8e5m2_apple9,
                                        nk_dots_pack_mxfp8e5m2_apple9, nk_dots_packed_mxfp8e5m2_apple9, backend);
        run_dots_symmetric<nk_mxfp8e5m2_k>(env, "dots_symmetric_mxfp8e5m2_apple9", nk_dots_symmetric_mxfp8e5m2_apple9,
                                           backend);
        run_angulars_packed<nk_mxfp8e5m2_k>(env, "angulars_packed_mxfp8e5m2_apple9", nk_dots_pack_size_mxfp8e5m2_apple9,
                                            nk_dots_pack_mxfp8e5m2_apple9, nk_angulars_packed_mxfp8e5m2_apple9,
                                            backend);
        run_angulars_symmetric<nk_mxfp8e5m2_k>(env, "angulars_symmetric_mxfp8e5m2_apple9",
                                               nk_angulars_symmetric_mxfp8e5m2_apple9, backend);
        run_euclideans_packed<nk_mxfp8e5m2_k>(env, "euclideans_packed_mxfp8e5m2_apple9",
                                              nk_dots_pack_size_mxfp8e5m2_apple9, nk_dots_pack_mxfp8e5m2_apple9,
                                              nk_euclideans_packed_mxfp8e5m2_apple9, backend);
        run_euclideans_symmetric<nk_mxfp8e5m2_k>(env, "euclideans_symmetric_mxfp8e5m2_apple9",
                                                 nk_euclideans_symmetric_mxfp8e5m2_apple9, backend);
        run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_apple9", nk_dots_pack_size_bf16_apple9,
                                   nk_dots_pack_bf16_apple9, nk_dots_packed_bf16_apple9, backend);
        run_dots_packed<nk_f16_k>(env, "dots_packed_f16_apple9", nk_dots_pack_size_f16_apple9, nk_dots_pack_f16_apple9,
                                  nk_dots_packed_f16_apple9, backend);
        run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_apple9", nk_dots_pack_size_e5m2_apple9,
                                   nk_dots_pack_e5m2_apple9, nk_dots_packed_e5m2_apple9, backend);
        run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_apple9", nk_dots_pack_size_e4m3_apple9,
                                   nk_dots_pack_e4m3_apple9, nk_dots_packed_e4m3_apple9, backend);
        run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_apple9", nk_dots_pack_size_e3m2_apple9,
                                   nk_dots_pack_e3m2_apple9, nk_dots_packed_e3m2_apple9, backend);
        run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_apple9", nk_dots_pack_size_e2m3_apple9,
                                   nk_dots_pack_e2m3_apple9, nk_dots_packed_e2m3_apple9, backend);
        run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_apple9", nk_dots_pack_size_e2m1_apple9,
                                   nk_dots_pack_e2m1_apple9, nk_dots_packed_e2m1_apple9, backend);

        run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_apple9", nk_dots_symmetric_bf16_apple9, backend);
        run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_apple9", nk_dots_symmetric_f16_apple9, backend);
        run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_apple9", nk_dots_symmetric_e4m3_apple9, backend);
        run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_apple9", nk_dots_pack_size_f16_apple9,
                                      nk_dots_pack_f16_apple9, nk_angulars_packed_f16_apple9, backend);
        run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_apple9", nk_angulars_symmetric_f16_apple9,
                                         backend);
        run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_apple9", nk_dots_pack_size_f16_apple9,
                                        nk_dots_pack_f16_apple9, nk_euclideans_packed_f16_apple9, backend);
        run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_apple9", nk_euclideans_symmetric_f16_apple9,
                                           backend);
        run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_apple9", nk_dots_pack_size_bf16_apple9,
                                       nk_dots_pack_bf16_apple9, nk_angulars_packed_bf16_apple9, backend);
        run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_apple9", nk_angulars_symmetric_bf16_apple9,
                                          backend);
        run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_apple9", nk_dots_pack_size_bf16_apple9,
                                         nk_dots_pack_bf16_apple9, nk_euclideans_packed_bf16_apple9, backend);
        run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_apple9",
                                            nk_euclideans_symmetric_bf16_apple9, backend);
        run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_apple9", nk_dots_pack_size_e4m3_apple9,
                                       nk_dots_pack_e4m3_apple9, nk_angulars_packed_e4m3_apple9, backend);
        run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_apple9", nk_angulars_symmetric_e4m3_apple9,
                                          backend);
        run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_apple9", nk_dots_pack_size_e4m3_apple9,
                                         nk_dots_pack_e4m3_apple9, nk_euclideans_packed_e4m3_apple9, backend);
        run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_apple9",
                                            nk_euclideans_symmetric_e4m3_apple9, backend);
        run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_apple9", nk_dots_pack_size_e5m2_apple9,
                                       nk_dots_pack_e5m2_apple9, nk_angulars_packed_e5m2_apple9, backend);
        run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_apple9", nk_angulars_symmetric_e5m2_apple9,
                                          backend);
        run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_apple9", nk_dots_pack_size_e5m2_apple9,
                                         nk_dots_pack_e5m2_apple9, nk_euclideans_packed_e5m2_apple9, backend);
        run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_apple9",
                                            nk_euclideans_symmetric_e5m2_apple9, backend);
        run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_apple9", nk_dots_pack_size_e3m2_apple9,
                                       nk_dots_pack_e3m2_apple9, nk_angulars_packed_e3m2_apple9, backend);
        run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_apple9", nk_angulars_symmetric_e3m2_apple9,
                                          backend);
        run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_apple9", nk_dots_pack_size_e3m2_apple9,
                                         nk_dots_pack_e3m2_apple9, nk_euclideans_packed_e3m2_apple9, backend);
        run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_apple9",
                                            nk_euclideans_symmetric_e3m2_apple9, backend);
        run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_apple9", nk_dots_pack_size_e2m3_apple9,
                                       nk_dots_pack_e2m3_apple9, nk_angulars_packed_e2m3_apple9, backend);
        run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_apple9", nk_angulars_symmetric_e2m3_apple9,
                                          backend);
        run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_apple9", nk_dots_pack_size_e2m3_apple9,
                                         nk_dots_pack_e2m3_apple9, nk_euclideans_packed_e2m3_apple9, backend);
        run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_apple9",
                                            nk_euclideans_symmetric_e2m3_apple9, backend);
        run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_apple9", nk_dots_pack_size_e2m1_apple9,
                                       nk_dots_pack_e2m1_apple9, nk_angulars_packed_e2m1_apple9, backend);
        run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_apple9", nk_angulars_symmetric_e2m1_apple9,
                                          backend);
        run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_apple9", nk_dots_pack_size_e2m1_apple9,
                                         nk_dots_pack_e2m1_apple9, nk_euclideans_packed_e2m1_apple9, backend);
        run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_apple9",
                                            nk_euclideans_symmetric_e2m1_apple9, backend);
        run_attention<nk_bf16_k>(env, "attention_packed_bf16_apple9", nk_attention_pack_size_bf16_apple9,
                                 nk_attention_pack_bf16_apple9, nk_attention_packed_bf16_apple9, backend);
        run_attention<nk_f16_k>(env, "attention_packed_f16_apple9", nk_attention_pack_size_f16_apple9,
                                nk_attention_pack_f16_apple9, nk_attention_packed_f16_apple9, backend);
        run_attention<nk_e4m3_k>(env, "attention_packed_e4m3_apple9", nk_attention_pack_size_e4m3_apple9,
                                 nk_attention_pack_e4m3_apple9, nk_attention_packed_e4m3_apple9, backend);
        run_attention<nk_nvfp4_k>(env, "attention_packed_nvfp4_apple9", nk_attention_pack_size_nvfp4_apple9,
                                  nk_attention_pack_nvfp4_apple9, nk_attention_packed_nvfp4_apple9, backend);
        run_attention<nk_mxfp4_k>(env, "attention_packed_mxfp4_apple9", nk_attention_pack_size_mxfp4_apple9,
                                  nk_attention_pack_mxfp4_apple9, nk_attention_packed_mxfp4_apple9, backend);
        run_attention<nk_mxfp6e2m3_k>(env, "attention_packed_mxfp6e2m3_apple9", nk_attention_pack_size_mxfp6e2m3_apple9,
                                      nk_attention_pack_mxfp6e2m3_apple9, nk_attention_packed_mxfp6e2m3_apple9,
                                      backend);
        run_attention<nk_mxfp6e3m2_k>(env, "attention_packed_mxfp6e3m2_apple9", nk_attention_pack_size_mxfp6e3m2_apple9,
                                      nk_attention_pack_mxfp6e3m2_apple9, nk_attention_packed_mxfp6e3m2_apple9,
                                      backend);
        run_attention<nk_mxfp8e4m3_k>(env, "attention_packed_mxfp8e4m3_apple9", nk_attention_pack_size_mxfp8e4m3_apple9,
                                      nk_attention_pack_mxfp8e4m3_apple9, nk_attention_packed_mxfp8e4m3_apple9,
                                      backend);
        run_attention<nk_mxfp8e5m2_k>(env, "attention_packed_mxfp8e5m2_apple9", nk_attention_pack_size_mxfp8e5m2_apple9,
                                      nk_attention_pack_mxfp8e5m2_apple9, nk_attention_packed_mxfp8e5m2_apple9,
                                      backend);
        run_attention_gradients<nk_bf16_k>(env, "attention_packed_gradients_bf16_apple9",
                                           nk_attention_pack_size_bf16_apple9, nk_attention_pack_bf16_apple9,
                                           nk_attention_packed_bf16_apple9, nk_attention_packed_gradients_bf16_apple9,
                                           backend);
    }
#endif // NUMKONG_TARGET_APPLE9
#if NUMKONG_TARGET_APPLE10
    if (!(enabled & nk_cap_apple10_k)) return nk::status_t::success_k;
    run_shapes_metal(env, "chamfers_packed_f16_apple10", backend, nk_chamfers_packed_f16_apple10);
    run_shapes_metal(env, "hausdorffs_packed_f16_apple10", backend, nk_hausdorffs_packed_f16_apple10);
    run_shapes_metal(env, "rmsds_packed_f16_apple10", backend, nk_rmsds_packed_f16_apple10);
    run_dots_packed<nk_nvfp4_k>(env, "dots_packed_nvfp4_apple10", nk_dots_pack_size_nvfp4_apple10,
                                nk_dots_pack_nvfp4_apple10, nk_dots_packed_nvfp4_apple10, backend);
    run_dots_symmetric<nk_nvfp4_k>(env, "dots_symmetric_nvfp4_apple10", nk_dots_symmetric_nvfp4_apple10, backend);
    run_angulars_packed<nk_nvfp4_k>(env, "angulars_packed_nvfp4_apple10", nk_dots_pack_size_nvfp4_apple10,
                                    nk_dots_pack_nvfp4_apple10, nk_angulars_packed_nvfp4_apple10, backend);
    run_angulars_symmetric<nk_nvfp4_k>(env, "angulars_symmetric_nvfp4_apple10", nk_angulars_symmetric_nvfp4_apple10,
                                       backend);
    run_euclideans_packed<nk_nvfp4_k>(env, "euclideans_packed_nvfp4_apple10", nk_dots_pack_size_nvfp4_apple10,
                                      nk_dots_pack_nvfp4_apple10, nk_euclideans_packed_nvfp4_apple10, backend);
    run_euclideans_symmetric<nk_nvfp4_k>(env, "euclideans_symmetric_nvfp4_apple10",
                                         nk_euclideans_symmetric_nvfp4_apple10, backend);
    run_dots_packed<nk_mxfp4_k>(env, "dots_packed_mxfp4_apple10", nk_dots_pack_size_mxfp4_apple10,
                                nk_dots_pack_mxfp4_apple10, nk_dots_packed_mxfp4_apple10, backend);
    run_dots_symmetric<nk_mxfp4_k>(env, "dots_symmetric_mxfp4_apple10", nk_dots_symmetric_mxfp4_apple10, backend);
    run_angulars_packed<nk_mxfp4_k>(env, "angulars_packed_mxfp4_apple10", nk_dots_pack_size_mxfp4_apple10,
                                    nk_dots_pack_mxfp4_apple10, nk_angulars_packed_mxfp4_apple10, backend);
    run_angulars_symmetric<nk_mxfp4_k>(env, "angulars_symmetric_mxfp4_apple10", nk_angulars_symmetric_mxfp4_apple10,
                                       backend);
    run_euclideans_packed<nk_mxfp4_k>(env, "euclideans_packed_mxfp4_apple10", nk_dots_pack_size_mxfp4_apple10,
                                      nk_dots_pack_mxfp4_apple10, nk_euclideans_packed_mxfp4_apple10, backend);
    run_euclideans_symmetric<nk_mxfp4_k>(env, "euclideans_symmetric_mxfp4_apple10",
                                         nk_euclideans_symmetric_mxfp4_apple10, backend);
    run_dots_packed<nk_mxfp6e2m3_k>(env, "dots_packed_mxfp6e2m3_apple10", nk_dots_pack_size_mxfp6e2m3_apple10,
                                    nk_dots_pack_mxfp6e2m3_apple10, nk_dots_packed_mxfp6e2m3_apple10, backend);
    run_dots_symmetric<nk_mxfp6e2m3_k>(env, "dots_symmetric_mxfp6e2m3_apple10", nk_dots_symmetric_mxfp6e2m3_apple10,
                                       backend);
    run_angulars_packed<nk_mxfp6e2m3_k>(env, "angulars_packed_mxfp6e2m3_apple10", nk_dots_pack_size_mxfp6e2m3_apple10,
                                        nk_dots_pack_mxfp6e2m3_apple10, nk_angulars_packed_mxfp6e2m3_apple10, backend);
    run_angulars_symmetric<nk_mxfp6e2m3_k>(env, "angulars_symmetric_mxfp6e2m3_apple10",
                                           nk_angulars_symmetric_mxfp6e2m3_apple10, backend);
    run_euclideans_packed<nk_mxfp6e2m3_k>(env, "euclideans_packed_mxfp6e2m3_apple10",
                                          nk_dots_pack_size_mxfp6e2m3_apple10, nk_dots_pack_mxfp6e2m3_apple10,
                                          nk_euclideans_packed_mxfp6e2m3_apple10, backend);
    run_euclideans_symmetric<nk_mxfp6e2m3_k>(env, "euclideans_symmetric_mxfp6e2m3_apple10",
                                             nk_euclideans_symmetric_mxfp6e2m3_apple10, backend);
    run_dots_packed<nk_mxfp6e3m2_k>(env, "dots_packed_mxfp6e3m2_apple10", nk_dots_pack_size_mxfp6e3m2_apple10,
                                    nk_dots_pack_mxfp6e3m2_apple10, nk_dots_packed_mxfp6e3m2_apple10, backend);
    run_dots_symmetric<nk_mxfp6e3m2_k>(env, "dots_symmetric_mxfp6e3m2_apple10", nk_dots_symmetric_mxfp6e3m2_apple10,
                                       backend);
    run_angulars_packed<nk_mxfp6e3m2_k>(env, "angulars_packed_mxfp6e3m2_apple10", nk_dots_pack_size_mxfp6e3m2_apple10,
                                        nk_dots_pack_mxfp6e3m2_apple10, nk_angulars_packed_mxfp6e3m2_apple10, backend);
    run_angulars_symmetric<nk_mxfp6e3m2_k>(env, "angulars_symmetric_mxfp6e3m2_apple10",
                                           nk_angulars_symmetric_mxfp6e3m2_apple10, backend);
    run_euclideans_packed<nk_mxfp6e3m2_k>(env, "euclideans_packed_mxfp6e3m2_apple10",
                                          nk_dots_pack_size_mxfp6e3m2_apple10, nk_dots_pack_mxfp6e3m2_apple10,
                                          nk_euclideans_packed_mxfp6e3m2_apple10, backend);
    run_euclideans_symmetric<nk_mxfp6e3m2_k>(env, "euclideans_symmetric_mxfp6e3m2_apple10",
                                             nk_euclideans_symmetric_mxfp6e3m2_apple10, backend);
    run_dots_packed<nk_mxfp8e4m3_k>(env, "dots_packed_mxfp8e4m3_apple10", nk_dots_pack_size_mxfp8e4m3_apple10,
                                    nk_dots_pack_mxfp8e4m3_apple10, nk_dots_packed_mxfp8e4m3_apple10, backend);
    run_dots_symmetric<nk_mxfp8e4m3_k>(env, "dots_symmetric_mxfp8e4m3_apple10", nk_dots_symmetric_mxfp8e4m3_apple10,
                                       backend);
    run_angulars_packed<nk_mxfp8e4m3_k>(env, "angulars_packed_mxfp8e4m3_apple10", nk_dots_pack_size_mxfp8e4m3_apple10,
                                        nk_dots_pack_mxfp8e4m3_apple10, nk_angulars_packed_mxfp8e4m3_apple10, backend);
    run_angulars_symmetric<nk_mxfp8e4m3_k>(env, "angulars_symmetric_mxfp8e4m3_apple10",
                                           nk_angulars_symmetric_mxfp8e4m3_apple10, backend);
    run_euclideans_packed<nk_mxfp8e4m3_k>(env, "euclideans_packed_mxfp8e4m3_apple10",
                                          nk_dots_pack_size_mxfp8e4m3_apple10, nk_dots_pack_mxfp8e4m3_apple10,
                                          nk_euclideans_packed_mxfp8e4m3_apple10, backend);
    run_euclideans_symmetric<nk_mxfp8e4m3_k>(env, "euclideans_symmetric_mxfp8e4m3_apple10",
                                             nk_euclideans_symmetric_mxfp8e4m3_apple10, backend);
    run_dots_packed<nk_mxfp8e5m2_k>(env, "dots_packed_mxfp8e5m2_apple10", nk_dots_pack_size_mxfp8e5m2_apple10,
                                    nk_dots_pack_mxfp8e5m2_apple10, nk_dots_packed_mxfp8e5m2_apple10, backend);
    run_dots_symmetric<nk_mxfp8e5m2_k>(env, "dots_symmetric_mxfp8e5m2_apple10", nk_dots_symmetric_mxfp8e5m2_apple10,
                                       backend);
    run_angulars_packed<nk_mxfp8e5m2_k>(env, "angulars_packed_mxfp8e5m2_apple10", nk_dots_pack_size_mxfp8e5m2_apple10,
                                        nk_dots_pack_mxfp8e5m2_apple10, nk_angulars_packed_mxfp8e5m2_apple10, backend);
    run_angulars_symmetric<nk_mxfp8e5m2_k>(env, "angulars_symmetric_mxfp8e5m2_apple10",
                                           nk_angulars_symmetric_mxfp8e5m2_apple10, backend);
    run_euclideans_packed<nk_mxfp8e5m2_k>(env, "euclideans_packed_mxfp8e5m2_apple10",
                                          nk_dots_pack_size_mxfp8e5m2_apple10, nk_dots_pack_mxfp8e5m2_apple10,
                                          nk_euclideans_packed_mxfp8e5m2_apple10, backend);
    run_euclideans_symmetric<nk_mxfp8e5m2_k>(env, "euclideans_symmetric_mxfp8e5m2_apple10",
                                             nk_euclideans_symmetric_mxfp8e5m2_apple10, backend);
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_apple10", nk_dots_pack_size_bf16_apple10,
                               nk_dots_pack_bf16_apple10, nk_dots_packed_bf16_apple10, backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_apple10", nk_dots_pack_size_f16_apple10, nk_dots_pack_f16_apple10,
                              nk_dots_packed_f16_apple10, backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_apple10", nk_dots_pack_size_e5m2_apple10,
                               nk_dots_pack_e5m2_apple10, nk_dots_packed_e5m2_apple10, backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_apple10", nk_dots_pack_size_e4m3_apple10,
                               nk_dots_pack_e4m3_apple10, nk_dots_packed_e4m3_apple10, backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_apple10", nk_dots_pack_size_e3m2_apple10,
                               nk_dots_pack_e3m2_apple10, nk_dots_packed_e3m2_apple10, backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_apple10", nk_dots_pack_size_e2m3_apple10,
                               nk_dots_pack_e2m3_apple10, nk_dots_packed_e2m3_apple10, backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_apple10", nk_dots_pack_size_e2m1_apple10,
                               nk_dots_pack_e2m1_apple10, nk_dots_packed_e2m1_apple10, backend);
    run_dots_packed<nk_i4_k>(env, "dots_packed_i4_apple10", nk_dots_pack_size_i4_apple10, nk_dots_pack_i4_apple10,
                             nk_dots_packed_i4_apple10, backend);
    run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_apple10", nk_dots_symmetric_i4_apple10, backend);
    run_dots_packed<nk_u4_k>(env, "dots_packed_u4_apple10", nk_dots_pack_size_u4_apple10, nk_dots_pack_u4_apple10,
                             nk_dots_packed_u4_apple10, backend);
    run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_apple10", nk_dots_symmetric_u4_apple10, backend);
    run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_apple10", nk_dots_pack_size_i4_apple10,
                                 nk_dots_pack_i4_apple10, nk_angulars_packed_i4_apple10, backend);
    run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_apple10", nk_angulars_symmetric_i4_apple10, backend);
    run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_apple10", nk_dots_pack_size_i4_apple10,
                                   nk_dots_pack_i4_apple10, nk_euclideans_packed_i4_apple10, backend);
    run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_apple10", nk_euclideans_symmetric_i4_apple10,
                                      backend);
    run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_apple10", nk_dots_pack_size_u4_apple10,
                                 nk_dots_pack_u4_apple10, nk_angulars_packed_u4_apple10, backend);
    run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_apple10", nk_angulars_symmetric_u4_apple10, backend);
    run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_apple10", nk_dots_pack_size_u4_apple10,
                                   nk_dots_pack_u4_apple10, nk_euclideans_packed_u4_apple10, backend);
    run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_apple10", nk_euclideans_symmetric_u4_apple10,
                                      backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_apple10", nk_dots_pack_size_i8_apple10, nk_dots_pack_i8_apple10,
                             nk_dots_packed_i8_apple10, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_apple10", nk_dots_pack_size_u8_apple10, nk_dots_pack_u8_apple10,
                             nk_dots_packed_u8_apple10, backend);

    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_apple10", nk_dots_symmetric_bf16_apple10, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_apple10", nk_dots_symmetric_f16_apple10, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_apple10", nk_dots_symmetric_e4m3_apple10, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_apple10", nk_dots_symmetric_i8_apple10, backend);
    run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_apple10", nk_dots_pack_size_i8_apple10,
                                 nk_dots_pack_i8_apple10, nk_angulars_packed_i8_apple10, backend);
    run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_apple10", nk_angulars_symmetric_i8_apple10, backend);
    run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_apple10", nk_dots_pack_size_i8_apple10,
                                   nk_dots_pack_i8_apple10, nk_euclideans_packed_i8_apple10, backend);
    run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_apple10", nk_euclideans_symmetric_i8_apple10,
                                      backend);
    run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_apple10", nk_dots_pack_size_u8_apple10,
                                 nk_dots_pack_u8_apple10, nk_angulars_packed_u8_apple10, backend);
    run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_apple10", nk_angulars_symmetric_u8_apple10, backend);
    run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_apple10", nk_dots_pack_size_u8_apple10,
                                   nk_dots_pack_u8_apple10, nk_euclideans_packed_u8_apple10, backend);
    run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_apple10", nk_euclideans_symmetric_u8_apple10,
                                      backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_apple10", nk_dots_pack_size_f16_apple10,
                                  nk_dots_pack_f16_apple10, nk_angulars_packed_f16_apple10, backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_apple10", nk_angulars_symmetric_f16_apple10, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_apple10", nk_dots_pack_size_f16_apple10,
                                    nk_dots_pack_f16_apple10, nk_euclideans_packed_f16_apple10, backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_apple10", nk_euclideans_symmetric_f16_apple10,
                                       backend);
    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_apple10", nk_dots_pack_size_bf16_apple10,
                                   nk_dots_pack_bf16_apple10, nk_angulars_packed_bf16_apple10, backend);
    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_apple10", nk_angulars_symmetric_bf16_apple10,
                                      backend);
    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_apple10", nk_dots_pack_size_bf16_apple10,
                                     nk_dots_pack_bf16_apple10, nk_euclideans_packed_bf16_apple10, backend);
    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_apple10", nk_euclideans_symmetric_bf16_apple10,
                                        backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_apple10", nk_dots_pack_size_e4m3_apple10,
                                   nk_dots_pack_e4m3_apple10, nk_angulars_packed_e4m3_apple10, backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_apple10", nk_angulars_symmetric_e4m3_apple10,
                                      backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_apple10", nk_dots_pack_size_e4m3_apple10,
                                     nk_dots_pack_e4m3_apple10, nk_euclideans_packed_e4m3_apple10, backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_apple10", nk_euclideans_symmetric_e4m3_apple10,
                                        backend);
    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_apple10", nk_dots_pack_size_e5m2_apple10,
                                   nk_dots_pack_e5m2_apple10, nk_angulars_packed_e5m2_apple10, backend);
    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_apple10", nk_angulars_symmetric_e5m2_apple10,
                                      backend);
    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_apple10", nk_dots_pack_size_e5m2_apple10,
                                     nk_dots_pack_e5m2_apple10, nk_euclideans_packed_e5m2_apple10, backend);
    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_apple10", nk_euclideans_symmetric_e5m2_apple10,
                                        backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_apple10", nk_dots_pack_size_e3m2_apple10,
                                   nk_dots_pack_e3m2_apple10, nk_angulars_packed_e3m2_apple10, backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_apple10", nk_angulars_symmetric_e3m2_apple10,
                                      backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_apple10", nk_dots_pack_size_e3m2_apple10,
                                     nk_dots_pack_e3m2_apple10, nk_euclideans_packed_e3m2_apple10, backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_apple10", nk_euclideans_symmetric_e3m2_apple10,
                                        backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_apple10", nk_dots_pack_size_e2m3_apple10,
                                   nk_dots_pack_e2m3_apple10, nk_angulars_packed_e2m3_apple10, backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_apple10", nk_angulars_symmetric_e2m3_apple10,
                                      backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_apple10", nk_dots_pack_size_e2m3_apple10,
                                     nk_dots_pack_e2m3_apple10, nk_euclideans_packed_e2m3_apple10, backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_apple10", nk_euclideans_symmetric_e2m3_apple10,
                                        backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_apple10", nk_dots_pack_size_e2m1_apple10,
                                   nk_dots_pack_e2m1_apple10, nk_angulars_packed_e2m1_apple10, backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_apple10", nk_angulars_symmetric_e2m1_apple10,
                                      backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_apple10", nk_dots_pack_size_e2m1_apple10,
                                     nk_dots_pack_e2m1_apple10, nk_euclideans_packed_e2m1_apple10, backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_apple10", nk_euclideans_symmetric_e2m1_apple10,
                                        backend);
    run_attention<nk_bf16_k>(env, "attention_packed_bf16_apple10", nk_attention_pack_size_bf16_apple10,
                             nk_attention_pack_bf16_apple10, nk_attention_packed_bf16_apple10, backend);
    run_attention<nk_f16_k>(env, "attention_packed_f16_apple10", nk_attention_pack_size_f16_apple10,
                            nk_attention_pack_f16_apple10, nk_attention_packed_f16_apple10, backend);
    run_attention<nk_e4m3_k>(env, "attention_packed_e4m3_apple10", nk_attention_pack_size_e4m3_apple10,
                             nk_attention_pack_e4m3_apple10, nk_attention_packed_e4m3_apple10, backend);
    run_attention<nk_i8_k>(env, "attention_packed_i8_apple10", nk_attention_pack_size_i8_apple10,
                           nk_attention_pack_i8_apple10, nk_attention_packed_i8_apple10, backend);
    run_attention<nk_nvfp4_k>(env, "attention_packed_nvfp4_apple10", nk_attention_pack_size_nvfp4_apple10,
                              nk_attention_pack_nvfp4_apple10, nk_attention_packed_nvfp4_apple10, backend);
    run_attention<nk_mxfp4_k>(env, "attention_packed_mxfp4_apple10", nk_attention_pack_size_mxfp4_apple10,
                              nk_attention_pack_mxfp4_apple10, nk_attention_packed_mxfp4_apple10, backend);
    run_attention<nk_mxfp6e2m3_k>(env, "attention_packed_mxfp6e2m3_apple10", nk_attention_pack_size_mxfp6e2m3_apple10,
                                  nk_attention_pack_mxfp6e2m3_apple10, nk_attention_packed_mxfp6e2m3_apple10, backend);
    run_attention<nk_mxfp6e3m2_k>(env, "attention_packed_mxfp6e3m2_apple10", nk_attention_pack_size_mxfp6e3m2_apple10,
                                  nk_attention_pack_mxfp6e3m2_apple10, nk_attention_packed_mxfp6e3m2_apple10, backend);
    run_attention<nk_mxfp8e4m3_k>(env, "attention_packed_mxfp8e4m3_apple10", nk_attention_pack_size_mxfp8e4m3_apple10,
                                  nk_attention_pack_mxfp8e4m3_apple10, nk_attention_packed_mxfp8e4m3_apple10, backend);
    run_attention<nk_mxfp8e5m2_k>(env, "attention_packed_mxfp8e5m2_apple10", nk_attention_pack_size_mxfp8e5m2_apple10,
                                  nk_attention_pack_mxfp8e5m2_apple10, nk_attention_packed_mxfp8e5m2_apple10, backend);
    run_attention_gradients<nk_bf16_k>(env, "attention_packed_gradients_bf16_apple10",
                                       nk_attention_pack_size_bf16_apple10, nk_attention_pack_bf16_apple10,
                                       nk_attention_packed_bf16_apple10, nk_attention_packed_gradients_bf16_apple10,
                                       backend);
#endif // NUMKONG_TARGET_APPLE10
#else  // !NUMKONG_ARCH_METAL_
    return nk::status_t::missing_gpu_k;
#endif // NUMKONG_ARCH_METAL_
    return nk::status_t::success_k;
}

} // namespace ashvardanian::numkong::bench
