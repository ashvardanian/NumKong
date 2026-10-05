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
#include <cstddef> // `std::size_t`, `std::ptrdiff_t`
#include <cstring> // `std::memcpy`, `std::memset`

#include <algorithm> // `std::min`, `std::max`
#include <bit>       // `std::bit_ceil`
#include <chrono>    // `std::chrono::steady_clock`

#include "numkong/numkong.h" // `nk_dots_packed_i8_metal`, `nk_stream_synchronize_metal`

#include "cross.hpp"

#if NUMKONG_ARCH_METAL_

namespace ashvardanian::numkong::bench {

/** Runs the Metal kernels on its queue over unified memory, timing windows of calls by wall
 *  clock through their synchronization. */
struct metal_backend_t : device_backend_t {
    static std::size_t token_rows(environment_t const &) noexcept { return 4096; }

    static std::vector<attention_shape_t> attention_shapes(environment_t const &) {
        return {{"prefill", 32, 8, 128, 4096, 4096}, {"decode", 32, 8, 128, 1, 4096}};
    }

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept {
        return nk_size_round_up_to_multiple_(row_bytes, 16);
    }

    /** Rotation sets of @p per_set each: enough to cover twice a 32 MB system-level cache,
     *  at most @c input_sets_count. */
    std::size_t input_sets(bytes_t per_set) const noexcept {
        std::size_t const cache_bytes = std::size_t(32) << 20;
        std::size_t const set_bytes = std::max(per_set.value, std::size_t(1));
        std::size_t const wanted = std::max(nk::divide_round_up(2 * cache_bytes, set_bytes), std::size_t(1));
        return std::min(std::bit_ceil(wanted), input_sets_count(per_set));
    }

    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        if (nk_status_t const status = synchronize(); status != nk_success_k) return status;
        std::memcpy(destination, source, bytes);
        return nk_success_k;
    }

    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        if (nk_status_t const status = synchronize(); status != nk_success_k) return status;
        std::memset(destination, 0, bytes);
        return nk_success_k;
    }

    template <typename kernel_type_, typename... arguments_types_>
    nk_status_t call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., memory.stream);
    }

    template <typename launch_type_>
    nk_status_t time(loop_t &loop, std::size_t sets_count, launch_type_ &launch) {
        using steady_clock_t = std::chrono::steady_clock;
        std::size_t calls = 0, window = 1;
        for ([[maybe_unused]] std::size_t call : loop) {
            auto const start = steady_clock_t::now();
            for (std::size_t index = 0; index != window; ++index)
                if (nk_status_t const status = launch((calls + index) & (sets_count - 1)); status != nk_success_k)
                    return status;
            if (nk_status_t const status = synchronize(); status != nk_success_k) return status;
            auto const elapsed = steady_clock_t::now() - start;
            loop.add_window(elapsed, window);
            calls += window;
            if (elapsed < std::chrono::milliseconds(1)) window *= 2;
        }
        return nk_success_k;
    }

    nk_status_t synchronize() noexcept { return nk_stream_synchronize_metal(memory.stream); }
};

} // namespace ashvardanian::numkong::bench

#endif // NUMKONG_ARCH_METAL_

namespace ashvardanian::numkong::bench {

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
    }
#endif // NUMKONG_TARGET_APPLE9
#if NUMKONG_TARGET_APPLE10
    if (!(enabled & nk_cap_apple10_k)) return nk::status_t::success_k;
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
#endif // NUMKONG_TARGET_APPLE10
#else  // !NUMKONG_ARCH_METAL_
    return nk::status_t::missing_gpu_k;
#endif // NUMKONG_ARCH_METAL_
    return nk::status_t::success_k;
}

} // namespace ashvardanian::numkong::bench
