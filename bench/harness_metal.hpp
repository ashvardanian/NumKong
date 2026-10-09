/**
 *  @file bench/harness_metal.hpp
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief The Metal backend that every @c bench/*_metal.cpp file times its rows with.
 */
#pragma once
#ifndef NUMKONG_BENCH_HARNESS_METAL_HPP
#define NUMKONG_BENCH_HARNESS_METAL_HPP

#include "harness.hpp" // `device_backend_t`, `attention_shape_t`, `loop_t`

#if NUMKONG_ARCH_METAL_

#include <cstring> // `std::memcpy`, `std::memset`

#include <algorithm> // `std::min`, `std::max`
#include <bit>       // `std::bit_ceil`
#include <chrono>    // `std::chrono::steady_clock`
#include <vector>    // `std::vector`

#include "numkong/numkong.h" // `nk_stream_synchronize_metal`

namespace ashvardanian::numkong::bench {

/** Runs the Metal kernels on its queue over unified memory, timing windows of calls by wall
 *  clock through their synchronization. */
struct metal_backend_t : device_backend_t {
    static std::size_t token_rows(environment_t const &) noexcept { return 4096; }

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
#endif // NUMKONG_BENCH_HARNESS_METAL_HPP
