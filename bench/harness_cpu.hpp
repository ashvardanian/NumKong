/**
 *  @file bench/harness_cpu.hpp
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief The host backend that every CPU benchmark times its rows with.
 */
#pragma once
#ifndef NUMKONG_BENCH_HARNESS_CPU_HPP
#define NUMKONG_BENCH_HARNESS_CPU_HPP

#include <cstring> // `std::memcpy`, `std::memset`

#include <vector> // `std::vector`

#include "harness.hpp" // `attention_shape_t`, `loop_t`, `input_sets_count`

namespace ashvardanian::numkong::bench {

/** Runs CPU kernels in place over host memory, timed by the loop's wall clock. */
struct host_backend_t {

    /** The allocator every kernel operand comes from. */
    template <typename value_type_>
    using allocator = nk::aligned_allocator<value_type_>;

    /** Row stride for @p row_bytes: exactly one row, keeping the tightest stride covered. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return row_bytes; }

    /** Rotation sets of @p per_set each, as many as @c input_sets_count allows. */
    std::size_t input_sets(bytes_t per_set) const noexcept { return input_sets_count(per_set); }

    /** Rows a token-row benchmark batches: one, as a row of the configured length already fills the
     *  caches. */
    static std::size_t token_rows(environment_t const &) noexcept { return 1; }

    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        std::memcpy(destination, source, bytes);
        return nk_success_k;
    }

    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        std::memset(destination, 0, bytes);
        return nk_success_k;
    }

    template <typename kernel_type_, typename... arguments_types_>
    nk_status_t call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., nullptr);
    }

    nk_status_t synchronize() noexcept { return nk_success_k; }

    template <typename launch_type_>
    nk_status_t time(loop_t &loop, std::size_t sets_count, launch_type_ &launch) {
        for (std::size_t call : loop)
            if (nk_status_t const status = launch(call & (sets_count - 1)); status != nk_success_k) return status;
        return nk_success_k;
    }
};

} // namespace ashvardanian::numkong::bench

#endif // NUMKONG_BENCH_HARNESS_CPU_HPP
