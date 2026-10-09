/**
 *  @file bench/harness_rocm.hpp
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief The ROCm backend that every @c bench/*_rocm.hip file times its rows with.
 */
#pragma once
#ifndef NUMKONG_BENCH_HARNESS_ROCM_HPP
#define NUMKONG_BENCH_HARNESS_ROCM_HPP

#include "harness.hpp" // `device_backend_t`, `attention_shape_t`, `loop_t`

#if NUMKONG_ARCH_ROCM_

#include <algorithm>   // `std::min`, `std::max`
#include <bit>         // `std::bit_ceil`
#include <chrono>      // `std::chrono::steady_clock`
#include <type_traits> // `std::is_same_v`
#include <vector>      // `std::vector`

#include <hip/hip_runtime.h>

namespace ashvardanian::numkong::bench {

/** Runs kernels over device memory, timing launch windows with ROCm events. */
struct rocm_backend_t : device_backend_t {
    std::size_t l2_bytes = 0;

    /** Row stride rounded to the kernels' 16-byte alignment. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept {
        return nk::round_up_to_multiple<16>(row_bytes);
    }

    /** Rotation sets of @p per_set each: enough to cover twice the L2, at most
     *  @c input_sets_count. */
    std::size_t input_sets(bytes_t per_set) const noexcept {
        std::size_t const set_bytes = std::max(per_set.value, std::size_t(1));
        std::size_t const wanted = std::max(nk::divide_round_up(2 * std::size_t(l2_bytes), set_bytes), std::size_t(1));
        return std::min(std::bit_ceil(wanted), input_sets_count(per_set));
    }

    /** Rows a token-row benchmark batches: a 4096-token prefill. */
    static std::size_t token_rows(environment_t const &) noexcept { return 4096; }

    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        return hipMemcpy(destination, source, bytes, hipMemcpyDefault) == hipSuccess ? nk_success_k
                                                                                     : nk_device_code_mismatch_k;
    }

    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        return hipMemset(destination, 0, bytes) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }

    template <typename kernel_type_, typename... arguments_types_>
    nk_status_t call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        auto const status = kernel(arguments..., memory.stream);
        if constexpr (std::is_same_v<decltype(status), hipError_t const>)
            return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
        else return status;
    }

    template <typename launch_type_>
    nk_status_t time(loop_t &loop, std::size_t sets_count, launch_type_ &launch) {
        hipEvent_t start = nullptr, stop = nullptr;
        if (hipEventCreate(&start) != hipSuccess) return nk_device_code_mismatch_k;
        if (hipEventCreate(&stop) != hipSuccess) {
            hipEventDestroy(start);
            return nk_device_code_mismatch_k;
        }
        nk_status_t status = nk_success_k;
        std::size_t calls = 0, window = 1;
        for ([[maybe_unused]] std::size_t call : loop) {
            if (hipEventRecord(start, (hipStream_t)memory.stream) != hipSuccess) {
                status = nk_device_code_mismatch_k;
                break;
            }
            for (std::size_t index = 0; index != window; ++index) {
                status = launch((calls + index) & (sets_count - 1));
                if (status != nk_success_k) break;
            }
            if (status != nk_success_k) break;
            if (hipEventRecord(stop, (hipStream_t)memory.stream) != hipSuccess ||
                hipEventSynchronize(stop) != hipSuccess) {
                status = nk_device_code_mismatch_k;
                break;
            }
            float milliseconds = 0;
            if (hipEventElapsedTime(&milliseconds, start, stop) != hipSuccess) {
                status = nk_device_code_mismatch_k;
                break;
            }
            std::chrono::duration<float, std::milli> const elapsed {milliseconds};
            loop.add_window(elapsed, window);
            calls += window;
            if (elapsed < std::chrono::milliseconds(1)) window *= 2;
        }
        hipError_t const start_status = hipEventDestroy(start), stop_status = hipEventDestroy(stop);
        if (status != nk_success_k) return status;
        return start_status == hipSuccess && stop_status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }

    nk_status_t synchronize() noexcept {
        return hipStreamSynchronize((hipStream_t)memory.stream) == hipSuccess ? nk_success_k
                                                                              : nk_device_code_mismatch_k;
    }
};

/** The element-wise ROCm rows, in their own file like the CPU's. */
void bench_each_rocm(environment_t const &env, rocm_backend_t const &backend, nk_capability_t enabled);

} // namespace ashvardanian::numkong::bench

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_BENCH_HARNESS_ROCM_HPP
