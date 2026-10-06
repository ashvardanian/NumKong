/**
 *  @file bench/harness_cuda.hpp
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief The CUDA backend that every @c bench/*_cuda.cu file times its rows with.
 */
#pragma once
#ifndef NUMKONG_BENCH_HARNESS_CUDA_HPP
#define NUMKONG_BENCH_HARNESS_CUDA_HPP

#include "harness.hpp" // `device_backend_t`, `attention_shape_t`, `loop_t`

#if NUMKONG_ARCH_CUDA_

#include <bit> // `std::bit_ceil`

#include <cuda_runtime.h>

namespace ashvardanian::numkong::bench {

/** An @c nk::vector in device memory. */
template <typename value_type_>
using cuda_device_vector = nk::vector<value_type_, nk::allocator<value_type_>>;

/** Runs kernels over device memory, timing launch windows with CUDA events. */
struct cuda_backend_t : device_backend_t {
    std::size_t l2_bytes = 0;

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes `cp.async` requires of A
     *  rows. */
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

    /** Prefill and decode segments at the end of a 4096-key cache, Llama-style heads. */
    static std::vector<attention_shape_t> attention_shapes(environment_t const &) {
        return {{"prefill", 32, 8, 128, 4096, 4096}, {"decode", 32, 8, 128, 1, 4096}};
    }

    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        return cudaMemcpy(destination, source, bytes, cudaMemcpyDefault) == cudaSuccess ? nk_success_k
                                                                                        : nk_device_code_mismatch_k;
    }

    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        return cudaMemset(destination, 0, bytes) == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }

    template <typename kernel_type_, typename... arguments_types_>
    nk_status_t call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        auto const status = kernel(arguments..., memory.stream);
        if constexpr (std::is_same_v<decltype(status), cudaError_t const>)
            return status == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
        else return status;
    }

    template <typename launch_type_>
    nk_status_t time(loop_t &loop, std::size_t sets_count, launch_type_ &launch) {
        cudaEvent_t start = nullptr, stop = nullptr;
        if (cudaEventCreate(&start) != cudaSuccess) return nk_device_code_mismatch_k;
        if (cudaEventCreate(&stop) != cudaSuccess) {
            cudaEventDestroy(start);
            return nk_device_code_mismatch_k;
        }
        nk_status_t status = nk_success_k;
        std::size_t calls = 0, window = 1;
        for ([[maybe_unused]] std::size_t call : loop) {
            if (cudaEventRecord(start, (cudaStream_t)memory.stream) != cudaSuccess) {
                status = nk_device_code_mismatch_k;
                break;
            }
            for (std::size_t index = 0; index != window; ++index) {
                status = launch((calls + index) & (sets_count - 1));
                if (status != nk_success_k) break;
            }
            if (status != nk_success_k) break;
            if (cudaEventRecord(stop, (cudaStream_t)memory.stream) != cudaSuccess ||
                cudaEventSynchronize(stop) != cudaSuccess) {
                status = nk_device_code_mismatch_k;
                break;
            }
            float milliseconds = 0;
            if (cudaEventElapsedTime(&milliseconds, start, stop) != cudaSuccess) {
                status = nk_device_code_mismatch_k;
                break;
            }
            std::chrono::duration<float, std::milli> const elapsed {milliseconds};
            loop.add_window(elapsed, window);
            calls += window;
            if (elapsed < std::chrono::milliseconds(1)) window *= 2;
        }
        cudaError_t const start_status = cudaEventDestroy(start), stop_status = cudaEventDestroy(stop);
        if (status != nk_success_k) return status;
        return start_status == cudaSuccess && stop_status == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }

    nk_status_t synchronize() noexcept {
        return cudaStreamSynchronize((cudaStream_t)memory.stream) == cudaSuccess ? nk_success_k
                                                                                 : nk_device_code_mismatch_k;
    }
};

/** The element-wise, conversion and reduction CUDA rows, a file per family like the CPU's. */
void bench_each_cuda(environment_t const &env, cuda_backend_t const &backend, nk_capability_t enabled);
void bench_cast_cuda(environment_t const &env, cuda_backend_t const &backend, nk_capability_t enabled);
void bench_reduce_cuda(environment_t const &env, cuda_backend_t const &backend, nk_capability_t enabled);

} // namespace ashvardanian::numkong::bench

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_BENCH_HARNESS_CUDA_HPP
