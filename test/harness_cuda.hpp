/**
 *  @file test/harness_cuda.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The CUDA runtime checks that the CUDA reference tests share.
 */
#pragma once
#ifndef NUMKONG_TEST_HARNESS_CUDA_HPP
#define NUMKONG_TEST_HARNESS_CUDA_HPP

#include "harness.hpp" // `record_failed_check`, `run_kernel_case`

#if NUMKONG_ARCH_CUDA_

#include <cuda_runtime.h> // `cudaError_t`, `cudaGetErrorString`, `cudaSetDevice`

namespace ashvardanian::numkong::test {

/** Holds when the CUDA runtime call that returned @p status succeeded; otherwise counts a failed
 *  check at the caller's line and prints the error under it. */
inline bool expect_device_success(cudaError_t status, std::source_location where = std::source_location::current()) {
    if (status == cudaSuccess) return true;
    record_failed_check(where);
    fmt::println(stderr, "    {}", cudaGetErrorString(status));
    return false;
}

/** Makes the CUDA device @p ordinal current for the scope's lifetime, then restores the old one. */
struct cuda_device_scope_t {
    int caller = 0;
    cudaError_t status;

    explicit cuda_device_scope_t(nk_size_t ordinal) noexcept : status(cudaGetDevice(&caller)) {
        if (status == cudaSuccess) status = cudaSetDevice(static_cast<int>(ordinal));
    }
    cuda_device_scope_t(cuda_device_scope_t const &) = delete;
    cuda_device_scope_t &operator=(cuda_device_scope_t const &) = delete;
    ~cuda_device_scope_t() noexcept {
        if (status == cudaSuccess) cudaSetDevice(caller);
    }
};

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_TEST_HARNESS_CUDA_HPP
