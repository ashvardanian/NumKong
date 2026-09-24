/**
 *  @file test/test.cuh
 *  @author Ash Vardanian
 *  @date September 23, 2026
 *  @brief CUDA memory and the CUDA backend the cross-kernel tests and benchmarks share.
 *
 *  Both allocators return @c nullptr on failure instead of throwing, so the non-throwing `try_*`
 *  factories report it. Benchmarks reach this header as `#include "../test/test.cuh"`.
 */
#pragma once
#ifndef NK_TEST_CUH
#define NK_TEST_CUH

#include <cstddef> // `std::size_t`, `std::ptrdiff_t`

#include <type_traits> // `std::true_type`, `std::is_same_v`

#include <cuda_runtime.h>

#include "numkong/vector.hpp" // `nk::vector`

#include "test.hpp" // `accumulation_t`, `host_backend_t`

namespace ashvardanian::numkong::test {

/** Unified memory: host and device both dereference it, so any @c nk::vector factory can use it. */
template <typename value_type_>
struct cuda_managed_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = cuda_managed_allocator<other_type_>;
    };

    constexpr cuda_managed_allocator() noexcept = default;

    template <typename other_type_>
    constexpr cuda_managed_allocator(cuda_managed_allocator<other_type_> const &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        void *pointer = nullptr;
        if (count == 0 || cudaMallocManaged(&pointer, count * sizeof(value_type), cudaMemAttachGlobal) != cudaSuccess)
            return nullptr;
        return static_cast<value_type *>(pointer);
    }

    void deallocate(value_type *pointer, std::size_t) noexcept {
        if (!pointer) return;
        [[maybe_unused]] cudaError_t const status = cudaFree(pointer);
    }

    template <typename other_type_>
    constexpr bool operator==(cuda_managed_allocator<other_type_> const &) const noexcept {
        return true;
    }
};

/** Device memory: only kernels dereference it, so build with @c try_empty and fill through
 *  @c cudaMemcpy. */
template <typename value_type_>
struct cuda_device_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = cuda_device_allocator<other_type_>;
    };

    constexpr cuda_device_allocator() noexcept = default;

    template <typename other_type_>
    constexpr cuda_device_allocator(cuda_device_allocator<other_type_> const &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        void *pointer = nullptr;
        if (count == 0 || cudaMalloc(&pointer, count * sizeof(value_type)) != cudaSuccess) return nullptr;
        return static_cast<value_type *>(pointer);
    }

    void deallocate(value_type *pointer, std::size_t) noexcept {
        if (!pointer) return;
        [[maybe_unused]] cudaError_t const status = cudaFree(pointer);
    }

    template <typename other_type_>
    constexpr bool operator==(cuda_device_allocator<other_type_> const &) const noexcept {
        return true;
    }
};

/** An `nk::vector` in device memory. */
template <typename value_type_>
using device_vector = vector<value_type_, cuda_device_allocator<value_type_>>;

/** Runs the CUDA kernels on @c cudaStreamPerThread over managed memory, keeping the first failed
 *  status. */
struct cuda_backend_t {

    /** The allocator every kernel operand comes from, readable by the host once the stream is
     *  synchronized. */
    template <typename value_type_>
    using allocator = cuda_managed_allocator<value_type_>;

    /** Where every call launches. */
    cudaStream_t stream = cudaStreamPerThread;

    /** The first failure since the last synchronization. */
    cudaError_t status = cudaSuccess;

    /** Dots of @p scalar_type_ accumulate integers exactly, F64 in Dot2, F32 in F64, and the rest
     *  in tensor cores. */
    template <typename scalar_type_>
    static constexpr accumulation_t dots_accumulation() noexcept {
        using result_t = typename scalar_type_::dot_result_t;
        if constexpr (is_integral_dtype<result_t>()) return accumulation_t::exact_k;
        else if constexpr (std::is_same_v<scalar_type_, f64_t>) return accumulation_t::dot2_k;
        else if constexpr (std::is_same_v<result_t, f64_t>) return accumulation_t::f64_k;
        else return accumulation_t::tensor_core_k;
    }

    /** Distances of @p scalar_type_ are judged by the accumulation of the dots beneath them. */
    template <typename scalar_type_>
    static constexpr accumulation_t spatials_accumulation() noexcept {
        return dots_accumulation<scalar_type_>();
    }

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes `cp.async` requires of A
     *  rows. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return (row_bytes + 15) / 16 * 16; }

    /** Copies @p bytes in whichever direction the pointers imply. */
    void copy(void *destination, void const *source, std::size_t bytes) noexcept {
        keep(cudaMemcpy(destination, source, bytes, cudaMemcpyDefault));
    }

    /** Zeroes @p bytes of a device buffer. */
    void zero(void *destination, std::size_t bytes) noexcept { keep(cudaMemset(destination, 0, bytes)); }

    /** Launches @p kernel with @p arguments on the stream. */
    template <typename kernel_type_, typename... arguments_types_>
    void call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        keep(kernel(arguments..., stream));
    }

    /** Calls @p kernel on operands it must refuse, reporting whether it returned
     *  @c cudaErrorMisalignedAddress unlaunched. */
    template <typename kernel_type_, typename... arguments_types_>
    bool refuses_misaligned(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., stream) == cudaErrorMisalignedAddress;
    }

    /** Waits for the stream, returning the name of the first failure since the last call, or
     *  @c nullptr. */
    char const *synchronize() noexcept {
        keep(cudaStreamSynchronize(stream));
        cudaError_t const failure = status;
        status = cudaSuccess;
        return failure == cudaSuccess ? nullptr : cudaGetErrorName(failure);
    }

    /** Remembers @p result unless an earlier failure is pending. */
    void keep(cudaError_t result) noexcept {
        if (status == cudaSuccess) status = result;
    }
};

} // namespace ashvardanian::numkong::test

#endif // NK_TEST_CUH
