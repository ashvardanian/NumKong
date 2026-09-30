/**
 *  @file test/harness.cuh
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The device backends of the CUDA and ROCm tests.
 *
 *  Each vendor gets its own runtime struct, compiled only by its own compiler, and one backend
 *  template runs over either, so a binary linking CUDA and ROCm units defines each name once.
 */
#pragma once
#ifndef NUMKONG_TEST_HARNESS_CUH
#define NUMKONG_TEST_HARNESS_CUH

#include "harness.hpp" // `error_stats_section_t`, `call_best`

namespace ashvardanian::numkong::test {

#if defined(__CUDACC__) && !defined(__HIP__)

/** The CUDA runtime calls a device backend makes. */
struct cuda_runtime_t {
    static constexpr nk::device_kind_t kind = nk::device_kind_t::cuda_k;

    /** The calling thread's own stream. */
    static void *stream() noexcept { return cudaStreamPerThread; }

    /** Allocates @p bytes that the host and the device both dereference, or returns @c nullptr. */
    static void *allocate_managed(std::size_t bytes) noexcept {
        void *pointer = nullptr;
        return cudaMallocManaged(&pointer, bytes, cudaMemAttachGlobal) == cudaSuccess ? pointer : nullptr;
    }

    static void free(void *pointer) noexcept { [[maybe_unused]] cudaError_t const status = cudaFree(pointer); }

    /** Copies @p bytes in whichever direction the pointers imply. */
    static nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        return cudaMemcpy(destination, source, bytes, cudaMemcpyDefault) == cudaSuccess ? nk_success_k
                                                                                        : nk_device_code_mismatch_k;
    }

    static nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        return cudaMemset(destination, 0, bytes) == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }

    /** Waits for everything queued on @p stream. */
    static nk_status_t synchronize(void *stream) noexcept {
        return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }
};
#endif // defined(__CUDACC__) && !defined(__HIP__)

#if defined(__HIP__)

/** The ROCm runtime calls a device backend makes. */
struct rocm_runtime_t {
    static constexpr nk::device_kind_t kind = nk::device_kind_t::rocm_k;

    /** The calling thread's own stream. */
    static void *stream() noexcept { return hipStreamPerThread; }

    /** Allocates @p bytes that the host and the device both dereference, or returns @c nullptr. */
    static void *allocate_managed(std::size_t bytes) noexcept {
        void *pointer = nullptr;
        return hipMallocManaged(&pointer, bytes, hipMemAttachGlobal) == hipSuccess ? pointer : nullptr;
    }

    static void free(void *pointer) noexcept { [[maybe_unused]] hipError_t const status = hipFree(pointer); }

    /** Copies @p bytes in whichever direction the pointers imply. */
    static nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        return hipMemcpy(destination, source, bytes, hipMemcpyDefault) == hipSuccess ? nk_success_k
                                                                                     : nk_device_code_mismatch_k;
    }

    static nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        return hipMemset(destination, 0, bytes) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }

    /** Waits for everything queued on @p stream. */
    static nk_status_t synchronize(void *stream) noexcept {
        return hipStreamSynchronize((hipStream_t)stream) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }
};
#endif // defined(__HIP__)

/** Unified memory of @p runtime_type_'s devices: host and device both dereference it, so any
 *  @c nk::vector factory can use it. A failed allocation returns @c nullptr instead of throwing,
 *  which the factories report. */
template <typename runtime_type_, typename value_type_>
struct managed_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = managed_allocator<runtime_type_, other_type_>;
    };

    constexpr managed_allocator() noexcept = default;

    template <typename other_type_>
    constexpr managed_allocator(managed_allocator<runtime_type_, other_type_> const &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        return count ? static_cast<value_type *>(runtime_type_::allocate_managed(count * sizeof(value_type))) : nullptr;
    }

    void deallocate(value_type *pointer, std::size_t) noexcept {
        if (pointer) runtime_type_::free(pointer);
    }

    template <typename other_type_>
    constexpr bool operator==(managed_allocator<runtime_type_, other_type_> const &) const noexcept {
        return true;
    }
};

/** Runs one vendor's kernels on its per-thread stream over managed memory, keeping the first failed
 *  status. */
template <typename runtime_type_>
struct device_backend {
    using runtime_t = runtime_type_;

    /** The allocator every kernel operand comes from, readable by the host once the stream is
     *  synchronized. */
    template <typename value_type_>
    using allocator = managed_allocator<runtime_type_, value_type_>;

    /** Where every call launches. */
    void *stream = runtime_type_::stream();

    /** The first failure since the last synchronization. */
    nk_status_t status = nk_success_k;

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept {
        return nk_size_round_up_to_multiple_(row_bytes, 16);
    }

    /** Copies @p bytes in whichever direction the pointers imply. */
    void copy(void *destination, void const *source, std::size_t bytes) noexcept {
        keep(runtime_type_::copy(destination, source, bytes));
    }

    /** Zeroes @p bytes of a device buffer. */
    void zero(void *destination, std::size_t bytes) noexcept { keep(runtime_type_::zero(destination, bytes)); }

    /** Launches @p kernel with @p arguments on the stream. */
    template <typename kernel_type_, typename... arguments_types_>
    void call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        keep(kernel(arguments..., stream));
    }

    /** Calls @p kernel on operands it must refuse, reporting whether it returned
     *  @c nk_misaligned_k unlaunched. */
    template <typename kernel_type_, typename... arguments_types_>
    bool refuses_misaligned(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., stream) == nk_misaligned_k;
    }

    /** Waits for the stream, returning the name of the first failure since the last call, or
     *  @c nullptr. */
    char const *synchronize() noexcept {
        keep(runtime_type_::synchronize(stream));
        nk_status_t const failure = status;
        status = nk_success_k;
        return failure == nk_success_k ? nullptr : nk_status_name(failure);
    }

    /** Remembers @p result unless an earlier failure is pending. */
    void keep(nk_status_t result) noexcept {
        if (status == nk_success_k) status = result;
    }
};

/** The capabilities of @p runtime_type_'s first device, where its backend launches. */
template <typename runtime_type_>
nk_capability_t device_capabilities() noexcept {
    auto const device = nk::device_t::make(runtime_type_::kind, 0);
    return device ? device.value.capabilities_enabled().value : 0;
}

/** The dispatch point @p best_ over the first device of @p runtime_type_, where its backend
 *  launches: callable like its capability kernels, and convertible to their function pointers. */
template <typename runtime_type_, auto best_>
inline constexpr auto device_best =
    [](auto... arguments) noexcept { return call_best<best_>(device_capabilities<runtime_type_>(), arguments...); };

#if defined(__CUDACC__) && !defined(__HIP__)
using cuda_backend_t = device_backend<cuda_runtime_t>;
#endif
#if defined(__HIP__)
using rocm_backend_t = device_backend<rocm_runtime_t>;
#endif

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_HARNESS_CUH
