/**
 *  @file test/cross_simt.cuh
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Batch operation tests - the device backend and dispatch scenario CUDA and ROCm share.
 *
 *  Included by one translation unit per binary, `cross_cuda.cu` or `cross_rocm.hip`, which adds the
 *  capability sections of its vendor. HIP answers the CUDA runtime calls the backend makes.
 */
#pragma once
#ifndef NUMKONG_TEST_CROSS_SIMT_CUH
#define NUMKONG_TEST_CROSS_SIMT_CUH

#include "harness.hpp" // `error_stats_section_t`, `call_best`
#include "cross.hpp"   // `test_dots_packed`, `attention_weights_t`

#if NUMKONG_ARCH_ROCM_
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
inline constexpr hipError_t cudaSuccess = hipSuccess;
inline constexpr hipMemcpyKind cudaMemcpyDefault = hipMemcpyDefault;
inline constexpr unsigned cudaMemAttachGlobal = hipMemAttachGlobal;
inline hipStream_t const cudaStreamPerThread = hipStreamPerThread;
inline hipError_t cudaMallocManaged(void **pointer, std::size_t bytes, unsigned flags) {
    return hipMallocManaged(pointer, bytes, flags);
}
inline hipError_t cudaFree(void *pointer) { return hipFree(pointer); }
inline hipError_t cudaMemcpy(void *destination, void const *source, std::size_t bytes, hipMemcpyKind kind) {
    return hipMemcpy(destination, source, bytes, kind);
}
inline hipError_t cudaMemset(void *destination, int value, std::size_t bytes) {
    return hipMemset(destination, value, bytes);
}
inline hipError_t cudaStreamSynchronize(hipStream_t stream) { return hipStreamSynchronize(stream); }
#endif

namespace ashvardanian::numkong::test {

/** Unified memory: host and device both dereference it, so any @c nk::vector factory can use it.
 *  A failed allocation returns @c nullptr instead of throwing, which `try_*` factories report. */
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

/** Runs the CUDA or ROCm kernels on the per-thread stream over managed memory, keeping the first
 *  failed status. */
struct cuda_backend_t {

    /** The allocator every kernel operand comes from, readable by the host once the stream is
     *  synchronized. */
    template <typename value_type_>
    using allocator = cuda_managed_allocator<value_type_>;

    /** Where every call launches. */
    void *stream = cudaStreamPerThread;

    /** The first failure since the last synchronization. */
    nk_status_t status = nk_success_k;

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
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
     *  @c nk_misaligned_k unlaunched. */
    template <typename kernel_type_, typename... arguments_types_>
    bool refuses_misaligned(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., stream) == nk_misaligned_k;
    }

    /** Waits for the stream, returning the name of the first failure since the last call, or
     *  @c nullptr. */
    char const *synchronize() noexcept {
        keep(cudaStreamSynchronize((cudaStream_t)stream));
        nk_status_t const failure = status;
        status = nk_success_k;
        return failure == nk_success_k ? nullptr : nk_status_to_string(failure);
    }

    /** Remembers @p result unless an earlier failure is pending. */
    void keep(nk_status_t result) noexcept {
        if (status == nk_success_k) status = result;
    }

    /** Remembers a runtime call's failure as the kernel it would have failed. */
    void keep(cudaError_t result) noexcept { keep(result == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k); }
};

/** The capabilities of this build's vendor's first device, where every backend launches. */
inline nk_capability_t simt_capabilities() noexcept {
#if NUMKONG_ARCH_ROCM_
    return nk::rocm_capabilities();
#else
    return nk::cuda_capabilities();
#endif
}

/** The dispatch point @p best_ in the shape of its capability kernels, over the capabilities of GPU 0, where
 *  the backend launches: callable like them, and convertible to their function pointers. */
template <auto best_>
inline constexpr auto gpu_best =
    [](auto... arguments) noexcept { return call_best<best_>(simt_capabilities(), arguments...); };

} // namespace ashvardanian::numkong::test

/** The dispatching entry points, over the capabilities of the device the backend launches on. */
void test_cross_dispatch() {
    using namespace ashvardanian::numkong::test;
    error_stats_section_t check(simt_capabilities());
    check.section("Cross Dispatch", nk_cap_cuda_k | nk_cap_rocm_k);
#if NUMKONG_HEADER_ONLY
    check("dots_packed_i8_dispatch", [] {
        return test_missing_library<nk_dots_packed_i8_best>(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr);
    });
#else
    check("dots_packed_i8_dispatch", test_dots_packed<i8_t, cuda_backend_t>, gpu_best<nk_dots_pack_size_i8_best>,
          gpu_best<nk_dots_pack_i8_best>, gpu_best<nk_dots_packed_i8_best>);
#endif
}

#endif // NUMKONG_TEST_CROSS_SIMT_CUH
