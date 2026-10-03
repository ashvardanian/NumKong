/**
 *  @file test/harness.cuh
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The device backends of the CUDA and ROCm tests.
 *
 *  One backend template runs over the library's runtime calls of either vendor, which the compiler
 *  of each unit picks, so a binary linking CUDA and ROCm units defines each name once.
 */
#pragma once
#ifndef NUMKONG_TEST_HARNESS_CUH
#define NUMKONG_TEST_HARNESS_CUH

#include "harness.hpp" // `error_stats_section_t`, `call_best`

namespace ashvardanian::numkong::test {

/** Unified memory from the library's @p allocate_ and @p free_: host and device both dereference
 *  it, so any @c nk::vector factory can use it. A failed allocation returns @c nullptr instead of
 *  throwing, which the factories report. */
template <auto allocate_, auto free_, typename value_type_>
struct unified_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = unified_allocator<allocate_, free_, other_type_>;
    };

    constexpr unified_allocator() noexcept = default;

    template <typename other_type_>
    constexpr unified_allocator(unified_allocator<allocate_, free_, other_type_> const &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        void *pointer = nullptr;
        nk_status_t const status = allocate_(count * sizeof(value_type), &pointer, nullptr);
        return status == nk_success_k ? static_cast<value_type *>(pointer) : nullptr;
    }

    void deallocate(value_type *pointer, std::size_t count) noexcept {
        [[maybe_unused]] nk_status_t const status = free_(pointer, count * sizeof(value_type), nullptr);
    }

    template <typename other_type_>
    constexpr bool operator==(unified_allocator<allocate_, free_, other_type_> const &) const noexcept {
        return true;
    }
};

/** Runs one vendor's kernels on a stream of its first device over unified memory, keeping the first
 *  failed status. The arguments are the library's runtime calls of that vendor. */
template <auto capabilities_enabled_, auto stream_init_, auto stream_free_, auto allocate_, auto free_,
          auto synchronize_>
struct device_backend {

    /** The allocator every kernel operand comes from, readable by the host once the stream is
     *  synchronized. */
    template <typename value_type_>
    using allocator = unified_allocator<allocate_, free_, value_type_>;

    /** Where every call launches. */
    void *stream = shared_stream();

    /** The first failure since the last synchronization. */
    nk_status_t status = nk_success_k;

    /** The capabilities of the first device, where every backend launches, or zero. */
    static nk_capability_t capabilities() noexcept {
        nk_capability_t capabilities = 0;
        return capabilities_enabled_(0, &capabilities) == nk_success_k ? capabilities : 0;
    }

    /** The one stream every backend of this vendor launches on, opened on the first device at first
     *  use and freed at exit. */
    static void *shared_stream() noexcept {
        struct owner_t {
            void *stream = nullptr;
            owner_t() noexcept { [[maybe_unused]] nk_status_t const status = stream_init_(0, &stream); }
            ~owner_t() noexcept { [[maybe_unused]] nk_status_t const status = stream_free_(stream); }
        };
        static owner_t const owner;
        return owner.stream;
    }

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept {
        return nk_size_round_up_to_multiple_(row_bytes, 16);
    }

    /** Copies @p bytes once every queued call has finished with them. */
    void copy(void *destination, void const *source, std::size_t bytes) noexcept {
        keep(synchronize_(stream));
        std::memcpy(destination, source, bytes);
    }

    /** Zeroes @p bytes once every queued call has finished with them. */
    void zero(void *destination, std::size_t bytes) noexcept {
        keep(synchronize_(stream));
        std::memset(destination, 0, bytes);
    }

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
        keep(synchronize_(stream));
        nk_status_t const failure = status;
        status = nk_success_k;
        return failure == nk_success_k ? nullptr : nk_status_name(failure);
    }

    /** Remembers @p result unless an earlier failure is pending. */
    void keep(nk_status_t result) noexcept {
        if (status == nk_success_k) status = result;
    }
};

/** The dispatch point @p best_ over the first device of @p backend_type_, where it launches:
 *  callable like its capability kernels, and convertible to their function pointers. */
template <typename backend_type_, auto best_>
inline constexpr auto device_best =
    [](auto... arguments) noexcept { return call_best<best_>(backend_type_::capabilities(), arguments...); };

#if defined(__HIP__)
using rocm_backend_t =
    device_backend<nk_rocm_capabilities_enabled, nk_rocm_stream_init, nk_rocm_stream_free,
                   nk_memory_allocate_unified_rocm, nk_memory_free_unified_rocm, nk_stream_synchronize_rocm>;
#elif defined(__CUDACC__)
using cuda_backend_t =
    device_backend<nk_cuda_capabilities_enabled, nk_cuda_stream_init, nk_cuda_stream_free,
                   nk_memory_allocate_unified_cuda, nk_memory_free_unified_cuda, nk_stream_synchronize_cuda>;
#endif

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_HARNESS_CUH
