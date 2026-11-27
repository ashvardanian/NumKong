/**
 *  @file test/backend_device.hpp
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief The backend the CUDA, ROCm and Metal suites run on: an owned stream over unified memory.
 *
 *  Only the linked suites include it, as it opens devices through the C++ SDK.
 */
#pragma once
#ifndef NUMKONG_TEST_BACKEND_DEVICE_HPP
#define NUMKONG_TEST_BACKEND_DEVICE_HPP

#include <memory> // `std::shared_ptr`, `std::make_shared`

#include "numkong/devices.hpp" // `nk::device_t`, `nk::stream_t`, `nk::unified_allocator`

#include "harness.hpp" // `test_run`, `allocator_of`

namespace ashvardanian::numkong::test {

/** Runs one vendor's kernels on an owned stream over unified memory. */
template <nk::device_kind_t kind_, auto synchronize_>
struct device_backend {

    /** The allocator every kernel operand comes from, readable by the host once the stream is
     *  synchronized. */
    template <typename value_type_>
    using allocator = nk::allocator<value_type_>;

    struct owner_t {
        nk::stream_t queue;
        [[no_unique_address]] nk::allocator<char> memory;
        nk_capability_t capabilities = 0;
        nk_status_t status;

        explicit owner_t(std::size_t ordinal) noexcept {
            auto const [device, made] = nk::device_t::make(kind_, ordinal);
            status = static_cast<nk_status_t>(made);
            if (status != nk_success_k) return;
            auto const enabled = device.capabilities_enabled();
            capabilities = enabled.value;
            status = static_cast<nk_status_t>(enabled.status);
            if (status != nk_success_k) return;
            auto created = nk::stream_t::make(device);
            status = static_cast<nk_status_t>(created.status);
            if (status != nk_success_k) return;
            queue = std::move(created.value);
            auto allocated = nk::unified_allocator<char>(capabilities, queue.get());
            memory = allocated.value;
            status = static_cast<nk_status_t>(allocated.status);
        }
    };

    std::shared_ptr<owner_t> owner;
    nk_stream_t const stream;

    /** The first failed call's status since the last @c synchronize, which skips later calls. */
    nk_status_t failure = nk_success_k;

    /** Opens the device of @c test_run that the device cases run on. */
    device_backend() : owner(std::make_shared<owner_t>(test_run.settings.device_ordinal)), stream(owner->queue.get()) {}

    nk_capability_t capabilities() const noexcept { return owner->capabilities; }

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept {
        return nk_size_round_up_to_multiple_(row_bytes, 16);
    }

    /** Copies @p bytes once every queued call has finished with them. */
    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        if (owner->status != nk_success_k) return owner->status;
        nk_status_t const status = synchronize_(stream);
        if (status != nk_success_k) return status;
        std::memcpy(destination, source, bytes);
        return nk_success_k;
    }

    /** Zeroes @p bytes once every queued call has finished with them. */
    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        if (owner->status != nk_success_k) return owner->status;
        nk_status_t const status = synchronize_(stream);
        if (status != nk_success_k) return status;
        std::memset(destination, 0, bytes);
        return nk_success_k;
    }

    /** Launches @p kernel with @p arguments on the stream, unless a call failed. */
    template <typename kernel_type_, typename... arguments_types_>
    void call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        if (owner->status == nk_success_k && failure == nk_success_k) failure = kernel(arguments..., stream);
    }

    /** Calls @p kernel on operands it must refuse, reporting whether it returned
     *  @c nk_misaligned_k unlaunched, and keeping no failure. */
    template <typename kernel_type_, typename... arguments_types_>
    bool refuses_misaligned(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., stream) == nk_misaligned_k;
    }

    /** Waits for submitted work, then returns and clears the first failure since the last call to
     *  it, or else the stream's. */
    nk_status_t synchronize() noexcept {
        if (owner->status != nk_success_k) return owner->status;
        nk_status_t const synchronized = synchronize_(stream);
        nk_status_t const failed = std::exchange(failure, nk_success_k);
        return failed != nk_success_k ? failed : synchronized;
    }
};

template <typename value_type_, nk::device_kind_t kind_, auto synchronize_>
auto allocator_of(device_backend<kind_, synchronize_> const &backend) noexcept {
    return nk::allocator<value_type_>(backend.owner->memory);
}

#if NUMKONG_ARCH_ROCM_
using rocm_backend_t = device_backend<nk::device_kind_t::rocm_k, nk_stream_synchronize_rocm>;
#endif // NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_CUDA_
using cuda_backend_t = device_backend<nk::device_kind_t::cuda_k, nk_stream_synchronize_cuda>;
#endif // NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_METAL_
using metal_backend_t = device_backend<nk::device_kind_t::metal_k, nk_stream_synchronize_metal>;
#endif // NUMKONG_ARCH_METAL_

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_BACKEND_DEVICE_HPP
