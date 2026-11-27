/**
 *  @file cpp/numkong/devices.hpp
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief Devices, their capability masks and streams, and the allocators over their memory.
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_DEVICES_HPP
#define NUMKONG_DEVICES_HPP

#include <cstddef> // `std::size_t`

#include <utility> // `std::move`, `std::exchange`

#include "numkong/capabilities.h" // `nk_capabilities_enabled_cpu`, `nk_stream_init_cuda`
#include "numkong/memory.h"       // `nk_allocator_init_unified_best`
#include "numkong/memory.hpp"     // `allocator`

namespace ashvardanian::numkong {

/** Which runtime a device belongs to, as the <tt>nk_\<kind\>_*</tt> C functions name it. */
enum class device_kind_t : int { cpu_k, cuda_k, rocm_k, metal_k };

/** One device NumKong can run kernels on: the host CPU, or a GPU by its runtime's own ordinal. */
class device_t {
    device_kind_t kind_;
    std::size_t ordinal_;

    constexpr device_t(device_kind_t kind, std::size_t ordinal) noexcept : kind_(kind), ordinal_(ordinal) {}

  public:
    /** The host CPU, which every build has. */
    static constexpr device_t cpu() noexcept { return {device_kind_t::cpu_k, 0}; }

    /** How many devices of @p kind the process sees: one CPU, or @c missing_gpu_k and zero GPUs. */
    static expected<std::size_t> count(device_kind_t kind) noexcept {
        nk_size_t count = 1;
        nk_status_t status = nk_success_k;
        switch (kind) {
        case device_kind_t::cpu_k: break;
        case device_kind_t::cuda_k: status = nk_count_devices_cuda(&count); break;
        case device_kind_t::rocm_k: status = nk_count_devices_rocm(&count); break;
        case device_kind_t::metal_k: status = nk_count_devices_metal(&count); break;
        }
        return {count, static_cast<status_t>(status)};
    }

    /** Device @p ordinal of @p kind, or @c missing_gpu_k past the last one. */
    static expected<device_t> make(device_kind_t kind, std::size_t ordinal) noexcept {
        auto const [devices, counted] = count(kind);
        status_t const status = succeeded(counted) && ordinal >= devices ? status_t::missing_gpu_k : counted;
        return {device_t {kind, ordinal}, status};
    }

    constexpr device_kind_t kind() const noexcept { return kind_; }
    constexpr std::size_t ordinal() const noexcept { return ordinal_; }

    /** What this device runs, whether or not this binary holds kernels for it; zero on failure. */
    expected<nk_capability_t> capabilities_detected() const noexcept {
        nk_capability_t capabilities = 0;
        nk_status_t status = nk_success_k;
        switch (kind_) {
        case device_kind_t::cpu_k: status = nk_capabilities_detected_cpu(&capabilities); break;
        case device_kind_t::cuda_k: status = nk_capabilities_detected_cuda(ordinal_, &capabilities); break;
        case device_kind_t::rocm_k: status = nk_capabilities_detected_rocm(ordinal_, &capabilities); break;
        case device_kind_t::metal_k: status = nk_capabilities_detected_metal(ordinal_, &capabilities); break;
        }
        return {capabilities, static_cast<status_t>(status)};
    }

    /** What this binary holds code for that this device runs, whether or not it offers the
     *  capabilities; zero past the last device. */
    nk_capability_t capabilities_compiled() const noexcept {
        nk_capability_t capabilities = 0;
        [[maybe_unused]] nk_status_t status = nk_success_k; // A failure leaves the mask empty
        switch (kind_) {
        case device_kind_t::cpu_k: status = nk_capabilities_compiled_cpu(&capabilities); break;
        case device_kind_t::cuda_k: status = nk_capabilities_compiled_cuda(ordinal_, &capabilities); break;
        case device_kind_t::rocm_k: status = nk_capabilities_compiled_rocm(ordinal_, &capabilities); break;
        case device_kind_t::metal_k: status = nk_capabilities_compiled_metal(ordinal_, &capabilities); break;
        }
        return capabilities;
    }

    /** Both at once: the mask for this device's dispatch points; zero on failure. */
    expected<nk_capability_t> capabilities_enabled() const noexcept {
        nk_capability_t capabilities = 0;
        nk_status_t status = nk_success_k;
        switch (kind_) {
        case device_kind_t::cpu_k: status = nk_capabilities_enabled_cpu(&capabilities); break;
        case device_kind_t::cuda_k: status = nk_capabilities_enabled_cuda(ordinal_, &capabilities); break;
        case device_kind_t::rocm_k: status = nk_capabilities_enabled_rocm(ordinal_, &capabilities); break;
        case device_kind_t::metal_k: status = nk_capabilities_enabled_metal(ordinal_, &capabilities); break;
        }
        return {capabilities, static_cast<status_t>(status)};
    }

    /** Prepares the calling thread for the CPU kernels of @p capabilities; GPUs have no such
     *  kernels, so they report @c missing_kernel_k. */
    status_t configure_thread(nk_capability_t capabilities) const noexcept {
        if (kind_ != device_kind_t::cpu_k) return status_t::missing_kernel_k;
        return static_cast<status_t>(nk_configure_thread_cpu(capabilities));
    }
};

/** The mask every verb dispatches over by default: the CPU's enabled capabilities, as
 *  @c device_t::cpu().capabilities_enabled() reports them. */
inline nk_capability_t default_capabilities() noexcept {
    nk_capability_t capabilities = 0;
    [[maybe_unused]] nk_status_t const status = nk_capabilities_enabled_cpu(&capabilities); // A failure leaves none
    return capabilities;
}

/** Owns a device stream; buffers using it must be released before the stream. */
class stream_t {
    nk_stream_t handle_ = nullptr;
    nk_status_t (*free_)(nk_stream_t) = nullptr;

  public:
    stream_t() noexcept = default;
    stream_t(stream_t const &) = delete;
    stream_t &operator=(stream_t const &) = delete;
    stream_t(stream_t &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)), free_(other.free_) {}
    stream_t &operator=(stream_t &&other) noexcept {
        if (this != &other) {
            if (handle_) { [[maybe_unused]] nk_status_t const status = free_(handle_); }
            handle_ = std::exchange(other.handle_, nullptr);
            free_ = other.free_;
        }
        return *this;
    }
    ~stream_t() {
        if (handle_) { [[maybe_unused]] nk_status_t const status = free_(handle_); }
    }

    static expected<stream_t> make(device_t device) noexcept {
        stream_t result;
        nk_status_t (*initialize)(nk_size_t, nk_stream_t *) = nullptr;
        switch (device.kind()) {
        case device_kind_t::cpu_k: return {std::move(result), status_t::success_k};
        case device_kind_t::cuda_k:
            initialize = nk_stream_init_cuda;
            result.free_ = nk_stream_free_cuda;
            break;
        case device_kind_t::rocm_k:
            initialize = nk_stream_init_rocm;
            result.free_ = nk_stream_free_rocm;
            break;
        case device_kind_t::metal_k:
            initialize = nk_stream_init_metal;
            result.free_ = nk_stream_free_metal;
            break;
        default: return {std::move(result), status_t::missing_gpu_k};
        }
        nk_status_t const status = initialize(device.ordinal(), &result.handle_);
        return {std::move(result), static_cast<status_t>(status)};
    }

    nk_stream_t get() const noexcept { return handle_; }
};

/** An allocator over the unified memory of the device @p capabilities and @p stream name, which
 *  the host reads once the stream is synchronized. */
template <typename value_type_>
expected<allocator<value_type_>> unified_allocator(nk_capability_t capabilities,
                                                   nk_stream_t stream = nullptr) noexcept {
    return allocator<value_type_>::make(capabilities, stream, nk_allocator_init_unified_best);
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_DEVICES_HPP
