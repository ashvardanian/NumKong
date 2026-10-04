/**
 *  @file bench/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batch operation benchmarks for the Metal kernels, the twin of `cross_cuda.cu`.
 *
 *  Runs the drivers of `cross.hpp` through a @c metal_backend_t, on the null stream, the library's
 *  queue on the system default device, over its unified memory. Metal has no C-level events, so
 *  every window of launches is timed by wall clock through the synchronization after it. Input sets
 *  rotate until their footprint is at least twice the system-level cache.
 */
#include <cstddef> // `std::size_t`, `std::ptrdiff_t`
#include <cstring> // `std::memcpy`, `std::memset`

#include <algorithm>   // `std::max`, `std::min`
#include <bit>         // `std::bit_ceil`
#include <chrono>      // `std::chrono::steady_clock`
#include <type_traits> // `std::true_type`

#include "numkong/numkong.h" // `nk_memory_allocate_unified_metal`, `nk_stream_synchronize_metal`

#include "cross.hpp"

#if NUMKONG_WITH_METAL

namespace ashvardanian::numkong::bench {

/** Unified memory of the system default device, which the host and its kernels both dereference,
 *  so any @c nk::vector factory can use it. */
template <typename value_type_>
struct metal_shared_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = metal_shared_allocator<other_type_>;
    };

    constexpr metal_shared_allocator() noexcept = default;
    template <typename other_type_>
    constexpr metal_shared_allocator(metal_shared_allocator<other_type_> const &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        void *pointer = nullptr;
        nk_status_t const status = nk_memory_allocate_unified_metal(count * sizeof(value_type), &pointer, nullptr);
        return status == nk_success_k ? static_cast<value_type *>(pointer) : nullptr;
    }
    void deallocate(value_type *pointer, std::size_t count) noexcept {
        [[maybe_unused]] nk_status_t const status = nk_memory_free_unified_metal(pointer, count * sizeof(value_type),
                                                                                 nullptr);
    }
    template <typename other_type_>
    constexpr bool operator==(metal_shared_allocator<other_type_> const &) const noexcept {
        return true;
    }
};

/** Runs the Metal kernels on the null stream over unified memory, timing windows of calls by wall
 *  clock through their synchronization. */
struct metal_backend_t {

    /** The allocator every kernel operand comes from, readable by the host once the stream is
     *  synchronized. */
    template <typename value_type_>
    using allocator = metal_shared_allocator<value_type_>;

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return (row_bytes + 15) / 16 * 16; }

    /** Rotation sets of @p per_set each: enough to cover twice a 32 MB system-level cache,
     *  at most @c input_sets_count. */
    std::size_t input_sets(bytes_t per_set) const noexcept {
        std::size_t const cache_bytes = std::size_t(32) << 20;
        std::size_t const set_bytes = std::max(per_set.value, std::size_t(1));
        std::size_t const wanted = std::max(nk::divide_round_up(2 * cache_bytes, set_bytes), std::size_t(1));
        return std::min(std::bit_ceil(wanted), input_sets_count(per_set));
    }

    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        if (nk_status_t const status = synchronize(); status != nk_success_k) return status;
        std::memcpy(destination, source, bytes);
        return nk_success_k;
    }

    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        if (nk_status_t const status = synchronize(); status != nk_success_k) return status;
        std::memset(destination, 0, bytes);
        return nk_success_k;
    }

    template <typename kernel_type_, typename... arguments_types_>
    nk_status_t call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., nullptr);
    }

    template <typename launch_type_>
    nk_status_t time(loop_t &loop, std::size_t sets_count, launch_type_ &launch) {
        using steady_clock_t = std::chrono::steady_clock;
        std::size_t calls = 0, window = 1;
        for ([[maybe_unused]] std::size_t call : loop) {
            auto const start = steady_clock_t::now();
            for (std::size_t index = 0; index != window; ++index)
                if (nk_status_t const status = launch((calls + index) & (sets_count - 1)); status != nk_success_k)
                    return status;
            if (nk_status_t const status = synchronize(); status != nk_success_k) return status;
            auto const elapsed = steady_clock_t::now() - start;
            loop.add_window(elapsed, window);
            calls += window;
            if (elapsed < std::chrono::milliseconds(1)) window *= 2;
        }
        return nk_success_k;
    }

    nk_status_t synchronize() noexcept { return nk_stream_synchronize_metal(nullptr); }
};

} // namespace ashvardanian::numkong::bench

#endif // NUMKONG_WITH_METAL

namespace ashvardanian::numkong::bench {

/** Every Metal baseline entry point beside every Apple9 and Apple10 one, so the matrix units'
 *  speedup shows, on devices whose families include each. */
void bench_cross_metal([[maybe_unused]] environment_t const &env) {
#if NUMKONG_WITH_METAL
    nk_capability_t detected = 0, enabled = 0;
    if (nk_metal_capabilities_detected(0, &detected) != nk_success_k || !detected)
        return fmt::println("- Metal: no device");
    char families[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_capabilities_name(detected, families, sizeof(families));
    fmt::println("- Metal: {}", families);

    if (nk_metal_capabilities_enabled(0, &enabled) != nk_success_k) enabled = 0;
    metal_backend_t const backend;
    if (enabled & nk_cap_metal_k) {
        run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_metal", nk_dots_pack_size_bf16_metal, nk_dots_pack_bf16_metal,
                                   nk_dots_packed_bf16_metal, backend);
        run_dots_packed<nk_f16_k>(env, "dots_packed_f16_metal", nk_dots_pack_size_f16_metal, nk_dots_pack_f16_metal,
                                  nk_dots_packed_f16_metal, backend);
        run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_metal", nk_dots_pack_size_e5m2_metal, nk_dots_pack_e5m2_metal,
                                   nk_dots_packed_e5m2_metal, backend);
        run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_metal", nk_dots_pack_size_e4m3_metal, nk_dots_pack_e4m3_metal,
                                   nk_dots_packed_e4m3_metal, backend);
        run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_metal", nk_dots_pack_size_e3m2_metal, nk_dots_pack_e3m2_metal,
                                   nk_dots_packed_e3m2_metal, backend);
        run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_metal", nk_dots_pack_size_e2m3_metal, nk_dots_pack_e2m3_metal,
                                   nk_dots_packed_e2m3_metal, backend);
        run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_metal", nk_dots_pack_size_e2m1_metal, nk_dots_pack_e2m1_metal,
                                   nk_dots_packed_e2m1_metal, backend);
        run_dots_packed<nk_i8_k>(env, "dots_packed_i8_metal", nk_dots_pack_size_i8_metal, nk_dots_pack_i8_metal,
                                 nk_dots_packed_i8_metal, backend);
        run_dots_packed<nk_u8_k>(env, "dots_packed_u8_metal", nk_dots_pack_size_u8_metal, nk_dots_pack_u8_metal,
                                 nk_dots_packed_u8_metal, backend);

        run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_metal", nk_dots_symmetric_bf16_metal, backend);
        run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_metal", nk_dots_symmetric_f16_metal, backend);
        run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_metal", nk_dots_symmetric_e4m3_metal, backend);
        run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_metal", nk_dots_symmetric_i8_metal, backend);
    }
#if NUMKONG_TARGET_APPLE9
    if (enabled & nk_cap_apple9_k) {
        run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_apple9", nk_dots_pack_size_bf16_apple9,
                                   nk_dots_pack_bf16_apple9, nk_dots_packed_bf16_apple9, backend);
        run_dots_packed<nk_f16_k>(env, "dots_packed_f16_apple9", nk_dots_pack_size_f16_apple9, nk_dots_pack_f16_apple9,
                                  nk_dots_packed_f16_apple9, backend);
        run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_apple9", nk_dots_pack_size_e5m2_apple9,
                                   nk_dots_pack_e5m2_apple9, nk_dots_packed_e5m2_apple9, backend);
        run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_apple9", nk_dots_pack_size_e4m3_apple9,
                                   nk_dots_pack_e4m3_apple9, nk_dots_packed_e4m3_apple9, backend);
        run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_apple9", nk_dots_pack_size_e3m2_apple9,
                                   nk_dots_pack_e3m2_apple9, nk_dots_packed_e3m2_apple9, backend);
        run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_apple9", nk_dots_pack_size_e2m3_apple9,
                                   nk_dots_pack_e2m3_apple9, nk_dots_packed_e2m3_apple9, backend);
        run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_apple9", nk_dots_pack_size_e2m1_apple9,
                                   nk_dots_pack_e2m1_apple9, nk_dots_packed_e2m1_apple9, backend);

        run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_apple9", nk_dots_symmetric_bf16_apple9, backend);
        run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_apple9", nk_dots_symmetric_f16_apple9, backend);
        run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_apple9", nk_dots_symmetric_e4m3_apple9, backend);
    }
#endif
#if NUMKONG_TARGET_APPLE10
    if (!(enabled & nk_cap_apple10_k)) return;
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_apple10", nk_dots_pack_size_bf16_apple10,
                               nk_dots_pack_bf16_apple10, nk_dots_packed_bf16_apple10, backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_apple10", nk_dots_pack_size_f16_apple10, nk_dots_pack_f16_apple10,
                              nk_dots_packed_f16_apple10, backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_apple10", nk_dots_pack_size_e5m2_apple10,
                               nk_dots_pack_e5m2_apple10, nk_dots_packed_e5m2_apple10, backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_apple10", nk_dots_pack_size_e4m3_apple10,
                               nk_dots_pack_e4m3_apple10, nk_dots_packed_e4m3_apple10, backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_apple10", nk_dots_pack_size_e3m2_apple10,
                               nk_dots_pack_e3m2_apple10, nk_dots_packed_e3m2_apple10, backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_apple10", nk_dots_pack_size_e2m3_apple10,
                               nk_dots_pack_e2m3_apple10, nk_dots_packed_e2m3_apple10, backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_apple10", nk_dots_pack_size_e2m1_apple10,
                               nk_dots_pack_e2m1_apple10, nk_dots_packed_e2m1_apple10, backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_apple10", nk_dots_pack_size_i8_apple10, nk_dots_pack_i8_apple10,
                             nk_dots_packed_i8_apple10, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_apple10", nk_dots_pack_size_u8_apple10, nk_dots_pack_u8_apple10,
                             nk_dots_packed_u8_apple10, backend);

    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_apple10", nk_dots_symmetric_bf16_apple10, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_apple10", nk_dots_symmetric_f16_apple10, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_apple10", nk_dots_symmetric_e4m3_apple10, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_apple10", nk_dots_symmetric_i8_apple10, backend);
#endif
#endif // NUMKONG_WITH_METAL
}

} // namespace ashvardanian::numkong::bench
