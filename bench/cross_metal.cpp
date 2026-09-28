/**
 *  @file bench/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batch operation benchmarks for the Metal kernels, the twin of `cross_cuda.cu`.
 *
 *  Runs the drivers of `cross.hpp` through a @c metal_backend_t, each run on a queue of its own,
 *  over that queue's shared memory. Metal has no C-level events, so a timed window is wall time
 *  from its first commit to the synchronization after its last; calls commit as they are encoded,
 *  so the host's encoding overlaps the device's work and large shapes measure the device. Input
 *  sets rotate until their footprint is at least twice the system-level cache.
 */
#include <cstddef> // `std::size_t`, `std::ptrdiff_t`
#include <cstring> // `std::memcpy`, `std::memset`

#include <algorithm>   // `std::clamp`, `std::max`
#include <bit>         // `std::bit_ceil`, `std::bit_floor`
#include <chrono>      // `std::chrono::steady_clock`
#include <type_traits> // `std::true_type`, `std::false_type`

#include "numkong/metal.h" // `nk_metal_allocate`, `nk_metal_free`
#include "numkong/numkong.h"

#include "cross.hpp"

#if NUMKONG_WITH_METAL

namespace ashvardanian::numkong::bench {

/** Shared memory both the host and @c queue's kernels dereference, so any @c nk::vector factory
 *  can use it. A default-constructed allocator has no queue and hands out nothing. */
template <typename value_type_>
struct metal_shared_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::false_type;

    template <typename other_type_>
    struct rebind {
        using other = metal_shared_allocator<other_type_>;
    };

    nk_metal_queue_t *queue = nullptr;

    constexpr metal_shared_allocator() noexcept = default;
    constexpr explicit metal_shared_allocator(nk_metal_queue_t &queue) noexcept : queue(&queue) {}
    template <typename other_type_>
    constexpr metal_shared_allocator(metal_shared_allocator<other_type_> const &other) noexcept : queue(other.queue) {}

    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        if (!queue) return nullptr;
        return static_cast<value_type *>(nk_metal_allocate(queue, count * sizeof(value_type)));
    }
    void deallocate(value_type *pointer, std::size_t) noexcept {
        if (pointer) nk_metal_free(queue, pointer);
    }
    template <typename other_type_>
    constexpr bool operator==(metal_shared_allocator<other_type_> const &other) const noexcept {
        return queue == other.queue;
    }
};

/** Runs the Metal kernels on @c queue over its shared memory, timing windows of calls by wall
 *  clock around a synchronization. */
struct metal_backend_t {

    /** The allocator every kernel operand comes from, readable by the host once the queue is
     *  synchronized. */
    template <typename value_type_>
    using allocator = metal_shared_allocator<value_type_>;

    /** Where every call encodes and every operand is recorded. Each copy opens its own on first use
     *  and closes it with itself, so a registered benchmark holds no device resources. */
    mutable nk_metal_queue_t queue {};

    /** The first failure since the last synchronization. */
    nk_status_t status = nk_success_k;

    metal_backend_t() noexcept = default;
    metal_backend_t(metal_backend_t const &) noexcept {}
    metal_backend_t &operator=(metal_backend_t const &) = delete;
    ~metal_backend_t() noexcept {
        if (queue.device) nk_metal_queue_free(&queue);
    }

    /** The queue, opened by the first call that needs it. */
    nk_metal_queue_t &opened() const noexcept {
        if (!queue.device && nk_metal_queue_init(&queue, 0) != nk_success_k) queue = {};
        return queue;
    }

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return (row_bytes + 15) / 16 * 16; }

    /** Rotation sets of @p bytes_per_set each: enough to cover twice a 32 MB system-level cache,
     *  rounded to a power of two within the budget. */
    std::size_t input_sets(std::size_t bytes_per_set) const noexcept {
        std::size_t const cache_bytes = std::size_t(32) << 20;
        std::size_t const set_bytes = std::max(bytes_per_set, std::size_t(1));
        std::size_t const wanted = std::max(nk::divide_round_up(2 * cache_bytes, set_bytes), std::size_t(1));
        std::size_t const affordable = std::max(bench_config.budget_bytes / set_bytes, std::size_t(1));
        return std::bit_floor(std::clamp(std::bit_ceil(wanted), std::size_t(1), affordable));
    }

    /** Copies @p bytes once every queued call has finished with them. */
    void copy(void *destination, void const *source, std::size_t bytes) noexcept {
        keep(nk_metal_synchronize(&opened()));
        std::memcpy(destination, source, bytes);
    }

    /** Zeroes @p bytes once every queued call has finished with them. */
    void zero(void *destination, std::size_t bytes) noexcept {
        keep(nk_metal_synchronize(&opened()));
        std::memset(destination, 0, bytes);
    }

    /** Encodes @p kernel with @p arguments on the queue. */
    template <typename kernel_type_, typename... arguments_types_>
    void call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        keep(kernel(arguments..., static_cast<void *>(&opened())));
    }

    /** Times batches of @p launch lasting at least a millisecond each, returning the call count. */
    template <typename launch_type_>
    std::size_t time(bm::State &state, std::size_t sets_count, launch_type_ &launch) {
        using clock_t = std::chrono::steady_clock;
        auto const calibration_start = clock_t::now();
        for (std::size_t index = 0; index != sets_count; ++index) launch(index);
        keep(nk_metal_synchronize(&opened()));
        double const calibration_seconds = std::chrono::duration<double>(clock_t::now() - calibration_start).count();
        double const per_call_seconds = std::max(calibration_seconds / double(sets_count), 1e-7);
        std::size_t const batch = std::clamp<std::size_t>(std::size_t(1e-3 / per_call_seconds) + 1, 1, 1 << 16);

        std::size_t calls = 0;
        for (auto _ : state) {
            auto const window_start = clock_t::now();
            for (std::size_t index = 0; index != batch && status == nk_success_k; ++index, ++calls)
                launch(calls & (sets_count - 1));
            keep(nk_metal_synchronize(&opened()));
            if (status != nk_success_k) break;
            state.SetIterationTime(std::chrono::duration<double>(clock_t::now() - window_start).count());
        }
        return calls;
    }

    /** Waits for the queue, returning the name of the first failure since the last call, or
     *  @c nullptr. */
    char const *synchronize() noexcept {
        keep(nk_metal_synchronize(&opened()));
        nk_status_t const failure = status;
        status = nk_success_k;
        return failure == nk_success_k ? nullptr : nk_status_to_string(failure);
    }

    /** Remembers @p result unless an earlier failure is pending. */
    void keep(nk_status_t result) noexcept {
        if (status == nk_success_k) status = result;
    }

    /** Reports the timed windows instead of the loop's own wall time. */
    static void configure(bm::internal::Benchmark *benchmark) { benchmark->UseManualTime(); }
};

/** Blocks recorded on @p backend's own queue. */
template <typename value_type_>
metal_shared_allocator<value_type_> allocator_of(metal_backend_t const &backend) noexcept {
    return metal_shared_allocator<value_type_>(backend.opened());
}

} // namespace ashvardanian::numkong::bench

using namespace ashvardanian::numkong::bench;

#endif // NUMKONG_WITH_METAL

/** Every Metal baseline entry point beside every Apple9 and Apple10 one, so the matrix units'
 *  speedup shows, on devices whose families include each. */
void bench_cross_metal() {
#if NUMKONG_WITH_METAL
    nk_capability_t detected = 0, enabled = 0;
    if (nk_metal_capabilities_detected(0, &detected) != nk_success_k || !detected)
        return fmt::println("- Metal: no device");
    char families[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_name_capabilities(detected, families, sizeof(families));
    fmt::println("- Metal: {}", families);

    if (nk_metal_capabilities_enabled(0, &enabled) != nk_success_k) enabled = 0;
    metal_backend_t const backend;
    if (enabled & nk_cap_metal_k) {
        run_dots_packed<nk_bf16_k>("dots_packed_bf16_metal", nk_dots_pack_size_bf16_metal, nk_dots_pack_bf16_metal,
                                   nk_dots_packed_bf16_metal, backend);
        run_dots_packed<nk_f16_k>("dots_packed_f16_metal", nk_dots_pack_size_f16_metal, nk_dots_pack_f16_metal,
                                  nk_dots_packed_f16_metal, backend);
        run_dots_packed<nk_e5m2_k>("dots_packed_e5m2_metal", nk_dots_pack_size_e5m2_metal, nk_dots_pack_e5m2_metal,
                                   nk_dots_packed_e5m2_metal, backend);
        run_dots_packed<nk_e4m3_k>("dots_packed_e4m3_metal", nk_dots_pack_size_e4m3_metal, nk_dots_pack_e4m3_metal,
                                   nk_dots_packed_e4m3_metal, backend);
        run_dots_packed<nk_e3m2_k>("dots_packed_e3m2_metal", nk_dots_pack_size_e3m2_metal, nk_dots_pack_e3m2_metal,
                                   nk_dots_packed_e3m2_metal, backend);
        run_dots_packed<nk_e2m3_k>("dots_packed_e2m3_metal", nk_dots_pack_size_e2m3_metal, nk_dots_pack_e2m3_metal,
                                   nk_dots_packed_e2m3_metal, backend);
        run_dots_packed<nk_e2m1_k>("dots_packed_e2m1_metal", nk_dots_pack_size_e2m1_metal, nk_dots_pack_e2m1_metal,
                                   nk_dots_packed_e2m1_metal, backend);
        run_dots_packed<nk_i8_k>("dots_packed_i8_metal", nk_dots_pack_size_i8_metal, nk_dots_pack_i8_metal,
                                 nk_dots_packed_i8_metal, backend);
        run_dots_packed<nk_u8_k>("dots_packed_u8_metal", nk_dots_pack_size_u8_metal, nk_dots_pack_u8_metal,
                                 nk_dots_packed_u8_metal, backend);

        run_dots_symmetric<nk_bf16_k>("dots_symmetric_bf16_metal", nk_dots_symmetric_bf16_metal, backend);
        run_dots_symmetric<nk_f16_k>("dots_symmetric_f16_metal", nk_dots_symmetric_f16_metal, backend);
        run_dots_symmetric<nk_e4m3_k>("dots_symmetric_e4m3_metal", nk_dots_symmetric_e4m3_metal, backend);
        run_dots_symmetric<nk_i8_k>("dots_symmetric_i8_metal", nk_dots_symmetric_i8_metal, backend);
    }
#if NUMKONG_TARGET_APPLE9
    if (enabled & nk_cap_apple9_k) {
        run_dots_packed<nk_bf16_k>("dots_packed_bf16_apple9", nk_dots_pack_size_bf16_apple9, nk_dots_pack_bf16_apple9,
                                   nk_dots_packed_bf16_apple9, backend);
        run_dots_packed<nk_f16_k>("dots_packed_f16_apple9", nk_dots_pack_size_f16_apple9, nk_dots_pack_f16_apple9,
                                  nk_dots_packed_f16_apple9, backend);
        run_dots_packed<nk_e5m2_k>("dots_packed_e5m2_apple9", nk_dots_pack_size_e5m2_apple9, nk_dots_pack_e5m2_apple9,
                                   nk_dots_packed_e5m2_apple9, backend);
        run_dots_packed<nk_e4m3_k>("dots_packed_e4m3_apple9", nk_dots_pack_size_e4m3_apple9, nk_dots_pack_e4m3_apple9,
                                   nk_dots_packed_e4m3_apple9, backend);
        run_dots_packed<nk_e3m2_k>("dots_packed_e3m2_apple9", nk_dots_pack_size_e3m2_apple9, nk_dots_pack_e3m2_apple9,
                                   nk_dots_packed_e3m2_apple9, backend);
        run_dots_packed<nk_e2m3_k>("dots_packed_e2m3_apple9", nk_dots_pack_size_e2m3_apple9, nk_dots_pack_e2m3_apple9,
                                   nk_dots_packed_e2m3_apple9, backend);
        run_dots_packed<nk_e2m1_k>("dots_packed_e2m1_apple9", nk_dots_pack_size_e2m1_apple9, nk_dots_pack_e2m1_apple9,
                                   nk_dots_packed_e2m1_apple9, backend);

        run_dots_symmetric<nk_bf16_k>("dots_symmetric_bf16_apple9", nk_dots_symmetric_bf16_apple9, backend);
        run_dots_symmetric<nk_f16_k>("dots_symmetric_f16_apple9", nk_dots_symmetric_f16_apple9, backend);
        run_dots_symmetric<nk_e4m3_k>("dots_symmetric_e4m3_apple9", nk_dots_symmetric_e4m3_apple9, backend);
    }
#endif
#if NUMKONG_TARGET_APPLE10
    if (!(enabled & nk_cap_apple10_k)) return;
    run_dots_packed<nk_bf16_k>("dots_packed_bf16_apple10", nk_dots_pack_size_bf16_apple10, nk_dots_pack_bf16_apple10,
                               nk_dots_packed_bf16_apple10, backend);
    run_dots_packed<nk_f16_k>("dots_packed_f16_apple10", nk_dots_pack_size_f16_apple10, nk_dots_pack_f16_apple10,
                              nk_dots_packed_f16_apple10, backend);
    run_dots_packed<nk_e5m2_k>("dots_packed_e5m2_apple10", nk_dots_pack_size_e5m2_apple10, nk_dots_pack_e5m2_apple10,
                               nk_dots_packed_e5m2_apple10, backend);
    run_dots_packed<nk_e4m3_k>("dots_packed_e4m3_apple10", nk_dots_pack_size_e4m3_apple10, nk_dots_pack_e4m3_apple10,
                               nk_dots_packed_e4m3_apple10, backend);
    run_dots_packed<nk_e3m2_k>("dots_packed_e3m2_apple10", nk_dots_pack_size_e3m2_apple10, nk_dots_pack_e3m2_apple10,
                               nk_dots_packed_e3m2_apple10, backend);
    run_dots_packed<nk_e2m3_k>("dots_packed_e2m3_apple10", nk_dots_pack_size_e2m3_apple10, nk_dots_pack_e2m3_apple10,
                               nk_dots_packed_e2m3_apple10, backend);
    run_dots_packed<nk_e2m1_k>("dots_packed_e2m1_apple10", nk_dots_pack_size_e2m1_apple10, nk_dots_pack_e2m1_apple10,
                               nk_dots_packed_e2m1_apple10, backend);
    run_dots_packed<nk_i8_k>("dots_packed_i8_apple10", nk_dots_pack_size_i8_apple10, nk_dots_pack_i8_apple10,
                             nk_dots_packed_i8_apple10, backend);
    run_dots_packed<nk_u8_k>("dots_packed_u8_apple10", nk_dots_pack_size_u8_apple10, nk_dots_pack_u8_apple10,
                             nk_dots_packed_u8_apple10, backend);

    run_dots_symmetric<nk_bf16_k>("dots_symmetric_bf16_apple10", nk_dots_symmetric_bf16_apple10, backend);
    run_dots_symmetric<nk_f16_k>("dots_symmetric_f16_apple10", nk_dots_symmetric_f16_apple10, backend);
    run_dots_symmetric<nk_e4m3_k>("dots_symmetric_e4m3_apple10", nk_dots_symmetric_e4m3_apple10, backend);
    run_dots_symmetric<nk_i8_k>("dots_symmetric_i8_apple10", nk_dots_symmetric_i8_apple10, backend);
#endif
#endif // NUMKONG_WITH_METAL
}
