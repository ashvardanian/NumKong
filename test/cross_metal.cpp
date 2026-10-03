/**
 *  @file test/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Batch operation tests - Metal capabilities.
 *
 *  Runs the dots scenarios of `cross.hpp` through a @c metal_backend_t for the Metal baseline and
 *  every Apple GPU family the device runs, against the serial `nk::` references on the host.
 */
#include "numkong/metal.h"
#include "numkong/memory.h" // `nk_memory_allocate_unified_metal`

#include "harness.hpp" // `error_stats_section_t`, `call_best`
#include "cross.hpp"   // `test_dots_packed`

namespace ashvardanian::numkong::test {

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

/** Runs the Metal kernels on the null stream, the library's queue on the system default device,
 *  over its unified memory, keeping the first failed status. */
struct metal_backend_t {

    /** The allocator every kernel operand comes from, readable by the host once the stream is
     *  synchronized. */
    template <typename value_type_>
    using allocator = metal_shared_allocator<value_type_>;

    /** The first failure since the last synchronization. */
    nk_status_t status = nk_success_k;

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return (row_bytes + 15) / 16 * 16; }

    /** Copies @p bytes once every queued call has finished with them. */
    void copy(void *destination, void const *source, std::size_t bytes) noexcept {
        keep(nk_stream_synchronize_metal(nullptr));
        std::memcpy(destination, source, bytes);
    }

    /** Zeroes @p bytes once every queued call has finished with them. */
    void zero(void *destination, std::size_t bytes) noexcept {
        keep(nk_stream_synchronize_metal(nullptr));
        std::memset(destination, 0, bytes);
    }

    /** Encodes @p kernel with @p arguments on the null stream. */
    template <typename kernel_type_, typename... arguments_types_>
    void call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        keep(kernel(arguments..., nullptr));
    }

    /** Calls @p kernel on operands it must refuse, reporting whether it returned
     *  @c nk_misaligned_k unencoded. */
    template <typename kernel_type_, typename... arguments_types_>
    bool refuses_misaligned(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., nullptr) == nk_misaligned_k;
    }

    /** Waits for the stream, returning the name of the first failure since the last call, or
     *  @c nullptr. */
    char const *synchronize() noexcept {
        keep(nk_stream_synchronize_metal(nullptr));
        nk_status_t const failure = status;
        status = nk_success_k;
        return failure == nk_success_k ? nullptr : nk_status_name(failure);
    }

    /** Remembers @p result unless an earlier failure is pending. */
    void keep(nk_status_t result) noexcept {
        if (status == nk_success_k) status = result;
    }
};

/** The capabilities of the first Metal device, where every backend encodes. */
inline nk_capability_t metal_capabilities() noexcept {
    auto const device = nk::device_t::make(nk::device_kind_t::metal_k, 0);
    return device ? device.value.capabilities_enabled().value : 0;
}

/** The dispatch point @p best_ in the shape of its capability kernels, over GPU 0's capabilities,
 *  where the backend encodes: callable like them, and convertible to their function pointers. */
template <auto best_>
inline constexpr auto gpu_best =
    [](auto... arguments) noexcept { return call_best<best_>(metal_capabilities(), arguments...); };

static error_stats_t test_metal_deferred_free(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    void *pointer = nullptr;
    stats.expect(nk_memory_allocate_unified_metal(16, &pointer, nullptr));
    if (!pointer) return stats;
    nk_metal_context_t *const context = nk_metal_context_(nullptr);
    os_unfair_lock_lock(&context->lock);
    nk_metal_pending_t *const grown = (nk_metal_pending_t *)nk_metal_reserve_(
        context->pending, context->pending_count, &context->pending_capacity, sizeof(nk_metal_pending_t));
    if (grown) {
        context->pending = grown;
        nk_metal_pending_t *pending = &grown[context->pending_count++];
        memset(pending, 0, sizeof(*pending));
        pending->queue = nk_metal_get_(context->queue, "retain");
        pending->waiters = 1; // A synchronizer has detached commands and is waiting outside the lock.
    }
    os_unfair_lock_unlock(&context->lock);
    stats.expect(grown != nullptr, "pending entry allocation failed");
    stats.expect(nk_memory_free_unified_metal(pointer, 16, nullptr));
    if (!grown) return stats;
    stats.expect(nk_stream_synchronize_metal(nullptr));
    os_unfair_lock_lock(&context->lock);
    nk_metal_pending_t *pending = nk_metal_pending_(context, context->queue);
    bool const retained = pending && pending->waiters == 1 && pending->frees_count == 1;
    if (pending) --pending->waiters;
    os_unfair_lock_unlock(&context->lock);
    stats.expect(retained, "another synchronizer released a buffer while a waiter was active");
    stats.expect(nk_stream_synchronize_metal(nullptr));
    return stats;
}

/** Every Metal baseline entry point, on any Apple GPU of family 7 or newer. */
static void test_cross_metal_baseline(error_stats_section_t &check) {
    metal_backend_t const backend {};
    check.section("Cross Metal", nk_cap_metal_k);
    check("deferred_free_metal", test_metal_deferred_free);
    check("dots_packed_i8_metal", test_dots_packed<i8_t, metal_backend_t>, backend, nk_dots_pack_size_i8_metal,
          nk_dots_pack_i8_metal, nk_dots_packed_i8_metal);
    check("dots_pack_i8_metal",
          test_dots_pack_layout<i8_t, metal_backend_t, nk_dots_pack_size_i8_metal, nk_dots_packed_shape_i8_metal,
                                nk_dots_pack_i8_metal>,
          backend);
    check("dots_contract_i8_metal",
          test_dots_launch_contract<i8_t, metal_backend_t, nk_dots_pack_size_i8_metal, nk_dots_packed_i8_metal,
                                    nk_dots_symmetric_i8_metal>,
          backend);
    check("dots_symmetric_i8_metal", test_dots_symmetric<i8_t, metal_backend_t>, backend, nk_dots_symmetric_i8_metal);
    check("dots_packed_u8_metal", test_dots_packed<u8_t, metal_backend_t>, backend, nk_dots_pack_size_u8_metal,
          nk_dots_pack_u8_metal, nk_dots_packed_u8_metal);
    check("dots_pack_u8_metal",
          test_dots_pack_layout<u8_t, metal_backend_t, nk_dots_pack_size_u8_metal, nk_dots_packed_shape_u8_metal,
                                nk_dots_pack_u8_metal>,
          backend);
    check("dots_contract_u8_metal",
          test_dots_launch_contract<u8_t, metal_backend_t, nk_dots_pack_size_u8_metal, nk_dots_packed_u8_metal,
                                    nk_dots_symmetric_u8_metal>,
          backend);
    check("dots_symmetric_u8_metal", test_dots_symmetric<u8_t, metal_backend_t>, backend, nk_dots_symmetric_u8_metal);
    check("dots_packed_i4_metal", test_dots_packed<i4x2_t, metal_backend_t>, backend, nk_dots_pack_size_i4_metal,
          nk_dots_pack_i4_metal, nk_dots_packed_i4_metal);
    check("dots_pack_i4_metal",
          test_dots_pack_layout<i4x2_t, metal_backend_t, nk_dots_pack_size_i4_metal, nk_dots_packed_shape_i4_metal,
                                nk_dots_pack_i4_metal>,
          backend);
    check("dots_contract_i4_metal",
          test_dots_launch_contract<i4x2_t, metal_backend_t, nk_dots_pack_size_i4_metal, nk_dots_packed_i4_metal,
                                    nk_dots_symmetric_i4_metal>,
          backend);
    check("dots_symmetric_i4_metal", test_dots_symmetric<i4x2_t, metal_backend_t>, backend, nk_dots_symmetric_i4_metal);
    check("dots_packed_u4_metal", test_dots_packed<u4x2_t, metal_backend_t>, backend, nk_dots_pack_size_u4_metal,
          nk_dots_pack_u4_metal, nk_dots_packed_u4_metal);
    check("dots_pack_u4_metal",
          test_dots_pack_layout<u4x2_t, metal_backend_t, nk_dots_pack_size_u4_metal, nk_dots_packed_shape_u4_metal,
                                nk_dots_pack_u4_metal>,
          backend);
    check("dots_contract_u4_metal",
          test_dots_launch_contract<u4x2_t, metal_backend_t, nk_dots_pack_size_u4_metal, nk_dots_packed_u4_metal,
                                    nk_dots_symmetric_u4_metal>,
          backend);
    check("dots_symmetric_u4_metal", test_dots_symmetric<u4x2_t, metal_backend_t>, backend, nk_dots_symmetric_u4_metal);
    check("dots_packed_f16_metal", test_dots_packed<f16_t, metal_backend_t>, backend, nk_dots_pack_size_f16_metal,
          nk_dots_pack_f16_metal, nk_dots_packed_f16_metal);
    check("dots_pack_f16_metal",
          test_dots_pack_layout<f16_t, metal_backend_t, nk_dots_pack_size_f16_metal, nk_dots_packed_shape_f16_metal,
                                nk_dots_pack_f16_metal>,
          backend);
    check("dots_contract_f16_metal",
          test_dots_launch_contract<f16_t, metal_backend_t, nk_dots_pack_size_f16_metal, nk_dots_packed_f16_metal,
                                    nk_dots_symmetric_f16_metal>,
          backend);
    check("dots_symmetric_f16_metal", test_dots_symmetric<f16_t, metal_backend_t>, backend,
          nk_dots_symmetric_f16_metal);
    check("dots_packed_bf16_metal", test_dots_packed<bf16_t, metal_backend_t>, backend, nk_dots_pack_size_bf16_metal,
          nk_dots_pack_bf16_metal, nk_dots_packed_bf16_metal);
    check("dots_pack_bf16_metal",
          test_dots_pack_layout<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_metal, nk_dots_packed_shape_bf16_metal,
                                nk_dots_pack_bf16_metal>,
          backend);
    check("dots_contract_bf16_metal",
          test_dots_launch_contract<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_metal, nk_dots_packed_bf16_metal,
                                    nk_dots_symmetric_bf16_metal>,
          backend);
    check("dots_symmetric_bf16_metal", test_dots_symmetric<bf16_t, metal_backend_t>, backend,
          nk_dots_symmetric_bf16_metal);
    check("dots_packed_e4m3_metal", test_dots_packed<e4m3_t, metal_backend_t>, backend, nk_dots_pack_size_e4m3_metal,
          nk_dots_pack_e4m3_metal, nk_dots_packed_e4m3_metal);
    check("dots_pack_e4m3_metal",
          test_dots_pack_layout<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_metal, nk_dots_packed_shape_e4m3_metal,
                                nk_dots_pack_e4m3_metal>,
          backend);
    check("dots_contract_e4m3_metal",
          test_dots_launch_contract<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_metal, nk_dots_packed_e4m3_metal,
                                    nk_dots_symmetric_e4m3_metal>,
          backend);
    check("dots_symmetric_e4m3_metal", test_dots_symmetric<e4m3_t, metal_backend_t>, backend,
          nk_dots_symmetric_e4m3_metal);
    check("dots_packed_e5m2_metal", test_dots_packed<e5m2_t, metal_backend_t>, backend, nk_dots_pack_size_e5m2_metal,
          nk_dots_pack_e5m2_metal, nk_dots_packed_e5m2_metal);
    check("dots_pack_e5m2_metal",
          test_dots_pack_layout<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_metal, nk_dots_packed_shape_e5m2_metal,
                                nk_dots_pack_e5m2_metal>,
          backend);
    check("dots_contract_e5m2_metal",
          test_dots_launch_contract<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_metal, nk_dots_packed_e5m2_metal,
                                    nk_dots_symmetric_e5m2_metal>,
          backend);
    check("dots_symmetric_e5m2_metal", test_dots_symmetric<e5m2_t, metal_backend_t>, backend,
          nk_dots_symmetric_e5m2_metal);
    check("dots_packed_e3m2_metal", test_dots_packed<e3m2_t, metal_backend_t>, backend, nk_dots_pack_size_e3m2_metal,
          nk_dots_pack_e3m2_metal, nk_dots_packed_e3m2_metal);
    check("dots_pack_e3m2_metal",
          test_dots_pack_layout<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_metal, nk_dots_packed_shape_e3m2_metal,
                                nk_dots_pack_e3m2_metal>,
          backend);
    check("dots_contract_e3m2_metal",
          test_dots_launch_contract<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_metal, nk_dots_packed_e3m2_metal,
                                    nk_dots_symmetric_e3m2_metal>,
          backend);
    check("dots_symmetric_e3m2_metal", test_dots_symmetric<e3m2_t, metal_backend_t>, backend,
          nk_dots_symmetric_e3m2_metal);
    check("dots_packed_e2m3_metal", test_dots_packed<e2m3_t, metal_backend_t>, backend, nk_dots_pack_size_e2m3_metal,
          nk_dots_pack_e2m3_metal, nk_dots_packed_e2m3_metal);
    check("dots_pack_e2m3_metal",
          test_dots_pack_layout<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_metal, nk_dots_packed_shape_e2m3_metal,
                                nk_dots_pack_e2m3_metal>,
          backend);
    check("dots_contract_e2m3_metal",
          test_dots_launch_contract<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_metal, nk_dots_packed_e2m3_metal,
                                    nk_dots_symmetric_e2m3_metal>,
          backend);
    check("dots_symmetric_e2m3_metal", test_dots_symmetric<e2m3_t, metal_backend_t>, backend,
          nk_dots_symmetric_e2m3_metal);
    check("dots_packed_e2m1_metal", test_dots_packed<e2m1x2_t, metal_backend_t>, backend, nk_dots_pack_size_e2m1_metal,
          nk_dots_pack_e2m1_metal, nk_dots_packed_e2m1_metal);
    check("dots_pack_e2m1_metal",
          test_dots_pack_layout<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_metal,
                                nk_dots_packed_shape_e2m1_metal, nk_dots_pack_e2m1_metal>,
          backend);
    check("dots_contract_e2m1_metal",
          test_dots_launch_contract<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_metal, nk_dots_packed_e2m1_metal,
                                    nk_dots_symmetric_e2m1_metal>,
          backend);
    check("dots_symmetric_e2m1_metal", test_dots_symmetric<e2m1x2_t, metal_backend_t>, backend,
          nk_dots_symmetric_e2m1_metal);
}

/** Every Apple9 entry point, on devices whose families include it. */
static void test_cross_apple9([[maybe_unused]] error_stats_section_t &check) {
#if NUMKONG_TARGET_APPLE9
    metal_backend_t const backend {};
    check.section("Cross Apple9", nk_cap_apple9_k);
    check("dots_packed_f16_apple9", test_dots_packed<f16_t, metal_backend_t>, backend, nk_dots_pack_size_f16_apple9,
          nk_dots_pack_f16_apple9, nk_dots_packed_f16_apple9);
    check("dots_pack_f16_apple9",
          test_dots_pack_layout<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple9, nk_dots_packed_shape_f16_apple9,
                                nk_dots_pack_f16_apple9>,
          backend);
    check("dots_contract_f16_apple9",
          test_dots_launch_contract<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple9, nk_dots_packed_f16_apple9,
                                    nk_dots_symmetric_f16_apple9>,
          backend);
    check("dots_symmetric_f16_apple9", test_dots_symmetric<f16_t, metal_backend_t>, backend,
          nk_dots_symmetric_f16_apple9);
    check("dots_packed_bf16_apple9", test_dots_packed<bf16_t, metal_backend_t>, backend, nk_dots_pack_size_bf16_apple9,
          nk_dots_pack_bf16_apple9, nk_dots_packed_bf16_apple9);
    check("dots_pack_bf16_apple9",
          test_dots_pack_layout<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple9,
                                nk_dots_packed_shape_bf16_apple9, nk_dots_pack_bf16_apple9>,
          backend);
    check("dots_contract_bf16_apple9",
          test_dots_launch_contract<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple9, nk_dots_packed_bf16_apple9,
                                    nk_dots_symmetric_bf16_apple9>,
          backend);
    check("dots_symmetric_bf16_apple9", test_dots_symmetric<bf16_t, metal_backend_t>, backend,
          nk_dots_symmetric_bf16_apple9);
    check("dots_packed_e4m3_apple9", test_dots_packed<e4m3_t, metal_backend_t>, backend, nk_dots_pack_size_e4m3_apple9,
          nk_dots_pack_e4m3_apple9, nk_dots_packed_e4m3_apple9);
    check("dots_pack_e4m3_apple9",
          test_dots_pack_layout<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple9,
                                nk_dots_packed_shape_e4m3_apple9, nk_dots_pack_e4m3_apple9>,
          backend);
    check("dots_contract_e4m3_apple9",
          test_dots_launch_contract<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple9, nk_dots_packed_e4m3_apple9,
                                    nk_dots_symmetric_e4m3_apple9>,
          backend);
    check("dots_symmetric_e4m3_apple9", test_dots_symmetric<e4m3_t, metal_backend_t>, backend,
          nk_dots_symmetric_e4m3_apple9);
    check("dots_packed_e5m2_apple9", test_dots_packed<e5m2_t, metal_backend_t>, backend, nk_dots_pack_size_e5m2_apple9,
          nk_dots_pack_e5m2_apple9, nk_dots_packed_e5m2_apple9);
    check("dots_pack_e5m2_apple9",
          test_dots_pack_layout<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple9,
                                nk_dots_packed_shape_e5m2_apple9, nk_dots_pack_e5m2_apple9>,
          backend);
    check("dots_contract_e5m2_apple9",
          test_dots_launch_contract<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple9, nk_dots_packed_e5m2_apple9,
                                    nk_dots_symmetric_e5m2_apple9>,
          backend);
    check("dots_symmetric_e5m2_apple9", test_dots_symmetric<e5m2_t, metal_backend_t>, backend,
          nk_dots_symmetric_e5m2_apple9);
    check("dots_packed_e3m2_apple9", test_dots_packed<e3m2_t, metal_backend_t>, backend, nk_dots_pack_size_e3m2_apple9,
          nk_dots_pack_e3m2_apple9, nk_dots_packed_e3m2_apple9);
    check("dots_pack_e3m2_apple9",
          test_dots_pack_layout<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple9,
                                nk_dots_packed_shape_e3m2_apple9, nk_dots_pack_e3m2_apple9>,
          backend);
    check("dots_contract_e3m2_apple9",
          test_dots_launch_contract<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple9, nk_dots_packed_e3m2_apple9,
                                    nk_dots_symmetric_e3m2_apple9>,
          backend);
    check("dots_symmetric_e3m2_apple9", test_dots_symmetric<e3m2_t, metal_backend_t>, backend,
          nk_dots_symmetric_e3m2_apple9);
    check("dots_packed_e2m3_apple9", test_dots_packed<e2m3_t, metal_backend_t>, backend, nk_dots_pack_size_e2m3_apple9,
          nk_dots_pack_e2m3_apple9, nk_dots_packed_e2m3_apple9);
    check("dots_pack_e2m3_apple9",
          test_dots_pack_layout<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple9,
                                nk_dots_packed_shape_e2m3_apple9, nk_dots_pack_e2m3_apple9>,
          backend);
    check("dots_contract_e2m3_apple9",
          test_dots_launch_contract<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple9, nk_dots_packed_e2m3_apple9,
                                    nk_dots_symmetric_e2m3_apple9>,
          backend);
    check("dots_symmetric_e2m3_apple9", test_dots_symmetric<e2m3_t, metal_backend_t>, backend,
          nk_dots_symmetric_e2m3_apple9);
    check("dots_packed_e2m1_apple9", test_dots_packed<e2m1x2_t, metal_backend_t>, backend,
          nk_dots_pack_size_e2m1_apple9, nk_dots_pack_e2m1_apple9, nk_dots_packed_e2m1_apple9);
    check("dots_pack_e2m1_apple9",
          test_dots_pack_layout<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple9,
                                nk_dots_packed_shape_e2m1_apple9, nk_dots_pack_e2m1_apple9>,
          backend);
    check("dots_contract_e2m1_apple9",
          test_dots_launch_contract<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple9,
                                    nk_dots_packed_e2m1_apple9, nk_dots_symmetric_e2m1_apple9>,
          backend);
    check("dots_symmetric_e2m1_apple9", test_dots_symmetric<e2m1x2_t, metal_backend_t>, backend,
          nk_dots_symmetric_e2m1_apple9);
#endif
}

/** Every Apple10 entry point, on devices whose families include it. */
static void test_cross_apple10([[maybe_unused]] error_stats_section_t &check) {
#if NUMKONG_TARGET_APPLE10
    metal_backend_t const backend {};
    check.section("Cross Apple10", nk_cap_apple10_k);
    check("dots_packed_i8_apple10", test_dots_packed<i8_t, metal_backend_t>, backend, nk_dots_pack_size_i8_apple10,
          nk_dots_pack_i8_apple10, nk_dots_packed_i8_apple10);
    check("dots_pack_i8_apple10",
          test_dots_pack_layout<i8_t, metal_backend_t, nk_dots_pack_size_i8_apple10, nk_dots_packed_shape_i8_apple10,
                                nk_dots_pack_i8_apple10>,
          backend);
    check("dots_contract_i8_apple10",
          test_dots_launch_contract<i8_t, metal_backend_t, nk_dots_pack_size_i8_apple10, nk_dots_packed_i8_apple10,
                                    nk_dots_symmetric_i8_apple10>,
          backend);
    check("dots_symmetric_i8_apple10", test_dots_symmetric<i8_t, metal_backend_t>, backend,
          nk_dots_symmetric_i8_apple10);
    check("dots_packed_u8_apple10", test_dots_packed<u8_t, metal_backend_t>, backend, nk_dots_pack_size_u8_apple10,
          nk_dots_pack_u8_apple10, nk_dots_packed_u8_apple10);
    check("dots_pack_u8_apple10",
          test_dots_pack_layout<u8_t, metal_backend_t, nk_dots_pack_size_u8_apple10, nk_dots_packed_shape_u8_apple10,
                                nk_dots_pack_u8_apple10>,
          backend);
    check("dots_contract_u8_apple10",
          test_dots_launch_contract<u8_t, metal_backend_t, nk_dots_pack_size_u8_apple10, nk_dots_packed_u8_apple10,
                                    nk_dots_symmetric_u8_apple10>,
          backend);
    check("dots_symmetric_u8_apple10", test_dots_symmetric<u8_t, metal_backend_t>, backend,
          nk_dots_symmetric_u8_apple10);
    check("dots_packed_f16_apple10", test_dots_packed<f16_t, metal_backend_t>, backend, nk_dots_pack_size_f16_apple10,
          nk_dots_pack_f16_apple10, nk_dots_packed_f16_apple10);
    check("dots_pack_f16_apple10",
          test_dots_pack_layout<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple10, nk_dots_packed_shape_f16_apple10,
                                nk_dots_pack_f16_apple10>,
          backend);
    check("dots_contract_f16_apple10",
          test_dots_launch_contract<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple10, nk_dots_packed_f16_apple10,
                                    nk_dots_symmetric_f16_apple10>,
          backend);
    check("dots_symmetric_f16_apple10", test_dots_symmetric<f16_t, metal_backend_t>, backend,
          nk_dots_symmetric_f16_apple10);
    check("dots_packed_bf16_apple10", test_dots_packed<bf16_t, metal_backend_t>, backend,
          nk_dots_pack_size_bf16_apple10, nk_dots_pack_bf16_apple10, nk_dots_packed_bf16_apple10);
    check("dots_pack_bf16_apple10",
          test_dots_pack_layout<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple10,
                                nk_dots_packed_shape_bf16_apple10, nk_dots_pack_bf16_apple10>,
          backend);
    check("dots_contract_bf16_apple10",
          test_dots_launch_contract<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple10,
                                    nk_dots_packed_bf16_apple10, nk_dots_symmetric_bf16_apple10>,
          backend);
    check("dots_symmetric_bf16_apple10", test_dots_symmetric<bf16_t, metal_backend_t>, backend,
          nk_dots_symmetric_bf16_apple10);
    check("dots_packed_e4m3_apple10", test_dots_packed<e4m3_t, metal_backend_t>, backend,
          nk_dots_pack_size_e4m3_apple10, nk_dots_pack_e4m3_apple10, nk_dots_packed_e4m3_apple10);
    check("dots_pack_e4m3_apple10",
          test_dots_pack_layout<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple10,
                                nk_dots_packed_shape_e4m3_apple10, nk_dots_pack_e4m3_apple10>,
          backend);
    check("dots_contract_e4m3_apple10",
          test_dots_launch_contract<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple10,
                                    nk_dots_packed_e4m3_apple10, nk_dots_symmetric_e4m3_apple10>,
          backend);
    check("dots_symmetric_e4m3_apple10", test_dots_symmetric<e4m3_t, metal_backend_t>, backend,
          nk_dots_symmetric_e4m3_apple10);
    check("dots_packed_e5m2_apple10", test_dots_packed<e5m2_t, metal_backend_t>, backend,
          nk_dots_pack_size_e5m2_apple10, nk_dots_pack_e5m2_apple10, nk_dots_packed_e5m2_apple10);
    check("dots_pack_e5m2_apple10",
          test_dots_pack_layout<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple10,
                                nk_dots_packed_shape_e5m2_apple10, nk_dots_pack_e5m2_apple10>,
          backend);
    check("dots_contract_e5m2_apple10",
          test_dots_launch_contract<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple10,
                                    nk_dots_packed_e5m2_apple10, nk_dots_symmetric_e5m2_apple10>,
          backend);
    check("dots_symmetric_e5m2_apple10", test_dots_symmetric<e5m2_t, metal_backend_t>, backend,
          nk_dots_symmetric_e5m2_apple10);
    check("dots_packed_e3m2_apple10", test_dots_packed<e3m2_t, metal_backend_t>, backend,
          nk_dots_pack_size_e3m2_apple10, nk_dots_pack_e3m2_apple10, nk_dots_packed_e3m2_apple10);
    check("dots_pack_e3m2_apple10",
          test_dots_pack_layout<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple10,
                                nk_dots_packed_shape_e3m2_apple10, nk_dots_pack_e3m2_apple10>,
          backend);
    check("dots_contract_e3m2_apple10",
          test_dots_launch_contract<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple10,
                                    nk_dots_packed_e3m2_apple10, nk_dots_symmetric_e3m2_apple10>,
          backend);
    check("dots_symmetric_e3m2_apple10", test_dots_symmetric<e3m2_t, metal_backend_t>, backend,
          nk_dots_symmetric_e3m2_apple10);
    check("dots_packed_e2m3_apple10", test_dots_packed<e2m3_t, metal_backend_t>, backend,
          nk_dots_pack_size_e2m3_apple10, nk_dots_pack_e2m3_apple10, nk_dots_packed_e2m3_apple10);
    check("dots_pack_e2m3_apple10",
          test_dots_pack_layout<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple10,
                                nk_dots_packed_shape_e2m3_apple10, nk_dots_pack_e2m3_apple10>,
          backend);
    check("dots_contract_e2m3_apple10",
          test_dots_launch_contract<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple10,
                                    nk_dots_packed_e2m3_apple10, nk_dots_symmetric_e2m3_apple10>,
          backend);
    check("dots_symmetric_e2m3_apple10", test_dots_symmetric<e2m3_t, metal_backend_t>, backend,
          nk_dots_symmetric_e2m3_apple10);
    check("dots_packed_e2m1_apple10", test_dots_packed<e2m1x2_t, metal_backend_t>, backend,
          nk_dots_pack_size_e2m1_apple10, nk_dots_pack_e2m1_apple10, nk_dots_packed_e2m1_apple10);
    check("dots_pack_e2m1_apple10",
          test_dots_pack_layout<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple10,
                                nk_dots_packed_shape_e2m1_apple10, nk_dots_pack_e2m1_apple10>,
          backend);
    check("dots_contract_e2m1_apple10",
          test_dots_launch_contract<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple10,
                                    nk_dots_packed_e2m1_apple10, nk_dots_symmetric_e2m1_apple10>,
          backend);
    check("dots_symmetric_e2m1_apple10", test_dots_symmetric<e2m1x2_t, metal_backend_t>, backend,
          nk_dots_symmetric_e2m1_apple10);
#endif
}

void test_cross_metal(error_stats_section_t &check) {
    test_cross_metal_baseline(check);
    test_cross_apple9(check);
    test_cross_apple10(check);
}

/** The dispatching entry points, over the capabilities of the device the backend encodes on. */
void test_cross_dispatch(error_stats_section_t &check) {
    check.section("Cross Dispatch", nk_cap_metal_k);
#if NUMKONG_HEADER_ONLY
    check("dots_packed_i8_dispatch", [](settings_t const &settings) {
        return test_missing_library<nk_dots_packed_i8_best>(settings, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0,
                                                            nullptr);
    });
#else
    metal_backend_t const backend {};
    check("dots_packed_i8_dispatch", test_dots_packed<i8_t, metal_backend_t>, backend,
          gpu_best<nk_dots_pack_size_i8_best>, gpu_best<nk_dots_pack_i8_best>, gpu_best<nk_dots_packed_i8_best>);
#endif
}

} // namespace ashvardanian::numkong::test
