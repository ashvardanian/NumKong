/**
 *  @file test/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Batch operation tests - Metal capabilities.
 *
 *  Runs the dots scenarios of `cross.hpp` through a @c metal_backend_t for the Metal baseline and every
 *  Apple GPU family the device runs, against the serial `nk::` references on the host.
 */
#include "numkong/metal.h" // `nk_metal_queue_t`

#include "harness.hpp" // `error_stats_section_t`, `call_best`
#include "cross.hpp"   // `test_dots_packed`

namespace ashvardanian::numkong::test {

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

/** Runs the Metal kernels on @c queue over its shared memory, keeping the first failed status.
 *  Unlike a CUDA stream, the queue is an explicit value: @c main opens it, and every allocator and
 *  backend holds a reference to it, since a kernel reaches only the blocks of its own queue. */
struct metal_backend_t {

    /** The allocator every kernel operand comes from, readable by the host once the queue is
     *  synchronized. */
    template <typename value_type_>
    using allocator = metal_shared_allocator<value_type_>;

    /** Where every call encodes and every operand is recorded, owned by @c main. */
    nk_metal_queue_t &queue;

    /** The first failure since the last synchronization. */
    nk_status_t status = nk_success_k;

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return (row_bytes + 15) / 16 * 16; }

    /** Copies @p bytes once every queued call has finished with them. */
    void copy(void *destination, void const *source, std::size_t bytes) noexcept {
        keep(nk_metal_synchronize(&queue));
        std::memcpy(destination, source, bytes);
    }

    /** Zeroes @p bytes once every queued call has finished with them. */
    void zero(void *destination, std::size_t bytes) noexcept {
        keep(nk_metal_synchronize(&queue));
        std::memset(destination, 0, bytes);
    }

    /** Encodes @p kernel with @p arguments on the queue. */
    template <typename kernel_type_, typename... arguments_types_>
    void call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        keep(kernel(arguments..., static_cast<void *>(&queue)));
    }

    /** Calls @p kernel on operands it must refuse, reporting whether it returned
     *  @c nk_misaligned_k unencoded. */
    template <typename kernel_type_, typename... arguments_types_>
    bool refuses_misaligned(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., static_cast<void *>(&queue)) == nk_misaligned_k;
    }

    /** Waits for the queue, returning the name of the first failure since the last call, or
     *  @c nullptr. */
    char const *synchronize() noexcept {
        keep(nk_metal_synchronize(&queue));
        nk_status_t const failure = status;
        status = nk_success_k;
        return failure == nk_success_k ? nullptr : nk_status_to_string(failure);
    }

    /** Remembers @p result unless an earlier failure is pending. */
    void keep(nk_status_t result) noexcept {
        if (status == nk_success_k) status = result;
    }
};

/** Blocks recorded on @p backend's own queue. */
template <typename value_type_>
metal_shared_allocator<value_type_> allocator_of(metal_backend_t const &backend) noexcept {
    return metal_shared_allocator<value_type_>(backend.queue);
}

/** The dispatch point @p best_ in the shape of its capability kernels, over the capabilities of GPU 0, where
 *  the backend encodes: callable like them, and convertible to their function pointers. */
template <auto best_>
inline constexpr auto gpu_best =
    [](auto... arguments) noexcept { return call_best<best_>(nk::metal_capabilities(), arguments...); };

} // namespace ashvardanian::numkong::test

using namespace ashvardanian::numkong::test;

/** Every Metal baseline entry point, on any Apple GPU of family 7 or newer. */
static void test_cross_metal_baseline(nk_metal_queue_t &queue) {
    metal_backend_t const backend {queue};
    error_stats_section_t check(nk::metal_capabilities());
    check.section("Cross Metal", nk_cap_metal_k);
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
static void test_cross_apple9([[maybe_unused]] nk_metal_queue_t &queue) {
#if NUMKONG_TARGET_APPLE9
    metal_backend_t const backend {queue};
    error_stats_section_t check(nk::metal_capabilities());
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
static void test_cross_apple10([[maybe_unused]] nk_metal_queue_t &queue) {
#if NUMKONG_TARGET_APPLE10
    metal_backend_t const backend {queue};
    error_stats_section_t check(nk::metal_capabilities());
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

void test_cross_metal(nk_metal_queue_t &queue) {
    test_cross_metal_baseline(queue);
    test_cross_apple9(queue);
    test_cross_apple10(queue);
}

/** The dispatching entry points, over the capabilities of the device the backend encodes on. */
void test_cross_dispatch([[maybe_unused]] nk_metal_queue_t &queue) {
    error_stats_section_t check(nk::metal_capabilities());
    check.section("Cross Dispatch", nk_cap_metal_k);
#if NUMKONG_HEADER_ONLY
    check("dots_packed_i8_dispatch", [] {
        return test_missing_library<nk_dots_packed_i8_best>(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr);
    });
#else
    metal_backend_t const backend {queue};
    check("dots_packed_i8_dispatch", test_dots_packed<i8_t, metal_backend_t>, backend,
          gpu_best<nk_dots_pack_size_i8_best>, gpu_best<nk_dots_pack_i8_best>, gpu_best<nk_dots_packed_i8_best>);
#endif
}
