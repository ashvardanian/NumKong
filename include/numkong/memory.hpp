/**
 *  @file include/numkong/memory.hpp
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief C++ allocation policies over the C allocator interface.
 */
#ifndef NUMKONG_MEMORY_HPP
#define NUMKONG_MEMORY_HPP

#include <cstddef> // `std::size_t`, `std::ptrdiff_t`

#include <limits>      // `std::numeric_limits`
#include <type_traits> // `std::true_type`, `std::false_type`

#include "numkong/memory.h"  // `nk_allocator_t`, `nk_allocator_init_heap`
#include "numkong/types.hpp" // `expected`

namespace ashvardanian::numkong {

/**
 *  @brief Cache-aligned allocator with non-throwing allocation.
 *  @tparam value_type_ Value type to allocate.
 *  @tparam alignment_ Alignment in bytes (default: @c nk_default_alignment_k).
 *
 *  Allocation returns @c nullptr on failure instead of throwing.
 *  It is stateless and always compares equal.
 */
template <typename value_type_, std::size_t alignment_ = nk_default_alignment_k>
struct aligned_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = aligned_allocator<other_type_, alignment_>;
    };

    static_assert(alignment_ && !(alignment_ & (alignment_ - 1)) && alignment_ >= alignof(value_type));
    static constexpr std::size_t alignment = alignment_;

    constexpr aligned_allocator() noexcept = default;

    template <typename other_type_>
    constexpr aligned_allocator(aligned_allocator<other_type_, alignment_> const &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t n) noexcept {
        if (n > (std::numeric_limits<std::size_t>::max)() / sizeof(value_type)) return nullptr;
        nk_allocator_t policy;
        [[maybe_unused]] nk_status_t const status = nk_allocator_init_heap(&policy, alignment_);
        return static_cast<value_type *>(policy.allocate(n * sizeof(value_type), policy.handle, nullptr));
    }

    void deallocate(value_type *pointer, std::size_t n) noexcept {
        nk_allocator_t policy;
        [[maybe_unused]] nk_status_t const status = nk_allocator_init_heap(&policy, alignment_);
        policy.free(pointer, n * sizeof(value_type), policy.handle, nullptr);
    }

    template <typename other_type_>
    constexpr bool operator==(aligned_allocator<other_type_, alignment_> const &) const noexcept {
        return true;
    }
};

/** Adapts a C allocation policy and caller-owned stream to the container allocator interface. */
template <typename value_type_>
struct allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::false_type;

    nk_allocator_t policy {};
    nk_stream_t stream = nullptr;

    template <typename other_type_>
    struct rebind {
        using other = allocator<other_type_>;
    };

    constexpr allocator() noexcept = default;

    /** An allocator over the policy @p initialize sets up for @p capabilities, through
     *  one of the C allocator initializers, on @p stream. */
    static expected<allocator> make(nk_capability_t capabilities, nk_stream_t stream,
                                    nk_status_t (*initialize)(nk_allocator_t *, nk_capability_t)) noexcept {
        nk_allocator_t policy {};
        nk_status_t const status = initialize(&policy, capabilities);
        return {allocator(policy, stream), static_cast<status_t>(status)};
    }

    constexpr allocator(nk_allocator_t policy, nk_stream_t stream = nullptr) noexcept
        : policy(policy), stream(stream) {}
    template <typename other_type_>
    constexpr allocator(allocator<other_type_> const &other) noexcept : policy(other.policy), stream(other.stream) {}

    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        if (!policy.allocate || !policy.free || count > (std::numeric_limits<std::size_t>::max)() / sizeof(value_type))
            return nullptr;
        count = count ? count : 1;
        void *pointer = policy.allocate(count * sizeof(value_type), policy.handle, stream);
        if (!pointer || reinterpret_cast<std::size_t>(pointer) % alignof(value_type) == 0)
            return static_cast<value_type *>(pointer);
        policy.free(pointer, count * sizeof(value_type), policy.handle, stream);
        return nullptr;
    }
    void deallocate(value_type *pointer, std::size_t count) noexcept {
        if (pointer) policy.free(pointer, (count ? count : 1) * sizeof(value_type), policy.handle, stream);
    }
    template <typename other_type_>
    constexpr bool operator==(allocator<other_type_> const &other) const noexcept {
        return policy.allocate == other.policy.allocate && policy.free == other.policy.free &&
               policy.handle == other.policy.handle && stream == other.stream;
    }
};

} // namespace ashvardanian::numkong

#endif // NUMKONG_MEMORY_HPP
