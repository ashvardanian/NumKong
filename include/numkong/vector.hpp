/**
 *  @brief NumKong Vector types for C++23 and newer.
 *  @file include/numkong/vector.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2026
 *
 *  Provides owning and non-owning vector types with signed indexing,
 *  strided views, and sub-byte element support.
 *
 *  - `nk::vector<T, A>`: Owning, non-resizable, SIMD-aligned
 *  - `nk::vector_view<T>`: Non-owning, const, strided
 *  - `nk::vector_span<T>`: Non-owning, mutable, strided
 *
 *  @section vector_terminology Terminology: Dimensions vs Values
 *
 *  The `nk::vector<value_type_>` container uses two distinct counts:
 *
 *  - `size()`: Number of logical dimensions (what kernels see).
 *    For `vector<i4x2_t>` with 100 dimensions, you have 100 nibbles.
 *
 *  - `size_values()`: Number of C++ container elements (`value_type_` instances).
 *    For `vector<i4x2_t>` with 100 dimensions, you have 50 values (2 dims/value).
 *
 *  @code
 *  auto v = nk::vector<float>::try_zeros(5);
 *  v.size();          // 5
 *  v[-1];             // last element (signed indexing)
 *  v[nk::range(0,3)]; // view of first 3 elements
 *  @endcode
 *
 *  See `types.hpp` for the full terminology reference.
 */

#ifndef NK_VECTOR_HPP
#define NK_VECTOR_HPP

#include <concepts>    // `std::integral`
#include <cstdlib>     // `std::aligned_alloc`, `std::free`
#include <cstring>     // `std::memset`
#include <iterator>    // `std::random_access_iterator_tag`
#include <memory>      // `std::allocator_traits`
#include <type_traits> // `std::conditional_t`
#include <utility>     // `std::exchange`, `std::swap`

#include "types.hpp"

namespace ashvardanian::numkong {

#pragma region Aligned Allocator

/**
 *  @brief Cache-aligned allocator with non-throwing allocation.
 *  @tparam value_type_ Value type to allocate.
 *  @tparam alignment_ Alignment in bytes (default: 64 for cache line).
 *
 *  This allocator uses `std::aligned_alloc` and returns `nullptr` on failure
 *  instead of throwing. It is stateless and always compares equal.
 */
template <typename value_type_, std::size_t alignment_ = 64>
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

    static constexpr std::size_t alignment = alignment_;

    constexpr aligned_allocator() noexcept = default;

    template <typename other_type_>
    constexpr aligned_allocator(aligned_allocator<other_type_, alignment_> const &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t n) noexcept {
        if (n == 0) return nullptr;
        std::size_t bytes = n * sizeof(value_type);
        // Round up to alignment boundary (required by aligned_alloc)
        std::size_t aligned_bytes = ((bytes + alignment_ - 1) / alignment_) * alignment_;
#if defined(_MSC_VER)
        return static_cast<value_type *>(::_aligned_malloc(aligned_bytes, alignment_));
#else
        return static_cast<value_type *>(std::aligned_alloc(alignment_, aligned_bytes));
#endif
    }

    void deallocate(value_type *p, std::size_t) noexcept {
#if defined(_MSC_VER)
        if (p) ::_aligned_free(p);
#else
        if (p) std::free(p);
#endif
    }

    template <typename other_type_>
    constexpr bool operator==(aligned_allocator<other_type_, alignment_> const &) const noexcept {
        return true;
    }
};

#pragma endregion Aligned Allocator

#pragma region Slicing Infrastructure

/** @brief Tag type for selecting all elements along a dimension. */
struct all_t {};

/** @brief Global instance of `all_t` for use in slicing expressions. */
inline constexpr all_t all {};

/** @brief Slicing descriptor: [start, stop) with optional step.
 *  `step` must be non-zero. Negative start/stop indices wrap from the end. */
struct range {
    std::ptrdiff_t start, stop, step;
    template <std::integral start_type_, std::integral stop_type_, std::integral step_type_ = int>
    constexpr range(start_type_ start, stop_type_ stop, step_type_ step = 1) noexcept
        : start(static_cast<std::ptrdiff_t>(start)), stop(static_cast<std::ptrdiff_t>(stop)),
          step(static_cast<std::ptrdiff_t>(step)) {}
};

/** @brief Resolve an integral index to an unsigned offset. Negative wraps from end. */
template <std::integral index_type_>
constexpr std::size_t resolve_index_(index_type_ idx, std::size_t extent) noexcept {
    if constexpr (std::signed_integral<index_type_>)
        return static_cast<std::size_t>(idx >= 0 ? idx : static_cast<std::ptrdiff_t>(extent) + idx);
    else return static_cast<std::size_t>(idx);
}

/** @brief Normalize any integral stride input to the signed internal representation. */
template <std::integral stride_type_>
constexpr std::ptrdiff_t resolve_stride_(stride_type_ stride) noexcept {
    return static_cast<std::ptrdiff_t>(stride);
}

/** @brief Normalize any unsigned extent input to the internal representation. */
template <std::unsigned_integral extent_type_>
constexpr std::size_t resolve_extent_(extent_type_ extent) noexcept {
    return static_cast<std::size_t>(extent);
}

/** @brief Resolve range start/stop against an extent (handles negatives). */
constexpr void resolve_range_(range const &r, std::size_t extent, //
                              std::size_t &out_start, std::size_t &out_stop) noexcept {
    out_start = resolve_index_(r.start, extent);
    out_stop = resolve_index_(r.stop, extent);
}

/** @brief Number of elements in a resolved range with the given step. */
constexpr std::size_t range_extent_(std::size_t start, std::size_t stop, std::ptrdiff_t step) noexcept {
    if (step > 0)
        return start < stop ? (stop - start + static_cast<std::size_t>(step) - 1) / static_cast<std::size_t>(step) : 0;
    else {
        auto abs_step = static_cast<std::size_t>(-step);
        return start > stop ? (start - stop + abs_step - 1) / abs_step : 0;
    }
}

#pragma endregion Slicing Infrastructure

#pragma region Forward Declarations

template <typename value_type_>
struct vector_view;
template <typename value_type_>
struct vector_span;

#pragma endregion Forward Declarations

#pragma region Dim Iterator

template <typename value_type_, typename allocator_type_>
struct vector;

/**
 *  @brief Random-access iterator over logical dimensions.
 *
 *  For sub-byte types (i4x2_t, u1x8_t), dereference returns a proxy reference.
 *  For normal types, dereference returns a direct reference.
 */
template <typename container_type_>
class dim_iterator {
    static constexpr bool is_const_ = std::is_const<container_type_>::value;
    using container_t = typename std::remove_reference<container_type_>::type;
    using container_ptr_t = std::conditional_t<is_const_, container_t const *, container_t *>;

    container_ptr_t container_;
    std::size_t index_;

  public:
    using container_type = container_type_;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    constexpr dim_iterator() noexcept : container_(nullptr), index_(0) {}
    constexpr dim_iterator(container_type &c, size_type i) noexcept : container_(&c), index_(i) {}

    constexpr decltype(auto) operator*() const noexcept { return (*container_)[index_]; }

    constexpr auto operator->() const noexcept { return &(*container_)[index_]; }

    constexpr decltype(auto) operator[](difference_type n) const noexcept {
        return (*container_)[static_cast<difference_type>(index_) + n];
    }

    constexpr dim_iterator &operator++() noexcept {
        ++index_;
        return *this;
    }
    constexpr dim_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++index_;
        return tmp;
    }
    constexpr dim_iterator &operator--() noexcept {
        --index_;
        return *this;
    }
    constexpr dim_iterator operator--(int) noexcept {
        auto tmp = *this;
        --index_;
        return tmp;
    }

    constexpr dim_iterator &operator+=(difference_type n) noexcept {
        index_ += n;
        return *this;
    }
    constexpr dim_iterator &operator-=(difference_type n) noexcept {
        index_ -= n;
        return *this;
    }
    constexpr dim_iterator operator+(difference_type n) const noexcept {
        return {*container_, index_ + static_cast<size_type>(n)};
    }
    constexpr dim_iterator operator-(difference_type n) const noexcept {
        return {*container_, index_ - static_cast<size_type>(n)};
    }
    constexpr difference_type operator-(dim_iterator const &other) const noexcept {
        return static_cast<difference_type>(index_) - static_cast<difference_type>(other.index_);
    }

    constexpr bool operator==(dim_iterator const &other) const noexcept { return index_ == other.index_; }
    constexpr bool operator!=(dim_iterator const &other) const noexcept { return index_ != other.index_; }
    constexpr bool operator<(dim_iterator const &other) const noexcept { return index_ < other.index_; }
    constexpr bool operator<=(dim_iterator const &other) const noexcept { return index_ <= other.index_; }
    constexpr bool operator>(dim_iterator const &other) const noexcept { return index_ > other.index_; }
    constexpr bool operator>=(dim_iterator const &other) const noexcept { return index_ >= other.index_; }

    constexpr size_type index() const noexcept { return index_; }

    friend constexpr dim_iterator operator+(difference_type n, dim_iterator const &it) noexcept { return it + n; }
};

/** Lightweight view yielding (index, value) pairs from a container's iterator. */
template <typename container_type_>
struct enumerate_view_ {
    container_type_ &container_;

    struct iterator_ {
        using inner_iterator = decltype(std::declval<container_type_ &>().begin());
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        inner_iterator it_;
        std::size_t index_;

        constexpr auto operator*() const noexcept { return std::pair<std::size_t, decltype(*it_)> {index_, *it_}; }
        constexpr iterator_ &operator++() noexcept {
            ++it_;
            ++index_;
            return *this;
        }
        constexpr iterator_ operator++(int) noexcept {
            auto t = *this;
            ++*this;
            return t;
        }
        constexpr bool operator==(iterator_ const &o) const noexcept { return it_ == o.it_; }
        constexpr bool operator!=(iterator_ const &o) const noexcept { return it_ != o.it_; }
    };

    constexpr iterator_ begin() noexcept { return {container_.begin(), 0}; }
    constexpr iterator_ end() noexcept { return {container_.end(), container_.size()}; }
};

/** Returns a view yielding (index, value) pairs over a vector, view, or span. */
template <typename container_type_>
constexpr enumerate_view_<container_type_> enumerate(container_type_ &c) noexcept {
    return {c};
}

template <typename container_type_>
constexpr enumerate_view_<container_type_ const> enumerate(container_type_ const &c) noexcept {
    return {c};
}

#pragma endregion Dim Iterator

#pragma region Vector View

/**
 *  @brief Non-owning, immutable, strided view into a vector.
 *  @tparam value_type_ Element type.
 *
 *  Supports signed indexing (`v[-1]` = last element) and strided access.
 *  For sub-byte types, only contiguous views are meaningful.
 */
template <typename value_type_>
struct vector_view {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using const_iterator = dim_iterator<vector_view const>;

  private:
    char const *data_ = nullptr;
    size_type dimensions_ = 0;
    difference_type stride_bytes_ = 0;

  public:
    constexpr vector_view() noexcept = default;

    template <std::unsigned_integral dims_type_, std::integral stride_type_>
    constexpr vector_view(char const *data, dims_type_ dims, stride_type_ stride_bytes) noexcept
        : data_(data), dimensions_(resolve_extent_(dims)), stride_bytes_(resolve_stride_(stride_bytes)) {}

    /** @brief Construct from contiguous typed pointer. */
    template <std::unsigned_integral dims_type_>
    constexpr vector_view(value_type const *data, dims_type_ dims) noexcept
        : data_(reinterpret_cast<char const *>(data)), dimensions_(resolve_extent_(dims)),
          stride_bytes_(static_cast<difference_type>(sizeof(value_type))) {}

    /** @brief Number of logical dimensions. */
    constexpr size_type size() const noexcept { return dimensions_; }

    /** @brief Check if empty. */
    constexpr bool empty() const noexcept { return dimensions_ == 0; }

    /** @brief Stride in bytes between consecutive elements. */
    constexpr difference_type stride_bytes() const noexcept { return stride_bytes_; }

    /** @brief True if elements are stored contiguously. */
    constexpr bool is_contiguous() const noexcept {
        return stride_bytes_ == static_cast<difference_type>(sizeof(value_type));
    }

    /** @brief Raw byte pointer to the first element. */
    constexpr char const *byte_data() const noexcept { return data_; }

    /** @brief Typed pointer (only valid if contiguous). */
    constexpr value_type const *data() const noexcept { return reinterpret_cast<value_type const *>(data_); }

    /** @brief Integral indexing: signed negatives wrap from end. */
    template <std::integral index_type_>
    decltype(auto) operator[](index_type_ idx) const noexcept {
        auto i = resolve_index_(idx, dimensions_);
        if constexpr (dimensions_per_value<value_type>() > 1) {
            constexpr auto dims_per_value = dimensions_per_value<value_type>();
            auto value_index = i / dims_per_value;
            auto sub_index = i % dims_per_value;
            using raw_type = typename raw_pod_type<value_type>::type;
            auto *base = const_cast<raw_type *>(
                reinterpret_cast<raw_type const *>(data_ + static_cast<difference_type>(value_index) * stride_bytes_));
            return sub_byte_ref<value_type>(base, sub_index).get();
        }
        else { return *reinterpret_cast<value_type const *>(data_ + static_cast<difference_type>(i) * stride_bytes_); }
    }

    /** @brief Sub-slice via range. */
    vector_view operator[](range r) const noexcept {
        size_type start, stop;
        resolve_range_(r, dimensions_, start, stop);
        auto count = range_extent_(start, stop, r.step);
        return {data_ + static_cast<difference_type>(start) * stride_bytes_, count, stride_bytes_ * r.step};
    }

    /** @brief Select all elements (identity). */
    vector_view operator[](all_t) const noexcept { return *this; }

    /** @brief Create a reversed view by negating the stride and pointing to the last element.
     *  Iterating the returned view visits elements in reverse order. */
    constexpr vector_view rev() const noexcept {
        if (dimensions_ == 0) return *this;
        return {data_ + static_cast<difference_type>(dimensions_ - 1) * stride_bytes_, dimensions_, -stride_bytes_};
    }

    /** @brief Dimension iterator to beginning. */
    const_iterator begin() const noexcept { return {*this, 0}; }
    const_iterator cbegin() const noexcept { return {*this, 0}; }

    /** @brief Dimension iterator to end. */
    const_iterator end() const noexcept { return {*this, dimensions_}; }
    const_iterator cend() const noexcept { return {*this, dimensions_}; }
};

#pragma endregion Vector View

#pragma region Vector Span

/**
 *  @brief Non-owning, mutable, strided view into a vector.
 *  @tparam value_type_ Element type.
 *
 *  Same as `vector_view` but allows mutation. Implicitly converts to `vector_view`.
 */
template <typename value_type_>
struct vector_span {
    static_assert(!std::is_const_v<value_type_>,
                  "vector_span requires a non-const value_type_; use vector_view<value_type_> for read-only access");

    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using iterator = dim_iterator<vector_span>;
    using const_iterator = dim_iterator<vector_span const>;

  private:
    char *data_ = nullptr;
    size_type dimensions_ = 0;
    difference_type stride_bytes_ = 0;

  public:
    constexpr vector_span() noexcept = default;

    template <std::unsigned_integral dims_type_, std::integral stride_type_>
    constexpr vector_span(char *data, dims_type_ dims, stride_type_ stride_bytes) noexcept
        : data_(data), dimensions_(resolve_extent_(dims)), stride_bytes_(resolve_stride_(stride_bytes)) {}

    /** @brief Construct from contiguous typed pointer. */
    template <std::unsigned_integral dims_type_>
    constexpr vector_span(value_type *data, dims_type_ dims) noexcept
        : data_(reinterpret_cast<char *>(data)), dimensions_(resolve_extent_(dims)),
          stride_bytes_(static_cast<difference_type>(sizeof(value_type))) {}

    /** @brief Number of logical dimensions. */
    constexpr size_type size() const noexcept { return dimensions_; }

    /** @brief Check if empty. */
    constexpr bool empty() const noexcept { return dimensions_ == 0; }

    /** @brief Stride in bytes. */
    constexpr difference_type stride_bytes() const noexcept { return stride_bytes_; }

    /** @brief True if contiguous. */
    constexpr bool is_contiguous() const noexcept {
        return stride_bytes_ == static_cast<difference_type>(sizeof(value_type));
    }

    /** @brief Raw byte pointer. */
    constexpr char *byte_data() const noexcept { return data_; }

    /** @brief Typed pointer (only valid if contiguous). */
    constexpr value_type *data() const noexcept { return reinterpret_cast<value_type *>(data_); }

    /** @brief Implicit conversion to const view. */
    constexpr operator vector_view<value_type>() const noexcept {
        return {static_cast<char const *>(data_), dimensions_, stride_bytes_};
    }

    /** @brief Mutable integral indexing. Signed negatives wrap from end. */
    template <std::integral index_type_>
    decltype(auto) operator[](index_type_ idx) const noexcept {
        auto i = resolve_index_(idx, dimensions_);
        if constexpr (dimensions_per_value<value_type>() > 1) {
            constexpr auto dims_per_value = dimensions_per_value<value_type>();
            auto value_index = i / dims_per_value;
            auto sub_index = i % dims_per_value;
            using raw_type = typename raw_pod_type<value_type>::type;
            auto *base = reinterpret_cast<raw_type *>(data_ +
                                                      static_cast<difference_type>(value_index) * stride_bytes_);
            return sub_byte_ref<value_type>(base, sub_index);
        }
        else { return *reinterpret_cast<value_type *>(data_ + static_cast<difference_type>(i) * stride_bytes_); }
    }

    /** @brief Sub-slice via range. */
    vector_span operator[](range r) const noexcept {
        size_type start, stop;
        resolve_range_(r, dimensions_, start, stop);
        auto count = range_extent_(start, stop, r.step);
        return {data_ + static_cast<difference_type>(start) * stride_bytes_, count, stride_bytes_ * r.step};
    }

    /** @brief Select all elements. */
    vector_span operator[](all_t) const noexcept { return *this; }

    /** @brief Dimension iterator to beginning. The handle is shallow-const, so iteration yields mutable refs. */
    iterator begin() const noexcept { return {const_cast<vector_span &>(*this), 0}; }
    const_iterator cbegin() const noexcept { return {*this, 0}; }

    /** @brief Dimension iterator to end. */
    iterator end() const noexcept { return {const_cast<vector_span &>(*this), dimensions_}; }
    const_iterator cend() const noexcept { return {*this, dimensions_}; }

    /** @brief Zero-fill every element. memset on the contiguous fast path,
     *  one memset per element on the strided slow path. */
    bool fill_zeros() noexcept {
        static_assert(is_memset_zero_safe_v<value_type>,
                      "fill_zeros requires a dtype whose binary-zero is the value-zero");
        if (data_ == nullptr || dimensions_ == 0) return true;
        if constexpr (dimensions_per_value<value_type>() > 1) {
            if (!is_contiguous()) return false;
            auto byte_count = divide_round_up(dimensions_, dimensions_per_value<value_type>()) * sizeof(value_type);
            std::memset(static_cast<void *>(data_), 0, byte_count);
            return true;
        }
        else {
            if (is_contiguous()) {
                std::memset(static_cast<void *>(data_), 0, dimensions_ * sizeof(value_type));
                return true;
            }
            for (size_type element_index = 0; element_index < dimensions_; ++element_index)
                std::memset(static_cast<void *>(data_ + static_cast<difference_type>(element_index) * stride_bytes_),
                            0, sizeof(value_type));
            return true;
        }
    }

    /** @brief Fill every element with `value`. memset-then-overlay strategy mirroring the
     *  free `fill` on tensor_span: 1-byte storage uses a single byte_pattern memset, multi-byte
     *  storage uses a typed scalar broadcast loop. */
    bool fill(value_type value) noexcept {
        if (!fill_zeros()) return false;
        value_type const default_value {};
        if (std::memcmp(&value, &default_value, sizeof(value_type)) == 0) return true;
        if constexpr (sizeof(value_type) == 1) {
            unsigned char byte_pattern;
            std::memcpy(&byte_pattern, &value, 1);
            if constexpr (dimensions_per_value<value_type>() > 1) {
                auto byte_count = divide_round_up(dimensions_, dimensions_per_value<value_type>()) * sizeof(value_type);
                std::memset(static_cast<void *>(data_), byte_pattern, byte_count);
                return true;
            }
            else {
                if (is_contiguous()) {
                    std::memset(static_cast<void *>(data_), byte_pattern, dimensions_ * sizeof(value_type));
                    return true;
                }
                for (size_type element_index = 0; element_index < dimensions_; ++element_index)
                    std::memset(
                        static_cast<void *>(data_ + static_cast<difference_type>(element_index) * stride_bytes_),
                        byte_pattern, sizeof(value_type));
                return true;
            }
        }
        else {
            if (is_contiguous()) {
                auto *typed_data = reinterpret_cast<value_type *>(data_);
                for (size_type element_index = 0; element_index < dimensions_; ++element_index)
                    typed_data[element_index] = value;
                return true;
            }
            for (size_type element_index = 0; element_index < dimensions_; ++element_index) {
                auto *target =
                    reinterpret_cast<value_type *>(data_ + static_cast<difference_type>(element_index) * stride_bytes_);
                *target = value;
            }
            return true;
        }
    }

    /** @brief Copy from a same-size view. memcpy on the contiguous fast path,
     *  per-element copy on the strided slow path. Returns false on size mismatch. */
    bool copy_from(vector_view<value_type> input) noexcept {
        if (input.size() != dimensions_) return false;
        if (dimensions_ == 0) return true;
        if constexpr (dimensions_per_value<value_type>() > 1) {
            if (!is_contiguous() || !input.is_contiguous()) return false;
            auto byte_count = divide_round_up(dimensions_, dimensions_per_value<value_type>()) * sizeof(value_type);
            std::memcpy(static_cast<void *>(data_), static_cast<void const *>(input.byte_data()), byte_count);
            return true;
        }
        else {
            if (is_contiguous() && input.is_contiguous()) {
                std::memcpy(static_cast<void *>(data_), static_cast<void const *>(input.byte_data()),
                            dimensions_ * sizeof(value_type));
                return true;
            }
            auto input_stride_bytes = input.stride_bytes();
            for (size_type element_index = 0; element_index < dimensions_; ++element_index) {
                auto *target =
                    reinterpret_cast<value_type *>(data_ + static_cast<difference_type>(element_index) * stride_bytes_);
                auto const *source = reinterpret_cast<value_type const *>(
                    input.byte_data() + static_cast<difference_type>(element_index) * input_stride_bytes);
                *target = *source;
            }
            return true;
        }
    }
};

#pragma endregion Vector Span

#pragma region Vector

/**
 *  @brief Owning, non-resizable, SIMD-aligned vector.
 *
 *  Size is fixed at construction. Use `try_zeros()` factory for
 *  non-throwing construction, or `from_raw()` to adopt existing memory.
 *
 *  Supports signed indexing (`v[-1]`), sub-byte types via proxy references,
 *  and slicing via `operator[](range)`.
 *
 *  @tparam value_type_ Element type.
 *  @tparam allocator_type_ Allocator (default: aligned_allocator).
 */
template <typename value_type_, typename allocator_type_ = aligned_allocator<value_type_>>
struct vector {
    using value_type = value_type_;
    using raw_value_type = typename raw_pod_type<value_type>::type;

    using allocator_type = allocator_type_;
    using alloc_traits = std::allocator_traits<allocator_type_>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type_ *;
    using const_pointer = value_type_ const *;

    using iterator = dim_iterator<vector>;
    using const_iterator = dim_iterator<vector const>;

  private:
    pointer data_ = nullptr;
    size_type dimensions_ = 0;
    [[no_unique_address]] allocator_type_ alloc_;

    /** @brief Convert dimension count to value count. */
    static constexpr size_type dims_to_values(size_type dims) noexcept {
        return divide_round_up(dims, dimensions_per_value<value_type>());
    }

  public:
    /** @brief Default constructor — empty vector with default allocator. */
    vector() noexcept = default;

    /** @brief Construct with custom allocator. */
    explicit vector(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    /** @brief Destructor — deallocates memory. */
    ~vector() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, dims_to_values(dimensions_));
    }

    /** @brief Move constructor. */
    vector(vector &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), dimensions_(std::exchange(other.dimensions_, 0)),
          alloc_(std::move(other.alloc_)) {}

    /** @brief Move assignment with allocator propagation. */
    vector &operator=(vector &&other) noexcept {
        if (this != &other) {
            if (data_) alloc_traits::deallocate(alloc_, data_, dims_to_values(dimensions_));
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(other.alloc_);
            data_ = std::exchange(other.data_, nullptr);
            dimensions_ = std::exchange(other.dimensions_, 0);
        }
        return *this;
    }

    vector(vector const &) = delete;
    vector &operator=(vector const &) = delete;

    /**
     *  @brief Factory: allocate a zero-initialized vector with `dims` dimensions.
     *  @return Non-empty vector on success, empty vector on allocation failure.
     */
    [[nodiscard]] static vector try_zeros(size_type dims, allocator_type_ alloc = {}) noexcept {
        vector v(alloc);
        size_type values = dims_to_values(dims);
        if (values == 0) return v;
        pointer ptr = alloc_traits::allocate(v.alloc_, values);
        if (!ptr) return v;
        if constexpr (is_memset_zero_safe_v<value_type_>) std::memset(ptr, 0, values * sizeof(value_type_));
        else
            for (size_type i = 0; i < values; ++i) ptr[i] = value_type_ {};
        v.data_ = ptr;
        v.dimensions_ = dims;
        return v;
    }

    /**
     *  @brief Factory: allocate a vector filled with ones.
     *  @return Non-empty vector on success, empty vector on allocation failure.
     */
    [[nodiscard]] static vector try_ones(size_type dims, allocator_type_ alloc = {}) noexcept {
        return try_full(dims, value_type_ {1}, alloc);
    }

    /**
     *  @brief Factory: allocate a vector filled with `val`.
     *  @return Non-empty vector on success, empty vector on allocation failure.
     */
    [[nodiscard]] static vector try_full(size_type dims, value_type_ val, allocator_type_ alloc = {}) noexcept {
        vector v(alloc);
        size_type values = dims_to_values(dims);
        if (values == 0) return v;
        pointer ptr = alloc_traits::allocate(v.alloc_, values);
        if (!ptr) return v;
        for (size_type i = 0; i < values; ++i) ptr[i] = val;
        v.data_ = ptr;
        v.dimensions_ = dims;
        return v;
    }

    /**
     *  @brief Factory: allocate an uninitialized vector.
     *  @return Non-empty vector on success, empty vector on allocation failure.
     *  @warning Contents are uninitialized. Caller must fill before reading.
     */
    [[nodiscard]] static vector try_empty(size_type dims, allocator_type_ alloc = {}) noexcept {
        vector v(alloc);
        size_type values = dims_to_values(dims);
        if (values == 0) return v;
        pointer ptr = alloc_traits::allocate(v.alloc_, values);
        if (!ptr) return v;
        v.data_ = ptr;
        v.dimensions_ = dims;
        return v;
    }

    /**
     *  @brief Factory: adopt raw memory. Caller transfers ownership.
     *  @param ptr Pointer to data (must have been allocated by `alloc`).
     *  @param dims Number of logical dimensions.
     *  @param alloc Allocator instance.
     */
    [[nodiscard]] static vector from_raw(pointer ptr, size_type dims, allocator_type_ alloc = {}) noexcept {
        vector v(alloc);
        v.data_ = ptr;
        v.dimensions_ = dims;
        return v;
    }

    /** @brief Swap with another vector. */
    void swap(vector &other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(dimensions_, other.dimensions_);
        if constexpr (alloc_traits::propagate_on_container_swap::value) swap(alloc_, other.alloc_);
    }

    /** @brief Number of logical dimensions. */
    size_type size() const noexcept { return dimensions_; }

    /** @brief Check if empty. */
    bool empty() const noexcept { return dimensions_ == 0; }

    /** @brief Number of storage values. */
    size_type size_values() const noexcept { return dims_to_values(dimensions_); }

    /** @brief Size in bytes. */
    size_type size_bytes() const noexcept { return dims_to_values(dimensions_) * sizeof(value_type_); }

    /** @brief Pointer to underlying data. */
    value_type *values_data() noexcept { return data_; }
    value_type const *values_data() const noexcept { return data_; }

    raw_value_type *raw_values_data() noexcept { return reinterpret_cast<raw_value_type *>(data_); }
    raw_value_type const *raw_values_data() const noexcept { return reinterpret_cast<raw_value_type const *>(data_); }

    /** @brief Get a copy of the allocator. */
    allocator_type get_allocator() const noexcept { return alloc_; }

    /**
     *  @brief Signed dimension access. Negative indices wrap from end.
     *  @retval For sub-byte types, returns proxy reference.
     *  @retval For normal types, returns direct reference.
     */
    template <std::integral index_type_>
    decltype(auto) operator[](index_type_ idx) noexcept {
        auto i = resolve_index_(idx, dimensions_);
        if constexpr (dimensions_per_value<value_type>() > 1)
            return sub_byte_ref<value_type>(reinterpret_cast<raw_value_type *>(data_), i);
        else return data_[i];
    }

    template <std::integral index_type_>
    decltype(auto) operator[](index_type_ idx) const noexcept {
        auto i = resolve_index_(idx, dimensions_);
        if constexpr (dimensions_per_value<value_type>() > 1)
            return sub_byte_ref<value_type>(
                       const_cast<raw_value_type *>(reinterpret_cast<raw_value_type const *>(data_)), i)
                .get();
        else return data_[i];
    }

    /** @brief Slice via range, returns a vector_span. */
    vector_span<value_type> operator[](range r) noexcept {
        size_type start, stop;
        resolve_range_(r, dimensions_, start, stop);
        auto count = range_extent_(start, stop, r.step);
        auto stride = static_cast<difference_type>(sizeof(value_type)) * r.step;
        return {reinterpret_cast<char *>(data_) +
                    static_cast<difference_type>(start) * static_cast<difference_type>(sizeof(value_type)),
                count, stride};
    }

    /** @brief Slice via range (const), returns a vector_view. */
    vector_view<value_type> operator[](range r) const noexcept {
        size_type start, stop;
        resolve_range_(r, dimensions_, start, stop);
        auto count = range_extent_(start, stop, r.step);
        auto stride = static_cast<difference_type>(sizeof(value_type)) * r.step;
        return {reinterpret_cast<char const *>(data_) +
                    static_cast<difference_type>(start) * static_cast<difference_type>(sizeof(value_type)),
                count, stride};
    }

    /** @brief Select all elements as a span. */
    vector_span<value_type> operator[](all_t) noexcept { return span(); }

    /** @brief Select all elements as a view. */
    vector_view<value_type> operator[](all_t) const noexcept { return view(); }

    /** @brief Create an immutable view. */
    vector_view<value_type> view() const noexcept { return {data_, dimensions_}; }

    /** @brief Create a mutable span. */
    vector_span<value_type> span() noexcept { return {data_, dimensions_}; }

    /** @brief Dimension iterator to beginning. */
    iterator begin() noexcept { return {*this, 0}; }
    const_iterator begin() const noexcept { return {*this, 0}; }
    const_iterator cbegin() const noexcept { return {*this, 0}; }

    /** @brief Dimension iterator to end. */
    iterator end() noexcept { return {*this, dimensions_}; }
    const_iterator end() const noexcept { return {*this, dimensions_}; }
    const_iterator cend() const noexcept { return {*this, dimensions_}; }

    /** @brief Zero-fill every element. */
    bool fill_zeros() noexcept { return span().fill_zeros(); }

    /** @brief Fill every element with `value`. */
    bool fill(value_type value) noexcept { return span().fill(value); }

    /** @brief Copy from a same-size view. */
    bool copy_from(vector_view<value_type> input) noexcept { return span().copy_from(input); }
};

/** @brief Non-member swap. */
template <typename value_type_, typename allocator_type_>
void swap(vector<value_type_, allocator_type_> &a, vector<value_type_, allocator_type_> &b) noexcept {
    a.swap(b);
}

#pragma endregion Vector

} // namespace ashvardanian::numkong

#endif // NK_VECTOR_HPP
