/**
 *  @brief NumKong Tensor types and tensor-level operations for C++20 and newer.
 *  @file include/numkong/tensor.hpp
 *  @author Ash Vardanian
 *  @date March 2026
 *
 *  Provides owning and non-owning N-dimensional tensor types:
 *
 *  - `nk::tensor<T, A, max_rank>`: Owning, non-resizable
 *  - `nk::tensor_view<T, max_rank>`: Non-owning, const
 *  - `nk::tensor_span<T, max_rank>`: Non-owning, mutable
 *  - `nk::matrix` / `nk::matrix_view` / `nk::matrix_span`: 2D aliases
 *
 *  Tensor-level free functions:
 *  - Non-allocating scalar results: `sum(view)`, `min(view)`, `max(view)`, etc.
 *  - Allocating ops: `try_add(a, b)`, `try_sum(view, axis)`, etc.
 *  - In-place into pre-allocated output: `add_into(a, b, out)`, etc.
 *
 *  Features:
 *  - Signed strides (ptrdiff_t) for reversed/transposed views
 *  - Signed indexing (negative = from end)
 *  - Variadic `operator()` for flat/exact access and trailing `slice` (C++20-portable);
 *    `operator[]` multi-arg sugar provided when the compiler supports P2128 (C++23).
 *  - Axis iteration (rows_views(), rows_spans(), axis_iterator)
 *  - Conversion to vector_view/vector_span for rank-1 tensors
 */

#ifndef NK_TENSOR_HPP
#define NK_TENSOR_HPP

#include <array>   // `std::array`
#include <cstdio>  // `std::fprintf`, `stderr`
#include <cstdlib> // `std::abort`
#include <cstring> // `std::memset`
#include <span>    // `std::span`
#include <tuple>   // `std::tuple_element_t`
#include <type_traits>

#include "vector.hpp" // `aligned_allocator`

// True when the compiler supports C++23 P2128 multi-arg `operator[]`. Under
// this gate we expose `t[a, b, c]` as sugar that delegates to `operator()`.
#if defined(__cpp_multidimensional_subscript) && __cpp_multidimensional_subscript >= 202110L
#define NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_ 1
#else
#define NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_ 0
#endif

namespace ashvardanian::numkong {

template <typename value_type_, std::size_t max_rank_>
struct tensor_view;
template <typename value_type_, std::size_t max_rank_>
struct tensor_span;
template <typename value_type_, typename allocator_type_, std::size_t max_rank_>
struct tensor;

struct tensor_slice_t {};
inline constexpr tensor_slice_t slice {};

template <typename... arg_types_>
struct trailing_tensor_slice_args_ : std::false_type {};

template <>
struct trailing_tensor_slice_args_<tensor_slice_t> : std::true_type {};

template <std::integral index_type_, typename... rest_types_>
struct trailing_tensor_slice_args_<index_type_, rest_types_...> : trailing_tensor_slice_args_<rest_types_...> {};

template <typename... rest_types_>
struct trailing_tensor_slice_args_<all_t, rest_types_...> : trailing_tensor_slice_args_<rest_types_...> {};

template <typename... rest_types_>
struct trailing_tensor_slice_args_<range, rest_types_...> : trailing_tensor_slice_args_<rest_types_...> {};

template <typename... arg_types_>
inline constexpr bool trailing_tensor_slice_args_v =
    trailing_tensor_slice_args_<std::remove_cvref_t<arg_types_>...>::value;

#if defined(NDEBUG)
#define nk_assert_(expr) ((void)0)
#else
extern "C" [[noreturn]] inline void nk_assert_failure(char const *expr, char const *file, int line) noexcept {
    std::fprintf(stderr, "NumKong assertion failed: %s (%s:%d)\n", expr, file, line);
    std::abort();
}
#define nk_assert_(expr) ((expr) ? (void)0 : nk_assert_failure(#expr, __FILE__, __LINE__))
#endif

#pragma region Shape Storage

/**
 *  @brief Inline fixed-capacity shape descriptor.
 *  @tparam max_rank_ Maximum number of dimensions supported.
 *
 *  Stores extents and signed strides for up to `max_rank_` dimensions.
 *  For `max_rank_=2` (matrix), this is only 40 bytes.
 *  For `max_rank_=64`, this is 1032 bytes.
 */
template <std::size_t max_rank_>
struct shape_storage_ {
    std::size_t extents[max_rank_] = {};
    std::ptrdiff_t strides[max_rank_] = {};
    std::size_t rank = 0;

    /** @brief Total number of elements. */
    constexpr std::size_t numel() const noexcept {
        std::size_t n = 1;
        for (std::size_t i = 0; i < rank; ++i) n *= extents[i];
        return n;
    }

    /** @brief Linearize multi-dimensional coordinates to a byte offset. */
    constexpr std::ptrdiff_t linearize(std::size_t const *coords) const noexcept {
        std::ptrdiff_t offset = 0;
        for (std::size_t i = 0; i < rank; ++i) offset += static_cast<std::ptrdiff_t>(coords[i]) * strides[i];
        return offset;
    }

    /** @brief Create contiguous (row-major) shape storage. */
    static constexpr shape_storage_ contiguous(std::size_t const *exts, std::size_t rank_val,
                                               std::size_t elem_bytes) noexcept {
        shape_storage_ s;
        s.rank = rank_val;
        auto stride = static_cast<std::ptrdiff_t>(elem_bytes);
        for (std::size_t i = rank_val; i > 0; --i) {
            s.extents[i - 1] = exts[i - 1];
            s.strides[i - 1] = stride;
            stride *= static_cast<std::ptrdiff_t>(exts[i - 1]);
        }
        return s;
    }
};

template <typename value_type_>
constexpr std::size_t dims_to_values_(std::size_t dims) noexcept {
    return divide_round_up(dims, static_cast<std::size_t>(dimensions_per_value<value_type_>()));
}

template <typename value_type_, std::size_t max_rank_>
constexpr std::size_t storage_values_for_shape_(shape_storage_<max_rank_> const &shape) noexcept {
    if (shape.rank == 0) return 1;
    std::size_t values = 1;
    for (std::size_t i = 0; i < shape.rank; ++i) {
        bool const is_last = i + 1 == shape.rank;
        values *= is_last ? dims_to_values_<value_type_>(shape.extents[i]) : shape.extents[i];
    }
    return values;
}

template <typename value_type_, std::size_t max_rank_>
constexpr shape_storage_<max_rank_> make_contiguous_shape_(std::size_t const *exts, std::size_t rank_val) noexcept {
    shape_storage_<max_rank_> s;
    s.rank = rank_val;
    auto stride = static_cast<std::ptrdiff_t>(sizeof(value_type_));
    for (std::size_t i = rank_val; i > 0; --i) {
        s.extents[i - 1] = exts[i - 1];
        s.strides[i - 1] = stride;
        auto const extent_factor = i == rank_val ? dims_to_values_<value_type_>(exts[i - 1])
                                                 : static_cast<std::size_t>(exts[i - 1]);
        stride *= static_cast<std::ptrdiff_t>(extent_factor);
    }
    return s;
}

template <typename value_type_, std::size_t max_rank_>
constexpr bool is_tensor_contiguous_(shape_storage_<max_rank_> const &shape) noexcept {
    if (shape.rank == 0) return true;
    auto expected = static_cast<std::ptrdiff_t>(sizeof(value_type_));
    for (std::size_t i = shape.rank; i > 0; --i) {
        if (shape.strides[i - 1] != expected) return false;
        auto const extent_factor = i == shape.rank ? dims_to_values_<value_type_>(shape.extents[i - 1])
                                                   : static_cast<std::size_t>(shape.extents[i - 1]);
        expected *= static_cast<std::ptrdiff_t>(extent_factor);
    }
    return true;
}

template <typename value_type_, std::size_t max_rank_>
constexpr bool packed_tensor_layout_supported_(shape_storage_<max_rank_> const &shape) noexcept {
    if constexpr (dimensions_per_value<value_type_>() == 1) return true;
    else return is_tensor_contiguous_<value_type_>(shape);
}

template <typename value_type_, std::size_t max_rank_, std::size_t... indices_, typename... index_types_>
constexpr std::array<std::size_t, sizeof...(indices_)> resolve_tensor_indices_(shape_storage_<max_rank_> const &shape,
                                                                               std::index_sequence<indices_...>,
                                                                               index_types_... idxs) noexcept {
    return {resolve_index_(idxs, shape.extents[indices_])...};
}

template <typename value_type_, std::size_t max_rank_, std::size_t extent_>
decltype(auto) tensor_lookup_resolved_(tensor_view<value_type_, max_rank_> input,
                                       std::span<std::size_t const, extent_> coords) noexcept;

template <typename value_type_, std::size_t max_rank_, std::size_t extent_>
decltype(auto) tensor_lookup_resolved_(tensor_span<value_type_, max_rank_> input,
                                       std::span<std::size_t const, extent_> coords) noexcept;

template <typename value_type_, std::size_t max_rank_, typename index_type_>
decltype(auto) tensor_flat_lookup_(tensor_view<value_type_, max_rank_> input, index_type_ idx) noexcept;

template <typename value_type_, std::size_t max_rank_, typename index_type_>
decltype(auto) tensor_flat_lookup_(tensor_span<value_type_, max_rank_> input, index_type_ idx) noexcept;

template <typename tensor_type_>
tensor_type_ tensor_slice_suffix_(tensor_type_ input, tensor_slice_t) noexcept;

template <typename tensor_type_, std::integral index_type_, typename... rest_types_>
tensor_type_ tensor_slice_suffix_(tensor_type_ input, index_type_ idx, rest_types_... rest) noexcept;

template <typename tensor_type_, typename... rest_types_>
tensor_type_ tensor_slice_suffix_(tensor_type_ input, all_t, rest_types_... rest) noexcept;

template <typename tensor_type_, typename... rest_types_>
tensor_type_ tensor_slice_suffix_(tensor_type_ input, range r, rest_types_... rest) noexcept;

#pragma endregion Shape Storage

#pragma region Tensor View

template <typename view_type_>
class axis_iterator;
template <typename view_type_>
class tensor_view_iterator_;
template <typename span_type_>
class tensor_span_iterator_;
template <typename iterator_type_>
struct tensor_dims_view_;

/**
 *  @brief Non-owning, immutable, N-dimensional view.
 *  @tparam value_type_ Element type.
 *  @tparam max_rank_ Maximum number of dimensions.
 */
template <typename value_type_, std::size_t max_rank_ = 8>
struct tensor_view {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

  private:
    char const *data_ = nullptr;
    shape_storage_<max_rank_> shape_;

  public:
    constexpr tensor_view() noexcept = default;

    constexpr tensor_view(char const *data, shape_storage_<max_rank_> const &shape) noexcept
        : data_(data), shape_(shape) {}

    /** @brief Convenience constructor for rank-2 views from typed pointer, rows, and cols. */
    tensor_view(value_type const *data, size_type rows, size_type cols) noexcept
        requires(max_rank_ >= 2)
        : data_(reinterpret_cast<char const *>(data)) {
        std::size_t extents[2] = {rows, cols};
        shape_ = make_contiguous_shape_<value_type_, max_rank_>(extents, 2);
    }

    /** @brief Number of dimensions. */
    constexpr size_type rank() const noexcept { return shape_.rank; }

    /** @brief Extent along the i-th dimension. */
    constexpr size_type extent(size_type i) const noexcept { return shape_.extents[i]; }

    /** @brief Stride in bytes along the i-th dimension (signed). */
    constexpr difference_type stride_bytes(size_type i) const noexcept { return shape_.strides[i]; }

    /** @brief Total number of elements. */
    constexpr size_type numel() const noexcept { return shape_.numel(); }

    /** @brief True if empty. */
    constexpr bool empty() const noexcept { return data_ == nullptr || shape_.numel() == 0; }

    /** @brief Raw byte pointer. */
    constexpr char const *byte_data() const noexcept { return data_; }

    /** @brief Typed pointer (assumes data is contiguous from this pointer). */
    constexpr value_type const *data() const noexcept { return reinterpret_cast<value_type const *>(data_); }

    /** @brief Access the shape storage. */
    constexpr shape_storage_<max_rank_> const &shape() const noexcept { return shape_; }

    /** @brief Slice along the leading dimension. */
    template <std::integral index_type_>
    tensor_view<value_type_, max_rank_> slice_leading(index_type_ idx) const noexcept {
        nk_assert_(shape_.rank >= 1);
        if (shape_.rank == 0) return {};
        auto i = resolve_index_(idx, shape_.extents[0]);
        auto offset = static_cast<difference_type>(i) * shape_.strides[0];
        shape_storage_<max_rank_> sub;
        sub.rank = shape_.rank - 1;
        for (size_type d = 0; d < sub.rank; ++d) {
            sub.extents[d] = shape_.extents[d + 1];
            sub.strides[d] = shape_.strides[d + 1];
        }
        return {data_ + offset, sub};
    }

    /** @brief Row access (alias for slice_leading). */
    template <std::integral index_type_>
    tensor_view<value_type_, max_rank_> row(index_type_ i) const noexcept {
        return slice_leading(i);
    }

    /** @brief Rank-0 scalar access. */
    template <std::integral index_type_>
    decltype(auto) operator[](index_type_ idx) const noexcept {
        return tensor_flat_lookup_(*this, idx);
    }

    /** @brief Exact multi-dimensional scalar lookup via call syntax (C++20-portable). */
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator()(index_types_... idxs) const noexcept {
        nk_assert_(shape_.rank == sizeof...(index_types_));
        auto coords = resolve_tensor_indices_<value_type_>(shape_, std::index_sequence_for<index_types_...> {},
                                                           idxs...);
        return tensor_lookup_resolved_(*this, std::span<std::size_t const, sizeof...(index_types_)>(coords));
    }

#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    /** @brief C++23 sugar: `t[i, j, k]` scalar lookup, delegates to `operator()`. */
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator[](index_types_... idxs) const noexcept {
        return (*this)(idxs...);
    }
#endif

    /** @brief Trailing `slice` returns the same view. */
    constexpr tensor_view operator[](tensor_slice_t) const noexcept { return *this; }

    /** @brief Prefix leading-axis slicing with a trailing `slice` marker (call syntax, C++20-portable). */
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    tensor_view operator()(first_type_ first, second_type_ second, rest_types_... rest) const noexcept {
        return tensor_slice_suffix_(*this, first, second, rest...);
    }

#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    /** @brief C++23 sugar: `t[i, nk::slice]` slicing, delegates to `operator()`. */
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    tensor_view operator[](first_type_ first, second_type_ second, rest_types_... rest) const noexcept {
        return (*this)(first, second, rest...);
    }
#endif

    /** @brief Rank-0 scalar access. */
    decltype(auto) scalar() const noexcept {
        nk_assert_(shape_.rank == 0);
        nk_assert_(data_ != nullptr);
        return *reinterpret_cast<value_type_ const *>(data_);
    }

    /** @brief Convert to vector_view (requires rank == 1). */
    vector_view<value_type> as_vector() const noexcept {
        nk_assert_(shape_.rank == 1);
        if (shape_.rank != 1) return {};
        return {data_, shape_.extents[0], shape_.strides[0]};
    }

    /** @brief Reinterpret as a 2D matrix view. Requires rank >= 2. */
    tensor_view<value_type_, 2> as_matrix() const noexcept {
        nk_assert_(shape_.rank >= 2);
        if (shape_.rank < 2) return {};
        shape_storage_<2> matrix_shape;
        matrix_shape.rank = 2;
        matrix_shape.extents[0] = shape_.extents[0];
        matrix_shape.extents[1] = shape_.extents[1];
        matrix_shape.strides[0] = shape_.strides[0];
        matrix_shape.strides[1] = shape_.strides[1];
        return {data_, matrix_shape};
    }

    /** @brief Check if the tensor is contiguous in memory. */
    constexpr bool is_contiguous() const noexcept { return is_tensor_contiguous_<value_type>(shape_); }

    /** @brief Transpose: reverse the order of all dimensions (swap extents and strides). */
    constexpr tensor_view transpose() const noexcept {
        if constexpr (dimensions_per_value<value_type>() > 1) {
            if (shape_.rank >= 2) return {};
        }
        if (shape_.rank < 2) return *this;
        auto transposed = shape_;
        for (size_type i = 0; i < transposed.rank / 2; ++i) {
            std::swap(transposed.extents[i], transposed.extents[transposed.rank - 1 - i]);
            std::swap(transposed.strides[i], transposed.strides[transposed.rank - 1 - i]);
        }
        return {data_, transposed};
    }

    /** @brief Reshape to new extents (requires contiguous layout and matching element count).
     *  Returns an empty view if the tensor is not contiguous or element counts don't match. */
    tensor_view reshape(std::initializer_list<size_type> new_extents) const noexcept {
        auto new_rank = new_extents.size();
        if (!is_contiguous() || new_rank > max_rank_ || new_rank == 0) return {};
        auto new_shape = make_contiguous_shape_<value_type, max_rank_>(new_extents.begin(), new_rank);
        if (storage_values_for_shape_<value_type>(new_shape) != storage_values_for_shape_<value_type>(shape_))
            return {};
        return {data_, new_shape};
    }

    /** @brief Range of sub-views along the leading dimension. */
    struct rows_views_t {
        tensor_view parent;
        axis_iterator<tensor_view> begin() const noexcept { return {parent, 0}; }
        axis_iterator<tensor_view> end() const noexcept { return {parent, parent.extent(0)}; }
    };

    rows_views_t rows() const noexcept { return {*this}; }

    static constexpr std::size_t max_rank = max_rank_;

    /** @brief Element iterator (begin): yields `(position, scalar)` pairs. */
    tensor_view_iterator_<tensor_view> begin() const noexcept { return {*this}; }
    /** @brief Element iterator (end). */
    tensor_view_iterator_<tensor_view> end() const noexcept { return {*this, true}; }
    /** @brief Number of logical scalar elements. */
    constexpr size_type size() const noexcept { return numel(); }
    /** @brief Dimension-only view: iterate scalars without positions. */
    tensor_dims_view_<tensor_view_iterator_<tensor_view>> dims() const noexcept {
        return {tensor_view_iterator_<tensor_view> {*this}, numel()};
    }

    /** @brief Flatten to 1D view (requires contiguous layout). Returns empty view if not contiguous. */
    tensor_view flatten() const noexcept { return reshape({numel()}); }

    /** @brief Remove dimensions of size 1. */
    tensor_view squeeze() const noexcept {
        auto result = shape_;
        size_type new_rank = 0;
        for (size_type i = 0; i < shape_.rank; ++i) {
            if (shape_.extents[i] != 1) {
                result.extents[new_rank] = shape_.extents[i];
                result.strides[new_rank] = shape_.strides[i];
                ++new_rank;
            }
        }
        if (new_rank == 0) {
            new_rank = 1;
            result.extents[0] = 1;
            result.strides[0] = static_cast<difference_type>(sizeof(value_type));
        }
        result.rank = new_rank;
        return {data_, result};
    }
};

#pragma endregion Tensor View

#pragma region Tensor Span

/**
 *  @brief Non-owning, mutable, N-dimensional view.
 *  @tparam value_type_ Element type.
 *  @tparam max_rank_ Maximum number of dimensions.
 */
template <typename value_type_, std::size_t max_rank_ = 8>
struct tensor_span {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

  private:
    char *data_ = nullptr;
    shape_storage_<max_rank_> shape_;

  public:
    constexpr tensor_span() noexcept = default;

    constexpr tensor_span(char *data, shape_storage_<max_rank_> const &shape) noexcept : data_(data), shape_(shape) {}

    /** @brief Convenience constructor for rank-2 spans from typed pointer, rows, and cols. */
    tensor_span(value_type *data, size_type rows, size_type cols) noexcept
        requires(max_rank_ >= 2)
        : data_(reinterpret_cast<char *>(data)) {
        std::size_t extents[2] = {rows, cols};
        shape_ = make_contiguous_shape_<value_type_, max_rank_>(extents, 2);
    }

    /** @brief Number of dimensions. */
    constexpr size_type rank() const noexcept { return shape_.rank; }
    /** @brief Extent along the i-th dimension. */
    constexpr size_type extent(size_type i) const noexcept { return shape_.extents[i]; }
    /** @brief Stride in bytes along the i-th dimension (signed). */
    constexpr difference_type stride_bytes(size_type i) const noexcept { return shape_.strides[i]; }
    /** @brief Total number of elements. */
    constexpr size_type numel() const noexcept { return shape_.numel(); }
    /** @brief True if empty. */
    constexpr bool empty() const noexcept { return data_ == nullptr || shape_.numel() == 0; }

    /** @brief Raw byte pointer. */
    constexpr char *byte_data() noexcept { return data_; }
    constexpr char const *byte_data() const noexcept { return data_; }
    /** @brief Typed pointer (assumes data is contiguous from this pointer). */
    constexpr value_type *data() noexcept { return reinterpret_cast<value_type *>(data_); }
    constexpr value_type const *data() const noexcept { return reinterpret_cast<value_type const *>(data_); }
    /** @brief Access the shape storage. */
    constexpr shape_storage_<max_rank_> const &shape() const noexcept { return shape_; }

    /** @brief Implicit conversion to const view. */
    constexpr operator tensor_view<value_type_, max_rank_>() const noexcept {
        return {static_cast<char const *>(data_), shape_};
    }

    /** @brief Slice along leading dimension. */
    template <std::integral index_type_>
    tensor_span slice_leading(index_type_ idx) const noexcept {
        nk_assert_(shape_.rank >= 1);
        if (shape_.rank == 0) return {};
        auto i = resolve_index_(idx, shape_.extents[0]);
        auto offset = static_cast<difference_type>(i) * shape_.strides[0];
        shape_storage_<max_rank_> sub;
        sub.rank = shape_.rank - 1;
        for (size_type d = 0; d < sub.rank; ++d) {
            sub.extents[d] = shape_.extents[d + 1];
            sub.strides[d] = shape_.strides[d + 1];
        }
        return {data_ + offset, sub};
    }

    /** @brief Mutable row access (alias for slice_leading). */
    template <std::integral index_type_>
    tensor_span row(index_type_ i) const noexcept {
        return slice_leading(i);
    }

    /** @brief Flat logical scalar access. */
    template <std::integral index_type_>
    decltype(auto) operator[](index_type_ idx) noexcept {
        return tensor_flat_lookup_(*this, idx);
    }

    /** @brief Const flat logical scalar access. */
    template <std::integral index_type_>
    decltype(auto) operator[](index_type_ idx) const noexcept {
        return tensor_flat_lookup_(static_cast<tensor_view<value_type_, max_rank_>>(*this), idx);
    }

    /** @brief Exact multi-dimensional scalar lookup via call syntax (C++20-portable). */
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator()(index_types_... idxs) noexcept {
        nk_assert_(shape_.rank == sizeof...(index_types_));
        auto coords = resolve_tensor_indices_<value_type_>(shape_, std::index_sequence_for<index_types_...> {},
                                                           idxs...);
        return tensor_lookup_resolved_(*this, std::span<std::size_t const, sizeof...(index_types_)>(coords));
    }

    /** @brief Const full-coordinate lookup via call syntax. */
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator()(index_types_... idxs) const noexcept {
        return static_cast<tensor_view<value_type_, max_rank_>>(*this)(idxs...);
    }

#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    /** @brief C++23 sugar: multi-arg `[]` scalar lookup, delegates to `operator()`. */
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator[](index_types_... idxs) noexcept {
        return (*this)(idxs...);
    }
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator[](index_types_... idxs) const noexcept {
        return (*this)(idxs...);
    }
#endif

    /** @brief Trailing `slice` returns the same span. */
    constexpr tensor_span operator[](tensor_slice_t) noexcept { return *this; }
    constexpr tensor_view<value_type_, max_rank_> operator[](tensor_slice_t) const noexcept {
        return static_cast<tensor_view<value_type_, max_rank_>>(*this);
    }

    /** @brief Prefix leading-axis slicing via call syntax (C++20-portable). */
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    tensor_span operator()(first_type_ first, second_type_ second, rest_types_... rest) noexcept {
        return tensor_slice_suffix_(*this, first, second, rest...);
    }

    /** @brief Const prefix leading-axis slicing via call syntax. */
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    tensor_view<value_type_, max_rank_> operator()(first_type_ first, second_type_ second,
                                                   rest_types_... rest) const noexcept {
        return tensor_slice_suffix_(static_cast<tensor_view<value_type_, max_rank_>>(*this), first, second, rest...);
    }

#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    /** @brief C++23 sugar: multi-arg `[]` slicing, delegates to `operator()`. */
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    tensor_span operator[](first_type_ first, second_type_ second, rest_types_... rest) noexcept {
        return (*this)(first, second, rest...);
    }
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    tensor_view<value_type_, max_rank_> operator[](first_type_ first, second_type_ second,
                                                   rest_types_... rest) const noexcept {
        return (*this)(first, second, rest...);
    }
#endif

    /** @brief Rank-0 mutable scalar access. */
    decltype(auto) scalar_ref() noexcept {
        nk_assert_(shape_.rank == 0);
        nk_assert_(data_ != nullptr);
        return *reinterpret_cast<value_type_ *>(data_);
    }

    /** @brief Rank-0 const scalar access. */
    decltype(auto) scalar() const noexcept { return static_cast<tensor_view<value_type_, max_rank_>>(*this).scalar(); }

    /** @brief Convert to vector_span (requires rank == 1). */
    vector_span<value_type> as_vector() noexcept {
        nk_assert_(shape_.rank == 1);
        if (shape_.rank != 1) return {};
        return {data_, shape_.extents[0], shape_.strides[0]};
    }

    /** @brief Convert to vector_view (requires rank == 1). */
    vector_view<value_type> as_vector() const noexcept {
        nk_assert_(shape_.rank == 1);
        if (shape_.rank != 1) return {};
        return {static_cast<char const *>(data_), shape_.extents[0], shape_.strides[0]};
    }

    /** @brief Reinterpret as a 2D matrix span. Requires rank >= 2. */
    tensor_span<value_type_, 2> as_matrix() noexcept {
        nk_assert_(shape_.rank >= 2);
        if (shape_.rank < 2) return {};
        shape_storage_<2> matrix_shape;
        matrix_shape.rank = 2;
        matrix_shape.extents[0] = shape_.extents[0];
        matrix_shape.extents[1] = shape_.extents[1];
        matrix_shape.strides[0] = shape_.strides[0];
        matrix_shape.strides[1] = shape_.strides[1];
        return {data_, matrix_shape};
    }

    /** @brief Reinterpret as a 2D const matrix view. Requires rank >= 2. */
    tensor_view<value_type_, 2> as_matrix() const noexcept {
        nk_assert_(shape_.rank >= 2);
        if (shape_.rank < 2) return {};
        shape_storage_<2> matrix_shape;
        matrix_shape.rank = 2;
        matrix_shape.extents[0] = shape_.extents[0];
        matrix_shape.extents[1] = shape_.extents[1];
        matrix_shape.strides[0] = shape_.strides[0];
        matrix_shape.strides[1] = shape_.strides[1];
        return {static_cast<char const *>(data_), matrix_shape};
    }

    /** @brief Check if contiguous in memory. */
    constexpr bool is_contiguous() const noexcept { return is_tensor_contiguous_<value_type>(shape_); }

    /** @brief Transpose: reverse the order of all dimensions (swap extents and strides). */
    constexpr tensor_span transpose() noexcept {
        if constexpr (dimensions_per_value<value_type>() > 1) {
            if (shape_.rank >= 2) return {};
        }
        if (shape_.rank < 2) return *this;
        auto transposed = shape_;
        for (size_type i = 0; i < transposed.rank / 2; ++i) {
            std::swap(transposed.extents[i], transposed.extents[transposed.rank - 1 - i]);
            std::swap(transposed.strides[i], transposed.strides[transposed.rank - 1 - i]);
        }
        return {data_, transposed};
    }

    /** @brief Reshape to new extents (requires contiguous layout and matching element count).
     *  Returns an empty span if not contiguous or element counts don't match. */
    tensor_span reshape(std::initializer_list<size_type> new_extents) noexcept {
        auto new_rank = new_extents.size();
        if (!is_contiguous() || new_rank > max_rank_ || new_rank == 0) return {};
        auto new_shape = make_contiguous_shape_<value_type, max_rank_>(new_extents.begin(), new_rank);
        if (storage_values_for_shape_<value_type>(new_shape) != storage_values_for_shape_<value_type>(shape_))
            return {};
        return {data_, new_shape};
    }

    /** @brief Range of mutable sub-spans along the leading dimension. */
    struct rows_spans_t {
        tensor_span parent;
        axis_iterator<tensor_span> begin() noexcept { return {parent, 0}; }
        axis_iterator<tensor_span> end() noexcept { return {parent, parent.extent(0)}; }
    };

    rows_spans_t rows() noexcept { return {*this}; }

    /** @brief Range of immutable sub-views along the leading dimension. */
    struct rows_views_t {
        tensor_view<value_type_, max_rank_> parent;
        axis_iterator<tensor_view<value_type_, max_rank_>> begin() const noexcept { return {parent, 0}; }
        axis_iterator<tensor_view<value_type_, max_rank_>> end() const noexcept { return {parent, parent.extent(0)}; }
    };

    rows_views_t rows() const noexcept {
        tensor_view<value_type_, max_rank_> v = *this;
        return {v};
    }

    static constexpr std::size_t max_rank = max_rank_;

    /** @brief Mutable element iterator (begin): yields `(position, ref_or_proxy)` pairs. */
    tensor_span_iterator_<tensor_span> begin() noexcept { return {*this}; }
    /** @brief Mutable element iterator (end). */
    tensor_span_iterator_<tensor_span> end() noexcept { return {*this, true}; }
    /** @brief Const element iterator (begin): yields `(position, scalar)` pairs. */
    tensor_view_iterator_<tensor_view<value_type_, max_rank_>> begin() const noexcept {
        return {static_cast<tensor_view<value_type_, max_rank_>>(*this)};
    }
    /** @brief Const element iterator (end). */
    tensor_view_iterator_<tensor_view<value_type_, max_rank_>> end() const noexcept {
        return {static_cast<tensor_view<value_type_, max_rank_>>(*this), true};
    }
    /** @brief Number of logical scalar elements. */
    constexpr size_type size() const noexcept { return numel(); }
    /** @brief Mutable dimension-only view. */
    tensor_dims_view_<tensor_span_iterator_<tensor_span>> dims() noexcept {
        return {tensor_span_iterator_<tensor_span> {*this}, numel()};
    }
    /** @brief Const dimension-only view. */
    tensor_dims_view_<tensor_view_iterator_<tensor_view<value_type_, max_rank_>>> dims() const noexcept {
        return {tensor_view_iterator_<tensor_view<value_type_, max_rank_>> {
                    static_cast<tensor_view<value_type_, max_rank_>>(*this)},
                numel()};
    }

    /** @brief Flatten to 1D span (requires contiguous layout). Returns empty span if not contiguous. */
    tensor_span flatten() noexcept { return reshape({numel()}); }

    /** @brief Remove dimensions of size 1. */
    tensor_span squeeze() noexcept {
        auto result = shape_;
        size_type new_rank = 0;
        for (size_type i = 0; i < shape_.rank; ++i) {
            if (shape_.extents[i] != 1) {
                result.extents[new_rank] = shape_.extents[i];
                result.strides[new_rank] = shape_.strides[i];
                ++new_rank;
            }
        }
        if (new_rank == 0) {
            new_rank = 1;
            result.extents[0] = 1;
            result.strides[0] = static_cast<difference_type>(sizeof(value_type));
        }
        result.rank = new_rank;
        return {data_, result};
    }
};

#pragma endregion Tensor Span

template <typename value_type_, std::size_t max_rank_, std::size_t extent_>
decltype(auto) tensor_lookup_resolved_(tensor_view<value_type_, max_rank_> input,
                                       std::span<std::size_t const, extent_> coords) noexcept {
    nk_assert_(input.byte_data() != nullptr);
    nk_assert_(coords.size() == input.rank());
    if constexpr (dimensions_per_value<value_type_>() > 1) {
        nk_assert_(packed_tensor_layout_supported_<value_type_>(input.shape()));
        auto offset = std::ptrdiff_t {};
        for (std::size_t i = 0; i + 1 < input.rank(); ++i)
            offset += static_cast<std::ptrdiff_t>(coords[i]) * input.stride_bytes(i);
        constexpr auto dims_per_value = dimensions_per_value<value_type_>();
        auto last_index = coords[input.rank() - 1];
        auto value_index = last_index / dims_per_value;
        auto sub_index = last_index % dims_per_value;
        using raw_type = typename raw_pod_type<value_type_>::type;
        auto *base = const_cast<raw_type *>(reinterpret_cast<raw_type const *>(
            input.byte_data() + offset +
            static_cast<std::ptrdiff_t>(value_index) * input.stride_bytes(input.rank() - 1)));
        return sub_byte_ref<value_type_>(base, sub_index).get();
    }
    else {
        auto offset = input.shape().linearize(coords.data());
        return *reinterpret_cast<value_type_ const *>(input.byte_data() + offset);
    }
}

template <typename value_type_, std::size_t max_rank_, std::size_t extent_>
decltype(auto) tensor_lookup_resolved_(tensor_span<value_type_, max_rank_> input,
                                       std::span<std::size_t const, extent_> coords) noexcept {
    nk_assert_(input.byte_data() != nullptr);
    nk_assert_(coords.size() == input.rank());
    if constexpr (dimensions_per_value<value_type_>() > 1) {
        nk_assert_(packed_tensor_layout_supported_<value_type_>(input.shape()));
        auto offset = std::ptrdiff_t {};
        for (std::size_t i = 0; i + 1 < input.rank(); ++i)
            offset += static_cast<std::ptrdiff_t>(coords[i]) * input.stride_bytes(i);
        constexpr auto dims_per_value = dimensions_per_value<value_type_>();
        auto last_index = coords[input.rank() - 1];
        auto value_index = last_index / dims_per_value;
        auto sub_index = last_index % dims_per_value;
        using raw_type = typename raw_pod_type<value_type_>::type;
        auto *base = reinterpret_cast<raw_type *>(input.byte_data() + offset +
                                                  static_cast<std::ptrdiff_t>(value_index) *
                                                      input.stride_bytes(input.rank() - 1));
        return sub_byte_ref<value_type_>(base, sub_index);
    }
    else {
        auto offset = input.shape().linearize(coords.data());
        return *reinterpret_cast<value_type_ *>(input.byte_data() + offset);
    }
}

template <typename value_type_, std::size_t max_rank_, typename index_type_>
decltype(auto) tensor_flat_lookup_(tensor_view<value_type_, max_rank_> input, index_type_ idx) noexcept {
    nk_assert_(input.byte_data() != nullptr);
    if constexpr (dimensions_per_value<value_type_>() > 1) nk_assert_(input.rank() > 0);
    auto flat = resolve_index_(idx, input.numel());
    if constexpr (dimensions_per_value<value_type_>() == 1) {
        if (input.rank() == 0) return input.scalar();
    }

    std::array<std::size_t, max_rank_> coords {};
    for (std::size_t dim = input.rank(); dim > 0; --dim) {
        auto axis = dim - 1;
        auto extent = input.extent(axis);
        coords[axis] = flat % extent;
        flat /= extent;
    }
    return tensor_lookup_resolved_(input, std::span<std::size_t const>(coords.data(), input.rank()));
}

template <typename value_type_, std::size_t max_rank_, typename index_type_>
decltype(auto) tensor_flat_lookup_(tensor_span<value_type_, max_rank_> input, index_type_ idx) noexcept {
    nk_assert_(input.byte_data() != nullptr);
    if constexpr (dimensions_per_value<value_type_>() > 1) nk_assert_(input.rank() > 0);
    auto flat = resolve_index_(idx, input.numel());
    if constexpr (dimensions_per_value<value_type_>() == 1) {
        if (input.rank() == 0) return input.scalar_ref();
    }

    std::array<std::size_t, max_rank_> coords {};
    for (std::size_t dim = input.rank(); dim > 0; --dim) {
        auto axis = dim - 1;
        auto extent = input.extent(axis);
        coords[axis] = flat % extent;
        flat /= extent;
    }
    return tensor_lookup_resolved_(input, std::span<std::size_t const>(coords.data(), input.rank()));
}

template <typename tensor_type_>
tensor_type_ tensor_slice_suffix_(tensor_type_ input, tensor_slice_t) noexcept {
    return input;
}

template <typename tensor_type_, std::integral index_type_, typename... rest_types_>
tensor_type_ tensor_slice_suffix_(tensor_type_ input, index_type_ idx, rest_types_... rest) noexcept {
    if constexpr (dimensions_per_value<typename tensor_type_::value_type>() > 1) {
        if constexpr (sizeof...(rest_types_) == 1)
            if (input.rank() <= 1) return {};
    }
    if (input.rank() == 0) return {};
    return tensor_slice_suffix_(input.slice_leading(idx), rest...);
}

template <typename tensor_type_, typename... rest_types_>
tensor_type_ tensor_slice_suffix_(tensor_type_ input, all_t, rest_types_... rest) noexcept {
    // `all` keeps the leading dimension intact — apply remaining args to inner dimensions.
    if (input.rank() == 0) return {};
    using size_type = typename tensor_type_::size_type;
    using difference_type = typename tensor_type_::difference_type;
    using shape_type = std::remove_cvref_t<decltype(input.shape())>;

    auto leading_extent = input.extent(0);
    auto leading_stride = input.stride_bytes(0);

    // Slice the first row to discover the resulting sub-shape.
    auto first_row = input.slice_leading(static_cast<size_type>(0));
    auto inner = tensor_slice_suffix_(first_row, rest...);

    // Build the output shape: leading dimension + inner dimensions.
    shape_type result_shape;
    result_shape.rank = 1 + inner.rank();
    result_shape.extents[0] = leading_extent;
    result_shape.strides[0] = leading_stride;
    for (size_type d = 0; d < inner.rank(); ++d) {
        result_shape.extents[1 + d] = inner.extent(d);
        result_shape.strides[1 + d] = inner.stride_bytes(d);
    }

    // The data pointer is the inner slice's offset relative to the first row,
    // applied to the original data pointer.
    using byte_ptr = decltype(input.byte_data());
    auto inner_byte_offset = inner.byte_data() - first_row.byte_data();
    return {const_cast<byte_ptr>(input.byte_data() + inner_byte_offset), result_shape};
}

template <typename tensor_type_, typename... rest_types_>
tensor_type_ tensor_slice_suffix_(tensor_type_ input, range r, rest_types_... rest) noexcept {
    if (input.rank() == 0) return {};
    using size_type = typename tensor_type_::size_type;
    using difference_type = typename tensor_type_::difference_type;
    using shape_type = std::remove_cvref_t<decltype(input.shape())>;

    auto leading_extent = input.extent(0);
    auto leading_stride = input.stride_bytes(0);
    auto start = resolve_index_(r.start, leading_extent);
    auto stop = resolve_index_(r.stop, leading_extent);
    auto step = r.step;
    if (start >= stop || step <= 0) return {};

    auto range_extent = static_cast<size_type>((stop - start + static_cast<size_type>(step) - 1) /
                                               static_cast<size_type>(step));
    auto range_stride = leading_stride * static_cast<difference_type>(step);
    auto data_offset = static_cast<difference_type>(start) * leading_stride;

    if constexpr (sizeof...(rest_types_) == 1 &&
                  std::is_same_v<std::tuple_element_t<0, std::tuple<std::remove_cvref_t<rest_types_>...>>,
                                 tensor_slice_t>) {
        // Fast path: range followed by just `slice` — no inner recursion needed.
        shape_type result_shape;
        result_shape.rank = input.rank();
        result_shape.extents[0] = range_extent;
        result_shape.strides[0] = range_stride;
        for (size_type d = 1; d < input.rank(); ++d) {
            result_shape.extents[d] = input.extent(d);
            result_shape.strides[d] = input.stride_bytes(d);
        }
        using byte_ptr = decltype(input.byte_data());
        return {const_cast<byte_ptr>(input.byte_data() + data_offset), result_shape};
    }
    else {
        // General path: recurse into inner dimensions (like `all_t` but with narrowed leading).
        auto first_row = input.slice_leading(static_cast<size_type>(start));
        auto inner = tensor_slice_suffix_(first_row, rest...);

        shape_type result_shape;
        result_shape.rank = 1 + inner.rank();
        result_shape.extents[0] = range_extent;
        result_shape.strides[0] = range_stride;
        for (size_type d = 0; d < inner.rank(); ++d) {
            result_shape.extents[1 + d] = inner.extent(d);
            result_shape.strides[1 + d] = inner.stride_bytes(d);
        }

        using byte_ptr = decltype(input.byte_data());
        auto inner_byte_offset = inner.byte_data() - first_row.byte_data();
        return {const_cast<byte_ptr>(input.byte_data() + data_offset + inner_byte_offset), result_shape};
    }
}

#pragma region Axis Iterator

/**
 *  @brief Random-access iterator over slices along the leading dimension.
 *  @tparam view_type_ Either `tensor_view` or `tensor_span`.
 *
 *  For a rank-2 matrix, iterating yields rank-1 row views/spans.
 *  Dereference calls `parent_.slice_leading(index_)` to produce each sub-view.
 */
template <typename view_type_>
class axis_iterator {
    using value_type = typename view_type_::value_type;
    using difference_type = std::ptrdiff_t;

    char const *data_ = nullptr;
    difference_type stride_ = 0;
    std::size_t index_ = 0;
    view_type_ parent_;

  public:
    using iterator_category = std::random_access_iterator_tag;

    constexpr axis_iterator() noexcept = default;

    constexpr axis_iterator(view_type_ const &parent, std::size_t index) noexcept
        : data_(parent.byte_data()), stride_(parent.stride_bytes(0)), index_(index), parent_(parent) {}

    constexpr view_type_ operator*() const noexcept {
        return parent_.slice_leading(static_cast<difference_type>(index_));
    }

    constexpr axis_iterator &operator++() noexcept {
        ++index_;
        return *this;
    }
    constexpr axis_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++index_;
        return tmp;
    }
    constexpr axis_iterator &operator--() noexcept {
        --index_;
        return *this;
    }
    constexpr axis_iterator operator--(int) noexcept {
        auto tmp = *this;
        --index_;
        return tmp;
    }

    constexpr axis_iterator operator+(difference_type n) const noexcept {
        auto copy = *this;
        copy.index_ += n;
        return copy;
    }
    constexpr axis_iterator operator-(difference_type n) const noexcept {
        auto copy = *this;
        copy.index_ -= n;
        return copy;
    }
    constexpr difference_type operator-(axis_iterator const &other) const noexcept {
        return static_cast<difference_type>(index_) - static_cast<difference_type>(other.index_);
    }

    constexpr bool operator==(axis_iterator const &other) const noexcept { return index_ == other.index_; }
    constexpr bool operator!=(axis_iterator const &other) const noexcept { return index_ != other.index_; }
    constexpr bool operator<(axis_iterator const &other) const noexcept { return index_ < other.index_; }

    constexpr std::size_t index() const noexcept { return index_; }
};

#pragma endregion Axis Iterator

#pragma region Tensor Element Iterators

/**
 *  @brief Forward iterator over all logical scalar elements of a const tensor view.
 *
 *  Yields `std::pair<index_type, scalar_type>` where `index_type` is an N-dimensional
 *  position array and `scalar_type` is the unpacked dimension scalar (a copy).
 *  For sub-byte types the innermost axis is split into per-dimension offsets.
 */
template <typename view_type_>
class tensor_view_iterator_ {
    using value_type_ = typename view_type_::value_type;
    static constexpr std::size_t max_rank_ = view_type_::max_rank;
    static constexpr unsigned dims_per_value_ = dimensions_per_value<value_type_>();

    char const *data_ = nullptr;
    std::size_t extents_[max_rank_] = {};
    std::ptrdiff_t strides_[max_rank_] = {};
    std::size_t ndim_ = 0;
    std::size_t indices_[max_rank_] = {};
    std::size_t remaining_ = 0;

  public:
    using index_type = std::array<std::size_t, max_rank_>;
    using scalar_type = typename value_type_::component_t;
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
    using value_type = std::pair<index_type, scalar_type>;

    constexpr tensor_view_iterator_() noexcept = default;

    constexpr tensor_view_iterator_(view_type_ const &parent, bool at_end = false) noexcept
        : data_(parent.byte_data()), ndim_(parent.rank()), remaining_(at_end || parent.empty() ? 0 : parent.numel()) {
        for (std::size_t i = 0; i < ndim_; ++i) {
            extents_[i] = parent.extent(i);
            strides_[i] = parent.stride_bytes(i);
        }
    }

    constexpr value_type operator*() const noexcept {
        index_type pos {};
        std::ptrdiff_t offset = 0;
        for (std::size_t d = 0; d + 1 < ndim_; ++d) {
            pos[d] = indices_[d];
            offset += static_cast<std::ptrdiff_t>(indices_[d]) * strides_[d];
        }
        if constexpr (dims_per_value_ == 1) {
            if (ndim_ == 0) return {pos, *reinterpret_cast<value_type_ const *>(data_)};
            std::size_t inner = indices_[ndim_ - 1];
            pos[ndim_ - 1] = inner;
            offset += static_cast<std::ptrdiff_t>(inner) * strides_[ndim_ - 1];
            return {pos, *reinterpret_cast<value_type_ const *>(data_ + offset)};
        }
        else {
            std::size_t inner = indices_[ndim_ - 1];
            pos[ndim_ - 1] = inner;
            std::size_t storage_idx = inner / dims_per_value_;
            std::size_t sub_idx = inner % dims_per_value_;
            offset += static_cast<std::ptrdiff_t>(storage_idx) * strides_[ndim_ - 1];
            using raw_type = typename raw_pod_type<value_type_>::type;
            auto *raw = const_cast<raw_type *>(reinterpret_cast<raw_type const *>(data_ + offset));
            return {pos, sub_byte_ref<value_type_>(raw, sub_idx).get()};
        }
    }

    constexpr tensor_view_iterator_ &operator++() noexcept {
        --remaining_;
        for (std::size_t d = ndim_; d > 0; --d) {
            if (++indices_[d - 1] < extents_[d - 1]) return *this;
            indices_[d - 1] = 0;
        }
        return *this;
    }

    constexpr tensor_view_iterator_ operator++(int) noexcept {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    constexpr bool operator==(tensor_view_iterator_ const &o) const noexcept { return remaining_ == o.remaining_; }
    constexpr bool operator!=(tensor_view_iterator_ const &o) const noexcept { return remaining_ != o.remaining_; }

    constexpr index_type position() const noexcept {
        index_type pos {};
        for (std::size_t d = 0; d < ndim_; ++d) pos[d] = indices_[d];
        return pos;
    }

    constexpr size_type remaining() const noexcept { return remaining_; }
};

/**
 *  @brief Forward iterator over all logical scalar elements of a mutable tensor span.
 *
 *  Yields `std::pair<index_type, T&>` for normal types or
 *  `std::pair<index_type, sub_byte_ref<T>>` for sub-byte types.
 */
template <typename span_type_>
class tensor_span_iterator_ {
    using value_type_ = typename span_type_::value_type;
    static constexpr std::size_t max_rank_ = span_type_::max_rank;
    static constexpr unsigned dims_per_value_ = dimensions_per_value<value_type_>();

    char *data_ = nullptr;
    std::size_t extents_[max_rank_] = {};
    std::ptrdiff_t strides_[max_rank_] = {};
    std::size_t ndim_ = 0;
    std::size_t indices_[max_rank_] = {};
    std::size_t remaining_ = 0;

  public:
    using index_type = std::array<std::size_t, max_rank_>;
    using value_reference_type = value_ref<value_type_>;
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
    using value_type = std::pair<index_type, value_reference_type>;

    constexpr tensor_span_iterator_() noexcept = default;

    constexpr tensor_span_iterator_(span_type_ &parent, bool at_end = false) noexcept
        : data_(parent.byte_data()), ndim_(parent.rank()), remaining_(at_end || parent.empty() ? 0 : parent.numel()) {
        for (std::size_t i = 0; i < ndim_; ++i) {
            extents_[i] = parent.extent(i);
            strides_[i] = parent.stride_bytes(i);
        }
    }

    value_type operator*() const noexcept {
        index_type pos {};
        std::ptrdiff_t offset = 0;
        for (std::size_t d = 0; d + 1 < ndim_; ++d) {
            pos[d] = indices_[d];
            offset += static_cast<std::ptrdiff_t>(indices_[d]) * strides_[d];
        }
        if constexpr (dims_per_value_ == 1) {
            if (ndim_ == 0) return {pos, *reinterpret_cast<value_type_ *>(data_)};
            std::size_t inner = indices_[ndim_ - 1];
            pos[ndim_ - 1] = inner;
            offset += static_cast<std::ptrdiff_t>(inner) * strides_[ndim_ - 1];
            return {pos, *reinterpret_cast<value_type_ *>(data_ + offset)};
        }
        else {
            std::size_t inner = indices_[ndim_ - 1];
            pos[ndim_ - 1] = inner;
            std::size_t storage_idx = inner / dims_per_value_;
            std::size_t sub_idx = inner % dims_per_value_;
            offset += static_cast<std::ptrdiff_t>(storage_idx) * strides_[ndim_ - 1];
            auto *raw = reinterpret_cast<typename raw_pod_type<value_type_>::type *>(data_ + offset);
            return {pos, sub_byte_ref<value_type_>(raw, sub_idx)};
        }
    }

    constexpr tensor_span_iterator_ &operator++() noexcept {
        --remaining_;
        for (std::size_t d = ndim_; d > 0; --d) {
            if (++indices_[d - 1] < extents_[d - 1]) return *this;
            indices_[d - 1] = 0;
        }
        return *this;
    }

    constexpr tensor_span_iterator_ operator++(int) noexcept {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    constexpr bool operator==(tensor_span_iterator_ const &o) const noexcept { return remaining_ == o.remaining_; }
    constexpr bool operator!=(tensor_span_iterator_ const &o) const noexcept { return remaining_ != o.remaining_; }

    constexpr index_type position() const noexcept {
        index_type pos {};
        for (std::size_t d = 0; d < ndim_; ++d) pos[d] = indices_[d];
        return pos;
    }

    constexpr size_type remaining() const noexcept { return remaining_; }
};

/**
 *  @brief Adapter view that strips positions from a tensor element iterator, yielding only scalars.
 *
 *  Works with both `tensor_view_iterator_` (yields scalar copies) and
 *  `tensor_span_iterator_` (yields references or sub-byte proxies).
 */
template <typename iterator_type_>
struct tensor_dims_view_ {
    iterator_type_ begin_;
    std::size_t size_;

    struct iterator_ {
        iterator_type_ it_;

        decltype(auto) operator*() const noexcept { return (*it_).second; }
        iterator_ &operator++() noexcept {
            ++it_;
            return *this;
        }
        iterator_ operator++(int) noexcept {
            auto tmp = *this;
            ++it_;
            return tmp;
        }
        bool operator==(iterator_ const &o) const noexcept { return it_ == o.it_; }
        bool operator!=(iterator_ const &o) const noexcept { return it_ != o.it_; }
    };

    iterator_ begin() const noexcept { return {begin_}; }
    iterator_ end() const noexcept {
        auto end_it = begin_;
        // Advance to end by constructing an iterator with remaining_==0
        // We stored size_ so we can compare via remaining counts
        iterator_type_ sentinel;
        return {sentinel};
    }
    std::size_t size() const noexcept { return size_; }
};

#pragma endregion Tensor Element Iterators

#pragma region Tensor

/**
 *  @brief Owning, non-resizable, N-dimensional tensor.
 *  @tparam value_type_ Element type.
 *  @tparam allocator_type_ Allocator.
 *  @tparam max_rank_ Maximum number of dimensions.
 *
 *  Fixed-size at construction. Use `try_zeros()` factory for non-throwing construction.
 */
template <typename value_type_, typename allocator_type_ = aligned_allocator<value_type_>, std::size_t max_rank_ = 8>
struct tensor {
    using value_type = value_type_;
    using allocator_type = allocator_type_;
    using alloc_traits = std::allocator_traits<allocator_type_>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type_ *;

    using view_type = tensor_view<value_type_, max_rank_>;
    using span_type = tensor_span<value_type_, max_rank_>;

  private:
    pointer data_ = nullptr;
    shape_storage_<max_rank_> shape_;
    [[no_unique_address]] allocator_type_ alloc_;

  public:
    tensor() noexcept = default;

    explicit tensor(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    ~tensor() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, storage_values_for_shape_<value_type_>(shape_));
    }

    tensor(tensor &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), shape_(std::exchange(other.shape_, {})),
          alloc_(std::move(other.alloc_)) {}

    tensor &operator=(tensor &&other) noexcept {
        if (this != &other) {
            if (data_) alloc_traits::deallocate(alloc_, data_, storage_values_for_shape_<value_type_>(shape_));
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(other.alloc_);
            data_ = std::exchange(other.data_, nullptr);
            shape_ = std::exchange(other.shape_, {});
        }
        return *this;
    }

    tensor(tensor const &) = delete;
    tensor &operator=(tensor const &) = delete;

    /**
     *  @brief Factory: allocate a zero-initialized tensor with the given extents.
     *  @param extents Extents (one per dimension), e.g. `{3, 4}`.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on failure.
     */
    [[nodiscard]] static tensor try_zeros(std::initializer_list<size_type> extents,
                                          allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        auto rank = extents.size();
        if (rank > max_rank_) return t;
        t.shape_ = make_contiguous_shape_<value_type_, max_rank_>(extents.begin(), rank);
        auto storage_values = storage_values_for_shape_<value_type_>(t.shape_);
        if (storage_values == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, storage_values);
        if (!ptr) return t;
        if constexpr (is_memset_zero_safe_v<value_type_>)
            std::memset(static_cast<void *>(ptr), 0, storage_values * sizeof(value_type_));
        else
            for (size_type i = 0; i < storage_values; ++i) ptr[i] = value_type_ {};
        t.data_ = ptr;
        return t;
    }

    /**
     *  @brief Factory: allocate a tensor filled with ones.
     *  @param extents Extents (one per dimension), e.g. `{3, 4}`.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on failure.
     */
    [[nodiscard]] static tensor try_ones(std::initializer_list<size_type> extents,
                                         allocator_type_ alloc = {}) noexcept {
        return try_full(extents, value_type_ {1}, alloc);
    }

    /**
     *  @brief Factory: allocate a tensor filled with a given value.
     *  @param extents Extents (one per dimension), e.g. `{3, 4}`.
     *  @param val Fill value.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on failure.
     */
    [[nodiscard]] static tensor try_full(std::initializer_list<size_type> extents, value_type_ val,
                                         allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        auto rank = extents.size();
        if (rank > max_rank_) return t;
        t.shape_ = make_contiguous_shape_<value_type_, max_rank_>(extents.begin(), rank);
        auto storage_values = storage_values_for_shape_<value_type_>(t.shape_);
        if (storage_values == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, storage_values);
        if (!ptr) return t;
        for (size_type i = 0; i < storage_values; ++i) ptr[i] = val;
        t.data_ = ptr;
        return t;
    }

    /**
     *  @brief Factory: allocate an uninitialized tensor.
     *  @param extents Extents (one per dimension), e.g. `{3, 4}`.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on failure.
     */
    [[nodiscard]] static tensor try_empty(std::initializer_list<size_type> extents,
                                          allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        auto rank = extents.size();
        if (rank > max_rank_) return t;
        t.shape_ = make_contiguous_shape_<value_type_, max_rank_>(extents.begin(), rank);
        auto storage_values = storage_values_for_shape_<value_type_>(t.shape_);
        if (storage_values == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, storage_values);
        if (!ptr) return t;
        t.data_ = ptr;
        return t;
    }

    /** @brief Factory: zero-initialized tensor from pointer + rank. */
    [[nodiscard]] static tensor try_zeros(size_type const *extents, size_type rank,
                                          allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        if (rank > max_rank_) return t;
        t.shape_ = make_contiguous_shape_<value_type_, max_rank_>(extents, rank);
        auto storage_values = storage_values_for_shape_<value_type_>(t.shape_);
        if (storage_values == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, storage_values);
        if (!ptr) return t;
        if constexpr (is_memset_zero_safe_v<value_type_>)
            std::memset(static_cast<void *>(ptr), 0, storage_values * sizeof(value_type_));
        else
            for (size_type i = 0; i < storage_values; ++i) ptr[i] = value_type_ {};
        t.data_ = ptr;
        return t;
    }

    /** @brief Factory: uninitialized tensor from pointer + rank. */
    [[nodiscard]] static tensor try_empty(size_type const *extents, size_type rank,
                                          allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        if (rank > max_rank_) return t;
        t.shape_ = make_contiguous_shape_<value_type_, max_rank_>(extents, rank);
        auto storage_values = storage_values_for_shape_<value_type_>(t.shape_);
        if (storage_values == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, storage_values);
        if (!ptr) return t;
        t.data_ = ptr;
        return t;
    }

    /** @brief Factory: filled tensor from pointer + rank. */
    [[nodiscard]] static tensor try_full(size_type const *extents, size_type rank, value_type_ val,
                                         allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        if (rank > max_rank_) return t;
        t.shape_ = make_contiguous_shape_<value_type_, max_rank_>(extents, rank);
        auto storage_values = storage_values_for_shape_<value_type_>(t.shape_);
        if (storage_values == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, storage_values);
        if (!ptr) return t;
        for (size_type i = 0; i < storage_values; ++i) ptr[i] = val;
        t.data_ = ptr;
        return t;
    }

    /**
     *  @brief Factory: create a rank-1 tensor from an initializer list of values.
     *  @param values Values to fill the tensor with.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on failure.
     */
    [[nodiscard]] static tensor try_from(std::initializer_list<value_type_> values,
                                         allocator_type_ alloc = {}) noexcept {
        tensor t = try_empty({values.size()}, alloc);
        if (t.empty()) return t;
        size_type index = 0;
        for (auto const &value : values) t.data_[index++] = value;
        return t;
    }

    /**
     *  @brief Factory: create a rank-2 tensor from a nested initializer list.
     *  @param rows Each inner list is a row. All rows must have the same length.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on ragged input or allocation failure.
     */
    [[nodiscard]] static tensor try_from(std::initializer_list<std::initializer_list<value_type_>> rows,
                                         allocator_type_ alloc = {}) noexcept
        requires(max_rank_ >= 2)
    {
        auto num_rows = rows.size();
        if (num_rows == 0) return tensor(alloc);
        auto num_cols = rows.begin()->size();
        for (auto const &row : rows)
            if (row.size() != num_cols) return tensor(alloc);
        tensor t = try_empty({num_rows, num_cols}, alloc);
        if (t.empty()) return t;
        size_type index = 0;
        for (auto const &row : rows)
            for (auto const &value : row) t.data_[index++] = value;
        return t;
    }

    /**
     *  @brief Factory: adopt raw memory.
     */
    [[nodiscard]] static tensor from_raw(pointer ptr, shape_storage_<max_rank_> const &shape,
                                         allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        t.data_ = ptr;
        t.shape_ = shape;
        return t;
    }

    /** @brief Number of dimensions. */
    constexpr size_type rank() const noexcept { return shape_.rank; }

    /** @brief Extent along dimension i. */
    constexpr size_type extent(size_type i) const noexcept { return shape_.extents[i]; }

    /** @brief Stride in bytes along dimension i (signed). */
    constexpr difference_type stride_bytes(size_type i) const noexcept { return shape_.strides[i]; }

    /** @brief Total number of elements. */
    constexpr size_type numel() const noexcept { return shape_.numel(); }

    /** @brief True if empty. */
    constexpr bool empty() const noexcept { return data_ == nullptr; }

    /** @brief Typed pointer to data. */
    pointer data() noexcept { return data_; }
    value_type const *data() const noexcept { return data_; }

    /** @brief Shape storage. */
    constexpr shape_storage_<max_rank_> const &shape() const noexcept { return shape_; }

    /** @brief Allocator. */
    allocator_type get_allocator() const noexcept { return alloc_; }

    /** @brief Create an immutable view. */
    view_type view() const noexcept { return {reinterpret_cast<char const *>(data_), shape_}; }

    /** @brief Create a mutable span. */
    span_type span() noexcept { return {reinterpret_cast<char *>(data_), shape_}; }

    /** @brief Range of immutable row views (slices along leading dimension). */
    struct rows_views_t {
        view_type parent;
        axis_iterator<view_type> begin() const noexcept { return {parent, 0}; }
        axis_iterator<view_type> end() const noexcept { return {parent, parent.extent(0)}; }
    };

    /** @brief Range of mutable row spans (slices along leading dimension). */
    struct rows_spans_t {
        span_type parent;
        axis_iterator<span_type> begin() noexcept { return {parent, 0}; }
        axis_iterator<span_type> end() noexcept { return {parent, parent.extent(0)}; }
    };

    /** @brief Iterate rows as immutable views. */
    rows_views_t rows_views() const noexcept { return {view()}; }

    /** @brief Iterate rows as mutable spans. */
    rows_spans_t rows_spans() noexcept { return {span()}; }

    /** @brief Iterate rows as immutable views (convenience alias for rows_views). */
    typename view_type::rows_views_t rows() const noexcept { return view().rows(); }

    /** @brief Iterate rows as mutable spans (convenience alias for rows_spans). */
    typename span_type::rows_spans_t rows() noexcept { return span().rows(); }

    /** @brief Const element iterator (begin). */
    tensor_view_iterator_<view_type> begin() const noexcept { return view().begin(); }
    /** @brief Const element iterator (end). */
    tensor_view_iterator_<view_type> end() const noexcept { return view().end(); }
    /** @brief Mutable element iterator (begin). */
    tensor_span_iterator_<span_type> begin() noexcept { return span().begin(); }
    /** @brief Mutable element iterator (end). */
    tensor_span_iterator_<span_type> end() noexcept { return span().end(); }
    /** @brief Number of logical scalar elements. */
    constexpr size_type size() const noexcept { return numel(); }
    /** @brief Const dimension-only view. */
    tensor_dims_view_<tensor_view_iterator_<view_type>> dims() const noexcept { return view().dims(); }
    /** @brief Mutable dimension-only view. */
    tensor_dims_view_<tensor_span_iterator_<span_type>> dims() noexcept { return span().dims(); }

    /** @brief Reinterpret as a 2D immutable matrix view. Requires rank >= 2. */
    tensor_view<value_type_, 2> as_matrix_view() const noexcept { return view().as_matrix(); }

    /** @brief Reinterpret as a 2D mutable matrix span. Requires rank >= 2. */
    tensor_span<value_type_, 2> as_matrix_span() noexcept { return span().as_matrix(); }

    /** @brief Transpose: reverse dimension order (immutable view). */
    view_type transpose() const noexcept { return view().transpose(); }

    /** @brief Transpose: reverse dimension order (mutable span). */
    span_type transpose() noexcept { return span().transpose(); }

    /** @brief Reshape (immutable view). Requires contiguous layout and matching element count. */
    view_type reshape(std::initializer_list<size_type> new_extents) const noexcept {
        return view().reshape(new_extents);
    }

    /** @brief Reshape (mutable span). Requires contiguous layout and matching element count. */
    span_type reshape(std::initializer_list<size_type> new_extents) noexcept { return span().reshape(new_extents); }

    /** @brief Check if contiguous in memory. Always true for freshly-constructed tensors. */
    constexpr bool is_contiguous() const noexcept { return view().is_contiguous(); }

    /** @brief Slice along leading dimension (immutable view). */
    template <std::integral index_type_>
    view_type slice_leading(index_type_ idx) const noexcept {
        return view().slice_leading(idx);
    }

    /** @brief Slice along leading dimension (mutable span). */
    template <std::integral index_type_>
    span_type slice_leading(index_type_ idx) noexcept {
        return span().slice_leading(idx);
    }

    /** @brief Row access (immutable view, alias for slice_leading). */
    template <std::integral index_type_>
    view_type row(index_type_ i) const noexcept {
        return view().slice_leading(i);
    }

    /** @brief Row access (mutable span, alias for slice_leading). */
    template <std::integral index_type_>
    span_type row(index_type_ i) noexcept {
        return span().slice_leading(i);
    }

    /** @brief Flat logical scalar access. */
    template <std::integral index_type_>
    decltype(auto) operator[](index_type_ idx) noexcept {
        return span()[idx];
    }

    /** @brief Const flat logical scalar access. */
    template <std::integral index_type_>
    decltype(auto) operator[](index_type_ idx) const noexcept {
        return view()[idx];
    }

    /** @brief Exact multi-dimensional scalar lookup via call syntax (C++20-portable). */
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator()(index_types_... idxs) noexcept {
        return span()(idxs...);
    }

    /** @brief Const multidimensional lookup via call syntax. */
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator()(index_types_... idxs) const noexcept {
        return view()(idxs...);
    }

#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    /** @brief C++23 sugar: multi-arg `[]` scalar lookup, delegates to `operator()`. */
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator[](index_types_... idxs) noexcept {
        return (*this)(idxs...);
    }
    template <std::integral... index_types_>
        requires(sizeof...(index_types_) >= 2)
    decltype(auto) operator[](index_types_... idxs) const noexcept {
        return (*this)(idxs...);
    }
#endif

    /** @brief Trailing `slice` returns the same tensor view/span category. */
    span_type operator[](tensor_slice_t) noexcept { return span(); }
    view_type operator[](tensor_slice_t) const noexcept { return view(); }

    /** @brief Prefix leading-axis slicing via call syntax (C++20-portable). */
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    span_type operator()(first_type_ first, second_type_ second, rest_types_... rest) noexcept {
        return tensor_slice_suffix_(span(), first, second, rest...);
    }

    /** @brief Const prefix leading-axis slicing via call syntax. */
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    view_type operator()(first_type_ first, second_type_ second, rest_types_... rest) const noexcept {
        return tensor_slice_suffix_(view(), first, second, rest...);
    }

#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    /** @brief C++23 sugar: multi-arg `[]` slicing, delegates to `operator()`. */
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    span_type operator[](first_type_ first, second_type_ second, rest_types_... rest) noexcept {
        return (*this)(first, second, rest...);
    }
    template <typename first_type_, typename second_type_, typename... rest_types_>
        requires(trailing_tensor_slice_args_v<first_type_, second_type_, rest_types_...>)
    view_type operator[](first_type_ first, second_type_ second, rest_types_... rest) const noexcept {
        return (*this)(first, second, rest...);
    }
#endif

    /** @brief Rank-0 mutable scalar access. */
    decltype(auto) scalar_ref() noexcept { return span().scalar_ref(); }

    /** @brief Rank-0 const scalar access. */
    decltype(auto) scalar() const noexcept { return view().scalar(); }

    /** @brief Convert to vector_view (requires rank == 1). */
    vector_view<value_type> as_vector_view() const noexcept { return view().as_vector(); }

    /** @brief Convert to vector_span (requires rank == 1). */
    vector_span<value_type> as_vector_span() noexcept { return span().as_vector(); }

    /** @brief Flatten (immutable view). Requires contiguous layout. */
    view_type flatten() const noexcept { return view().flatten(); }

    /** @brief Flatten (mutable span). Requires contiguous layout. */
    span_type flatten() noexcept { return span().flatten(); }

    /** @brief Squeeze (immutable view). Removes size-1 dimensions. */
    view_type squeeze() const noexcept { return view().squeeze(); }

    /** @brief Squeeze (mutable span). Removes size-1 dimensions. */
    span_type squeeze() noexcept { return span().squeeze(); }
};

/** @brief Non-member swap. */
template <typename V, typename A, std::size_t R>
void swap(tensor<V, A, R> &a, tensor<V, A, R> &b) noexcept {
    auto tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

#pragma endregion Tensor

#pragma region Matrix Aliases

/** @brief 2D owning matrix (max_rank = 2, smaller shape_storage). */
template <typename value_type_, typename allocator_type_ = aligned_allocator<value_type_>>
using matrix = tensor<value_type_, allocator_type_, 2>;

/** @brief 2D immutable view. */
template <typename value_type_>
using matrix_view = tensor_view<value_type_, 2>;

/** @brief 2D mutable span. */
template <typename value_type_>
using matrix_span = tensor_span<value_type_, 2>;

#pragma endregion Matrix Aliases

} // namespace ashvardanian::numkong

namespace ashvardanian::numkong {

#pragma region Enums and Result Types

/** @brief Controls whether reduction collapses or preserves the reduced axis. */
enum keep_dims_t : bool { collapse_dims_k = false, keep_dims_k = true };

/** @brief Result of moments(): Σxᵢ and Σxᵢ². */
template <typename sum_type_, typename sumsq_type_>
struct moments_result {
    sum_type_ sum {};
    sumsq_type_ sumsq {};
};

/** @brief Result of minmax(): min/max values with their indices. */
template <typename minmax_value_type_>
struct minmax_result {
    minmax_value_type_ min_value {};
    std::size_t min_index = 0;
    minmax_value_type_ max_value {};
    std::size_t max_index = 0;
};

#pragma endregion Enums and Result Types

#pragma region Helpers

/** @brief Compute output shape with one axis removed (or set to 1 if keep_dims). */
template <typename value_type_, std::size_t max_rank_>
shape_storage_<max_rank_> reduced_shape_(shape_storage_<max_rank_> const &in, std::size_t axis,
                                         keep_dims_t keep_dims) noexcept {
    std::size_t out_extents[max_rank_];
    std::size_t out_rank = 0;
    for (std::size_t i = 0; i < in.rank; ++i) {
        if (i == axis) {
            if (keep_dims) out_extents[out_rank++] = 1;
        }
        else { out_extents[out_rank++] = in.extents[i]; }
    }
    return make_contiguous_shape_<value_type_, max_rank_>(out_extents, out_rank);
}

/** @brief Validate that two views have matching shapes. */
template <typename value_type_, std::size_t max_rank_>
bool shapes_match_(tensor_view<value_type_, max_rank_> a, tensor_view<value_type_, max_rank_> b) noexcept {
    if (a.rank() != b.rank()) return false;
    for (std::size_t i = 0; i < a.rank(); ++i)
        if (a.extent(i) != b.extent(i)) return false;
    return true;
}

/** @brief Validate shape match between view and span. */
template <typename in_type_, typename out_type_, std::size_t max_rank_>
bool shapes_match_out_(tensor_view<in_type_, max_rank_> a, tensor_span<out_type_, max_rank_> out) noexcept {
    if (a.rank() != out.rank()) return false;
    for (std::size_t i = 0; i < a.rank(); ++i)
        if (a.extent(i) != out.extent(i)) return false;
    return true;
}

template <typename value_type_, std::size_t max_rank_>
bool tensor_layout_supported_(tensor_view<value_type_, max_rank_> input) noexcept {
    return packed_tensor_layout_supported_<value_type_>(input.shape());
}

template <typename value_type_, std::size_t max_rank_>
bool tensor_layout_supported_(tensor_span<value_type_, max_rank_> input) noexcept {
    return packed_tensor_layout_supported_<value_type_>(input.shape());
}

template <typename value_type_, std::size_t max_rank_>
bool shape_matches_(shape_storage_<max_rank_> const &expected, tensor_span<value_type_, max_rank_> actual) noexcept {
    if (expected.rank != actual.rank()) return false;
    for (std::size_t i = 0; i < expected.rank; ++i)
        if (expected.extents[i] != actual.extent(i)) return false;
    return true;
}

template <typename value_type_, std::size_t max_rank_>
struct normalized_rank1_lane_ {
    value_type_ const *data = nullptr;
    std::size_t count = 0;
    std::size_t stride_bytes = sizeof(value_type_);
    bool reversed = false;
};

template <typename value_type_, std::size_t max_rank_>
bool can_reduce_rank1_with_kernel_(tensor_view<value_type_, max_rank_> input) noexcept {
    if (input.rank() != 1 || input.byte_data() == nullptr || !tensor_layout_supported_(input)) return false;
    if constexpr (dimensions_per_value<value_type_>() > 1) return input.is_contiguous();
    return input.stride_bytes(0) != 0;
}

template <typename value_type_, std::size_t max_rank_>
bool can_apply_rank1_data_kernel_(tensor_view<value_type_, max_rank_> input) noexcept {
    if (input.rank() != 1 || input.byte_data() == nullptr || !tensor_layout_supported_(input)) return false;
    return input.is_contiguous();
}

template <typename value_type_, std::size_t max_rank_>
bool can_apply_rank1_data_kernel_(tensor_span<value_type_, max_rank_> output) noexcept {
    if (output.rank() != 1 || output.byte_data() == nullptr || !tensor_layout_supported_(output)) return false;
    return output.is_contiguous();
}

template <typename value_type_, std::size_t max_rank_>
normalized_rank1_lane_<value_type_, max_rank_> normalize_rank1_lane_(
    tensor_view<value_type_, max_rank_> input) noexcept {
    normalized_rank1_lane_<value_type_, max_rank_> lane;
    if (input.rank() != 1 || input.byte_data() == nullptr) return lane;
    lane.count = input.extent(0);
    if (lane.count == 0) return lane;
    auto stride = input.stride_bytes(0);
    if constexpr (dimensions_per_value<value_type_>() > 1) {
        if (!input.is_contiguous()) return {};
        lane.data = input.data();
        lane.stride_bytes = sizeof(value_type_);
        lane.reversed = false;
        return lane;
    }
    if (stride >= 0) {
        lane.data = input.data();
        lane.stride_bytes = static_cast<std::size_t>(stride);
        lane.reversed = false;
    }
    else {
        lane.data = reinterpret_cast<value_type_ const *>(input.byte_data() + (lane.count - 1) * stride);
        lane.stride_bytes = static_cast<std::size_t>(-stride);
        lane.reversed = true;
    }
    return lane;
}

template <typename value_type_, std::size_t max_rank_, typename lane_fn_>
bool for_each_axis_lane_(tensor_view<value_type_, max_rank_> input, std::size_t axis, lane_fn_ &&lane_fn) noexcept {
    if (axis >= input.rank() || !tensor_layout_supported_(input) || input.byte_data() == nullptr) return false;

    shape_storage_<max_rank_> lane_shape;
    lane_shape.rank = 1;
    lane_shape.extents[0] = input.extent(axis);
    lane_shape.strides[0] = input.stride_bytes(axis);

    std::size_t remaining_dims[max_rank_] = {};
    std::size_t remaining_count = 0;
    for (std::size_t dim = 0; dim < input.rank(); ++dim) {
        if (dim != axis) remaining_dims[remaining_count++] = dim;
    }

    if (remaining_count == 0) return lane_fn(tensor_view<value_type_, max_rank_> {input.byte_data(), lane_shape}, 0);

    std::size_t total_lanes = 1;
    for (std::size_t i = 0; i < remaining_count; ++i) total_lanes *= input.extent(remaining_dims[i]);

    // When non-axis dims form a uniform-stride progression, iterate with a constant byte
    // increment instead of recomputing offsets from multi-dimensional coordinates.
    {
        std::size_t other_extents[max_rank_];
        std::ptrdiff_t other_strides[max_rank_];
        for (std::size_t i = 0; i < remaining_count; ++i) {
            other_extents[i] = input.extent(remaining_dims[i]);
            other_strides[i] = input.stride_bytes(remaining_dims[i]);
        }
        bool all_collapse = true;
        auto expected_stride = other_strides[remaining_count - 1];
        for (std::size_t i = remaining_count - 1; i > 0 && all_collapse; --i) {
            expected_stride *= static_cast<std::ptrdiff_t>(other_extents[i]);
            if (other_strides[i - 1] != expected_stride) all_collapse = false;
        }
        if (all_collapse) {
            auto lane_byte_increment = other_strides[remaining_count - 1];
            auto *ptr = input.byte_data();
            for (std::size_t lane_index = 0; lane_index < total_lanes; ++lane_index, ptr += lane_byte_increment) {
                if (!lane_fn(tensor_view<value_type_, max_rank_> {ptr, lane_shape}, lane_index)) return false;
            }
            return true;
        }
    }

    std::size_t coords[max_rank_] = {};
    for (std::size_t lane_index = 0; lane_index < total_lanes; ++lane_index) {
        auto offset = std::ptrdiff_t {};
        for (std::size_t i = 0; i < remaining_count; ++i)
            offset += static_cast<std::ptrdiff_t>(coords[i]) * input.stride_bytes(remaining_dims[i]);
        if (!lane_fn(tensor_view<value_type_, max_rank_> {input.byte_data() + offset, lane_shape}, lane_index))
            return false;

        for (std::size_t i = remaining_count; i > 0; --i) {
            auto coord_index = i - 1;
            auto dim = remaining_dims[coord_index];
            if (++coords[coord_index] < input.extent(dim)) break;
            coords[coord_index] = 0;
        }
    }
    return true;
}

/** @brief Count trailing dimensions that are contiguous across all stride arrays.
 *  Returns how many rightmost dims can be collapsed into a single contiguous slice. */
template <typename value_type_, std::size_t max_rank_>
std::size_t shared_contiguous_tail_dims_(std::size_t rank, std::size_t const *extents,
                                         std::initializer_list<std::ptrdiff_t const *> all_strides) noexcept {
    if constexpr (dimensions_per_value<value_type_>() > 1) return 0;
    std::size_t tail = 0;
    for (std::size_t i = rank; i > 0; --i) {
        auto dim = i - 1;
        auto expected = static_cast<std::ptrdiff_t>(sizeof(value_type_));
        for (std::size_t d = rank; d > dim + 1; --d) expected *= static_cast<std::ptrdiff_t>(extents[d - 1]);
        bool all_match = true;
        for (auto const *strides : all_strides) {
            if (strides[dim] != expected) {
                all_match = false;
                break;
            }
        }
        if (!all_match) break;
        ++tail;
    }
    return tail;
}

/** @brief Collapse contiguous trailing dimensions of a tensor_view into one. */
template <typename value_type_, std::size_t max_rank_>
tensor_view<value_type_, max_rank_> collapse_contiguous_tail_(tensor_view<value_type_, max_rank_> input,
                                                              std::size_t tail_dims) noexcept {
    shape_storage_<max_rank_> s;
    s.rank = input.rank() - tail_dims + 1;
    for (std::size_t i = 0; i + tail_dims < input.rank(); ++i) {
        s.extents[i] = input.extent(i);
        s.strides[i] = input.stride_bytes(i);
    }
    std::size_t product = 1;
    for (std::size_t i = input.rank() - tail_dims; i < input.rank(); ++i) product *= input.extent(i);
    s.extents[s.rank - 1] = product;
    s.strides[s.rank - 1] = static_cast<std::ptrdiff_t>(sizeof(value_type_));
    return {input.byte_data(), s};
}

/** @brief Collapse contiguous trailing dimensions of a tensor_span into one. */
template <typename value_type_, std::size_t max_rank_>
tensor_span<value_type_, max_rank_> collapse_contiguous_tail_(tensor_span<value_type_, max_rank_> input,
                                                              std::size_t tail_dims) noexcept {
    shape_storage_<max_rank_> s;
    s.rank = input.rank() - tail_dims + 1;
    for (std::size_t i = 0; i + tail_dims < input.rank(); ++i) {
        s.extents[i] = input.extent(i);
        s.strides[i] = input.stride_bytes(i);
    }
    std::size_t product = 1;
    for (std::size_t i = input.rank() - tail_dims; i < input.rank(); ++i) product *= input.extent(i);
    s.extents[s.rank - 1] = product;
    s.strides[s.rank - 1] = static_cast<std::ptrdiff_t>(sizeof(value_type_));
    return {input.byte_data(), s};
}

/** @brief Unary elementwise traversal: validates shapes, recurses on rank≥2, calls leaf on rank-1 slices. */
template <typename value_type_, std::size_t max_rank_, typename leaf_fn_>
bool elementwise_into_(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output,
                       leaf_fn_ &&leaf) noexcept {
    if (!shapes_match_out_(input, output) || !tensor_layout_supported_(input) || !tensor_layout_supported_(output))
        return false;
    if (input.empty()) return true;
    if (input.rank() >= 2) {
        auto tail = shared_contiguous_tail_dims_<value_type_, max_rank_>(
            input.rank(), input.shape().extents, {input.shape().strides, output.shape().strides});
        if (tail >= 2)
            return elementwise_into_<value_type_, max_rank_>(collapse_contiguous_tail_(input, tail),
                                                             collapse_contiguous_tail_(output, tail),
                                                             std::forward<leaf_fn_>(leaf));
        for (std::size_t i = 0; i < input.extent(0); ++i) {
            auto idx = static_cast<std::ptrdiff_t>(i);
            if (!elementwise_into_<value_type_, max_rank_>(input.slice_leading(idx), output.slice_leading(idx), leaf))
                return false;
        }
        return true;
    }
    if (!can_apply_rank1_data_kernel_(input) || !can_apply_rank1_data_kernel_(output)) return false;
    leaf(input, output);
    return true;
}

/** @brief Binary elementwise traversal: validates shapes, recurses on rank≥2, calls leaf on rank-1 slices. */
template <typename value_type_, std::size_t max_rank_, typename leaf_fn_>
bool elementwise_into_(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
                       tensor_span<value_type_, max_rank_> output, leaf_fn_ &&leaf) noexcept {
    if (!shapes_match_(lhs, rhs) || !shapes_match_out_(lhs, output) || !tensor_layout_supported_(lhs) ||
        !tensor_layout_supported_(rhs) || !tensor_layout_supported_(output))
        return false;
    if (lhs.empty()) return true;
    if (lhs.rank() >= 2) {
        auto tail = shared_contiguous_tail_dims_<value_type_, max_rank_>(
            lhs.rank(), lhs.shape().extents, {lhs.shape().strides, rhs.shape().strides, output.shape().strides});
        if (tail >= 2)
            return elementwise_into_<value_type_, max_rank_>(
                collapse_contiguous_tail_(lhs, tail), collapse_contiguous_tail_(rhs, tail),
                collapse_contiguous_tail_(output, tail), std::forward<leaf_fn_>(leaf));
        for (std::size_t i = 0; i < lhs.extent(0); ++i) {
            auto idx = static_cast<std::ptrdiff_t>(i);
            if (!elementwise_into_<value_type_, max_rank_>(lhs.slice_leading(idx), rhs.slice_leading(idx),
                                                           output.slice_leading(idx), leaf))
                return false;
        }
        return true;
    }
    if (!can_apply_rank1_data_kernel_(lhs) || !can_apply_rank1_data_kernel_(rhs) ||
        !can_apply_rank1_data_kernel_(output))
        return false;
    leaf(lhs, rhs, output);
    return true;
}

/** @brief Ternary elementwise traversal: validates shapes, recurses on rank≥2, calls leaf on rank-1 slices. */
template <typename value_type_, std::size_t max_rank_, typename leaf_fn_>
bool elementwise_into_(tensor_view<value_type_, max_rank_> a, tensor_view<value_type_, max_rank_> b,
                       tensor_view<value_type_, max_rank_> c, tensor_span<value_type_, max_rank_> output,
                       leaf_fn_ &&leaf) noexcept {
    if (!shapes_match_(a, b) || !shapes_match_(a, c) || !shapes_match_out_(a, output) || !tensor_layout_supported_(a) ||
        !tensor_layout_supported_(b) || !tensor_layout_supported_(c) || !tensor_layout_supported_(output))
        return false;
    if (a.empty()) return true;
    if (a.rank() >= 2) {
        auto tail = shared_contiguous_tail_dims_<value_type_, max_rank_>(
            a.rank(), a.shape().extents,
            {a.shape().strides, b.shape().strides, c.shape().strides, output.shape().strides});
        if (tail >= 2)
            return elementwise_into_<value_type_, max_rank_>(
                collapse_contiguous_tail_(a, tail), collapse_contiguous_tail_(b, tail),
                collapse_contiguous_tail_(c, tail), collapse_contiguous_tail_(output, tail),
                std::forward<leaf_fn_>(leaf));
        for (std::size_t i = 0; i < a.extent(0); ++i) {
            auto idx = static_cast<std::ptrdiff_t>(i);
            if (!elementwise_into_<value_type_, max_rank_>(a.slice_leading(idx), b.slice_leading(idx),
                                                           c.slice_leading(idx), output.slice_leading(idx), leaf))
                return false;
        }
        return true;
    }
    if (!can_apply_rank1_data_kernel_(a) || !can_apply_rank1_data_kernel_(b) || !can_apply_rank1_data_kernel_(c) ||
        !can_apply_rank1_data_kernel_(output))
        return false;
    leaf(a, b, c, output);
    return true;
}

#pragma endregion Helpers

} // namespace ashvardanian::numkong

#endif // NK_TENSOR_HPP
