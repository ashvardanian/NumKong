/**
 *  @file include/numkong/matrix.hpp
 *  @author Ash Vardanian
 *  @date March 5, 2026
 *  @brief NumKong packed_matrix type for efficient GEMM.
 *
 *  Provides a pre-packed matrix type for cache-efficient GEMM, plus the @c packed_maxsim and
 *  @c packed_attention sets. Each owner packs through the size and pack callables its @c make
 *  takes, which the @c numkong::cpp verbs @c nk::dots_pack, @c nk::maxsim_pack and
 *  @c nk::attention_pack bind to the dispatching kernels.
 *
 *  @code{.cpp}
 *  auto [b, b_status] = nk::matrix<nk::f32_t>::zeros({256, 512});
 *  if (nk::failed(b_status)) return b_status;
 *  auto [packed, packed_status] = nk::dots_pack<nk::f32_t>(b.view());
 *  if (nk::failed(packed_status)) return packed_status;
 *  // multiply many times with different A matrices
 *  @endcode
 */

#ifndef NUMKONG_MATRIX_HPP
#define NUMKONG_MATRIX_HPP

#include <cstdint> // `std::uint32_t`

#include <memory>  // `std::allocator_traits`
#include <utility> // `std::exchange`

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

/** Whether @p tokens is a non-empty @b [tokens,heads,depth] view or span with contiguous heads, the
 *  one layout the attention kernels stride over. */
template <typename tokens_type_>
constexpr bool attention_rows_supported_(tokens_type_ const &tokens) noexcept {
    auto const element_bytes = static_cast<std::ptrdiff_t>(sizeof(typename tokens_type_::value_type));
    return tokens.rank() == 3 && tokens.extent(1) != 0 && tokens.extent(2) != 0 &&
           tokens.stride_bytes(2) == element_bytes &&
           tokens.stride_bytes(1) == static_cast<std::ptrdiff_t>(tokens.extent(2)) * element_bytes &&
           tokens.stride_bytes(0) >= 0;
}

#pragma region Packed Containers

/**
 *  @brief Owning, move-only, pre-packed matrix for efficient GEMM.
 *  @tparam value_type_ Element type, e.g., f32_t, bf16_t.
 *  @tparam allocator_type_ Allocator for the packed buffer, default aligned_allocator<char>.
 *
 *  Holds a matrix B pre-arranged into a cache-friendly layout. @c nk::dots_pack packs one through
 *  the dispatching kernels, and <tt>make()</tt> through any pair of callables; pass it to
 *  <tt>dots_packed()</tt> for computation.
 */
template <a_numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
struct packed_matrix {
    using value_type = value_type_;
    using result_type = typename value_type_::dot_result_t;
    using allocator_type = allocator_type_;
    using alloc_traits = std::allocator_traits<allocator_type_>;
    using size_type = std::size_t;

  private:
    char *data_ = nullptr;
    size_type size_bytes_ = 0;
    size_type rows_ = 0;  // n (number of rows in B)
    size_type depth_ = 0; // k (number of columns in B)
    [[no_unique_address]] allocator_type_ alloc_;

  public:
    packed_matrix() noexcept = default;

    explicit packed_matrix(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    ~packed_matrix() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
    }

    packed_matrix(packed_matrix &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_bytes_(std::exchange(other.size_bytes_, 0)),
          rows_(std::exchange(other.rows_, 0)), depth_(std::exchange(other.depth_, 0)),
          alloc_(std::move(other.alloc_)) {}

    packed_matrix &operator=(packed_matrix &&other) noexcept {
        if (this != &other) {
            if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(other.alloc_);
            data_ = std::exchange(other.data_, nullptr);
            size_bytes_ = std::exchange(other.size_bytes_, 0);
            rows_ = std::exchange(other.rows_, 0);
            depth_ = std::exchange(other.depth_, 0);
        }
        return *this;
    }

    packed_matrix(packed_matrix const &) = delete;
    packed_matrix &operator=(packed_matrix const &) = delete;

    /**
     *  @brief Packs a 2D matrix_view into the layout @p pack writes.
     *  @param[in] b 2D matrix view. Uses extents[0] as rows, extents[1] as depth.
     *  @param[in] alloc Allocator instance.
     *  @param[in] pack_size Measures the pack as <tt>pack_size(rows, depth, &bytes)</tt>, like
     *      one capability's @c nk_dots_pack_size_* kernel.
     *  @param[in] pack Packs as <tt>pack(b.data(), rows, depth, row_stride_bytes, packed)</tt>; pair it
     *      with @p pack_size.
     *  @return The packed matrix.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k A rank below 2, a strided row, or a negative row stride.
     *  - The failure @p pack_size or @p pack reports.
     */
    template <typename pack_size_type_, typename pack_type_>
    static expected<packed_matrix> make(matrix_view<value_type_> b, allocator_type_ alloc, pack_size_type_ pack_size,
                                        pack_type_ pack) noexcept {
        if (b.rank() < 2 || b.stride_bytes(1) != sizeof(value_type_) || b.stride_bytes(0) < 0)
            return {packed_matrix(alloc), status_t::unexpected_dimensions_k};
        packed_matrix pm(alloc);
        pm.rows_ = b.extent(0);
        pm.depth_ = b.extent(1);
        nk_size_t bytes = 0;
        if (status_t const sized = static_cast<status_t>(pack_size(pm.rows_, pm.depth_, &bytes)); failed(sized))
            return {packed_matrix(alloc), sized};
        pm.size_bytes_ = bytes;

        pm.data_ = alloc_traits::allocate(pm.alloc_, pm.size_bytes_);
        if (!pm.data_) return {packed_matrix(alloc), status_t::bad_alloc_k};

        if (status_t const packed = static_cast<status_t>(
                pack(b.data(), pm.rows_, pm.depth_, static_cast<size_type>(b.stride_bytes(0)), pm.data_));
            failed(packed))
            return {packed_matrix(alloc), packed};
        return {std::move(pm), status_t::success_k};
    }

    /** Number of rows in the packed matrix (n). */
    constexpr size_type rows() const noexcept { return rows_; }

    /** Number of columns / depth (k). */
    constexpr size_type depth() const noexcept { return depth_; }

    /** Size of the packed buffer in bytes. */
    constexpr size_type size_bytes() const noexcept { return size_bytes_; }

    /** True if no matrix is packed. */
    constexpr bool empty() const noexcept { return data_ == nullptr; }

    /** Raw pointer to the packed data. */
    constexpr void const *data() const noexcept { return data_; }
};

/**
 *  @brief Pre-packed vector set for MaxSim — ColBERT late-interaction.
 *
 *  MaxSim computes Σᵢ minⱼ angular(qᵢ, dⱼ) using quantized i8 screening followed by full-precision
 *  refinement. Both queries and documents must be independently packed before calling <tt>maxsim()</tt>.
 *
 *  Supported types: bf16_t, f32_t, f16_t.
 */
template <a_numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
class packed_maxsim {
    using alloc_traits = std::allocator_traits<allocator_type_>;

    char *data_ = nullptr;
    std::size_t size_bytes_ = 0;
    std::size_t vector_count_ = 0;
    std::size_t depth_ = 0;
    [[no_unique_address]] allocator_type_ alloc_;

  public:
    packed_maxsim() noexcept = default;
    explicit packed_maxsim(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    ~packed_maxsim() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
    }

    packed_maxsim(packed_maxsim &&o) noexcept
        : data_(std::exchange(o.data_, nullptr)), size_bytes_(std::exchange(o.size_bytes_, 0)),
          vector_count_(std::exchange(o.vector_count_, 0)), depth_(std::exchange(o.depth_, 0)),
          alloc_(std::move(o.alloc_)) {}

    packed_maxsim &operator=(packed_maxsim &&o) noexcept {
        if (this != &o) {
            if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(o.alloc_);
            data_ = std::exchange(o.data_, nullptr);
            size_bytes_ = std::exchange(o.size_bytes_, 0);
            vector_count_ = std::exchange(o.vector_count_, 0);
            depth_ = std::exchange(o.depth_, 0);
        }
        return *this;
    }

    packed_maxsim(packed_maxsim const &) = delete;
    packed_maxsim &operator=(packed_maxsim const &) = delete;

    /** Packs a 2D matrix of vectors like @c packed_matrix::make, with
     *  <tt>pack_size(vector_count, depth, &bytes)</tt> and
     *  <tt>pack(raw_values, vector_count, depth, row_stride_bytes, packed)</tt>. */
    template <typename pack_size_type_, typename pack_type_>
    static expected<packed_maxsim> make(matrix_view<value_type_> vectors, allocator_type_ alloc,
                                        pack_size_type_ pack_size, pack_type_ pack) noexcept {
        if (vectors.rank() < 2 || vectors.stride_bytes(1) != sizeof(value_type_) || vectors.stride_bytes(0) < 0)
            return {packed_maxsim(alloc), status_t::unexpected_dimensions_k};
        packed_maxsim pm(alloc);
        pm.vector_count_ = vectors.extent(0);
        pm.depth_ = vectors.extent(1);
        nk_size_t bytes = 0;
        if (status_t const sized = static_cast<status_t>(pack_size(pm.vector_count_, pm.depth_, &bytes)); failed(sized))
            return {packed_maxsim(alloc), sized};
        pm.size_bytes_ = bytes;

        pm.data_ = alloc_traits::allocate(pm.alloc_, pm.size_bytes_);
        if (!pm.data_) return {packed_maxsim(alloc), status_t::bad_alloc_k};

        if (status_t const packed = static_cast<status_t>(
                pack(reinterpret_cast<typename value_type_::raw_t const *>(vectors.data()), pm.vector_count_, pm.depth_,
                     static_cast<std::size_t>(vectors.stride_bytes(0)), pm.data_));
            failed(packed))
            return {packed_maxsim(alloc), packed};
        return {std::move(pm), status_t::success_k};
    }

    std::size_t vector_count() const noexcept { return vector_count_; }
    std::size_t rows() const noexcept { return vector_count_; }
    std::size_t depth() const noexcept { return depth_; }
    bool empty() const noexcept { return data_ == nullptr; }
    void const *data() const noexcept { return data_; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }
};

/**
 *  @brief Pre-packed ragged KV-cache for scaled-dot-product attention.
 *
 *  Packs K and V once, keeping a copy of the segment offsets after the packed bytes, so the view
 *  overloads of <tt>attention_packed()</tt> run self-attention against it without restating the batch.
 *
 *  Supported types: bf16_t, e4m3_t, i8_t.
 */
template <a_numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
class packed_attention {
    using alloc_traits = std::allocator_traits<allocator_type_>;

    char *data_ = nullptr;
    std::size_t size_bytes_ = 0;
    std::size_t key_value_head_count_ = 0;
    std::size_t depth_ = 0;
    std::size_t segment_count_ = 0;
    [[no_unique_address]] allocator_type_ alloc_;

  public:
    packed_attention() noexcept = default;
    explicit packed_attention(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    ~packed_attention() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
    }

    packed_attention(packed_attention &&o) noexcept
        : data_(std::exchange(o.data_, nullptr)), size_bytes_(std::exchange(o.size_bytes_, 0)),
          key_value_head_count_(std::exchange(o.key_value_head_count_, 0)), depth_(std::exchange(o.depth_, 0)),
          segment_count_(std::exchange(o.segment_count_, 0)), alloc_(std::move(o.alloc_)) {}

    packed_attention &operator=(packed_attention &&o) noexcept {
        if (this != &o) {
            if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(o.alloc_);
            data_ = std::exchange(o.data_, nullptr);
            size_bytes_ = std::exchange(o.size_bytes_, 0);
            key_value_head_count_ = std::exchange(o.key_value_head_count_, 0);
            depth_ = std::exchange(o.depth_, 0);
            segment_count_ = std::exchange(o.segment_count_, 0);
        }
        return *this;
    }

    packed_attention(packed_attention const &) = delete;
    packed_attention &operator=(packed_attention const &) = delete;

    /**
     *  @brief Size, allocate and pack a ragged batch of K and V tokens.
     *  @param[in] keys @b [tokens,key_value_heads,depth] view whose heads sit back to back along
     *      its depth stride, so a transposed depth packs without a copy.
     *  @param[in] values @b [tokens,key_value_heads,depth] view laid out like @p keys.
     *  @param[in] key_offsets The slot boundaries of the segments, one more than there are
     *      segments, which must not decrease.
     *  @param[in] key_lengths Keys each segment holds, at most its slot's width; zeros mark padding
     *      slots. Empty means every slot is full.
     *  @param[in] alloc Allocator instance.
     *  @param[in] pack_size Measures the pack as
     *      <tt>pack_size(key_value_heads, depth, token_count, segment_count, &bytes)</tt>, like
     *      one capability's @c nk_attention_pack_size_* kernel.
     *  @param[in] pack Packs the operands, then the pack, in the order of the C pack kernels.
     *  @return The pack.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k The views or segments disagree or overrun the tokens.
     *  - The failure @p pack_size or @p pack reports.
     */
    template <std::size_t max_rank_, typename pack_size_type_, typename pack_type_>
    static expected<packed_attention> make(tensor_view<value_type_, max_rank_> keys,
                                           tensor_view<value_type_, max_rank_> values,
                                           vector_view<std::uint32_t> key_offsets,
                                           vector_view<std::uint32_t> key_lengths, allocator_type_ alloc,
                                           pack_size_type_ pack_size, pack_type_ pack) noexcept {
        bool const has_lengths = key_lengths.data() != nullptr;
        std::size_t const segment_count = key_offsets.size() ? key_offsets.size() - 1 : 0;
        bool shaped = attention_rows_supported_(keys) && attention_rows_supported_(values) &&
                      key_offsets.size() == segment_count + 1 && key_offsets.is_contiguous() &&
                      (!has_lengths || (key_lengths.size() == segment_count && key_lengths.is_contiguous()));
        for (std::size_t axis = 0; shaped && axis < 3; ++axis) shaped = keys.extent(axis) == values.extent(axis);
        std::size_t token_count = 0;
        for (std::size_t segment = 0; shaped && segment < segment_count; ++segment) {
            std::size_t const first = key_offsets[segment], slot_end = key_offsets[segment + 1];
            shaped = first <= slot_end;
            std::size_t const count = shaped ? (has_lengths ? key_lengths[segment] : slot_end - first) : 0;
            shaped = shaped && count <= slot_end - first && first + count <= keys.extent(0);
            token_count += count;
        }
        if (!shaped) return {packed_attention(alloc), status_t::unexpected_dimensions_k};

        packed_attention pa(alloc);
        pa.key_value_head_count_ = keys.extent(1);
        pa.depth_ = keys.extent(2);
        nk_size_t bytes = 0;
        if (status_t const sized = static_cast<status_t>(
                pack_size(pa.key_value_head_count_, pa.depth_, token_count, segment_count, &bytes));
            failed(sized))
            return {packed_attention(alloc), sized};
        pa.size_bytes_ = bytes;
        pa.segment_count_ = segment_count;
        pa.data_ = alloc_traits::allocate(pa.alloc_, pa.size_bytes_);
        if (!pa.data_) return {packed_attention(alloc), status_t::bad_alloc_k};

        if (status_t const packed = static_cast<status_t>(
                pack(pa.key_value_head_count_, pa.depth_, key_offsets.data(), key_lengths.data(), segment_count,
                     keys.data(), static_cast<std::size_t>(keys.stride_bytes(0)), values.data(),
                     static_cast<std::size_t>(values.stride_bytes(0)), pa.data_));
            failed(packed))
            return {packed_attention(alloc), packed};
        return {std::move(pa), status_t::success_k};
    }

    std::size_t key_value_head_count() const noexcept { return key_value_head_count_; }
    std::size_t depth() const noexcept { return depth_; }
    std::size_t segment_count() const noexcept { return segment_count_; }
    bool empty() const noexcept { return data_ == nullptr; }
    void const *data() const noexcept { return data_; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }

    /** Read through @c nk::attention_key_offsets, a library export. */
    vector_view<std::uint32_t> key_offsets() const = delete;

    /** Read through @c nk::attention_key_lengths. */
    vector_view<std::uint32_t> key_lengths() const = delete;
};

#pragma endregion Packed Containers

} // namespace ashvardanian::numkong

#endif // NUMKONG_MATRIX_HPP
