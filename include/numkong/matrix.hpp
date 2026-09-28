/**
 *  @file include/numkong/matrix.hpp
 *  @author Ash Vardanian
 *  @date March 5, 2026
 *  @brief NumKong packed_matrix type for efficient GEMM.
 *
 *  Provides a pre-packed matrix type over @c dots_pack and @c dots_packed for cache-efficient GEMM,
 *  plus the @c packed_maxsim and @c packed_attention sets over their own packing kernels.
 *
 *  @code{.cpp}
 *  auto [b, b_status] = nk::tensor<nk::f32_t>::zeros({256, 512});
 *  if (nk::failed(b_status)) return b_status;
 *  auto [packed, packed_status] = nk::packed_matrix<nk::f32_t>::make(b.view());
 *  if (nk::failed(packed_status)) return packed_status;
 *  // multiply many times with different A matrices
 *  @endcode
 */

#ifndef NUMKONG_MATRIX_HPP
#define NUMKONG_MATRIX_HPP

#include <cstring>
#include <type_traits>

#include "numkong/attention.h"
#include "numkong/dots.h"
#include "numkong/maxsim.h"
#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Packing Utilities

/**
 *  @brief Estimates the memory requirements for packed B matrix.
 *  @param[in] row_count Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row (k)
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @return Size in bytes for row-major B data plus stride metadata, or zero when no capability in
 *      @p capabilities packs @p in_type_
 *
 *  @tparam in_type_ Input element type
 */
template <numeric_dtype in_type_>
size_t dots_pack_size(size_t row_count, size_t depth, nk_capability_t capabilities = cpu_capabilities()) {
    nk_size_t bytes = 0;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t>)
            return nk_dots_pack_size_f64_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, f32_t>)
            return nk_dots_pack_size_f32_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, f16_t>)
            return nk_dots_pack_size_f16_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, bf16_t>)
            return nk_dots_pack_size_bf16_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, i8_t>)
            return nk_dots_pack_size_i8_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, u8_t>)
            return nk_dots_pack_size_u8_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return nk_dots_pack_size_e4m3_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, e5m2_t>)
            return nk_dots_pack_size_e5m2_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, e2m3_t>)
            return nk_dots_pack_size_e2m3_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
            return nk_dots_pack_size_e2m1_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, e3m2_t>)
            return nk_dots_pack_size_e3m2_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, u4x2_t>)
            return nk_dots_pack_size_u4_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, i4x2_t>)
            return nk_dots_pack_size_i4_best(row_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
    }
    // We need enough space for the pointer to the original B matrix and its stride
    return sizeof(void *) + sizeof(size_t);
}

/**
 *  @brief Packs matrix B into row-major form for efficient dots_packed access.
 *  @param[in] b Input matrix B, row-major, of shape @b [row_count,depth]
 *  @param[in] row_count Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row (k)
 *  @param[in] b_stride_in_bytes Stride between rows of B in bytes
 *  @param[out] b_packed Output buffer for packed row-major B with metadata
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input element type
 */
template <numeric_dtype in_type_>
status_t dots_pack(in_type_ const *b, size_t row_count, size_t depth, size_t b_stride_in_bytes, void *b_packed,
                   nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) {
    using raw_t = typename in_type_::raw_t;
    raw_t const *b_raw = reinterpret_cast<raw_t const *>(b);

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t>)
            return static_cast<status_t>(nk_dots_pack_f64_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                               row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t>)
            return static_cast<status_t>(nk_dots_pack_f32_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                               row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t>)
            return static_cast<status_t>(nk_dots_pack_f16_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                               row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_dots_pack_bf16_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                                row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t>)
            return static_cast<status_t>(nk_dots_pack_i8_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                              row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t>)
            return static_cast<status_t>(nk_dots_pack_u8_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                              row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_dots_pack_e4m3_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                                row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t>)
            return static_cast<status_t>(nk_dots_pack_e5m2_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                                row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t>)
            return static_cast<status_t>(nk_dots_pack_e2m3_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                                row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
            return static_cast<status_t>(nk_dots_pack_e2m1_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                                row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t>)
            return static_cast<status_t>(nk_dots_pack_e3m2_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                                row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t>)
            return static_cast<status_t>(nk_dots_pack_u4_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                              row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t>)
            return static_cast<status_t>(nk_dots_pack_i4_best(b_raw, row_count, depth, b_stride_in_bytes, b_packed, 0,
                                                              row_count, capabilities, stream));
    }
    // Persist the pointer to the original B matrix and its stride
    char *b_packed_bytes = reinterpret_cast<char *>(b_packed);
    std::memcpy(b_packed_bytes, &b, sizeof(void *));
    std::memcpy(b_packed_bytes + sizeof(void *), &b_stride_in_bytes, sizeof(size_t));
    return status_t::success_k;
}

/**
 *  @brief Estimates the memory requirements for a maxsim packed vector set.
 *  @param[in] vector_count Number of vectors to pack.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template.
 *  @return Size in bytes for the packed buffer, or zero when no capability in @p capabilities packs
 *      @p in_type_.
 *
 *  @tparam in_type_ Input element type (bf16_t, f32_t, f16_t).
 */
template <numeric_dtype in_type_>
std::size_t maxsim_pack_size(std::size_t vector_count, std::size_t depth,
                             nk_capability_t capabilities = cpu_capabilities()) {
    nk_size_t bytes = 0;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, bf16_t>)
            return nk_maxsim_pack_size_bf16_best(vector_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, f32_t>)
            return nk_maxsim_pack_size_f32_best(vector_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
        else if constexpr (std::is_same_v<in_type_, f16_t>)
            return nk_maxsim_pack_size_f16_best(vector_count, depth, capabilities, &bytes) == nk_success_k ? bytes : 0;
    }
    return sizeof(void *) + sizeof(std::size_t);
}

/**
 *  @brief Packs vectors into a backend-specific layout for maxsim computation.
 *  @param[in] vectors Input vectors in row-major order.
 *  @param[in] vector_count Number of vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[in] stride Row stride in bytes for the input vectors.
 *  @param[out] packed Output packed buffer from maxsim_pack_size.
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input element type (bf16_t, f32_t, f16_t).
 */
template <numeric_dtype in_type_>
status_t maxsim_pack(typename in_type_::raw_t const *vectors, std::size_t vector_count, std::size_t depth,
                     std::size_t stride, void *packed, nk_capability_t capabilities = cpu_capabilities(),
                     void *stream = nullptr) {
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(
                nk_maxsim_pack_bf16_best(vectors, vector_count, depth, stride, packed, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t>)
            return static_cast<status_t>(
                nk_maxsim_pack_f32_best(vectors, vector_count, depth, stride, packed, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t>)
            return static_cast<status_t>(
                nk_maxsim_pack_f16_best(vectors, vector_count, depth, stride, packed, capabilities, stream));
    }
    char *packed_bytes = reinterpret_cast<char *>(packed);
    std::memcpy(packed_bytes, &vectors, sizeof(void *));
    std::memcpy(packed_bytes + sizeof(void *), &stride, sizeof(std::size_t));
    return status_t::success_k;
}

/**
 *  @brief Sizes the packed KV-cache of a ragged batch of segments.
 *  @param[in] key_value_head_count Number of K/V heads, a nonzero divisor of the query head count.
 *  @param[in] depth Head dimension; any value ≥ 1.
 *  @param[in] segment_lengths Live token counts, one per segment; zeros allowed.
 *  @param[in] segment_count Number of segments packed together.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial reference.
 *  @return The size in bytes, or @c missing_kernel_k when no capability in @p capabilities packs
 *      @p in_type_.
 *
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 */
template <numeric_dtype in_type_>
expected<std::size_t> attention_pack_size(std::size_t key_value_head_count, std::size_t depth,
                                          nk_u32_t const *segment_lengths, std::size_t segment_count,
                                          nk_capability_t capabilities = cpu_capabilities()) {
    nk_size_t bytes = 0;
    nk_status_t status = nk_missing_kernel_k;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, bf16_t>)
            status = nk_attention_pack_size_bf16_best(key_value_head_count, depth, segment_lengths, segment_count,
                                                      capabilities, &bytes);
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            status = nk_attention_pack_size_e4m3_best(key_value_head_count, depth, segment_lengths, segment_count,
                                                      capabilities, &bytes);
        else if constexpr (std::is_same_v<in_type_, i8_t>)
            status = nk_attention_pack_size_i8_best(key_value_head_count, depth, segment_lengths, segment_count,
                                                    capabilities, &bytes);
    }
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_attention_pack_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count,
                                                    &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_attention_pack_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count,
                                                    &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_attention_pack_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count, &bytes);
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs ragged K/V token matrices into a backend-opaque KV-cache blob.
 *  @param[in] keys,values Token-major matrices, one row of @p key_value_head_count × @p depth
 *      elements per token.
 *  @param[in] segment_offsets Start token of each segment, @p segment_count + 1 prefix sums.
 *  @param[in] segment_lengths Live token counts, one per segment; zeros mark padding slots.
 *  @param[in] keys_stride_in_bytes Row (token) stride of @p keys in bytes.
 *  @param[in] values_stride_in_bytes Row (token) stride of @p values in bytes.
 *  @param[out] key_value_packed 64-byte-aligned buffer of @c attention_pack_size bytes.
 *  @param[in] task_begin First task of a window over the segments × K/V heads grid.
 *  @param[in] task_end End of that half-open window, so callers can shard packing across threads.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial reference.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 */
template <numeric_dtype in_type_>
status_t attention_pack(in_type_ const *keys, in_type_ const *values, std::size_t key_value_head_count,
                        std::size_t depth, nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                        std::size_t segment_count, std::size_t keys_stride_in_bytes, std::size_t values_stride_in_bytes,
                        void *key_value_packed, std::size_t task_begin = 0,
                        std::size_t task_end = static_cast<std::size_t>(-1),
                        nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) {
    using raw_t = typename in_type_::raw_t;
    raw_t const *keys_raw = reinterpret_cast<raw_t const *>(keys);
    raw_t const *values_raw = reinterpret_cast<raw_t const *>(values);
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_attention_pack_bf16_best(
                keys_raw, values_raw, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                keys_stride_in_bytes, values_stride_in_bytes, key_value_packed, task_begin, task_end, capabilities,
                stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_attention_pack_e4m3_best(
                keys_raw, values_raw, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                keys_stride_in_bytes, values_stride_in_bytes, key_value_packed, task_begin, task_end, capabilities,
                stream));
        else if constexpr (std::is_same_v<in_type_, i8_t>)
            return static_cast<status_t>(
                nk_attention_pack_i8_best(keys_raw, values_raw, key_value_head_count, depth, segment_offsets,
                                          segment_lengths, segment_count, keys_stride_in_bytes, values_stride_in_bytes,
                                          key_value_packed, task_begin, task_end, capabilities, stream));
    }
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_attention_pack_bf16_serial(
            keys_raw, values_raw, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
            keys_stride_in_bytes, values_stride_in_bytes, key_value_packed, task_begin, task_end, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_attention_pack_e4m3_serial(
            keys_raw, values_raw, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
            keys_stride_in_bytes, values_stride_in_bytes, key_value_packed, task_begin, task_end, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_attention_pack_i8_serial(
            keys_raw, values_raw, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
            keys_stride_in_bytes, values_stride_in_bytes, key_value_packed, task_begin, task_end, stream));
    else return status_t::missing_kernel_k;
}

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

#pragma endregion Packing Utilities

#pragma region Packed Containers

/**
 *  @brief Owning, move-only, pre-packed matrix for efficient GEMM.
 *  @tparam value_type_ Element type, e.g., f32_t, bf16_t.
 *  @tparam allocator_type_ Allocator for the packed buffer, default aligned_allocator<char>.
 *
 *  Wraps @c dots_pack to pre-arrange a matrix B into a cache-friendly layout.
 *  Use `make()` to create from a matrix_view, then pass to `dots_packed()` for computation.
 */
template <numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
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
     *  @brief Pack a 2D matrix_view into cache-efficient layout.
     *  @param[in] b 2D matrix view. Uses extents[0] as rows, extents[1] as depth.
     *  @param[in] alloc Allocator instance.
     *  @return The packed matrix; @c unexpected_dimensions_k for a rank below 2,
     *      @c missing_kernel_k when nothing packs @p value_type_, @c bad_alloc_k, or the failure
     *      the packing kernel itself reports.
     */
    static expected<packed_matrix> make(matrix_view<value_type_> b, allocator_type_ alloc = {}) noexcept {
        if (b.rank() < 2) return {packed_matrix(alloc), status_t::unexpected_dimensions_k};
        packed_matrix pm(alloc);
        pm.rows_ = b.extent(0);
        pm.depth_ = b.extent(1);
        pm.size_bytes_ = dots_pack_size<value_type_>(pm.rows_, pm.depth_);
        if (pm.size_bytes_ == 0) return {packed_matrix(alloc), status_t::missing_kernel_k};

        pm.data_ = alloc_traits::allocate(pm.alloc_, pm.size_bytes_);
        if (!pm.data_) return {packed_matrix(alloc), status_t::bad_alloc_k};

        if (status_t status = dots_pack<value_type_>(b.data(), pm.rows_, pm.depth_,
                                                     static_cast<size_type>(b.stride_bytes(0)), pm.data_);
            failed(status))
            return {packed_matrix(alloc), status};
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
 *  refinement. Both queries and documents must be independently packed before calling `maxsim()`.
 *
 *  Supported types: bf16_t, f32_t, f16_t.
 */
template <numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
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

    /** Pack a 2D matrix of vectors; fails like @c packed_matrix::make. */
    static expected<packed_maxsim> make(matrix_view<value_type_> vectors, allocator_type_ alloc = {}) noexcept {
        if (vectors.rank() < 2) return {packed_maxsim(alloc), status_t::unexpected_dimensions_k};
        packed_maxsim pm(alloc);
        pm.vector_count_ = vectors.extent(0);
        pm.depth_ = vectors.extent(1);
        pm.size_bytes_ = maxsim_pack_size<value_type_>(pm.vector_count_, pm.depth_);
        if (pm.size_bytes_ == 0) return {packed_maxsim(alloc), status_t::missing_kernel_k};

        pm.data_ = alloc_traits::allocate(pm.alloc_, pm.size_bytes_);
        if (!pm.data_) return {packed_maxsim(alloc), status_t::bad_alloc_k};

        if (status_t status = maxsim_pack<value_type_>(
                reinterpret_cast<typename value_type_::raw_t const *>(vectors.data()), pm.vector_count_, pm.depth_,
                static_cast<std::size_t>(vectors.stride_bytes(0)), pm.data_);
            failed(status))
            return {packed_maxsim(alloc), status};
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
 *  overloads of `attention_bidirectional_packed()` and `attention_causal_packed()` run
 *  self-attention against it without restating the batch.
 *
 *  Supported types: bf16_t, e4m3_t, i8_t.
 */
template <numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
class packed_attention {
    using alloc_traits = std::allocator_traits<allocator_type_>;

    char *data_ = nullptr;
    std::size_t size_bytes_ = 0;
    std::size_t key_value_head_count_ = 0;
    std::size_t depth_ = 0;
    std::size_t segment_count_ = 0;
    [[no_unique_address]] allocator_type_ alloc_;

    std::size_t allocated_bytes_() const noexcept { return size_bytes_ + (segment_count_ + 1) * sizeof(nk_u32_t); }

  public:
    packed_attention() noexcept = default;
    explicit packed_attention(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    ~packed_attention() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, allocated_bytes_());
    }

    packed_attention(packed_attention &&o) noexcept
        : data_(std::exchange(o.data_, nullptr)), size_bytes_(std::exchange(o.size_bytes_, 0)),
          key_value_head_count_(std::exchange(o.key_value_head_count_, 0)), depth_(std::exchange(o.depth_, 0)),
          segment_count_(std::exchange(o.segment_count_, 0)), alloc_(std::move(o.alloc_)) {}

    packed_attention &operator=(packed_attention &&o) noexcept {
        if (this != &o) {
            if (data_) alloc_traits::deallocate(alloc_, data_, allocated_bytes_());
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
     *  @param[in] keys,values @b [tokens,key_value_heads,depth] views with contiguous heads.
     *  @param[in] segment_offsets Start token of each segment, holding one more entry than
     *      @p segment_lengths.
     *  @param[in] segment_lengths Live token counts, one per segment; zeros mark padding slots.
     *  @param[in] alloc Allocator instance.
     *  @param[in] capabilities Capabilities to pack for; attend with the same ones.
     *  @return The pack; @c unexpected_dimensions_k when the views or segments disagree or overrun
     *      the tokens, @c missing_kernel_k, @c bad_alloc_k, or the packing kernel's failure.
     */
    template <std::size_t max_rank_>
    static expected<packed_attention> make(tensor_view<value_type_, max_rank_> keys,
                                           tensor_view<value_type_, max_rank_> values,
                                           vector_view<nk_u32_t> segment_offsets, vector_view<nk_u32_t> segment_lengths,
                                           allocator_type_ alloc = {},
                                           nk_capability_t capabilities = cpu_capabilities()) noexcept {
        std::size_t const segment_count = segment_lengths.size();
        bool shaped = attention_rows_supported_(keys) && attention_rows_supported_(values) &&
                      segment_offsets.size() == segment_count + 1 && segment_offsets.is_contiguous() &&
                      segment_lengths.is_contiguous();
        for (std::size_t axis = 0; shaped && axis < 3; ++axis) shaped = keys.extent(axis) == values.extent(axis);
        for (std::size_t segment = 0; shaped && segment < segment_count; ++segment)
            shaped = std::size_t {segment_offsets[segment]} + segment_lengths[segment] <= keys.extent(0);
        if (!shaped) return {packed_attention(alloc), status_t::unexpected_dimensions_k};

        packed_attention pa(alloc);
        pa.key_value_head_count_ = keys.extent(1);
        pa.depth_ = keys.extent(2);
        auto size = attention_pack_size<value_type_>(pa.key_value_head_count_, pa.depth_, segment_lengths.data(),
                                                     segment_count, capabilities);
        if (!size) return {packed_attention(alloc), size.status};
        pa.size_bytes_ = size.value;
        pa.segment_count_ = segment_count;
        pa.data_ = alloc_traits::allocate(pa.alloc_, pa.allocated_bytes_());
        if (!pa.data_) return {packed_attention(alloc), status_t::bad_alloc_k};

        if (status_t status = attention_pack<value_type_>(keys.data(), values.data(), pa.key_value_head_count_,
                                                          pa.depth_, segment_offsets.data(), segment_lengths.data(),
                                                          segment_count, static_cast<std::size_t>(keys.stride_bytes(0)),
                                                          static_cast<std::size_t>(values.stride_bytes(0)), pa.data_, 0,
                                                          static_cast<std::size_t>(-1), capabilities);
            failed(status))
            return {packed_attention(alloc), status};
        std::memcpy(pa.data_ + pa.size_bytes_, segment_offsets.data(), (segment_count + 1) * sizeof(nk_u32_t));
        return {std::move(pa), status_t::success_k};
    }

    std::size_t key_value_head_count() const noexcept { return key_value_head_count_; }
    std::size_t depth() const noexcept { return depth_; }
    std::size_t segment_count() const noexcept { return segment_count_; }
    bool empty() const noexcept { return data_ == nullptr; }
    void const *data() const noexcept { return data_; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }

    /** The pack-time segment offsets, which the view overloads take as the query offsets. */
    vector_view<nk_u32_t> segment_offsets() const noexcept {
        if (!data_) return {};
        return {reinterpret_cast<nk_u32_t const *>(data_ + size_bytes_), segment_count_ + 1};
    }
};

#pragma endregion Packed Containers

} // namespace ashvardanian::numkong

#endif // NUMKONG_MATRIX_HPP
