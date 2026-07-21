/**
 *  @brief NumKong packed_matrix type for efficient GEMM.
 *  @file include/numkong/matrix.hpp
 *  @author Ash Vardanian
 *  @date March 2026
 *
 *  Provides a pre-packed matrix type that wraps `dots_pack` / `dots_packed` for
 *  cache-efficient matrix multiplication.
 *
 *  @code
 *  auto b = nk::tensor<nk::f32_t>::try_zeros({256, 512});
 *  auto packed = nk::packed_matrix<nk::f32_t>::try_pack(b.view());
 *  // multiply many times with different A matrices
 *  @endcode
 */

#ifndef NK_MATRIX_HPP
#define NK_MATRIX_HPP

#include <cstring>
#include <type_traits>

#include "numkong/dots.h"
#include "numkong/maxsim.h"
#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Packing Utilities

/**
 *  @brief Estimates the memory requirements for packed B matrix.
 *  @param[in] row_count Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row (k)
 *  @return Size in bytes for row-major B data plus stride metadata
 *
 *  @tparam in_type_ Input element type
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_API_COMPTIME size_t dots_pack_size(size_t row_count, size_t depth) {
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) return nk_dots_pack_size_f64(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) return nk_dots_pack_size_f32(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) return nk_dots_pack_size_f16(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) return nk_dots_pack_size_bf16(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) return nk_dots_pack_size_i8(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) return nk_dots_pack_size_u8(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) return nk_dots_pack_size_e4m3(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) return nk_dots_pack_size_e5m2(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd) return nk_dots_pack_size_e2m3(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd) return nk_dots_pack_size_e3m2(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) return nk_dots_pack_size_u4(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) return nk_dots_pack_size_i4(row_count, depth);
    else {
        // We need enough space for the pointer to the original B matrix and its stride
        return sizeof(void *) + sizeof(size_t);
    }
}

/**
 *  @brief Packs matrix B into row-major form for efficient dots_packed access.
 *  @param[in] b Input matrix B in row-major form [row_count x depth]
 *  @param[in] row_count Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row (k)
 *  @param[in] b_stride_in_bytes Stride between rows of B in bytes
 *  @param[out] b_packed Output buffer for packed row-major B with metadata
 *
 *  @tparam in_type_ Input element type
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_API_COMPTIME void dots_pack(in_type_ const *b, size_t row_count, size_t depth, size_t b_stride_in_bytes,
                               void *b_packed) {
    using raw_t = typename in_type_::raw_t;
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_dots_pack_f64(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                         row_count);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_dots_pack_f32(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                         row_count);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_dots_pack_f16(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                         row_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_dots_pack_bf16(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                          row_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_dots_pack_i8(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                        row_count);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_dots_pack_u8(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                        row_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_dots_pack_e4m3(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                          row_count);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_dots_pack_e5m2(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                          row_count);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd)
        nk_dots_pack_e2m3(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                          row_count);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd)
        nk_dots_pack_e3m2(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                          row_count);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd)
        nk_dots_pack_u4(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                        row_count);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd)
        nk_dots_pack_i4(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed, 0,
                        row_count);
    else {
        // Persist the pointer to the original B matrix and its stride
        char *b_packed_bytes = reinterpret_cast<char *>(b_packed);
        std::memcpy(b_packed_bytes, &b, sizeof(void *));
        std::memcpy(b_packed_bytes + sizeof(void *), &b_stride_in_bytes, sizeof(size_t));
    }
}

/**
 *  @brief Estimates the memory requirements for a maxsim packed vector set.
 *  @param[in] vector_count Number of vectors to pack.
 *  @param[in] depth Number of dimensions per vector.
 *  @return Size in bytes for the packed buffer.
 *
 *  @tparam in_type_ Input element type (bf16_t, f32_t, f16_t).
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`.
 */
template <numeric_dtype in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_API_COMPTIME std::size_t maxsim_pack_size(std::size_t vector_count, std::size_t depth) {
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, bf16_t> && simd) return nk_maxsim_pack_size_bf16(vector_count, depth);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) return nk_maxsim_pack_size_f32(vector_count, depth);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) return nk_maxsim_pack_size_f16(vector_count, depth);
    else return sizeof(void *) + sizeof(std::size_t);
}

/**
 *  @brief Packs vectors into a backend-specific layout for maxsim computation.
 *  @param[in] vectors Input vectors in row-major order.
 *  @param[in] vector_count Number of vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[in] stride Row stride in bytes for the input vectors.
 *  @param[out] packed Output packed buffer from maxsim_pack_size.
 *
 *  @tparam in_type_ Input element type (bf16_t, f32_t, f16_t).
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`.
 */
template <numeric_dtype in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_API_COMPTIME void maxsim_pack(typename in_type_::raw_t const *vectors, std::size_t vector_count, std::size_t depth,
                                 std::size_t stride, void *packed) {
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_maxsim_pack_bf16(vectors, vector_count, depth, stride, packed);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_maxsim_pack_f32(vectors, vector_count, depth, stride, packed);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_maxsim_pack_f16(vectors, vector_count, depth, stride, packed);
    else {
        char *packed_bytes = reinterpret_cast<char *>(packed);
        std::memcpy(packed_bytes, &vectors, sizeof(void *));
        std::memcpy(packed_bytes + sizeof(void *), &stride, sizeof(std::size_t));
    }
}

#pragma endregion Packing Utilities

#pragma region Packed Containers

/**
 *  @brief Owning, move-only, pre-packed matrix for efficient GEMM.
 *  @tparam value_type_ Element type (e.g., f32_t, bf16_t).
 *  @tparam allocator_type_ Allocator for the packed buffer (default: aligned_allocator<char>).
 *
 *  Wraps `dots_pack` to pre-arrange a matrix B into a cache-friendly layout.
 *  Use `try_pack()` to create from a matrix_view, then pass to `dots_packed()` for computation.
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
     *  @param b 2D matrix view. Uses extents[0] as rows, extents[1] as depth.
     *  @param alloc Allocator instance.
     *  @return Non-empty packed_matrix on success, empty on failure.
     */
    [[nodiscard]] static packed_matrix try_pack(matrix_view<value_type_> b, allocator_type_ alloc = {}) noexcept {
        packed_matrix pm(alloc);
        if (b.rank() < 2) return pm;

        pm.rows_ = b.extent(0);
        pm.depth_ = b.extent(1);
        pm.size_bytes_ = dots_pack_size<value_type_>(pm.rows_, pm.depth_);
        if (pm.size_bytes_ == 0) return pm;

        pm.data_ = alloc_traits::allocate(pm.alloc_, pm.size_bytes_);
        if (!pm.data_) {
            pm.size_bytes_ = 0;
            return pm;
        }

        dots_pack<value_type_>(b.data(), pm.rows_, pm.depth_, static_cast<size_type>(b.stride_bytes(0)), pm.data_);
        return pm;
    }

    /** @brief Number of rows in the packed matrix (n). */
    constexpr size_type rows() const noexcept { return rows_; }

    /** @brief Number of columns / depth (k). */
    constexpr size_type depth() const noexcept { return depth_; }

    /** @brief Size of the packed buffer in bytes. */
    constexpr size_type size_bytes() const noexcept { return size_bytes_; }

    /** @brief True if no matrix is packed. */
    constexpr bool empty() const noexcept { return data_ == nullptr; }

    /** @brief Raw pointer to the packed data. */
    constexpr void const *data() const noexcept { return data_; }
};

/**
 *  @brief Pre-packed vector set for MaxSim (ColBERT late-interaction).
 *
 *  MaxSim computes Σᵢ minⱼ angular(qᵢ, dⱼ) using quantized i8 screening
 *  followed by full-precision refinement. Both queries and documents must
 *  be independently packed before calling `maxsim()`.
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

    /** @brief Pack a 2D matrix of vectors. Returns empty on failure. */
    [[nodiscard]] static packed_maxsim try_pack(matrix_view<value_type_> vectors, allocator_type_ alloc = {}) noexcept {
        packed_maxsim pm(alloc);
        if (vectors.rank() < 2) return pm;

        pm.vector_count_ = vectors.extent(0);
        pm.depth_ = vectors.extent(1);
        pm.size_bytes_ = maxsim_pack_size<value_type_>(pm.vector_count_, pm.depth_);
        if (pm.size_bytes_ == 0) return pm;

        pm.data_ = alloc_traits::allocate(pm.alloc_, pm.size_bytes_);
        if (!pm.data_) {
            pm.size_bytes_ = 0;
            return pm;
        }

        maxsim_pack<value_type_>(reinterpret_cast<typename value_type_::raw_t const *>(vectors.data()),
                                 pm.vector_count_, pm.depth_, static_cast<std::size_t>(vectors.stride_bytes(0)),
                                 pm.data_);
        return pm;
    }

    std::size_t vector_count() const noexcept { return vector_count_; }
    std::size_t rows() const noexcept { return vector_count_; }
    std::size_t depth() const noexcept { return depth_; }
    bool empty() const noexcept { return data_ == nullptr; }
    void const *data() const noexcept { return data_; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }
};

#pragma endregion Packed Containers

} // namespace ashvardanian::numkong

#endif // NK_MATRIX_HPP
