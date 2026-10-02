/**
 *  @file include/numkong/dots.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for multi-target dot-product kernels.
 */
#ifndef NUMKONG_DOTS_HPP
#define NUMKONG_DOTS_HPP

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "numkong/dot.h"
#include "numkong/dots.h"
#include "numkong/sets.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Reference unpacked GEMM: C = A × Bᵀ (row-major A and B, B transposed).
 *
 *  This matches BLAS sgemm/dgemm with CblasNoTrans for A and CblasTrans for B.
 *  Useful as a reference implementation for validating BLAS/MKL/Accelerate.
 *
 *  @param[in] a Row-major matrix A of shape @b [rows,depth]
 *  @param[in] b Row-major matrix B of shape @b [columns,depth], accessed as Bᵀ
 *  @param[out] c Row-major output matrix C of shape @b [rows,columns]
 *  @param[in] row_count Rows of A and C (m)
 *  @param[in] column_count Rows of B and columns of C (n)
 *  @param[in] depth Columns of A and B (k). Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between rows of A in bytes
 *  @param[in] b_stride_in_bytes Stride between rows of B in bytes
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @tparam in_type_ Input element type (e.g., f32_t, bf16_t)
 *  @tparam result_type_ Accumulator/output type (e.g., f32_t, f118_t for high precision)
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t>
void dots_unpacked(in_type_ const *a, in_type_ const *b, result_type_ *c, std::size_t row_count,
                   std::size_t column_count, std::size_t depth, std::size_t a_stride_in_bytes,
                   std::size_t b_stride_in_bytes, std::size_t c_stride_in_bytes) noexcept {
    char const *a_bytes = reinterpret_cast<char const *>(a);
    char const *b_bytes = reinterpret_cast<char const *>(b);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t const depth_values = depth / dimensions_per_value<in_type_>();

    for (std::size_t i = 0; i < row_count; i++) {
        in_type_ const *a_row = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
        for (std::size_t j = 0; j < column_count; j++) {
            in_type_ const *b_row = reinterpret_cast<in_type_ const *>(b_bytes + j * b_stride_in_bytes);
            result_type_ sum {};
            for (std::size_t l = 0; l < depth_values; l++) sum = fma(a_row[l], b_row[l], sum);
            c_row[j] = sum;
        }
    }
}

/**
 *  @brief Conjugated unpacked dot products: C = A × Bᴴ (Hermitian inner product, row-major)
 *
 *  Same as @c dots_unpacked, but conjugates elements of B before multiplication. For real types
 *  this is identical to @c dots_unpacked. For complex types this computes the standard Hermitian
 *  inner product matching `cblas_{c,z}gemm` with @c CblasConjTrans.
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t>
void dots_unpacked_conjugated(in_type_ const *a, in_type_ const *b, result_type_ *c, std::size_t row_count,
                              std::size_t column_count, std::size_t depth, std::size_t a_stride_in_bytes,
                              std::size_t b_stride_in_bytes, std::size_t c_stride_in_bytes) noexcept {
    char const *a_bytes = reinterpret_cast<char const *>(a);
    char const *b_bytes = reinterpret_cast<char const *>(b);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t const depth_values = depth / dimensions_per_value<in_type_>();

    for (std::size_t i = 0; i < row_count; i++) {
        in_type_ const *a_row = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
        for (std::size_t j = 0; j < column_count; j++) {
            in_type_ const *b_row = reinterpret_cast<in_type_ const *>(b_bytes + j * b_stride_in_bytes);
            result_type_ sum {};
            for (std::size_t l = 0; l < depth_values; l++) sum = fcma(b_row[l], a_row[l], sum);
            c_row[j] = sum;
        }
    }
}

/**
 *  @brief Packed dot products (batch matrix multiply): C = A × B (row-major)
 *  @param[in] a Matrix A of shape @b [rows,depth]
 *  @param[in] b_packed Packed matrix B of shape @b [depth,columns], with stride metadata appended
 *  @param[out] c Output matrix C of shape @b [rows,columns]
 *  @param[in] row_count Rows of A and C
 *  @param[in] column_count Columns of B and C
 *  @param[in] depth Columns of A, rows of B, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between rows of A in bytes
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template, which reads
 *      packs of the same zero mask
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Accumulator/output type, defaults to @c in_type_::dot_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t>
status_t dots_packed(in_type_ const *a, void const *b_packed, result_type_ *c, std::size_t row_count,
                     std::size_t column_count, std::size_t depth, std::size_t a_stride_in_bytes,
                     std::size_t c_stride_in_bytes, nk_capability_t capabilities = default_capabilities(),
                     void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::dot_result_t>;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_f64_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                 column_count, depth, a_stride_in_bytes, 0,
                                                                 c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_f32_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                 column_count, depth, a_stride_in_bytes, 0,
                                                                 c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_f16_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                 column_count, depth, a_stride_in_bytes, 0,
                                                                 c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_bf16_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                  column_count, depth, a_stride_in_bytes, 0,
                                                                  c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_i8_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                column_count, depth, a_stride_in_bytes, 0,
                                                                c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_u8_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                column_count, depth, a_stride_in_bytes, 0,
                                                                c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_e4m3_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                  column_count, depth, a_stride_in_bytes, 0,
                                                                  c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_e5m2_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                  column_count, depth, a_stride_in_bytes, 0,
                                                                  c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_e2m3_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                  column_count, depth, a_stride_in_bytes, 0,
                                                                  c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_e2m1_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                  column_count, depth, a_stride_in_bytes, 0,
                                                                  c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_e3m2_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                  column_count, depth, a_stride_in_bytes, 0,
                                                                  c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_u4_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                column_count, depth, a_stride_in_bytes, 0,
                                                                c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_dots_packed_i4_best(&a->raw_, NUMKONG_NULL, b_packed, &c->raw_, row_count,
                                                                column_count, depth, a_stride_in_bytes, 0,
                                                                c_stride_in_bytes, capabilities, stream));
    }
    in_type_ const *b;
    std::size_t b_stride_in_bytes;
    char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
    std::memcpy(&b, b_packed_bytes, sizeof(void *));
    std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(std::size_t));
    dots_unpacked<in_type_, result_type_>(a, b, c, row_count, column_count, depth, a_stride_in_bytes, b_stride_in_bytes,
                                          c_stride_in_bytes);
    return status_t::success_k;
}

/**
 *  @brief Symmetric dot products: C = A × Aᵀ where C[i,j] = ⟨A[i], A[j]⟩
 *  @param[in] a Matrix A of shape @b [vectors,depth]
 *  @param[in] vectors_count Number of vectors
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between vectors in A
 *  @param[out] c Output matrix C of shape @b [vectors,vectors]
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] row_start First row of C to compute, 0 by default
 *  @param[in] row_count Rows of C to compute, all by default
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Accumulator/output type, defaults to @c in_type_::dot_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t>
status_t dots_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth, std::size_t a_stride_in_bytes,
                        result_type_ *c, std::size_t c_stride_in_bytes, std::size_t row_start = 0,
                        std::size_t row_count = std::numeric_limits<std::size_t>::max(),
                        nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_f64_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                    a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                    row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_f32_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                    a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                    row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_f16_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                    a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                    row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_bf16_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                     a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                     row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_i8_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                   a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                   row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_u8_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                   a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                   row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_e4m3_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                     a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                     row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_e5m2_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                     a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                     row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_e2m3_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                     a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                     row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_e2m1_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                     a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                     row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_e3m2_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                     a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                     row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_u4_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                   a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                   row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_dots_symmetric_i4_best(&a->raw_, NUMKONG_NULL, vectors_count, depth,
                                                                   a_stride_in_bytes, 0, &c->raw_, c_stride_in_bytes,
                                                                   row_start, row_count, capabilities, stream));
    }
    std::size_t depth_values = depth / dimensions_per_value<in_type_>();
    char const *a_bytes = reinterpret_cast<char const *>(a);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;

    for (std::size_t i = row_start; i < row_end; i++) {
        in_type_ const *a_i = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
        for (std::size_t j = 0; j < vectors_count; j++) {
            in_type_ const *a_j = reinterpret_cast<in_type_ const *>(a_bytes + j * a_stride_in_bytes);
            result_type_ sum {};
            for (std::size_t l = 0; l < depth_values; l++) sum = fma(a_i[l], a_j[l], sum);
            c_row[j] = sum;
        }
    }
    return status_t::success_k;
}

/**
 *  @brief Symmetric Hamming distance matrix: C[i,j] = hamming(A[i], A[j])
 *  @param[in] a Input matrix, shape @b [vectors_count,depth]
 *  @param[in] vectors_count Number of vectors
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Row stride in bytes
 *  @param[out] c Output matrix, shape @b [vectors_count,vectors_count]
 *  @param[in] c_stride_in_bytes Output row stride in bytes
 *  @param[in] row_start Starting row index (default 0)
 *  @param[in] row_count Number of rows to compute (default all)
 *
 *  Computes Hamming distances between all pairs of binary vectors. For @c u1x8_t inputs, distances
 *  are exact bit counts, returned as @c u32_t.
 *
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input element type (u1x8_t)
 *  @tparam result_type_ Output type (u32_t for Hamming distances)
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::hamming_result_t>
status_t hammings_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth,
                            std::size_t a_stride_in_bytes, result_type_ *c, std::size_t c_stride_in_bytes,
                            std::size_t row_start = 0, std::size_t row_count = std::numeric_limits<std::size_t>::max(),
                            nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::hamming_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
            return static_cast<status_t>(nk_hammings_symmetric_u1_best(&a->raw_, vectors_count, depth,
                                                                       a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                       row_start, row_count, capabilities, stream));
    }
    using raw_t = typename in_type_::raw_t;
    std::size_t depth_bytes = depth / dimensions_per_value<in_type_>();
    char const *a_bytes = reinterpret_cast<char const *>(a);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;

    for (std::size_t i = row_start; i < row_end; i++) {
        raw_t const *a_i = reinterpret_cast<raw_t const *>(a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);

        for (std::size_t j = 0; j < vectors_count; j++) {
            raw_t const *a_j = reinterpret_cast<raw_t const *>(a_bytes + j * a_stride_in_bytes);
            typename result_type_::raw_t distance = 0;
            for (std::size_t b = 0; b < depth_bytes; b++) {
                auto xor_val = a_i[b] ^ a_j[b];
                distance += std::popcount(static_cast<unsigned>(xor_val));
            }
            c_row[j] = result_type_::from_raw(distance);
        }
    }
    return status_t::success_k;
}

/**
 *  @brief Computes Hamming distances between rows of A and columns of packed B.
 *  @param[in] a Pointer to the first matrix (m x k).
 *  @param[in] b_packed Pointer to the packed second matrix (k x n).
 *  @param[out] c Pointer to the output matrix (m x n).
 *  @param[in] row_count Number of rows in A (m).
 *  @param[in] column_count Number of columns in B (n).
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between consecutive rows of A in bytes.
 *  @param[in] c_stride_in_bytes Stride between consecutive rows of C in bytes.
 *
 *  Computes Hamming distances between binary vectors using optimized packed format. For @c u1x8_t
 *  inputs, distances are exact bit counts, returned as @c u32_t.
 *
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input element type (u1x8_t)
 *  @tparam result_type_ Output type (u32_t for Hamming distances)
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::hamming_result_t>
status_t hammings_packed(in_type_ const *a, void const *b_packed, result_type_ *c, std::size_t row_count,
                         std::size_t column_count, std::size_t depth, std::size_t a_stride_in_bytes = 0,
                         std::size_t c_stride_in_bytes = 0, nk_capability_t capabilities = default_capabilities(),
                         void *stream = nullptr) noexcept {
    // Compute default strides
    if (!a_stride_in_bytes) a_stride_in_bytes = depth / dimensions_per_value<in_type_>() * sizeof(in_type_);
    if (!c_stride_in_bytes) c_stride_in_bytes = column_count * sizeof(result_type_);

    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::hamming_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
            return static_cast<status_t>(nk_hammings_packed_u1_best(
                reinterpret_cast<nk_u1x8_t const *>(a), b_packed, reinterpret_cast<nk_u32_t *>(c), row_count,
                column_count, depth, a_stride_in_bytes, c_stride_in_bytes, capabilities, stream));
    }
    // Scalar fallback: extract pointer and stride from b_packed, then compute directly
    in_type_ const *b;
    std::size_t b_stride_in_bytes;
    char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
    std::memcpy(&b, b_packed_bytes, sizeof(void *));
    std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(std::size_t));

    // Compute Hamming distances using unpacked matrices
    char const *a_bytes = reinterpret_cast<char const *>(a);
    char const *b_bytes = reinterpret_cast<char const *>(b);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t depth_bytes = depth / dimensions_per_value<in_type_>();

    for (std::size_t i = 0; i < row_count; i++) {
        typename in_type_::raw_t const *a_row = reinterpret_cast<typename in_type_::raw_t const *>(
            a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);

        for (std::size_t j = 0; j < column_count; j++) {
            typename in_type_::raw_t const *b_row = reinterpret_cast<typename in_type_::raw_t const *>(
                b_bytes + j * b_stride_in_bytes);

            // Compute Hamming distance: XOR then popcount
            typename result_type_::raw_t distance = 0;
            for (std::size_t byte_idx = 0; byte_idx < depth_bytes; byte_idx++) {
                auto xor_val = a_row[byte_idx] ^ b_row[byte_idx];
                distance += std::popcount(static_cast<unsigned>(xor_val));
            }
            c_row[j] = result_type_::from_raw(distance);
        }
    }
    return status_t::success_k;
}

/** Symmetric Jaccard distance matrix: C[i,j] = jaccard(A[i], A[j]). */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::jaccard_result_t>
status_t jaccards_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth,
                            std::size_t a_stride_in_bytes, result_type_ *c, std::size_t c_stride_in_bytes,
                            std::size_t row_start = 0, std::size_t row_count = std::numeric_limits<std::size_t>::max(),
                            nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::jaccard_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
            return static_cast<status_t>(nk_jaccards_symmetric_u1_best(&a->raw_, vectors_count, depth,
                                                                       a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                       row_start, row_count, capabilities, stream));
    }
    using raw_t = typename in_type_::raw_t;
    std::size_t depth_bytes = depth / dimensions_per_value<in_type_>();
    char const *a_bytes = reinterpret_cast<char const *>(a);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;

    for (std::size_t i = row_start; i < row_end; i++) {
        raw_t const *a_i = reinterpret_cast<raw_t const *>(a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);

        for (std::size_t j = 0; j < vectors_count; j++) {
            raw_t const *a_j = reinterpret_cast<raw_t const *>(a_bytes + j * a_stride_in_bytes);
            unsigned intersection = 0, union_ = 0;
            for (std::size_t b = 0; b < depth_bytes; b++) {
                intersection += std::popcount(static_cast<unsigned>(a_i[b] & a_j[b]));
                union_ += std::popcount(static_cast<unsigned>(a_i[b] | a_j[b]));
            }
            c_row[j] = result_type_::from_raw(
                union_ ? 1.0f - static_cast<float>(intersection) / static_cast<float>(union_) : 0.0f);
        }
    }
    return status_t::success_k;
}

/** Computes Jaccard distances between rows of A and columns of packed B. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::jaccard_result_t>
status_t jaccards_packed(in_type_ const *a, void const *b_packed, result_type_ *c, std::size_t row_count,
                         std::size_t column_count, std::size_t depth, std::size_t a_stride_in_bytes = 0,
                         std::size_t c_stride_in_bytes = 0, nk_capability_t capabilities = default_capabilities(),
                         void *stream = nullptr) noexcept {
    if (!a_stride_in_bytes) a_stride_in_bytes = depth / dimensions_per_value<in_type_>() * sizeof(in_type_);
    if (!c_stride_in_bytes) c_stride_in_bytes = column_count * sizeof(result_type_);

    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::jaccard_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
            return static_cast<status_t>(nk_jaccards_packed_u1_best(
                reinterpret_cast<nk_u1x8_t const *>(a), b_packed, reinterpret_cast<nk_f32_t *>(c), row_count,
                column_count, depth, a_stride_in_bytes, c_stride_in_bytes, capabilities, stream));
    }
    // Scalar fallback: extract pointer and stride from b_packed, then compute directly
    in_type_ const *b;
    std::size_t b_stride_in_bytes;
    char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
    std::memcpy(&b, b_packed_bytes, sizeof(void *));
    std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(std::size_t));

    char const *a_bytes = reinterpret_cast<char const *>(a);
    char const *b_bytes = reinterpret_cast<char const *>(b);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t depth_bytes = depth / dimensions_per_value<in_type_>();

    for (std::size_t i = 0; i < row_count; i++) {
        typename in_type_::raw_t const *a_row = reinterpret_cast<typename in_type_::raw_t const *>(
            a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);

        for (std::size_t j = 0; j < column_count; j++) {
            typename in_type_::raw_t const *b_row = reinterpret_cast<typename in_type_::raw_t const *>(
                b_bytes + j * b_stride_in_bytes);
            unsigned intersection = 0, union_ = 0;
            for (std::size_t byte_idx = 0; byte_idx < depth_bytes; byte_idx++) {
                intersection += std::popcount(static_cast<unsigned>(a_row[byte_idx] & b_row[byte_idx]));
                union_ += std::popcount(static_cast<unsigned>(a_row[byte_idx] | b_row[byte_idx]));
            }
            c_row[j] = result_type_::from_raw(
                union_ ? 1.0f - static_cast<float>(intersection) / static_cast<float>(union_) : 0.0f);
        }
    }
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Concept Constrained Symmetric Dot Products

/** C = A × Aᵀ where C[i,j] = ⟨A[i], A[j]⟩. @c unexpected_dimensions_k unless the output is square
 *  over the input rows. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::dot_result_t> output_matrix_>
status_t dots_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    return numkong::dots_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                                static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                                static_cast<std::size_t>(output.stride_bytes(0)));
}

/** Partitioned symmetric dot products for parallel row-range work. @c unexpected_dimensions_k
 *  unless the output is square over the input rows. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::dot_result_t> output_matrix_>
status_t dots_symmetric(input_matrix_ const &input, output_matrix_ output, std::size_t row_start,
                        std::size_t row_count) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    return numkong::dots_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                                static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                                static_cast<std::size_t>(output.stride_bytes(0)), row_start, row_count);
}

/** Allocating symmetric dot products: C = A × Aᵀ. Empty for an empty input, or the allocation's or
 *  the kernel's failure. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::dot_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::dot_result_t>)
expected<matrix<typename value_type_::dot_result_t, allocator_type_>> dots_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::dot_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::zeros({num_vectors, num_vectors}, alloc);
    if (!result) return result;
    if (status_t status = dots_symmetric<value_type_>(input, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Symmetric Hamming distances: C[i,j] = hamming(A[i], A[j]). @c unexpected_dimensions_k unless the
 *  output is square over the input rows. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::hamming_result_t> output_matrix_>
status_t hammings_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    return numkong::hammings_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                                    static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                                    static_cast<std::size_t>(output.stride_bytes(0)));
}

/** Allocating symmetric Hamming distances. Empty for an empty input, or the allocation's or
 *  the kernel's failure. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::hamming_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::hamming_result_t>)
expected<matrix<typename value_type_::hamming_result_t, allocator_type_>> hammings_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::hamming_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::zeros({num_vectors, num_vectors}, alloc);
    if (!result) return result;
    if (status_t status = hammings_symmetric<value_type_>(input, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Symmetric Jaccard distances: C[i,j] = jaccard(A[i], A[j]). @c unexpected_dimensions_k unless the
 *  output is square over the input rows. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::jaccard_result_t> output_matrix_>
status_t jaccards_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    return numkong::jaccards_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                                    static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                                    static_cast<std::size_t>(output.stride_bytes(0)));
}

/** Allocating symmetric Jaccard distances. Empty for an empty input, or the allocation's or
 *  the kernel's failure. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::jaccard_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::jaccard_result_t>)
expected<matrix<typename value_type_::jaccard_result_t, allocator_type_>> jaccards_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::jaccard_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::zeros({num_vectors, num_vectors}, alloc);
    if (!result) return result;
    if (status_t status = jaccards_symmetric<value_type_>(input, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

#pragma endregion Concept Constrained Symmetric Dot Products

#pragma region Concept Constrained Packed Dot Products

/** Packed dot products: C = A × B_packedᵀ. @c unexpected_dimensions_k for an empty pack, a rank
 *  below 2 or a mismatched shape. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::dot_result_t> output_matrix_>
status_t dots_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    return numkong::dots_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                             packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                             static_cast<std::size_t>(c.stride_bytes(0)));
}

/** Allocating packed dot products: C = A × B_packedᵀ. @c unexpected_dimensions_k for an empty pack
 *  or a rank below 2, or the allocation's or the kernel's failure. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::dot_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::dot_result_t>)
expected<matrix<typename value_type_::dot_result_t, allocator_type_>> dots_packed(input_matrix_ const &a,
                                                                                  packed_type_ const &packed_b,
                                                                                  allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::dot_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto c = out_t::uninitialized({a.extent(0), packed_b.rows()}, alloc);
    if (!c) return c;
    if (status_t status = dots_packed<value_type_>(a, packed_b, c.value.as_matrix_span()); failed(status))
        return {out_t(alloc), status};
    return c;
}

/** Packed Hamming distances: C = hamming(A, B_packed). @c unexpected_dimensions_k for an empty
 *  pack, a rank below 2 or a mismatched shape. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::hamming_result_t> output_matrix_>
status_t hammings_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    return numkong::hammings_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                                 packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                                 static_cast<std::size_t>(c.stride_bytes(0)));
}

/** Allocating packed Hamming distances. @c unexpected_dimensions_k for an empty pack or a rank
 *  below 2, or the allocation's or the kernel's failure. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::hamming_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::hamming_result_t>)
expected<matrix<typename value_type_::hamming_result_t, allocator_type_>> hammings_packed(
    input_matrix_ const &a, packed_type_ const &packed_b, allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::hamming_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto c = out_t::uninitialized({a.extent(0), packed_b.rows()}, alloc);
    if (!c) return c;
    if (status_t status = hammings_packed<value_type_>(a, packed_b, c.value.as_matrix_span()); failed(status))
        return {out_t(alloc), status};
    return c;
}

/** Packed Jaccard distances: C = jaccard(A, B_packed). @c unexpected_dimensions_k for an empty
 *  pack, a rank below 2 or a mismatched shape. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::jaccard_result_t> output_matrix_>
status_t jaccards_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    return numkong::jaccards_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                                 packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                                 static_cast<std::size_t>(c.stride_bytes(0)));
}

/** Allocating packed Jaccard distances. @c unexpected_dimensions_k for an empty pack or a rank
 *  below 2, or the allocation's or the kernel's failure. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::jaccard_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::jaccard_result_t>)
expected<matrix<typename value_type_::jaccard_result_t, allocator_type_>> jaccards_packed(
    input_matrix_ const &a, packed_type_ const &packed_b, allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::jaccard_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto c = out_t::uninitialized({a.extent(0), packed_b.rows()}, alloc);
    if (!c) return c;
    if (status_t status = jaccards_packed<value_type_>(a, packed_b, c.value.as_matrix_span()); failed(status))
        return {out_t(alloc), status};
    return c;
}

#pragma endregion Concept Constrained Packed Dot Products

} // namespace ashvardanian::numkong

#endif // NUMKONG_DOTS_HPP
