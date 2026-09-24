/**
 *  @file include/numkong/dots.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for multi-target dot-product kernels.
 */
#ifndef NK_DOTS_HPP
#define NK_DOTS_HPP

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
void dots_unpacked(in_type_ const *a, in_type_ const *b, result_type_ *c, size_t row_count, size_t column_count,
                   size_t depth, size_t a_stride_in_bytes, size_t b_stride_in_bytes,
                   size_t c_stride_in_bytes) noexcept {
    char const *a_bytes = reinterpret_cast<char const *>(a);
    char const *b_bytes = reinterpret_cast<char const *>(b);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t const depth_values = depth / dimensions_per_value<in_type_>();

    for (size_t i = 0; i < row_count; i++) {
        in_type_ const *a_row = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
        for (size_t j = 0; j < column_count; j++) {
            in_type_ const *b_row = reinterpret_cast<in_type_ const *>(b_bytes + j * b_stride_in_bytes);
            result_type_ sum {};
            for (size_t l = 0; l < depth_values; l++) sum = fma(a_row[l], b_row[l], sum);
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
void dots_unpacked_conjugated(in_type_ const *a, in_type_ const *b, result_type_ *c, size_t row_count,
                              size_t column_count, size_t depth, size_t a_stride_in_bytes, size_t b_stride_in_bytes,
                              size_t c_stride_in_bytes) noexcept {
    char const *a_bytes = reinterpret_cast<char const *>(a);
    char const *b_bytes = reinterpret_cast<char const *>(b);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t const depth_values = depth / dimensions_per_value<in_type_>();

    for (size_t i = 0; i < row_count; i++) {
        in_type_ const *a_row = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
        for (size_t j = 0; j < column_count; j++) {
            in_type_ const *b_row = reinterpret_cast<in_type_ const *>(b_bytes + j * b_stride_in_bytes);
            result_type_ sum {};
            for (size_t l = 0; l < depth_values; l++) sum = fcma(b_row[l], a_row[l], sum);
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
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Accumulator/output type, defaults to @c in_type_::dot_result_t
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when @c prefer_simd_k
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void dots_packed(in_type_ const *a, void const *b_packed, result_type_ *c, size_t row_count, size_t column_count,
                 size_t depth, size_t a_stride_in_bytes, size_t c_stride_in_bytes) noexcept {
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::dot_result_t>;
    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_dots_packed_f64(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                           c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_dots_packed_f32(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                           c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_dots_packed_f16(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                           c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_dots_packed_bf16(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_dots_packed_i8(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                          c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_dots_packed_u8(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                          c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_dots_packed_e4m3(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_dots_packed_e5m2(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_dots_packed_e2m3(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
        nk_dots_packed_e2m1(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_dots_packed_e3m2(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_dots_packed_u4(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                          c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_dots_packed_i4(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                          c_stride_in_bytes);
    else {
        in_type_ const *b;
        size_t b_stride_in_bytes;
        char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
        std::memcpy(&b, b_packed_bytes, sizeof(void *));
        std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(size_t));
        dots_unpacked<in_type_, result_type_>(a, b, c, row_count, column_count, depth, a_stride_in_bytes,
                                              b_stride_in_bytes, c_stride_in_bytes);
    }
}

/**
 *  @brief Symmetric dot products: C = A × Aᵀ where C[i,j] = ⟨A[i], A[j]⟩
 *  @param[in] a Matrix A of shape @b [vectors,depth]
 *  @param[in] vectors_count Number of vectors
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between vectors in A
 *  @param[out] c Output matrix C of shape @b [vectors,vectors]
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Accumulator/output type, defaults to @c in_type_::dot_result_t
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when @c prefer_simd_k
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void dots_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth, std::size_t a_stride_in_bytes,
                    result_type_ *c, std::size_t c_stride_in_bytes, std::size_t row_start = 0,
                    std::size_t row_count = std::numeric_limits<std::size_t>::max()) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_dots_symmetric_f64(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                              row_count);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_dots_symmetric_f32(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                              row_count);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_dots_symmetric_f16(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                              row_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_dots_symmetric_bf16(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                               row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_dots_symmetric_i8(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                             row_count);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_dots_symmetric_u8(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                             row_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_dots_symmetric_e4m3(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                               row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_dots_symmetric_e5m2(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                               row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_dots_symmetric_e2m3(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                               row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
        nk_dots_symmetric_e2m1(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                               row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_dots_symmetric_e3m2(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                               row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_dots_symmetric_u4(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                             row_count);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_dots_symmetric_i4(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                             row_count);
    else {
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
    }
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
 *  @tparam in_type_ Input element type (u1x8_t)
 *  @tparam result_type_ Output type (u32_t for Hamming distances)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when @c prefer_simd_k
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::hamming_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void hammings_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth, std::size_t a_stride_in_bytes,
                        result_type_ *c, std::size_t c_stride_in_bytes, std::size_t row_start = 0,
                        std::size_t row_count = std::numeric_limits<std::size_t>::max()) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::hamming_result_t>;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
        nk_hammings_symmetric_u1(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                 row_start, row_count);
    else {
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
    }
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
 *  @tparam in_type_ Input element type (u1x8_t)
 *  @tparam result_type_ Output type (u32_t for Hamming distances)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when @c prefer_simd_k
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::hamming_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void hammings_packed(in_type_ const *a, void const *b_packed, result_type_ *c, std::size_t row_count,
                     std::size_t column_count, std::size_t depth, std::size_t a_stride_in_bytes = 0,
                     std::size_t c_stride_in_bytes = 0) noexcept {
    // Compute default strides
    if (!a_stride_in_bytes) a_stride_in_bytes = depth / dimensions_per_value<in_type_>() * sizeof(in_type_);
    if (!c_stride_in_bytes) c_stride_in_bytes = column_count * sizeof(result_type_);

    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::hamming_result_t>;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch) {
        nk_hammings_packed_u1(reinterpret_cast<nk_u1x8_t const *>(a), b_packed, reinterpret_cast<nk_u32_t *>(c),
                              row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    }
    else {
        // Scalar fallback: extract pointer and stride from b_packed, then compute directly
        in_type_ const *b;
        size_t b_stride_in_bytes;
        char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
        std::memcpy(&b, b_packed_bytes, sizeof(void *));
        std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(size_t));

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
    }
}

/** Symmetric Jaccard distance matrix: C[i,j] = jaccard(A[i], A[j]). */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::jaccard_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void jaccards_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth, std::size_t a_stride_in_bytes,
                        result_type_ *c, std::size_t c_stride_in_bytes, std::size_t row_start = 0,
                        std::size_t row_count = std::numeric_limits<std::size_t>::max()) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::jaccard_result_t>;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
        nk_jaccards_symmetric_u1(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                 row_start, row_count);
    else {
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
    }
}

/** Computes Jaccard distances between rows of A and columns of packed B. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::jaccard_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void jaccards_packed(in_type_ const *a, void const *b_packed, result_type_ *c, std::size_t row_count,
                     std::size_t column_count, std::size_t depth, std::size_t a_stride_in_bytes = 0,
                     std::size_t c_stride_in_bytes = 0) noexcept {
    if (!a_stride_in_bytes) a_stride_in_bytes = depth / dimensions_per_value<in_type_>() * sizeof(in_type_);
    if (!c_stride_in_bytes) c_stride_in_bytes = column_count * sizeof(result_type_);

    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::jaccard_result_t>;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch) {
        nk_jaccards_packed_u1(reinterpret_cast<nk_u1x8_t const *>(a), b_packed, reinterpret_cast<nk_f32_t *>(c),
                              row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    }
    else {
        // Scalar fallback: extract pointer and stride from b_packed, then compute directly
        in_type_ const *b;
        size_t b_stride_in_bytes;
        char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
        std::memcpy(&b, b_packed_bytes, sizeof(void *));
        std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(size_t));

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
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Concept Constrained Symmetric Dot Products

/** C = A × Aᵀ where C[i,j] = ⟨A[i], A[j]⟩. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::dot_result_t> output_matrix_>
bool dots_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::dots_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                         static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                         static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** Partitioned symmetric dot products for parallel row-range work. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::dot_result_t> output_matrix_>
bool dots_symmetric(input_matrix_ const &input, output_matrix_ output, std::size_t row_start,
                    std::size_t row_count) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::dots_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                         static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                         static_cast<std::size_t>(output.stride_bytes(0)), row_start, row_count);
    return true;
}

/** Allocating symmetric dot products: C = A × Aᵀ. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::dot_result_t>>
matrix<typename value_type_::dot_result_t, allocator_type_> try_dots_symmetric(input_matrix_ const &input) noexcept {
    using result_t = typename value_type_::dot_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!dots_symmetric<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** Symmetric Hamming distances: C[i,j] = hamming(A[i], A[j]). */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::hamming_result_t> output_matrix_>
bool hammings_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::hammings_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                             static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                             static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** Allocating symmetric Hamming distances. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::hamming_result_t>>
matrix<typename value_type_::hamming_result_t, allocator_type_> try_hammings_symmetric(
    input_matrix_ const &input) noexcept {
    using result_t = typename value_type_::hamming_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!hammings_symmetric<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** Symmetric Jaccard distances: C[i,j] = jaccard(A[i], A[j]). */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::jaccard_result_t> output_matrix_>
bool jaccards_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::jaccards_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                             static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                             static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** Allocating symmetric Jaccard distances. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::jaccard_result_t>>
matrix<typename value_type_::jaccard_result_t, allocator_type_> try_jaccards_symmetric(
    input_matrix_ const &input) noexcept {
    using result_t = typename value_type_::jaccard_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!jaccards_symmetric<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

#pragma endregion Concept Constrained Symmetric Dot Products

#pragma region Concept Constrained Packed Dot Products

/** Packed dot products: C = A × B_packedᵀ. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::dot_result_t> output_matrix_>
bool dots_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return false;
    if (a.extent(1) != packed_b.depth()) return false;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return false;
    numkong::dots_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                      packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                      static_cast<std::size_t>(c.stride_bytes(0)));
    return true;
}

/** Allocating packed dot products: C = A × B_packedᵀ. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::dot_result_t>>
matrix<typename value_type_::dot_result_t, allocator_type_> try_dots_packed(input_matrix_ const &a,
                                                                            packed_type_ const &packed_b) noexcept {
    using result_t = typename value_type_::dot_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return out_t {};
    auto c = out_t::try_empty({a.extent(0), packed_b.rows()});
    if (c.empty()) return c;
    if (!dots_packed<value_type_>(a, packed_b, c.as_matrix_span())) return out_t {};
    return c;
}

/** Packed Hamming distances: C = hamming(A, B_packed). */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::hamming_result_t> output_matrix_>
bool hammings_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return false;
    if (a.extent(1) != packed_b.depth()) return false;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return false;
    numkong::hammings_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                          packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                          static_cast<std::size_t>(c.stride_bytes(0)));
    return true;
}

/** Allocating packed Hamming distances. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::hamming_result_t>>
matrix<typename value_type_::hamming_result_t, allocator_type_> try_hammings_packed(
    input_matrix_ const &a, packed_type_ const &packed_b) noexcept {
    using result_t = typename value_type_::hamming_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return out_t {};
    auto c = out_t::try_empty({a.extent(0), packed_b.rows()});
    if (c.empty()) return c;
    if (!hammings_packed<value_type_>(a, packed_b, c.as_matrix_span())) return out_t {};
    return c;
}

/** Packed Jaccard distances: C = jaccard(A, B_packed). */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::jaccard_result_t> output_matrix_>
bool jaccards_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return false;
    if (a.extent(1) != packed_b.depth()) return false;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return false;
    numkong::jaccards_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                          packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                          static_cast<std::size_t>(c.stride_bytes(0)));
    return true;
}

/** Allocating packed Jaccard distances. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::jaccard_result_t>>
matrix<typename value_type_::jaccard_result_t, allocator_type_> try_jaccards_packed(
    input_matrix_ const &a, packed_type_ const &packed_b) noexcept {
    using result_t = typename value_type_::jaccard_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return out_t {};
    auto c = out_t::try_empty({a.extent(0), packed_b.rows()});
    if (c.empty()) return c;
    if (!jaccards_packed<value_type_>(a, packed_b, c.as_matrix_span())) return out_t {};
    return c;
}

#pragma endregion Concept Constrained Packed Dot Products

} // namespace ashvardanian::numkong

#endif // NK_DOTS_HPP
