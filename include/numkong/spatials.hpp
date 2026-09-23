/**
 *  @brief C++ wrappers for SIMD-accelerated Batched Spatial Distance Matrices.
 *  @file include/numkong/spatials.hpp
 *  @author Ash Vardanian
 *  @date March 2026
 */
#ifndef NK_SPATIALS_HPP
#define NK_SPATIALS_HPP

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "numkong/spatials.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Symmetric angular distance matrix: C[i,j] = angular(A[i], A[j])
 *  @param[in] a Matrix A [vectors_count x depth]
 *  @param[in] vectors_count Number of vectors (n)
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between vectors in A
 *  @param[out] c Output matrix C [n x n]
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] row_start Starting row index (default 0)
 *  @param[in] row_count Number of rows to compute (default all)
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to `in_type_::angular_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::angular_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void angulars_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth, std::size_t a_stride_in_bytes,
                        result_type_ *c, std::size_t c_stride_in_bytes, std::size_t row_start = 0,
                        std::size_t row_count = std::numeric_limits<std::size_t>::max()) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::angular_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_angulars_symmetric_f64(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                  row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_angulars_symmetric_f32(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                  row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_angulars_symmetric_f16(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                  row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_angulars_symmetric_bf16(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_angulars_symmetric_e4m3(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_angulars_symmetric_e5m2(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_angulars_symmetric_e2m3(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
        nk_angulars_symmetric_e2m1(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_angulars_symmetric_e3m2(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_angulars_symmetric_i8(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                 row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_angulars_symmetric_u8(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                 row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_angulars_symmetric_i4(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                 row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_angulars_symmetric_u4(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                 row_start, row_count);
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
                result_type_ ab {}, aa {}, bb {};
                for (std::size_t l = 0; l < depth_values; l++) {
                    ab = fma(a_i[l], a_j[l], ab);
                    aa = fma(a_i[l], a_i[l], aa);
                    bb = fma(a_j[l], a_j[l], bb);
                }
                result_type_ cos_sim = ab / (aa.sqrt() * bb.sqrt());
                result_type_ distance = result_type_(1) - cos_sim;
                c_row[j] = distance > result_type_(0) ? distance : result_type_(0);
            }
        }
    }
}

/**
 *  @brief Symmetric Euclidean distance matrix: C[i,j] = euclidean(A[i], A[j])
 *  @param[in] a Matrix A [vectors_count x depth]
 *  @param[in] vectors_count Number of vectors (n)
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between vectors in A
 *  @param[out] c Output matrix C [n x n]
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] row_start Starting row index (default 0)
 *  @param[in] row_count Number of rows to compute (default all)
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to `in_type_::euclidean_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::euclidean_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void euclideans_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth,
                          std::size_t a_stride_in_bytes, result_type_ *c, std::size_t c_stride_in_bytes,
                          std::size_t row_start = 0,
                          std::size_t row_count = std::numeric_limits<std::size_t>::max()) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::euclidean_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_euclideans_symmetric_f64(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                    row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_euclideans_symmetric_f32(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                    row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_euclideans_symmetric_f16(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                    row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_euclideans_symmetric_bf16(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_euclideans_symmetric_e4m3(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_euclideans_symmetric_e5m2(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_euclideans_symmetric_e2m3(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
        nk_euclideans_symmetric_e2m1(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_euclideans_symmetric_e3m2(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_euclideans_symmetric_i8(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_euclideans_symmetric_u8(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_euclideans_symmetric_i4(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_euclideans_symmetric_u4(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
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
                for (std::size_t l = 0; l < depth_values; l++) sum = fdsa(a_i[l], a_j[l], sum);
                c_row[j] = sum.sqrt();
            }
        }
    }
}

/**
 *  @brief Packed angular distances: C = angular(A, B_packed)
 *  @param[in] a Matrix A [row_count x depth]
 *  @param[in] b_packed Packed B matrix (produced by nk_dots_pack_*)
 *  @param[out] c Output matrix C [row_count x column_count]
 *  @param[in] row_count Rows of A and C (m)
 *  @param[in] column_count Columns of B and C (n)
 *  @param[in] depth Shared inner dimension (k). Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between rows of A in bytes
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to `in_type_::angular_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::angular_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void angulars_packed(in_type_ const *a, void const *b_packed, result_type_ *c, size_t row_count, size_t column_count,
                     size_t depth, size_t a_stride_in_bytes, size_t c_stride_in_bytes) noexcept {
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::angular_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_angulars_packed_f64(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                               c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_angulars_packed_f32(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                               c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_angulars_packed_f16(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                               c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_angulars_packed_bf16(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_angulars_packed_e4m3(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_angulars_packed_e5m2(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_angulars_packed_e2m3(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
        nk_angulars_packed_e2m1(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_angulars_packed_e3m2(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_angulars_packed_i8(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                              c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_angulars_packed_u8(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                              c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_angulars_packed_i4(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                              c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_angulars_packed_u4(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                              c_stride_in_bytes);
    else {
        // Scalar fallback: extract pointer and stride, compute pairwise angular distances
        in_type_ const *b;
        size_t b_stride_in_bytes;
        char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
        std::memcpy(&b, b_packed_bytes, sizeof(void *));
        std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(size_t));

        char const *a_bytes = reinterpret_cast<char const *>(a);
        char const *b_bytes = reinterpret_cast<char const *>(b);
        char *c_bytes = reinterpret_cast<char *>(c);
        std::size_t depth_values = depth / dimensions_per_value<in_type_>();

        for (size_t i = 0; i < row_count; i++) {
            in_type_ const *a_row = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
            result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
            for (size_t j = 0; j < column_count; j++) {
                in_type_ const *b_row = reinterpret_cast<in_type_ const *>(b_bytes + j * b_stride_in_bytes);
                result_type_ ab {}, aa {}, bb {};
                for (std::size_t l = 0; l < depth_values; l++) {
                    ab = fma(a_row[l], b_row[l], ab);
                    aa = fma(a_row[l], a_row[l], aa);
                    bb = fma(b_row[l], b_row[l], bb);
                }
                result_type_ cos_sim = ab / (aa.sqrt() * bb.sqrt());
                result_type_ distance = result_type_(1) - cos_sim;
                c_row[j] = distance > result_type_(0) ? distance : result_type_(0);
            }
        }
    }
}

/**
 *  @brief Packed Euclidean distances: C = euclidean(A, B_packed)
 *  @param[in] a Matrix A [row_count x depth]
 *  @param[in] b_packed Packed B matrix (produced by nk_dots_pack_*)
 *  @param[out] c Output matrix C [row_count x column_count]
 *  @param[in] row_count Rows of A and C (m)
 *  @param[in] column_count Columns of B and C (n)
 *  @param[in] depth Shared inner dimension (k). Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between rows of A in bytes
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to `in_type_::euclidean_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::euclidean_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void euclideans_packed(in_type_ const *a, void const *b_packed, result_type_ *c, size_t row_count, size_t column_count,
                       size_t depth, size_t a_stride_in_bytes, size_t c_stride_in_bytes) noexcept {
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::euclidean_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_euclideans_packed_f64(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                 c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_euclideans_packed_f32(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                 c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_euclideans_packed_f16(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                 c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_euclideans_packed_bf16(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_euclideans_packed_e4m3(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_euclideans_packed_e5m2(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_euclideans_packed_e2m3(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
        nk_euclideans_packed_e2m1(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_euclideans_packed_e3m2(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_euclideans_packed_i8(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_euclideans_packed_u8(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_euclideans_packed_i4(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_euclideans_packed_u4(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride_in_bytes,
                                c_stride_in_bytes);
    else {
        // Scalar fallback: extract pointer and stride, compute pairwise euclidean distances
        in_type_ const *b;
        size_t b_stride_in_bytes;
        char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
        std::memcpy(&b, b_packed_bytes, sizeof(void *));
        std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(size_t));

        char const *a_bytes = reinterpret_cast<char const *>(a);
        char const *b_bytes = reinterpret_cast<char const *>(b);
        char *c_bytes = reinterpret_cast<char *>(c);
        std::size_t depth_values = depth / dimensions_per_value<in_type_>();

        for (size_t i = 0; i < row_count; i++) {
            in_type_ const *a_row = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
            result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
            for (size_t j = 0; j < column_count; j++) {
                in_type_ const *b_row = reinterpret_cast<in_type_ const *>(b_bytes + j * b_stride_in_bytes);
                result_type_ sum {};
                for (std::size_t l = 0; l < depth_values; l++) sum = fdsa(a_row[l], b_row[l], sum);
                c_row[j] = sum.sqrt();
            }
        }
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Concept Constrained Symmetric Spatial Distances

/** @brief Symmetric angular distances: C[i,j] = angular(A[i], A[j]). */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::angular_result_t> output_matrix_>
bool angulars_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::angulars_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                             static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                             static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** @brief Allocating symmetric angular distances. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::angular_result_t>>
matrix<typename value_type_::angular_result_t, allocator_type_> try_angulars_symmetric(
    input_matrix_ const &input) noexcept {
    using result_t = typename value_type_::angular_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!angulars_symmetric<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Symmetric Euclidean distances: C[i,j] = euclidean(A[i], A[j]). */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::euclidean_result_t> output_matrix_>
bool euclideans_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::euclideans_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                               static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                               static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** @brief Allocating symmetric Euclidean distances. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::euclidean_result_t>>
matrix<typename value_type_::euclidean_result_t, allocator_type_> try_euclideans_symmetric(
    input_matrix_ const &input) noexcept {
    using result_t = typename value_type_::euclidean_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!euclideans_symmetric<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Partitioned symmetric angular distances for parallel row-range work. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::angular_result_t> output_matrix_>
bool angulars_symmetric(input_matrix_ const &input, output_matrix_ &&output, std::size_t row_start,
                        std::size_t row_count) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::angulars_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                             static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                             static_cast<std::size_t>(output.stride_bytes(0)), row_start, row_count);
    return true;
}

/** @brief Partitioned symmetric Euclidean distances for parallel row-range work. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::euclidean_result_t> output_matrix_>
bool euclideans_symmetric(input_matrix_ const &input, output_matrix_ &&output, std::size_t row_start,
                          std::size_t row_count) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::euclideans_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                               static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                               static_cast<std::size_t>(output.stride_bytes(0)), row_start, row_count);
    return true;
}

#pragma endregion Concept Constrained Symmetric Spatial Distances

#pragma region Concept Constrained Packed Spatial Distances

/** @brief Packed angular distances: C = angular(A, B_packed). */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::angular_result_t> output_matrix_>
bool angulars_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return false;
    if (a.extent(1) != packed_b.depth()) return false;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return false;
    numkong::angulars_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                          packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                          static_cast<std::size_t>(c.stride_bytes(0)));
    return true;
}

/** @brief Allocating packed angular distances. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::angular_result_t>>
matrix<typename value_type_::angular_result_t, allocator_type_> try_angulars_packed(
    input_matrix_ const &a, packed_type_ const &packed_b) noexcept {
    using result_t = typename value_type_::angular_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return out_t {};
    auto c = out_t::try_empty({a.extent(0), packed_b.rows()});
    if (c.empty()) return c;
    if (!angulars_packed<value_type_>(a, packed_b, c.as_matrix_span())) return out_t {};
    return c;
}

/** @brief Packed Euclidean distances: C = euclidean(A, B_packed). */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::euclidean_result_t> output_matrix_>
bool euclideans_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return false;
    if (a.extent(1) != packed_b.depth()) return false;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return false;
    numkong::euclideans_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                            packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                            static_cast<std::size_t>(c.stride_bytes(0)));
    return true;
}

/** @brief Allocating packed Euclidean distances. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::euclidean_result_t>>
matrix<typename value_type_::euclidean_result_t, allocator_type_> try_euclideans_packed(
    input_matrix_ const &a, packed_type_ const &packed_b) noexcept {
    using result_t = typename value_type_::euclidean_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return out_t {};
    auto c = out_t::try_empty({a.extent(0), packed_b.rows()});
    if (c.empty()) return c;
    if (!euclideans_packed<value_type_>(a, packed_b, c.as_matrix_span())) return out_t {};
    return c;
}

#pragma endregion Concept Constrained Packed Spatial Distances

} // namespace ashvardanian::numkong

#endif // NK_SPATIALS_HPP
