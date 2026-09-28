/**
 *  @file include/numkong/spatials.hpp
 *  @author Ash Vardanian
 *  @date March 6, 2026
 *  @brief C++ wrappers for SIMD-accelerated batched spatial distance matrices.
 */
#ifndef NUMKONG_SPATIALS_HPP
#define NUMKONG_SPATIALS_HPP

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "numkong/spatials.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Symmetric angular distance matrix: C[i,j] = angular(A[i], A[j])
 *  @param[in] a Matrix A, shape @b [vectors_count,depth]
 *  @param[in] vectors_count Number of vectors, n
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between vectors in A
 *  @param[out] c Output matrix C, shape @b [n,n]
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] row_start Starting row index, default 0
 *  @param[in] row_count Number of rows to compute, default all
 *
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to @c in_type_::angular_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::angular_result_t>
status_t angulars_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth,
                            std::size_t a_stride_in_bytes, result_type_ *c, std::size_t c_stride_in_bytes,
                            std::size_t row_start = 0, std::size_t row_count = std::numeric_limits<std::size_t>::max(),
                            nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::angular_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_f64_best(&a->raw_, vectors_count, depth,
                                                                        a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                        row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_f32_best(&a->raw_, vectors_count, depth,
                                                                        a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                        row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_f16_best(&a->raw_, vectors_count, depth,
                                                                        a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                        row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_bf16_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                         row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_e4m3_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                         row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_e5m2_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                         row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_e2m3_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                         row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_e2m1_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                         row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_e3m2_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                         row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_i8_best(&a->raw_, vectors_count, depth,
                                                                       a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                       row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_u8_best(&a->raw_, vectors_count, depth,
                                                                       a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                       row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_i4_best(&a->raw_, vectors_count, depth,
                                                                       a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                       row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_symmetric_u4_best(&a->raw_, vectors_count, depth,
                                                                       a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
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
    return status_t::success_k;
}

/**
 *  @brief Symmetric Euclidean distance matrix: C[i,j] = euclidean(A[i], A[j])
 *  @param[in] a Matrix A, shape @b [vectors_count,depth]
 *  @param[in] vectors_count Number of vectors, n
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between vectors in A
 *  @param[out] c Output matrix C, shape @b [n,n]
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] row_start Starting row index, default 0
 *  @param[in] row_count Number of rows to compute, default all
 *
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to @c in_type_::euclidean_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::euclidean_result_t>
status_t euclideans_symmetric(in_type_ const *a, std::size_t vectors_count, std::size_t depth,
                              std::size_t a_stride_in_bytes, result_type_ *c, std::size_t c_stride_in_bytes,
                              std::size_t row_start = 0,
                              std::size_t row_count = std::numeric_limits<std::size_t>::max(),
                              nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = vectors_count;
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::euclidean_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_euclideans_symmetric_f64_best(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_,
                                                 c_stride_in_bytes, row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_euclideans_symmetric_f32_best(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_,
                                                 c_stride_in_bytes, row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_euclideans_symmetric_f16_best(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_,
                                                 c_stride_in_bytes, row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(
                nk_euclideans_symmetric_bf16_best(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_,
                                                  c_stride_in_bytes, row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(
                nk_euclideans_symmetric_e4m3_best(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_,
                                                  c_stride_in_bytes, row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(
                nk_euclideans_symmetric_e5m2_best(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_,
                                                  c_stride_in_bytes, row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(
                nk_euclideans_symmetric_e2m3_best(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_,
                                                  c_stride_in_bytes, row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
            return static_cast<status_t>(
                nk_euclideans_symmetric_e2m1_best(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_,
                                                  c_stride_in_bytes, row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(
                nk_euclideans_symmetric_e3m2_best(&a->raw_, vectors_count, depth, a_stride_in_bytes, &c->raw_,
                                                  c_stride_in_bytes, row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_symmetric_i8_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                         row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_symmetric_u8_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                         row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_symmetric_i4_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                                                         row_start, row_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_symmetric_u4_best(&a->raw_, vectors_count, depth,
                                                                         a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
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
            for (std::size_t l = 0; l < depth_values; l++) sum = fdsa(a_i[l], a_j[l], sum);
            c_row[j] = sum.sqrt();
        }
    }
    return status_t::success_k;
}

/**
 *  @brief Packed angular distances: C = angular(A, B_packed)
 *  @param[in] a Matrix A, shape @b [row_count,depth]
 *  @param[in] b_packed Packed B matrix, produced by nk_dots_pack_*
 *  @param[out] c Output matrix C, shape @b [row_count,column_count]
 *  @param[in] row_count Rows of A and C, m
 *  @param[in] column_count Columns of B and C, n
 *  @param[in] depth Shared inner dimension, k, in multiples of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between rows of A in bytes
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to @c in_type_::angular_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::angular_result_t>
status_t angulars_packed(in_type_ const *a, void const *b_packed, result_type_ *c, size_t row_count,
                         size_t column_count, size_t depth, size_t a_stride_in_bytes, size_t c_stride_in_bytes,
                         nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::angular_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_f64_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                     column_count, depth, a_stride_in_bytes,
                                                                     c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_f32_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                     column_count, depth, a_stride_in_bytes,
                                                                     c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_f16_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                     column_count, depth, a_stride_in_bytes,
                                                                     c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_bf16_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_e4m3_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_e5m2_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_e2m3_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_e2m1_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_e3m2_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_i8_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                    column_count, depth, a_stride_in_bytes,
                                                                    c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_u8_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                    column_count, depth, a_stride_in_bytes,
                                                                    c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_i4_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                    column_count, depth, a_stride_in_bytes,
                                                                    c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_angulars_packed_u4_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                    column_count, depth, a_stride_in_bytes,
                                                                    c_stride_in_bytes, capabilities, stream));
    }
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
    return status_t::success_k;
}

/**
 *  @brief Packed Euclidean distances: C = euclidean(A, B_packed)
 *  @param[in] a Matrix A, shape @b [row_count,depth]
 *  @param[in] b_packed Packed B matrix, produced by nk_dots_pack_*
 *  @param[out] c Output matrix C, shape @b [row_count,column_count]
 *  @param[in] row_count Rows of A and C, m
 *  @param[in] column_count Columns of B and C, n
 *  @param[in] depth Shared inner dimension, k, in multiples of the values per byte.
 *  @param[in] a_stride_in_bytes Stride between rows of A in bytes
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to @c in_type_::euclidean_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::euclidean_result_t>
status_t euclideans_packed(in_type_ const *a, void const *b_packed, result_type_ *c, size_t row_count,
                           size_t column_count, size_t depth, size_t a_stride_in_bytes, size_t c_stride_in_bytes,
                           nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::euclidean_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_f64_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                       column_count, depth, a_stride_in_bytes,
                                                                       c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_f32_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                       column_count, depth, a_stride_in_bytes,
                                                                       c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_f16_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                       column_count, depth, a_stride_in_bytes,
                                                                       c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_bf16_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                        column_count, depth, a_stride_in_bytes,
                                                                        c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_e4m3_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                        column_count, depth, a_stride_in_bytes,
                                                                        c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_e5m2_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                        column_count, depth, a_stride_in_bytes,
                                                                        c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_e2m3_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                        column_count, depth, a_stride_in_bytes,
                                                                        c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_e2m1_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                        column_count, depth, a_stride_in_bytes,
                                                                        c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_e3m2_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                        column_count, depth, a_stride_in_bytes,
                                                                        c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_i8_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_u8_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_i4_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_euclideans_packed_u4_best(&a->raw_, b_packed, &c->raw_, row_count,
                                                                      column_count, depth, a_stride_in_bytes,
                                                                      c_stride_in_bytes, capabilities, stream));
    }
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
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Concept Constrained Symmetric Spatial Distances

/** Symmetric angular distances: C[i,j] = angular(A[i], A[j]). @c unexpected_dimensions_k unless the
 *  output is square over the input rows. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::angular_result_t> output_matrix_>
status_t angulars_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    return numkong::angulars_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                                    static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                                    static_cast<std::size_t>(output.stride_bytes(0)));
}

/** Allocating symmetric angular distances. Empty for an empty input, or the allocation's or
 *  the kernel's failure. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::angular_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::angular_result_t>)
expected<matrix<typename value_type_::angular_result_t, allocator_type_>> angulars_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::angular_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::zeros({num_vectors, num_vectors}, alloc);
    if (!result) return result;
    if (status_t status = angulars_symmetric<value_type_>(input, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Symmetric Euclidean distances: C[i,j] = euclidean(A[i], A[j]). @c unexpected_dimensions_k unless
 *  the output is square over the input rows. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::euclidean_result_t> output_matrix_>
status_t euclideans_symmetric(input_matrix_ const &input, output_matrix_ &&output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    return numkong::euclideans_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                                      static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                                      static_cast<std::size_t>(output.stride_bytes(0)));
}

/** Allocating symmetric Euclidean distances. Empty for an empty input, or the allocation's or
 *  the kernel's failure. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::euclidean_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::euclidean_result_t>)
expected<matrix<typename value_type_::euclidean_result_t, allocator_type_>> euclideans_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::euclidean_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::zeros({num_vectors, num_vectors}, alloc);
    if (!result) return result;
    if (status_t status = euclideans_symmetric<value_type_>(input, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Partitioned symmetric angular distances for parallel row-range work. @c unexpected_dimensions_k
 *  unless the output is square over the input rows. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::angular_result_t> output_matrix_>
status_t angulars_symmetric(input_matrix_ const &input, output_matrix_ &&output, std::size_t row_start,
                            std::size_t row_count) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    return numkong::angulars_symmetric<value_type_>(
        input.data(), num_vectors, input.extent(1), static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
        static_cast<std::size_t>(output.stride_bytes(0)), row_start, row_count);
}

/** Partitioned symmetric Euclidean distances for parallel row-range work. @c
 *  unexpected_dimensions_k unless the output is square over the input rows. */
template <numeric_dtype value_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::euclidean_result_t> output_matrix_>
status_t euclideans_symmetric(input_matrix_ const &input, output_matrix_ &&output, std::size_t row_start,
                              std::size_t row_count) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    return numkong::euclideans_symmetric<value_type_>(
        input.data(), num_vectors, input.extent(1), static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
        static_cast<std::size_t>(output.stride_bytes(0)), row_start, row_count);
}

#pragma endregion Concept Constrained Symmetric Spatial Distances

#pragma region Concept Constrained Packed Spatial Distances

/** Packed angular distances: C = angular(A, B_packed). @c unexpected_dimensions_k for an empty
 *  pack, a rank below 2 or a mismatched shape. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::angular_result_t> output_matrix_>
status_t angulars_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    return numkong::angulars_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                                 packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                                 static_cast<std::size_t>(c.stride_bytes(0)));
}

/** Allocating packed angular distances. @c unexpected_dimensions_k for an empty pack or a rank
 *  below 2, or the allocation's or the kernel's failure. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::angular_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::angular_result_t>)
expected<matrix<typename value_type_::angular_result_t, allocator_type_>> angulars_packed(
    input_matrix_ const &a, packed_type_ const &packed_b, allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::angular_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto c = out_t::uninitialized({a.extent(0), packed_b.rows()}, alloc);
    if (!c) return c;
    if (status_t status = angulars_packed<value_type_>(a, packed_b, c.value.as_matrix_span()); failed(status))
        return {out_t(alloc), status};
    return c;
}

/** Packed Euclidean distances: C = euclidean(A, B_packed). @c unexpected_dimensions_k for an empty
 *  pack, a rank below 2 or a mismatched shape. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          mutable_matrix_of<typename value_type_::euclidean_result_t> output_matrix_>
status_t euclideans_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    return numkong::euclideans_packed<value_type_>(a.data(), packed_b.data(), c.data(), a.extent(0), packed_b.rows(),
                                                   packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)),
                                                   static_cast<std::size_t>(c.stride_bytes(0)));
}

/** Allocating packed Euclidean distances. @c unexpected_dimensions_k for an empty pack or a rank
 *  below 2, or the allocation's or the kernel's failure. */
template <numeric_dtype value_type_, packed_matrix_like packed_type_, const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::euclidean_result_t>>
    requires(!mutable_matrix_of<allocator_type_, typename value_type_::euclidean_result_t>)
expected<matrix<typename value_type_::euclidean_result_t, allocator_type_>> euclideans_packed(
    input_matrix_ const &a, packed_type_ const &packed_b, allocator_type_ alloc = {}) noexcept {
    using result_t = typename value_type_::euclidean_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto c = out_t::uninitialized({a.extent(0), packed_b.rows()}, alloc);
    if (!c) return c;
    if (status_t status = euclideans_packed<value_type_>(a, packed_b, c.value.as_matrix_span()); failed(status))
        return {out_t(alloc), status};
    return c;
}

#pragma endregion Concept Constrained Packed Spatial Distances

} // namespace ashvardanian::numkong

#endif // NUMKONG_SPATIALS_HPP
