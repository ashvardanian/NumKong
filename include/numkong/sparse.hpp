/**
 *  @brief C++ bindings for sparse-vector kernels.
 *  @file include/numkong/sparse.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_SPARSE_HPP
#define NK_SPARSE_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/sparse.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Count intersection of two sorted index arrays
 *  @param[in] a,b Sorted index arrays (ascending, unique elements)
 *  @param[in] a_length,b_length Number of elements in each array
 *  @param[out] result Optional buffer for the matched indices, at least `min(a_length, b_length)` long
 *  @param[out] count Output intersection count
 *
 *  @tparam index_type_ Index type (u16_t, u32_t, u64_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype index_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void sparse_intersect(index_type_ const *a, index_type_ const *b, std::size_t a_length, std::size_t b_length,
                      index_type_ *result, nk_size_t *count) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k;
    typename index_type_::raw_t *result_raw = result ? &result->raw_ : nullptr;

    if constexpr (std::is_same_v<index_type_, u16_t> && simd)
        nk_sparse_intersect_u16(&a->raw_, &b->raw_, a_length, b_length, result_raw, count);
    else if constexpr (std::is_same_v<index_type_, u32_t> && simd)
        nk_sparse_intersect_u32(&a->raw_, &b->raw_, a_length, b_length, result_raw, count);
    else if constexpr (std::is_same_v<index_type_, u64_t> && simd)
        nk_sparse_intersect_u64(&a->raw_, &b->raw_, a_length, b_length, result_raw, count);
    // Scalar fallback
    else {
        nk_size_t c = 0;
        std::size_t i = 0, j = 0;
        while (i < a_length && j < b_length) {
            if (a[i] < b[j]) i++;
            else if (b[j] < a[i]) j++;
            else {
                if (result) result[c] = a[i];
                c++, i++, j++;
            }
        }
        *count = c;
    }
}

/**
 *  @brief Sparse weighted dot product: Σ aₖ × bₖ over shared indices
 *  @param[in] a,b Sorted index arrays (ascending, unique elements)
 *  @param[in] a_weights,b_weights Weights corresponding to indices
 *  @param[in] a_length,b_length Number of elements in each array
 *  @param[out] product Output dot product
 *
 *  @tparam index_type_ Index type (u16_t, u32_t, u64_t)
 *  @tparam weight_t Weight type (bf16_t for u16 indices, f32_t for u32 indices)
 *  @tparam result_type_ Result type, defaults to `f32_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 *
 *  @note Computes sum of a_weights[i] * b_weights[j] for all i,j where a[i] == b[j]
 */
template <numeric_dtype index_type_, numeric_dtype weight_t,
          numeric_dtype result_type_ = typename weight_t::dot_result_t, allow_simd_t allow_simd_ = prefer_simd_k>
void sparse_dot(index_type_ const *a, index_type_ const *b, weight_t const *a_weights, weight_t const *b_weights,
                std::size_t a_length, std::size_t b_length, result_type_ *product) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<result_type_, typename weight_t::dot_result_t>;

    // u16 indices + bf16 weights -> f32 product
    if constexpr (std::is_same_v<index_type_, u16_t> && std::is_same_v<weight_t, bf16_t> && simd)
        nk_sparse_dot_u16bf16(&a->raw_, &b->raw_, &a_weights->raw_, &b_weights->raw_, a_length, b_length,
                              &product->raw_);
    else if constexpr (std::is_same_v<index_type_, u32_t> && std::is_same_v<weight_t, f32_t> && simd)
        nk_sparse_dot_u32f32(&a->raw_, &b->raw_, &a_weights->raw_, &b_weights->raw_, a_length, b_length,
                             &product->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        std::size_t i = 0, j = 0;
        while (i < a_length && j < b_length) {
            if (a[i] < b[j]) i++;
            else if (b[j] < a[i]) j++;
            else sum = fma<weight_t, result_type_>(a_weights[i], b_weights[j], sum), i++, j++;
        }
        *product = sum;
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

template <numeric_dtype index_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void sparse_intersect(vector_view<index_type_> a, vector_view<index_type_> b, nk_size_t *count) noexcept {
    sparse_intersect<index_type_, allow_simd_>(a.data(), b.data(), a.size(), b.size(), nullptr, count);
}

template <numeric_dtype index_type_, numeric_dtype weight_t,
          numeric_dtype result_type_ = typename weight_t::dot_result_t, allow_simd_t allow_simd_ = prefer_simd_k>
void sparse_dot(vector_view<index_type_> a, vector_view<index_type_> b, vector_view<weight_t> a_weights,
                vector_view<weight_t> b_weights, result_type_ *product) noexcept {
    sparse_dot<index_type_, weight_t, result_type_, allow_simd_>(a.data(), b.data(), a_weights.data(), b_weights.data(),
                                                                 a.size(), b.size(), product);
}

} // namespace ashvardanian::numkong

#endif // NK_SPARSE_HPP
