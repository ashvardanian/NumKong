/**
 *  @file include/numkong/sparse.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for sparse-vector kernels.
 */
#ifndef NUMKONG_SPARSE_HPP
#define NUMKONG_SPARSE_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/sparse.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Count intersection of two sorted index arrays
 *  @param[in] a,b Sorted index arrays (ascending, unique elements)
 *  @param[in] a_length Number of elements in @p a
 *  @param[in] b_length Number of elements in @p b
 *  @param[out] result Optional buffer for matches, at least min( @p a_length, @p b_length) long
 *  @param[out] count Output intersection count
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam index_type_ Index type (u16_t, u32_t, u64_t)
 */
template <numeric_dtype index_type_>
status_t sparse_intersect(index_type_ const *a, index_type_ const *b, std::size_t a_length, std::size_t b_length,
                          index_type_ *result, std::size_t *count, nk_capability_t capabilities = cpu_capabilities(),
                          void *stream = nullptr) noexcept {
    typename index_type_::raw_t *result_raw = result ? &result->raw_ : nullptr;
    nk_size_t found = 0;
    if (capabilities) {
        if constexpr (std::is_same_v<index_type_, u16_t>) {
            nk_status_t status = nk_sparse_intersect_u16_best(&a->raw_, &b->raw_, a_length, b_length, result_raw,
                                                              &found, capabilities, stream);
            *count = static_cast<std::size_t>(found);
            return static_cast<status_t>(status);
        }
        else if constexpr (std::is_same_v<index_type_, u32_t>) {
            nk_status_t status = nk_sparse_intersect_u32_best(&a->raw_, &b->raw_, a_length, b_length, result_raw,
                                                              &found, capabilities, stream);
            *count = static_cast<std::size_t>(found);
            return static_cast<status_t>(status);
        }
        else if constexpr (std::is_same_v<index_type_, u64_t>) {
            nk_status_t status = nk_sparse_intersect_u64_best(&a->raw_, &b->raw_, a_length, b_length, result_raw,
                                                              &found, capabilities, stream);
            *count = static_cast<std::size_t>(found);
            return static_cast<status_t>(status);
        }
    }
    std::size_t c = 0, i = 0, j = 0;
    while (i < a_length && j < b_length) {
        if (a[i] < b[j]) i++;
        else if (b[j] < a[i]) j++;
        else {
            if (result) result[c] = a[i];
            c++, i++, j++;
        }
    }
    *count = c;
    return status_t::success_k;
}

/**
 *  @brief Sparse weighted dot product: Σ aₖ × bₖ over shared indices
 *  @param[in] a,b Sorted index arrays (ascending, unique elements)
 *  @param[in] a_weights Weights corresponding to indices of @p a
 *  @param[in] b_weights Weights corresponding to indices of @p b
 *  @param[in] a_length Number of elements in @p a
 *  @param[in] b_length Number of elements in @p b
 *  @param[out] product Output dot product
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam index_type_ Index type (u16_t, u32_t, u64_t)
 *  @tparam weight_t Weight type (bf16_t for u16 indices, f32_t for u32 indices)
 *  @tparam result_type_ Result type, defaults to @c f32_t
 *
 *  @note Computes sum of @p a_weights[i] * @p b_weights[j] for all i,j where a[i] == b[j]
 */
template <numeric_dtype index_type_, numeric_dtype weight_t,
          numeric_dtype result_type_ = typename weight_t::dot_result_t>
status_t sparse_dot(index_type_ const *a, index_type_ const *b, weight_t const *a_weights, weight_t const *b_weights,
                    std::size_t a_length, std::size_t b_length, result_type_ *product,
                    nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename weight_t::dot_result_t>;

    if (capabilities) {
        // u16 indices + bf16 weights → f32 product
        if constexpr (std::is_same_v<index_type_, u16_t> && std::is_same_v<weight_t, bf16_t> && dispatch)
            return static_cast<status_t>(nk_sparse_dot_u16bf16_best(&a->raw_, &b->raw_, &a_weights->raw_,
                                                                    &b_weights->raw_, a_length, b_length,
                                                                    &product->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<index_type_, u32_t> && std::is_same_v<weight_t, f32_t> && dispatch)
            return static_cast<status_t>(nk_sparse_dot_u32f32_best(&a->raw_, &b->raw_, &a_weights->raw_,
                                                                   &b_weights->raw_, a_length, b_length, &product->raw_,
                                                                   capabilities, stream));
    }
    result_type_ sum {};
    std::size_t i = 0, j = 0;
    while (i < a_length && j < b_length) {
        if (a[i] < b[j]) i++;
        else if (b[j] < a[i]) j++;
        else sum = fma<weight_t, result_type_>(a_weights[i], b_weights[j], sum), i++, j++;
    }
    *product = sum;
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

template <numeric_dtype index_type_>
status_t sparse_intersect(vector_view<index_type_> a, vector_view<index_type_> b, std::size_t *count,
                          nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    return sparse_intersect<index_type_>(a.data(), b.data(), a.size(), b.size(), nullptr, count, capabilities, stream);
}

template <numeric_dtype index_type_, numeric_dtype weight_t,
          numeric_dtype result_type_ = typename weight_t::dot_result_t>
status_t sparse_dot(vector_view<index_type_> a, vector_view<index_type_> b, vector_view<weight_t> a_weights,
                    vector_view<weight_t> b_weights, result_type_ *product,
                    nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    if (a_weights.size() != a.size() || b_weights.size() != b.size()) return status_t::unexpected_dimensions_k;
    return sparse_dot<index_type_, weight_t, result_type_>(a.data(), b.data(), a_weights.data(), b_weights.data(),
                                                           a.size(), b.size(), product, capabilities, stream);
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_SPARSE_HPP
