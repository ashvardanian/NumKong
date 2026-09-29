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
                          index_type_ *result, std::size_t *count,
                          nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
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
                    nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
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

/** Counts the indices two sorted index runs share; @c unexpected_dimensions_k for a strided run or
 *  one ending mid-value. */
template <numeric_dtype index_type_, vector_of<index_type_> a_type_, vector_of<index_type_> b_type_>
expected<std::size_t> sparse_intersect(a_type_ const &a, b_type_ const &b,
                                       nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<index_type_ const>(a);
    auto b_values = contiguous_values_<index_type_ const>(b);
    if (!a_values || !b_values) return {0, status_t::unexpected_dimensions_k};
    std::size_t count = 0;
    status_t status = sparse_intersect<index_type_>(a_values.value.data(), b_values.value.data(), a_values.value.size(),
                                                    b_values.value.size(), nullptr, &count, capabilities);
    return {count, status};
}

/** Σ of weight products over the indices two sorted runs share; @c unexpected_dimensions_k when a
 *  weight run's size differs from its index run's, or any run is strided or ends mid-value. */
template <numeric_dtype index_type_, numeric_dtype weight_t,
          numeric_dtype result_type_ = typename weight_t::dot_result_t, vector_of<index_type_> a_type_,
          vector_of<index_type_> b_type_, vector_of<weight_t> a_weights_type_, vector_of<weight_t> b_weights_type_>
expected<result_type_> sparse_dot(a_type_ const &a, b_type_ const &b, a_weights_type_ const &a_weights,
                                  b_weights_type_ const &b_weights,
                                  nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<index_type_ const>(a);
    auto b_values = contiguous_values_<index_type_ const>(b);
    auto a_weights_values = contiguous_values_<weight_t const>(a_weights);
    auto b_weights_values = contiguous_values_<weight_t const>(b_weights);
    if (!a_values || !b_values || !a_weights_values || !b_weights_values ||
        a_weights_values.value.size() != a_values.value.size() ||
        b_weights_values.value.size() != b_values.value.size())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ product {};
    status_t status = sparse_dot<index_type_, weight_t, result_type_>(
        a_values.value.data(), b_values.value.data(), a_weights_values.value.data(), b_weights_values.value.data(),
        a_values.value.size(), b_values.value.size(), &product, capabilities);
    return {product, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_SPARSE_HPP
