/**
 *  @file include/numkong/set.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for set-intersection kernels.
 */
#ifndef NUMKONG_SET_HPP
#define NUMKONG_SET_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/set.h"
#include "numkong/sets.h"
#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Hamming distance: Σ(aᵢ ⊕ bᵢ)
 *  @param[in] a,b Input vectors
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output count
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type (u1x8_t or u8_t)
 *  @tparam result_type_ Accumulator type, defaults to @c in_type_::hamming_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::hamming_result_t>
status_t hamming(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r,
                 nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::hamming_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
            return static_cast<status_t>(nk_hamming_u1_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_hamming_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    }
    constexpr std::size_t dims_per_value = dimensions_per_value<in_type_>();
    std::size_t n = d / dims_per_value;
    typename result_type_::raw_t count = 0;
    for (std::size_t i = 0; i < n; i++) count += count_differences(a[i], b[i]);
    *r = result_type_::from_raw(count);
    return status_t::success_k;
}

/**
 *  @brief Jaccard distance: 1 − |A ∩ B| / |A ∪ B|
 *  @param[in] a,b Input vectors
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output distance
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  For u1x8_t bit vectors, uses popcount(AND) / popcount(OR). For u16_t/u32_t element vectors, uses
 *  count of matching elements / total.
 *
 *  @tparam in_type_ Input vector element type (u1x8_t, u16_t, or u32_t)
 *  @tparam result_type_ Accumulator type, defaults to @c in_type_::jaccard_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::jaccard_result_t>
status_t jaccard(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r,
                 nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::jaccard_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
            return static_cast<status_t>(nk_jaccard_u1_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u16_t> && dispatch)
            return static_cast<status_t>(nk_jaccard_u16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u32_t> && dispatch)
            return static_cast<status_t>(nk_jaccard_u32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    }
    constexpr std::size_t dims_per_value = dimensions_per_value<in_type_>();
    std::size_t n = d / dims_per_value;
    std::uint32_t intersection_count = 0, union_count = 0;
    for (std::size_t i = 0; i < n; i++)
        intersection_count += count_intersection(a[i], b[i]), union_count += count_union(a[i], b[i]);
    if (union_count == 0) *r = result_type_();
    else *r = result_type_(1) - result_type_(intersection_count) / result_type_(union_count);
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

/** Hamming distance between two bit runs of equal dimensions; @c unexpected_dimensions_k
 *  when they differ or either run is strided or ends mid-value. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::hamming_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_>
expected<result_type_> hamming(a_type_ const &a, b_type_ const &b,
                               nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = hamming<in_type_, result_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result,
                                                      capabilities);
    return {result, status};
}

/** Jaccard distance between two bit runs, failing like the concept @c hamming. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::jaccard_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_>
expected<result_type_> jaccard(a_type_ const &a, b_type_ const &b,
                               nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = jaccard<in_type_, result_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result,
                                                      capabilities);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_SET_HPP
