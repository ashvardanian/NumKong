/**
 *  @file include/numkong/set.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for set-intersection kernels.
 */
#ifndef NK_SET_HPP
#define NK_SET_HPP

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
 *
 *  @tparam in_type_ Input vector element type (u1x8_t or u8_t)
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::hamming_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::hamming_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void hamming(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::hamming_result_t>;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && simd) nk_hamming_u1(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_hamming_u8(&a->raw_, &b->raw_, d, &r->raw_);
    else {
        constexpr std::size_t dims_per_value = dimensions_per_value<in_type_>();
        std::size_t n = d / dims_per_value;
        typename result_type_::raw_t count = 0;
        for (std::size_t i = 0; i < n; i++) count += count_differences(a[i], b[i]);
        *r = result_type_::from_raw(count);
    }
}

/**
 *  @brief Jaccard distance: 1 − |A ∩ B| / |A ∪ B|
 *  @param[in] a,b Input vectors
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output distance
 *
 *  For u1x8_t bit vectors, uses popcount(AND) / popcount(OR). For u16_t/u32_t element vectors, uses
 *  count of matching elements / total.
 *
 *  @tparam in_type_ Input vector element type (u1x8_t, u16_t, or u32_t)
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::jaccard_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::jaccard_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void jaccard(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::jaccard_result_t>;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && simd) nk_jaccard_u1(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd) nk_jaccard_u16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd) nk_jaccard_u32(&a->raw_, &b->raw_, d, &r->raw_);
    else {
        constexpr std::size_t dims_per_value = dimensions_per_value<in_type_>();
        std::size_t n = d / dims_per_value;
        std::uint32_t intersection_count = 0, union_count = 0;
        for (std::size_t i = 0; i < n; i++)
            intersection_count += count_intersection(a[i], b[i]), union_count += count_union(a[i], b[i]);
        if (union_count == 0) *r = result_type_();
        else *r = result_type_(1) - result_type_(intersection_count) / result_type_(union_count);
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::hamming_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k, std::size_t max_rank_a_, std::size_t max_rank_b_>
void hamming(tensor_view<in_type_, max_rank_a_> a, tensor_view<in_type_, max_rank_b_> b, std::size_t d,
             result_type_ *r) noexcept {
    hamming<in_type_, result_type_, allow_simd_>(a.data(), b.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::hamming_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void hamming(vector_view<in_type_> a, vector_view<in_type_> b, std::size_t d, result_type_ *r) noexcept {
    hamming<in_type_, result_type_, allow_simd_>(a.data(), b.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::jaccard_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k, std::size_t max_rank_a_, std::size_t max_rank_b_>
void jaccard(tensor_view<in_type_, max_rank_a_> a, tensor_view<in_type_, max_rank_b_> b, std::size_t d,
             result_type_ *r) noexcept {
    jaccard<in_type_, result_type_, allow_simd_>(a.data(), b.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::jaccard_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void jaccard(vector_view<in_type_> a, vector_view<in_type_> b, std::size_t d, result_type_ *r) noexcept {
    jaccard<in_type_, result_type_, allow_simd_>(a.data(), b.data(), d, r);
}

} // namespace ashvardanian::numkong

#endif // NK_SET_HPP
