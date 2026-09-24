/**
 *  @file include/numkong/probability.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ wrappers for SIMD-accelerated similarity measures for probability distributions.
 */
#ifndef NK_PROBABILITY_HPP
#define NK_PROBABILITY_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/probability.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Kullback-Leibler divergence: Σ pᵢ × log(pᵢ / qᵢ)
 *  @param[in] p,q First and second probability distributions
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] r Pointer to output divergence value
 *
 *  @tparam in_type_ Input distribution type (probability vectors)
 *  @tparam result_type_ Result type, defaults to @c in_type_::probability_result_t
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when @c prefer_simd_k
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void kld(in_type_ const *p, in_type_ const *q, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::probability_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_kld_f64(&p->raw_, &q->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_kld_f32(&p->raw_, &q->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_kld_f16(&p->raw_, &q->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_kld_bf16(&p->raw_, &q->raw_, d, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < d; i++) {
            result_type_ pi(p[i]), qi(q[i]);
            if (pi > result_type_(0)) sum = sum + pi * (pi / qi).log();
        }
        *r = sum;
    }
}

/**
 *  @brief Jensen-Shannon distance: √(½ × (KL(p‖m) + KL(q‖m))), where m = (p + q) / 2
 *  @param[in] p,q First and second probability distributions
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] r Pointer to output distance value
 *
 *  @tparam in_type_ Input distribution type (probability vectors)
 *  @tparam result_type_ Result type, defaults to @c in_type_::probability_result_t
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when @c prefer_simd_k
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void jsd(in_type_ const *p, in_type_ const *q, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::probability_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_jsd_f64(&p->raw_, &q->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_jsd_f32(&p->raw_, &q->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_jsd_f16(&p->raw_, &q->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_jsd_bf16(&p->raw_, &q->raw_, d, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        result_type_ half(0.5);
        for (std::size_t i = 0; i < d; i++) {
            result_type_ pi(p[i]), qi(q[i]);
            result_type_ mi = half * (pi + qi);
            if (pi > result_type_(0)) sum = sum + pi * (pi / mi).log();
            if (qi > result_type_(0)) sum = sum + qi * (qi / mi).log();
        }
        // JSD distance = sqrt(divergence / 2), clamped to non-negative
        result_type_ divergence = half * sum;
        *r = divergence > result_type_(0) ? divergence.sqrt() : result_type_(0);
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k, std::size_t max_rank_a_, std::size_t max_rank_b_>
void kld(tensor_view<in_type_, max_rank_a_> p, tensor_view<in_type_, max_rank_b_> q, std::size_t d,
         result_type_ *r) noexcept {
    kld<in_type_, result_type_, allow_simd_>(p.data(), q.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void kld(vector_view<in_type_> p, vector_view<in_type_> q, std::size_t d, result_type_ *r) noexcept {
    kld<in_type_, result_type_, allow_simd_>(p.data(), q.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k, std::size_t max_rank_a_, std::size_t max_rank_b_>
void jsd(tensor_view<in_type_, max_rank_a_> p, tensor_view<in_type_, max_rank_b_> q, std::size_t d,
         result_type_ *r) noexcept {
    jsd<in_type_, result_type_, allow_simd_>(p.data(), q.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void jsd(vector_view<in_type_> p, vector_view<in_type_> q, std::size_t d, result_type_ *r) noexcept {
    jsd<in_type_, result_type_, allow_simd_>(p.data(), q.data(), d, r);
}

} // namespace ashvardanian::numkong

#endif // NK_PROBABILITY_HPP
