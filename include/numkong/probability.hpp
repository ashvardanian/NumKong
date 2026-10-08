/**
 *  @file include/numkong/probability.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ wrappers for SIMD-accelerated similarity measures for probability distributions.
 */
#ifndef NUMKONG_PROBABILITY_HPP
#define NUMKONG_PROBABILITY_HPP

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
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input distribution type (probability vectors)
 *  @tparam result_type_ Result type, defaults to @c in_type_::probability_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t>
status_t kld(in_type_ const *p, in_type_ const *q, std::size_t d, result_type_ *r,
             nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::probability_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_kld_f64_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_kld_f32_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_kld_f16_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_kld_bf16_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    }
    result_type_ sum {};
    for (std::size_t i = 0; i < d; i++) {
        result_type_ pi(p[i]), qi(q[i]);
        if (pi > result_type_(0)) sum = sum + pi * (pi / qi).log();
    }
    *r = sum;
    return status_t::success_k;
}

/**
 *  @brief Jensen-Shannon distance: √(½ × (KL(p‖m) + KL(q‖m))), where m = (p + q) / 2
 *  @param[in] p,q First and second probability distributions
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input distribution type (probability vectors)
 *  @tparam result_type_ Result type, defaults to @c in_type_::probability_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t>
status_t jsd(in_type_ const *p, in_type_ const *q, std::size_t d, result_type_ *r,
             nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::probability_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_jsd_f64_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_jsd_f32_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_jsd_f16_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_jsd_bf16_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    }
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
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

/** Kullback-Leibler divergence of two distributions with equal dimensions;
 *  @c unexpected_dimensions_k when they differ or either run is strided or ends mid-value. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_>
expected<result_type_> kld(a_type_ const &a, b_type_ const &b,
                           nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = kld<in_type_, result_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result,
                                                  capabilities);
    return {result, status};
}

/** Jensen-Shannon divergence of two distributions, failing like the concept @c kld. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::probability_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_>
expected<result_type_> jsd(a_type_ const &a, b_type_ const &b,
                           nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = jsd<in_type_, result_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result,
                                                  capabilities);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_PROBABILITY_HPP
