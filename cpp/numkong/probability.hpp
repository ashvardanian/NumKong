/**
 *  @file cpp/numkong/probability.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ wrappers for SIMD-accelerated similarity measures for probability distributions.
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_PROBABILITY_HPP
#define NUMKONG_PROBABILITY_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/probability.h"

#include "numkong/tensor.hpp"
#include "numkong/types.hpp"
#include "numkong/devices.hpp" // `default_capabilities`

namespace ashvardanian::numkong {

/**
 *  @brief Kullback-Leibler divergence: Σ pᵢ × log(pᵢ / qᵢ)
 *  @param[in] p First probability distribution
 *  @param[in] q Second probability distribution
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] r Pointer to output divergence value
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input distribution type (probability vectors)
 */
template <a_numeric_dtype in_type_>
status_t kld(in_type_ const *p, in_type_ const *q, std::size_t d, typename in_type_::probability_result_t *r,
             nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_kld_f64_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_kld_f32_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_kld_f16_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_kld_bf16_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_kld has no kernel for this dtype");
}

/**
 *  @brief Jensen-Shannon distance: √(½ × (KL(p‖m) + KL(q‖m))), where m = (p + q) / 2
 *  @param[in] p First probability distribution
 *  @param[in] q Second probability distribution
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input distribution type (probability vectors)
 */
template <a_numeric_dtype in_type_>
status_t jsd(in_type_ const *p, in_type_ const *q, std::size_t d, typename in_type_::probability_result_t *r,
             nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_jsd_f64_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_jsd_f32_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_jsd_f16_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_jsd_bf16_best(&p->raw_, &q->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_jsd has no kernel for this dtype");
}

/** Kullback-Leibler divergence of two distributions with equal dimensions;
 *  @c unexpected_dimensions_k when they differ or either run is strided or ends mid-value. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_>
expected<typename in_type_::probability_result_t> kld(a_type_ const &a, b_type_ const &b,
                                                      nk_capability_t capabilities = default_capabilities(),
                                                      nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::probability_result_t result {};
    status_t status = kld<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result, capabilities,
                                    stream);
    return {result, status};
}

/** Jensen-Shannon divergence of two distributions, failing like the concept @c kld. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_>
expected<typename in_type_::probability_result_t> jsd(a_type_ const &a, b_type_ const &b,
                                                      nk_capability_t capabilities = default_capabilities(),
                                                      nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::probability_result_t result {};
    status_t status = jsd<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result, capabilities,
                                    stream);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_PROBABILITY_HPP
