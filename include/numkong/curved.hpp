/**
 *  @file include/numkong/curved.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief Curved-space kernels: bilinear, mahalanobis.
 */
#ifndef NUMKONG_CURVED_HPP
#define NUMKONG_CURVED_HPP

#include <cstdint>     // `std::uint32_t`
#include <type_traits> // `std::is_same_v`

#include "numkong/curved.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Bilinear form: aᵀ × C × b where C is a d × d matrix (row-major)
 *  @param[in] a,b Input vectors of length d
 *  @param[in] c Matrix of size dxd (row-major)
 *  @param[in] d Number of dimensions
 *  @param[out] r Pointer to output value
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type (real or complex)
 *  @tparam result_type_ Accumulator type, defaults to @c in_type_::curved_result_t
 *
 *  @note For weighted inner products, Mahalanobis distance, etc.
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t>
status_t bilinear(in_type_ const *a, in_type_ const *b, in_type_ const *c, std::size_t d, result_type_ *r,
                  nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::curved_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_bilinear_f64_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_bilinear_f32_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_bilinear_f16_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(
                nk_bilinear_bf16_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f64c_t> && dispatch)
            return static_cast<status_t>(
                nk_bilinear_f64c_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32c_t> && dispatch)
            return static_cast<status_t>(
                nk_bilinear_f32c_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16c_t> && dispatch)
            return static_cast<status_t>(
                nk_bilinear_f16c_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16c_t> && dispatch)
            return static_cast<status_t>(
                nk_bilinear_bf16c_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    }
    result_type_ sum {};
    for (std::size_t i = 0; i < d; i++) {
        for (std::size_t j = 0; j < d; j++) {
            sum = sum + result_type_(a[i]) * result_type_(c[i * d + j]) * result_type_(b[j]);
        }
    }
    *r = sum;
    return status_t::success_k;
}

/**
 *  @brief Mahalanobis distance: √((a−b)ᵀ × C × (a−b)) where C is a d × d matrix (row-major)
 *  @param[in] a,b Input vectors of length d
 *  @param[in] c Covariance matrix of size dxd (row-major)
 *  @param[in] d Number of dimensions
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type, defaults to @c in_type_::curved_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t>
status_t mahalanobis(in_type_ const *a, in_type_ const *b, in_type_ const *c, std::size_t d, result_type_ *r,
                     nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::curved_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_mahalanobis_f64_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_mahalanobis_f32_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_mahalanobis_f16_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(
                nk_mahalanobis_bf16_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    }
    result_type_ sum {};
    for (std::size_t i = 0; i < d; i++) {
        result_type_ di = result_type_(a[i]) - result_type_(b[i]);
        for (std::size_t j = 0; j < d; j++) {
            result_type_ dj = result_type_(a[j]) - result_type_(b[j]);
            sum = sum + di * result_type_(c[i * d + j]) * dj;
        }
    }
    *r = sum.sqrt();
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

/** aᵀ × C × b over runs of @p in_type_, with @p c holding the row-major square matrix;
 *  @c unexpected_dimensions_k when the sizes disagree or any run is strided or ends mid-value. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_, vector_of<in_type_> c_type_>
expected<result_type_> bilinear(a_type_ const &a, b_type_ const &b, c_type_ const &c,
                                nk_capability_t capabilities = cpu_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    auto c_values = contiguous_values_<in_type_ const>(c);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || !c_values || b_values.value.size() * dimensions_per_value<in_type_>() != dimensions ||
        c_values.value.size() * dimensions_per_value<in_type_>() != dimensions * dimensions)
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = bilinear<in_type_, result_type_>(a_values.value.data(), b_values.value.data(),
                                                       c_values.value.data(), dimensions, &result, capabilities);
    return {result, status};
}

/** Mahalanobis distance of @p a and @p b under the row-major square matrix @p c,
 *  failing like the concept @c bilinear. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_, vector_of<in_type_> c_type_>
expected<result_type_> mahalanobis(a_type_ const &a, b_type_ const &b, c_type_ const &c,
                                   nk_capability_t capabilities = cpu_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    auto c_values = contiguous_values_<in_type_ const>(c);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || !c_values || b_values.value.size() * dimensions_per_value<in_type_>() != dimensions ||
        c_values.value.size() * dimensions_per_value<in_type_>() != dimensions * dimensions)
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = mahalanobis<in_type_, result_type_>(a_values.value.data(), b_values.value.data(),
                                                          c_values.value.data(), dimensions, &result, capabilities);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_CURVED_HPP
