/**
 *  @file cpp/numkong/curved.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief Curved-space kernels: bilinear, mahalanobis.
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_CURVED_HPP
#define NUMKONG_CURVED_HPP

#include <cstdint>     // `std::uint32_t`
#include <type_traits> // `std::is_same_v`

#include "numkong/curved.h"

#include "numkong/tensor.hpp"
#include "numkong/types.hpp"
#include "numkong/devices.hpp" // `default_capabilities`

namespace ashvardanian::numkong {

/**
 *  @brief Bilinear form: aᵀ × C × b where C is a d × d matrix (row-major)
 *  @param[in] a First input vector of length d
 *  @param[in] b Second input vector of length d
 *  @param[in] c Matrix of size dxd (row-major)
 *  @param[in] d Number of dimensions
 *  @param[out] r Pointer to output value
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type (real or complex)
 *
 *  @note For weighted inner products, Mahalanobis distance, etc.
 */
template <a_numeric_dtype in_type_>
status_t bilinear(in_type_ const *a, in_type_ const *b, in_type_ const *c, std::size_t d,
                  typename in_type_::curved_result_t *r, nk_capability_t capabilities = default_capabilities(),
                  nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(
            nk_bilinear_f64_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(
            nk_bilinear_f32_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(
            nk_bilinear_f16_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(
            nk_bilinear_bf16_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f64c_t>)
        return static_cast<status_t>(
            nk_bilinear_f64c_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32c_t>)
        return static_cast<status_t>(
            nk_bilinear_f32c_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16c_t>)
        return static_cast<status_t>(
            nk_bilinear_f16c_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16c_t>)
        return static_cast<status_t>(
            nk_bilinear_bf16c_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_bilinear has no kernel for this dtype");
}

/**
 *  @brief Mahalanobis distance: √((a−b)ᵀ × C × (a−b)) where C is a d × d matrix (row-major)
 *  @param[in] a First input vector of length d
 *  @param[in] b Second input vector of length d
 *  @param[in] c Covariance matrix of size dxd (row-major)
 *  @param[in] d Number of dimensions
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 */
template <a_numeric_dtype in_type_>
status_t mahalanobis(in_type_ const *a, in_type_ const *b, in_type_ const *c, std::size_t d,
                     typename in_type_::curved_result_t *r, nk_capability_t capabilities = default_capabilities(),
                     nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(
            nk_mahalanobis_f64_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(
            nk_mahalanobis_f32_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(
            nk_mahalanobis_f16_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(
            nk_mahalanobis_bf16_best(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_mahalanobis has no kernel for this dtype");
}

/** aᵀ × C × b over runs of @p in_type_, with @p c holding the row-major square matrix;
 *  @c unexpected_dimensions_k when the sizes disagree or any run is strided or ends mid-value. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_,
          a_vector_of<in_type_> c_type_>
expected<typename in_type_::curved_result_t> bilinear(a_type_ const &a, b_type_ const &b, c_type_ const &c,
                                                      nk_capability_t capabilities = default_capabilities(),
                                                      nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    auto c_values = contiguous_values_<in_type_ const>(c);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || !c_values || b_values.value.size() * dimensions_per_value<in_type_>() != dimensions ||
        c_values.value.size() * dimensions_per_value<in_type_>() != dimensions * dimensions)
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::curved_result_t result {};
    status_t status = bilinear<in_type_>(a_values.value.data(), b_values.value.data(), c_values.value.data(),
                                         dimensions, &result, capabilities, stream);
    return {result, status};
}

/** Mahalanobis distance of @p a and @p b under the row-major square matrix @p c,
 *  failing like the concept @c bilinear. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_,
          a_vector_of<in_type_> c_type_>
expected<typename in_type_::curved_result_t> mahalanobis(a_type_ const &a, b_type_ const &b, c_type_ const &c,
                                                         nk_capability_t capabilities = default_capabilities(),
                                                         nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    auto c_values = contiguous_values_<in_type_ const>(c);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || !c_values || b_values.value.size() * dimensions_per_value<in_type_>() != dimensions ||
        c_values.value.size() * dimensions_per_value<in_type_>() != dimensions * dimensions)
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::curved_result_t result {};
    status_t status = mahalanobis<in_type_>(a_values.value.data(), b_values.value.data(), c_values.value.data(),
                                            dimensions, &result, capabilities, stream);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_CURVED_HPP
