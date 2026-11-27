/**
 *  @file cpp/numkong/spatial.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ wrappers for SIMD-accelerated spatial similarity measures.
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_SPATIAL_HPP
#define NUMKONG_SPATIAL_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/spatial.h"

#include "numkong/tensor.hpp"
#include "numkong/types.hpp"
#include "numkong/devices.hpp" // `default_capabilities`

namespace ashvardanian::numkong {

/**
 *  @brief L₂ (Euclidean) distance: √Σ(aᵢ − bᵢ)²
 *  @param[in] a First vector
 *  @param[in] b Second vector
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 */
template <a_numeric_dtype in_type_>
status_t euclidean(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::euclidean_result_t *r,
                   nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_euclidean_f64_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_euclidean_f32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_euclidean_f16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_euclidean_bf16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_euclidean_e4m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(nk_euclidean_e5m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(nk_euclidean_e2m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(nk_euclidean_e3m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_euclidean_i8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_euclidean_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_euclidean_i4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_euclidean_u4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_euclidean has no kernel for this dtype");
}

/**
 *  @brief Squared L₂ distance: Σ(aᵢ − bᵢ)²
 *  @param[in] a First vector
 *  @param[in] b Second vector
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 */
template <a_numeric_dtype in_type_>
status_t sqeuclidean(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::sqeuclidean_result_t *r,
                     nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_sqeuclidean_f64_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_sqeuclidean_f32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_sqeuclidean_f16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_sqeuclidean_bf16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_sqeuclidean_e4m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(nk_sqeuclidean_e5m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(nk_sqeuclidean_e2m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(nk_sqeuclidean_e3m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_sqeuclidean_i8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_sqeuclidean_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_sqeuclidean_i4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_sqeuclidean_u4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_sqeuclidean has no kernel for this dtype");
}

/**
 *  @brief Angular similarity (cosine): ⟨a,b⟩ / (‖a‖ × ‖b‖)
 *  @param[in] a First vector
 *  @param[in] b Second vector
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 */
template <a_numeric_dtype in_type_>
status_t angular(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::angular_result_t *r,
                 nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_angular_f64_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_angular_f32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_angular_f16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_angular_bf16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_angular_e4m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(nk_angular_e5m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(nk_angular_e2m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(nk_angular_e3m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_angular_i8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_angular_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_angular_i4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_angular_u4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_angular has no kernel for this dtype");
}

/** Euclidean distance between two runs of @p in_type_ with equal dimensions;
 *  @c unexpected_dimensions_k when they differ or either run is strided or ends mid-value. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_>
expected<typename in_type_::euclidean_result_t> euclidean(a_type_ const &a, b_type_ const &b,
                                                          nk_capability_t capabilities = default_capabilities(),
                                                          nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::euclidean_result_t result {};
    status_t status = euclidean<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result,
                                          capabilities, stream);
    return {result, status};
}

/** Squared Euclidean distance between two runs, failing like the concept @c euclidean. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_>
expected<typename in_type_::sqeuclidean_result_t> sqeuclidean(a_type_ const &a, b_type_ const &b,
                                                              nk_capability_t capabilities = default_capabilities(),
                                                              nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::sqeuclidean_result_t result {};
    status_t status = sqeuclidean<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result,
                                            capabilities, stream);
    return {result, status};
}

/** Angular distance between two runs, failing like the concept @c euclidean. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_>
expected<typename in_type_::angular_result_t> angular(a_type_ const &a, b_type_ const &b,
                                                      nk_capability_t capabilities = default_capabilities(),
                                                      nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::angular_result_t result {};
    status_t status = angular<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result, capabilities,
                                        stream);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_SPATIAL_HPP
