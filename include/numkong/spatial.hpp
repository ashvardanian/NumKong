/**
 *  @file include/numkong/spatial.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ wrappers for SIMD-accelerated spatial similarity measures.
 */
#ifndef NUMKONG_SPATIAL_HPP
#define NUMKONG_SPATIAL_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/spatial.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief L₂ (Euclidean) distance: √Σ(aᵢ − bᵢ)²
 *  @param[in] a,b First and second vectors
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type, defaults to @c in_type_::euclidean_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::euclidean_result_t>
status_t euclidean(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r,
                   nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::euclidean_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_f64_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_f32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_f16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_bf16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_e4m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_e5m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_e2m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_e3m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_i8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_i4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_euclidean_u4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    }
    result_type_ sum {};
    for (std::size_t i = 0; i < d / dimensions_per_value<in_type_>(); i++) sum = fdsa(a[i], b[i], sum);
    *r = sum.sqrt();
    return status_t::success_k;
}

/**
 *  @brief Squared L₂ distance: Σ(aᵢ − bᵢ)²
 *  @param[in] a,b First and second vectors
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type, defaults to @c in_type_::sqeuclidean_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::sqeuclidean_result_t>
status_t sqeuclidean(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r,
                     nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::sqeuclidean_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_sqeuclidean_f64_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_sqeuclidean_f32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_sqeuclidean_f16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(
                nk_sqeuclidean_bf16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(
                nk_sqeuclidean_e4m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(
                nk_sqeuclidean_e5m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(
                nk_sqeuclidean_e2m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(
                nk_sqeuclidean_e3m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_sqeuclidean_i8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_sqeuclidean_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_sqeuclidean_i4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_sqeuclidean_u4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    }
    result_type_ sum {};
    for (std::size_t i = 0; i < d / dimensions_per_value<in_type_>(); i++) sum = fdsa(a[i], b[i], sum);
    *r = sum;
    return status_t::success_k;
}

/**
 *  @brief Angular similarity (cosine): ⟨a,b⟩ / (‖a‖ × ‖b‖)
 *  @param[in] a,b First and second vectors
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output distance value
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type, defaults to @c in_type_::angular_result_t
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::angular_result_t>
status_t angular(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r,
                 nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::angular_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_angular_f64_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_angular_f32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_angular_f16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_angular_bf16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(nk_angular_e4m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(nk_angular_e5m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(nk_angular_e2m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(nk_angular_e3m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_angular_i8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_angular_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_angular_i4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_angular_u4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    }
    result_type_ ab {}, aa {}, bb {};
    for (std::size_t i = 0; i < d / dimensions_per_value<in_type_>(); i++) {
        ab = fma(a[i], b[i], ab);
        aa = fma(a[i], a[i], aa);
        bb = fma(b[i], b[i], bb);
    }
    result_type_ const zero(0), one(1);
    result_type_ distance = one - ab / (aa.sqrt() * bb.sqrt());
    // Two zero norms give 0, one zero norm or a zero dot 1, and NaN stays, as in `spatial.h`
    if (aa == zero && bb == zero) distance = zero;
    else if (ab == zero || aa == zero || bb == zero) distance = one;
    *r = distance < zero ? zero : distance;
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

/** Euclidean distance between two runs of @p in_type_ with equal dimensions;
 *  @c unexpected_dimensions_k when they differ or either run is strided or ends mid-value. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::euclidean_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_>
expected<result_type_> euclidean(a_type_ const &a, b_type_ const &b,
                                 nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = euclidean<in_type_, result_type_>(a_values.value.data(), b_values.value.data(), dimensions,
                                                        &result, capabilities);
    return {result, status};
}

/** Squared Euclidean distance between two runs, failing like the concept @c euclidean. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::sqeuclidean_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_>
expected<result_type_> sqeuclidean(a_type_ const &a, b_type_ const &b,
                                   nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = sqeuclidean<in_type_, result_type_>(a_values.value.data(), b_values.value.data(), dimensions,
                                                          &result, capabilities);
    return {result, status};
}

/** Angular distance between two runs, failing like the concept @c euclidean. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::angular_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_>
expected<result_type_> angular(a_type_ const &a, b_type_ const &b,
                               nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = angular<in_type_, result_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result,
                                                      capabilities);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_SPATIAL_HPP
