/**
 *  @file include/numkong/dot.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for dot-product kernels: ⟨a,b⟩ = Σ aᵢ × bᵢ
 */
#ifndef NUMKONG_DOT_HPP
#define NUMKONG_DOT_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/dot.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t>
status_t dot(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r,
             nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_dot_f32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_dot_f64_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_dot_f16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_dot_bf16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(nk_dot_e4m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(nk_dot_e5m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(nk_dot_e2m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
            return static_cast<status_t>(nk_dot_e2m1_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(nk_dot_e3m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_dot_i8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_dot_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32c_t> && dispatch)
            return static_cast<status_t>(nk_dot_f32c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f64c_t> && dispatch)
            return static_cast<status_t>(nk_dot_f64c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16c_t> && dispatch)
            return static_cast<status_t>(nk_dot_f16c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16c_t> && dispatch)
            return static_cast<status_t>(nk_dot_bf16c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_dot_i4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_dot_u4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
            return static_cast<status_t>(nk_dot_u1_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    }
    result_type_ sum {};
    std::size_t n = d / dimensions_per_value<in_type_>();
    for (std::size_t i = 0; i < n; i++) sum = fma(a[i], b[i], sum);
    *r = sum;
    return status_t::success_k;
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t>
status_t vdot(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r,
              nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f32c_t> && dispatch)
            return static_cast<status_t>(nk_vdot_f32c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f64c_t> && dispatch)
            return static_cast<status_t>(nk_vdot_f64c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16c_t> && dispatch)
            return static_cast<status_t>(nk_vdot_f16c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16c_t> && dispatch)
            return static_cast<status_t>(nk_vdot_bf16c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    }
    result_type_ sum {};
    for (std::size_t i = 0; i < d; i++) sum = fcma(a[i], b[i], sum);
    *r = sum;
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

/** ⟨a,b⟩ over two runs of @p in_type_ with equal dimensions; @c unexpected_dimensions_k when they
 *  differ or either run is strided or ends mid-value. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_>
expected<result_type_> dot(a_type_ const &a, b_type_ const &b,
                           nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = dot<in_type_, result_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result,
                                                  capabilities);
    return {result, status};
}

/** Conjugated ⟨a,b⟩ over two complex runs, failing like the concept @c dot. */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          vector_of<in_type_> a_type_, vector_of<in_type_> b_type_>
expected<result_type_> vdot(a_type_ const &a, b_type_ const &b,
                            nk_capability_t capabilities = default_capabilities()) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    result_type_ result {};
    status_t status = vdot<in_type_, result_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result,
                                                   capabilities);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_DOT_HPP
