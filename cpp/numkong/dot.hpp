/**
 *  @file cpp/numkong/dot.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for dot-product kernels: ⟨a,b⟩ = Σ aᵢ × bᵢ
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_DOT_HPP
#define NUMKONG_DOT_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/dot.h"

#include "numkong/tensor.hpp"
#include "numkong/types.hpp"
#include "numkong/devices.hpp" // `default_capabilities`

namespace ashvardanian::numkong {

template <a_numeric_dtype in_type_>
status_t dot(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::dot_result_t *r,
             nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_dot_f32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_dot_f64_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_dot_f16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_dot_bf16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_dot_e4m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(nk_dot_e5m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(nk_dot_e2m3_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        return static_cast<status_t>(nk_dot_e2m1_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(nk_dot_e3m2_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_dot_i8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_dot_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32c_t>)
        return static_cast<status_t>(nk_dot_f32c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f64c_t>)
        return static_cast<status_t>(nk_dot_f64c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16c_t>)
        return static_cast<status_t>(nk_dot_f16c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16c_t>)
        return static_cast<status_t>(nk_dot_bf16c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_dot_i4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_dot_u4_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(nk_dot_u1_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_dot has no kernel for this dtype");
}

template <a_numeric_dtype in_type_>
status_t vdot(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::dot_result_t *r,
              nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f32c_t>)
        return static_cast<status_t>(nk_vdot_f32c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f64c_t>)
        return static_cast<status_t>(nk_vdot_f64c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16c_t>)
        return static_cast<status_t>(nk_vdot_f16c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16c_t>)
        return static_cast<status_t>(nk_vdot_bf16c_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_vdot has no kernel for this dtype");
}

/** ⟨a,b⟩ over two runs of @p in_type_ with equal dimensions; @c unexpected_dimensions_k when they
 *  differ or either run is strided or ends mid-value. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_>
expected<typename in_type_::dot_result_t> dot(a_type_ const &a, b_type_ const &b,
                                              nk_capability_t capabilities = default_capabilities(),
                                              nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::dot_result_t result {};
    status_t status = dot<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result, capabilities,
                                    stream);
    return {result, status};
}

/** Conjugated ⟨a,b⟩ over two complex runs, failing like the concept @c dot. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_>
expected<typename in_type_::dot_result_t> vdot(a_type_ const &a, b_type_ const &b,
                                               nk_capability_t capabilities = default_capabilities(),
                                               nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::dot_result_t result {};
    status_t status = vdot<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result, capabilities,
                                     stream);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_DOT_HPP
