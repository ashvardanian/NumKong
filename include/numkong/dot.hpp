/**
 *  @file include/numkong/dot.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for dot-product kernels: ⟨a,b⟩ = Σ aᵢ × bᵢ
 */
#ifndef NK_DOT_HPP
#define NK_DOT_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/dot.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void dot(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_dot_f32(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_dot_f64(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_dot_f16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_dot_bf16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) nk_dot_e4m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) nk_dot_e5m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd) nk_dot_e2m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && simd) nk_dot_e2m1(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd) nk_dot_e3m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_dot_i8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_dot_u8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32c_t> && simd) nk_dot_f32c(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd) nk_dot_f64c(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) nk_dot_i4(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) nk_dot_u4(&a->raw_, &b->raw_, d, &r->raw_);
    else {
        result_type_ sum {};
        std::size_t n = d / dimensions_per_value<in_type_>();
        for (std::size_t i = 0; i < n; i++) sum = fma(a[i], b[i], sum);
        *r = sum;
    }
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void vdot(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if constexpr (std::is_same_v<in_type_, f32c_t> && simd) nk_vdot_f32c(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd) nk_vdot_f64c(&a->raw_, &b->raw_, d, &r->raw_);
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < d; i++) sum = fcma(a[i], b[i], sum);
        *r = sum;
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k, std::size_t max_rank_a_, std::size_t max_rank_b_>
void dot(tensor_view<in_type_, max_rank_a_> a, tensor_view<in_type_, max_rank_b_> b, std::size_t d,
         result_type_ *r) noexcept {
    dot<in_type_, result_type_, allow_simd_>(a.data(), b.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void dot(vector_view<in_type_> a, vector_view<in_type_> b, std::size_t d, result_type_ *r) noexcept {
    dot<in_type_, result_type_, allow_simd_>(a.data(), b.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k, std::size_t max_rank_a_, std::size_t max_rank_b_>
void vdot(tensor_view<in_type_, max_rank_a_> a, tensor_view<in_type_, max_rank_b_> b, std::size_t d,
          result_type_ *r) noexcept {
    vdot<in_type_, result_type_, allow_simd_>(a.data(), b.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void vdot(vector_view<in_type_> a, vector_view<in_type_> b, std::size_t d, result_type_ *r) noexcept {
    vdot<in_type_, result_type_, allow_simd_>(a.data(), b.data(), d, r);
}

} // namespace ashvardanian::numkong

#endif // NK_DOT_HPP
