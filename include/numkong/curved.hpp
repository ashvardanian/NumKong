/**
 *  @file include/numkong/curved.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief Curved-space kernels: bilinear, mahalanobis.
 */
#ifndef NK_CURVED_HPP
#define NK_CURVED_HPP

#include <cstdint>     // `std::uint32_t`
#include <type_traits> // `std::is_same_v`

#include "numkong/curved.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Bilinear form: aᵀ × C × b where C is a d×d matrix (row-major)
 *  @param[in] a,b Input vectors of length d
 *  @param[in] c Matrix of size dxd (row-major)
 *  @param[in] d Number of dimensions
 *  @param[out] r Pointer to output value
 *
 *  @tparam in_type_ Input vector element type (real or complex)
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::curved_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 *
 *  @note For weighted inner products, Mahalanobis distance, etc.
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void bilinear(in_type_ const *a, in_type_ const *b, in_type_ const *c, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::curved_result_t>;

    // Real types
    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_bilinear_f64(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_bilinear_f32(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_bilinear_f16(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_bilinear_bf16(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    // Complex types
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd)
        nk_bilinear_f64c(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32c_t> && simd)
        nk_bilinear_f32c(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16c_t> && simd)
        nk_bilinear_f16c(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16c_t> && simd)
        nk_bilinear_bf16c(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < d; i++) {
            for (std::size_t j = 0; j < d; j++) {
                sum = sum + result_type_(a[i]) * result_type_(c[i * d + j]) * result_type_(b[j]);
            }
        }
        *r = sum;
    }
}

/**
 *  @brief Mahalanobis distance: √((a−b)ᵀ × C × (a−b)) where C is a d×d matrix (row-major)
 *  @param[in] a,b Input vectors of length d
 *  @param[in] c Covariance matrix of size dxd (row-major)
 *  @param[in] d Number of dimensions
 *  @param[out] r Pointer to output distance value
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::curved_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void mahalanobis(in_type_ const *a, in_type_ const *b, in_type_ const *c, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::curved_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_mahalanobis_f64(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_mahalanobis_f32(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_mahalanobis_f16(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_mahalanobis_bf16(&a->raw_, &b->raw_, &c->raw_, d, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < d; i++) {
            result_type_ di = result_type_(a[i]) - result_type_(b[i]);
            for (std::size_t j = 0; j < d; j++) {
                result_type_ dj = result_type_(a[j]) - result_type_(b[j]);
                sum = sum + di * result_type_(c[i * d + j]) * dj;
            }
        }
        *r = sum.sqrt();
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k, std::size_t max_rank_a_, std::size_t max_rank_b_,
          std::size_t max_rank_c_>
void bilinear(tensor_view<in_type_, max_rank_a_> a, tensor_view<in_type_, max_rank_b_> b,
              tensor_view<in_type_, max_rank_c_> c, std::size_t d, result_type_ *r) noexcept {
    bilinear<in_type_, result_type_, allow_simd_>(a.data(), b.data(), c.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void bilinear(vector_view<in_type_> a, vector_view<in_type_> b, vector_view<in_type_> c, std::size_t d,
              result_type_ *r) noexcept {
    bilinear<in_type_, result_type_, allow_simd_>(a.data(), b.data(), c.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k, std::size_t max_rank_a_, std::size_t max_rank_b_,
          std::size_t max_rank_c_>
void mahalanobis(tensor_view<in_type_, max_rank_a_> a, tensor_view<in_type_, max_rank_b_> b,
                 tensor_view<in_type_, max_rank_c_> c, std::size_t d, result_type_ *r) noexcept {
    mahalanobis<in_type_, result_type_, allow_simd_>(a.data(), b.data(), c.data(), d, r);
}

template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::curved_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void mahalanobis(vector_view<in_type_> a, vector_view<in_type_> b, vector_view<in_type_> c, std::size_t d,
                 result_type_ *r) noexcept {
    mahalanobis<in_type_, result_type_, allow_simd_>(a.data(), b.data(), c.data(), d, r);
}

} // namespace ashvardanian::numkong

#endif // NK_CURVED_HPP
