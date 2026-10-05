/**
 *  @file include/numkong/each.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ wrappers for SIMD-accelerated elementwise arithmetic.
 */
#ifndef NUMKONG_EACH_HPP
#define NUMKONG_EACH_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/each.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Elementwise addition: cᵢ = aᵢ + bᵢ
 *  @param[in] a,b Input vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] c Output vector
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 */
template <numeric_dtype in_type_>
status_t add(in_type_ const *a, in_type_ const *b, std::size_t d, in_type_ *c,
             nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t>)
            return static_cast<status_t>(nk_each_sum_f64_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t>)
            return static_cast<status_t>(nk_each_sum_f32_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t>)
            return static_cast<status_t>(nk_each_sum_f16_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_each_sum_bf16_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t>)
            return static_cast<status_t>(nk_each_sum_i8_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t>)
            return static_cast<status_t>(nk_each_sum_u8_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i16_t>)
            return static_cast<status_t>(nk_each_sum_i16_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u16_t>)
            return static_cast<status_t>(nk_each_sum_u16_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i32_t>)
            return static_cast<status_t>(nk_each_sum_i32_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u32_t>)
            return static_cast<status_t>(nk_each_sum_u32_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i64_t>)
            return static_cast<status_t>(nk_each_sum_i64_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u64_t>)
            return static_cast<status_t>(nk_each_sum_u64_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32c_t>)
            return static_cast<status_t>(nk_each_sum_f32c_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f64c_t>)
            return static_cast<status_t>(nk_each_sum_f64c_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_each_sum_e4m3_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t>)
            return static_cast<status_t>(nk_each_sum_e5m2_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t>)
            return static_cast<status_t>(nk_each_sum_e2m3_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t>)
            return static_cast<status_t>(nk_each_sum_e3m2_best(&a->raw_, &b->raw_, d, &c->raw_, capabilities, stream));
    }
    for (std::size_t i = 0; i < d; i++) c[i] = saturating_add(a[i], b[i]);
    return status_t::success_k;
}

/**
 *  @brief Elementwise scale: cᵢ = α × aᵢ + β
 *  @param[in] a Input vector
 *  @param[in] d Number of dimensions in input vector
 *  @param[in] alpha,beta Scale and shift coefficients
 *  @param[out] c Output vector
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 *  @tparam precision_type_ Precision type for scalar fallback computations, defaults to @c in_type_
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_>
status_t scale(in_type_ const *a, std::size_t d, typename in_type_::scale_t const *alpha,
               typename in_type_::scale_t const *beta, in_type_ *c,
               nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<precision_type_, in_type_>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_f64_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_f32_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_f16_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_bf16_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_i8_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_u8_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_i16_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_u16_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i32_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_i32_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u32_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_u32_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i64_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_i64_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u64_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_u64_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32c_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_f32c_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f64c_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_f64c_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_e4m3_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_e5m2_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_e2m3_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(
                nk_each_scale_e3m2_best(&a->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
    }
    // Scalar fallback with high-precision intermediates
    for (std::size_t i = 0; i < d; i++) {
        precision_type_ const result = precision_type_(a[i]) * precision_type_(*alpha) + precision_type_(*beta);
        if constexpr (dispatch) c[i] = result;
        else c[i] = result.template to<in_type_>();
    }
    return status_t::success_k;
}

/**
 *  @brief Blend: cᵢ = α × aᵢ + β × bᵢ
 *  @param[in] a,b Input vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[in] alpha,beta Weight coefficients
 *  @param[out] c Output vector
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 *  @tparam precision_type_ Precision type for scalar fallback computations, defaults to @c in_type_
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_>
status_t blend(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::scale_t const *alpha,
               typename in_type_::scale_t const *beta, in_type_ *c,
               nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<precision_type_, in_type_>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_f64_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_f32_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_f16_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_bf16_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_i8_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_u8_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_i16_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_u16_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i32_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_i32_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u32_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_u32_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i64_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_i64_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u64_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_u64_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32c_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_f32c_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f64c_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_f64c_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_e4m3_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_e5m2_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_e2m3_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(
                nk_each_blend_e3m2_best(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_, capabilities, stream));
    }
    // Scalar fallback with high-precision intermediates
    for (std::size_t i = 0; i < d; i++) {
        precision_type_ const result = precision_type_(a[i]) * precision_type_(*alpha) +
                                       precision_type_(b[i]) * precision_type_(*beta);
        if constexpr (dispatch) c[i] = result;
        else c[i] = result.template to<in_type_>();
    }
    return status_t::success_k;
}

/**
 *  @brief Elementwise FMA: outᵢ = α × aᵢ × bᵢ + β × cᵢ
 *  @param[in] a,b,c Input vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[in] alpha,beta Coefficients
 *  @param[out] out Output vector
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 *  @tparam precision_type_ Precision type for scalar fallback computations, defaults to @c in_type_
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_>
status_t fma(in_type_ const *a, in_type_ const *b, in_type_ const *c, std::size_t d,
             typename in_type_::scale_t const *alpha, typename in_type_::scale_t const *beta, in_type_ *out,
             nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<precision_type_, in_type_>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_f64_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_f32_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_f16_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_bf16_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_i8_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_u8_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_i16_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u16_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_u16_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i32_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_i32_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u32_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_u32_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i64_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_i64_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u64_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_u64_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32c_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_f32c_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f64c_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_f64c_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_e4m3_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_e5m2_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_e2m3_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(
                nk_each_fma_e3m2_best(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_, capabilities, stream));
    }
    // Scalar fallback with high-precision intermediates
    for (std::size_t i = 0; i < d; i++) {
        precision_type_ const result = precision_type_(a[i]) * precision_type_(b[i]) * precision_type_(*alpha) +
                                       precision_type_(c[i]) * precision_type_(*beta);
        if constexpr (dispatch) out[i] = result;
        else out[i] = result.template to<in_type_>();
    }
    return status_t::success_k;
}

/**
 *  @brief Fused SwiGLU: yᵢ = silu(gateᵢ · @p gate_scale) · upᵢ · @p output_scale.
 *
 *  With a null @p up this reduces to plain SiLU.
 *
 *  @param[in] gate Gate input, shape @b [rows,columns]
 *  @param[in] up Up input, same shape as @p gate; @c nullptr collapses to plain SiLU
 *  @param[out] y Output, same shape and dtype as @p gate; may alias @p gate
 *  @param[in] rows Logical row count
 *  @param[in] columns Logical column count
 *  @param[in] gate_stride Row stride of @p gate in bytes
 *  @param[in] up_stride Row stride of @p up in bytes
 *  @param[in] y_stride Row stride of @p y in bytes
 *  @param[in] gate_scale Scalar applied to each loaded gate element, the E4M3 descale or 1.0
 *  @param[in] output_scale Scalar applied to each result before it is stored
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 */
template <numeric_dtype in_type_>
status_t swiglu(in_type_ const *gate, in_type_ const *up, in_type_ *y, std::size_t rows, std::size_t columns,
                std::size_t gate_stride, std::size_t up_stride, std::size_t y_stride, f32_t gate_scale = 1.0f,
                f32_t output_scale = 1.0f, nk_capability_t capabilities = default_capabilities(),
                void *stream = nullptr) noexcept {
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f32_t>)
            return static_cast<status_t>(
                nk_each_swiglu_f32_best(&gate->raw_, up ? &up->raw_ : nullptr, &y->raw_, rows, columns, gate_stride,
                                        up_stride, y_stride, gate_scale.raw_, output_scale.raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t>)
            return static_cast<status_t>(
                nk_each_swiglu_f16_best(&gate->raw_, up ? &up->raw_ : nullptr, &y->raw_, rows, columns, gate_stride,
                                        up_stride, y_stride, gate_scale.raw_, output_scale.raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_each_swiglu_bf16_best(
                &gate->raw_, up ? &up->raw_ : nullptr, &y->raw_, rows, columns, gate_stride, up_stride, y_stride,
                gate_scale.raw_, output_scale.raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_each_swiglu_e4m3_best(
                &gate->raw_, up ? &up->raw_ : nullptr, &y->raw_, rows, columns, gate_stride, up_stride, y_stride,
                gate_scale.raw_, output_scale.raw_, capabilities, stream));
    }
    // Scalar fallback for other numeric dtypes or a mask of no capability.
    for (std::size_t row = 0; row < rows; ++row) {
        in_type_ const *gate_row = reinterpret_cast<in_type_ const *>(reinterpret_cast<char const *>(gate) +
                                                                      row * gate_stride);
        in_type_ const *up_row =
            up ? reinterpret_cast<in_type_ const *>(reinterpret_cast<char const *>(up) + row * up_stride) : nullptr;
        in_type_ *output_row = reinterpret_cast<in_type_ *>(reinterpret_cast<char *>(y) + row * y_stride);
        // SiLU(g) = g / (1 + exp(-g)), with exp from the type method, like the sin/cos fallbacks.
        for (std::size_t column = 0; column < columns; ++column) {
            float gate_value = static_cast<float>(gate_row[column]) * gate_scale.raw_;
            float result = gate_value / (1.0f + static_cast<float>(f32_t(-gate_value).exp()));
            if (up_row) result *= static_cast<float>(up_row[column]);
            output_row[column] = f32_t(result * output_scale.raw_).template to<in_type_>();
        }
    }
    return status_t::success_k;
}

/**
 *  @brief Grouped RMSNorm: yᵢ = xᵢ · rsqrt(mean(x²) + epsilon) · gammaᵢ
 *
 *  Each row holds @p groups independent @p columns-vectors, normalized separately.
 *
 *  @param[in] x Input matrix, shaped @b [rows,groups,columns], each group packed contiguously
 *  @param[in] gamma Per-column gain, length @p columns, shared by groups; @c nullptr is unit scale.
 *  @param[out] y Output matrix, same shape and dtype as @p x; may alias @p x
 *  @param[in] rows,groups,columns Logical shape
 *  @param[in] x_stride, @p y_stride Row (outer) strides in bytes
 *  @param[in] epsilon Variance epsilon added before the reciprocal square root
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 */
template <numeric_dtype in_type_>
status_t rmsnorm(in_type_ const *x, f32_t const *gamma, in_type_ *y, std::size_t rows, std::size_t groups,
                 std::size_t columns, std::size_t x_stride, std::size_t y_stride, f32_t epsilon,
                 nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    nk_f32_t const *gamma_raw = gamma ? &gamma->raw_ : nullptr;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f32_t>)
            return static_cast<status_t>(nk_each_rmsnorm_f32_best(&x->raw_, gamma_raw, &y->raw_, rows, groups, columns,
                                                                  x_stride, y_stride, epsilon.raw_, capabilities,
                                                                  stream));
        else if constexpr (std::is_same_v<in_type_, f16_t>)
            return static_cast<status_t>(nk_each_rmsnorm_f16_best(&x->raw_, gamma_raw, &y->raw_, rows, groups, columns,
                                                                  x_stride, y_stride, epsilon.raw_, capabilities,
                                                                  stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_each_rmsnorm_bf16_best(&x->raw_, gamma_raw, &y->raw_, rows, groups, columns,
                                                                   x_stride, y_stride, epsilon.raw_, capabilities,
                                                                   stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_each_rmsnorm_e4m3_best(&x->raw_, gamma_raw, &y->raw_, rows, groups, columns,
                                                                   x_stride, y_stride, epsilon.raw_, capabilities,
                                                                   stream));
    }
    // Scalar fallback for other numeric dtypes or a mask of no capability.
    for (std::size_t row = 0; row < rows; ++row) {
        in_type_ const *row_input = reinterpret_cast<in_type_ const *>(reinterpret_cast<char const *>(x) +
                                                                       row * x_stride);
        in_type_ *row_output = reinterpret_cast<in_type_ *>(reinterpret_cast<char *>(y) + row * y_stride);
        for (std::size_t group = 0; group < groups; ++group) {
            in_type_ const *group_input = row_input + group * columns;
            in_type_ *group_output = row_output + group * columns;
            double mean_square = 0;
            for (std::size_t column = 0; column < columns; ++column) {
                float value = static_cast<float>(group_input[column]);
                mean_square += static_cast<double>(value) * static_cast<double>(value);
            }
            float inverse_rms = static_cast<float>(
                f32_t(static_cast<float>(mean_square / static_cast<double>(columns)) + epsilon.raw_).rsqrt());
            for (std::size_t column = 0; column < columns; ++column) {
                float value = static_cast<float>(group_input[column]);
                float gamma_value = gamma ? static_cast<float>(gamma[column]) : 1.0f;
                group_output[column] = f32_t(value * inverse_rms * gamma_value).template to<in_type_>();
            }
        }
    }
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Vector Elementwise

/** Elementwise cᵢ = aᵢ + bᵢ over runs of equal dimensions; @c unexpected_dimensions_k when they
 *  differ or any run is strided or ends mid-value. */
template <numeric_dtype in_type_, vector_of<in_type_> a_type_, vector_of<in_type_> b_type_,
          mutable_vector_of<in_type_> output_type_>
status_t add(a_type_ const &a, b_type_ const &b, output_type_ &&output,
             nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    auto output_values = contiguous_values_<in_type_>(output);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || !output_values ||
        b_values.value.size() * dimensions_per_value<in_type_>() != dimensions ||
        output_values.value.size() * dimensions_per_value<in_type_>() != dimensions)
        return status_t::unexpected_dimensions_k;
    return add<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, output_values.value.data(),
                         capabilities, stream);
}

/** Elementwise cᵢ = α × aᵢ + β, failing like the concept @c add. */
template <numeric_dtype in_type_, vector_of<in_type_> a_type_, mutable_vector_of<in_type_> output_type_>
    requires(!requires(a_type_ const &tensor) { tensor.rank(); }) // tensors take the tensor overload below
status_t scale(a_type_ const &a, typename in_type_::scale_t alpha, typename in_type_::scale_t beta,
               output_type_ &&output, nk_capability_t capabilities = default_capabilities(),
               void *stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto output_values = contiguous_values_<in_type_>(output);
    std::size_t const dimensions = output_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !output_values || dimensions != a_values.value.size() * dimensions_per_value<in_type_>())
        return status_t::unexpected_dimensions_k;
    return scale<in_type_>(a_values.value.data(), a_values.value.size() * dimensions_per_value<in_type_>(), &alpha,
                           &beta, output_values.value.data(), capabilities, stream);
}

/** Elementwise cᵢ = α × aᵢ + β × bᵢ, failing like the concept @c add. */
template <numeric_dtype in_type_, vector_of<in_type_> a_type_, vector_of<in_type_> b_type_,
          mutable_vector_of<in_type_> output_type_>
    requires(!requires(a_type_ const &tensor) { tensor.rank(); })
status_t blend(a_type_ const &a, b_type_ const &b, typename in_type_::scale_t alpha, typename in_type_::scale_t beta,
               output_type_ &&output, nk_capability_t capabilities = default_capabilities(),
               void *stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    auto output_values = contiguous_values_<in_type_>(output);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || !output_values ||
        b_values.value.size() * dimensions_per_value<in_type_>() != dimensions ||
        output_values.value.size() * dimensions_per_value<in_type_>() != dimensions)
        return status_t::unexpected_dimensions_k;
    return blend<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &alpha, &beta,
                           output_values.value.data(), capabilities, stream);
}

/** Elementwise outᵢ = α × aᵢ × bᵢ + β × cᵢ, failing like the concept @c add. */
template <numeric_dtype in_type_, vector_of<in_type_> a_type_, vector_of<in_type_> b_type_, vector_of<in_type_> c_type_,
          mutable_vector_of<in_type_> output_type_>
    requires(!requires(a_type_ const &tensor) { tensor.rank(); })
status_t fma(a_type_ const &a, b_type_ const &b, c_type_ const &c, typename in_type_::scale_t alpha,
             typename in_type_::scale_t beta, output_type_ &&output,
             nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    auto c_values = contiguous_values_<in_type_ const>(c);
    auto output_values = contiguous_values_<in_type_>(output);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || !c_values || !output_values ||
        b_values.value.size() * dimensions_per_value<in_type_>() != dimensions ||
        c_values.value.size() * dimensions_per_value<in_type_>() != dimensions ||
        output_values.value.size() * dimensions_per_value<in_type_>() != dimensions)
        return status_t::unexpected_dimensions_k;
    return fma<in_type_>(a_values.value.data(), b_values.value.data(), c_values.value.data(), dimensions, &alpha, &beta,
                         output_values.value.data(), capabilities, stream);
}

#pragma endregion Vector Elementwise

#pragma region Tensor Elementwise

/** Fused SwiGLU over @b [rows,columns] matrices into a matching output span — @p up empty means
 *  SiLU; @c unexpected_dimensions_k when the shapes disagree. */
template <numeric_dtype value_type_>
status_t swiglu(matrix_view<value_type_> gate, matrix_view<value_type_> up, matrix_span<value_type_> output,
                f32_t gate_scale = 1.0f, f32_t output_scale = 1.0f) noexcept {
    bool const has_up = !up.empty();
    if (gate.extent(0) != output.extent(0) || gate.extent(1) != output.extent(1))
        return status_t::unexpected_dimensions_k;
    if (has_up && (up.extent(0) != gate.extent(0) || up.extent(1) != gate.extent(1)))
        return status_t::unexpected_dimensions_k;
    value_type_ const *up_ptr = has_up ? up.data() : nullptr;
    std::size_t const up_stride = has_up ? static_cast<std::size_t>(up.stride_bytes(0)) : 0;
    return numkong::swiglu<value_type_>(gate.data(), up_ptr, output.data(), gate.extent(0), gate.extent(1),
                                        static_cast<std::size_t>(gate.stride_bytes(0)), up_stride,
                                        static_cast<std::size_t>(output.stride_bytes(0)), gate_scale, output_scale);
}

/** Allocating SwiGLU returning a fresh matrix — @p up empty means SiLU; empty for an empty
 *  @p gate, or the allocation's or the kernel's failure. */
template <numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, 2>> swiglu(matrix_view<value_type_> gate, matrix_view<value_type_> up,
                                                         f32_t gate_scale = 1.0f, f32_t output_scale = 1.0f,
                                                         allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, 2>;
    if (gate.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &gate_shape = gate.shape();
    auto result = out_tensor_t::uninitialized(gate_shape.extents, gate_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = swiglu<value_type_>(gate, up, result.value.span(), gate_scale, output_scale); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Grouped RMSNorm over a matrix of @c rows rows, @p groups groups and @c columns columns, into a
 *  matching output span; @c unexpected_dimensions_k when the shapes, the groups or @p gamma
 *  disagree with each other. */
template <numeric_dtype value_type_>
status_t rmsnorm(matrix_view<value_type_> input, vector_view<f32_t> gamma, matrix_span<value_type_> output,
                 std::size_t groups, f32_t epsilon) noexcept {
    if (input.extent(0) != output.extent(0) || input.extent(1) != output.extent(1))
        return status_t::unexpected_dimensions_k;
    std::size_t const columns_total = input.extent(1);
    if (groups == 0 || columns_total % groups != 0) return status_t::unexpected_dimensions_k;
    if (!gamma.empty() && gamma.size() != columns_total / groups) return status_t::unexpected_dimensions_k;
    f32_t const *gamma_ptr = gamma.empty() ? nullptr : gamma.data();
    return numkong::rmsnorm<value_type_>(input.data(), gamma_ptr, output.data(), input.extent(0), groups,
                                         columns_total / groups, static_cast<std::size_t>(input.stride_bytes(0)),
                                         static_cast<std::size_t>(output.stride_bytes(0)), epsilon);
}

/** Allocating grouped RMSNorm returning a fresh matrix, empty for an empty @p input, or the
 *  allocation's or the kernel's failure. */
template <numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, 2>> rmsnorm(matrix_view<value_type_> input, vector_view<f32_t> gamma,
                                                          std::size_t groups, f32_t epsilon,
                                                          allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, 2>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = rmsnorm<value_type_>(input, gamma, result.value.span(), groups, epsilon); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Scale: output[i] = α × input[i] + β. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t scale(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t alpha,
               typename value_type_::scale_t beta, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [&](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            return numkong::scale<value_type_>(in.data(), in.extent(0), &alpha, &beta, out.data());
        });
}

/** Allocating scale: result[i] = α × input[i] + β; empty for an empty @p input, or the
 *  allocation's or the kernel's failure. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> scale(tensor_view<value_type_, max_rank_> input,
                                                                typename value_type_::scale_t alpha,
                                                                typename value_type_::scale_t beta,
                                                                allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = scale<value_type_, max_rank_>(input, alpha, beta, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Blend: each output is α times the left operand plus β times the right operand. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t blend(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
               typename value_type_::scale_t alpha, typename value_type_::scale_t beta,
               tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        lhs, rhs, output,
        [&](tensor_view<value_type_, max_rank_> l, tensor_view<value_type_, max_rank_> r,
            tensor_span<value_type_, max_rank_> out) {
            return numkong::blend<value_type_>(l.data(), r.data(), l.extent(0), &alpha, &beta, out.data());
        });
}

/** Allocating blend: each result is α times the left operand plus β times the right operand;
 *  @c unexpected_dimensions_k when the operands disagree, empty for empty ones, or the
 *  allocation's or the kernel's failure. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> blend(tensor_view<value_type_, max_rank_> lhs,
                                                                tensor_view<value_type_, max_rank_> rhs,
                                                                typename value_type_::scale_t alpha,
                                                                typename value_type_::scale_t beta,
                                                                allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs)) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (lhs.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = blend<value_type_, max_rank_>(lhs, rhs, alpha, beta, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** FMA: each output is α times the product of both operands plus β times the addend. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t fma(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
             tensor_view<value_type_, max_rank_> addend, typename value_type_::scale_t alpha,
             typename value_type_::scale_t beta, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        lhs, rhs, addend, output,
        [&](tensor_view<value_type_, max_rank_> a, tensor_view<value_type_, max_rank_> b,
            tensor_view<value_type_, max_rank_> c, tensor_span<value_type_, max_rank_> out) {
            return numkong::fma<value_type_>(a.data(), b.data(), c.data(), a.extent(0), &alpha, &beta, out.data());
        });
}

/** Allocating FMA: each result is α times the product of both operands plus β times the addend;
 *  fails like the allocating @c blend. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> fma(tensor_view<value_type_, max_rank_> lhs,
                                                              tensor_view<value_type_, max_rank_> rhs,
                                                              tensor_view<value_type_, max_rank_> addend,
                                                              typename value_type_::scale_t alpha,
                                                              typename value_type_::scale_t beta,
                                                              allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs) || !shapes_match_(lhs, addend))
        return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (lhs.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = fma<value_type_, max_rank_>(lhs, rhs, addend, alpha, beta, result.value.span());
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Elementwise addition: each output is the sum of the two operands. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t add(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
             tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        lhs, rhs, output,
        [](tensor_view<value_type_, max_rank_> l, tensor_view<value_type_, max_rank_> r,
           tensor_span<value_type_, max_rank_> out) {
            return numkong::add<value_type_>(l.data(), r.data(), l.extent(0), out.data());
        });
}

/** Allocating elementwise add: the result is the sum of the two operands; fails like the
 *  allocating @c blend. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> add(tensor_view<value_type_, max_rank_> lhs,
                                                              tensor_view<value_type_, max_rank_> rhs,
                                                              allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs)) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (lhs.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = add<value_type_, max_rank_>(lhs, rhs, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Elementwise add scalar: output[i] = input[i] + scalar. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t add(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t scalar,
             tensor_span<value_type_, max_rank_> output) noexcept {
    typename value_type_::scale_t one {1};
    return scale<value_type_, max_rank_>(input, one, scalar, output);
}

/** Allocating add scalar; fails like the allocating @c scale. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> add(tensor_view<value_type_, max_rank_> input,
                                                              typename value_type_::scale_t scalar,
                                                              allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = add<value_type_, max_rank_>(input, scalar, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Elementwise subtraction: each output is the left operand minus the right operand. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t sub(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
             tensor_span<value_type_, max_rank_> output) noexcept {
    typename value_type_::scale_t alpha {1}, beta {-1};
    return blend<value_type_, max_rank_>(lhs, rhs, alpha, beta, output);
}

/** Allocating elementwise sub; fails like the allocating @c blend. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> sub(tensor_view<value_type_, max_rank_> lhs,
                                                              tensor_view<value_type_, max_rank_> rhs,
                                                              allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs)) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (lhs.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = sub<value_type_, max_rank_>(lhs, rhs, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Elementwise sub scalar: output[i] = input[i] − scalar. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t sub(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t scalar,
             tensor_span<value_type_, max_rank_> output) noexcept {
    typename value_type_::scale_t one {1};
    typename value_type_::scale_t neg_scalar = -scalar;
    return scale<value_type_, max_rank_>(input, one, neg_scalar, output);
}

/** Allocating sub scalar; fails like the allocating @c scale. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> sub(tensor_view<value_type_, max_rank_> input,
                                                              typename value_type_::scale_t scalar,
                                                              allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = sub<value_type_, max_rank_>(input, scalar, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Elementwise multiplication: each output is the product of the two operands. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t mul(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
             tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        lhs, rhs, output,
        [](tensor_view<value_type_, max_rank_> l, tensor_view<value_type_, max_rank_> r,
           tensor_span<value_type_, max_rank_> out) {
            typename value_type_::scale_t alpha {1}, beta {0};
            return numkong::fma<value_type_>(l.data(), r.data(), out.data(), l.extent(0), &alpha, &beta, out.data());
        });
}

/** Allocating elementwise multiply; fails like the allocating @c blend. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> mul(tensor_view<value_type_, max_rank_> lhs,
                                                              tensor_view<value_type_, max_rank_> rhs,
                                                              allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs)) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (lhs.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::zeros(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = mul<value_type_, max_rank_>(lhs, rhs, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Elementwise multiply by scalar: output[i] = input[i] × scalar. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t mul(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t scalar,
             tensor_span<value_type_, max_rank_> output) noexcept {
    typename value_type_::scale_t zero {0};
    return scale<value_type_, max_rank_>(input, scalar, zero, output);
}

/** Allocating multiply by scalar; fails like the allocating @c scale. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> mul(tensor_view<value_type_, max_rank_> input,
                                                              typename value_type_::scale_t scalar,
                                                              allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = mul<value_type_, max_rank_>(input, scalar, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

#pragma endregion Tensor Elementwise

} // namespace ashvardanian::numkong

#endif // NUMKONG_EACH_HPP
