/**
 *  @brief C++ bindings for trigonometric kernels.
 *  @file include/numkong/trigonometry.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_TRIGONOMETRY_HPP
#define NK_TRIGONOMETRY_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/trigonometry.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Array sine: outᵢ = sin(inᵢ)
 *  @param[in] in Input array
 *  @param[in] n Number of elements
 *  @param[out] out Output array
 *
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam precision_type_ Precision type for scalar fallback, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void sin(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_trig_sin_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_trig_sin_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_trig_sin_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = in_type_(precision_type_(in[i]).sin());
    }
}

/**
 *  @brief Array cosine: outᵢ = cos(inᵢ)
 *  @param[in] in Input array
 *  @param[in] n Number of elements
 *  @param[out] out Output array
 *
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam precision_type_ Precision type for scalar fallback, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void cos(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_trig_cos_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_trig_cos_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_trig_cos_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = in_type_(precision_type_(in[i]).cos());
    }
}

/**
 *  @brief Array arctangent: outᵢ = arctan(inᵢ)
 *  @param[in] in Input array
 *  @param[in] n Number of elements
 *  @param[out] out Output array
 *
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam precision_type_ Precision type for scalar fallback, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void atan(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_trig_atan_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_trig_atan_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_trig_atan_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = in_type_(precision_type_(in[i]).atan());
    }
}

/**
 *  @brief NeoX split-half rotary position embedding (RoPE): rotates channel pairs by per-token angles.
 *
 *  Rotates each pair `(i, i + half_dim)` of every head by the `[rows, half_dim]` angle grids.
 *
 *  @param[in] x `rows × (heads · 2·half_dim)` input token matrix
 *  @param[out] y Output, same shape and dtype as x; may alias x for in-place rotation
 *  @param[in] cos,sin `[rows, half_dim]` per-token angle grids, shared across heads
 *  @param[in] rows,heads Token count and heads per token
 *  @param[in] half_dim Half the head dimension; channel `i` pairs with `i + half_dim`
 *  @param[in] x_row_stride,y_row_stride Row (token) strides in bytes
 *  @param[in] input_scale Scalar folded onto every loaded element (E4M3 descale; 1.0 for BF16/F32)
 *
 *  @tparam in_type_ Element type
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void rope(in_type_ const *x, in_type_ *y, f32_t const *cos, f32_t const *sin, std::size_t rows, std::size_t heads,
          std::size_t half_dim, std::size_t x_row_stride, std::size_t y_row_stride, float input_scale = 1.0f) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k;
    if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_trig_rope_f32(&x->raw_, &y->raw_, &cos->raw_, &sin->raw_, rows, heads, half_dim, x_row_stride, y_row_stride,
                         input_scale);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_trig_rope_bf16(&x->raw_, &y->raw_, &cos->raw_, &sin->raw_, rows, heads, half_dim, x_row_stride, y_row_stride,
                          input_scale);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_trig_rope_e4m3(&x->raw_, &y->raw_, &cos->raw_, &sin->raw_, rows, heads, half_dim, x_row_stride, y_row_stride,
                          input_scale);
    // Scalar fallback for other numeric dtypes or when SIMD is disabled.
    else {
        for (std::size_t row = 0; row < rows; ++row) {
            f32_t const *cos_row = cos + row * half_dim;
            f32_t const *sin_row = sin + row * half_dim;
            in_type_ const *x_row = reinterpret_cast<in_type_ const *>(reinterpret_cast<char const *>(x) +
                                                                       row * x_row_stride);
            in_type_ *y_row = reinterpret_cast<in_type_ *>(reinterpret_cast<char *>(y) + row * y_row_stride);
            for (std::size_t head = 0; head < heads; ++head) {
                in_type_ const *x_base = x_row + head * 2 * half_dim;
                in_type_ *y_base = y_row + head * 2 * half_dim;
                for (std::size_t i = 0; i < half_dim; ++i) {
                    float low = static_cast<float>(x_base[i]) * input_scale;
                    float high = static_cast<float>(x_base[i + half_dim]) * input_scale;
                    float cosine = static_cast<float>(cos_row[i]), sine = static_cast<float>(sin_row[i]);
                    y_base[i] = f32_t(low * cosine - high * sine).template to<in_type_>();
                    y_base[i + half_dim] = f32_t(low * sine + high * cosine).template to<in_type_>();
                }
            }
        }
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Tensor Trigonometric

/** @brief Elementwise sin into pre-allocated output. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool sin(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            numkong::sin<value_type_>(in.data(), in.extent(0), out.data());
        });
}

/** @brief Allocating sin. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_sin(tensor_view<value_type_, max_rank_> input) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!sin<value_type_, max_rank_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise cos into pre-allocated output. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool cos(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            numkong::cos<value_type_>(in.data(), in.extent(0), out.data());
        });
}

/** @brief Allocating cos. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_cos(tensor_view<value_type_, max_rank_> input) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!cos<value_type_, max_rank_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise atan into pre-allocated output. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool atan(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            numkong::atan<value_type_>(in.data(), in.extent(0), out.data());
        });
}

/** @brief In-place NeoX split-half RoPE over a `[rows, heads·2·half_dim]` matrix span. */
template <numeric_dtype value_type_>
bool rope(matrix_view<value_type_> x, matrix_span<value_type_> y, vector_view<f32_t> cos, vector_view<f32_t> sin,
          std::size_t heads, std::size_t half_dim, float input_scale = 1.0f) noexcept {
    if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1)) return false;
    if (x.extent(1) < heads * 2 * half_dim) return false;
    if (cos.size() < x.extent(0) * half_dim || sin.size() < x.extent(0) * half_dim) return false;
    numkong::rope<value_type_>(x.data(), y.data(), cos.data(), sin.data(), x.extent(0), heads, half_dim,
                               static_cast<std::size_t>(x.stride_bytes(0)), static_cast<std::size_t>(y.stride_bytes(0)),
                               input_scale);
    return true;
}

/** @brief Allocating atan. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_atan(tensor_view<value_type_, max_rank_> input) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!atan<value_type_, max_rank_>(input, result.span())) return out_tensor_t {};
    return result;
}

#pragma endregion Tensor Trigonometric

} // namespace ashvardanian::numkong

#endif // NK_TRIGONOMETRY_HPP
