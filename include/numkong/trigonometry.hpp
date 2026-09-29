/**
 *  @file include/numkong/trigonometry.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for trigonometric kernels.
 */
#ifndef NUMKONG_TRIGONOMETRY_HPP
#define NUMKONG_TRIGONOMETRY_HPP

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
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam precision_type_ Precision type for scalar fallback, defaults to @c in_type_
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_>
status_t sin(in_type_ const *in, std::size_t n, in_type_ *out, nk_capability_t capabilities = default_capabilities(),
             void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<in_type_, precision_type_>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_trig_sin_f64_best(&in->raw_, n, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_trig_sin_f32_best(&in->raw_, n, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_trig_sin_f16_best(&in->raw_, n, &out->raw_, capabilities, stream));
    }
    for (std::size_t i = 0; i < n; i++) out[i] = in_type_(precision_type_(in[i]).sin());
    return status_t::success_k;
}

/**
 *  @brief Array cosine: outᵢ = cos(inᵢ)
 *  @param[in] in Input array
 *  @param[in] n Number of elements
 *  @param[out] out Output array
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam precision_type_ Precision type for scalar fallback, defaults to @c in_type_
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_>
status_t cos(in_type_ const *in, std::size_t n, in_type_ *out, nk_capability_t capabilities = default_capabilities(),
             void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<in_type_, precision_type_>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_trig_cos_f64_best(&in->raw_, n, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_trig_cos_f32_best(&in->raw_, n, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_trig_cos_f16_best(&in->raw_, n, &out->raw_, capabilities, stream));
    }
    for (std::size_t i = 0; i < n; i++) out[i] = in_type_(precision_type_(in[i]).cos());
    return status_t::success_k;
}

/**
 *  @brief Array arctangent: outᵢ = arctan(inᵢ)
 *  @param[in] in Input array
 *  @param[in] n Number of elements
 *  @param[out] out Output array
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam precision_type_ Precision type for scalar fallback, defaults to @c in_type_
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_>
status_t atan(in_type_ const *in, std::size_t n, in_type_ *out, nk_capability_t capabilities = default_capabilities(),
              void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<in_type_, precision_type_>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_trig_atan_f64_best(&in->raw_, n, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_trig_atan_f32_best(&in->raw_, n, &out->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_trig_atan_f16_best(&in->raw_, n, &out->raw_, capabilities, stream));
    }
    for (std::size_t i = 0; i < n; i++) out[i] = in_type_(precision_type_(in[i]).atan());
    return status_t::success_k;
}

/**
 *  @brief NeoX split-half RoPE: rotates channel pairs of every head by per-token angles.
 *
 *  Rotates each pair `(i, i + @p half_dim)` of every head by the `[rows, @p half_dim]` angle grids.
 *
 *  @param[in] x Row-major matrix of shape @b [rows,channels]; channels = heads · 2 · @p half_dim
 *  @param[out] y Output, same shape and dtype as x; may alias x for in-place rotation
 *  @param[in] cos,sin `[rows, @p half_dim]` per-token angle grids, shared across heads
 *  @param[in] rows,heads Token count and heads per token
 *  @param[in] half_dim Half the head dimension; channel @c i pairs with `i + half_dim`
 *  @param[in] x_row_stride Row (token) stride of x in bytes
 *  @param[in] y_row_stride Row (token) stride of y in bytes
 *  @param[in] input_scale Scalar folded onto every loaded element (E4M3 descale; 1.0 for BF16/F32)
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 */
template <numeric_dtype in_type_>
status_t rope(in_type_ const *x, in_type_ *y, f32_t const *cos, f32_t const *sin, std::size_t rows, std::size_t heads,
              std::size_t half_dim, std::size_t x_row_stride, std::size_t y_row_stride, f32_t input_scale = 1.0f,
              nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f32_t>)
            return static_cast<status_t>(nk_trig_rope_f32_best(&x->raw_, &y->raw_, &cos->raw_, &sin->raw_, rows, heads,
                                                               half_dim, x_row_stride, y_row_stride, input_scale.raw_,
                                                               capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_trig_rope_bf16_best(&x->raw_, &y->raw_, &cos->raw_, &sin->raw_, rows, heads,
                                                                half_dim, x_row_stride, y_row_stride, input_scale.raw_,
                                                                capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_trig_rope_e4m3_best(&x->raw_, &y->raw_, &cos->raw_, &sin->raw_, rows, heads,
                                                                half_dim, x_row_stride, y_row_stride, input_scale.raw_,
                                                                capabilities, stream));
    }
    // Scalar fallback for other numeric dtypes or a mask of no capability.
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
                float low = static_cast<float>(x_base[i]) * input_scale.raw_;
                float high = static_cast<float>(x_base[i + half_dim]) * input_scale.raw_;
                float cosine = static_cast<float>(cos_row[i]), sine = static_cast<float>(sin_row[i]);
                y_base[i] = f32_t(low * cosine - high * sine).template to<in_type_>();
                y_base[i + half_dim] = f32_t(low * sine + high * cosine).template to<in_type_>();
            }
        }
    }
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Vector Trigonometric

/** Elementwise sin of one run into another of equal dimensions; @c unexpected_dimensions_k
 *  when they differ or either run is strided or ends mid-value. */
template <numeric_dtype in_type_, vector_of<in_type_> input_type_, mutable_vector_of<in_type_> output_type_>
status_t sin(input_type_ const &input, output_type_ &&output, nk_capability_t capabilities = default_capabilities(),
             void *stream = nullptr) noexcept {
    auto input_values = contiguous_values_<in_type_ const>(input);
    auto output_values = contiguous_values_<in_type_>(output);
    std::size_t const dimensions = input_values.value.size() * dimensions_per_value<in_type_>();
    if (!input_values || !output_values || dimensions != output_values.value.size() * dimensions_per_value<in_type_>())
        return status_t::unexpected_dimensions_k;
    return sin<in_type_>(input_values.value.data(), dimensions, output_values.value.data(), capabilities, stream);
}

/** Elementwise cos of one run into another, failing like the concept @c sin. */
template <numeric_dtype in_type_, vector_of<in_type_> input_type_, mutable_vector_of<in_type_> output_type_>
status_t cos(input_type_ const &input, output_type_ &&output, nk_capability_t capabilities = default_capabilities(),
             void *stream = nullptr) noexcept {
    auto input_values = contiguous_values_<in_type_ const>(input);
    auto output_values = contiguous_values_<in_type_>(output);
    std::size_t const dimensions = input_values.value.size() * dimensions_per_value<in_type_>();
    if (!input_values || !output_values || dimensions != output_values.value.size() * dimensions_per_value<in_type_>())
        return status_t::unexpected_dimensions_k;
    return cos<in_type_>(input_values.value.data(), dimensions, output_values.value.data(), capabilities, stream);
}

/** Elementwise atan of one run into another, failing like the concept @c sin. */
template <numeric_dtype in_type_, vector_of<in_type_> input_type_, mutable_vector_of<in_type_> output_type_>
status_t atan(input_type_ const &input, output_type_ &&output, nk_capability_t capabilities = default_capabilities(),
              void *stream = nullptr) noexcept {
    auto input_values = contiguous_values_<in_type_ const>(input);
    auto output_values = contiguous_values_<in_type_>(output);
    std::size_t const dimensions = input_values.value.size() * dimensions_per_value<in_type_>();
    if (!input_values || !output_values || dimensions != output_values.value.size() * dimensions_per_value<in_type_>())
        return status_t::unexpected_dimensions_k;
    return atan<in_type_>(input_values.value.data(), dimensions, output_values.value.data(), capabilities, stream);
}

#pragma endregion Vector Trigonometric

#pragma region Tensor Trigonometric

/** Elementwise sin into pre-allocated output. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t sin(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            return numkong::sin<value_type_>(in.data(), in.extent(0), out.data());
        });
}

/** Allocating sin; empty for an empty @p input, or the allocation's or the kernel's failure. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> sin(tensor_view<value_type_, max_rank_> input,
                                                              allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = sin<value_type_, max_rank_>(input, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Elementwise cos into pre-allocated output. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t cos(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            return numkong::cos<value_type_>(in.data(), in.extent(0), out.data());
        });
}

/** Allocating cos; empty for an empty @p input, or the allocation's or the kernel's failure. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> cos(tensor_view<value_type_, max_rank_> input,
                                                              allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = cos<value_type_, max_rank_>(input, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Elementwise atan into pre-allocated output. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
status_t atan(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            return numkong::atan<value_type_>(in.data(), in.extent(0), out.data());
        });
}

/** In-place NeoX split-half RoPE over a @b [rows,channels] matrix span, channels being heads times
 *  2 · @p half_dim; @c unexpected_dimensions_k when the shapes or the tables are too small. */
template <numeric_dtype value_type_>
status_t rope(matrix_view<value_type_> x, matrix_span<value_type_> y, vector_view<f32_t> cos, vector_view<f32_t> sin,
              std::size_t heads, std::size_t half_dim, f32_t input_scale = 1.0f) noexcept {
    if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1)) return status_t::unexpected_dimensions_k;
    if (x.extent(1) < heads * 2 * half_dim) return status_t::unexpected_dimensions_k;
    if (cos.size() < x.extent(0) * half_dim || sin.size() < x.extent(0) * half_dim)
        return status_t::unexpected_dimensions_k;
    return numkong::rope<value_type_>(x.data(), y.data(), cos.data(), sin.data(), x.extent(0), heads, half_dim,
                                      static_cast<std::size_t>(x.stride_bytes(0)),
                                      static_cast<std::size_t>(y.stride_bytes(0)), input_scale);
}

/** Allocating atan; empty for an empty @p input, or the allocation's or the kernel's failure. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
expected<tensor<value_type_, allocator_type_, max_rank_>> atan(tensor_view<value_type_, max_rank_> input,
                                                               allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::uninitialized(input_shape.extents, input_shape.rank, alloc);
    if (!result) return result;
    if (status_t status = atan<value_type_, max_rank_>(input, result.value.span()); failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

#pragma endregion Tensor Trigonometric

} // namespace ashvardanian::numkong

#endif // NUMKONG_TRIGONOMETRY_HPP
