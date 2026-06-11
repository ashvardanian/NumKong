/**
 *  @brief C++ wrappers for SIMD-accelerated type casting.
 *  @file include/numkong/cast.hpp
 *  @author Ash Vardanian
 *  @date March 20, 2026
 */
#ifndef NK_CAST_HPP
#define NK_CAST_HPP

#include <cstddef> // `std::size_t`

#include "numkong/cast.h"

#include "numkong/tensor.hpp"
#include "numkong/types.hpp"
#include "numkong/vector.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Elementwise type-cast from one numeric type to another.
 *  @param[in] from Input array of `n` elements.
 *  @param[in] n Number of elements.
 *  @param[out] to Output array of `n` elements.
 *
 *  @tparam from_type_ Source element type.
 *  @tparam to_type_ Destination element type.
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`.
 */
template <numeric_dtype from_type_, numeric_dtype to_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void cast(from_type_ const *from, std::size_t n, to_type_ *to) noexcept {
    if constexpr (allow_simd_ == prefer_simd_k) nk_cast(from, from_type_::dtype(), n, to, to_type_::dtype());
    else nk_cast_serial(from, from_type_::dtype(), n, to, to_type_::dtype());
}

/** @brief Elementwise type-cast between vector views. Sizes must match. */
template <numeric_dtype from_type_, numeric_dtype to_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void cast(vector_view<from_type_> from, vector_span<to_type_> to) noexcept {
    std::size_t n = from.size() < to.size() ? from.size() : to.size();
    cast<from_type_, to_type_, allow_simd_>(from.data(), n, to.data());
}

#pragma region Block-Scaled Casts

/**
 *  @brief One side of a block-scaled cast, already marshalled to C-ABI terms.
 *
 *  `elements`/`scales` are the byte buffers (`scales` is NULL for a plain f32 side). `tensor_scale`
 *  is a value carrier: seed it from the source on decode/transcode, or leave it zero on a destination
 *  so the kernel derives it and writes it back. `elements`/`scales` are held as `void *` even for a
 *  source — the kernel only ever reads a source side, so the const_cast in the builders is safe.
 */
struct block_scaled_operand_ {
    void *elements = nullptr;
    void *scales = nullptr;
    nk_scalar_buffer_t tensor_scale = {};
    bool has_tensor_scale = false;
    nk_block_scaled_format_t format = nk_plain(nk_f32_k);
};

/** @brief A plain contiguous f32 side. */
inline block_scaled_operand_ plain_f32_operand_(void const *bytes) noexcept {
    block_scaled_operand_ operand;
    operand.elements = const_cast<void *>(bytes);
    return operand;
}

/** @brief A block-scaled side; @p tensor_scale seeds the value (source) or is 0 to derive (destination). */
template <typename format_>
block_scaled_operand_ scaled_operand_(void const *elements, void const *scales, float tensor_scale) noexcept {
    block_scaled_operand_ operand;
    operand.elements = const_cast<void *>(elements);
    operand.scales = const_cast<void *>(scales);
    operand.tensor_scale.f32 = tensor_scale;
    operand.has_tensor_scale = format_::has_tensor_scale();
    operand.format = nk_block_scaled_format_of_dtype(format_::dtype());
    return operand;
}

/** @brief The single place the block-scaled C kernel is invoked; selects SIMD vs serial. */
template <allow_simd_t allow_simd_>
inline void block_scaled_cast_(block_scaled_operand_ const &source, block_scaled_operand_ &destination,
                               std::size_t count) noexcept {
    nk_scalar_buffer_t const *source_scale = source.has_tensor_scale ? &source.tensor_scale : nullptr;
    nk_scalar_buffer_t *destination_scale = destination.has_tensor_scale ? &destination.tensor_scale : nullptr;
    if constexpr (allow_simd_ == prefer_simd_k)
        nk_cast_block_scaled(source.elements, source.scales, source_scale, &source.format, destination.elements,
                             destination.scales, destination_scale, &destination.format, count);
    else
        nk_cast_block_scaled_serial(source.elements, source.scales, source_scale, &source.format, destination.elements,
                                    destination.scales, destination_scale, &destination.format, count);
}

/** @brief Writes a destination span's per-tensor scale slot from the derived value (NVFP4 only). */
template <typename format_>
void store_derived_tensor_scale_(scaled_tensor_span<format_> const &destination,
                                 block_scaled_operand_ const &operand) noexcept {
    if constexpr (format_::has_tensor_scale())
        if (destination.tensor_scale_slot()) *destination.tensor_scale_slot() = operand.tensor_scale.f32;
}

/**
 *  @brief Encode (quantize) a dense f32 tensor into a preallocated block-scaled tensor.
 *
 *  Derives the per-tensor scale (NVFP4) over the whole tensor in one contiguous pass and writes it
 *  back through the destination's scale slot. The source must be contiguous (the scale is a
 *  whole-tensor reduction). Allocate the destination with `scaled_tensor<format_>::try_empty(...)`.
 */
template <typename format_, std::size_t max_rank_, allow_simd_t allow_simd_ = prefer_simd_k>
void cast(tensor_view<f32_t, max_rank_> from, scaled_tensor_span<format_, max_rank_> to) noexcept {
    auto source = plain_f32_operand_(from.byte_data());
    auto destination =
        scaled_operand_<format_>(to.elements().byte_data(), to.block_scales().byte_data(), /*derive*/ 0.0f);
    block_scaled_cast_<allow_simd_>(source, destination, from.numel());
    store_derived_tensor_scale_(to, destination);
}

/** @brief Encode (quantize) a dense f32 vector into a preallocated single-row block-scaled tensor. */
template <typename format_, allow_simd_t allow_simd_ = prefer_simd_k>
void cast(vector_view<f32_t> from, scaled_tensor_span<format_> to) noexcept {
    auto source = plain_f32_operand_(from.byte_data());
    auto destination =
        scaled_operand_<format_>(to.elements().byte_data(), to.block_scales().byte_data(), /*derive*/ 0.0f);
    block_scaled_cast_<allow_simd_>(source, destination, from.size());
    store_derived_tensor_scale_(to, destination);
}

/** @brief Decode (dequantize) one block-scaled row into a dense f32 vector. */
template <typename format_, std::size_t max_rank_, allow_simd_t allow_simd_ = prefer_simd_k>
void cast(scaled_tensor_view<format_, max_rank_> from, vector_span<f32_t> to) noexcept {
    float tensor_scale = 0.0f;
    if constexpr (format_::has_tensor_scale()) tensor_scale = from.tensor_scale();
    auto source = scaled_operand_<format_>(from.elements().byte_data(), from.block_scales().byte_data(), tensor_scale);
    auto destination = plain_f32_operand_(to.data());
    std::size_t count = from.numel() < to.size() ? from.numel() : to.size();
    block_scaled_cast_<allow_simd_>(source, destination, count);
}

/**
 *  @brief Decode (dequantize) a block-scaled tensor into a dense f32 tensor.
 *
 *  Contiguous tensors decode in a single kernel call; a strided view (e.g. a block-aligned column
 *  tile) decodes row by row, since each row is a contiguous run even when the tile is not.
 */
template <typename format_, std::size_t max_rank_, allow_simd_t allow_simd_ = prefer_simd_k>
void cast(scaled_tensor_view<format_, max_rank_> from, tensor_span<f32_t, max_rank_> to) noexcept {
    bool const contiguous =
        from.elements().is_contiguous() && from.block_scales().is_contiguous() && to.is_contiguous();
    if (from.rank() <= 1 || contiguous) {
        float tensor_scale = 0.0f;
        if constexpr (format_::has_tensor_scale()) tensor_scale = from.tensor_scale();
        auto source =
            scaled_operand_<format_>(from.elements().byte_data(), from.block_scales().byte_data(), tensor_scale);
        auto destination = plain_f32_operand_(to.byte_data());
        std::size_t count = from.numel() < to.numel() ? from.numel() : to.numel();
        block_scaled_cast_<allow_simd_>(source, destination, count);
        return;
    }
    std::size_t rows = from.extent(0) < to.extent(0) ? from.extent(0) : to.extent(0);
    for (std::size_t i = 0; i < rows; ++i) cast<format_, max_rank_, allow_simd_>(from.row(i), to.slice_leading(i));
}

/**
 *  @brief Transcode one block-scaled tensor into another (e.g. MXFP8 → NVFP4).
 *
 *  One contiguous pass, so a per-tensor scale on the destination (NVFP4) is derived over the whole
 *  tensor and written back through its scale slot. Both operands must be contiguous.
 */
template <typename from_format_, typename to_format_, std::size_t max_rank_, allow_simd_t allow_simd_ = prefer_simd_k>
void cast(scaled_tensor_view<from_format_, max_rank_> from, scaled_tensor_span<to_format_, max_rank_> to) noexcept {
    float from_tensor_scale = 0.0f;
    if constexpr (from_format_::has_tensor_scale()) from_tensor_scale = from.tensor_scale();
    auto source =
        scaled_operand_<from_format_>(from.elements().byte_data(), from.block_scales().byte_data(), from_tensor_scale);
    auto destination =
        scaled_operand_<to_format_>(to.elements().byte_data(), to.block_scales().byte_data(), /*derive*/ 0.0f);
    block_scaled_cast_<allow_simd_>(source, destination, from.numel());
    store_derived_tensor_scale_(to, destination);
}

#pragma endregion Block-Scaled Casts

} // namespace ashvardanian::numkong

#endif // NK_CAST_HPP
