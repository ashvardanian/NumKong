/**
 *  @file include/numkong/cast.hpp
 *  @author Ash Vardanian
 *  @date March 20, 2026
 *  @brief C++ wrappers for SIMD-accelerated type casting.
 */
#ifndef NUMKONG_CAST_HPP
#define NUMKONG_CAST_HPP

#include <cstddef> // `std::size_t`

#include "numkong/cast.h"

#include "numkong/tensor.hpp"
#include "numkong/types.hpp"
#include "numkong/vector.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Elementwise type-cast from one numeric type to another.
 *  @param[in] from Input array of @p n elements.
 *  @param[in] n Counts dimensions, a multiple of the values per byte.
 *  @param[out] to Output array of @p n elements.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam from_type_ Source element type.
 *  @tparam to_type_ Destination element type.
 */
template <numeric_dtype from_type_, numeric_dtype to_type_>
status_t cast(from_type_ const *from, std::size_t n, to_type_ *to, nk_capability_t capabilities = cpu_capabilities(),
              void *stream = nullptr) noexcept {
    if (!capabilities)
        return static_cast<status_t>(nk_cast_serial(from, from_type_::dtype(), n, to, to_type_::dtype(), stream));
    return static_cast<status_t>(
        nk_cast_best(from, from_type_::dtype(), n, to, to_type_::dtype(), capabilities, stream));
}

/** Elementwise type-cast of one run into another of equal dimensions; @c unexpected_dimensions_k
 *  when they differ or either run is strided or ends mid-value. */
template <numeric_dtype from_type_, numeric_dtype to_type_, vector_of<from_type_> from_vector_type_,
          mutable_vector_of<to_type_> to_vector_type_>
status_t cast(from_vector_type_ const &from, to_vector_type_ &&to, nk_capability_t capabilities = cpu_capabilities(),
              void *stream = nullptr) noexcept {
    auto from_values = contiguous_values_<from_type_ const>(from);
    auto to_values = contiguous_values_<to_type_>(to);
    std::size_t const dimensions = from_values.value.size() * dimensions_per_value<from_type_>();
    if (!from_values || !to_values || dimensions != to_values.value.size() * dimensions_per_value<to_type_>())
        return status_t::unexpected_dimensions_k;
    return cast<from_type_, to_type_>(from_values.value.data(), dimensions, to_values.value.data(), capabilities,
                                      stream);
}

#pragma region Block Scaled Casts

/**
 *  @brief One side of a block-scaled cast, already marshalled to C-ABI terms.
 *
 *  @c elements and @c scales are the byte buffers, with @c scales NULL for a plain f32 side.
 *  @c tensor_scale is a value carrier: seed it from the source on decode or transcode, or leave it
 *  zero on a destination so the kernel derives it and writes it back. Both buffers stay `void *`
 *  even for a source — the kernel only reads a source side, so the builders' const_cast is safe.
 */
struct block_scaled_operand_ {
    void *elements = nullptr;
    void *scales = nullptr;
    nk_scalar_buffer_t tensor_scale = {};
    bool has_tensor_scale = false;
    nk_block_scaled_format_t format = nk_plain(nk_f32_k);
};

/** A plain contiguous f32 side. */
inline block_scaled_operand_ plain_f32_operand_(void const *bytes) noexcept {
    block_scaled_operand_ operand;
    operand.elements = const_cast<void *>(bytes);
    return operand;
}

/** A block-scaled side; @p tensor_scale seeds the value on a source, or is 0 to derive it on a
 *  destination. */
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

/** The one place the block-scaled C kernel is invoked; no capability runs the serial reference. */
inline status_t block_scaled_cast_(block_scaled_operand_ const &source, block_scaled_operand_ &destination,
                                   std::size_t count, nk_capability_t capabilities, void *stream) noexcept {
    nk_scalar_buffer_t const *source_scale = source.has_tensor_scale ? &source.tensor_scale : nullptr;
    nk_scalar_buffer_t *destination_scale = destination.has_tensor_scale ? &destination.tensor_scale : nullptr;
    if (!capabilities)
        return static_cast<status_t>(nk_cast_block_scaled_serial(
            source.elements, source.scales, source_scale, &source.format, destination.elements, destination.scales,
            destination_scale, &destination.format, count, stream));
    return static_cast<status_t>(nk_cast_block_scaled_best(source.elements, source.scales, source_scale, &source.format,
                                                           destination.elements, destination.scales, destination_scale,
                                                           &destination.format, count, capabilities, stream));
}

/** Writes a destination span's per-tensor scale slot from the derived value, NVFP4 only. */
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
 *  whole-tensor reduction). Allocate the destination with
 *  `scaled_tensor<format_>::uninitialized(...)`, which reports the allocation's own status.
 */
template <typename format_, std::size_t max_rank_>
status_t cast(tensor_view<f32_t, max_rank_> from, scaled_tensor_span<format_, max_rank_> to,
              nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    auto source = plain_f32_operand_(from.byte_data());
    auto destination = scaled_operand_<format_>(to.elements().byte_data(), to.block_scales().byte_data(),
                                                /*derive*/ 0.0f);
    status_t const status = block_scaled_cast_(source, destination, from.numel(), capabilities, stream);
    if (succeeded(status)) store_derived_tensor_scale_(to, destination);
    return status;
}

/** Encode, or quantize, a dense f32 vector into a preallocated single-row block-scaled tensor. */
template <typename format_>
status_t cast(vector_view<f32_t> from, scaled_tensor_span<format_> to,
              nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    auto source = plain_f32_operand_(from.byte_data());
    auto destination = scaled_operand_<format_>(to.elements().byte_data(), to.block_scales().byte_data(),
                                                /*derive*/ 0.0f);
    status_t const status = block_scaled_cast_(source, destination, from.size(), capabilities, stream);
    if (succeeded(status)) store_derived_tensor_scale_(to, destination);
    return status;
}

/** Decode, or dequantize, one block-scaled row into a dense f32 vector. */
template <typename format_, std::size_t max_rank_>
status_t cast(scaled_tensor_view<format_, max_rank_> from, vector_span<f32_t> to,
              nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    float tensor_scale = 0.0f;
    if constexpr (format_::has_tensor_scale()) tensor_scale = from.tensor_scale().raw_;
    auto source = scaled_operand_<format_>(from.elements().byte_data(), from.block_scales().byte_data(), tensor_scale);
    auto destination = plain_f32_operand_(to.data());
    std::size_t count = from.numel() < to.size() ? from.numel() : to.size();
    return block_scaled_cast_(source, destination, count, capabilities, stream);
}

/**
 *  @brief Decode (dequantize) a block-scaled tensor into a dense f32 tensor.
 *
 *  Contiguous tensors decode in a single kernel call; a strided view (e.g. a block-aligned column
 *  tile) decodes row by row, since each row is a contiguous run even when the tile is not.
 */
template <typename format_, std::size_t max_rank_>
status_t cast(scaled_tensor_view<format_, max_rank_> from, tensor_span<f32_t, max_rank_> to,
              nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    bool const contiguous = from.elements().is_contiguous() && from.block_scales().is_contiguous() &&
                            to.is_contiguous();
    if (from.rank() <= 1 || contiguous) {
        float tensor_scale = 0.0f;
        if constexpr (format_::has_tensor_scale()) tensor_scale = from.tensor_scale().raw_;
        auto source = scaled_operand_<format_>(from.elements().byte_data(), from.block_scales().byte_data(),
                                               tensor_scale);
        auto destination = plain_f32_operand_(to.byte_data());
        std::size_t count = from.numel() < to.numel() ? from.numel() : to.numel();
        return block_scaled_cast_(source, destination, count, capabilities, stream);
    }
    std::size_t rows = from.extent(0) < to.extent(0) ? from.extent(0) : to.extent(0);
    for (std::size_t i = 0; i < rows; ++i)
        if (status_t const status = cast<format_, max_rank_>(from.row(i), to.slice_leading(i), capabilities, stream);
            failed(status))
            return status;
    return status_t::success_k;
}

/**
 *  @brief Transcode one block-scaled tensor into another (e.g. MXFP8 → NVFP4).
 *
 *  One contiguous pass, so a per-tensor scale on the destination (NVFP4) is derived over the whole
 *  tensor and written back through its scale slot. Both operands must be contiguous.
 */
template <typename from_format_, typename to_format_, std::size_t max_rank_>
status_t cast(scaled_tensor_view<from_format_, max_rank_> from, scaled_tensor_span<to_format_, max_rank_> to,
              nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) noexcept {
    float from_tensor_scale = 0.0f;
    if constexpr (from_format_::has_tensor_scale()) from_tensor_scale = from.tensor_scale().raw_;
    auto source = scaled_operand_<from_format_>(from.elements().byte_data(), from.block_scales().byte_data(),
                                                from_tensor_scale);
    auto destination = scaled_operand_<to_format_>(to.elements().byte_data(), to.block_scales().byte_data(),
                                                   /*derive*/ 0.0f);
    status_t const status = block_scaled_cast_(source, destination, from.numel(), capabilities, stream);
    if (succeeded(status)) store_derived_tensor_scale_(to, destination);
    return status;
}

#pragma endregion Block Scaled Casts

} // namespace ashvardanian::numkong

#endif // NUMKONG_CAST_HPP
