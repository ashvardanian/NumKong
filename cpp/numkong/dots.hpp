/**
 *  @file cpp/numkong/dots.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for multi-target dot-product kernels.
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_DOTS_HPP
#define NUMKONG_DOTS_HPP

#include <cstdint>

#include <limits>
#include <type_traits>

#include "numkong/dot.h"
#include "numkong/dots.h"
#include "numkong/sets.h"

#include "numkong/devices.hpp" // `default_capabilities`
#include "numkong/matrix.hpp"  // `packed_matrix`

namespace ashvardanian::numkong {

/**
 *  @brief Estimates the memory requirements for packed B matrix.
 *  @param[in] columns Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row (k)
 *  @param[in] capabilities Capabilities to pick from
 *  @return Size in bytes of the pack @c dots_pack writes with the same @p capabilities
 *
 *  @tparam in_type_ Input type, a plain dtype or a block-scaled format
 */
template <a_numeric_dtype in_type_>
expected<std::size_t> dots_pack_size(std::size_t columns, std::size_t depth,
                                     nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_dots_pack_size_f64_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_dots_pack_size_f32_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_dots_pack_size_f16_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_dots_pack_size_bf16_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_dots_pack_size_i8_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        status = nk_dots_pack_size_u8_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_dots_pack_size_e4m3_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        status = nk_dots_pack_size_e5m2_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        status = nk_dots_pack_size_e2m3_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        status = nk_dots_pack_size_e2m1_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        status = nk_dots_pack_size_e3m2_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        status = nk_dots_pack_size_u4_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        status = nk_dots_pack_size_i4_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u1x8_t>)
        status = nk_dots_pack_size_u1_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_dots_pack_size_nvfp4_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_dots_pack_size_mxfp4_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_dots_pack_size_mxfp6e2m3_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_dots_pack_size_mxfp6e3m2_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_dots_pack_size_mxfp8e4m3_best(columns, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_dots_pack_size_mxfp8e5m2_best(columns, depth, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_dots_pack_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs matrix B into the layout @c dots_packed reads.
 *  @param[in] b Input matrix B, row-major, of shape @b [columns,depth], see @c operand_pointer;
 *      packing fuses no transposition, so a transposed B is transposed in a separate pass first
 *  @param[in] columns Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row (k)
 *  @param[in] b_stride Stride between rows of B in bytes
 *  @param[out] b_packed Output buffer of @c dots_pack_size bytes for the same @p capabilities
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input type, a plain dtype or a block-scaled format
 */
template <a_numeric_dtype in_type_>
status_t dots_pack(operand_pointer<in_type_> b, std::size_t columns, std::size_t depth, std::size_t b_stride,
                   void *b_packed, nk_capability_t capabilities = default_capabilities(),
                   nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(
            nk_dots_pack_f64_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(
            nk_dots_pack_f32_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(
            nk_dots_pack_f16_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(
            nk_dots_pack_bf16_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(
            nk_dots_pack_i8_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(
            nk_dots_pack_u8_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(
            nk_dots_pack_e4m3_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(
            nk_dots_pack_e5m2_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(
            nk_dots_pack_e2m3_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        return static_cast<status_t>(
            nk_dots_pack_e2m1_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(
            nk_dots_pack_e3m2_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(
            nk_dots_pack_u4_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(
            nk_dots_pack_i4_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(
            nk_dots_pack_u1_best(&b->raw_, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(
            nk_dots_pack_nvfp4_best(b, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(
            nk_dots_pack_mxfp4_best(b, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(
            nk_dots_pack_mxfp6e2m3_best(b, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(
            nk_dots_pack_mxfp6e3m2_best(b, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(
            nk_dots_pack_mxfp8e4m3_best(b, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(
            nk_dots_pack_mxfp8e5m2_best(b, columns, depth, b_stride, b_packed, 0, columns, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_dots_pack has no kernel for this dtype");
}

/**
 *  @brief Packs matrix B once into an owner that every @c dots_packed, @c angulars_packed and
 *      @c euclideans_packed call reuses.
 *  @param[in] b 2D matrix view, row-major, of shape @b [rows,depth].
 *  @param[in] alloc Allocator of the packed buffer.
 *  @param[in] capabilities Capabilities to pack for; multiply with the same ones.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *  @return The packed matrix, failing like @c packed_matrix::make.
 */
template <a_numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
expected<packed_matrix<value_type_, allocator_type_>> dots_pack(matrix_view<value_type_> b, allocator_type_ alloc = {},
                                                                nk_capability_t capabilities = default_capabilities(),
                                                                nk_stream_t stream = nullptr) noexcept {
    auto const pack_size = [capabilities](nk_size_t rows, nk_size_t depth, nk_size_t *bytes) noexcept {
        auto const [size, status] = dots_pack_size<value_type_>(rows, depth, capabilities);
        *bytes = size;
        return status;
    };
    auto const pack = [capabilities, stream](operand_pointer<value_type_> values, nk_size_t rows, nk_size_t depth,
                                             nk_size_t stride, void *packed) noexcept {
        return dots_pack<value_type_>(values, rows, depth, stride, packed, capabilities, stream);
    };
    return packed_matrix<value_type_, allocator_type_>::make(b, alloc, pack_size, pack);
}

/**
 *  @brief Returns the workspace bytes of @c dots_packed, zero when no scratch is needed.
 *  @param[in] rows Rows of A and C
 *  @param[in] column_count Columns of B and C
 *  @param[in] depth Columns of A, rows of B
 *  @param[in] capabilities Capabilities to pick from
 */
template <a_numeric_dtype in_type_>
expected<std::size_t> dots_packed_workspace_size(std::size_t rows, std::size_t column_count, std::size_t depth,
                                                 nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_dots_packed_workspace_size_f64_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_dots_packed_workspace_size_f32_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_dots_packed_workspace_size_f16_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_dots_packed_workspace_size_bf16_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_dots_packed_workspace_size_i8_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        status = nk_dots_packed_workspace_size_u8_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_dots_packed_workspace_size_e4m3_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        status = nk_dots_packed_workspace_size_e5m2_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        status = nk_dots_packed_workspace_size_e2m3_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        status = nk_dots_packed_workspace_size_e2m1_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        status = nk_dots_packed_workspace_size_e3m2_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        status = nk_dots_packed_workspace_size_u4_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        status = nk_dots_packed_workspace_size_i4_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u1x8_t>)
        status = nk_dots_packed_workspace_size_u1_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_dots_packed_workspace_size_nvfp4_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_dots_packed_workspace_size_mxfp4_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_dots_packed_workspace_size_mxfp6e2m3_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_dots_packed_workspace_size_mxfp6e3m2_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_dots_packed_workspace_size_mxfp8e4m3_best(rows, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_dots_packed_workspace_size_mxfp8e5m2_best(rows, column_count, depth, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_dots_packed_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Returns the workspace bytes of @c dots_symmetric, zero when no scratch is needed.
 *  @param[in] vector_count Number of vectors
 *  @param[in] depth Counts dimensions
 *  @param[in] rows_begin First row of C to compute
 *  @param[in] rows_end Row of C after the last
 *  @param[in] capabilities Capabilities to pick from
 */
template <a_numeric_dtype in_type_>
expected<std::size_t> dots_symmetric_workspace_size(std::size_t vector_count, std::size_t depth,
                                                    std::size_t rows_begin = 0,
                                                    std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                                                    nk_capability_t capabilities = default_capabilities()) noexcept {
    if (rows_end > vector_count) rows_end = vector_count;
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_dots_symmetric_workspace_size_f64_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                           &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_dots_symmetric_workspace_size_f32_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                           &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_dots_symmetric_workspace_size_f16_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                           &bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_dots_symmetric_workspace_size_bf16_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                            &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_dots_symmetric_workspace_size_i8_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                          &bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        status = nk_dots_symmetric_workspace_size_u8_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                          &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_dots_symmetric_workspace_size_e4m3_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                            &bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        status = nk_dots_symmetric_workspace_size_e5m2_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                            &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        status = nk_dots_symmetric_workspace_size_e2m3_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                            &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        status = nk_dots_symmetric_workspace_size_e2m1_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                            &bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        status = nk_dots_symmetric_workspace_size_e3m2_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                            &bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        status = nk_dots_symmetric_workspace_size_u4_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                          &bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        status = nk_dots_symmetric_workspace_size_i4_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                          &bytes);
    else if constexpr (std::is_same_v<in_type_, u1x8_t>)
        status = nk_dots_symmetric_workspace_size_u1_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                          &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_dots_symmetric_workspace_size_nvfp4_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                             &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_dots_symmetric_workspace_size_mxfp4_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                             &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_dots_symmetric_workspace_size_mxfp6e2m3_best(vector_count, depth, rows_begin, rows_end,
                                                                 capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_dots_symmetric_workspace_size_mxfp6e3m2_best(vector_count, depth, rows_begin, rows_end,
                                                                 capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_dots_symmetric_workspace_size_mxfp8e4m3_best(vector_count, depth, rows_begin, rows_end,
                                                                 capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_dots_symmetric_workspace_size_mxfp8e5m2_best(vector_count, depth, rows_begin, rows_end,
                                                                 capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_dots_symmetric_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packed dot products (batch matrix multiply): C = A × B (row-major)
 *  @param[in] a Matrix A of shape @b [rows,depth], see @c operand_pointer
 *  @param[in] b_packed Matrix B of shape @b [depth,columns], which @c dots_pack packed for
 *      the same @p capabilities
 *  @param[out] c Output matrix C of shape @b [rows,columns]
 *  @param[in] rows Rows of A and C
 *  @param[in] column_count Columns of B and C
 *  @param[in] depth Columns of A, rows of B, a multiple of the values per byte.
 *  @param[in] a_stride Stride between rows of A in bytes
 *  @param[in] c_stride Stride between rows of C in bytes
 *  @param[in] workspace Scratch aligned to @c nk_default_alignment_k, sized by the matching query,
 *      or empty for the scratch-free path
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input type, a plain dtype or a block-scaled format
 */
template <a_numeric_dtype in_type_>
status_t dots_packed(operand_pointer<in_type_> a, void const *b_packed, typename in_type_::dot_result_t *c,
                     std::size_t rows, std::size_t column_count, std::size_t depth, std::size_t a_stride,
                     std::size_t c_stride, std::span<std::byte> workspace = {},
                     nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_dots_packed_f64_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                             a_stride, c_stride, workspace.data(), workspace.size(),
                                                             capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_dots_packed_f32_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                             a_stride, c_stride, workspace.data(), workspace.size(),
                                                             capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_dots_packed_f16_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                             a_stride, c_stride, workspace.data(), workspace.size(),
                                                             capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_dots_packed_bf16_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                              a_stride, c_stride, workspace.data(), workspace.size(),
                                                              capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_dots_packed_i8_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                            a_stride, c_stride, workspace.data(), workspace.size(),
                                                            capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_dots_packed_u8_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                            a_stride, c_stride, workspace.data(), workspace.size(),
                                                            capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_dots_packed_e4m3_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                              a_stride, c_stride, workspace.data(), workspace.size(),
                                                              capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(nk_dots_packed_e5m2_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                              a_stride, c_stride, workspace.data(), workspace.size(),
                                                              capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(nk_dots_packed_e2m3_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                              a_stride, c_stride, workspace.data(), workspace.size(),
                                                              capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        return static_cast<status_t>(nk_dots_packed_e2m1_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                              a_stride, c_stride, workspace.data(), workspace.size(),
                                                              capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(nk_dots_packed_e3m2_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                              a_stride, c_stride, workspace.data(), workspace.size(),
                                                              capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_dots_packed_u4_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                            a_stride, c_stride, workspace.data(), workspace.size(),
                                                            capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_dots_packed_i4_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                            a_stride, c_stride, workspace.data(), workspace.size(),
                                                            capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(nk_dots_packed_u1_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                            a_stride, c_stride, workspace.data(), workspace.size(),
                                                            capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_dots_packed_nvfp4_best(a, b_packed, &c->raw_, rows, column_count, depth,
                                                               a_stride, c_stride, workspace.data(), workspace.size(),
                                                               capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_dots_packed_mxfp4_best(a, b_packed, &c->raw_, rows, column_count, depth,
                                                               a_stride, c_stride, workspace.data(), workspace.size(),
                                                               capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(nk_dots_packed_mxfp6e2m3_best(a, b_packed, &c->raw_, rows, column_count, depth,
                                                                   a_stride, c_stride, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(nk_dots_packed_mxfp6e3m2_best(a, b_packed, &c->raw_, rows, column_count, depth,
                                                                   a_stride, c_stride, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(nk_dots_packed_mxfp8e4m3_best(a, b_packed, &c->raw_, rows, column_count, depth,
                                                                   a_stride, c_stride, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(nk_dots_packed_mxfp8e5m2_best(a, b_packed, &c->raw_, rows, column_count, depth,
                                                                   a_stride, c_stride, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_dots_packed has no kernel for this dtype");
}

/**
 *  @brief Symmetric dot products: C = A × Aᵀ where C[i,j] = ⟨A[i], A[j]⟩
 *  @param[in] a Matrix A of shape @b [vectors,depth], see @c operand_pointer
 *  @param[in] vector_count Number of vectors
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride Stride between vectors in A
 *  @param[out] c Output matrix C of shape @b [vectors,vectors]
 *  @param[in] c_stride Stride between rows of C in bytes
 *  @param[in] rows_begin First row of C to compute, 0 by default
 *  @param[in] rows_end Row of C after the last, clamped to @p vector_count, all by default
 *  @param[in] workspace Scratch aligned to @c nk_default_alignment_k, sized by the matching query,
 *      or empty for the scratch-free path
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input type, a plain dtype or a block-scaled format
 */
template <a_numeric_dtype in_type_>
status_t dots_symmetric(operand_pointer<in_type_> a, std::size_t vector_count, std::size_t depth, std::size_t a_stride,
                        typename in_type_::dot_result_t *c, std::size_t c_stride, std::size_t rows_begin = 0,
                        std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                        std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                        nk_stream_t stream = nullptr) noexcept {
    if (rows_end > vector_count) rows_end = vector_count;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_dots_symmetric_f64_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                c_stride, rows_begin, rows_end, workspace.data(),
                                                                workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_dots_symmetric_f32_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                c_stride, rows_begin, rows_end, workspace.data(),
                                                                workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_dots_symmetric_f16_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                c_stride, rows_begin, rows_end, workspace.data(),
                                                                workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_dots_symmetric_bf16_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                 c_stride, rows_begin, rows_end, workspace.data(),
                                                                 workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_dots_symmetric_i8_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                               c_stride, rows_begin, rows_end, workspace.data(),
                                                               workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_dots_symmetric_u8_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                               c_stride, rows_begin, rows_end, workspace.data(),
                                                               workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_dots_symmetric_e4m3_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                 c_stride, rows_begin, rows_end, workspace.data(),
                                                                 workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(nk_dots_symmetric_e5m2_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                 c_stride, rows_begin, rows_end, workspace.data(),
                                                                 workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(nk_dots_symmetric_e2m3_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                 c_stride, rows_begin, rows_end, workspace.data(),
                                                                 workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        return static_cast<status_t>(nk_dots_symmetric_e2m1_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                 c_stride, rows_begin, rows_end, workspace.data(),
                                                                 workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(nk_dots_symmetric_e3m2_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                 c_stride, rows_begin, rows_end, workspace.data(),
                                                                 workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_dots_symmetric_u4_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                               c_stride, rows_begin, rows_end, workspace.data(),
                                                               workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_dots_symmetric_i4_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                               c_stride, rows_begin, rows_end, workspace.data(),
                                                               workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(nk_dots_symmetric_u1_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                               c_stride, rows_begin, rows_end, workspace.data(),
                                                               workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_dots_symmetric_nvfp4_best(a, vector_count, depth, a_stride, &c->raw_, c_stride,
                                                                  rows_begin, rows_end, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_dots_symmetric_mxfp4_best(a, vector_count, depth, a_stride, &c->raw_, c_stride,
                                                                  rows_begin, rows_end, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(nk_dots_symmetric_mxfp6e2m3_best(a, vector_count, depth, a_stride, &c->raw_,
                                                                      c_stride, rows_begin, rows_end, workspace.data(),
                                                                      workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(nk_dots_symmetric_mxfp6e3m2_best(a, vector_count, depth, a_stride, &c->raw_,
                                                                      c_stride, rows_begin, rows_end, workspace.data(),
                                                                      workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(nk_dots_symmetric_mxfp8e4m3_best(a, vector_count, depth, a_stride, &c->raw_,
                                                                      c_stride, rows_begin, rows_end, workspace.data(),
                                                                      workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(nk_dots_symmetric_mxfp8e5m2_best(a, vector_count, depth, a_stride, &c->raw_,
                                                                      c_stride, rows_begin, rows_end, workspace.data(),
                                                                      workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_dots_symmetric has no kernel for this dtype");
}

/** Returns the workspace bytes of @c hammings_packed, zero when no scratch is needed. */
template <a_numeric_dtype in_type_>
expected<std::size_t> hammings_packed_workspace_size(std::size_t rows, std::size_t column_count, std::size_t depth,
                                                     nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        status = nk_hammings_packed_workspace_size_u1_best(rows, column_count, depth, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_hammings_packed_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** Returns the workspace bytes of @c hammings_symmetric, zero when no scratch is needed. */
template <a_numeric_dtype in_type_>
expected<std::size_t> hammings_symmetric_workspace_size(
    std::size_t vector_count, std::size_t depth, std::size_t rows_begin = 0,
    std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
    nk_capability_t capabilities = default_capabilities()) noexcept {
    if (rows_end > vector_count) rows_end = vector_count;
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        status = nk_hammings_symmetric_workspace_size_u1_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                              &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_hammings_symmetric_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** Returns the workspace bytes of @c jaccards_packed, zero when no scratch is needed. */
template <a_numeric_dtype in_type_>
expected<std::size_t> jaccards_packed_workspace_size(std::size_t rows, std::size_t column_count, std::size_t depth,
                                                     nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        status = nk_jaccards_packed_workspace_size_u1_best(rows, column_count, depth, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_jaccards_packed_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** Returns the workspace bytes of @c jaccards_symmetric, zero when no scratch is needed. */
template <a_numeric_dtype in_type_>
expected<std::size_t> jaccards_symmetric_workspace_size(
    std::size_t vector_count, std::size_t depth, std::size_t rows_begin = 0,
    std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
    nk_capability_t capabilities = default_capabilities()) noexcept {
    if (rows_end > vector_count) rows_end = vector_count;
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        status = nk_jaccards_symmetric_workspace_size_u1_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                              &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_jaccards_symmetric_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Symmetric Hamming distance matrix: C[i,j] = hamming(A[i], A[j])
 *  @param[in] a Input matrix, shape @b [vector_count,depth]
 *  @param[in] vector_count Number of vectors
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride Row stride in bytes
 *  @param[out] c Output matrix, shape @b [vector_count,vector_count]
 *  @param[in] c_stride Output row stride in bytes
 *  @param[in] rows_begin First row to compute (default 0)
 *  @param[in] rows_end Row after the last to compute, clamped to @p vector_count (default all)
 *
 *  Computes Hamming distances between all pairs of binary vectors. For @c u1x8_t inputs, distances
 *  are exact bit counts, returned as @c u32_t.
 *
 *  @param[in] workspace At least what @c hammings_symmetric_workspace_size reports, or empty
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input element type (u1x8_t)
 */
template <a_numeric_dtype in_type_>
status_t hammings_symmetric(in_type_ const *a, std::size_t vector_count, std::size_t depth, std::size_t a_stride,
                            typename in_type_::hamming_result_t *c, std::size_t c_stride, std::size_t rows_begin = 0,
                            std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                            std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                            nk_stream_t stream = nullptr) noexcept {
    if (rows_end > vector_count) rows_end = vector_count;
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(nk_hammings_symmetric_u1_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                   c_stride, rows_begin, rows_end, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_hammings_symmetric has no kernel for this dtype");
}

/**
 *  @brief Computes Hamming distances between rows of A and columns of packed B.
 *  @param[in] a Pointer to the first matrix (m x k).
 *  @param[in] b_packed Pointer to the second matrix (k x n), which @c dots_pack packed for
 *      the same capabilities.
 *  @param[out] c Pointer to the output matrix (m x n).
 *  @param[in] rows Number of rows in A (m).
 *  @param[in] column_count Number of columns in B (n).
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride Stride between consecutive rows of A in bytes.
 *  @param[in] c_stride Stride between consecutive rows of C in bytes.
 *
 *  Computes Hamming distances between binary vectors using optimized packed format. For @c u1x8_t
 *  inputs, distances are exact bit counts, returned as @c u32_t.
 *
 *  @param[in] workspace At least what @c hammings_packed_workspace_size reports, or empty.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input element type (u1x8_t)
 */
template <a_numeric_dtype in_type_>
status_t hammings_packed(in_type_ const *a, void const *b_packed, typename in_type_::hamming_result_t *c,
                         std::size_t rows, std::size_t column_count, std::size_t depth, std::size_t a_stride = 0,
                         std::size_t c_stride = 0, std::span<std::byte> workspace = {},
                         nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if (!a_stride) a_stride = depth / dimensions_per_value<in_type_>() * sizeof(in_type_);
    if (!c_stride) c_stride = column_count * sizeof(typename in_type_::hamming_result_t);
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(nk_hammings_packed_u1_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                                a_stride, c_stride, workspace.data(), workspace.size(),
                                                                capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_hammings_packed has no kernel for this dtype");
}

/** Symmetric Jaccard distance matrix: C[i,j] = jaccard(A[i], A[j]). */
template <a_numeric_dtype in_type_>
status_t jaccards_symmetric(in_type_ const *a, std::size_t vector_count, std::size_t depth, std::size_t a_stride,
                            typename in_type_::jaccard_result_t *c, std::size_t c_stride, std::size_t rows_begin = 0,
                            std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                            std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                            nk_stream_t stream = nullptr) noexcept {
    if (rows_end > vector_count) rows_end = vector_count;
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(nk_jaccards_symmetric_u1_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                   c_stride, rows_begin, rows_end, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_jaccards_symmetric has no kernel for this dtype");
}

/** Computes Jaccard distances between rows of A and columns of packed B. */
template <a_numeric_dtype in_type_>
status_t jaccards_packed(in_type_ const *a, void const *b_packed, typename in_type_::jaccard_result_t *c,
                         std::size_t rows, std::size_t column_count, std::size_t depth, std::size_t a_stride = 0,
                         std::size_t c_stride = 0, std::span<std::byte> workspace = {},
                         nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if (!a_stride) a_stride = depth / dimensions_per_value<in_type_>() * sizeof(in_type_);
    if (!c_stride) c_stride = column_count * sizeof(typename in_type_::jaccard_result_t);
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(nk_jaccards_packed_u1_best(&a->raw_, b_packed, &c->raw_, rows, column_count, depth,
                                                                a_stride, c_stride, workspace.data(), workspace.size(),
                                                                capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_jaccards_packed has no kernel for this dtype");
}

#pragma region Concept Constrained Symmetric Dot Products

/** C = A × Aᵀ where C[i,j] = ⟨A[i], A[j]⟩, taking the pointer overload's trailing parameters.
 *  @c unexpected_dimensions_k unless the output is square over the input rows. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::dot_result_t> output_matrix_>
status_t dots_symmetric(input_matrix_ const &input, output_matrix_ &&output, std::size_t rows_begin = 0,
                        std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                        std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                        nk_stream_t stream = nullptr) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    if (input.stride_bytes(1) != sizeof(value_type_) || input.stride_bytes(0) < 0 ||
        output.stride_bytes(1) != sizeof(typename value_type_::dot_result_t) || output.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::dots_symmetric<value_type_>(reinterpret_cast<value_type_ const *>(input.data()), num_vectors,
                                                input.extent(1), static_cast<std::size_t>(input.stride_bytes(0)),
                                                reinterpret_cast<typename value_type_::dot_result_t *>(output.data()),
                                                static_cast<std::size_t>(output.stride_bytes(0)), rows_begin, rows_end,
                                                workspace, capabilities, stream);
}

/** Allocating symmetric dot products: C = A × Aᵀ, with any workspace in the result's block. Empty
 *  for an empty input, or the allocation's or the kernel's failure. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::dot_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::dot_result_t>)
expected<matrix<typename value_type_::dot_result_t, allocator_type_>> dots_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::dot_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto workspace_bytes = dots_symmetric_workspace_size<value_type_>(num_vectors, input.extent(1), 0, num_vectors,
                                                                      capabilities);
    if (!workspace_bytes) return {out_tensor_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto result = uninitialized_with_workspace_<out_tensor_t>({num_vectors, num_vectors}, workspace_bytes.value, alloc,
                                                              workspace);
    if (!result) return result;
    if (status_t status = result.value.span().fill_zeros(); failed(status)) return {out_tensor_t(alloc), status};
    if (status_t status = dots_symmetric<value_type_>(input, result.value.span(), 0, num_vectors, workspace,
                                                      capabilities, stream);
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Symmetric Hamming distances: C[i,j] = hamming(A[i], A[j]), trailing like @c dots_symmetric.
 *  @c unexpected_dimensions_k unless the output is square over the input rows. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::hamming_result_t> output_matrix_>
status_t hammings_symmetric(input_matrix_ const &input, output_matrix_ &&output, std::size_t rows_begin = 0,
                            std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                            std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                            nk_stream_t stream = nullptr) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    if (input.stride_bytes(1) != sizeof(value_type_) || input.stride_bytes(0) < 0 ||
        output.stride_bytes(1) != sizeof(typename value_type_::hamming_result_t) || output.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::hammings_symmetric<value_type_>(
        reinterpret_cast<value_type_ const *>(input.data()), num_vectors, input.extent(1),
        static_cast<std::size_t>(input.stride_bytes(0)),
        reinterpret_cast<typename value_type_::hamming_result_t *>(output.data()),
        static_cast<std::size_t>(output.stride_bytes(0)), rows_begin, rows_end, workspace, capabilities, stream);
}

/** Allocating symmetric Hamming distances, with any workspace in the result's block. Empty for an
 *  empty input, or the allocation's or the kernel's failure. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::hamming_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::hamming_result_t>)
expected<matrix<typename value_type_::hamming_result_t, allocator_type_>> hammings_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::hamming_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto workspace_bytes = hammings_symmetric_workspace_size<value_type_>(num_vectors, input.extent(1), 0, num_vectors,
                                                                          capabilities);
    if (!workspace_bytes) return {out_tensor_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto result = uninitialized_with_workspace_<out_tensor_t>({num_vectors, num_vectors}, workspace_bytes.value, alloc,
                                                              workspace);
    if (!result) return result;
    if (status_t status = result.value.span().fill_zeros(); failed(status)) return {out_tensor_t(alloc), status};
    if (status_t status = hammings_symmetric<value_type_>(input, result.value.span(), 0, num_vectors, workspace,
                                                          capabilities, stream);
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Symmetric Jaccard distances: C[i,j] = jaccard(A[i], A[j]), trailing like @c dots_symmetric.
 *  @c unexpected_dimensions_k unless the output is square over the input rows. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::jaccard_result_t> output_matrix_>
status_t jaccards_symmetric(input_matrix_ const &input, output_matrix_ &&output, std::size_t rows_begin = 0,
                            std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                            std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                            nk_stream_t stream = nullptr) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    if (input.stride_bytes(1) != sizeof(value_type_) || input.stride_bytes(0) < 0 ||
        output.stride_bytes(1) != sizeof(typename value_type_::jaccard_result_t) || output.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::jaccards_symmetric<value_type_>(
        reinterpret_cast<value_type_ const *>(input.data()), num_vectors, input.extent(1),
        static_cast<std::size_t>(input.stride_bytes(0)),
        reinterpret_cast<typename value_type_::jaccard_result_t *>(output.data()),
        static_cast<std::size_t>(output.stride_bytes(0)), rows_begin, rows_end, workspace, capabilities, stream);
}

/** Allocating symmetric Jaccard distances, with any workspace in the result's block. Empty for an
 *  empty input, or the allocation's or the kernel's failure. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::jaccard_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::jaccard_result_t>)
expected<matrix<typename value_type_::jaccard_result_t, allocator_type_>> jaccards_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::jaccard_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto workspace_bytes = jaccards_symmetric_workspace_size<value_type_>(num_vectors, input.extent(1), 0, num_vectors,
                                                                          capabilities);
    if (!workspace_bytes) return {out_tensor_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto result = uninitialized_with_workspace_<out_tensor_t>({num_vectors, num_vectors}, workspace_bytes.value, alloc,
                                                              workspace);
    if (!result) return result;
    if (status_t status = result.value.span().fill_zeros(); failed(status)) return {out_tensor_t(alloc), status};
    if (status_t status = jaccards_symmetric<value_type_>(input, result.value.span(), 0, num_vectors, workspace,
                                                          capabilities, stream);
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

#pragma endregion Concept Constrained Symmetric Dot Products

#pragma region Concept Constrained Packed Dot Products

/** Packed dot products: C = A × B_packedᵀ, taking the pointer overload's trailing parameters.
 *  @c unexpected_dimensions_k for an empty pack, a rank below 2 or a mismatched shape. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::dot_result_t> output_matrix_>
status_t dots_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c,
                     std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                     nk_stream_t stream = nullptr) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    if (a.stride_bytes(1) != sizeof(value_type_) || a.stride_bytes(0) < 0 ||
        c.stride_bytes(1) != sizeof(typename value_type_::dot_result_t) || c.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::dots_packed<value_type_>(
        reinterpret_cast<value_type_ const *>(a.data()), packed_b.data(),
        reinterpret_cast<typename value_type_::dot_result_t *>(c.data()), a.extent(0), packed_b.rows(),
        packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)), static_cast<std::size_t>(c.stride_bytes(0)),
        workspace, capabilities, stream);
}

/** Allocating packed dot products into one block with any workspace.
 *  @c unexpected_dimensions_k for an empty pack or a rank below 2, else the failure of the
 *  allocation or of the kernel itself. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::dot_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::dot_result_t>)
expected<matrix<typename value_type_::dot_result_t, allocator_type_>> dots_packed(
    input_matrix_ const &a, packed_type_ const &packed_b, allocator_type_ alloc = {},
    nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::dot_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto workspace_bytes = dots_packed_workspace_size<value_type_>(a.extent(0), packed_b.rows(), packed_b.depth(),
                                                                   capabilities);
    if (!workspace_bytes) return {out_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto c = uninitialized_with_workspace_<out_t>({a.extent(0), packed_b.rows()}, workspace_bytes.value, alloc,
                                                  workspace);
    if (!c) return c;
    if (status_t status = dots_packed<value_type_>(a, packed_b, c.value.as_matrix_span(), workspace, capabilities,
                                                   stream);
        failed(status))
        return {out_t(alloc), status};
    return c;
}

/** Packed Hamming distances: C = hamming(A, B_packed), trailing like @c dots_packed.
 *  @c unexpected_dimensions_k for an empty pack, a rank below 2 or a mismatched shape. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::hamming_result_t> output_matrix_>
status_t hammings_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c,
                         std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                         nk_stream_t stream = nullptr) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    if (a.stride_bytes(1) != sizeof(value_type_) || a.stride_bytes(0) < 0 ||
        c.stride_bytes(1) != sizeof(typename value_type_::hamming_result_t) || c.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::hammings_packed<value_type_>(
        reinterpret_cast<value_type_ const *>(a.data()), packed_b.data(),
        reinterpret_cast<typename value_type_::hamming_result_t *>(c.data()), a.extent(0), packed_b.rows(),
        packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)), static_cast<std::size_t>(c.stride_bytes(0)),
        workspace, capabilities, stream);
}

/** Allocating packed Hamming distances into one block with any workspace.
 *  @c unexpected_dimensions_k for an empty pack or a rank below 2, else the failure of the
 *  allocation or of the kernel itself. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::hamming_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::hamming_result_t>)
expected<matrix<typename value_type_::hamming_result_t, allocator_type_>> hammings_packed(
    input_matrix_ const &a, packed_type_ const &packed_b, allocator_type_ alloc = {},
    nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::hamming_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto workspace_bytes = hammings_packed_workspace_size<value_type_>(a.extent(0), packed_b.rows(), packed_b.depth(),
                                                                       capabilities);
    if (!workspace_bytes) return {out_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto c = uninitialized_with_workspace_<out_t>({a.extent(0), packed_b.rows()}, workspace_bytes.value, alloc,
                                                  workspace);
    if (!c) return c;
    if (status_t status = hammings_packed<value_type_>(a, packed_b, c.value.as_matrix_span(), workspace, capabilities,
                                                       stream);
        failed(status))
        return {out_t(alloc), status};
    return c;
}

/** Packed Jaccard distances: C = jaccard(A, B_packed), trailing like @c dots_packed.
 *  @c unexpected_dimensions_k for an empty pack, a rank below 2 or a mismatched shape. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::jaccard_result_t> output_matrix_>
status_t jaccards_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c,
                         std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                         nk_stream_t stream = nullptr) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    if (a.stride_bytes(1) != sizeof(value_type_) || a.stride_bytes(0) < 0 ||
        c.stride_bytes(1) != sizeof(typename value_type_::jaccard_result_t) || c.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::jaccards_packed<value_type_>(
        reinterpret_cast<value_type_ const *>(a.data()), packed_b.data(),
        reinterpret_cast<typename value_type_::jaccard_result_t *>(c.data()), a.extent(0), packed_b.rows(),
        packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)), static_cast<std::size_t>(c.stride_bytes(0)),
        workspace, capabilities, stream);
}

/** Allocating packed Jaccard distances into one block with any workspace.
 *  @c unexpected_dimensions_k for an empty pack or a rank below 2, else the failure of the
 *  allocation or of the kernel itself. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::jaccard_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::jaccard_result_t>)
expected<matrix<typename value_type_::jaccard_result_t, allocator_type_>> jaccards_packed(
    input_matrix_ const &a, packed_type_ const &packed_b, allocator_type_ alloc = {},
    nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::jaccard_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto workspace_bytes = jaccards_packed_workspace_size<value_type_>(a.extent(0), packed_b.rows(), packed_b.depth(),
                                                                       capabilities);
    if (!workspace_bytes) return {out_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto c = uninitialized_with_workspace_<out_t>({a.extent(0), packed_b.rows()}, workspace_bytes.value, alloc,
                                                  workspace);
    if (!c) return c;
    if (status_t status = jaccards_packed<value_type_>(a, packed_b, c.value.as_matrix_span(), workspace, capabilities,
                                                       stream);
        failed(status))
        return {out_t(alloc), status};
    return c;
}

#pragma endregion Concept Constrained Packed Dot Products

} // namespace ashvardanian::numkong

#endif // NUMKONG_DOTS_HPP
