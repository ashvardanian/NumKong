/**
 *  @file cpp/numkong/spatials.hpp
 *  @author Ash Vardanian
 *  @date March 6, 2026
 *  @brief C++ wrappers for SIMD-accelerated batched spatial distance matrices.
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_SPATIALS_HPP
#define NUMKONG_SPATIALS_HPP

#include <cstdint>

#include <limits>
#include <type_traits>

#include "numkong/spatials.h"

#include "numkong/tensor.hpp"
#include "numkong/types.hpp"
#include "numkong/devices.hpp" // `default_capabilities`

namespace ashvardanian::numkong {

/**
 *  @brief Returns the workspace bytes of @c angulars_symmetric, zero when no scratch is needed.
 *  @param[in] vector_count Number of vectors.
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] rows_begin First row of the result to compute, 0 by default.
 *  @param[in] rows_end Row after the last of the result, clamped to @p vector_count.
 *  @param[in] capabilities Capabilities to pick from
 */
template <a_numeric_dtype in_type_>
expected<std::size_t> angulars_symmetric_workspace_size(
    std::size_t vector_count, std::size_t depth, std::size_t rows_begin = 0,
    std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
    nk_capability_t capabilities = default_capabilities()) noexcept {
    if (rows_end > vector_count) rows_end = vector_count;
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_angulars_symmetric_workspace_size_f64_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                               &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_angulars_symmetric_workspace_size_f32_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                               &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_angulars_symmetric_workspace_size_f16_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                               &bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_angulars_symmetric_workspace_size_bf16_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_angulars_symmetric_workspace_size_i8_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                              &bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        status = nk_angulars_symmetric_workspace_size_u8_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                              &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_angulars_symmetric_workspace_size_e4m3_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        status = nk_angulars_symmetric_workspace_size_e5m2_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        status = nk_angulars_symmetric_workspace_size_e2m3_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        status = nk_angulars_symmetric_workspace_size_e2m1_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        status = nk_angulars_symmetric_workspace_size_e3m2_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        status = nk_angulars_symmetric_workspace_size_u4_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                              &bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        status = nk_angulars_symmetric_workspace_size_i4_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                              &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_angulars_symmetric_workspace_size_nvfp4_best(vector_count, depth, rows_begin, rows_end,
                                                                 capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_angulars_symmetric_workspace_size_mxfp4_best(vector_count, depth, rows_begin, rows_end,
                                                                 capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_angulars_symmetric_workspace_size_mxfp6e2m3_best(vector_count, depth, rows_begin, rows_end,
                                                                     capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_angulars_symmetric_workspace_size_mxfp6e3m2_best(vector_count, depth, rows_begin, rows_end,
                                                                     capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_angulars_symmetric_workspace_size_mxfp8e4m3_best(vector_count, depth, rows_begin, rows_end,
                                                                     capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_angulars_symmetric_workspace_size_mxfp8e5m2_best(vector_count, depth, rows_begin, rows_end,
                                                                     capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_angulars_symmetric_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** The workspace bytes of @c euclideans_symmetric, queried like the angular one. */
template <a_numeric_dtype in_type_>
expected<std::size_t> euclideans_symmetric_workspace_size(
    std::size_t vector_count, std::size_t depth, std::size_t rows_begin = 0,
    std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
    nk_capability_t capabilities = default_capabilities()) noexcept {
    if (rows_end > vector_count) rows_end = vector_count;
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_euclideans_symmetric_workspace_size_f64_best(vector_count, depth, rows_begin, rows_end,
                                                                 capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_euclideans_symmetric_workspace_size_f32_best(vector_count, depth, rows_begin, rows_end,
                                                                 capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_euclideans_symmetric_workspace_size_f16_best(vector_count, depth, rows_begin, rows_end,
                                                                 capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_euclideans_symmetric_workspace_size_bf16_best(vector_count, depth, rows_begin, rows_end,
                                                                  capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_euclideans_symmetric_workspace_size_i8_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        status = nk_euclideans_symmetric_workspace_size_u8_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_euclideans_symmetric_workspace_size_e4m3_best(vector_count, depth, rows_begin, rows_end,
                                                                  capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        status = nk_euclideans_symmetric_workspace_size_e5m2_best(vector_count, depth, rows_begin, rows_end,
                                                                  capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        status = nk_euclideans_symmetric_workspace_size_e2m3_best(vector_count, depth, rows_begin, rows_end,
                                                                  capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        status = nk_euclideans_symmetric_workspace_size_e2m1_best(vector_count, depth, rows_begin, rows_end,
                                                                  capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        status = nk_euclideans_symmetric_workspace_size_e3m2_best(vector_count, depth, rows_begin, rows_end,
                                                                  capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        status = nk_euclideans_symmetric_workspace_size_u4_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        status = nk_euclideans_symmetric_workspace_size_i4_best(vector_count, depth, rows_begin, rows_end, capabilities,
                                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_euclideans_symmetric_workspace_size_nvfp4_best(vector_count, depth, rows_begin, rows_end,
                                                                   capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_euclideans_symmetric_workspace_size_mxfp4_best(vector_count, depth, rows_begin, rows_end,
                                                                   capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_euclideans_symmetric_workspace_size_mxfp6e2m3_best(vector_count, depth, rows_begin, rows_end,
                                                                       capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_euclideans_symmetric_workspace_size_mxfp6e3m2_best(vector_count, depth, rows_begin, rows_end,
                                                                       capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_euclideans_symmetric_workspace_size_mxfp8e4m3_best(vector_count, depth, rows_begin, rows_end,
                                                                       capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_euclideans_symmetric_workspace_size_mxfp8e5m2_best(vector_count, depth, rows_begin, rows_end,
                                                                       capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_euclideans_symmetric_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** The workspace bytes of @c angulars_packed, queried like @c angulars_symmetric_workspace_size. */
template <a_numeric_dtype in_type_>
expected<std::size_t> angulars_packed_workspace_size(std::size_t row_count, std::size_t column_count, std::size_t depth,
                                                     nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_angulars_packed_workspace_size_f64_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_angulars_packed_workspace_size_f32_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_angulars_packed_workspace_size_f16_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_angulars_packed_workspace_size_bf16_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_angulars_packed_workspace_size_i8_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        status = nk_angulars_packed_workspace_size_u8_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_angulars_packed_workspace_size_e4m3_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        status = nk_angulars_packed_workspace_size_e5m2_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        status = nk_angulars_packed_workspace_size_e2m3_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        status = nk_angulars_packed_workspace_size_e2m1_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        status = nk_angulars_packed_workspace_size_e3m2_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        status = nk_angulars_packed_workspace_size_u4_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        status = nk_angulars_packed_workspace_size_i4_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_angulars_packed_workspace_size_nvfp4_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_angulars_packed_workspace_size_mxfp4_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_angulars_packed_workspace_size_mxfp6e2m3_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_angulars_packed_workspace_size_mxfp6e3m2_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_angulars_packed_workspace_size_mxfp8e4m3_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_angulars_packed_workspace_size_mxfp8e5m2_best(row_count, column_count, depth, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_angulars_packed_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/** The workspace bytes of @c euclideans_packed, queried like the angular one. */
template <a_numeric_dtype in_type_>
expected<std::size_t> euclideans_packed_workspace_size(std::size_t row_count, std::size_t column_count,
                                                       std::size_t depth,
                                                       nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_euclideans_packed_workspace_size_f64_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_euclideans_packed_workspace_size_f32_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_euclideans_packed_workspace_size_f16_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_euclideans_packed_workspace_size_bf16_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_euclideans_packed_workspace_size_i8_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        status = nk_euclideans_packed_workspace_size_u8_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_euclideans_packed_workspace_size_e4m3_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        status = nk_euclideans_packed_workspace_size_e5m2_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        status = nk_euclideans_packed_workspace_size_e2m3_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        status = nk_euclideans_packed_workspace_size_e2m1_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        status = nk_euclideans_packed_workspace_size_e3m2_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        status = nk_euclideans_packed_workspace_size_u4_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        status = nk_euclideans_packed_workspace_size_i4_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_euclideans_packed_workspace_size_nvfp4_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_euclideans_packed_workspace_size_mxfp4_best(row_count, column_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_euclideans_packed_workspace_size_mxfp6e2m3_best(row_count, column_count, depth, capabilities,
                                                                    &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_euclideans_packed_workspace_size_mxfp6e3m2_best(row_count, column_count, depth, capabilities,
                                                                    &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_euclideans_packed_workspace_size_mxfp8e4m3_best(row_count, column_count, depth, capabilities,
                                                                    &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_euclideans_packed_workspace_size_mxfp8e5m2_best(row_count, column_count, depth, capabilities,
                                                                    &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_euclideans_packed_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Symmetric Angular distance matrix: C[i,j] = angular(A[i], A[j])
 *  @param[in] a Matrix A, shape @b [vector_count,depth], see @c operand_pointer
 *  @param[in] vector_count Number of vectors, n
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride Stride between vectors in A
 *  @param[out] c Output matrix C, shape @b [n,n]
 *  @param[in] c_stride Stride between rows of C in bytes
 *  @param[in] rows_begin First row of C to compute, 0 by default
 *  @param[in] rows_end Row of C after the last, clamped to @p vector_count, all by default
 *
 *  @param[in] workspace Scratch aligned to @c nk_default_alignment_k, sized by the matching query,
 *      or empty for the scratch-free path
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input type, a plain dtype or a block-scaled format
 */
template <a_numeric_dtype in_type_>
status_t angulars_symmetric(operand_pointer<in_type_> a, std::size_t vector_count, std::size_t depth,
                            std::size_t a_stride, typename in_type_::angular_result_t *c, std::size_t c_stride,
                            std::size_t rows_begin = 0,
                            std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                            std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                            nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_angulars_symmetric_f64_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                    c_stride, rows_begin, rows_end, workspace.data(),
                                                                    workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_angulars_symmetric_f32_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                    c_stride, rows_begin, rows_end, workspace.data(),
                                                                    workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_angulars_symmetric_f16_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                    c_stride, rows_begin, rows_end, workspace.data(),
                                                                    workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_angulars_symmetric_bf16_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_angulars_symmetric_i8_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                   c_stride, rows_begin, rows_end, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_angulars_symmetric_u8_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                   c_stride, rows_begin, rows_end, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_angulars_symmetric_e4m3_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(nk_angulars_symmetric_e5m2_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(nk_angulars_symmetric_e2m3_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        return static_cast<status_t>(nk_angulars_symmetric_e2m1_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(nk_angulars_symmetric_e3m2_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_angulars_symmetric_u4_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                   c_stride, rows_begin, rows_end, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_angulars_symmetric_i4_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                   c_stride, rows_begin, rows_end, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_angulars_symmetric_nvfp4_best(a, vector_count, depth, a_stride, &c->raw_,
                                                                      c_stride, rows_begin, rows_end, workspace.data(),
                                                                      workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_angulars_symmetric_mxfp4_best(a, vector_count, depth, a_stride, &c->raw_,
                                                                      c_stride, rows_begin, rows_end, workspace.data(),
                                                                      workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(
            nk_angulars_symmetric_mxfp6e2m3_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                                 rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(
            nk_angulars_symmetric_mxfp6e3m2_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                                 rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(
            nk_angulars_symmetric_mxfp8e4m3_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                                 rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(
            nk_angulars_symmetric_mxfp8e5m2_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                                 rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_angulars_symmetric has no kernel for this dtype");
}

/**
 *  @brief Symmetric Euclidean distance matrix: C[i,j] = euclidean(A[i], A[j])
 *  @param[in] a Matrix A, shape @b [vector_count,depth], see @c operand_pointer
 *  @param[in] vector_count Number of vectors, n
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] a_stride Stride between vectors in A
 *  @param[out] c Output matrix C, shape @b [n,n]
 *  @param[in] c_stride Stride between rows of C in bytes
 *  @param[in] rows_begin First row of C to compute, 0 by default
 *  @param[in] rows_end Row of C after the last, clamped to @p vector_count, all by default
 *
 *  @param[in] workspace Scratch aligned to @c nk_default_alignment_k, sized by the matching query,
 *      or empty for the scratch-free path
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input type, a plain dtype or a block-scaled format
 */
template <a_numeric_dtype in_type_>
status_t euclideans_symmetric(operand_pointer<in_type_> a, std::size_t vector_count, std::size_t depth,
                              std::size_t a_stride, typename in_type_::euclidean_result_t *c, std::size_t c_stride,
                              std::size_t rows_begin = 0,
                              std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                              std::span<std::byte> workspace = {},
                              nk_capability_t capabilities = default_capabilities(),
                              nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_euclideans_symmetric_f64_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                      c_stride, rows_begin, rows_end, workspace.data(),
                                                                      workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_euclideans_symmetric_f32_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                      c_stride, rows_begin, rows_end, workspace.data(),
                                                                      workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_euclideans_symmetric_f16_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                      c_stride, rows_begin, rows_end, workspace.data(),
                                                                      workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_bf16_best(&a->raw_, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                              rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_euclideans_symmetric_i8_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_euclideans_symmetric_u8_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_e4m3_best(&a->raw_, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                              rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_e5m2_best(&a->raw_, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                              rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_e2m3_best(&a->raw_, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                              rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_e2m1_best(&a->raw_, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                              rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_e3m2_best(&a->raw_, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                              rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_euclideans_symmetric_u4_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_euclideans_symmetric_i4_best(&a->raw_, vector_count, depth, a_stride, &c->raw_,
                                                                     c_stride, rows_begin, rows_end, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_nvfp4_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                               rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_mxfp4_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                               rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_mxfp6e2m3_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                                   rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_mxfp6e3m2_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                                   rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_mxfp8e4m3_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                                   rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(
            nk_euclideans_symmetric_mxfp8e5m2_best(a, vector_count, depth, a_stride, &c->raw_, c_stride, rows_begin,
                                                   rows_end, workspace.data(), workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_euclideans_symmetric has no kernel for this dtype");
}

/**
 *  @brief Packed angular distances: C = angular(A, B_packed)
 *  @param[in] a Matrix A, shape @b [row_count,depth], see @c operand_pointer
 *  @param[in] b_packed Packed B matrix, produced by @c dots_pack with the same @p capabilities
 *  @param[out] c Output matrix C, shape @b [row_count,column_count]
 *  @param[in] row_count Rows of A and C, m
 *  @param[in] column_count Columns of B and C, n
 *  @param[in] depth Shared inner dimension, k, in multiples of the values per byte.
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
status_t angulars_packed(operand_pointer<in_type_> a, void const *b_packed, typename in_type_::angular_result_t *c,
                         std::size_t row_count, std::size_t column_count, std::size_t depth, std::size_t a_stride,
                         std::size_t c_stride, std::span<std::byte> workspace = {},
                         nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(nk_angulars_packed_f64_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                 depth, a_stride, c_stride, workspace.data(),
                                                                 workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_angulars_packed_f32_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                 depth, a_stride, c_stride, workspace.data(),
                                                                 workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_angulars_packed_f16_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                 depth, a_stride, c_stride, workspace.data(),
                                                                 workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_angulars_packed_bf16_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_angulars_packed_i8_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                depth, a_stride, c_stride, workspace.data(),
                                                                workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_angulars_packed_u8_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                depth, a_stride, c_stride, workspace.data(),
                                                                workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_angulars_packed_e4m3_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(nk_angulars_packed_e5m2_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(nk_angulars_packed_e2m3_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        return static_cast<status_t>(nk_angulars_packed_e2m1_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(nk_angulars_packed_e3m2_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_angulars_packed_u4_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                depth, a_stride, c_stride, workspace.data(),
                                                                workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_angulars_packed_i4_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                depth, a_stride, c_stride, workspace.data(),
                                                                workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_angulars_packed_nvfp4_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                   depth, a_stride, c_stride, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_angulars_packed_mxfp4_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                   depth, a_stride, c_stride, workspace.data(),
                                                                   workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(nk_angulars_packed_mxfp6e2m3_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                       depth, a_stride, c_stride, workspace.data(),
                                                                       workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(nk_angulars_packed_mxfp6e3m2_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                       depth, a_stride, c_stride, workspace.data(),
                                                                       workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(nk_angulars_packed_mxfp8e4m3_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                       depth, a_stride, c_stride, workspace.data(),
                                                                       workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(nk_angulars_packed_mxfp8e5m2_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                       depth, a_stride, c_stride, workspace.data(),
                                                                       workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_angulars_packed has no kernel for this dtype");
}

/**
 *  @brief Packed euclidean distances: C = euclidean(A, B_packed)
 *  @param[in] a Matrix A, shape @b [row_count,depth], see @c operand_pointer
 *  @param[in] b_packed Packed B matrix, produced by @c dots_pack with the same @p capabilities
 *  @param[out] c Output matrix C, shape @b [row_count,column_count]
 *  @param[in] row_count Rows of A and C, m
 *  @param[in] column_count Columns of B and C, n
 *  @param[in] depth Shared inner dimension, k, in multiples of the values per byte.
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
status_t euclideans_packed(operand_pointer<in_type_> a, void const *b_packed, typename in_type_::euclidean_result_t *c,
                           std::size_t row_count, std::size_t column_count, std::size_t depth, std::size_t a_stride,
                           std::size_t c_stride, std::span<std::byte> workspace = {},
                           nk_capability_t capabilities = default_capabilities(),
                           nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(
            nk_euclideans_packed_f64_best(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride,
                                          c_stride, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(
            nk_euclideans_packed_f32_best(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride,
                                          c_stride, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(
            nk_euclideans_packed_f16_best(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride,
                                          c_stride, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(
            nk_euclideans_packed_bf16_best(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride,
                                           c_stride, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_euclideans_packed_i8_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_euclideans_packed_u8_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(
            nk_euclideans_packed_e4m3_best(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride,
                                           c_stride, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(
            nk_euclideans_packed_e5m2_best(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride,
                                           c_stride, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(
            nk_euclideans_packed_e2m3_best(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride,
                                           c_stride, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        return static_cast<status_t>(
            nk_euclideans_packed_e2m1_best(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride,
                                           c_stride, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(
            nk_euclideans_packed_e3m2_best(&a->raw_, b_packed, &c->raw_, row_count, column_count, depth, a_stride,
                                           c_stride, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(nk_euclideans_packed_u4_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(nk_euclideans_packed_i4_best(&a->raw_, b_packed, &c->raw_, row_count, column_count,
                                                                  depth, a_stride, c_stride, workspace.data(),
                                                                  workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_euclideans_packed_nvfp4_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                     depth, a_stride, c_stride, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_euclideans_packed_mxfp4_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                     depth, a_stride, c_stride, workspace.data(),
                                                                     workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(nk_euclideans_packed_mxfp6e2m3_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                         depth, a_stride, c_stride, workspace.data(),
                                                                         workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(nk_euclideans_packed_mxfp6e3m2_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                         depth, a_stride, c_stride, workspace.data(),
                                                                         workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(nk_euclideans_packed_mxfp8e4m3_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                         depth, a_stride, c_stride, workspace.data(),
                                                                         workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(nk_euclideans_packed_mxfp8e5m2_best(a, b_packed, &c->raw_, row_count, column_count,
                                                                         depth, a_stride, c_stride, workspace.data(),
                                                                         workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_euclideans_packed has no kernel for this dtype");
}

#pragma region Concept Constrained Symmetric Spatial Distances

/** Symmetric angular distances: C[i,j] = angular(A[i], A[j]), with the pointer overload's tail.
 *  @c unexpected_dimensions_k unless the output is square over the input rows. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::angular_result_t> output_matrix_>
status_t angulars_symmetric(input_matrix_ const &input, output_matrix_ &&output, std::size_t rows_begin = 0,
                            std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                            std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                            nk_stream_t stream = nullptr) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    if (input.stride_bytes(1) != sizeof(value_type_) || input.stride_bytes(0) < 0 ||
        output.stride_bytes(1) != sizeof(typename value_type_::angular_result_t) || output.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::angulars_symmetric<value_type_>(
        reinterpret_cast<value_type_ const *>(input.data()), num_vectors, input.extent(1),
        static_cast<std::size_t>(input.stride_bytes(0)),
        reinterpret_cast<typename value_type_::angular_result_t *>(output.data()),
        static_cast<std::size_t>(output.stride_bytes(0)), rows_begin, rows_end, workspace, capabilities, stream);
}

/** Allocating symmetric angular distances, with any workspace in the result's block. Empty for an
 *  empty input, or the allocation's or the kernel's failure. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::angular_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::angular_result_t>)
expected<matrix<typename value_type_::angular_result_t, allocator_type_>> angulars_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::angular_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto workspace_bytes = angulars_symmetric_workspace_size<value_type_>(num_vectors, input.extent(1), 0, num_vectors,
                                                                          capabilities);
    if (!workspace_bytes) return {out_tensor_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto result = uninitialized_with_workspace_<out_tensor_t>({num_vectors, num_vectors}, workspace_bytes.value, alloc,
                                                              workspace);
    if (!result) return result;
    if (status_t status = result.value.span().fill_zeros(); failed(status)) return {out_tensor_t(alloc), status};
    if (status_t status = angulars_symmetric<value_type_>(input, result.value.span(), 0, num_vectors, workspace,
                                                          capabilities, stream);
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Symmetric Euclidean distances: C[i,j] = euclidean(A[i], A[j]), trailing like the angular one.
 *  @c unexpected_dimensions_k unless the output is square over the input rows. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::euclidean_result_t> output_matrix_>
status_t euclideans_symmetric(input_matrix_ const &input, output_matrix_ &&output, std::size_t rows_begin = 0,
                              std::size_t rows_end = (std::numeric_limits<std::size_t>::max)(),
                              std::span<std::byte> workspace = {},
                              nk_capability_t capabilities = default_capabilities(),
                              nk_stream_t stream = nullptr) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return status_t::unexpected_dimensions_k;
    if (input.stride_bytes(1) != sizeof(value_type_) || input.stride_bytes(0) < 0 ||
        output.stride_bytes(1) != sizeof(typename value_type_::euclidean_result_t) || output.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::euclideans_symmetric<value_type_>(
        reinterpret_cast<value_type_ const *>(input.data()), num_vectors, input.extent(1),
        static_cast<std::size_t>(input.stride_bytes(0)),
        reinterpret_cast<typename value_type_::euclidean_result_t *>(output.data()),
        static_cast<std::size_t>(output.stride_bytes(0)), rows_begin, rows_end, workspace, capabilities, stream);
}

/** Allocating symmetric Euclidean distances, with any workspace in the result's block. Empty for an
 *  empty input, or the allocation's or the kernel's failure. */
template <a_numeric_dtype value_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::euclidean_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::euclidean_result_t>)
expected<matrix<typename value_type_::euclidean_result_t, allocator_type_>> euclideans_symmetric(
    input_matrix_ const &input, allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::euclidean_result_t;
    using out_tensor_t = matrix<result_t, allocator_type_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    std::size_t num_vectors = input.extent(0);
    auto workspace_bytes = euclideans_symmetric_workspace_size<value_type_>(num_vectors, input.extent(1), 0,
                                                                            num_vectors, capabilities);
    if (!workspace_bytes) return {out_tensor_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto result = uninitialized_with_workspace_<out_tensor_t>({num_vectors, num_vectors}, workspace_bytes.value, alloc,
                                                              workspace);
    if (!result) return result;
    if (status_t status = result.value.span().fill_zeros(); failed(status)) return {out_tensor_t(alloc), status};
    if (status_t status = euclideans_symmetric<value_type_>(input, result.value.span(), 0, num_vectors, workspace,
                                                            capabilities, stream);
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

#pragma endregion Concept Constrained Symmetric Spatial Distances

#pragma region Concept Constrained Packed Spatial Distances

/** Packed angular distances: C = angular(A, B_packed), taking the pointer overload's trailing
 *  parameters. @c unexpected_dimensions_k for an empty pack, a rank below 2 or a shape mismatch. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::angular_result_t> output_matrix_>
status_t angulars_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c,
                         std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                         nk_stream_t stream = nullptr) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    if (a.stride_bytes(1) != sizeof(value_type_) || a.stride_bytes(0) < 0 ||
        c.stride_bytes(1) != sizeof(typename value_type_::angular_result_t) || c.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::angulars_packed<value_type_>(
        reinterpret_cast<value_type_ const *>(a.data()), packed_b.data(),
        reinterpret_cast<typename value_type_::angular_result_t *>(c.data()), a.extent(0), packed_b.rows(),
        packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)), static_cast<std::size_t>(c.stride_bytes(0)),
        workspace, capabilities, stream);
}

/** Allocating packed angular distances into one block with any workspace.
 *  @c unexpected_dimensions_k for an empty pack or a rank below 2, else the failure of the
 *  allocation or of the kernel itself. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::angular_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::angular_result_t>)
expected<matrix<typename value_type_::angular_result_t, allocator_type_>> angulars_packed(
    input_matrix_ const &a, packed_type_ const &packed_b, allocator_type_ alloc = {},
    nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::angular_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto workspace_bytes = angulars_packed_workspace_size<value_type_>(a.extent(0), packed_b.rows(), packed_b.depth(),
                                                                       capabilities);
    if (!workspace_bytes) return {out_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto c = uninitialized_with_workspace_<out_t>({a.extent(0), packed_b.rows()}, workspace_bytes.value, alloc,
                                                  workspace);
    if (!c) return c;
    if (status_t status = angulars_packed<value_type_>(a, packed_b, c.value.as_matrix_span(), workspace, capabilities,
                                                       stream);
        failed(status))
        return {out_t(alloc), status};
    return c;
}

/** Packed Euclidean distances: C = euclidean(A, B_packed), trailing like the angular one.
 *  @c unexpected_dimensions_k for an empty pack, a rank below 2 or a mismatched shape. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          a_mutable_matrix_of<typename value_type_::euclidean_result_t> output_matrix_>
status_t euclideans_packed(input_matrix_ const &a, packed_type_ const &packed_b, output_matrix_ &&c,
                           std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                           nk_stream_t stream = nullptr) noexcept {
    if (packed_b.empty() || a.rank() < 2 || c.rank() < 2) return status_t::unexpected_dimensions_k;
    if (a.extent(1) != packed_b.depth()) return status_t::unexpected_dimensions_k;
    if (c.extent(0) != a.extent(0) || c.extent(1) != packed_b.rows()) return status_t::unexpected_dimensions_k;
    if (a.stride_bytes(1) != sizeof(value_type_) || a.stride_bytes(0) < 0 ||
        c.stride_bytes(1) != sizeof(typename value_type_::euclidean_result_t) || c.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    return numkong::euclideans_packed<value_type_>(
        reinterpret_cast<value_type_ const *>(a.data()), packed_b.data(),
        reinterpret_cast<typename value_type_::euclidean_result_t *>(c.data()), a.extent(0), packed_b.rows(),
        packed_b.depth(), static_cast<std::size_t>(a.stride_bytes(0)), static_cast<std::size_t>(c.stride_bytes(0)),
        workspace, capabilities, stream);
}

/** Allocating packed Euclidean distances into one block with any workspace.
 *  @c unexpected_dimensions_k for an empty pack or a rank below 2, else the failure of the
 *  allocation or of the kernel itself. */
template <a_numeric_dtype value_type_, a_packed_matrix packed_type_, a_const_matrix_of<value_type_> input_matrix_,
          typename allocator_type_ = aligned_allocator<typename value_type_::euclidean_result_t>>
    requires(!a_mutable_matrix_of<allocator_type_, typename value_type_::euclidean_result_t>)
expected<matrix<typename value_type_::euclidean_result_t, allocator_type_>> euclideans_packed(
    input_matrix_ const &a, packed_type_ const &packed_b, allocator_type_ alloc = {},
    nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::euclidean_result_t;
    using out_t = matrix<result_t, allocator_type_>;
    if (packed_b.empty() || a.rank() < 2) return {out_t(alloc), status_t::unexpected_dimensions_k};
    auto workspace_bytes = euclideans_packed_workspace_size<value_type_>(a.extent(0), packed_b.rows(), packed_b.depth(),
                                                                         capabilities);
    if (!workspace_bytes) return {out_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto c = uninitialized_with_workspace_<out_t>({a.extent(0), packed_b.rows()}, workspace_bytes.value, alloc,
                                                  workspace);
    if (!c) return c;
    if (status_t status = euclideans_packed<value_type_>(a, packed_b, c.value.as_matrix_span(), workspace, capabilities,
                                                         stream);
        failed(status))
        return {out_t(alloc), status};
    return c;
}

#pragma endregion Concept Constrained Packed Spatial Distances

} // namespace ashvardanian::numkong

#endif // NUMKONG_SPATIALS_HPP
