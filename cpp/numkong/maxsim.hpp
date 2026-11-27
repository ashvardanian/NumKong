/**
 *  @file cpp/numkong/maxsim.hpp
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *  @brief C++ bindings for multi-target MaxSim, ColBERT late-interaction kernels.
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_MAXSIM_HPP
#define NUMKONG_MAXSIM_HPP

#include <cstddef>
#include <type_traits>

#include "numkong/maxsim.h"
#include "numkong/devices.hpp" // `default_capabilities`
#include "numkong/matrix.hpp"  // `packed_maxsim`

namespace ashvardanian::numkong {

/**
 *  @brief Estimates the memory requirements for a maxsim packed vector set.
 *  @param[in] vector_count Number of vectors to pack.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[in] capabilities Capabilities to pick from.
 *  @return Size in bytes of the pack @c maxsim_pack writes with the same @p capabilities.
 *
 *  @tparam in_type_ Input element type (bf16_t, f32_t, f16_t).
 */
template <a_numeric_dtype in_type_>
expected<std::size_t> maxsim_pack_size(std::size_t vector_count, std::size_t depth,
                                       nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_maxsim_pack_size_bf16_best(vector_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_maxsim_pack_size_f32_best(vector_count, depth, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_maxsim_pack_size_f16_best(vector_count, depth, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_maxsim_pack_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs vectors into a backend-specific layout for maxsim computation.
 *  @param[in] vectors Input vectors in row-major order. Packing fuses no transposition, so
 *      depth-major vectors are transposed in a separate pass first.
 *  @param[in] vector_count Number of vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[in] stride Row stride in bytes for the input vectors.
 *  @param[out] packed Output packed buffer from maxsim_pack_size.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input element type (bf16_t, f32_t, f16_t).
 */
template <a_numeric_dtype in_type_>
status_t maxsim_pack(typename in_type_::raw_t const *vectors, std::size_t vector_count, std::size_t depth,
                     std::size_t stride, void *packed, nk_capability_t capabilities = default_capabilities(),
                     nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(
            nk_maxsim_pack_bf16_best(vectors, vector_count, depth, stride, packed, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(
            nk_maxsim_pack_f32_best(vectors, vector_count, depth, stride, packed, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(
            nk_maxsim_pack_f16_best(vectors, vector_count, depth, stride, packed, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_maxsim_pack has no kernel for this dtype");
}

/** Packs a 2D matrix of vectors into an owner for @c maxsim, for the @p capabilities scoring will
 *  use; fails like @c packed_maxsim::make. */
template <a_numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
expected<packed_maxsim<value_type_, allocator_type_>> maxsim_pack(matrix_view<value_type_> vectors,
                                                                  allocator_type_ alloc = {},
                                                                  nk_capability_t capabilities = default_capabilities(),
                                                                  nk_stream_t stream = nullptr) noexcept {
    auto const pack_size = [capabilities](nk_size_t count, nk_size_t depth, nk_size_t *bytes) noexcept {
        auto const [size, status] = maxsim_pack_size<value_type_>(count, depth, capabilities);
        *bytes = size;
        return status;
    };
    auto const pack = [capabilities, stream](typename value_type_::raw_t const *values, nk_size_t count,
                                             nk_size_t depth, nk_size_t stride, void *packed) noexcept {
        return maxsim_pack<value_type_>(values, count, depth, stride, packed, capabilities, stream);
    };
    return packed_maxsim<value_type_, allocator_type_>::make(vectors, alloc, pack_size, pack);
}

/**
 *  @brief Computes angular distance late-interaction on pre-packed vectors.
 *
 *  Returns Σᵢ minⱼ angular(qᵢ, dⱼ).
 *
 *  @param[in] query_packed Query vectors from @c maxsim_pack for the same @p capabilities.
 *  @param[in] document_packed Document vectors from @c maxsim_pack for the same @p capabilities.
 *  @param[in] query_count Number of query vectors.
 *  @param[in] document_count Number of document vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[out] result Sum of per-query minimum angular distances.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input element type: @c bf16_t, @c f32_t or @c f16_t.
 */
template <a_numeric_dtype in_type_>
status_t maxsim_packed(void const *query_packed, void const *document_packed, std::size_t query_count,
                       std::size_t document_count, std::size_t depth, typename in_type_::maxsim_result_t *result,
                       nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_maxsim_packed_bf16_best(
            query_packed, document_packed, query_count, document_count, depth, &result->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_maxsim_packed_f32_best(
            query_packed, document_packed, query_count, document_count, depth, &result->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_maxsim_packed_f16_best(
            query_packed, document_packed, query_count, document_count, depth, &result->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_maxsim_packed has no kernel for this dtype");
}

/** MaxSim: Σᵢ minⱼ angular(qᵢ, dⱼ) on vectors packed for the same @p capabilities;
 *  @c unexpected_dimensions_k for an empty pack or mismatched depths. */
template <a_numeric_dtype value_type_>
expected<typename value_type_::maxsim_result_t> maxsim(packed_maxsim<value_type_> const &queries,
                                                       packed_maxsim<value_type_> const &documents,
                                                       nk_capability_t capabilities = default_capabilities(),
                                                       nk_stream_t stream = nullptr) noexcept {
    using result_t = typename value_type_::maxsim_result_t;
    result_t result {};
    if (queries.empty() || documents.empty() || queries.depth() != documents.depth())
        return {result, status_t::unexpected_dimensions_k};
    status_t status = maxsim_packed<value_type_>(queries.data(), documents.data(), queries.vector_count(),
                                                 documents.vector_count(), queries.depth(), &result, capabilities,
                                                 stream);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_MAXSIM_HPP
