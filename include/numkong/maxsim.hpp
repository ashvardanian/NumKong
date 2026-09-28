/**
 *  @file include/numkong/maxsim.hpp
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *  @brief C++ bindings for multi-target MaxSim, ColBERT late-interaction kernels.
 */
#ifndef NUMKONG_MAXSIM_HPP
#define NUMKONG_MAXSIM_HPP

#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>

#include "numkong/maxsim.h"
#include "numkong/types.hpp"
#include "numkong/spatial.hpp" // angular<>

namespace ashvardanian::numkong {

/**
 *  @brief Computes angular distance late-interaction on pre-packed vectors.
 *
 *  Returns Σᵢ minⱼ angular(qᵢ, dⱼ).
 *
 *  @param[in] query_packed Packed query vectors.
 *  @param[in] document_packed Packed document vectors.
 *  @param[in] query_count Number of query vectors.
 *  @param[in] document_count Number of document vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[out] result Sum of per-query minimum angular distances.
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template, which reads packs of
 *      the same zero mask.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input element type: @c bf16_t, @c f32_t or @c f16_t.
 *  @tparam result_type_ Result type, defaults to @c in_type_::maxsim_result_t.
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::maxsim_result_t>
nk_status_t maxsim_packed(void const *query_packed, void const *document_packed, std::size_t query_count,
                          std::size_t document_count, std::size_t depth, result_type_ *result,
                          nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) {
    constexpr bool dispatch = std::is_same_v<result_type_, typename in_type_::maxsim_result_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return nk_maxsim_packed_bf16_best(query_packed, document_packed, query_count, document_count, depth,
                                              &result->raw_, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return nk_maxsim_packed_f32_best(query_packed, document_packed, query_count, document_count, depth,
                                             &result->raw_, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return nk_maxsim_packed_f16_best(query_packed, document_packed, query_count, document_count, depth,
                                             &result->raw_, capabilities, stream);
    }
    typename in_type_::raw_t const *q_ptr;
    std::size_t q_stride;
    char const *q_bytes = reinterpret_cast<char const *>(query_packed);
    std::memcpy(&q_ptr, q_bytes, sizeof(void *));
    std::memcpy(&q_stride, q_bytes + sizeof(void *), sizeof(std::size_t));

    typename in_type_::raw_t const *d_ptr;
    std::size_t d_stride;
    char const *d_bytes = reinterpret_cast<char const *>(document_packed);
    std::memcpy(&d_ptr, d_bytes, sizeof(void *));
    std::memcpy(&d_stride, d_bytes + sizeof(void *), sizeof(std::size_t));

    return maxsim_reference<in_type_, result_type_>(q_ptr, query_count, q_stride, d_ptr, document_count, d_stride,
                                                    depth, result, capabilities);
}

/**
 *  @brief Exhaustive angular reference for testing: Σᵢ minⱼ angular(qᵢ, dⱼ).
 *
 *  Computes all pairwise angular distances and picks the minimum per query.
 *  Uses f64 accumulator for precision.
 *
 *  @param[in] queries Query vectors in row-major order.
 *  @param[in] query_count Number of query vectors.
 *  @param[in] query_stride Row stride in bytes for query vectors.
 *  @param[in] documents Document vectors in row-major order.
 *  @param[in] document_count Number of document vectors.
 *  @param[in] document_stride Row stride in bytes for document vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[out] result Pointer to store the sum of per-query minimum angular distances.
 *  @param[in] capabilities Capabilities the angular distances pick from, or zero for the C++ template.
 *
 *  @tparam in_type_ Input element type: @c bf16_t, @c f32_t or @c f16_t.
 *  @tparam result_type_ Result type, defaults to @c in_type_::angular_result_t.
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::angular_result_t>
nk_status_t maxsim_reference(typename in_type_::raw_t const *queries, std::size_t query_count, std::size_t query_stride,
                             typename in_type_::raw_t const *documents, std::size_t document_count,
                             std::size_t document_stride, std::size_t depth, result_type_ *result,
                             nk_capability_t capabilities = cpu_capabilities()) {
    result_type_ total_angular_distance {};

    for (std::size_t query_index = 0; query_index < query_count; query_index++) {
        in_type_ const *query_row = reinterpret_cast<in_type_ const *>(reinterpret_cast<char const *>(queries) +
                                                                       query_index * query_stride);

        result_type_ min_angular = result_type_::finite_max();

        for (std::size_t document_index = 0; document_index < document_count; document_index++) {
            in_type_ const *document_row = reinterpret_cast<in_type_ const *>(
                reinterpret_cast<char const *>(documents) + document_index * document_stride);

            result_type_ angular_distance {};
            nk_status_t const status = angular<in_type_, result_type_>(query_row, document_row, depth,
                                                                       &angular_distance, capabilities);
            if (status != nk_success_k) return status;

            if (angular_distance < min_angular) min_angular = angular_distance;
        }

        total_angular_distance = total_angular_distance + min_angular;
    }

    *result = total_angular_distance;
    return nk_success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/matrix.hpp"

namespace ashvardanian::numkong {

/** MaxSim: Σᵢ minⱼ angular(qᵢ, dⱼ) on pre-packed vectors. */
template <numeric_dtype value_type_>
typename value_type_::maxsim_result_t maxsim(packed_maxsim<value_type_> const &queries,
                                             packed_maxsim<value_type_> const &documents) noexcept {
    using result_t = typename value_type_::maxsim_result_t;
    result_t result {};
    if (queries.empty() || documents.empty()) return result;
    if (queries.depth() != documents.depth()) return result;
    maxsim_packed<value_type_>(queries.data(), documents.data(), queries.vector_count(), documents.vector_count(),
                               queries.depth(), &result);
    return result;
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_MAXSIM_HPP
