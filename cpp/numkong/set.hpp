/**
 *  @file cpp/numkong/set.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for set-intersection kernels.
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_SET_HPP
#define NUMKONG_SET_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/set.h"
#include "numkong/sets.h"
#include "numkong/tensor.hpp"
#include "numkong/types.hpp"
#include "numkong/devices.hpp" // `default_capabilities`

namespace ashvardanian::numkong {

/**
 *  @brief Hamming distance: Σ(aᵢ ⊕ bᵢ)
 *  @param[in] a First input vector
 *  @param[in] b Second input vector
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output count
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type (u1x8_t or u8_t)
 */
template <a_numeric_dtype in_type_>
status_t hamming(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::hamming_result_t *r,
                 nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(nk_hamming_u1_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(nk_hamming_u8_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_hamming has no kernel for this dtype");
}

/**
 *  @brief Jaccard distance: 1 − |A ∩ B| / |A ∪ B|
 *  @param[in] a First input vector
 *  @param[in] b Second input vector
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[out] r Pointer to output distance
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  For u1x8_t bit vectors, uses popcount(AND) / popcount(OR). For u16_t/u32_t element vectors, uses
 *  count of matching elements / total.
 *
 *  @tparam in_type_ Input vector element type (u1x8_t, u16_t, or u32_t)
 */
template <a_numeric_dtype in_type_>
status_t jaccard(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::jaccard_result_t *r,
                 nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(nk_jaccard_u1_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u16_t>)
        return static_cast<status_t>(nk_jaccard_u16_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u32_t>)
        return static_cast<status_t>(nk_jaccard_u32_best(&a->raw_, &b->raw_, d, &r->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_jaccard has no kernel for this dtype");
}

/** Hamming distance between two bit runs of equal dimensions; @c unexpected_dimensions_k
 *  when they differ or either run is strided or ends mid-value. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_>
expected<typename in_type_::hamming_result_t> hamming(a_type_ const &a, b_type_ const &b,
                                                      nk_capability_t capabilities = default_capabilities(),
                                                      nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::hamming_result_t result {};
    status_t status = hamming<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result, capabilities,
                                        stream);
    return {result, status};
}

/** Jaccard distance between two bit runs, failing like the concept @c hamming. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> a_type_, a_vector_of<in_type_> b_type_>
expected<typename in_type_::jaccard_result_t> jaccard(a_type_ const &a, b_type_ const &b,
                                                      nk_capability_t capabilities = default_capabilities(),
                                                      nk_stream_t stream = nullptr) noexcept {
    auto a_values = contiguous_values_<in_type_ const>(a);
    auto b_values = contiguous_values_<in_type_ const>(b);
    std::size_t const dimensions = a_values.value.size() * dimensions_per_value<in_type_>();
    if (!a_values || !b_values || dimensions != b_values.value.size() * dimensions_per_value<in_type_>())
        return {{}, status_t::unexpected_dimensions_k};
    typename in_type_::jaccard_result_t result {};
    status_t status = jaccard<in_type_>(a_values.value.data(), b_values.value.data(), dimensions, &result, capabilities,
                                        stream);
    return {result, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_SET_HPP
