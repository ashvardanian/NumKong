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
    if constexpr (dimensions_per_value<from_type_>() > 1 || dimensions_per_value<to_type_>() > 1)
        cast<from_type_, to_type_, allow_simd_>(from.data(), n, to.data());
    else
        nk_cast_strided_(from.byte_data(), from_type_::dtype(), from.stride_bytes(), n, to.byte_data(),
                         to_type_::dtype(), to.stride_bytes(), allow_simd_ == prefer_simd_k ? nk_cast : nk_cast_serial);
}

} // namespace ashvardanian::numkong

#endif // NK_CAST_HPP
