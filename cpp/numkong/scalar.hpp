/**
 *  @file cpp/numkong/scalar.hpp
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief C++ verbs over the scalar math of @c c/numkong/scalar.h: square roots, fused
 *      multiply-add, saturating integer arithmetic and the total order of the narrow floats.
 *
 *  Each verb runs the multiversioned kernel the mask picks. The plain overloads take the default
 *  mask and return the value; the overloads taking a mask and a stream return an @c expected, which
 *  holds @c missing_kernel_k for a stream or for a mask without a CPU capability, as scalar kernels
 *  run on the CPU only.
 *
 *  @code{.cpp}
 *  nk::f32_t root = nk::sqrt(nk::f32_t {2.0f});
 *  int sign = nk::order(nk::f16_t::one(), nk::f16_t::zero());             // 1
 *  nk::i8_t clamped = nk::saturating_add(nk::i8_t {100}, nk::i8_t {100}); // 127
 *  @endcode
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_SCALAR_HPP
#define NUMKONG_SCALAR_HPP

#include <type_traits> // `std::is_same_v`

#include "numkong/scalar.h"

#include "numkong/devices.hpp" // `default_capabilities`
#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/** The status of a scalar verb under an explicit mask and stream: scalar kernels run on the CPU. */
inline status_t scalar_status_(nk_capability_t capabilities, nk_stream_t stream) noexcept {
    bool const runs = !stream && (capabilities & nk_cap_cpus_k);
    return runs ? status_t::success_k : status_t::missing_kernel_k;
}

template <a_numeric_dtype scalar_type_>
scalar_type_ sqrt_(scalar_type_ value, nk_capability_t capabilities) noexcept {
    if constexpr (std::is_same_v<scalar_type_, f32_t>) return f32_t {nk_f32_sqrt_best(value.raw_, capabilities)};
    else if constexpr (std::is_same_v<scalar_type_, f64_t>) return f64_t {nk_f64_sqrt_best(value.raw_, capabilities)};
    else if constexpr (std::is_same_v<scalar_type_, f16_t>)
        return f16_t::from_raw(nk_f16_sqrt_best(value.raw_, capabilities));
    else static_assert(sizeof(scalar_type_) == 0, "nk_*_sqrt has no kernel for this dtype");
}

/** The square root of @p value, from @c nk_f32_sqrt_best and its @c f64 and @c f16 twins. */
template <a_numeric_dtype scalar_type_>
scalar_type_ sqrt(scalar_type_ value) noexcept {
    return sqrt_(value, default_capabilities());
}

/** @copydoc sqrt */
template <a_numeric_dtype scalar_type_>
expected<scalar_type_> sqrt(scalar_type_ value, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    status_t const status = scalar_status_(capabilities, stream);
    return {succeeded(status) ? sqrt_(value, capabilities) : scalar_type_ {}, status};
}

template <a_numeric_dtype scalar_type_>
scalar_type_ rsqrt_(scalar_type_ value, nk_capability_t capabilities) noexcept {
    if constexpr (std::is_same_v<scalar_type_, f32_t>) return f32_t {nk_f32_rsqrt_best(value.raw_, capabilities)};
    else if constexpr (std::is_same_v<scalar_type_, f64_t>) return f64_t {nk_f64_rsqrt_best(value.raw_, capabilities)};
    else if constexpr (std::is_same_v<scalar_type_, f16_t>)
        return f16_t::from_raw(nk_f16_rsqrt_best(value.raw_, capabilities));
    else static_assert(sizeof(scalar_type_) == 0, "nk_*_rsqrt has no kernel for this dtype");
}

/** The reciprocal square root of @p value, from @c nk_f32_rsqrt_best and its twins. */
template <a_numeric_dtype scalar_type_>
scalar_type_ rsqrt(scalar_type_ value) noexcept {
    return rsqrt_(value, default_capabilities());
}

/** @copydoc rsqrt */
template <a_numeric_dtype scalar_type_>
expected<scalar_type_> rsqrt(scalar_type_ value, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    status_t const status = scalar_status_(capabilities, stream);
    return {succeeded(status) ? rsqrt_(value, capabilities) : scalar_type_ {}, status};
}

template <a_numeric_dtype scalar_type_>
scalar_type_ fma_(scalar_type_ a, scalar_type_ b, scalar_type_ c, nk_capability_t capabilities) noexcept {
    if constexpr (std::is_same_v<scalar_type_, f32_t>)
        return f32_t {nk_f32_fma_best(a.raw_, b.raw_, c.raw_, capabilities)};
    else if constexpr (std::is_same_v<scalar_type_, f64_t>)
        return f64_t {nk_f64_fma_best(a.raw_, b.raw_, c.raw_, capabilities)};
    else if constexpr (std::is_same_v<scalar_type_, f16_t>)
        return f16_t::from_raw(nk_f16_fma_best(a.raw_, b.raw_, c.raw_, capabilities));
    else static_assert(sizeof(scalar_type_) == 0, "nk_*_fma has no kernel for this dtype");
}

/** @p a × @p b + @p c rounded once, from @c nk_f32_fma_best and its twins. */
template <a_numeric_dtype scalar_type_>
scalar_type_ fma(scalar_type_ a, scalar_type_ b, scalar_type_ c) noexcept {
    return fma_(a, b, c, default_capabilities());
}

/** @copydoc fma */
template <a_numeric_dtype scalar_type_>
expected<scalar_type_> fma(scalar_type_ a, scalar_type_ b, scalar_type_ c, nk_capability_t capabilities,
                           nk_stream_t stream) noexcept {
    status_t const status = scalar_status_(capabilities, stream);
    return {succeeded(status) ? fma_(a, b, c, capabilities) : scalar_type_ {}, status};
}

template <a_numeric_dtype scalar_type_>
scalar_type_ saturating_add_(scalar_type_ a, scalar_type_ b, nk_capability_t capabilities) noexcept {
    using raw_t = typename scalar_type_::raw_t;
    raw_t raw;
    if constexpr (std::is_same_v<scalar_type_, u8_t>) raw = nk_u8_saturating_add_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i8_t>)
        raw = nk_i8_saturating_add_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, u16_t>)
        raw = nk_u16_saturating_add_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i16_t>)
        raw = nk_i16_saturating_add_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, u32_t>)
        raw = nk_u32_saturating_add_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i32_t>)
        raw = nk_i32_saturating_add_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, u64_t>)
        raw = nk_u64_saturating_add_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i64_t>)
        raw = nk_i64_saturating_add_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i4x2_t>)
        raw = nk_i4x2_saturating_add_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, u4x2_t>)
        raw = nk_u4x2_saturating_add_best(a.raw_, b.raw_, capabilities);
    else static_assert(sizeof(scalar_type_) == 0, "nk_*_saturating_add has no kernel for this dtype");
    return scalar_type_::from_raw(raw);
}

/** @p a + @p b clamped to an integer type's range, from @c nk_i8_saturating_add_best and twins. */
template <a_numeric_dtype scalar_type_>
scalar_type_ saturating_add(scalar_type_ a, scalar_type_ b) noexcept {
    return saturating_add_(a, b, default_capabilities());
}

/** @copydoc saturating_add */
template <a_numeric_dtype scalar_type_>
expected<scalar_type_> saturating_add(scalar_type_ a, scalar_type_ b, nk_capability_t capabilities,
                                      nk_stream_t stream) noexcept {
    status_t const status = scalar_status_(capabilities, stream);
    return {succeeded(status) ? saturating_add_(a, b, capabilities) : scalar_type_ {}, status};
}

template <a_numeric_dtype scalar_type_>
scalar_type_ saturating_mul_(scalar_type_ a, scalar_type_ b, nk_capability_t capabilities) noexcept {
    using raw_t = typename scalar_type_::raw_t;
    raw_t raw;
    if constexpr (std::is_same_v<scalar_type_, u8_t>) raw = nk_u8_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i8_t>)
        raw = nk_i8_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, u16_t>)
        raw = nk_u16_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i16_t>)
        raw = nk_i16_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, u32_t>)
        raw = nk_u32_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i32_t>)
        raw = nk_i32_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, u64_t>)
        raw = nk_u64_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i64_t>)
        raw = nk_i64_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, i4x2_t>)
        raw = nk_i4x2_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, u4x2_t>)
        raw = nk_u4x2_saturating_mul_best(a.raw_, b.raw_, capabilities);
    else static_assert(sizeof(scalar_type_) == 0, "nk_*_saturating_mul has no kernel for this dtype");
    return scalar_type_::from_raw(raw);
}

/** @p a × @p b clamped to an integer type's range, from @c nk_i8_saturating_mul_best and twins. */
template <a_numeric_dtype scalar_type_>
scalar_type_ saturating_mul(scalar_type_ a, scalar_type_ b) noexcept {
    return saturating_mul_(a, b, default_capabilities());
}

/** @copydoc saturating_mul */
template <a_numeric_dtype scalar_type_>
expected<scalar_type_> saturating_mul(scalar_type_ a, scalar_type_ b, nk_capability_t capabilities,
                                      nk_stream_t stream) noexcept {
    status_t const status = scalar_status_(capabilities, stream);
    return {succeeded(status) ? saturating_mul_(a, b, capabilities) : scalar_type_ {}, status};
}

template <a_numeric_dtype scalar_type_>
int order_(scalar_type_ a, scalar_type_ b, nk_capability_t capabilities) noexcept {
    if constexpr (std::is_same_v<scalar_type_, f16_t>) return nk_f16_order_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, bf16_t>) return nk_bf16_order_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e4m3_t>) return nk_e4m3_order_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e5m2_t>) return nk_e5m2_order_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e2m3_t>) return nk_e2m3_order_best(a.raw_, b.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e3m2_t>) return nk_e3m2_order_best(a.raw_, b.raw_, capabilities);
    else static_assert(sizeof(scalar_type_) == 0, "nk_*_order has no kernel for this dtype");
}

/**
 *  @brief The total order of two narrow floats, -NaN < -Inf < ... < +Inf < +NaN, from
 *      @c nk_f16_order_best and its twins.
 *  @return Negative, zero or positive, as @p a sorts before, with or after @p b.
 */
template <a_numeric_dtype scalar_type_>
int order(scalar_type_ a, scalar_type_ b) noexcept {
    return order_(a, b, default_capabilities());
}

/** @copydoc order */
template <a_numeric_dtype scalar_type_>
expected<int> order(scalar_type_ a, scalar_type_ b, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    status_t const status = scalar_status_(capabilities, stream);
    return {succeeded(status) ? order_(a, b, capabilities) : 0, status};
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_SCALAR_HPP
