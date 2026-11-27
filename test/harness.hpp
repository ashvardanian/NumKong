/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief C++ test suite with precision analysis using double-double arithmetic.
 *
 *  This test suite compares NumKong operations against high-precision references, like our
 *  @c f118_t double-double type, and reports ULP, absolute, and relative error statistics. Every
 *  kernel is a case labelled @c family.dtype.capability , like @c dot.f32.haswell , skipped on a
 *  machine without that capability.
 *
 *  Environment Variables, each overridden by its flag, like <tt>--time-limit 200ms</tt>:
 *
 *  @verbatim
 *  Variable                     Flag                   Default    Meaning
 *  NUMKONG_SEED                 --seed                 42         Unsigned 64-bit seed for random inputs, or random
 *  NUMKONG_FILTER               --filter               none       Regex searched in case labels
 *  NUMKONG_SCALE                --scale                1          Multiplies each case's time limit, like 0.1 or 10
 *  NUMKONG_TRAP_ON_FAILURE      --trap-on-failure      0          Anything but 0 traps at the first failed check
 *  NUMKONG_NO_SKIPS             --no-skips             0          Anything but 0 exits 3 when a case skipped
 *  NUMKONG_DEVICES              --devices              auto       Comma-separated backend:ordinal, like cuda:0,rocm:1
 *  NUMKONG_TIME_LIMIT           --time-limit           1s         Randomized trials per case, like 200ms, 1s or 2m
 *  NUMKONG_DIMS                 --dims                 1536       Dot and spatial dimension, the first of a list
 *  NUMKONG_CURVED_DIMS          --curved-dims          64         Curved dimension, the first of a list
 *  NUMKONG_SPARSE_DIMS          --sparse-dims          256        Sparse dimension, the first of a list
 *  NUMKONG_SHAPE_POINTS         --shape-points         32         Points per cloud for shape tests
 *  NUMKONG_DIMS_HEIGHT          --dims-height          1024       GEMM M dimension, the first of a list
 *  NUMKONG_DIMS_WIDTH           --dims-width           128        GEMM N dimension, the first of a list
 *  NUMKONG_DIMS_DEPTH           --dims-depth           1536       GEMM K dimension, the first of a list
 *  NUMKONG_IN_QEMU              --in-qemu              0          Anything but 0 shrinks shapes for emulated runs
 *  NUMKONG_ASSERT               --assert               1          0 only reports a failed accuracy check
 *  NUMKONG_ULP_THRESHOLD_F32    --ulp-threshold-f32    4          Max allowed ULP for f32
 *  NUMKONG_ULP_THRESHOLD_F16    --ulp-threshold-f16    32         Max allowed ULP for f16
 *  NUMKONG_ULP_THRESHOLD_BF16   --ulp-threshold-bf16   256        Max allowed ULP for bf16
 *  NUMKONG_SCALE_THRESHOLD      --scale-threshold      0.02       Max error over the largest reference, in (0, 1]
 *  NUMKONG_MAX_COORD_ANGLE      --max-coord-angle      180        Max geospatial angular separation, in (0, 180]
 *  NUMKONG_RANDOM_DISTRIBUTION  --random-distribution  lognormal  uniform, lognormal or cauchy
 *  @endverbatim
 */

#pragma once
#ifndef NUMKONG_TEST_HARNESS_HPP
#define NUMKONG_TEST_HARNESS_HPP

#include <cctype>  // `std::tolower`
#include <cmath>   // `std::fabs`, `std::isnan`, `std::ldexp`, `std::ilogb`
#include <cstddef> // `std::ptrdiff_t`
#include <cstdint> // `std::uint64_t`, `std::int32_t`, `std::int64_t`
#include <cstdio>  // `std::FILE`, `std::fflush`, `stdout`, `stderr`
#include <cstdlib> // `std::abort`, `std::exit`, `std::getenv`, `std::strtod`
#include <cstring> // `std::memcpy`, `std::strcmp`

#include <algorithm>       // `std::equal`
#include <array>           // `std::array`
#include <atomic>          // `std::atomic`
#include <bit>             // `std::countr_zero`
#include <charconv>        // `std::from_chars`
#include <chrono>          // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <complex>         // `std::complex`
#include <functional>      // `std::invoke`
#include <limits>          // `std::numeric_limits`
#include <new>             // `std::bad_alloc`
#include <optional>        // `std::optional`
#include <random>          // `std::random_device`
#include <regex>           // `std::regex`, `std::regex_search`
#include <source_location> // `std::source_location`
#include <string>          // `std::string`
#include <string_view>     // `std::string_view`
#include <system_error>    // `std::errc`
#include <thread>          // `std::this_thread::yield`
#include <tuple>           // `std::tuple`, `std::get`
#include <type_traits>     // `std::is_same_v`, `std::type_identity_t`
#include <utility>         // `std::index_sequence`, `std::move`
#include <vector>          // `std::vector`

#include <fmt/chrono.h> // `std::chrono::milliseconds` through `{}`
#include <fmt/format.h> // `fmt::print`, `fmt::println`, enums with `format_as`

#ifndef NUMKONG_ALLOW_ISA_REDIRECT
#define NUMKONG_ALLOW_ISA_REDIRECT 0
#endif

/** Optional BLAS/MKL integration for precision comparison */
#ifndef NUMKONG_COMPARE_TO_BLAS
#define NUMKONG_COMPARE_TO_BLAS 0
#endif // NUMKONG_COMPARE_TO_BLAS
#ifndef NUMKONG_COMPARE_TO_MKL
#define NUMKONG_COMPARE_TO_MKL 0
#endif // NUMKONG_COMPARE_TO_MKL
#ifndef NUMKONG_COMPARE_TO_ACCELERATE
#define NUMKONG_COMPARE_TO_ACCELERATE 0
#endif // NUMKONG_COMPARE_TO_ACCELERATE

/* Include reference library headers - MKL, Accelerate, or generic CBLAS */
#if NUMKONG_COMPARE_TO_MKL
#include <mkl.h> // MKL includes its own CBLAS interface
#elif NUMKONG_COMPARE_TO_ACCELERATE
#include <Accelerate/Accelerate.h> // Apple Accelerate framework
#elif NUMKONG_COMPARE_TO_BLAS
#include <cblas.h> // Generic CBLAS (OpenBLAS, etc.)
#endif             // NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE || NUMKONG_COMPARE_TO_BLAS

/* In tests we want to make sure our custom floating-point routines are used instead of
 * compiler-provided native types. */
#undef NUMKONG_NATIVE_F16
#define NUMKONG_NATIVE_F16 0
#undef NUMKONG_NATIVE_BF16
#define NUMKONG_NATIVE_BF16 0

#include "numkong/numkong.h" // the C ABI and its dispatch points
#include "numkong/types.hpp"
#include "numkong/tensor.hpp"
#include "numkong/matrix.hpp"
#include "numkong/vector.hpp" // `nk::vector`
#include "numkong/memory.hpp" // `nk::allocator`

#if NUMKONG_OS_WINDOWS_
#include <csignal> // `std::signal`, `std::raise`
#include <io.h>    // `_write`
#elif !NUMKONG_ARCH_WASM_
#include <csignal>  // `std::signal`, `std::raise`, `SIGTRAP`
#include <unistd.h> // `write`, `STDERR_FILENO`
#endif
#if (defined(__GLIBC__) || NUMKONG_OS_APPLE_) && __has_include(<execinfo.h>)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif

namespace nk = ashvardanian::numkong;

namespace ashvardanian::numkong::test {

using nk::bf16_t;
using nk::bf16c_t;
using nk::e2m1x2_t;
using nk::e2m3_t;
using nk::e3m2_t;
using nk::e4m3_t;
using nk::e5m2_t;
using nk::f118_t;
using nk::f118c_t;
using nk::f16_t;
using nk::f16c_t;
using nk::f32_t;
using nk::f32c_t;
using nk::f64_t;
using nk::f64c_t;
using nk::i16_t;
using nk::i32_t;
using nk::i4x2_t;
using nk::i64_t;
using nk::i8_t;
using nk::mxfp4_t;
using nk::mxfp8e4m3_t;
using nk::mxfp8e5m2_t;
using nk::nvfp4_t;
using nk::u16_t;
using nk::u1x8_t;
using nk::u32_t;
using nk::u4x2_t;
using nk::u64_t;
using nk::u8_t;
using nk::ue4m3_t;
using nk::ue8m0_t;

using nk::mxfp4_t;
using nk::mxfp6e2m3_t;
using nk::mxfp6e3m2_t;
using nk::mxfp8e4m3_t;
using nk::mxfp8e5m2_t;
using nk::mxint8_t;
using nk::nvfp4_t;

#pragma region Scalar References

/** Whether @p scalar_type_ is a narrow float whose type holds storage only, so its value is read
 *  through the dispatched scalar conversion exports, like @c nk_f16_to_f32_best. */
template <typename scalar_type_>
constexpr bool is_narrow_float() noexcept {
    return std::is_same_v<scalar_type_, f16_t> || std::is_same_v<scalar_type_, bf16_t> ||
           std::is_same_v<scalar_type_, e4m3_t> || std::is_same_v<scalar_type_, e5m2_t> ||
           std::is_same_v<scalar_type_, e2m3_t> || std::is_same_v<scalar_type_, e3m2_t> ||
           std::is_same_v<scalar_type_, ue8m0_t> || std::is_same_v<scalar_type_, ue4m3_t>;
}

/** The CPU's enabled mask, the default the dispatch points pick from in every suite. */
inline nk_capability_t enabled_capabilities() noexcept {
    static nk_capability_t const capabilities = [] {
        nk_capability_t enabled = 0;
        nk_unused_(nk_capabilities_enabled_cpu(&enabled));
        return enabled;
    }();
    return capabilities;
}

/** The value of one narrow float as a @c float, through its dispatched conversion export. */
template <typename scalar_type_>
float narrow_to_f32(scalar_type_ value) noexcept {
    nk_capability_t const capabilities = enabled_capabilities();
    float result;
    if constexpr (std::is_same_v<scalar_type_, f16_t>) nk_f16_to_f32_best(&value.raw_, &result, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, bf16_t>) nk_bf16_to_f32_best(&value.raw_, &result, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e4m3_t>) nk_e4m3_to_f32_best(&value.raw_, &result, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e5m2_t>) nk_e5m2_to_f32_best(&value.raw_, &result, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e2m3_t>) nk_e2m3_to_f32_best(&value.raw_, &result, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e3m2_t>) nk_e3m2_to_f32_best(&value.raw_, &result, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, ue8m0_t>) nk_ue8m0_to_f32_best(&value.raw_, &result, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, ue4m3_t>) nk_ue4m3_to_f32_best(&value.raw_, &result, capabilities);
    else static_assert(sizeof(scalar_type_) == 0, "not a narrow float");
    return result;
}

/** One @c float rounded into a narrow float, through its dispatched conversion export. */
template <typename scalar_type_>
scalar_type_ f32_to_narrow(float value) noexcept {
    nk_capability_t const capabilities = enabled_capabilities();
    scalar_type_ result;
    if constexpr (std::is_same_v<scalar_type_, f16_t>) nk_f32_to_f16_best(&value, &result.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, bf16_t>) nk_f32_to_bf16_best(&value, &result.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e4m3_t>) nk_f32_to_e4m3_best(&value, &result.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e5m2_t>) nk_f32_to_e5m2_best(&value, &result.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e2m3_t>) nk_f32_to_e2m3_best(&value, &result.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, e3m2_t>) nk_f32_to_e3m2_best(&value, &result.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, ue8m0_t>) nk_f32_to_ue8m0_best(&value, &result.raw_, capabilities);
    else if constexpr (std::is_same_v<scalar_type_, ue4m3_t>) nk_f32_to_ue4m3_best(&value, &result.raw_, capabilities);
    else static_assert(sizeof(scalar_type_) == 0, "not a narrow float");
    return result;
}

template <typename value_type_>
struct tracked;

/** Whether @p type_ is a @c tracked reference, which widens through the value it tracks. */
template <typename type_>
constexpr bool is_tracked = false;
template <typename value_type_>
constexpr bool is_tracked<tracked<value_type_>> = true;

/** @p value in @p target_type_: a narrow float through @c float, a half-precision complex part by
 *  part, an integer wrapper through its raw value, and anything else as cast. */
template <typename target_type_, typename scalar_type_>
target_type_ widen(scalar_type_ value) noexcept {
    if constexpr (std::is_same_v<target_type_, scalar_type_>) return value;
    else if constexpr (is_tracked<target_type_>)
        return target_type_(widen<typename target_type_::reference_t>(value));
    else if constexpr (is_narrow_float<scalar_type_>()) return target_type_(narrow_to_f32(value));
    else if constexpr (std::is_same_v<scalar_type_, f16c_t> || std::is_same_v<scalar_type_, bf16c_t>) {
        using component_t = typename target_type_::component_t;
        return target_type_(widen<component_t>(value.real()), widen<component_t>(value.imag()));
    }
    else if constexpr (!std::is_constructible_v<target_type_, scalar_type_>) return target_type_(value.raw_);
    else return static_cast<target_type_>(value);
}

/** @p value rounded into @p scalar_type_, the way a kernel's result is: a narrow float through
 *  @c float and its conversion export, anything else as cast. */
template <typename scalar_type_, typename value_type_>
scalar_type_ narrow(value_type_ value) noexcept;

/** Whether @p a sorts before @p b by value: narrow floats as the floats they widen to, the rest by
 *  their own comparison, which keeps 64-bit integers exact. */
template <typename scalar_type_>
bool sorts_before_reference(scalar_type_ a, scalar_type_ b) noexcept {
    if constexpr (is_narrow_float<scalar_type_>()) return narrow_to_f32(a) < narrow_to_f32(b);
    else return a < b;
}

/** The value of any real scalar as a @c double, for printing and tolerances. */
template <typename scalar_type_>
double to_double(scalar_type_ scalar) noexcept {
    if constexpr (std::is_arithmetic_v<scalar_type_>) return static_cast<double>(scalar);
    else if constexpr (is_narrow_float<scalar_type_>()) return narrow_to_f32(scalar);
    else if constexpr (is_tracked<scalar_type_>) return to_double(scalar.value);
    else if constexpr (!std::is_constructible_v<double, scalar_type_>) return static_cast<double>(scalar.raw_);
    else return static_cast<double>(scalar);
}

/** @p value rounded into @p scalar_type_, the way a kernel's result is: a narrow float through
 *  @c float and its conversion export, an integer to nearest even and saturated, through @c f118_t,
 *  and anything else as cast. */
template <typename scalar_type_, typename value_type_>
scalar_type_ narrow(value_type_ value) noexcept {
    if constexpr (std::is_same_v<scalar_type_, value_type_>) return value;
    else if constexpr (is_narrow_float<scalar_type_>())
        return f32_to_narrow<scalar_type_>(static_cast<float>(to_double(value)));
    else if constexpr (std::is_same_v<scalar_type_, f16c_t> || std::is_same_v<scalar_type_, bf16c_t>) {
        using component_t = typename scalar_type_::component_t;
        return scalar_type_(narrow<component_t>(value.real()), narrow<component_t>(value.imag()));
    }
    else if constexpr (is_narrow_float<value_type_>()) return narrow<scalar_type_>(narrow_to_f32(value));
    else if constexpr (std::is_same_v<value_type_, f118_t> || std::is_same_v<value_type_, f118c_t>)
        return value.template to<scalar_type_>();
    else if constexpr (nk::is_integral_dtype<scalar_type_>() && !nk::is_integral_dtype<value_type_>())
        return f118_t(to_double(value)).template to<scalar_type_>();
    else return static_cast<scalar_type_>(value);
}

/** FMA helper template for baseline dot-product implementations. */
template <typename in_type_, typename accumulator_type_>
    requires(dimensions_per_value<in_type_>() == 1)
inline accumulator_type_ fma_reference(in_type_ a, in_type_ b, accumulator_type_ accumulator) noexcept {
    return accumulator + widen<accumulator_type_>(a) * widen<accumulator_type_>(b);
}

/** FMA helper template for baseline conjugate complex dot-product implementations. */
template <typename in_type_, typename accumulator_type_>
    requires(dimensions_per_value<in_type_>() == 1)
inline accumulator_type_ fcma_reference(in_type_ a, in_type_ b, accumulator_type_ accumulator) noexcept {
    return accumulator + widen<accumulator_type_>(a.conj()) * widen<accumulator_type_>(b);
}

/** Fused addition of squared differences for baseline L2 implementations. */
template <typename in_type_, typename accumulator_type_>
    requires(dimensions_per_value<in_type_>() == 1)
inline accumulator_type_ fdsa_reference(in_type_ a, in_type_ b, accumulator_type_ accumulator) noexcept {
    accumulator_type_ const d = widen<accumulator_type_>(a) - widen<accumulator_type_>(b);
    return accumulator + d * d;
}

/** @p a + @p b clamped to an integer wrapper's range, in its own type without a wider one. */
template <typename integer_type_>
integer_type_ saturating_add_integers(integer_type_ a, integer_type_ b) noexcept {
    using raw_t = typename integer_type_::raw_t;
    using limits_t = std::numeric_limits<raw_t>;
    raw_t const x = a.raw_, y = b.raw_;
    if (y > 0 && x > (limits_t::max)() - y) return integer_type_::finite_max();
    if constexpr (limits_t::is_signed)
        if (y < 0 && x < (limits_t::min)() - y) return integer_type_::finite_min();
    return integer_type_::from_raw(static_cast<raw_t>(x + y));
}

/** @p a × @p b clamped to an integer wrapper's range, through 64-bit magnitudes. */
template <typename integer_type_>
integer_type_ saturating_mul_integers(integer_type_ a, integer_type_ b) noexcept {
    using raw_t = typename integer_type_::raw_t;
    using limits_t = std::numeric_limits<raw_t>;
    raw_t const x = a.raw_, y = b.raw_;
    auto const magnitude = [](raw_t value) noexcept {
        return value < 0 ? 0 - static_cast<std::uint64_t>(value) : static_cast<std::uint64_t>(value);
    };
    bool const negative = limits_t::is_signed && ((x < 0) != (y < 0));
    std::uint64_t const x_magnitude = magnitude(x), y_magnitude = magnitude(y);
    std::uint64_t const bound = static_cast<std::uint64_t>((limits_t::max)()) + (negative ? 1 : 0);
    if (x_magnitude != 0 && y_magnitude > bound / x_magnitude)
        return negative ? integer_type_::finite_min() : integer_type_::finite_max();
    std::uint64_t const product = x_magnitude * y_magnitude;
    return integer_type_::from_raw(static_cast<raw_t>(negative ? 0 - product : product));
}

/** Free-standing saturating addition for baseline implementations. */
template <typename accumulator_type_, typename in_type_>
    requires(std::is_same<accumulator_type_, in_type_>::value || dimensions_per_value<in_type_>() == 1)
accumulator_type_ saturating_add_reference(accumulator_type_ a, in_type_ b) noexcept {
    if constexpr (nk::is_integral_dtype<accumulator_type_>())
        return saturating_add_integers(a, widen<accumulator_type_>(b));
    else if constexpr (is_narrow_float<accumulator_type_>()) {
        double const sum = to_double(a) + to_double(b);
        double const limit = to_double(accumulator_type_::finite_max());
        return narrow<accumulator_type_>(sum > limit ? limit : sum < -limit ? -limit : sum);
    }
    else return a.saturating_add(widen<accumulator_type_>(b));
}

/** Free-standing saturating multiplication for baseline implementations. */
template <typename accumulator_type_, typename in_type_>
    requires(std::is_same<accumulator_type_, in_type_>::value || dimensions_per_value<in_type_>() == 1)
accumulator_type_ saturating_mul_reference(accumulator_type_ a, in_type_ b) noexcept {
    if constexpr (nk::is_integral_dtype<accumulator_type_>())
        return saturating_mul_integers(a, widen<accumulator_type_>(b));
    else return a.saturating_mul(widen<accumulator_type_>(b));
}

/** Operands of the free-standing FMA: scalars into any accumulator, packed values into their own
 *  type. */
template <typename in_type_, typename accumulator_type_>
concept an_fma_operand_pair = dimensions_per_value<in_type_>() == 1 || std::is_same_v<accumulator_type_, in_type_>;

/** Signed scalars into an unsigned integer accumulator, which a negative operand cannot be cast
 *  into. */
template <typename in_type_, typename accumulator_type_>
concept a_signed_into_unsigned_fma_operand_pair =
    an_fma_operand_pair<in_type_, accumulator_type_> && dimensions_per_value<in_type_>() == 1 &&
    is_signed_dtype<in_type_>() && is_integral_dtype<accumulator_type_>() && !is_signed_dtype<accumulator_type_>();

/** Free-standing saturating FMA, a*b + c, for baseline implementations. */
template <typename in_type_, typename accumulator_type_>
    requires an_fma_operand_pair<in_type_, accumulator_type_>
accumulator_type_ saturating_fma_reference(in_type_ a, in_type_ b, accumulator_type_ accumulator) noexcept {
    return saturating_add_reference(accumulator,
                                    saturating_mul_reference(widen<accumulator_type_>(a), widen<accumulator_type_>(b)));
}

/** Saturating FMA of signed operands into an unsigned integer accumulator, multiplied in @c i64_t
 *  first. */
template <typename in_type_, typename accumulator_type_>
    requires a_signed_into_unsigned_fma_operand_pair<in_type_, accumulator_type_>
accumulator_type_ saturating_fma_reference(in_type_ a, in_type_ b, accumulator_type_ accumulator) noexcept {
    std::int64_t const product = saturating_mul_integers(widen<i64_t>(a), widen<i64_t>(b));
    if (product >= 0)
        return saturating_add_integers(accumulator, accumulator_type_(static_cast<std::uint64_t>(product)));
    return accumulator.saturating_sub(accumulator_type_(0 - static_cast<std::uint64_t>(product)));
}

/** FMA specialization for i4x2_t, signed 4-bit packed pairs. */
template <typename accumulator_type_>
constexpr accumulator_type_ fma_reference(i4x2_t a, i4x2_t b, accumulator_type_ accumulator) noexcept {
    return accumulator +
           accumulator_type_(nk_i32_t(a.first()) * nk_i32_t(b.first()) + nk_i32_t(a.second()) * nk_i32_t(b.second()));
}

/** FMA specialization for u4x2_t, unsigned 4-bit packed pairs. */
template <typename accumulator_type_>
constexpr accumulator_type_ fma_reference(u4x2_t a, u4x2_t b, accumulator_type_ accumulator) noexcept {
    return accumulator +
           accumulator_type_(nk_u32_t(a.first()) * nk_u32_t(b.first()) + nk_u32_t(a.second()) * nk_u32_t(b.second()));
}

/** FMA specialization for e2m1x2_t, FP4 packed pairs; both nibble products are exact in f32. */
template <typename accumulator_type_>
constexpr accumulator_type_ fma_reference(e2m1x2_t a, e2m1x2_t b, accumulator_type_ accumulator) noexcept {
    return accumulator + accumulator_type_(a.first() * b.first() + a.second() * b.second());
}

/** FMA specialization for u1x8_t, 8 packed bits. Counts matching set bits, popcount of AND. */
template <typename accumulator_type_>
constexpr accumulator_type_ fma_reference(u1x8_t a, u1x8_t b, accumulator_type_ accumulator) noexcept {
    return accumulator + accumulator_type_(std::popcount(static_cast<unsigned>(a.raw() & b.raw())));
}

/** Squared difference specialization for i4x2_t, signed 4-bit packed pairs. */
template <typename accumulator_type_>
constexpr accumulator_type_ fdsa_reference(i4x2_t a, i4x2_t b, accumulator_type_ accumulator) noexcept {
    nk_i32_t first_difference = nk_i32_t(a.first()) - nk_i32_t(b.first());
    nk_i32_t second_difference = nk_i32_t(a.second()) - nk_i32_t(b.second());
    return accumulator + accumulator_type_(first_difference * first_difference + second_difference * second_difference);
}

/** Squared difference specialization for u4x2_t, unsigned 4-bit packed pairs. */
template <typename accumulator_type_>
constexpr accumulator_type_ fdsa_reference(u4x2_t a, u4x2_t b, accumulator_type_ accumulator) noexcept {
    nk_i32_t first_difference = nk_i32_t(a.first()) - nk_i32_t(b.first());
    nk_i32_t second_difference = nk_i32_t(a.second()) - nk_i32_t(b.second());
    return accumulator + accumulator_type_(first_difference * first_difference + second_difference * second_difference);
}

/** Saturating addition specialization for i4x2_t, signed 4-bit packed pairs. */
template <typename accumulator_type_>
accumulator_type_ saturating_add_reference(accumulator_type_ a, i4x2_t b) noexcept {
    return saturating_add_reference(saturating_add_reference(a, b.first()), b.second());
}

/** Saturating addition specialization for u4x2_t, unsigned 4-bit packed pairs. */
template <typename accumulator_type_>
accumulator_type_ saturating_add_reference(accumulator_type_ a, u4x2_t b) noexcept {
    return saturating_add_reference(saturating_add_reference(a, b.first()), b.second());
}

/** Saturating FMA specialization for i4x2_t, signed 4-bit packed pairs. */
template <typename accumulator_type_>
    requires(!is_integral_dtype<accumulator_type_>())
constexpr accumulator_type_ saturating_fma_reference(i4x2_t a, i4x2_t b, accumulator_type_ accumulator) noexcept {
    return fma_reference(a, b, accumulator);
}

/** Saturating FMA specialization for u4x2_t, unsigned 4-bit packed pairs. */
template <typename accumulator_type_>
    requires(!is_integral_dtype<accumulator_type_>())
constexpr accumulator_type_ saturating_fma_reference(u4x2_t a, u4x2_t b, accumulator_type_ accumulator) noexcept {
    return fma_reference(a, b, accumulator);
}

/** Count element-level differences. For scalar types, returns 1 if different, 0 if equal. */
template <typename scalar_type_>
constexpr unsigned count_differences(scalar_type_ a, scalar_type_ b) noexcept {
    return (a != b) ? 1 : 0;
}

/** Count bit-level differences for u1x8_t, 8 packed bits. Returns popcount of XOR. */
constexpr unsigned count_differences(u1x8_t a, u1x8_t b) noexcept { return a.hamming(b); }

/** Count element-level intersection. For scalar types, returns 1 if equal, 0 if different. */
template <typename scalar_type_>
constexpr unsigned count_intersection(scalar_type_ a, scalar_type_ b) noexcept {
    return (a == b) ? 1 : 0;
}

/** Count element-level union. For scalar types, always returns 1, one element per pair. */
template <typename scalar_type_>
constexpr unsigned count_union(scalar_type_ a, scalar_type_ b) noexcept {
    nk_unused_(a), nk_unused_(b);
    return 1;
}

/** Count bit-level intersection for u1x8_t, 8 packed bits. Returns popcount of AND. */
constexpr unsigned count_intersection(u1x8_t a, u1x8_t b) noexcept { return a.intersection(b); }

/** Count bit-level union for u1x8_t, 8 packed bits. Returns popcount of OR. */
constexpr unsigned count_union(u1x8_t a, u1x8_t b) noexcept { return a.union_size(b); }

/** Decodes @p count dimensions of @p values into @p decoded, bits and nibbles included. */
template <typename scalar_type_>
void decode_values(scalar_type_ const *values, std::size_t count, double *decoded) noexcept {
    constexpr std::size_t per_value = nk::dimensions_per_value<scalar_type_>();
    for (std::size_t i = 0; i != count; ++i) {
        if constexpr (per_value == 1) decoded[i] = to_double(values[i]);
        else {
            scalar_type_ value = values[i / per_value];
            if constexpr (std::is_same_v<scalar_type_, e2m1x2_t>)
                decoded[i] = e2m1x2_t::nibble_to_f32(value[static_cast<unsigned>(i % per_value)].get());
            else decoded[i] = static_cast<double>(value[static_cast<unsigned>(i % per_value)].get());
        }
    }
}

/** Encodes @p count dimensions of @p decoded into @p values, rounding as @c narrow does; a bit or
 *  an integer nibble takes its nearest integer, and an E2M1 nibble its nearest code. */
template <typename scalar_type_>
void encode_values(double const *decoded, std::size_t count, scalar_type_ *values) noexcept {
    constexpr std::size_t per_value = nk::dimensions_per_value<scalar_type_>();
    for (std::size_t i = 0; i != count; ++i) {
        if constexpr (per_value == 1) values[i] = narrow<scalar_type_>(decoded[i]);
        else if constexpr (std::is_same_v<scalar_type_, e2m1x2_t>) {
            std::uint8_t nearest = 0;
            for (std::uint8_t code = 1; code != 16; ++code)
                if (std::fabs(e2m1x2_t::nibble_to_f32(code) - decoded[i]) <
                    std::fabs(e2m1x2_t::nibble_to_f32(nearest) - decoded[i]))
                    nearest = code;
            values[i / per_value][static_cast<unsigned>(i % per_value)] = nearest;
        }
        else values[i / per_value][static_cast<unsigned>(i % per_value)] = static_cast<int>(std::nearbyint(decoded[i]));
    }
}

#pragma endregion Scalar References

using steady_clock_t = std::chrono::steady_clock;

/** The moment a kernel's randomized trials stop. */
struct deadline_t {
    steady_clock_t::time_point moment;

    /** Whether the clock, read on every call, has not reached @c moment yet. */
    bool ahead() const noexcept { return steady_clock_t::now() < moment; }
};

/** A size in bytes, kept apart from element counts. */
struct bytes_t {
    std::size_t value = 0;
};

/** A positive whole number, like "64". */
inline std::optional<std::size_t> parse_count(std::string_view text) noexcept {
    std::size_t count = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc {} || end != text.data() + text.size() || !count) return std::nullopt;
    return count;
}

/** Positive counts separated by commas, like "64,128". */
inline std::optional<std::vector<std::size_t>> parse_dims(std::string_view text) {
    std::vector<std::size_t> dims;
    for (std::size_t start = 0;;) {
        std::size_t const comma = text.find(',', start);
        std::optional<std::size_t> const dim = parse_count(text.substr(start, comma - start));
        if (!dim) return std::nullopt;
        dims.push_back(*dim);
        if (comma == std::string_view::npos) return dims;
        start = comma + 1;
    }
}

/**
 *  @brief Maps an input type to the type its reference results are computed in.
 *
 *  - f32/f64 input → f118_t (need full double-double precision)
 *  - Complex f32c/f64c input → f118c_t
 *  - Smaller complex (f16c, bf16c) → f64c_t (52-bit mantissa >> 7-10 bit mantissa)
 *  - Everything else (integers, bf16, f16, etc.) → f64_t
 */
template <typename input_type_>
using reference_for = std::conditional_t<
    std::is_same_v<input_type_, f32_t> || std::is_same_v<input_type_, f64_t>, f118_t,
    std::conditional_t<
        nk::is_complex_dtype<input_type_>(),
        std::conditional_t<std::is_same_v<input_type_, f32c_t> || std::is_same_v<input_type_, f64c_t>, f118c_t, f64c_t>,
        f64_t>>;

/** The distribution @c fill_random draws test inputs from. */
enum class distribution_t { uniform_k, lognormal_k, cauchy_k };

/** The name @c NUMKONG_RANDOM_DISTRIBUTION spells @p distribution with. */
inline std::string_view format_as(distribution_t distribution) noexcept {
    switch (distribution) {
    case distribution_t::uniform_k: return "uniform";
    case distribution_t::lognormal_k: return "lognormal";
    case distribution_t::cauchy_k: return "cauchy";
    }
    return "unknown";
}

/** The distribution named @p text, like "lognormal". */
inline std::optional<distribution_t> parse_distribution(std::string_view text) noexcept {
    for (distribution_t distribution :
         {distribution_t::uniform_k, distribution_t::lognormal_k, distribution_t::cauchy_k})
        if (format_as(distribution) == text) return distribution;
    return std::nullopt;
}

/** What a kernel failing its accuracy check does to the exit status. */
enum class on_failure_t : unsigned char { report_k, exit_k };

enum class comparison_family_t {
    exact_k,
    approximate_k,
    normalized_reduction_k,
    bounded_k,
};
enum class comparison_failure_mode_t { exact_distance_k, ulp_threshold_k, scale_threshold_k, bound_threshold_k };

struct comparison_family_spec_t {
    comparison_failure_mode_t failure_mode;
    std::array<char const *, 5> column_labels;
};

inline constexpr comparison_family_spec_t comparison_family_spec(comparison_family_t family) noexcept {
    switch (family) {
    case comparison_family_t::exact_k:
        return {comparison_failure_mode_t::exact_distance_k, {"max_dist", "mean_dist", "max_abs", "mismatch", "exact"}};
    case comparison_family_t::approximate_k:
        return {comparison_failure_mode_t::ulp_threshold_k, {"max_abs", "max_rel", "mean_ulp", "max_ulp", "exact"}};
    case comparison_family_t::normalized_reduction_k:
        // Normalized weighted reductions (attention outputs) legitimately pass through zero,
        // where ULP distance explodes on quantization-level noise; the meaningful bound is
        // the absolute error relative to the largest reference magnitude.
        return {comparison_failure_mode_t::scale_threshold_k, {"max_abs", "max_rel", "mean_ulp", "max_ulp", "exact"}};
    case comparison_family_t::bounded_k:
        // Each result carries its own error bound, derived from the arithmetic the backend accumulates with.
        return {comparison_failure_mode_t::bound_threshold_k, {"max_abs", "max_rel", "max_bound", "max_ulp", "exact"}};
    }
    return {comparison_failure_mode_t::ulp_threshold_k, {"max_abs", "max_rel", "mean_ulp", "max_ulp", "exact"}};
}

/** The facts this binary and this machine report: the library version, and the capabilities
 *  compiled in and detected. */
struct machine_t {
    std::array<unsigned, 3> version {NUMKONG_VERSION_MAJOR, NUMKONG_VERSION_MINOR, NUMKONG_VERSION_PATCH};
    nk_capability_t compiled = 0;
    nk_capability_t detected = 0;
};

/** The capabilities this CPU runs, whether or not this binary holds them. */
inline nk_capability_t capabilities_detected_cpu() noexcept {
    nk_capability_t capabilities = 0;
    return nk_capabilities_detected_cpu(&capabilities) == nk_success_k ? capabilities : nk_cap_serial_k;
}

/** The CPU capabilities this binary holds, whether or not this CPU runs them. */
inline nk_capability_t capabilities_compiled_cpu() noexcept {
    nk_capability_t capabilities = 0;
    return nk_capabilities_compiled_cpu(&capabilities) == nk_success_k ? capabilities : nk_cap_serial_k;
}

/** Probes the capabilities @c machine_t reports. */
inline machine_t probe_machine() noexcept {
    machine_t machine;
    machine.compiled = capabilities_compiled_cpu();
    machine.detected = capabilities_detected_cpu();
    return machine;
}

/** A capability mask, formatted as the comma-separated names @c nk_capabilities_name spells. */
struct capability_names_t {
    nk_capability_t capabilities = 0;
};

} // namespace ashvardanian::numkong::test

/** Writes a @ref capability_names_t through @c nk_capabilities_name , its one staging buffer. */
template <>
struct fmt::formatter<ashvardanian::numkong::test::capability_names_t> {
    constexpr auto parse(fmt::format_parse_context &context) { return context.begin(); }
    template <typename context_type_>
    auto format(ashvardanian::numkong::test::capability_names_t const &names, context_type_ &context) const {
        char spelled[NUMKONG_CAPABILITIES_NAME_CAPACITY];
        std::size_t const length = nk_capabilities_name(names.capabilities, spelled, sizeof(spelled));
        return std::copy_n(spelled, length, context.out());
    }
};

namespace ashvardanian::numkong::test {

/** Prints the version line, then the capabilities as "- Compiled for:" and "- This machine:". */
inline void print(machine_t const &machine, std::FILE *stream = stdout) {
    fmt::println(stream, "NumKong {}.{}.{}", machine.version[0], machine.version[1], machine.version[2]);
    fmt::println(stream, "- Compiled for: {}", capability_names_t {machine.compiled});
    fmt::println(stream, "- This machine: {}", capability_names_t {machine.detected});
}

/** Stops in an attached debugger, or dumps core without one; SIGTRAP skips the crash handlers. */
inline void trap_into_debugger() noexcept {
#if NUMKONG_OS_WINDOWS_
    __debugbreak();
#elif NUMKONG_ARCH_WASM_
    __builtin_trap();
#else
    std::raise(SIGTRAP);
#endif
}

/** Arms the crash handlers, defined below the run whose case they name. */
inline void install_test_signal_handlers() noexcept;

/** Writes one line to @p stream, formatted from @p format and @p args, then its newline. */
template <typename... args_types_>
void print_line(std::FILE *stream, fmt::format_string<args_types_...> format, args_types_ &&...args) {
    fmt::println(stream, format, std::forward<args_types_>(args)...);
}

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const *variable, char const *text, char const *expected) noexcept {
    print_line(stderr, "{}=\"{}\" does not parse, expected {}", variable, text, expected);
    std::exit(1);
}

/** The text of a setting: its @p flag when given, else its environment @p variable , else null. */
inline char const *setting_text(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    for (int index = 1; index < argc; ++index) {
        std::string_view const argument {argv[index]};
        if (argument == flag)
            return index + 1 < argc && std::strncmp(argv[index + 1], "--", 2) != 0 ? argv[index + 1] : "1";
        if (argument.size() > flag.size() && argument.starts_with(flag) && argument[flag.size()] == '=')
            return argv[index] + flag.size() + 1;
    }
    char const *text = std::getenv(variable);
    return text && *text ? text : nullptr;
}

/** A switch, @p fallback when unset: 0, false, off and no turn it off, anything else on. */
inline bool read_flag_setting(int argc, char **argv, char const *variable, std::string_view flag,
                              bool fallback = false) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    for (char const *off : {"0", "false", "off", "no"})
        if (std::strcmp(text, off) == 0) return false;
    return true;
}

/** A whole number, @p fallback when unset. */
inline std::size_t read_count_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                      std::size_t fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::size_t count = 0;
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "a whole number");
    return count;
}

/** A positive real, @p fallback when unset. */
inline double read_real_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                double fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    char *end = nullptr;
    double const value = std::strtod(text, &end);
    if (*end != '\0' || !(value > 0) || !std::isfinite(value))
        exit_unparsed(variable, text, "a positive number like 0.1 or 10");
    return value;
}

/** A positive duration such as 200ms, 10s or 2m, @p fallback when unset. */
inline std::chrono::milliseconds read_duration_setting(int argc, char **argv, char const *variable,
                                                       std::string_view flag,
                                                       std::chrono::milliseconds fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::uint32_t count = 0;
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (parse_status == std::errc {} && count != 0) {
        if (unit == "ms") return std::chrono::milliseconds(count);
        if (unit == "s") return std::chrono::seconds(count);
        if (unit == "m") return std::chrono::minutes(count);
    }
    exit_unparsed(variable, text, "a duration like 200ms, 10s or 2m");
}

/** The run's seed: 42 when unset, drawn afresh for @c random , else any unsigned 64-bit value. */
inline std::uint64_t read_seed_setting(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return 42;
    std::string_view const spelled {text};
    if (spelled == "random") {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }
    std::uint64_t seed = 0;
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: those its regex matches somewhere, every label while unset. */
struct regex_filter_t {
    std::string_view text;
    std::regex regex {""};

    bool selects(std::string_view label) const noexcept { return std::regex_search(label.begin(), label.end(), regex); }
};

/** The filter a setting spells, compiled once; one that does not compile exits as a bad setting. */
inline regex_filter_t read_regex_filter_setting(int argc, char **argv, char const *variable,
                                                std::string_view flag) noexcept {
    regex_filter_t filter;
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.text = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.assign(filter.text.begin(), filter.text.end());
    }
    catch (std::regex_error const &) {
        exit_unparsed(variable, text, "a regular expression");
    }
#else
    filter.regex.assign(filter.text.begin(), filter.text.end());
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    regex_filter_t regex_filter;
    char const *program = "";
};

/** SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits. */
constexpr std::uint64_t split_mix_bits(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** The @p index -th stream of the family @p name under @p seed , identical in every language. */
constexpr std::uint64_t stream_key(std::uint64_t seed, std::string_view name, std::uint64_t index = 0) noexcept {
    std::uint64_t hashed = 0xCBF29CE484222325ull;
    for (char const character : name) hashed = (hashed ^ static_cast<std::uint8_t>(character)) * 0x100000001B3ull;
    return split_mix_bits(split_mix_bits(seed ^ hashed) + index);
}

/**
 *  @brief SplitMix64 over one 64-bit counter, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t counter = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : counter(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(counter += 0x9E3779B97F4A7C15ull); }

    /** A uniform draw below @p bound , zero for an empty bound, without the bias of a modulo. */
    std::uint64_t below(std::uint64_t bound) noexcept {
        if (bound == 0) return 0;
        std::uint64_t const rejection_floor = (0u - bound) % bound;
        std::uint64_t draw = (*this)();
        while (draw < rejection_floor) draw = (*this)();
        return draw % bound;
    }
};

#pragma endregion Shared Settings

/** Every test setting, with its default as the initializer, read once by @c read_test_settings . */
struct test_settings_t : common_settings_t {
    double scale = 1.0;
    bool trap_on_failure = false;
    bool refuse_skips = false;

    /** How long each case's randomized trials run, before @c NUMKONG_SCALE multiplies it. */
    std::chrono::milliseconds time_limit = std::chrono::seconds(1);

    /** Random distribution for test inputs. */
    distribution_t random_distribution = distribution_t::lognormal_k;

    /** What a failed accuracy check does to its case. */
    on_failure_t on_failure = on_failure_t::exit_k;

    /** Whether to shrink shapes for emulated SIMD. */
    bool in_qemu = false;

    /** Max allowed ULP for f32. */
    std::size_t ulp_threshold_f32 = 4;

    /** Max allowed ULP for f16. */
    std::size_t ulp_threshold_f16 = 32;

    /** Max allowed ULP for bf16. */
    std::size_t ulp_threshold_bf16 = 256;

    /** Max absolute error as a fraction of the largest reference magnitude, for the
     *  normalized-reduction family. */
    double scale_threshold = 0.02;

    /** For dot products, spatial metrics. */
    std::size_t dense_dimensions = 1536;

    /** GEMM M dimension. */
    std::size_t matrix_height = 1024;

    /** GEMM N dimension. */
    std::size_t matrix_width = 128;

    /** GEMM K dimension. */
    std::size_t matrix_depth = 1536;

    /** For curved metrics, quadratic in dimensions. */
    std::size_t curved_dimensions = 64;

    /** For sparse set intersection and sparse dot. */
    std::size_t sparse_dimensions = 256;

    /** Points per cloud for the many-to-many shape distances. */
    std::size_t shape_points = 32;

    /** Max geospatial angular separation. */
    float max_coord_angle_degrees = 180.0f;

    /** The vendor's ordinal of the GPU the device cases run on. */
    std::size_t device_ordinal = 0;

    /** What the cases in progress may run: this CPU's capabilities, then each device's ones. */
    nk_capability_t capabilities = 0;

    /** The deadline of trials starting now. */
    deadline_t deadline() const noexcept {
        return {steady_clock_t::now() + std::chrono::duration_cast<steady_clock_t::duration>(time_limit * scale)};
    }

    std::uint64_t ulp_threshold_for(nk_dtype_t dtype) const noexcept {
        if (dtype == nk_bf16_k) return ulp_threshold_bf16;
        if (dtype == nk_f16_k) return ulp_threshold_f16;
        return ulp_threshold_f32;
    }
};

inline test_settings_t read_test_settings(int argc, char **argv) noexcept;

#pragma region Shared Run

/** What a case reports beyond its checks: that it ran, or why it skipped. */
struct case_result_t {
    char const *skip_reason = nullptr;

    [[nodiscard]] static constexpr case_result_t skipped(char const *reason) noexcept { return {reason}; }
};

/**
 *  @brief The run in progress: the harness's one sanctioned mutable global.
 *
 *  Checks fire on worker threads and inside helpers as well as in a case's own body, so the run
 *  counts failed checks in one place and @ref run_test_case diffs that count around each case,
 *  rather than threading a tally through every call. After a crash, printing @c test_run from a
 *  core dump names the case in progress and the last failed check, and the signal handlers print
 *  the case too.
 */
struct test_run_t {
    test_settings_t settings {};
    char const *current_case = nullptr;
    std::atomic<std::size_t> failed_checks {0};
    std::atomic<char const *> last_failed_file {nullptr};
    std::atomic<std::uint_least32_t> last_failed_line {0};
    std::size_t passed_cases = 0;
    std::size_t failed_cases = 0;
    std::size_t skipped_cases = 0;
};

inline test_run_t test_run;

/** How long any wait in a test may take; a passing case never waits it out, so it bounds hangs. */
inline constexpr std::chrono::seconds wait_limit_k {120};

/** Yields until @p predicate holds; false when @ref wait_limit_k passed first. */
template <typename predicate_type_>
bool spin_until(predicate_type_ &&predicate) noexcept {
    auto const deadline = std::chrono::steady_clock::now() + wait_limit_k;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

/** @p baseline iterations under the @c SCALE setting, never fewer than one. */
inline std::size_t scaled_iterations(std::size_t baseline) noexcept {
    double const scaled = to_double(baseline) * test_run.settings.scale;
    return scaled < 1 ? 1 : static_cast<std::size_t>(std::llround(scaled));
}

/** The step that visits a @p coverage fraction of a sweep, at least one. */
inline std::size_t sweep_stride(double coverage) noexcept {
    double const stride = 1.0 / coverage;
    return stride < 1 ? 1 : static_cast<std::size_t>(std::llround(stride));
}

/** The @p index -th stream of the case in progress, so a filtered run draws what a full one did. */
inline std::uint64_t case_stream_key(std::uint64_t index = 0) noexcept {
    return stream_key(test_run.settings.seed, test_run.current_case ? test_run.current_case : "", index);
}

/** Line-buffers stdout, prints the machine, reads the settings, arms the crash handlers and
 *  prints the settings line, before the first case. */
inline void start_test_run(int argc, char **argv) noexcept {
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ); // before any output, or the C library keeps its buffering
    print(probe_machine());
    test_run.settings = read_test_settings(argc, argv);
    install_test_signal_handlers();
    print_line(stdout, "- Seed: {}, scale: {}", test_run.settings.seed, test_run.settings.scale);
    print_line(
        stdout, "- Filter: {}",
        test_run.settings.regex_filter.text.empty() ? std::string_view("none") : test_run.settings.regex_filter.text);
}

/** Counts one failed check at @p where , and traps when the run asked to stop at the first. */
inline void record_failed_check(std::source_location where) noexcept {
    std::fflush(stdout); // so a log that merges both streams keeps the failure under its case
    print_line(stderr, "  check failed at {}:{}", where.file_name(), where.line());
    test_run.last_failed_file.store(where.file_name(), std::memory_order_relaxed);
    test_run.last_failed_line.store(where.line(), std::memory_order_relaxed);
    test_run.failed_checks.fetch_add(1, std::memory_order_relaxed);
    if (test_run.settings.trap_on_failure) trap_into_debugger();
}

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line. */
inline bool expect_true(bool is_satisfied, std::source_location where = std::source_location::current()) noexcept {
    if (!is_satisfied) record_failed_check(where);
    return is_satisfied;
}

/** A case label spelled as a regex that matches only itself, every metacharacter escaped. */
struct regex_escaped_t {
    std::string_view label;
};

} // namespace ashvardanian::numkong::test

/** Writes a @ref regex_escaped_t straight to the output, a backslash before each metacharacter. */
template <>
struct fmt::formatter<ashvardanian::numkong::test::regex_escaped_t> {
    constexpr auto parse(fmt::format_parse_context &context) { return context.begin(); }
    template <typename context_type_>
    auto format(ashvardanian::numkong::test::regex_escaped_t const &escaped, context_type_ &context) const {
        auto output = context.out();
        for (char const character : escaped.label) {
            if (std::string_view {"\\^$.|?*+()[]{}"}.find(character) != std::string_view::npos) *output++ = '\\';
            *output++ = character;
        }
        return output;
    }
};

namespace ashvardanian::numkong::test {

/** Prints one case's verdict, and the line that reruns it alone when it failed. */
inline void report_test_case(char const *label, std::size_t failed_checks, char const *skip_reason,
                             double seconds) noexcept {
    if (skip_reason) {
        print_line(stdout, "- {} ... skipped: {}", label, skip_reason);
        return;
    }
    if (failed_checks == 0) {
        print_line(stdout, "- {} ... ok ({:.2f} s)", label, seconds);
        return;
    }
    std::fflush(stdout);
    print_line(stderr, "- {} ... FAILED: {} failed check(s)", label, failed_checks);
    print_line(stderr, "  rerun: {} --seed {} --filter '^{}$'", test_run.settings.program, test_run.settings.seed,
               regex_escaped_t {label});
}

/** Runs one case unless the filter leaves it out, and records whether it passed or skipped. */
template <typename case_function_type_, typename... argument_types_>
void run_test_case(char const *label, case_function_type_ &&case_function, argument_types_ &&...arguments) noexcept {
    if (!test_run.settings.regex_filter.selects(label)) return;
    test_run.current_case = label;
    std::size_t const failed_before = test_run.failed_checks.load();
    auto const started = std::chrono::steady_clock::now();
    case_result_t result {};
    using returned_type = std::invoke_result_t<case_function_type_, argument_types_...>;
    if constexpr (std::is_void_v<returned_type>)
        std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    else result = std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    std::chrono::duration<double> const elapsed = std::chrono::steady_clock::now() - started;
    std::size_t const failed_checks = test_run.failed_checks.load() - failed_before;
    if (result.skip_reason) ++test_run.skipped_cases;
    else if (failed_checks != 0) ++test_run.failed_cases;
    else ++test_run.passed_cases;
    report_test_case(label, failed_checks, result.skip_reason, elapsed.count());
    test_run.current_case = nullptr;
}

/** The exit code: 1 when a case or a check outside any case failed, or nothing ran; 3 when
 *  skips are refused and one happened; else 0. */
inline int finish_test_run() noexcept {
    print_line(stdout, "{} passed, {} failed, {} skipped, seed {}", test_run.passed_cases, test_run.failed_cases,
               test_run.skipped_cases, test_run.settings.seed);
    bool const failed = test_run.failed_cases != 0 || test_run.failed_checks.load() != 0;
    if (failed || test_run.passed_cases + test_run.skipped_cases == 0) return 1;
    if (test_run.skipped_cases != 0 && test_run.settings.refuse_skips) return 3;
    return 0;
}

#pragma endregion Shared Run

/** Reads every @c test_settings_t setting, shrinking the shapes under emulation. */
inline test_settings_t read_test_settings(int argc, char **argv) noexcept {
    test_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "NUMKONG_SEED", "--seed");
    settings.regex_filter = read_regex_filter_setting(argc, argv, "NUMKONG_FILTER", "--filter");
    settings.program = argc != 0 ? argv[0] : "";
    settings.scale = read_real_setting(argc, argv, "NUMKONG_SCALE", "--scale", 1.0);
    settings.trap_on_failure = read_flag_setting(argc, argv, "NUMKONG_TRAP_ON_FAILURE", "--trap-on-failure");
    settings.refuse_skips = read_flag_setting(argc, argv, "NUMKONG_NO_SKIPS", "--no-skips");
    // Python and JS take a comma list of dimensions, of which C++ runs the first
    auto const first_dim = [&](char const *variable, std::string_view flag, std::size_t fallback) noexcept {
        char const *text = setting_text(argc, argv, variable, flag);
        if (!text) return fallback;
        std::optional<std::vector<std::size_t>> const dims = parse_dims(text);
        if (!dims) exit_unparsed(variable, text, "positive counts like 64,128");
        return dims->front();
    };
    settings.time_limit = read_duration_setting(argc, argv, "NUMKONG_TIME_LIMIT", "--time-limit", settings.time_limit);
    if (char const *text = setting_text(argc, argv, "NUMKONG_RANDOM_DISTRIBUTION", "--random-distribution")) {
        std::optional<distribution_t> const distribution = parse_distribution(text);
        if (!distribution) exit_unparsed("NUMKONG_RANDOM_DISTRIBUTION", text, "uniform, lognormal or cauchy");
        settings.random_distribution = *distribution;
    }
    if (!read_flag_setting(argc, argv, "NUMKONG_ASSERT", "--assert", true))
        settings.on_failure = on_failure_t::report_k;
    settings.in_qemu = read_flag_setting(argc, argv, "NUMKONG_IN_QEMU", "--in-qemu");
    settings.ulp_threshold_f32 = read_count_setting(argc, argv, "NUMKONG_ULP_THRESHOLD_F32", "--ulp-threshold-f32",
                                                    settings.ulp_threshold_f32);
    settings.ulp_threshold_f16 = read_count_setting(argc, argv, "NUMKONG_ULP_THRESHOLD_F16", "--ulp-threshold-f16",
                                                    settings.ulp_threshold_f16);
    settings.ulp_threshold_bf16 = read_count_setting(argc, argv, "NUMKONG_ULP_THRESHOLD_BF16", "--ulp-threshold-bf16",
                                                     settings.ulp_threshold_bf16);
    settings.scale_threshold = read_real_setting(argc, argv, "NUMKONG_SCALE_THRESHOLD", "--scale-threshold",
                                                 settings.scale_threshold);
    if (settings.scale_threshold > 1)
        exit_unparsed("NUMKONG_SCALE_THRESHOLD",
                      setting_text(argc, argv, "NUMKONG_SCALE_THRESHOLD", "--scale-threshold"), "a number in (0, 1]");
    settings.dense_dimensions = first_dim("NUMKONG_DIMS", "--dims", settings.dense_dimensions);
    settings.matrix_height = first_dim("NUMKONG_DIMS_HEIGHT", "--dims-height", settings.matrix_height);
    settings.matrix_width = first_dim("NUMKONG_DIMS_WIDTH", "--dims-width", settings.matrix_width);
    settings.matrix_depth = first_dim("NUMKONG_DIMS_DEPTH", "--dims-depth", settings.matrix_depth);
    settings.curved_dimensions = first_dim("NUMKONG_CURVED_DIMS", "--curved-dims", settings.curved_dimensions);
    settings.sparse_dimensions = first_dim("NUMKONG_SPARSE_DIMS", "--sparse-dims", settings.sparse_dimensions);
    settings.shape_points = read_count_setting(argc, argv, "NUMKONG_SHAPE_POINTS", "--shape-points",
                                               settings.shape_points);
    settings.max_coord_angle_degrees = static_cast<float>(read_real_setting(
        argc, argv, "NUMKONG_MAX_COORD_ANGLE", "--max-coord-angle", settings.max_coord_angle_degrees));
    if (settings.max_coord_angle_degrees > 180)
        exit_unparsed("NUMKONG_MAX_COORD_ANGLE",
                      setting_text(argc, argv, "NUMKONG_MAX_COORD_ANGLE", "--max-coord-angle"), "a number in (0, 180]");

    // Shrink dimensions for QEMU — divides whatever value is currently stored,
    // so explicit env-var overrides are proportionally reduced too.
    if (settings.in_qemu) {
        settings.dense_dimensions = nk::max_of<std::size_t>(1, settings.dense_dimensions / 4);
        settings.matrix_height = nk::max_of<std::size_t>(1, settings.matrix_height / 4);
        settings.matrix_width = nk::max_of<std::size_t>(1, settings.matrix_width / 4);
        settings.matrix_depth = nk::max_of<std::size_t>(1, settings.matrix_depth / 4);
        settings.shape_points = nk::max_of<std::size_t>(1, settings.shape_points / 4);
    }
    settings.capabilities = capabilities_detected_cpu();
    return settings;
}

/** Prints NumKong's own settings as "- Name: value", in the grammar each parses from. */
inline void print(test_settings_t const &settings) {
    fmt::println("- Time limit: {}", settings.time_limit);
    fmt::println("- Random distribution: {}", settings.random_distribution);
    fmt::println("- Assert: {}", settings.on_failure == on_failure_t::exit_k);
    fmt::println("- In QEMU: {}", settings.in_qemu);
    fmt::println("- ULP threshold f32: {}", settings.ulp_threshold_f32);
    fmt::println("- ULP threshold f16: {}", settings.ulp_threshold_f16);
    fmt::println("- ULP threshold bf16: {}", settings.ulp_threshold_bf16);
    fmt::println("- Scale threshold: {}", settings.scale_threshold);
    fmt::println("- Dims: {}", settings.dense_dimensions);
    fmt::println("- Dims height: {}", settings.matrix_height);
    fmt::println("- Dims width: {}", settings.matrix_width);
    fmt::println("- Dims depth: {}", settings.matrix_depth);
    fmt::println("- Curved dims: {}", settings.curved_dimensions);
    fmt::println("- Sparse dims: {}", settings.sparse_dimensions);
    fmt::println("- Shape points: {}", settings.shape_points);
    fmt::println("- Max coord angle: {}", settings.max_coord_angle_degrees);
}

/** Names the signal and the case in progress through raw writes, which a crashing thread holding
 *  the stdio lock still gets out, then prints a backtrace where <tt>execinfo</tt> has one, and
 *  re-raises, so a core still dumps. */
#if !NUMKONG_ARCH_WASM_
inline void test_fatal_signal_handler(int signal_number) noexcept {
    char const *name = "a fatal signal";
    switch (signal_number) {
    case SIGSEGV: name = "SIGSEGV"; break;
    case SIGABRT: name = "SIGABRT"; break;
    case SIGILL: name = "SIGILL"; break;
    case SIGFPE: name = "SIGFPE"; break;
#if defined(SIGBUS)
    case SIGBUS: name = "SIGBUS"; break;
#endif
    }
    auto const emit = [](char const *text) noexcept {
#if NUMKONG_OS_WINDOWS_
        _write(2, text, static_cast<unsigned>(std::strlen(text)));
#else
        [[maybe_unused]] auto const written = write(STDERR_FILENO, text, std::strlen(text));
#endif
    };
    emit("\n*** "), emit(name), emit(" in case '"), emit(test_run.current_case ? test_run.current_case : "none");
    emit("' - backtrace follows ***\n");
#if (defined(__GLIBC__) || NUMKONG_OS_APPLE_) && __has_include(<execinfo.h>)
    void *frames[64];
    int const frames_count = backtrace(frames, static_cast<int>(sizeof(frames) / sizeof(frames[0])));
    // The `_fd` form writes without allocating, which is what makes it usable from a handler
    backtrace_symbols_fd(frames, frames_count, STDERR_FILENO);
#endif
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}
#endif

/** Routes every fatal signal but SIGTRAP through @c test_fatal_signal_handler ; WebAssembly has
 *  none. */
inline void install_test_signal_handlers() noexcept {
#if !NUMKONG_ARCH_WASM_
    std::signal(SIGSEGV, test_fatal_signal_handler);
    std::signal(SIGABRT, test_fatal_signal_handler);
    std::signal(SIGILL, test_fatal_signal_handler);
    std::signal(SIGFPE, test_fatal_signal_handler);
#if defined(SIGBUS) // Windows has no bus error to catch
    std::signal(SIGBUS, test_fatal_signal_handler);
#endif
#endif
}

/** Whether the CPU, or the device the cases run on, has @p capability : <tt>#if NUMKONG_TARGET_X</tt>
 *  says built, this says runnable. */
inline bool capable(nk_capability_t capability) noexcept { return (test_run.settings.capabilities & capability) != 0; }

/** Why a case skips without @p capability : <tt>needs</tt> and the constant of its lowest one, as
 *  <tt>nk_cap_haswell_k</tt>. */
inline char const *needs_capability(nk_capability_t capability) noexcept {
    using reason_t = std::array<char, NUMKONG_CAPABILITIES_NAME_CAPACITY>;
    static std::array<reason_t, 64> const reasons = [] {
        std::array<reason_t, 64> spelled {};
        for (std::size_t bit = 0; bit != spelled.size(); ++bit)
            fmt::format_to_n(spelled[bit].data(), spelled[bit].size() - 1, "needs nk_cap_{}_k",
                             capability_names_t {nk_capability_t {1} << bit});
        return spelled;
    }();
    return reasons[static_cast<std::size_t>(std::countr_zero(capability))].data();
}

/**
 *  @brief A format string bundled with the @c std::source_location of the call that spelled it.
 *
 *  The implicit conversion from a string literal captures the caller's line behind the variadic
 *  arguments of a check's context, where a defaulted trailing parameter cannot go, and checks the
 *  pattern against those arguments at compile time, as @c fmt::format_string does.
 */
template <typename... args_types_>
struct located_format {
    fmt::format_string<args_types_...> format;
    std::source_location location;

    template <typename text_type_>
    consteval located_format(text_type_ const &text,
                             std::source_location location = std::source_location::current()) noexcept
        : format(text), location(location) {}
};

/** Prints @p context formatted with @p args , indented under a failed check's location line. */
template <typename... args_types_>
void print_context_(located_format<std::type_identity_t<args_types_>...> context, args_types_ const &...args) noexcept {
    fmt::print(stderr, "    ");
    fmt::vprint(stderr, context.format.get(), fmt::make_format_args(args...));
    fmt::print(stderr, "\n");
}

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line and prints
 *  @p context formatted with @p args beneath it. */
template <typename... args_types_>
bool expect_true(bool is_satisfied, located_format<std::type_identity_t<args_types_>...> context,
                 args_types_ const &...args) noexcept {
    if (is_satisfied) return true;
    record_failed_check(context.location);
    print_context_<args_types_...>(context, args...);
    return false;
}

/** A value as a failed check prints it: as {fmt} spells it, or else as the @c double it converts
 *  to. */
template <typename value_type_>
auto printable_(value_type_ const &value) noexcept {
    if constexpr (fmt::is_formattable<value_type_>::value) return value;
    else return to_double(value);
}

/** Holds when @p actual equals @p wanted ; otherwise counts a failed check at the caller's line and
 *  prints both. */
template <typename actual_type_, typename wanted_type_>
bool expect_equal(actual_type_ const &actual, wanted_type_ const &wanted,
                  std::source_location where = std::source_location::current()) {
    if (actual == wanted) return true;
    record_failed_check(where);
    fmt::println(stderr, "    expected {}, got {}", printable_(wanted), printable_(actual));
    return false;
}

/** The same, printing @p context formatted with @p args under both values. */
template <typename actual_type_, typename wanted_type_, typename... args_types_>
bool expect_equal(actual_type_ const &actual, wanted_type_ const &wanted,
                  located_format<std::type_identity_t<args_types_>...> context, args_types_ const &...args) {
    if (actual == wanted) return true;
    record_failed_check(context.location);
    fmt::println(stderr, "    expected {}, got {}", printable_(wanted), printable_(actual));
    print_context_<args_types_...>(context, args...);
    return false;
}

/** Whether @p actual is within @p tolerance of @p wanted , relative to the largest of 1 and both
 *  magnitudes. */
template <typename actual_type_, typename wanted_type_>
bool near_(actual_type_ const &actual, wanted_type_ const &wanted, double tolerance) noexcept {
    double const actual_value = to_double(actual), wanted_value = to_double(wanted);
    double const magnitude = std::fmax(1.0, std::fmax(std::fabs(actual_value), std::fabs(wanted_value)));
    return std::fabs(actual_value - wanted_value) <= tolerance * magnitude;
}

/** Holds when @p actual is within @p tolerance of @p wanted , relative to the largest of 1 and both
 *  magnitudes; otherwise counts a failed check at the caller's line and prints both. */
template <typename actual_type_, typename wanted_type_>
bool expect_near(actual_type_ const &actual, wanted_type_ const &wanted, double tolerance,
                 std::source_location where = std::source_location::current()) {
    if (near_(actual, wanted, tolerance)) return true;
    record_failed_check(where);
    fmt::println(stderr, "    expected {} within {}, got {}", printable_(wanted), tolerance, printable_(actual));
    return false;
}

/** The same, printing @p context formatted with @p args under both values. */
template <typename actual_type_, typename wanted_type_, typename... args_types_>
bool expect_near(actual_type_ const &actual, wanted_type_ const &wanted, double tolerance,
                 located_format<std::type_identity_t<args_types_>...> context, args_types_ const &...args) {
    if (near_(actual, wanted, tolerance)) return true;
    record_failed_check(context.location);
    fmt::println(stderr, "    expected {} within {}, got {}", printable_(wanted), tolerance, printable_(actual));
    print_context_<args_types_...>(context, args...);
    return false;
}

/** Calls the dispatch point @p best_ over @p capabilities with the arguments of its capability
 *  kernels, whose last one, the stream or a pack size's output, follows the mask. */
template <auto best_, typename... arguments_types_>
nk_status_t call_best(nk_capability_t capabilities, arguments_types_... arguments) noexcept {
    std::tuple<arguments_types_...> const tuple {arguments...};
    return [&]<std::size_t... indices_>(std::index_sequence<indices_...>) {
        return best_(std::get<indices_>(tuple)..., capabilities, std::get<sizeof...(indices_)>(tuple));
    }(std::make_index_sequence<sizeof...(arguments_types_) - 1> {});
}

/** The dispatch point @p best_ in the shape of its capability kernels, over the CPU capabilities
 *  this process enables: callable like them, and convertible to their function pointers. */
template <auto best_>
inline constexpr auto cpu_best =
    [](auto... arguments) noexcept { return call_best<best_>(enabled_capabilities(), arguments...); };

/**
 *  @brief Compute the ULP, Units in Last Place, distance between two floating-point values.
 *
 *  ULP distance is the number of representable floating-point numbers between a and b.
 *  This is the gold standard for comparing floating-point implementations.
 *
 *  Uses the XOR transformation from Bruce Dawson's algorithm to handle all sign combinations.
 *
 *  @see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 *  @see https://en.wikipedia.org/wiki/Unit_in_the_last_place
 */
template <typename scalar_type_>
std::uint64_t ulp_distance(scalar_type_ a, scalar_type_ b) noexcept {
    // Handle special cases - skip float checks for integer types
    if constexpr (!nk::is_integral_dtype<scalar_type_>()) {
        if (std::isnan(static_cast<double>(a)) || std::isnan(static_cast<double>(b)))
            return (std::numeric_limits<std::uint64_t>::max)();
    }
    if (a == b) return 0; // Also handles +0 == -0

    // Use the XOR transformation from Bruce Dawson's "Comparing Floating Point Numbers"
    // This transforms float bit patterns to an ordered integer representation where
    // the integer difference equals the ULP distance.
    if constexpr (sizeof(scalar_type_) == 4) {
        std::int32_t ia, ib;
        std::memcpy(&ia, &a, sizeof(ia));
        std::memcpy(&ib, &b, sizeof(ib));

        // Transform negative floats: flip all bits except sign to reverse their ordering
        // This makes the integer representation monotonically ordered with float value
        if (ia < 0) ia ^= 0x7FFFFFFF;
        if (ib < 0) ib ^= 0x7FFFFFFF;

        // Compute absolute difference using 64-bit arithmetic to avoid overflow
        std::int64_t diff = static_cast<std::int64_t>(ia) - static_cast<std::int64_t>(ib);
        return static_cast<std::uint64_t>(diff < 0 ? -diff : diff);
    }
    else if constexpr (sizeof(scalar_type_) == 8) {
        std::int64_t ia, ib;
        std::memcpy(&ia, &a, sizeof(ia));
        std::memcpy(&ib, &b, sizeof(ib));

        if (ia < 0) ia ^= 0x7FFFFFFFFFFFFFFFLL;
        if (ib < 0) ib ^= 0x7FFFFFFFFFFFFFFFLL;

        // For 64-bit, handle potential overflow in subtraction when signs differ
        if ((ia >= 0) != (ib >= 0)) {
            // Different signs after transformation: distance = |ia| + |ib|
            // Safe negation that handles INT64_MIN
            auto safe_abs = [](std::int64_t x) -> std::uint64_t {
                return x < 0 ? static_cast<std::uint64_t>(~x) + 1 : static_cast<std::uint64_t>(x);
            };
            return safe_abs(ia) + safe_abs(ib);
        }
        // Same sign: simple subtraction (no overflow possible)
        return ia >= ib ? static_cast<std::uint64_t>(ia - ib) : static_cast<std::uint64_t>(ib - ia);
    }
    else {
        // For f16/bf16, convert to f32 and compute there
        return ulp_distance(static_cast<float>(a), static_cast<float>(b));
    }
}

/** @copydoc ulp_distance, for narrow floats as the floats they widen to. */
template <typename scalar_type_>
    requires(is_narrow_float<scalar_type_>())
std::uint64_t ulp_distance(scalar_type_ a, scalar_type_ b) noexcept {
    return ulp_distance(narrow_to_f32(a), narrow_to_f32(b));
}

template <typename scalar_type_>
std::uint64_t integer_distance(scalar_type_ a, scalar_type_ b) noexcept {
    auto ordered = [](scalar_type_ value) noexcept -> std::uint64_t {
        if constexpr (nk::is_signed_dtype<scalar_type_>())
            return static_cast<std::uint64_t>(static_cast<std::int64_t>(value)) ^ (1ull << 63);
        else return static_cast<std::uint64_t>(value);
    };
    std::uint64_t a_ordered = ordered(a), b_ordered = ordered(b);
    return a_ordered >= b_ordered ? a_ordered - b_ordered : b_ordered - a_ordered;
}

#pragma region Tracked References

/** The square root of @p value in a reference type: @c f118_t and complex ones by their own
 *  arithmetic, @c f64_t and @c f32_t by @c std::sqrt, as a reference never runs a kernel. */
template <typename value_type_>
value_type_ square_root_reference(value_type_ value) noexcept {
    if constexpr (std::is_same_v<value_type_, f64_t> || std::is_same_v<value_type_, f32_t>)
        return value_type_(std::sqrt(value.raw_));
    else return value.sqrt();
}

/** The reciprocal square root of @p value in a reference type, like @c square_root_reference. */
template <typename value_type_>
value_type_ reciprocal_square_root_reference(value_type_ value) noexcept {
    if constexpr (std::is_same_v<value_type_, f64_t> || std::is_same_v<value_type_, f32_t>)
        return value_type_(1 / std::sqrt(value.raw_));
    else return value.rsqrt();
}

/**
 *  @brief A reference value that also carries the scale of any correct kernel's rounding error.
 *
 *  The arithmetic computing @c value also sums the magnitudes of the terms it combines, and counts
 *  the most roundings on any path from an input. No summation order rounds more often than the
 *  serial reference, so the bound holds for every kernel. Complex values take their modulus, which
 *  bounds both parts of their sums and products.
 */
template <typename value_type_>
struct tracked {
    using reference_t = value_type_;

    value_type_ value {};
    double magnitude = 0;

    /** Roundings on the longest path through products, quotients and functions: a term's own. */
    std::size_t term_roundings = 0;

    /** Roundings on the longest path through sums and differences: the reduction's. */
    std::size_t sum_roundings = 0;

    static constexpr nk_dtype_t dtype() noexcept { return value_type_::dtype(); }
    static constexpr bool is_integer() noexcept { return false; }
    static tracked finite_max() noexcept { return tracked(value_type_::finite_max()); }

    tracked() = default;
    tracked(value_type_ value, double magnitude, std::size_t term_roundings, std::size_t sum_roundings) noexcept
        : value(value), magnitude(magnitude), term_roundings(term_roundings), sum_roundings(sum_roundings) {}

    /** An input, exact in the reference type and not rounded yet. */
    template <typename input_type_>
    explicit tracked(input_type_ input) noexcept
        : value(widen<value_type_>(input)), magnitude(to_double(value.abs())) {}

    friend tracked operator+(tracked const &a, tracked const &b) noexcept {
        return {a.value + b.value, a.magnitude + b.magnitude, nk::max_of(a.term_roundings, b.term_roundings),
                nk::max_of(a.sum_roundings, b.sum_roundings) + 1};
    }
    friend tracked operator-(tracked const &a, tracked const &b) noexcept {
        return {a.value - b.value, a.magnitude + b.magnitude, nk::max_of(a.term_roundings, b.term_roundings),
                nk::max_of(a.sum_roundings, b.sum_roundings) + 1};
    }
    friend tracked operator*(tracked const &a, tracked const &b) noexcept {
        // Each part of a complex product also adds two real products
        std::size_t const own_roundings = value_type_::is_complex() ? 2 : 1;
        return {a.value * b.value, a.magnitude * b.magnitude, a.term_roundings + b.term_roundings + own_roundings,
                a.sum_roundings + b.sum_roundings};
    }

    /** Relative errors add in a quotient as in a product, and the division rounds once more. */
    friend tracked operator/(tracked const &a, tracked const &b) noexcept {
        double const divisor = to_double(b.value.abs());
        return {a.value / b.value, a.magnitude * b.magnitude / (divisor * divisor),
                a.term_roundings + b.term_roundings + 1, a.sum_roundings + b.sum_roundings};
    }
    friend bool operator>(tracked const &a, tracked const &b) noexcept { return a.value > b.value; }
    friend bool operator<(tracked const &a, tracked const &b) noexcept { return a.value < b.value; }
    friend bool operator==(tracked const &a, tracked const &b) noexcept { return a.value == b.value; }
    tracked saturating_add(tracked const &other) const noexcept { return *this + other; }
    tracked saturating_mul(tracked const &other) const noexcept { return *this * other; }

    /** The radicand's error shrinks by the derivative 1 / (2√v), and the root rounds once more. */
    tracked sqrt() const noexcept {
        value_type_ const root = square_root_reference(value);
        double const root_magnitude = to_double(root);
        return {root, nk::max_of(magnitude / (2 * root_magnitude), root_magnitude), term_roundings + 1, sum_roundings};
    }

    /** The argument's error shrinks by the derivative 1 / v, and the logarithm rounds once more. */
    tracked log() const noexcept {
        value_type_ const logarithm = value.log();
        double const argument = to_double(value.abs());
        return {logarithm, nk::max_of(magnitude / argument, to_double(logarithm.abs())), term_roundings + 1,
                sum_roundings};
    }
};

/** The reference a sum over @p input_type_ is checked against: integer results exact in their own
 *  type, floating ones tracked. */
template <typename input_type_, typename result_type_>
using bounded_reference_for =
    std::conditional_t<nk::is_integral_dtype<result_type_>(), result_type_, tracked<reference_for<input_type_>>>;

/** Half a unit in the last place of @p scalar_type_ at @p value, subnormals included: what rounding
 *  a result into that type adds. Integers step by one everywhere. */
template <typename scalar_type_>
double half_ulp(double value) noexcept {
    if constexpr (nk::is_integral_dtype<scalar_type_>()) return 0.5;
    else {
        using component_t = typename scalar_type_::component_t;
        int const smallest_normal_exponent = std::ilogb(to_double(component_t::positive_min()));
        int const exponent = nk::max_of(std::ilogb(value), smallest_normal_exponent);
        return std::ldexp(1.0, exponent - static_cast<int>(component_t::mantissa_bits()) - 1);
    }
}

#pragma endregion Tracked References

/** Accumulator for error statistics across multiple test trials. */
struct error_stats_t {
    comparison_family_t family = comparison_family_t::approximate_k;
    nk_f64_t term_error_bound = 0;
    nk_f64_t sum_error_bound = 0;

    /** The type of the results recorded, which picks their ULP threshold. */
    nk_dtype_t result_dtype = nk_dtype_unknown_k;

    nk_f64_t max_absolute_error = 0;
    nk_f64_t max_relative_error = 0;
    nk_f64_t max_reference = 0;
    std::uint64_t max_ulp = 0;
    f118_t sum_ulp = f118_t();

    nk_f64_t max_bound_ratio = 0;

    std::size_t count = 0;
    std::size_t exact_matches = 0;
    std::size_t failed_expectations = 0;
    bool saw_floating_distance = false;
    char const *first_failure = nullptr;

    explicit error_stats_t(comparison_family_t family = comparison_family_t::approximate_k) noexcept : family(family) {}

    /** Judges results against tracked references, each term adding up to @p term_error_bound of
     *  Σ|terms|, like the `nk_*_error_bound` helpers return; zero demands exact results. */
    explicit error_stats_t(nk_f64_t term_error_bound) noexcept : error_stats_t(term_error_bound, term_error_bound) {}

    /** Judges results whose terms round within @p term_error_bound and whose sums, kept in a wider
     *  type, round within @p sum_error_bound. */
    error_stats_t(nk_f64_t term_error_bound, nk_f64_t sum_error_bound) noexcept
        : family(term_error_bound == 0 ? comparison_family_t::exact_k : comparison_family_t::bounded_k),
          term_error_bound(term_error_bound), sum_error_bound(sum_error_bound) {}

    /** Records a boolean property, which @p property names when it does not hold. Only the first
     *  property to fail records a failed check at @p where , as a property can fail on every trial,
     *  and none does under @c NUMKONG_ASSERT=0. */
    bool expect_true(bool is_met, char const *property,
                     std::source_location where = std::source_location::current()) noexcept {
        if (!is_met && !first_failure) {
            first_failure = property;
            if (test_run.settings.on_failure == on_failure_t::exit_k) {
                record_failed_check(where);
                fmt::println(stderr, "    {}", property);
            }
        }
        failed_expectations += !is_met;
        accumulate(static_cast<int>(is_met), 1);
        return is_met;
    }

    /** Records a kernel's @p status, failing it by the status's name when it wrote no result, and
     *  returns whether it succeeded. */
    bool expect_status(nk_status_t status, std::source_location where = std::source_location::current()) noexcept {
        return expect_true(status == nk_success_k, nk_status_name(status), where);
    }

    /** Records a C++ wrapper's @p status, like the C one. */
    bool expect_status(nk::status_t status, std::source_location where = std::source_location::current()) noexcept {
        return expect_status(static_cast<nk_status_t>(status), where);
    }

    /** Records one result against its reference, failing on NaN or when the error exceeds @p bound
     *  plus the rounding into @p actual_type_. Past that type's finite range, a saturated or
     *  overflowed result of the reference's sign is exact, and non-finite references are only
     *  counted, as @c accumulate_scalar does. */
    template <typename actual_type_, typename expected_type_>
    void accumulate_bounded(actual_type_ actual, expected_type_ expected, nk_f64_t bound) noexcept {
        nk_f64_t const result = to_double(actual), reference = to_double(expected);
        if (!std::isfinite(reference)) return accumulate_scalar(actual, reference);
        nk_f64_t const limit = to_double(actual_type_::finite_max());
        bool const saturated = std::fabs(reference) > limit && std::fabs(result) >= limit &&
                               std::signbit(result) == std::signbit(reference);
        nk_f64_t const error = saturated || result == reference ? 0 : std::fabs(result - reference);
        nk_f64_t const ratio = error == 0 ? 0 : error / (bound + half_ulp<actual_type_>(reference));
        max_bound_ratio = std::isnan(ratio) ? std::numeric_limits<nk_f64_t>::infinity()
                                            : nk::max_of(max_bound_ratio, ratio);
        accumulate_scalar(actual, reference);
    }

    template <typename actual_type_, typename expected_type_>
    void accumulate(actual_type_ actual, expected_type_ expected) noexcept {
        if constexpr (nk::is_complex_dtype<actual_type_>())
            accumulate_scalar(actual.real(), expected.real()), accumulate_scalar(actual.imag(), expected.imag());
        else accumulate_scalar(actual, expected);
    }

    /** Records a result against a tracked reference: integers exactly once it rounds into them,
     *  floats within @c term_error_bound of its magnitude per rounding of a term, and
     *  @c sum_error_bound per rounding of a sum. Once that magnitude passes the result's finite
     *  range, a partial sum may overflow before later terms cancel, so either infinity passes. */
    template <typename actual_type_, typename value_type_>
    void accumulate(actual_type_ actual, tracked<value_type_> const &expected) noexcept {
        nk_f64_t const bound = ((expected.term_roundings + 1) * term_error_bound +
                                expected.sum_roundings * sum_error_bound) *
                               expected.magnitude;
        if constexpr (nk::is_integral_dtype<actual_type_>())
            accumulate_scalar(actual, narrow<actual_type_>(expected.value));
        else if constexpr (nk::is_complex_dtype<actual_type_>())
            accumulate_bounded(actual.real(), expected.value.real(), bound),
                accumulate_bounded(actual.imag(), expected.value.imag(), bound);
        else if (std::isinf(to_double(actual)) && expected.magnitude > to_double(actual_type_::finite_max()))
            accumulate_scalar(actual, to_double(actual));
        else accumulate_bounded(actual, expected.value, bound);
    }

    template <typename actual_type_, typename expected_type_>
    void accumulate_scalar(actual_type_ actual, expected_type_ expected) noexcept {
        actual_type_ expected_as_actual;
        if constexpr (std::is_same_v<expected_type_, f118_t> || std::is_same_v<expected_type_, f118c_t>)
            expected_as_actual = narrow<actual_type_>(expected);
        else if constexpr (nk::is_integral_dtype<expected_type_>() && nk::is_integral_dtype<actual_type_>())
            expected_as_actual = actual_type_(expected);
        else expected_as_actual = narrow<actual_type_>(to_double(expected));

        std::uint64_t ulps;
        if constexpr (nk::is_integral_dtype<actual_type_>())
            ulps = family == comparison_family_t::exact_k ? integer_distance(actual, expected_as_actual)
                                                          : ulp_distance(actual, expected_as_actual);
        else ulps = ulp_distance(actual, expected_as_actual);

        // A NaN reference makes the distance meaningless; a NaN result against a number fails
        if constexpr (!nk::is_integral_dtype<actual_type_>())
            if (ulps == (std::numeric_limits<std::uint64_t>::max)()) {
                if (!std::isnan(to_double(expected_as_actual))) expect_true(false, "NaN result for a number");
                return;
            }

        if constexpr (!nk::is_integral_dtype<actual_type_>()) saw_floating_distance = true;
        if constexpr (requires { actual_type_::dtype(); }) result_dtype = actual_type_::dtype();

        if constexpr (!nk::is_integral_dtype<actual_type_>() || std::is_integral_v<actual_type_>) {
            nk_f64_t exp_f64 = to_double(expected_as_actual);
            nk_f64_t act_f64 = to_double(actual);

            nk_f64_t absolute_error = std::fabs(exp_f64 - act_f64);
            nk_f64_t relative_error = exp_f64 != 0 ? absolute_error / std::fabs(exp_f64) : absolute_error;

            max_absolute_error = nk::max_of(max_absolute_error, absolute_error);
            max_relative_error = nk::max_of(max_relative_error, relative_error);
            max_reference = nk::max_of(max_reference, std::fabs(exp_f64));
        }

        // Always update ULP metrics (works for both integer and float)
        max_ulp = nk::max_of(max_ulp, ulps);
        sum_ulp += f118_t(ulps);

        count++;
        if (ulps == 0) exact_matches++;
    }

    nk_f64_t mean_ulp() const noexcept {
        // On 32-bit WASM we need this ugly casting sequence to avoid adding more `f118_t` constructors
        return count > 0 ? to_double(sum_ulp / f118_t(to_double(count))) : 0;
    }
    std::size_t mismatches() const noexcept { return count - exact_matches; }

    void reset() noexcept {
        error_stats_t fresh {family};
        fresh.term_error_bound = term_error_bound;
        *this = fresh;
    }

    void merge(error_stats_t const &other) noexcept {
        if (other.count == 0) return;
        if (count == 0) family = other.family, term_error_bound = other.term_error_bound;
        else expect_true(family == other.family, "merged stats of another comparison family");
        max_absolute_error = nk::max_of(max_absolute_error, other.max_absolute_error);
        max_relative_error = nk::max_of(max_relative_error, other.max_relative_error);
        max_reference = nk::max_of(max_reference, other.max_reference);
        max_ulp = nk::max_of(max_ulp, other.max_ulp);
        sum_ulp += other.sum_ulp;
        max_bound_ratio = nk::max_of(max_bound_ratio, other.max_bound_ratio);
        count += other.count;
        exact_matches += other.exact_matches;
        failed_expectations += other.failed_expectations;
        saw_floating_distance = saw_floating_distance || other.saw_floating_distance;
    }
};

/** Whether @p stats break their family's bound, or a property they recorded failed. */
inline bool should_fail(error_stats_t const &stats) noexcept {
    if (stats.failed_expectations) return true;
    comparison_family_spec_t const spec = comparison_family_spec(stats.family);
    switch (spec.failure_mode) {
    case comparison_failure_mode_t::exact_distance_k:
        if (!stats.saw_floating_distance) return stats.max_ulp > 0;
        return stats.max_ulp > test_run.settings.ulp_threshold_for(stats.result_dtype);
    case comparison_failure_mode_t::ulp_threshold_k:
        return stats.max_ulp > test_run.settings.ulp_threshold_for(stats.result_dtype);
    case comparison_failure_mode_t::scale_threshold_k:
        return stats.max_absolute_error > test_run.settings.scale_threshold * stats.max_reference;
    case comparison_failure_mode_t::bound_threshold_k: return !(stats.max_bound_ratio <= 1);
    }
    return false;
}

/** Prints @p stats under their case's verdict, each figure after the name of its family's
 *  column. */
inline void print_stats_row(error_stats_t const &stats) noexcept {
    std::array<char const *, 5> const &names = comparison_family_spec(stats.family).column_labels;
    switch (stats.family) {
    case comparison_family_t::exact_k:
        fmt::println("    {} {}, {} {:.1f}, {} {:.2e}, {} {}, {} {}", names[0], stats.max_ulp, names[1],
                     stats.mean_ulp(), names[2], stats.max_absolute_error, names[3], stats.mismatches(), names[4],
                     stats.exact_matches);
        break;
    case comparison_family_t::approximate_k:
    case comparison_family_t::normalized_reduction_k:
        fmt::println("    {} {:.2e}, {} {:.2e}, {} {:.2e}, {} {}, {} {}", names[0], stats.max_absolute_error, names[1],
                     stats.max_relative_error, names[2], stats.mean_ulp(), names[3], stats.max_ulp, names[4],
                     stats.exact_matches);
        break;
    case comparison_family_t::bounded_k:
        fmt::println("    {} {:.2e}, {} {:.2e}, {} {:.2e}, {} {}, {} {}", names[0], stats.max_absolute_error, names[1],
                     stats.max_relative_error, names[2], stats.max_bound_ratio, names[3], stats.max_ulp, names[4],
                     stats.exact_matches);
        break;
    }
    // The counters say how many properties failed; this says which one.
    if (stats.first_failure && (stats.mismatches() || stats.failed_expectations))
        fmt::println("    first failure: {}", stats.first_failure);
    if (test_run.settings.on_failure == on_failure_t::report_k && should_fail(stats))
        fmt::println("    fails its accuracy check, which NUMKONG_ASSERT=0 only reports");
    std::fflush(stdout);
}

/**
 *  @brief Runs @p test_fn over @p arguments as the case @p label , skipped without @p capability ,
 *      and prints the statistics it returns beneath the case's verdict.
 *
 *  Statistics past their family's bound fail the case unless @c NUMKONG_ASSERT=0.
 */
template <typename test_function_type_, typename... arguments_types_>
void run_kernel_case(char const *label, test_function_type_ test_fn, nk_capability_t capability,
                     arguments_types_ &&...arguments) {
    std::optional<error_stats_t> stats;
    run_test_case(label, [&]() -> case_result_t {
        if (!capable(capability)) return case_result_t::skipped(needs_capability(capability));
        std::size_t const failed_before = test_run.failed_checks.load();
        stats = test_fn(std::forward<arguments_types_>(arguments)...);
        // A kernel failing only its statistics still records a check, so a trap stops after it too
        if (test_run.failed_checks.load() == failed_before && test_run.settings.on_failure == on_failure_t::exit_k)
            expect_true(!should_fail(*stats), "the statistics below break their family's bound");
        return {};
    });
    if (stats) print_stats_row(*stats);
}

/** The same for a test templated or overloaded on its kernels' types, which @p kernels pick. */
template <typename stats_type_ = error_stats_t,
          typename... kernels_types_> // ! `stats_type_` makes this the more specialized
void run_kernel_case(char const *label, std::type_identity_t<stats_type_ (*)(kernels_types_...)> test_fn,
                     nk_capability_t capability, kernels_types_... kernels) {
    run_kernel_case(label, [&] { return test_fn(kernels...); }, capability);
}

/** Factory function to allocate vectors, potentially raising bad-allocs. */
template <typename type_>
[[nodiscard]] nk::vector<type_> make_vector(std::size_t n) {
    auto result = nk::vector<type_>::zeros(n);
#if defined(__cpp_exceptions) && __cpp_exceptions
    if (!result) throw std::bad_alloc();
#else
    if (!result) std::abort();
#endif
    return std::move(result.value);
}

/** Shapes @p vector to @p count values, growing its storage only when a case needs more. */
template <typename value_type_, typename allocator_type_>
void resize_vector(nk::vector<value_type_, allocator_type_> &vector, std::size_t count) {
    if (nk::failed(vector.reserve(count)) || nk::failed(vector.resize(count))) std::abort();
}

/** Shapes @p matrix to @p rows × @p columns, growing its storage only when a case needs more. */
template <typename value_type_>
nk::matrix_span<value_type_> resize_matrix(nk::matrix<value_type_> &matrix, std::size_t rows, std::size_t columns) {
    if (nk::failed(matrix.reserve(rows * columns)) || nk::failed(matrix.resize({rows, columns}))) std::abort();
    return matrix.span();
}

/**
 *  @brief Fill buffer with random values, respecting the distribution setting.
 *
 *  Draws like the SDK's @c nk::fill_uniform, @c nk::fill_lognormal and @c nk::fill_cauchy with
 *  their default bounds, per @c NUMKONG_RANDOM_DISTRIBUTION, rounding into narrow floats through
 *  their conversion exports, so header-only suites draw the same inputs.
 */
template <typename scalar_type_, typename allocator_type_, typename generator_type_>
void fill_random(generator_type_ &generator, nk::vector<scalar_type_, allocator_type_> &vector) {
    scalar_type_ *values = vector.values_data();
    std::size_t const count = vector.size_values();
    distribution_t const distribution = test_run.settings.random_distribution;
    auto const clamp = [](auto value, auto low, auto high) noexcept {
        return value < low ? low : value > high ? high : value;
    };
    if constexpr (nk::dimensions_per_value<scalar_type_>() > 1) {
        // Packed bits and nibbles take random bytes, or the clamped draws of their nibbles
        if (distribution == distribution_t::uniform_k) {
            std::uniform_int_distribution<std::uint32_t> bytes(0, 255);
            for (std::size_t i = 0; i < count; ++i)
                values[i] = scalar_type_::from_raw(static_cast<typename scalar_type_::raw_t>(bytes(generator)));
        }
        else {
            std::lognormal_distribution<float> lognormal(0.0f, 0.5f);
            std::cauchy_distribution<float> cauchy(0.0f, 1.0f);
            for (std::size_t i = 0; i < count; ++i)
                for (std::size_t j = 0; j < nk::dimensions_per_value<scalar_type_>(); ++j) {
                    float const draw = distribution == distribution_t::lognormal_k ? lognormal(generator)
                                                                                   : cauchy(generator);
                    values[i][j] = static_cast<std::uint8_t>(clamp(draw, 0.0f, 255.0f));
                }
        }
    }
    else if constexpr (nk::is_integral_dtype<scalar_type_>()) {
        using small_t = std::conditional_t<nk::is_signed_dtype<scalar_type_>(), std::int32_t, std::uint32_t>;
        using large_t = std::conditional_t<nk::is_signed_dtype<scalar_type_>(), std::int64_t, std::uint64_t>;
        using draw_t = std::conditional_t<sizeof(scalar_type_) <= 4, small_t, large_t>;
        using float_t = std::conditional_t<sizeof(scalar_type_) <= 4, float, double>;
        float_t const low = static_cast<float_t>(to_double(nk::finite_min<scalar_type_>())),
                      high = static_cast<float_t>(to_double(nk::finite_max<scalar_type_>()));
        if (distribution == distribution_t::uniform_k) {
            std::uniform_int_distribution<draw_t> uniform(nk::finite_min<scalar_type_>(),
                                                          nk::finite_max<scalar_type_>());
            for (std::size_t i = 0; i < count; ++i) values[i] = static_cast<scalar_type_>(uniform(generator));
        }
        else if (distribution == distribution_t::lognormal_k) {
            std::lognormal_distribution<float_t> lognormal(0, float_t(0.5));
            std::bernoulli_distribution sign(0.5);
            constexpr float_t signs[] = {1, -1};
            for (std::size_t i = 0; i < count; ++i) {
                float_t draw = lognormal(generator);
                if constexpr (nk::is_signed_dtype<scalar_type_>()) draw *= signs[sign(generator)];
                values[i] = static_cast<scalar_type_>(clamp(draw, low, high));
            }
        }
        else {
            std::cauchy_distribution<float_t> cauchy(0, 1);
            for (std::size_t i = 0; i < count; ++i)
                values[i] = static_cast<scalar_type_>(clamp(cauchy(generator), low, high));
        }
    }
    else if constexpr (nk::is_complex_dtype<scalar_type_>()) {
        using component_t = typename scalar_type_::component_t;
        using float_t = std::conditional_t<sizeof(component_t) <= 4, float, double>;
        float_t const high = static_cast<float_t>(to_double(nk::finite_max<component_t>())),
                      low = static_cast<float_t>(to_double(nk::finite_min<component_t>()));
        std::uniform_real_distribution<float_t> uniform(-10, 10);
        std::lognormal_distribution<float_t> lognormal(0, float_t(0.5));
        std::cauchy_distribution<float_t> cauchy(0, 1);
        std::bernoulli_distribution sign(0.5);
        constexpr float_t signs[] = {1, -1};
        auto const draw = [&]() noexcept -> float_t {
            if (distribution == distribution_t::uniform_k) return uniform(generator);
            if (distribution == distribution_t::lognormal_k)
                return clamp(lognormal(generator) * signs[sign(generator)], low, high);
            return clamp(cauchy(generator), low, high);
        };
        for (std::size_t i = 0; i < count; ++i) {
            float_t const real = draw(), imag = draw();
            values[i] = scalar_type_(narrow<component_t>(real), narrow<component_t>(imag));
        }
    }
    else {
        using float_t = std::conditional_t<sizeof(scalar_type_) <= 4, float, double>;
        float_t const high = static_cast<float_t>(to_double(nk::finite_max<scalar_type_>())),
                      low = static_cast<float_t>(to_double(nk::finite_min<scalar_type_>()));
        if (distribution == distribution_t::uniform_k) {
            std::uniform_real_distribution<float_t> uniform(-10, 10);
            for (std::size_t i = 0; i < count; ++i) values[i] = narrow<scalar_type_>(uniform(generator));
        }
        else if (distribution == distribution_t::lognormal_k) {
            std::lognormal_distribution<float_t> lognormal(0, float_t(0.5));
            std::bernoulli_distribution sign(0.5);
            constexpr float_t signs[] = {1, -1};
            for (std::size_t i = 0; i < count; ++i) {
                float_t draw = lognormal(generator);
                if constexpr (nk::is_signed_dtype<scalar_type_>()) draw *= signs[sign(generator)];
                values[i] = narrow<scalar_type_>(clamp(draw, low, high));
            }
        }
        else {
            std::cauchy_distribution<float_t> cauchy(0, 1);
            for (std::size_t i = 0; i < count; ++i)
                values[i] = narrow<scalar_type_>(clamp(cauchy(generator), low, high));
        }
    }
}

/** Fills @p vector with uniformly random bits, reaching every NaN, infinity, subnormal and
 *  out-of-range value its type can hold. */
template <typename scalar_type_, typename allocator_type_, typename generator_type_>
void fill_random_bits(generator_type_ &generator, nk::vector<scalar_type_, allocator_type_> &vector) {
    std::uniform_int_distribution<unsigned> byte(0, 255);
    auto *bytes = reinterpret_cast<std::uint8_t *>(vector.raw_values_data());
    for (std::size_t index = 0; index < vector.size_bytes(); index++)
        bytes[index] = static_cast<std::uint8_t>(byte(generator));
}

#pragma region Backends

/** Runs CPU kernels in place: operands in host memory, direct calls, results readable at once,
 *  keeping the first failed status. */
struct host_backend_t {

    /** The allocator every kernel operand comes from. */
    template <typename value_type_>
    using allocator = aligned_allocator<value_type_>;

    /** The first failed call's status since the last @c synchronize, which skips later calls. */
    nk_status_t failure = nk_success_k;

    /** Row stride for @p row_bytes: exactly one row, keeping the tightest stride covered. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return row_bytes; }

    /** Copies @p bytes between host buffers. */
    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        std::memcpy(destination, source, bytes);
        return nk_success_k;
    }

    /** Zeroes @p bytes of a host buffer. */
    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        std::memset(destination, 0, bytes);
        return nk_success_k;
    }

    /** Calls @p kernel with @p arguments and the null stream of the CPU, unless a call failed. */
    template <typename kernel_type_, typename... arguments_types_>
    void call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        if (failure == nk_success_k) failure = kernel(arguments..., nullptr);
    }

    /** Returns and clears the first failure since the last call to it. */
    nk_status_t synchronize() noexcept { return std::exchange(failure, nk_success_k); }
};

template <typename backend_type_>
backend_type_ make_backend() {
    return {};
}

/** The allocator @p backend hands out @p value_type_ from: stateless, unless the backend's memory
 *  belongs to a context it holds and it overloads this. */
template <typename value_type_, typename backend_type_>
typename backend_type_::template allocator<value_type_> allocator_of(backend_type_ const &) noexcept {
    return {};
}

/** A vector of @p value_type_ in the memory the kernels of @p backend_type_ read and write. */
template <typename value_type_, typename backend_type_>
using backend_vector = nk::vector<value_type_, typename backend_type_::template allocator<value_type_>>;

/** A zeroed vector of @p count values in @p backend memory, aborting on exhaustion. */
template <typename value_type_, typename backend_type_>
[[nodiscard]] backend_vector<value_type_, backend_type_> make_vector(backend_type_ const &backend, std::size_t count) {
    auto [vector, status] = backend_vector<value_type_, backend_type_>::zeros(count,
                                                                              allocator_of<value_type_>(backend));
    if (nk::failed(status)) std::abort();
    return std::move(vector);
}

#pragma endregion Backends

/** The cases of each test module, which register them through @c run_kernel_case . */
void run_cast_cases();
void run_reduce_cases();
void run_dot_cases();
void run_spatial_cases();
void run_set_cases();
void run_curved_cases();
void run_probability_cases();
void run_each_cases();
void run_trigonometry_cases();
void run_geospatial_cases();
void run_shapes_cases();
void run_sparse_cases();
void run_vector_cases();
void run_tensor_cases();
void run_maxsim_cases();
void run_each_cuda_cases();
void run_each_rocm_cases();
void run_each_metal_cases();
void run_cast_cuda_cases();
void run_cast_rocm_cases();
void run_cast_metal_cases();
void run_reduce_cuda_cases();
void run_shapes_cuda_cases();
void run_shapes_rocm_cases();
void run_shapes_metal_cases();
void run_tensor_cuda_cases();

/** The cross/batch cases, one ISA-family file each. */
void run_cross_serial_cases();
void run_cross_runtime_cases();
void run_cross_x8664_cases();
void run_cross_x8664amx_cases();
void run_cross_arm64_cases();
void run_cross_blas_cases();
void run_cross_riscv64_cases();
void run_cross_ppc64_cases();
void run_cross_loongarch64_cases();
void run_cross_wasm_cases();
void run_cross_cuda_cases();
void run_cross_rocm_cases();
void run_cross_metal_cases();

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_HARNESS_HPP
