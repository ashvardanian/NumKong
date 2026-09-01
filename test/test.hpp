/**
 *  @brief C++ test suite with precision analysis using double-double arithmetic.
 *  @file test/test.hpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  This test suite compares NumKong operations against high-precision references,
 *  like our `f118_t` double-double type, and reports ULP, absolute, and relative error statistics.
 *
 *  Environment Variables:
 *    NK_FILTER=<pattern>           - Filter tests by name RegEx (default: run all)
 *    NK_SEED=N                     - RNG seed (default: 42)
 *
 *    NK_DENSE_DIMENSIONS=N         - Vector dimension for dot/spatial tests (default: 1536)
 *    NK_CURVED_DIMENSIONS=N        - Vector dimension for curved tests (default: 64)
 *    NK_SPARSE_DIMENSIONS=N        - Vector dimension for sparse tests (default: 256)
 *    NK_MESH_POINTS=N              - Point count for mesh tests (default: 1000)
 *    NK_MATRIX_HEIGHT=N            - GEMM M dimension (default: 1024)
 *    NK_MATRIX_WIDTH=N             - GEMM N dimension (default: 128)
 *    NK_MATRIX_DEPTH=N             - GEMM K dimension (default: 1536)
 *
 *    NK_IN_QEMU                    - Set when running under QEMU (relaxes accuracy thresholds)
 *    NK_TEST_ASSERT=1              - Assert on failed accuracy checks (default: 0)
 *    NK_TEST_VERBOSE=1             - Show per-dimension ULP breakdown (default: 0)
 *    NK_ULP_THRESHOLD_F32=N        - Max allowed ULP for f32 (default: 4)
 *    NK_ULP_THRESHOLD_F16=N        - Max allowed ULP for f16 (default: 32)
 *    NK_ULP_THRESHOLD_BF16=N       - Max allowed ULP for bf16 (default: 256)
 *    NK_BUDGET_SECS=<seconds>      - Time budget per kernel in seconds (default: 1)
 *    NK_RANDOM_DISTRIBUTION=<type> - Random distribution: uniform_k|lognormal_k|cauchy_k (default: lognormal_k)
 */

#pragma once
#ifndef NK_TEST_HPP
#define NK_TEST_HPP

#include <cmath>   // `std::fabs`, `std::isnan`, `std::isinf`
#include <cstdint> // `std::uint64_t`, `std::int32_t`, `std::int64_t`
#include <cstdio>  // `std::printf`, `std::fflush`
#include <cstdlib> // `std::abort`
#include <cstring> // `std::memcpy`, `std::strstr`

#include <algorithm> // `std::min`, `std::max`
#include <array>     // `std::array`
#include <cassert>   // `assert`
#include <chrono>    // `std::chrono::steady_clock`, `std::chrono::duration_cast`
#include <complex>   // `std::complex`
#include <limits>    // `std::numeric_limits`
#include <new>       // `std::bad_alloc`
#include <optional>  // `std::optional`

#if NK_TEST_USE_OPENMP
#include <omp.h>
#endif

#ifndef NK_ALLOW_ISA_REDIRECT
#define NK_ALLOW_ISA_REDIRECT 0
#endif

// Optional BLAS/MKL integration for precision comparison
#ifndef NK_COMPARE_TO_BLAS
#define NK_COMPARE_TO_BLAS 0
#endif
#ifndef NK_COMPARE_TO_MKL
#define NK_COMPARE_TO_MKL 0
#endif
#ifndef NK_COMPARE_TO_ACCELERATE
#define NK_COMPARE_TO_ACCELERATE 0
#endif

// Include reference library headers - MKL, Accelerate, or generic CBLAS
#if NK_COMPARE_TO_MKL
#include <mkl.h> // MKL includes its own CBLAS interface
#elif NK_COMPARE_TO_ACCELERATE
#include <Accelerate/Accelerate.h> // Apple Accelerate framework
#elif NK_COMPARE_TO_BLAS
#include <cblas.h> // Generic CBLAS (OpenBLAS, etc.)
#endif

// Intests we want to make sure our custom floating-point routines are used instead of
// compiler-provided native types.
#undef NK_NATIVE_F16
#define NK_NATIVE_F16 0
#undef NK_NATIVE_BF16
#define NK_NATIVE_BF16 0

#include "numkong/capabilities.h" // `nk_capabilities`, `nk_capability_t`
#include "numkong/types.hpp"
#include "numkong/tensor.hpp"
#include "numkong/dots.hpp"
#include "numkong/maxsim.hpp"
#include "numkong/matrix.hpp"
#include "numkong/reduce.hpp"
#include "numkong/each.hpp"
#include "numkong/trigonometry.hpp"
#include "numkong/spatials.hpp"
#include "numkong/random.hpp" // `nk::fill_uniform`

namespace nk = ashvardanian::numkong;

using nk::bf16_t;
using nk::bf16c_t;
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
using nk::u16_t;
using nk::u1x8_t;
using nk::u32_t;
using nk::u4x2_t;
using nk::u64_t;
using nk::u8_t;

using steady_clock = std::chrono::steady_clock;
using time_point = steady_clock::time_point;

/**
 *  @brief Maps scalar types to appropriate reference types for ULP testing.
 *
 *  Two template parameters: input type and result type (defaulting to input).
 *  - f32/f64 input → f118_t (need full double-double precision)
 *  - Complex f32c/f64c input → f118c_t
 *  - Smaller complex (f16c, bf16c) → f64c_t (52-bit mantissa >> 7-10 bit mantissa)
 *  - Everything else (integers, bf16, f16, etc.) → f64_t
 */
template <typename input_type_, typename result_type_ = input_type_>
using reference_for = std::conditional_t<
    std::is_same_v<input_type_, f32_t> || std::is_same_v<input_type_, f64_t>, f118_t,
    std::conditional_t<
        nk::is_complex_dtype<input_type_>(),
        std::conditional_t<std::is_same_v<input_type_, f32c_t> || std::is_same_v<input_type_, f64c_t>, f118c_t, f64c_t>,
        f64_t>>;

template class nk::vector<int>;
template class nk::vector<nk::i32_t>;
template class nk::vector<nk::u1x8_t>;
template class nk::vector<nk::i4x2_t>;
template class nk::vector<nk::f64c_t>;
template class nk::vector<std::complex<float>>;

enum class random_distribution_kind_t { uniform_k, lognormal_k, cauchy_k };
enum class comparison_family_t {
    exact_k,
    approximate_k,
    probability_k,
    geospatial_k,
};
enum class comparison_failure_mode_t { exact_distance_k, ulp_threshold_k };

struct comparison_family_spec_t {
    comparison_failure_mode_t failure_mode;
    std::array<char const *, 5> column_labels;
};

inline constexpr comparison_family_spec_t comparison_family_spec(comparison_family_t family) noexcept {
    switch (family) {
    case comparison_family_t::exact_k:
        return {comparison_failure_mode_t::exact_distance_k, {"max_dist", "mean_dist", "max_abs", "mismatch", "exact"}};
    case comparison_family_t::probability_k:
        return {comparison_failure_mode_t::ulp_threshold_k, {"max_abs", "mean_abs", "max_rel", "mean_rel", "mean_ulp"}};
    case comparison_family_t::geospatial_k:
        return {comparison_failure_mode_t::ulp_threshold_k, {"max_abs", "mean_abs", "max_rel", "mean_ulp", "max_ulp"}};
    case comparison_family_t::approximate_k:
        return {comparison_failure_mode_t::ulp_threshold_k, {"max_abs", "max_rel", "mean_ulp", "max_ulp", "exact"}};
    }
    return {comparison_failure_mode_t::ulp_threshold_k, {"max_abs", "max_rel", "mean_ulp", "max_ulp", "exact"}};
}

struct test_config_t {
    /** Assert on failed accuracy checks. Override: `NK_TEST_ASSERT=1`. */
    bool assert_on_failure = false;
    /** Show per-dimension ULP breakdown. Override: `NK_TEST_VERBOSE=1`. */
    bool verbose = false;
    /** Relaxed accuracy for emulated SIMD. Override: `NK_IN_QEMU`. */
    bool running_in_qemu = false;
    /** Max allowed ULP for f32. Override: `NK_ULP_THRESHOLD_F32`. */
    std::uint64_t ulp_threshold_f32 = 4;
    /** Max allowed ULP for f16. Override: `NK_ULP_THRESHOLD_F16`. */
    std::uint64_t ulp_threshold_f16 = 32;
    /** Max allowed ULP for bf16. Override: `NK_ULP_THRESHOLD_BF16`. */
    std::uint64_t ulp_threshold_bf16 = 256;
    /** Time budget per kernel in milliseconds. Override: `NK_BUDGET_SECS`. */
    std::size_t time_budget_ms = 1000;
    /** Random seed for reproducible tests. Override: `NK_SEED`. */
    std::uint32_t seed = 42;
    /** Filter tests by name (regex or substring). Override: `NK_FILTER`. */
    char const *filter = nullptr;
    /** Random distribution for test inputs. Override: `NK_RANDOM_DISTRIBUTION`. */
    random_distribution_kind_t distribution = random_distribution_kind_t::lognormal_k;

    /** For dot products, spatial metrics. Override: `NK_DENSE_DIMENSIONS`. */
    std::size_t dense_dimensions = 1536;
    /** For curved metrics (quadratic in dims). Override: `NK_CURVED_DIMENSIONS`. */
    std::size_t curved_dimensions = 64;
    /** For sparse set intersection and sparse dot. Override: `NK_SPARSE_DIMENSIONS`. */
    std::size_t sparse_dimensions = 256;
    /** Number of 3D points for RMSD, Kabsch. Override: `NK_MESH_POINTS`. */
    std::size_t mesh_points = 1000;
    /** GEMM M dimension. Override: `NK_MATRIX_HEIGHT`. */
    std::size_t matrix_height = 1024;
    /** GEMM N dimension. Override: `NK_MATRIX_WIDTH`. */
    std::size_t matrix_width = 128;
    /** GEMM K dimension. Override: `NK_MATRIX_DEPTH`. */
    std::size_t matrix_depth = 1536;
    /** Max angular separation in degrees for geospatial tests. Override: `NK_MAX_COORD_ANGLE`. */
    float max_coord_angle = 180.0f;
    /** Count of kernels that failed the configured accuracy checks. */
    std::size_t failure_count = 0;

    bool should_run(char const *test_name) const;

    std::uint64_t ulp_threshold_for(char const *kernel_name) const noexcept {
        if (std::strstr(kernel_name, "_bf16")) return ulp_threshold_bf16;
        if (std::strstr(kernel_name, "_f16")) return ulp_threshold_f16;
        return ulp_threshold_f32;
    }

    char const *distribution_name() const noexcept {
        switch (distribution) {
        case random_distribution_kind_t::uniform_k: return "uniform";
        case random_distribution_kind_t::lognormal_k: return "lognormal";
        case random_distribution_kind_t::cauchy_k: return "cauchy";
        default: return "unknown";
        }
    }
};

extern test_config_t global_config;
void print_stats_header(comparison_family_t family) noexcept;
struct error_stats_t;
bool should_fail(char const *kernel_name, error_stats_t const &stats) noexcept;
void print_stats_row(char const *kernel_name, error_stats_t const &stats) noexcept;

/**
 *  @brief Tracks the currently-running kernel name for SIGILL diagnostics.
 *  Set before each kernel call, cleared after. A signal handler installed in
 *  main() reads this to log the culprit before the process exits.
 */
inline char const *volatile nk_test_current_kernel_ = nullptr;

/** @brief `#if NK_TARGET_X` says the compiler could build a kernel; this says the CPU can run it. */
inline bool isa_available(nk_capability_t cap) noexcept {
    static nk_capability_t const caps = nk_capabilities();
    return (caps & cap) != 0;
}

struct error_stats_section_t {
    char const *title = nullptr;
    nk_capability_t required = nk_cap_serial_k;
    std::optional<comparison_family_t> last_family = std::nullopt;
    bool emitted_any = false;

    error_stats_section_t() noexcept = default;

    /** @brief Restart under a new heading, for kernels needing @p cap. */
    void section(char const *heading, nk_capability_t cap) noexcept {
        title = heading;
        required = cap;
        emitted_any = false;
        last_family.reset();
    }

    template <typename test_function_type_, typename... args_types_>
    void operator()(char const *kernel_name, test_function_type_ test_fn, args_types_ &&...args) {
        if (!isa_available(required) || !global_config.should_run(kernel_name)) return;

        nk_test_current_kernel_ = kernel_name;
        auto stats = test_fn(std::forward<args_types_>(args)...);
        nk_test_current_kernel_ = nullptr;

        if (!emitted_any) {
            if (title) {
                std::puts("");
                std::printf("%s:\n", title);
            }
            emitted_any = true;
        }
        if (last_family != stats.family) {
            print_stats_header(stats.family);
            last_family = stats.family;
        }
        print_stats_row(kernel_name, stats);
        if (global_config.assert_on_failure && should_fail(kernel_name, stats)) ++global_config.failure_count;
    }
};

inline time_point test_start_time() { return steady_clock::now(); }

inline bool within_time_budget(time_point start) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start).count();
    return elapsed < static_cast<long long>(global_config.time_budget_ms);
}

/**
 *  @brief Compute ULP (Units in Last Place) distance between two floating-point values.
 *
 *  ULP distance is the number of representable floating-point numbers between a and b.
 *  This is the gold standard for comparing floating-point implementations.
 *
 *  Uses the XOR transformation from Bruce Dawson's algorithm to handle all sign combinations:
 *  @see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 *  @see https://en.wikipedia.org/wiki/Unit_in_the_last_place
 */
template <typename scalar_type_>
std::uint64_t ulp_distance(scalar_type_ a, scalar_type_ b) noexcept {
    // Handle special cases - skip float checks for integer types
    if constexpr (!nk::is_integral_dtype<scalar_type_>()) {
        if (std::isnan(static_cast<double>(a)) || std::isnan(static_cast<double>(b)))
            return std::numeric_limits<std::uint64_t>::max();
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

/**
 *  @brief Accumulator for error statistics across multiple test trials.
 */
struct error_stats_t {
    comparison_family_t family = comparison_family_t::approximate_k;

    nk_f64_t min_abs_err = std::numeric_limits<nk_f64_t>::max();
    nk_f64_t max_abs_err = 0;
    nk_f64_t sum_abs_err = 0;

    nk_f64_t min_rel_err = std::numeric_limits<nk_f64_t>::max();
    nk_f64_t max_rel_err = 0;
    nk_f64_t sum_rel_err = 0;

    std::uint64_t min_ulp = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_ulp = 0;
    f118_t sum_ulp = f118_t();

    std::size_t count = 0;
    std::size_t exact_matches = 0;
    bool saw_floating_distance = false;
    char const *first_failure = nullptr;

    explicit error_stats_t(comparison_family_t family = comparison_family_t::approximate_k) noexcept : family(family) {}

    /** @brief Record a boolean property; @p property names it in the report when it does not hold. */
    void expect(bool held, char const *property) noexcept {
        if (!held && !first_failure) first_failure = property;
        accumulate(static_cast<int>(held), 1);
    }

    template <typename actual_type_, typename expected_type_>
    void accumulate(actual_type_ actual, expected_type_ expected) noexcept {
        if constexpr (nk::is_complex_dtype<actual_type_>())
            accumulate_scalar(actual.real(), expected.real()), accumulate_scalar(actual.imag(), expected.imag());
        else accumulate_scalar(actual, expected);
    }

    template <typename actual_type_, typename expected_type_>
    void accumulate_scalar(actual_type_ actual, expected_type_ expected) noexcept {
        actual_type_ expected_as_actual;
        if constexpr (std::is_same_v<expected_type_, f118_t> || std::is_same_v<expected_type_, f118c_t>)
            expected_as_actual = expected.template to<actual_type_>();
        else if constexpr (nk::is_integral_dtype<expected_type_>() && nk::is_integral_dtype<actual_type_>())
            expected_as_actual = actual_type_(expected);
        else expected_as_actual = actual_type_(static_cast<double>(expected));

        bool const use_integer_distance = family == comparison_family_t::exact_k &&
                                          nk::is_integral_dtype<actual_type_>();
        std::uint64_t ulps = use_integer_distance ? integer_distance(actual, expected_as_actual)
                                                  : ulp_distance(actual, expected_as_actual);

        // Skip NaN/Inf pairs — sentinel value means the comparison is meaningless
        if (ulps == std::numeric_limits<std::uint64_t>::max()) return;

        if constexpr (!nk::is_integral_dtype<actual_type_>()) saw_floating_distance = true;

        if constexpr (!nk::is_integral_dtype<actual_type_>() || std::is_integral_v<actual_type_>) {
            nk_f64_t exp_f64 = static_cast<nk_f64_t>(expected_as_actual);
            nk_f64_t act_f64 = static_cast<nk_f64_t>(actual);

            nk_f64_t abs_err = std::fabs(exp_f64 - act_f64);
            nk_f64_t rel_err = exp_f64 != 0 ? abs_err / std::fabs(exp_f64) : abs_err;

            min_abs_err = std::min(min_abs_err, abs_err);
            max_abs_err = std::max(max_abs_err, abs_err);
            sum_abs_err += abs_err;
            min_rel_err = std::min(min_rel_err, rel_err);
            max_rel_err = std::max(max_rel_err, rel_err);
            sum_rel_err += rel_err;
        }

        // Always update ULP metrics (works for both integer and float)
        min_ulp = std::min(min_ulp, ulps);
        max_ulp = std::max(max_ulp, ulps);
        sum_ulp += f118_t(ulps);

        count++;
        if (ulps == 0) exact_matches++;
    }

    nk_f64_t mean_abs_err() const noexcept { return count > 0 ? sum_abs_err / count : 0; }
    nk_f64_t mean_rel_err() const noexcept { return count > 0 ? sum_rel_err / count : 0; }
    nk_f64_t mean_ulp() const noexcept {
        // On 32-bit WASM we need this ugly casting sequence to avoid adding more `f118_t` constructors
        return count > 0 ? static_cast<double>(sum_ulp / f118_t(static_cast<double>(count))) : 0;
    }
    std::size_t mismatches() const noexcept { return count - exact_matches; }

    void reset() noexcept {
        comparison_family_t const current_family = family;
        *this = error_stats_t {current_family};
    }

    void merge(error_stats_t const &other) noexcept {
        if (other.count == 0) return;
        if (count == 0) family = other.family;
        else assert(family == other.family && "Can't merge stats from different comparison families");
        min_abs_err = std::min(min_abs_err, other.min_abs_err);
        max_abs_err = std::max(max_abs_err, other.max_abs_err);
        sum_abs_err += other.sum_abs_err;
        min_rel_err = std::min(min_rel_err, other.min_rel_err);
        max_rel_err = std::max(max_rel_err, other.max_rel_err);
        sum_rel_err += other.sum_rel_err;
        min_ulp = std::min(min_ulp, other.min_ulp);
        max_ulp = std::max(max_ulp, other.max_ulp);
        sum_ulp += other.sum_ulp;
        count += other.count;
        exact_matches += other.exact_matches;
        saw_floating_distance = saw_floating_distance || other.saw_floating_distance;
    }
};

inline bool should_fail(char const *kernel_name, error_stats_t const &stats) noexcept {
    comparison_family_spec_t const spec = comparison_family_spec(stats.family);
    switch (spec.failure_mode) {
    case comparison_failure_mode_t::exact_distance_k:
        if (!stats.saw_floating_distance) return stats.max_ulp > 0;
        return stats.max_ulp > global_config.ulp_threshold_for(kernel_name);
    case comparison_failure_mode_t::ulp_threshold_k:
        return stats.max_ulp > global_config.ulp_threshold_for(kernel_name);
    }
    return false;
}

inline void print_stats_row(char const *kernel_name, error_stats_t const &stats) noexcept {
    switch (stats.family) {
    case comparison_family_t::exact_k:
        std::printf("%-40s %12llu %10.1f %12.2e %12zu %10zu\n", kernel_name,
                    static_cast<unsigned long long>(stats.max_ulp), stats.mean_ulp(), stats.max_abs_err,
                    stats.mismatches(), stats.exact_matches);
        break;
    case comparison_family_t::probability_k:
        std::printf("%-40s %12.2e %10.2e %12.2e %12.2e %10.2e\n", kernel_name, stats.max_abs_err, stats.mean_abs_err(),
                    stats.max_rel_err, stats.mean_rel_err(), stats.mean_ulp());
        break;
    case comparison_family_t::geospatial_k:
        std::printf("%-40s %12.2e %10.2e %12.2e %12.1f %10llu\n", kernel_name, stats.max_abs_err, stats.mean_abs_err(),
                    stats.max_rel_err, stats.mean_ulp(), static_cast<unsigned long long>(stats.max_ulp));
        break;
    case comparison_family_t::approximate_k:
        std::printf("%-40s %12.2e %10.2e %12.2e %12llu %10zu\n", kernel_name, stats.max_abs_err, stats.max_rel_err,
                    stats.mean_ulp(), static_cast<unsigned long long>(stats.max_ulp), stats.exact_matches);
        break;
    }
    // The counters say how many properties failed; this says which one.
    if (stats.first_failure && stats.mismatches()) std::printf("    first failure: %s\n", stats.first_failure);
    std::fflush(stdout);
}

/**
 *  @brief Factory function to allocate vectors, potentially raising bad-allocs.
 */
template <typename type_>
[[nodiscard]] nk::vector<type_> make_vector(std::size_t n) {
    auto result = nk::vector<type_>::try_zeros(n);
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
    if (result.empty() && n > 0) throw std::bad_alloc();
#else
    if (result.empty() && n > 0) std::abort();
#endif
    return result;
}

/**
 *  @brief Fill buffer with random values, respecting global distribution setting.
 *
 *  Dispatches to appropriate nk::fill_* library function based on `global_config.distribution`.
 *  Infers sensible bounds from type's representable range.
 */
template <typename scalar_type_, typename generator_type_>
void fill_random(generator_type_ &generator, nk::vector<scalar_type_> &vector) {
    switch (global_config.distribution) {
    case random_distribution_kind_t::uniform_k:
        nk::fill_uniform(generator, vector.values_data(), vector.size_values());
        break;
    case random_distribution_kind_t::lognormal_k:
        nk::fill_lognormal(generator, vector.values_data(), vector.size_values());
        break;
    case random_distribution_kind_t::cauchy_k:
        nk::fill_cauchy(generator, vector.values_data(), vector.size_values());
        break;
    }
}

// Forward declarations for test modules
void test_casts();
void test_reduce();
void test_dot();
void test_spatial();
void test_set();
void test_curved();
void test_probability();
void test_each();
void test_trigonometry();
void test_geospatial();
void test_mesh();
void test_sparse();
void test_vector_types();
void test_tensor_ops();
void test_maxsim();

// Forward declarations for cross/batch tests (ISA-family files)
void test_cross_serial();
void test_cross_x86();
void test_cross_amx();
void test_cross_arm();
void test_cross_sme();
void test_cross_blas();
void test_cross_rvv();
void test_cross_power();
void test_cross_loongarch();
void test_cross_wasm();

#endif // NK_TEST_HPP
