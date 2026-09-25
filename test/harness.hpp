/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief C++ test suite with precision analysis using double-double arithmetic.
 *
 *  This test suite compares NumKong operations against high-precision references, like our
 *  @c f118_t double-double type, and reports ULP, absolute, and relative error statistics.
 *
 *  Environment Variables:
 *
 *  @verbatim
 *  NUMKONG_FILTER=<pattern>           - Filter tests by name RegEx (default: run all)
 *  NUMKONG_SEED=N                     - RNG seed, or random to draw one (default: 42)
 *
 *  NUMKONG_DENSE_DIMENSIONS=N[,...]   - Vector dimension for dot/spatial tests (default: 1536)
 *  NUMKONG_CURVED_DIMENSIONS=N[,...]  - Vector dimension for curved tests (default: 64)
 *  NUMKONG_SPARSE_DIMENSIONS=N[,...]  - Vector dimension for sparse tests (default: 256)
 *  NUMKONG_MESH_POINTS=N              - Point count for mesh tests (default: 1000)
 *  NUMKONG_MATRIX_HEIGHT=N[,...]      - GEMM M dimension (default: 1024)
 *  NUMKONG_MATRIX_WIDTH=N[,...]       - GEMM N dimension (default: 128)
 *  NUMKONG_MATRIX_DEPTH=N[,...]       - GEMM K dimension (default: 1536)
 *
 *  NUMKONG_IN_QEMU=1                  - Shrink shapes for emulated runs; unset, 0 or false keep them
 *  NUMKONG_ASSERT=1                   - Exit 1 when any kernel fails its accuracy check (default: 0)
 *  NUMKONG_VERBOSE=1                  - Show per-dimension ULP breakdown (default: 0)
 *  NUMKONG_ULP_THRESHOLD_F32=N        - Max allowed ULP for f32 (default: 4)
 *  NUMKONG_ULP_THRESHOLD_F16=N        - Max allowed ULP for f16 (default: 32)
 *  NUMKONG_ULP_THRESHOLD_BF16=N       - Max allowed ULP for bf16 (default: 256)
 *  NUMKONG_BUDGET_SECS=<seconds>      - Time budget per kernel in seconds (default: 1)
 *  NUMKONG_RANDOM_DISTRIBUTION=<type> - uniform_k, lognormal_k or cauchy_k (default: lognormal_k)
 *  @endverbatim
 */

#pragma once
#ifndef NUMKONG_TEST_HARNESS_HPP
#define NUMKONG_TEST_HARNESS_HPP

#include <cmath>   // `std::fabs`, `std::isnan`, `std::isinf`
#include <cstdint> // `std::uint64_t`, `std::int32_t`, `std::int64_t`
#include <cstdio>  // `std::fflush`, `stdout`, `stderr`
#include <cstdlib> // `std::abort`, `std::getenv`, `std::strtod`
#include <cstring> // `std::memcpy`, `std::strstr`, `std::strcmp`, `std::strcspn`

#include <algorithm>    // `std::min`, `std::max`
#include <array>        // `std::array`
#include <cassert>      // `assert`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`, `std::chrono::duration`
#include <complex>      // `std::complex`
#include <limits>       // `std::numeric_limits`
#include <new>          // `std::bad_alloc`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`
#include <system_error> // `std::errc`
#include <type_traits>  // `std::is_same_v`

#include <fmt/base.h> // `fmt::print`, `fmt::println`

#if __has_include(<regex.h>)
#include <regex.h>
#define NUMKONG_HAS_POSIX_REGEX_ 1
#else
#include <regex>
#define NUMKONG_HAS_POSIX_REGEX_ 0
#endif

#ifndef NUMKONG_ALLOW_ISA_REDIRECT
#define NUMKONG_ALLOW_ISA_REDIRECT 0
#endif

/** Optional BLAS/MKL integration for precision comparison */
#ifndef NUMKONG_COMPARE_TO_BLAS
#define NUMKONG_COMPARE_TO_BLAS 0
#endif
#ifndef NUMKONG_COMPARE_TO_MKL
#define NUMKONG_COMPARE_TO_MKL 0
#endif
#ifndef NUMKONG_COMPARE_TO_ACCELERATE
#define NUMKONG_COMPARE_TO_ACCELERATE 0
#endif

/* Include reference library headers - MKL, Accelerate, or generic CBLAS */
#if NUMKONG_COMPARE_TO_MKL
#include <mkl.h> // MKL includes its own CBLAS interface
#elif NUMKONG_COMPARE_TO_ACCELERATE
#include <Accelerate/Accelerate.h> // Apple Accelerate framework
#elif NUMKONG_COMPARE_TO_BLAS
#include <cblas.h> // Generic CBLAS (OpenBLAS, etc.)
#endif

/* In tests we want to make sure our custom floating-point routines are used instead of
 * compiler-provided native types. */
#undef NUMKONG_NATIVE_F16
#define NUMKONG_NATIVE_F16 0
#undef NUMKONG_NATIVE_BF16
#define NUMKONG_NATIVE_BF16 0

#include "numkong/capabilities.h" // `nk_capabilities_detected`, `nk_capability_t`
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
#include "numkong/vector.hpp" // `nk::aligned_allocator`

namespace nk = ashvardanian::numkong;

template class nk::vector<int>;
template class nk::vector<nk::i32_t>;
template class nk::vector<nk::u1x8_t>;
template class nk::vector<nk::i4x2_t>;
template class nk::vector<nk::f64c_t>;
template class nk::vector<std::complex<float>>;

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
using nk::u16_t;
using nk::u1x8_t;
using nk::u32_t;
using nk::u4x2_t;
using nk::u64_t;
using nk::u8_t;
using nk::ue4m3_t;
using nk::ue8m0_t;

using nk::mxfp4_t;
using nk::mxfp6_e2m3_t;
using nk::mxfp6_e3m2_t;
using nk::mxfp8_e4m3_t;
using nk::mxfp8_e5m2_t;
using nk::mxint8_t;
using nk::nvfp4_t;

using steady_clock = std::chrono::steady_clock;
using time_point = steady_clock::time_point;

/** Reads @p name from the environment as @p value_type_, or @p fallback when it is unset or empty.
 *  Aborts, naming the variable, when its text does not parse: a typo never passes as a default. */
template <typename value_type_>
[[nodiscard]] value_type_ env_variable(char const *name, value_type_ fallback) noexcept {
    char const *const text = std::getenv(name);
    if (!text || !*text) return fallback;
    if constexpr (std::is_same_v<value_type_, char const *>) return text;
    else if constexpr (std::is_same_v<value_type_, bool>) return std::strcmp(text, "0") && std::strcmp(text, "false");
    else {
        value_type_ value {};
        char *stop = nullptr;
        if constexpr (std::is_floating_point_v<value_type_>) value = static_cast<value_type_>(std::strtod(text, &stop));
        else {
            auto const [end, error] = std::from_chars(text, text + std::strlen(text), value);
            stop = error == std::errc {} ? const_cast<char *>(end) : const_cast<char *>(text);
        }
        if (stop != text && *stop == '\0') return value;
        fmt::println(stderr, "{}=\"{}\" does not parse", name, text);
        std::abort();
    }
}

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

enum class random_distribution_kind_t { uniform_k, lognormal_k, cauchy_k };
enum class comparison_family_t {
    exact_k,
    approximate_k,
    normalized_reduction_k,
    probability_k,
    geospatial_k,
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
    case comparison_family_t::probability_k:
        return {comparison_failure_mode_t::ulp_threshold_k, {"max_abs", "mean_abs", "max_rel", "mean_rel", "mean_ulp"}};
    case comparison_family_t::geospatial_k:
        return {comparison_failure_mode_t::ulp_threshold_k, {"max_abs", "mean_abs", "max_rel", "mean_ulp", "max_ulp"}};
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

struct test_config_t {

    /** Exit 1 when any kernel fails its accuracy check. Override: `NUMKONG_ASSERT=1`. */
    bool assert_on_failure = false;

    /** Show per-dimension ULP breakdown. Override: `NUMKONG_VERBOSE=1`. */
    bool verbose = false;

    /** Shrinks shapes for emulated SIMD. Override: @c NUMKONG_IN_QEMU. */
    bool running_in_qemu = false;

    /** Max allowed ULP for f32. Override: @c NUMKONG_ULP_THRESHOLD_F32. */
    std::uint64_t ulp_threshold_f32 = 4;

    /** Max allowed ULP for f16. Override: @c NUMKONG_ULP_THRESHOLD_F16. */
    std::uint64_t ulp_threshold_f16 = 32;

    /** Max allowed ULP for bf16. Override: @c NUMKONG_ULP_THRESHOLD_BF16. */
    std::uint64_t ulp_threshold_bf16 = 256;

    /** Max absolute error as a fraction of the largest reference magnitude, for the
     *  normalized-reduction family. Override: @c NUMKONG_SCALE_THRESHOLD. */
    nk_f64_t scale_threshold = 0.02;

    /** Time budget per kernel in seconds. Override: @c NUMKONG_BUDGET_SECS. */
    double budget_seconds = 1;

    /** The binary's @c argv[0], which closes every rerun line. */
    char const *program = "";

    /** Random seed for reproducible tests. Override: @c NUMKONG_SEED, where random draws one. */
    std::uint32_t seed = 42;

    /** Filter tests by name (regex or substring), set through @c set_filter. Override: @c NUMKONG_FILTER. */
    char const *filter = nullptr;
#if NUMKONG_HAS_POSIX_REGEX_
    regex_t filter_regex {};
    bool filter_compiled = false;
#else
    std::optional<std::regex> filter_regex;
#endif

    /** Random distribution for test inputs. Override: @c NUMKONG_RANDOM_DISTRIBUTION. */
    random_distribution_kind_t distribution = random_distribution_kind_t::lognormal_k;

    /** For dot products, spatial metrics. Override: @c NUMKONG_DENSE_DIMENSIONS. */
    std::size_t dense_dimensions = 1536;

    /** For curved metrics, quadratic in dimensions. Override: @c NUMKONG_CURVED_DIMENSIONS. */
    std::size_t curved_dimensions = 64;

    /** For sparse set intersection and sparse dot. Override: @c NUMKONG_SPARSE_DIMENSIONS. */
    std::size_t sparse_dimensions = 256;

    /** Number of 3D points for RMSD, Kabsch. Override: @c NUMKONG_MESH_POINTS. */
    std::size_t mesh_points = 1000;

    /** GEMM M dimension. Override: @c NUMKONG_MATRIX_HEIGHT. */
    std::size_t matrix_height = 1024;

    /** GEMM N dimension. Override: @c NUMKONG_MATRIX_WIDTH. */
    std::size_t matrix_width = 128;

    /** GEMM K dimension. Override: @c NUMKONG_MATRIX_DEPTH. */
    std::size_t matrix_depth = 1536;

    /** Max geospatial angular separation, in degrees. Override: @c NUMKONG_MAX_COORD_ANGLE. */
    float max_coord_angle = 180.0f;

    /** Count of kernels that ran their accuracy checks. */
    std::size_t kernel_count = 0;

    /** Count of kernels that failed the configured accuracy checks. */
    std::size_t failure_count = 0;

    /** Compiles @p pattern as an extended regex; an invalid one matches as a substring instead. */
    void set_filter(char const *pattern) noexcept {
        filter = pattern;
#if NUMKONG_HAS_POSIX_REGEX_
        if (filter_compiled) regfree(&filter_regex);
        filter_compiled = pattern && regcomp(&filter_regex, pattern, REG_EXTENDED | REG_NOSUB) == 0;
#else
        filter_regex.reset();
        try {
            if (pattern) filter_regex.emplace(pattern);
        }
        catch (std::regex_error const &) {
        }
#endif
    }

    bool should_run(char const *test_name) const noexcept {
        if (!filter) return true;
#if NUMKONG_HAS_POSIX_REGEX_
        if (filter_compiled) return regexec(&filter_regex, test_name, 0, nullptr, 0) == 0;
#else
        if (filter_regex) return std::regex_search(test_name, *filter_regex);
#endif
        return std::strstr(test_name, filter) != nullptr;
    }

    /** Applies the `NUMKONG_*` environment overrides, which the command line, parsed afterwards,
     *  overrides in turn. */
    void load_environment() noexcept {
        auto const positive = [](char const *name, auto fallback) noexcept {
            auto const value = env_variable(name, fallback);
            if (value > 0) return value;
            fmt::println(stderr, "{} must be positive", name);
            std::abort();
        };
        // Python and JS take a comma list of dimensions, of which C++ runs the first
        auto const dimension = [](char const *name, std::size_t fallback) noexcept {
            char const *const text = env_variable<char const *>(name, nullptr);
            if (!text) return fallback;
            char const *const first_end = text + std::strcspn(text, ",");
            std::size_t value = 0;
            auto const [end, error] = std::from_chars(text, first_end, value);
            if (error == std::errc {} && end == first_end && value > 0) return value;
            fmt::println(stderr, "{}=\"{}\" does not start with a positive count", name, text);
            std::abort();
        };
        running_in_qemu = env_variable("NUMKONG_IN_QEMU", running_in_qemu);
        assert_on_failure = env_variable("NUMKONG_ASSERT", assert_on_failure);
        verbose = env_variable("NUMKONG_VERBOSE", verbose);
        ulp_threshold_f32 = env_variable("NUMKONG_ULP_THRESHOLD_F32", ulp_threshold_f32);
        ulp_threshold_f16 = env_variable("NUMKONG_ULP_THRESHOLD_F16", ulp_threshold_f16);
        ulp_threshold_bf16 = env_variable("NUMKONG_ULP_THRESHOLD_BF16", ulp_threshold_bf16);
        scale_threshold = env_variable("NUMKONG_SCALE_THRESHOLD", scale_threshold);
        bool const random_seed = std::strcmp(env_variable("NUMKONG_SEED", ""), "random") == 0;
        seed = random_seed ? std::random_device {}() : env_variable("NUMKONG_SEED", seed);
        set_filter(env_variable("NUMKONG_FILTER", filter)); // e.g., "dot", "angular", "kld"
        // A zero or negative budget keeps the default, rather than running no iterations at all
        if (double const budget = env_variable("NUMKONG_BUDGET_SECS", 0.0); budget > 0) budget_seconds = budget;

        if (char const *const text = env_variable<char const *>("NUMKONG_RANDOM_DISTRIBUTION", nullptr)) {
            if (std::strcmp(text, "uniform_k") == 0) distribution = random_distribution_kind_t::uniform_k;
            else if (std::strcmp(text, "cauchy_k") == 0) distribution = random_distribution_kind_t::cauchy_k;
            else if (std::strcmp(text, "lognormal_k") == 0) distribution = random_distribution_kind_t::lognormal_k;
            else {
                fmt::println(stderr, "NUMKONG_RANDOM_DISTRIBUTION=\"{}\" is not uniform_k, lognormal_k or cauchy_k",
                             text);
                std::abort();
            }
        }

        dense_dimensions = dimension("NUMKONG_DENSE_DIMENSIONS", dense_dimensions);
        curved_dimensions = dimension("NUMKONG_CURVED_DIMENSIONS", curved_dimensions);
        sparse_dimensions = dimension("NUMKONG_SPARSE_DIMENSIONS", sparse_dimensions);
        mesh_points = positive("NUMKONG_MESH_POINTS", mesh_points);
        matrix_height = dimension("NUMKONG_MATRIX_HEIGHT", matrix_height);
        matrix_width = dimension("NUMKONG_MATRIX_WIDTH", matrix_width);
        matrix_depth = dimension("NUMKONG_MATRIX_DEPTH", matrix_depth);
        max_coord_angle = positive("NUMKONG_MAX_COORD_ANGLE", max_coord_angle);

        // Shrink dimensions for QEMU — divides whatever value is currently stored,
        // so explicit env-var overrides are proportionally reduced too.
        if (running_in_qemu) {
            dense_dimensions = std::max<std::size_t>(1, dense_dimensions / 4);
            matrix_height = std::max<std::size_t>(1, matrix_height / 4);
            matrix_width = std::max<std::size_t>(1, matrix_width / 4);
            matrix_depth = std::max<std::size_t>(1, matrix_depth / 4);
            mesh_points = std::max<std::size_t>(1, mesh_points / 4);
        }
    }

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

inline void print_stats_header(comparison_family_t family) noexcept {
    comparison_family_spec_t const spec = comparison_family_spec(family);
    fmt::println("{:<40} {:>12} {:>10} {:>12} {:>12} {:>10}\n", "Kernel", spec.column_labels[0], spec.column_labels[1],
                 spec.column_labels[2], spec.column_labels[3], spec.column_labels[4]);
}

struct error_stats_t;
bool should_fail(char const *kernel_name, error_stats_t const &stats) noexcept;
void print_stats_row(char const *kernel_name, error_stats_t const &stats) noexcept;

/** Names the running kernel for SIGILL diagnostics: set before each kernel call, cleared after, and
 *  read by the signal handler that main() installs to log the culprit before the process exits. */
extern char const *volatile nk_test_current_kernel_;

struct error_stats_section_t {
    char const *title = nullptr;
    nk_capability_t required = nk_cap_serial_k;
    nk_capability_t available;
    std::optional<comparison_family_t> last_family = std::nullopt;
    bool emitted_any = false;

    /** Runs only kernels whose family is in @p available: `#if NUMKONG_TARGET_X` says built, this says
     *  runnable. */
    explicit error_stats_section_t(nk_capability_t available = nk_capabilities_detected()) noexcept
        : available(available) {}

    /** Restart under a new heading, for kernels needing @p cap. */
    void section(char const *heading, nk_capability_t cap) noexcept {
        title = heading;
        required = cap;
        emitted_any = false;
        last_family.reset();
    }

    /** Runs @p test_fn over @p kernels, deducing a scenario's kernel types from the kernels
     *  themselves. */
    template <typename stats_type_ = error_stats_t, typename... kernels_types_>
    void operator()(char const *kernel_name, stats_type_ (*test_fn)(kernels_types_...), kernels_types_... kernels) {
        (*this)(kernel_name, [&] { return test_fn(kernels...); });
    }

    template <typename test_function_type_, typename... args_types_>
    void operator()(char const *kernel_name, test_function_type_ test_fn, args_types_ &&...args) {
        if ((available & required) == 0 || !global_config.should_run(kernel_name)) return;

        nk_test_current_kernel_ = kernel_name;
        auto stats = test_fn(std::forward<args_types_>(args)...);
        nk_test_current_kernel_ = nullptr;

        if (!emitted_any) {
            if (title) fmt::println("\n{}:", title);
            emitted_any = true;
        }
        if (last_family != stats.family) {
            print_stats_header(stats.family);
            last_family = stats.family;
        }
        print_stats_row(kernel_name, stats);
        ++global_config.kernel_count;
        if (should_fail(kernel_name, stats)) {
            ++global_config.failure_count;
            fmt::println("  rerun: NUMKONG_SEED={} NUMKONG_FILTER='^{}$' {}", global_config.seed, kernel_name,
                         global_config.program);
        }
    }
};

inline time_point test_start_time() { return steady_clock::now(); }

inline bool within_time_budget(time_point start) {
    return std::chrono::duration<double>(steady_clock::now() - start).count() < global_config.budget_seconds;
}

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

/** Accumulator for error statistics across multiple test trials. */
struct error_stats_t {
    comparison_family_t family = comparison_family_t::approximate_k;

    nk_f64_t min_abs_err = std::numeric_limits<nk_f64_t>::max();
    nk_f64_t max_abs_err = 0;
    nk_f64_t max_reference = 0;
    nk_f64_t sum_abs_err = 0;

    nk_f64_t min_rel_err = std::numeric_limits<nk_f64_t>::max();
    nk_f64_t max_rel_err = 0;
    nk_f64_t sum_rel_err = 0;

    std::uint64_t min_ulp = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_ulp = 0;
    f118_t sum_ulp = f118_t();

    nk_f64_t max_bound_ratio = 0;

    std::size_t count = 0;
    std::size_t exact_matches = 0;
    std::size_t failed_expectations = 0;
    bool saw_floating_distance = false;
    char const *first_failure = nullptr;

    explicit error_stats_t(comparison_family_t family = comparison_family_t::approximate_k) noexcept : family(family) {}

    /** Record a boolean property; @p property names it in the report when it does not hold. */
    void expect(bool held, char const *property) noexcept {
        if (!held && !first_failure) first_failure = property;
        failed_expectations += !held;
        accumulate(static_cast<int>(held), 1);
    }

    /** Records one result against its reference, failing when error exceeds @p bound, or on NaN. */
    template <typename actual_type_>
    void accumulate_bounded(actual_type_ actual, nk_f64_t expected, nk_f64_t bound) noexcept {
        nk_f64_t const error = std::fabs(static_cast<nk_f64_t>(actual) - expected);
        nk_f64_t const ratio = error == 0 ? 0 : error / bound;
        max_bound_ratio = std::isnan(ratio) ? std::numeric_limits<nk_f64_t>::infinity()
                                            : std::max(max_bound_ratio, ratio);
        accumulate_scalar(actual, expected);
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

        // Skip NaN/Inf pairs, whose sentinel distance means the comparison is meaningless; integers have none
        if constexpr (!nk::is_integral_dtype<actual_type_>())
            if (ulps == std::numeric_limits<std::uint64_t>::max()) return;

        if constexpr (!nk::is_integral_dtype<actual_type_>()) saw_floating_distance = true;

        if constexpr (!nk::is_integral_dtype<actual_type_>() || std::is_integral_v<actual_type_>) {
            nk_f64_t exp_f64 = static_cast<nk_f64_t>(expected_as_actual);
            nk_f64_t act_f64 = static_cast<nk_f64_t>(actual);

            nk_f64_t abs_err = std::fabs(exp_f64 - act_f64);
            nk_f64_t rel_err = exp_f64 != 0 ? abs_err / std::fabs(exp_f64) : abs_err;

            min_abs_err = std::min(min_abs_err, abs_err);
            max_abs_err = std::max(max_abs_err, abs_err);
            max_reference = std::max(max_reference, std::fabs(exp_f64));
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
        max_reference = std::max(max_reference, other.max_reference);
        sum_abs_err += other.sum_abs_err;
        min_rel_err = std::min(min_rel_err, other.min_rel_err);
        max_rel_err = std::max(max_rel_err, other.max_rel_err);
        sum_rel_err += other.sum_rel_err;
        min_ulp = std::min(min_ulp, other.min_ulp);
        max_ulp = std::max(max_ulp, other.max_ulp);
        sum_ulp += other.sum_ulp;
        max_bound_ratio = std::max(max_bound_ratio, other.max_bound_ratio);
        count += other.count;
        exact_matches += other.exact_matches;
        failed_expectations += other.failed_expectations;
        saw_floating_distance = saw_floating_distance || other.saw_floating_distance;
    }
};

inline bool should_fail(char const *kernel_name, error_stats_t const &stats) noexcept {
    if (stats.failed_expectations) return true;
    comparison_family_spec_t const spec = comparison_family_spec(stats.family);
    switch (spec.failure_mode) {
    case comparison_failure_mode_t::exact_distance_k:
        if (!stats.saw_floating_distance) return stats.max_ulp > 0;
        return stats.max_ulp > global_config.ulp_threshold_for(kernel_name);
    case comparison_failure_mode_t::ulp_threshold_k:
        return stats.max_ulp > global_config.ulp_threshold_for(kernel_name);
    case comparison_failure_mode_t::scale_threshold_k:
        return stats.max_abs_err > global_config.scale_threshold * stats.max_reference;
    case comparison_failure_mode_t::bound_threshold_k: return !(stats.max_bound_ratio <= 1);
    }
    return false;
}

inline void print_stats_row(char const *kernel_name, error_stats_t const &stats) noexcept {
    switch (stats.family) {
    case comparison_family_t::exact_k:
        fmt::println("{:<40} {:>12} {:>10.1f} {:>12.2e} {:>12} {:>10}", kernel_name, stats.max_ulp, stats.mean_ulp(),
                     stats.max_abs_err, stats.mismatches(), stats.exact_matches);
        break;
    case comparison_family_t::probability_k:
        fmt::println("{:<40} {:>12.2e} {:>10.2e} {:>12.2e} {:>12.2e} {:>10.2e}", kernel_name, stats.max_abs_err,
                     stats.mean_abs_err(), stats.max_rel_err, stats.mean_rel_err(), stats.mean_ulp());
        break;
    case comparison_family_t::geospatial_k:
        fmt::println("{:<40} {:>12.2e} {:>10.2e} {:>12.2e} {:>12.1f} {:>10}", kernel_name, stats.max_abs_err,
                     stats.mean_abs_err(), stats.max_rel_err, stats.mean_ulp(), stats.max_ulp);
        break;
    case comparison_family_t::approximate_k:
    case comparison_family_t::normalized_reduction_k:
        fmt::println("{:<40} {:>12.2e} {:>10.2e} {:>12.2e} {:>12} {:>10}", kernel_name, stats.max_abs_err,
                     stats.max_rel_err, stats.mean_ulp(), stats.max_ulp, stats.exact_matches);
        break;
    case comparison_family_t::bounded_k:
        fmt::println("{:<40} {:>12.2e} {:>10.2e} {:>12.2e} {:>12} {:>10}", kernel_name, stats.max_abs_err,
                     stats.max_rel_err, stats.max_bound_ratio, stats.max_ulp, stats.exact_matches);
        break;
    }
    // The counters say how many properties failed; this says which one.
    if (stats.first_failure && (stats.mismatches() || stats.failed_expectations))
        fmt::println("    first failure: {}", stats.first_failure);
    std::fflush(stdout);
}

/** Factory function to allocate vectors, potentially raising bad-allocs. */
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
 *  Dispatches to the matching `nk::fill_*` library function based on `global_config.distribution`.
 *  Infers sensible bounds from type's representable range.
 */
template <typename scalar_type_, typename allocator_type_, typename generator_type_>
void fill_random(generator_type_ &generator, nk::vector<scalar_type_, allocator_type_> &vector) {
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

#pragma region Host Backend

/** The arithmetic a backend accumulates products with, setting how far its results may land from
 *  the reference. */
enum class accumulation_t {

    /** Judged by the comparison family's own ULP or scale thresholds. */
    family_thresholds_k,

    /** Integer products summed exactly, compared bit for bit. */
    exact_k,

    /** F64 with TwoProd and TwoSum: two ulp plus 4 · γ² of Σ|a · b|. */
    dot2_k,

    /** Products exact in F64 and summed in F64: (depth + 1) · 2⁻⁵³ of Σ|a · b|, or exact. */
    f64_k,

    /** Products exact in F32 and summed in F32: (depth + 1) · 2⁻²⁴ of Σ|a · b|, or exact. */
    f32_k,

    /** 32-deep MMA blocks into F32, truncated to ~22 bits: (depth / 32 + 1) · 2⁻²² of Σ|a · b|. */
    tensor_core_k,
};

/** Runs CPU kernels in place: operands in host memory, direct calls, results readable at once. */
struct host_backend_t {

    /** The allocator every kernel operand comes from. */
    template <typename value_type_>
    using allocator = aligned_allocator<value_type_>;

    /** Dots of @p scalar_type_ accumulate integers exactly, F64 in Dot2, F32 in F64, and narrower
     *  floats in F32. */
    template <typename scalar_type_>
    static constexpr accumulation_t dots_accumulation() noexcept {
        using result_t = typename scalar_type_::dot_result_t;
        if constexpr (is_integral_dtype<result_t>()) return accumulation_t::exact_k;
        else if constexpr (std::is_same_v<scalar_type_, f64_t>) return accumulation_t::dot2_k;
        else if constexpr (std::is_same_v<result_t, f64_t>) return accumulation_t::f64_k;
        else return accumulation_t::f32_k;
    }

    /** Distances of @p scalar_type_ keep the comparison family's ULP thresholds. */
    template <typename scalar_type_>
    static constexpr accumulation_t spatials_accumulation() noexcept {
        return accumulation_t::family_thresholds_k;
    }

    /** Row stride for @p row_bytes: exactly one row, keeping the tightest stride covered. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return row_bytes; }

    /** Copies @p bytes between host buffers. */
    void copy(void *destination, void const *source, std::size_t bytes) noexcept {
        std::memcpy(destination, source, bytes);
    }

    /** Zeroes @p bytes of a host buffer. */
    void zero(void *destination, std::size_t bytes) noexcept { std::memset(destination, 0, bytes); }

    /** Calls @p kernel with @p arguments. */
    template <typename kernel_type_, typename... arguments_types_>
    void call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        kernel(arguments...);
    }

    /** Nothing runs asynchronously on the host, so nothing fails later. */
    char const *synchronize() noexcept { return nullptr; }
};

#pragma endregion Host Backend

#pragma region Suite Header

/** Prints the library version, the kits compiled in, and the kits this machine offers, once per
 *  binary. */
inline void log_environment() {
    char compiled[NUMKONG_CAPABILITIES_NAME_CAPACITY], detected[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_name_capabilities(nk_capabilities_compiled(), compiled, sizeof(compiled));
    nk_name_capabilities(nk_capabilities_detected(), detected, sizeof(detected));
    fmt::println("NumKong {}.{}.{}", NUMKONG_VERSION_MAJOR, NUMKONG_VERSION_MINOR, NUMKONG_VERSION_PATCH);
    fmt::println("- Compiled for: {}", compiled);
    fmt::println("- This machine: {}", detected);
}

#pragma endregion Suite Header

} // namespace ashvardanian::numkong::test

/** Forward declarations for test modules. */
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

/** Forward declarations for cross/batch tests, ISA-family files. */
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

#endif // NUMKONG_TEST_HARNESS_HPP
