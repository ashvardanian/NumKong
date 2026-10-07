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
 *  Variable                     Default    Meaning
 *  NUMKONG_DEVICES              auto       Comma-separated backend:ordinal, like cuda:0,rocm:1
 *  NUMKONG_FILTER               none       Regex over kernel names, or a substring if not a regex
 *  NUMKONG_SEED                 42         32-bit seed for random inputs, or random
 *  NUMKONG_TIME_LIMIT           1s         Randomized trials per kernel, like 200ms or 1s
 *  NUMKONG_DIMS                 1536       Vector dimension for dot and spatial tests, the first of a list
 *  NUMKONG_CURVED_DIMS          64         Vector dimension for curved tests, the first of a list
 *  NUMKONG_SPARSE_DIMS          256        Vector dimension for sparse tests, the first of a list
 *  NUMKONG_MESH_POINTS          1000       Point count for mesh tests
 *  NUMKONG_DIMS_HEIGHT          1024       GEMM M dimension, the first of a list
 *  NUMKONG_DIMS_WIDTH           128        GEMM N dimension, the first of a list
 *  NUMKONG_DIMS_DEPTH           1536       GEMM K dimension, the first of a list
 *  NUMKONG_IN_QEMU              0          Shrink shapes for emulated runs: 0, 1, true or false
 *  NUMKONG_ASSERT               1          Exit 1 on a failed accuracy check; 0 only reports it
 *  NUMKONG_ULP_THRESHOLD_F32    4          Max allowed ULP for f32
 *  NUMKONG_ULP_THRESHOLD_F16    32         Max allowed ULP for f16
 *  NUMKONG_ULP_THRESHOLD_BF16   256        Max allowed ULP for bf16
 *  NUMKONG_SCALE_THRESHOLD      0.02       Max error over the largest reference, for attention
 *  NUMKONG_MAX_COORD_ANGLE      180        Max geospatial angular separation in degrees, in [0, 180]
 *  NUMKONG_RANDOM_DISTRIBUTION  lognormal  uniform, lognormal or cauchy
 *  @endverbatim
 */

#pragma once
#ifndef NUMKONG_TEST_HARNESS_HPP
#define NUMKONG_TEST_HARNESS_HPP

#include <cassert> // `assert`
#include <cctype>  // `std::tolower`
#include <cmath>   // `std::fabs`, `std::isnan`, `std::ldexp`, `std::ilogb`
#include <cstddef> // `std::ptrdiff_t`
#include <cstdint> // `std::uint64_t`, `std::int32_t`, `std::int64_t`
#include <cstdio>  // `std::fflush`, `stdout`, `stderr`
#include <cstdlib> // `std::abort`, `std::exit`, `std::getenv`, `std::strtod`
#include <cstring> // `std::memcpy`, `std::strcmp`

#include <algorithm>    // `std::min`, `std::max`
#include <array>        // `std::array`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <complex>      // `std::complex`
#include <limits>       // `std::numeric_limits`
#include <memory>       // `std::shared_ptr`, `std::make_shared`
#include <new>          // `std::bad_alloc`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`
#include <regex>        // `std::regex`, `std::regex_search`
#include <string>       // `std::string`, `std::to_string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <tuple>        // `std::tuple`, `std::get`
#include <type_traits>  // `std::is_same_v`
#include <utility>      // `std::index_sequence`, `std::move`
#include <vector>       // `std::vector`

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

#include "numkong/memory.h"
#include "numkong/capabilities.h" // `nk_cpu_capabilities_detected`
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
#include "numkong/vector.hpp" // `nk::vector`
#include "numkong/memory.hpp" // `nk::allocator`, `nk::stream_t`

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

using steady_clock_t = std::chrono::steady_clock;
using time_point_t = steady_clock_t::time_point;

/** A 32-bit generator seed, kept apart from counts so neither passes for the other. */
struct seed_t {
    std::uint32_t value = 0;
};

/** A positive thread count: "0" in the environment resolves to every core this process may use. */
struct threads_t {
    std::size_t count = 0;
};

/** A size in bytes, kept apart from element counts. */
struct bytes_t {
    std::size_t value = 0;
};

/** The text of the environment variable @p name, or nothing when it is unset or empty. */
inline std::optional<std::string_view> env_text(char const *name) noexcept {
    char const *const text = std::getenv(name);
    if (!text || !*text) return std::nullopt;
    return std::string_view(text);
}

/** Parses the environment variable @p name with @p parse, or returns @p fallback when it is unset
 *  or empty. Text that does not parse prints `NAME="text" does not parse, expected <expected>` and
 *  exits with status 1, which leaves crash handlers quiet. */
template <typename value_type_, typename parse_type_>
[[nodiscard]] value_type_ env_parsed(char const *name, value_type_ fallback, parse_type_ &&parse,
                                     char const *expected) noexcept {
    std::optional<std::string_view> const text = env_text(name);
    if (!text) return fallback;
    if (std::optional<value_type_> value = parse(*text)) return *std::move(value);
    fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", name, *text, expected);
    std::exit(1);
}

/** A positive whole number, like "64". */
inline std::optional<std::size_t> parse_count(std::string_view text) noexcept {
    std::size_t count = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc {} || end != text.data() + text.size() || !count) return std::nullopt;
    return count;
}

/** A positive duration in whole milliseconds or seconds, like "200ms" or "10s". */
inline std::optional<std::chrono::milliseconds> parse_duration(std::string_view text) noexcept {
    std::uint32_t count = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc {} || !count) return std::nullopt;
    std::string_view const unit = text.substr(end - text.data());
    if (unit == "ms") return std::chrono::milliseconds(count);
    if (unit == "s") return std::chrono::seconds(count);
    return std::nullopt;
}

/** A positive size in whole bytes or binary units of any case, like "4096", "64KB" or "1GB". */
inline std::optional<bytes_t> parse_size(std::string_view text) noexcept {
    std::size_t bytes = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), bytes);
    if (error != std::errc {} || !bytes) return std::nullopt;
    std::string_view const unit = text.substr(end - text.data());
    auto const same_letter = [](char given, char lower) noexcept {
        return std::tolower(static_cast<unsigned char>(given)) == lower;
    };
    for (std::string_view const known : std::array<std::string_view, 5> {"", "kb", "mb", "gb", "tb"}) {
        if (std::equal(unit.begin(), unit.end(), known.begin(), known.end(), same_letter)) return bytes_t {bytes};
        if (bytes > std::numeric_limits<std::size_t>::max() / 1024) return std::nullopt;
        bytes *= 1024;
    }
    return std::nullopt;
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

/** A 32-bit seed, or "random" for a fresh draw from @c std::random_device. */
inline std::optional<seed_t> parse_seed(std::string_view text) noexcept {
    if (text == "random") return seed_t {static_cast<std::uint32_t>(std::random_device {}())};
    std::uint32_t seed = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), seed);
    if (error != std::errc {} || end != text.data() + text.size()) return std::nullopt;
    return seed_t {seed};
}

inline std::size_t env_count(char const *name, std::size_t fallback) noexcept {
    return env_parsed(name, fallback, parse_count, "a positive count");
}

inline std::chrono::milliseconds env_duration(char const *name, std::chrono::milliseconds fallback) noexcept {
    return env_parsed(name, fallback, parse_duration, "a duration like 200ms or 10s");
}

inline bytes_t env_size(char const *name, bytes_t fallback) noexcept {
    return env_parsed(name, fallback, parse_size, "a size like 4096, 64KB or 1GB");
}

inline bool env_flag(char const *name, bool fallback) noexcept {
    auto const parse_flag = [](std::string_view text) noexcept -> std::optional<bool> {
        if (text == "1" || text == "true") return true;
        if (text == "0" || text == "false") return false;
        return std::nullopt;
    };
    return env_parsed(name, fallback, parse_flag, "0, 1, true or false");
}

inline seed_t env_seed(char const *name, seed_t fallback) noexcept {
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random");
}

/** Spells @p duration as a user types it: whole seconds as "10s", anything else as "1500ms". */
inline std::string spell_duration(std::chrono::milliseconds duration) {
    auto const count = duration.count();
    return count % 1000 ? std::to_string(count) + "ms" : std::to_string(count / 1000) + "s";
}

/** Spells @p size as a user types it, in the largest binary unit dividing it, like "256MB". */
inline std::string spell_size(bytes_t size) {
    constexpr std::array<char const *, 5> units {"", "KB", "MB", "GB", "TB"};
    std::size_t bytes = size.value, power = 0;
    while (power + 1 != units.size() && bytes && bytes % 1024 == 0) bytes /= 1024, ++power;
    return std::to_string(bytes) + units[power];
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

/** Input shapes: as set, or divided by four for emulated SIMD. */
enum class emulation_t : unsigned char { native_k, emulated_k };
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

/** A requested GPU, before checking whether its runtime and ordinal are available. */
struct device_selection_t {
    nk::device_kind_t backend;
    std::size_t ordinal;
};

inline std::string_view device_name(nk::device_kind_t kind) noexcept {
    switch (kind) {
    case nk::device_kind_t::cpu_k: return "cpu";
    case nk::device_kind_t::cuda_k: return "cuda";
    case nk::device_kind_t::rocm_k: return "rocm";
    case nk::device_kind_t::metal_k: return "metal";
    }
    return "unrecognized";
}

inline std::optional<std::vector<device_selection_t>> parse_devices(std::string_view text) {
    std::vector<device_selection_t> devices;
    do {
        std::size_t const comma = text.find(',');
        std::string_view const entry = text.substr(0, comma);
        std::size_t const colon = entry.find(':');
        if (colon == std::string_view::npos) return std::nullopt;
        std::string_view const vendor = entry.substr(0, colon);
        nk::device_kind_t backend;
        if (vendor == "cuda") backend = nk::device_kind_t::cuda_k;
        else if (vendor == "rocm") backend = nk::device_kind_t::rocm_k;
        else if (vendor == "metal") backend = nk::device_kind_t::metal_k;
        else return std::nullopt;
        std::string_view const number = entry.substr(colon + 1);
        std::size_t ordinal = 0;
        auto const [end, error] = std::from_chars(number.data(), number.data() + number.size(), ordinal);
        if (error != std::errc {} || end != number.data() + number.size()) return std::nullopt;
        devices.push_back({backend, ordinal});
        if (comma == std::string_view::npos) return devices;
        text.remove_prefix(comma + 1);
    } while (!text.empty());
    return std::nullopt;
}

/** Every test setting, with its default as the initializer, filled once by @c read_settings. */
struct settings_t {

    std::optional<std::vector<device_selection_t>> devices;
    nk::device_t device = nk::device_t::cpu();

    /** Tests to run, by ECMAScript regex or else substring. */
    std::string_view filter;
    std::optional<std::regex> filter_regex;

    /** Random seed for reproducible tests. */
    seed_t seed {42};

    /** How long each kernel's randomized trials run. */
    std::chrono::milliseconds time_limit_per_kernel = std::chrono::seconds(1);

    /** Random distribution for test inputs. */
    distribution_t distribution = distribution_t::lognormal_k;

    /** What a failed accuracy check does to the exit status. */
    on_failure_t on_failure = on_failure_t::exit_k;

    /** Whether to shrink shapes for emulated SIMD. */
    emulation_t emulation = emulation_t::native_k;

    /** Max allowed ULP for f32. */
    std::uint64_t ulp_threshold_f32 = 4;

    /** Max allowed ULP for f16. */
    std::uint64_t ulp_threshold_f16 = 32;

    /** Max allowed ULP for bf16. */
    std::uint64_t ulp_threshold_bf16 = 256;

    /** Max absolute error as a fraction of the largest reference magnitude, for the
     *  normalized-reduction family. */
    nk_f64_t scale_threshold = 0.02;

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

    /** Number of 3D points for RMSD, Kabsch. */
    std::size_t mesh_points = 1000;

    /** Max geospatial angular separation. */
    float max_coord_angle_degrees = 180.0f;

    /** The binary's @c argv[0], which closes every rerun line. */
    std::string_view program;

    /** Whether @p name passes the filter. */
    bool selects(std::string_view name) const {
        if (filter.empty()) return true;
        if (filter_regex) return std::regex_search(name.begin(), name.end(), *filter_regex);
        return name.find(filter) != std::string_view::npos;
    }

    std::uint64_t ulp_threshold_for(nk_dtype_t dtype) const noexcept {
        if (dtype == nk_bf16_k) return ulp_threshold_bf16;
        if (dtype == nk_f16_k) return ulp_threshold_f16;
        return ulp_threshold_f32;
    }
};

/** Reads every @c settings_t variable, rejecting a zero count, for the binary named @p program. */
inline settings_t read_settings(char const *program) noexcept {
    settings_t settings;
    settings.program = program;
    // Python and JS take a comma list of dimensions, of which C++ runs the first
    auto const first_dim = [](char const *name, std::size_t fallback) noexcept {
        return env_parsed(name, std::vector {fallback}, parse_dims, "positive counts like 64,128").front();
    };
    auto const env_ulps = [](char const *name, std::uint64_t fallback) noexcept {
        auto const parse = [](std::string_view text) noexcept -> std::optional<std::uint64_t> {
            std::uint64_t ulps = 0;
            auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), ulps);
            if (error != std::errc {} || end != text.data() + text.size()) return std::nullopt;
            return ulps;
        };
        return env_parsed(name, fallback, parse, "an unsigned integer");
    };
    auto const number_within = [](char const *name, double fallback, double low, double high,
                                  char const *expected) noexcept {
        auto const parse = [=](std::string_view text) -> std::optional<double> {
            std::string const terminated(text);
            char *end = nullptr;
            double const value = std::strtod(terminated.c_str(), &end);
            if (end != terminated.c_str() + terminated.size() || !(value >= low && value <= high)) return std::nullopt;
            return value;
        };
        return env_parsed(name, fallback, parse, expected);
    };
    if (env_text("NUMKONG_DEVICES"))
        settings.devices = env_parsed("NUMKONG_DEVICES", std::vector<device_selection_t> {}, parse_devices,
                                      "comma-separated devices such as cuda:0,rocm:1");
    settings.filter = env_text("NUMKONG_FILTER").value_or("");
    if (!settings.filter.empty()) {
#if defined(__cpp_exceptions) && __cpp_exceptions
        try {
            settings.filter_regex.emplace(settings.filter.begin(), settings.filter.end());
        }
        catch (std::regex_error const &) {
        }
#else
        settings.filter_regex.emplace(settings.filter.begin(), settings.filter.end());
#endif
    }
    settings.seed = env_seed("NUMKONG_SEED", settings.seed);
    settings.time_limit_per_kernel = env_duration("NUMKONG_TIME_LIMIT", settings.time_limit_per_kernel);
    settings.distribution = env_parsed("NUMKONG_RANDOM_DISTRIBUTION", settings.distribution, parse_distribution,
                                       "uniform, lognormal or cauchy");
    settings.on_failure = env_flag("NUMKONG_ASSERT", settings.on_failure == on_failure_t::exit_k)
                              ? on_failure_t::exit_k
                              : on_failure_t::report_k;
    settings.emulation = env_flag("NUMKONG_IN_QEMU", settings.emulation == emulation_t::emulated_k)
                             ? emulation_t::emulated_k
                             : emulation_t::native_k;
    settings.ulp_threshold_f32 = env_ulps("NUMKONG_ULP_THRESHOLD_F32", settings.ulp_threshold_f32);
    settings.ulp_threshold_f16 = env_ulps("NUMKONG_ULP_THRESHOLD_F16", settings.ulp_threshold_f16);
    settings.ulp_threshold_bf16 = env_ulps("NUMKONG_ULP_THRESHOLD_BF16", settings.ulp_threshold_bf16);
    settings.scale_threshold = number_within("NUMKONG_SCALE_THRESHOLD", settings.scale_threshold, 0, 1,
                                             "a number in [0, 1]");
    settings.dense_dimensions = first_dim("NUMKONG_DIMS", settings.dense_dimensions);
    settings.matrix_height = first_dim("NUMKONG_DIMS_HEIGHT", settings.matrix_height);
    settings.matrix_width = first_dim("NUMKONG_DIMS_WIDTH", settings.matrix_width);
    settings.matrix_depth = first_dim("NUMKONG_DIMS_DEPTH", settings.matrix_depth);
    settings.curved_dimensions = first_dim("NUMKONG_CURVED_DIMS", settings.curved_dimensions);
    settings.sparse_dimensions = first_dim("NUMKONG_SPARSE_DIMS", settings.sparse_dimensions);
    settings.mesh_points = env_count("NUMKONG_MESH_POINTS", settings.mesh_points);
    settings.max_coord_angle_degrees = static_cast<float>(
        number_within("NUMKONG_MAX_COORD_ANGLE", settings.max_coord_angle_degrees, 0, 180, "a number in [0, 180]"));

    // Shrink dimensions for QEMU — divides whatever value is currently stored,
    // so explicit env-var overrides are proportionally reduced too.
    if (settings.emulation == emulation_t::emulated_k) {
        settings.dense_dimensions = std::max<std::size_t>(1, settings.dense_dimensions / 4);
        settings.matrix_height = std::max<std::size_t>(1, settings.matrix_height / 4);
        settings.matrix_width = std::max<std::size_t>(1, settings.matrix_width / 4);
        settings.matrix_depth = std::max<std::size_t>(1, settings.matrix_depth / 4);
        settings.mesh_points = std::max<std::size_t>(1, settings.mesh_points / 4);
    }
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar it parses from, then a rerun template. */
inline void print(settings_t const &settings) {
    if (settings.devices) {
        for (device_selection_t const &device : *settings.devices)
            fmt::println("- Device: {}:{}", device_name(device.backend), device.ordinal);
    }
    else fmt::println("- Devices: auto");
    fmt::println("- Seed: {}", settings.seed.value);
    fmt::println("- Filter: {}", settings.filter.empty() ? std::string_view("none") : settings.filter);
    fmt::println("- Time limit: {}", spell_duration(settings.time_limit_per_kernel));
    fmt::println("- Random distribution: {}", settings.distribution);
    fmt::println("- Assert: {}", settings.on_failure == on_failure_t::exit_k);
    fmt::println("- In QEMU: {}", settings.emulation == emulation_t::emulated_k);
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
    fmt::println("- Mesh points: {}", settings.mesh_points);
    fmt::println("- Max coord angle: {}", settings.max_coord_angle_degrees);
    fmt::println("- Rerun one test: NUMKONG_SEED={} NUMKONG_FILTER='^<name>$' {}", settings.seed.value,
                 settings.program);
}

/** The facts this binary and this machine report: the library version, and the capabilities
 *  compiled in and detected. */
struct machine_t {
    std::array<unsigned, 3> version {NUMKONG_VERSION_MAJOR, NUMKONG_VERSION_MINOR, NUMKONG_VERSION_PATCH};
    nk_capability_t compiled = 0;
    nk_capability_t detected = 0;
};

/** Everything a test reads, built once in @c main and passed down by reference. */
struct environment_t {
    settings_t settings;
    machine_t machine;
};

inline void print_stats_header(comparison_family_t family) noexcept {
    comparison_family_spec_t const spec = comparison_family_spec(family);
    fmt::println("{:<40} {:>12} {:>10} {:>12} {:>12} {:>10}\n", "Kernel", spec.column_labels[0], spec.column_labels[1],
                 spec.column_labels[2], spec.column_labels[3], spec.column_labels[4]);
}

struct error_stats_t;
bool should_fail(settings_t const &settings, error_stats_t const &stats) noexcept;
void print_stats_row(char const *kernel_name, error_stats_t const &stats) noexcept;

/** Names the running kernel for SIGILL diagnostics: set before each kernel call, cleared after, and
 *  read by the signal handler that main() installs to log the culprit before the process exits. */
extern char const *volatile nk_test_current_kernel_;

/** The capabilities this CPU runs, whether or not this binary holds them. */
inline nk_capability_t cpu_capabilities_detected() noexcept {
    nk_capability_t capabilities = 0;
    return nk_cpu_capabilities_detected(&capabilities) == nk_success_k ? capabilities : nk_cap_serial_k;
}

/** The CPU capabilities this binary holds, whether or not this CPU runs them. */
inline nk_capability_t cpu_capabilities_compiled() noexcept {
    nk_capability_t capabilities = 0;
    return nk_cpu_capabilities_compiled(&capabilities) == nk_success_k ? capabilities : nk_cap_serial_k;
}

/** Probes the capabilities @c machine_t reports. */
inline machine_t probe_machine() noexcept {
    machine_t machine;
    machine.compiled = cpu_capabilities_compiled();
    machine.detected = cpu_capabilities_detected();
    return machine;
}

/** Prints the version line, then the capabilities as "- Compiled for:" and "- This machine:". */
inline void print(machine_t const &machine) {
    char compiled[NUMKONG_CAPABILITIES_NAME_CAPACITY], detected[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_capabilities_name(machine.compiled, compiled, sizeof(compiled));
    nk_capabilities_name(machine.detected, detected, sizeof(detected));
    fmt::println("NumKong {}.{}.{}", machine.version[0], machine.version[1], machine.version[2]);
    fmt::println("- Compiled for: {}", compiled);
    fmt::println("- This machine: {}", detected);
}

/** A mask of no capability, so the C++ wrappers run their templates: the references every
 *  capability is checked against. */
inline constexpr nk_capability_t no_tiers_k = 0;

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
    [](auto... arguments) noexcept { return call_best<best_>(nk::default_capabilities(), arguments...); };

/** Runs each test under the settings, prints its row, and counts the kernels checked and failed:
 *  one per binary, built in @c main and passed to every test family. */
struct error_stats_section_t {
    settings_t settings;
    nk_capability_t available;
    char const *title = nullptr;
    nk_capability_t required = nk_cap_serial_k;
    std::optional<comparison_family_t> last_family = std::nullopt;
    std::size_t kernel_count = 0;
    std::size_t failure_count = 0;

    /** Runs only kernels whose family is in @p available: `#if NUMKONG_TARGET_X` says built, this says
     *  runnable. */
    error_stats_section_t(settings_t const &settings, nk_capability_t available) noexcept
        : settings(settings), available(available) {}

    /** Restart under a new heading, for kernels needing @p cap, or say why they skip. */
    void section(char const *heading, nk_capability_t cap) noexcept {
        title = heading;
        required = cap;
        last_family.reset();
        if (available & required) return;
        char missing_names[NUMKONG_CAPABILITIES_NAME_CAPACITY];
        nk_capabilities_name(required & ~available, missing_names, sizeof(missing_names));
        fmt::println("\n{}: skipped, {} not detected", heading, missing_names);
    }

    /** Runs @p test_fn over @p kernels, deducing a scenario's kernel types from the kernels
     *  themselves. */
    template <typename stats_type_ = error_stats_t, typename... kernels_types_>
    void operator()(char const *kernel_name,
                    std::type_identity_t<stats_type_ (*)(settings_t const &, kernels_types_...)> test_fn,
                    kernels_types_... kernels) {
        (*this)(kernel_name, [&](settings_t const &) { return test_fn(settings, kernels...); });
    }

    template <typename test_function_type_, typename... args_types_>
    void operator()(char const *kernel_name, test_function_type_ test_fn, args_types_ &&...args) {
        if ((available & required) == 0 || !settings.selects(kernel_name)) return;

        nk_test_current_kernel_ = kernel_name;
        auto stats = test_fn(settings, std::forward<args_types_>(args)...);
        nk_test_current_kernel_ = nullptr;

        if (!last_family && title) fmt::println("\n{}:", title);
        if (last_family != stats.family) {
            print_stats_header(stats.family);
            last_family = stats.family;
        }
        print_stats_row(kernel_name, stats);
        ++kernel_count;
        if (should_fail(settings, stats)) {
            ++failure_count;
            fmt::println("  rerun: NUMKONG_SEED={} NUMKONG_FILTER='^{}$' {}", settings.seed.value, kernel_name,
                         settings.program);
        }
    }
};

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

#pragma region Tracked References

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
        : value(static_cast<value_type_>(input)), magnitude(static_cast<double>(value.abs())) {}

    friend tracked operator+(tracked const &a, tracked const &b) noexcept {
        return {a.value + b.value, a.magnitude + b.magnitude, std::max(a.term_roundings, b.term_roundings),
                std::max(a.sum_roundings, b.sum_roundings) + 1};
    }
    friend tracked operator-(tracked const &a, tracked const &b) noexcept {
        return {a.value - b.value, a.magnitude + b.magnitude, std::max(a.term_roundings, b.term_roundings),
                std::max(a.sum_roundings, b.sum_roundings) + 1};
    }
    friend tracked operator*(tracked const &a, tracked const &b) noexcept {
        // Each part of a complex product also adds two real products
        std::size_t const own_roundings = value_type_::is_complex() ? 2 : 1;
        return {a.value * b.value, a.magnitude * b.magnitude, a.term_roundings + b.term_roundings + own_roundings,
                a.sum_roundings + b.sum_roundings};
    }

    /** Relative errors add in a quotient as in a product, and the division rounds once more. */
    friend tracked operator/(tracked const &a, tracked const &b) noexcept {
        double const divisor = static_cast<double>(b.value.abs());
        return {a.value / b.value, a.magnitude * b.magnitude / (divisor * divisor),
                a.term_roundings + b.term_roundings + 1, a.sum_roundings + b.sum_roundings};
    }
    friend bool operator>(tracked const &a, tracked const &b) noexcept { return a.value > b.value; }
    friend bool operator<(tracked const &a, tracked const &b) noexcept { return a.value < b.value; }
    tracked saturating_add(tracked const &other) const noexcept { return *this + other; }
    tracked saturating_mul(tracked const &other) const noexcept { return *this * other; }

    /** The radicand's error shrinks by the derivative 1 / (2√v), and the root rounds once more. */
    tracked sqrt() const noexcept {
        value_type_ const root = value.sqrt();
        double const root_magnitude = static_cast<double>(root);
        return {root, std::max(magnitude / (2 * root_magnitude), root_magnitude), term_roundings + 1, sum_roundings};
    }

    /** The argument's error shrinks by the derivative 1 / v, and the logarithm rounds once more. */
    tracked log() const noexcept {
        value_type_ const logarithm = value.log();
        double const argument = static_cast<double>(value.abs());
        return {logarithm, std::max(magnitude / argument, static_cast<double>(logarithm.abs())), term_roundings + 1,
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
        int const smallest_normal_exponent = std::ilogb(static_cast<double>(component_t::positive_min()));
        int const exponent = std::max(std::ilogb(value), smallest_normal_exponent);
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

    /** Judges results against tracked references, each term adding up to @p term_error_bound of
     *  Σ|terms|, like the `nk_*_error_bound` helpers return; zero demands exact results. */
    explicit error_stats_t(nk_f64_t term_error_bound) noexcept : error_stats_t(term_error_bound, term_error_bound) {}

    /** Judges results whose terms round within @p term_error_bound and whose sums, kept in a wider
     *  type, round within @p sum_error_bound. */
    error_stats_t(nk_f64_t term_error_bound, nk_f64_t sum_error_bound) noexcept
        : family(term_error_bound == 0 ? comparison_family_t::exact_k : comparison_family_t::bounded_k),
          term_error_bound(term_error_bound), sum_error_bound(sum_error_bound) {}

    /** Record a boolean property; @p property names it in the report when it does not hold. */
    void expect(bool held, char const *property) noexcept {
        if (!held && !first_failure) first_failure = property;
        failed_expectations += !held;
        accumulate(static_cast<int>(held), 1);
    }

    /** Record a kernel's @p status, failing it by the status's name when it wrote no result. */
    void expect(nk_status_t status) noexcept {
        if (status != nk_success_k) expect(false, nk_status_name(status));
    }

    /** Record a C++ wrapper's @p status, like the C one. */
    void expect(nk::status_t status) noexcept { expect(static_cast<nk_status_t>(status)); }

    /** Records one result against its reference, failing on NaN or when the error exceeds @p bound
     *  plus the rounding into @p actual_type_. Past that type's finite range, a saturated or
     *  overflowed result of the reference's sign is exact, and non-finite references are only
     *  counted, as @c accumulate_scalar does. */
    template <typename actual_type_, typename expected_type_>
    void accumulate_bounded(actual_type_ actual, expected_type_ expected, nk_f64_t bound) noexcept {
        nk_f64_t const result = static_cast<nk_f64_t>(actual), reference = static_cast<nk_f64_t>(expected);
        if (!std::isfinite(reference)) return accumulate_scalar(actual, reference);
        nk_f64_t const limit = static_cast<nk_f64_t>(actual_type_::finite_max());
        bool const saturated = std::fabs(reference) > limit && std::fabs(result) >= limit &&
                               std::signbit(result) == std::signbit(reference);
        nk_f64_t const error = saturated || result == reference ? 0 : std::fabs(result - reference);
        nk_f64_t const ratio = error == 0 ? 0 : error / (bound + half_ulp<actual_type_>(reference));
        max_bound_ratio = std::isnan(ratio) ? std::numeric_limits<nk_f64_t>::infinity()
                                            : std::max(max_bound_ratio, ratio);
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
     *  @c sum_error_bound per rounding of a sum. */
    template <typename actual_type_, typename value_type_>
    void accumulate(actual_type_ actual, tracked<value_type_> const &expected) noexcept {
        nk_f64_t const bound = ((expected.term_roundings + 1) * term_error_bound +
                                expected.sum_roundings * sum_error_bound) *
                               expected.magnitude;
        if constexpr (nk::is_integral_dtype<actual_type_>())
            accumulate_scalar(actual, expected.value.template to<actual_type_>());
        else if constexpr (nk::is_complex_dtype<actual_type_>())
            accumulate_bounded(actual.real(), expected.value.real(), bound),
                accumulate_bounded(actual.imag(), expected.value.imag(), bound);
        else accumulate_bounded(actual, expected.value, bound);
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

        // A NaN reference makes the distance meaningless; a NaN result against a number fails
        if constexpr (!nk::is_integral_dtype<actual_type_>())
            if (ulps == std::numeric_limits<std::uint64_t>::max()) {
                if (!std::isnan(static_cast<double>(expected_as_actual))) expect(false, "NaN result for a number");
                return;
            }

        if constexpr (!nk::is_integral_dtype<actual_type_>()) saw_floating_distance = true;
        if constexpr (requires { actual_type_::dtype(); }) result_dtype = actual_type_::dtype();

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
        error_stats_t fresh {family};
        fresh.term_error_bound = term_error_bound;
        *this = fresh;
    }

    void merge(error_stats_t const &other) noexcept {
        if (other.count == 0) return;
        if (count == 0) family = other.family, term_error_bound = other.term_error_bound;
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

#if NUMKONG_HEADER_ONLY

/** The header-only stub of @p best_, called with the arguments of its capability kernels, which it
 *  never reads, reports the missing library over every mask. */
template <auto best_, typename... arguments_types_>
error_stats_t test_missing_library(settings_t const &, arguments_types_... arguments) {
    error_stats_t stats(comparison_family_t::exact_k);
    stats.expect(call_best<best_>(nk_cap_any_k, arguments...) == nk_missing_library_k,
                 "a header-only dispatch point ran a kernel");
    return stats;
}
#endif // NUMKONG_HEADER_ONLY

inline bool should_fail(settings_t const &settings, error_stats_t const &stats) noexcept {
    if (stats.failed_expectations) return true;
    comparison_family_spec_t const spec = comparison_family_spec(stats.family);
    switch (spec.failure_mode) {
    case comparison_failure_mode_t::exact_distance_k:
        if (!stats.saw_floating_distance) return stats.max_ulp > 0;
        return stats.max_ulp > settings.ulp_threshold_for(stats.result_dtype);
    case comparison_failure_mode_t::ulp_threshold_k:
        return stats.max_ulp > settings.ulp_threshold_for(stats.result_dtype);
    case comparison_failure_mode_t::scale_threshold_k:
        return stats.max_abs_err > settings.scale_threshold * stats.max_reference;
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
    auto result = nk::vector<type_>::zeros(n);
#if defined(__cpp_exceptions) && __cpp_exceptions
    if (!result) throw std::bad_alloc();
#else
    if (!result) std::abort();
#endif
    return std::move(result.value);
}

/**
 *  @brief Fill buffer with random values, respecting the distribution setting.
 *
 *  Dispatches to the matching `nk::fill_*` library function based on @p settings.
 *  Infers sensible bounds from type's representable range.
 */
template <typename scalar_type_, typename allocator_type_, typename generator_type_>
void fill_random(settings_t const &settings, generator_type_ &generator,
                 nk::vector<scalar_type_, allocator_type_> &vector) {
    switch (settings.distribution) {
    case distribution_t::uniform_k: nk::fill_uniform(generator, vector.values_data(), vector.size_values()); break;
    case distribution_t::lognormal_k: nk::fill_lognormal(generator, vector.values_data(), vector.size_values()); break;
    case distribution_t::cauchy_k: nk::fill_cauchy(generator, vector.values_data(), vector.size_values()); break;
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

#pragma region Host Backend

/** Runs CPU kernels in place: operands in host memory, direct calls, results readable at once,
 *  keeping the first failed status. */
struct host_backend_t {

    /** The allocator every kernel operand comes from. */
    template <typename value_type_>
    using allocator = aligned_allocator<value_type_>;

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

    /** Calls @p kernel with @p arguments and the null stream of the CPU. */
    template <typename kernel_type_, typename... arguments_types_>
    nk_status_t call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., nullptr);
    }

    /** Waits for submitted work. */
    nk_status_t synchronize() noexcept { return nk_success_k; }
};

#if NUMKONG_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)
inline bool cuda_check_(cudaError_t error, char const *expression, char const *file, int line) {
    if (error == cudaSuccess) return true;
    fmt::println(stderr, "CUDA error {} at {}:{}: {}", expression, file, line, cudaGetErrorString(error));
    return false;
}

#define nk_cuda_assert_(expression)                                                    \
    do {                                                                               \
        if (!cuda_check_((expression), #expression, __FILE__, __LINE__)) return false; \
    } while (0)

struct cuda_device_scope_t {
    int caller = 0;
    cudaError_t status;

    explicit cuda_device_scope_t(nk_size_t ordinal) noexcept : status(cudaGetDevice(&caller)) {
        if (status == cudaSuccess) status = cudaSetDevice(static_cast<int>(ordinal));
    }
    cuda_device_scope_t(cuda_device_scope_t const &) = delete;
    cuda_device_scope_t &operator=(cuda_device_scope_t const &) = delete;
    ~cuda_device_scope_t() noexcept {
        if (status == cudaSuccess) cudaSetDevice(caller);
    }
};
#endif // NUMKONG_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)

/** Runs one vendor's kernels on an owned stream over unified memory. */
template <auto synchronize_>
struct device_backend {

    /** The allocator every kernel operand comes from, readable by the host once the stream is
     *  synchronized. */
    template <typename value_type_>
    using allocator = nk::allocator<value_type_>;

    struct owner_t {
        nk::stream_t queue;
        nk::allocator<char> memory;
        nk_capability_t capabilities = 0;
        nk_status_t status;

        explicit owner_t(nk::device_t device) noexcept {
            auto const enabled = device.capabilities_enabled();
            capabilities = enabled.value;
            status = static_cast<nk_status_t>(enabled.status);
            if (status != nk_success_k) return;
            auto created = nk::stream_t::make(device);
            status = static_cast<nk_status_t>(created.status);
            if (status != nk_success_k) return;
            queue = std::move(created.value);
            auto allocated = nk::allocator<char>::make(capabilities, queue.get());
            memory = allocated.value;
            status = static_cast<nk_status_t>(allocated.status);
        }
    };

    std::shared_ptr<owner_t> owner;
    void *const stream;

    explicit device_backend(nk::device_t device)
        : owner(std::make_shared<owner_t>(device)), stream(owner->queue.get()) {}

    nk_capability_t capabilities() const noexcept { return owner->capabilities; }

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes the A contract requires. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept {
        return nk_size_round_up_to_multiple_(row_bytes, 16);
    }

    /** Copies @p bytes once every queued call has finished with them. */
    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        if (owner->status != nk_success_k) return owner->status;
        nk_status_t const status = synchronize_(stream);
        if (status != nk_success_k) return status;
        std::memcpy(destination, source, bytes);
        return nk_success_k;
    }

    /** Zeroes @p bytes once every queued call has finished with them. */
    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        if (owner->status != nk_success_k) return owner->status;
        nk_status_t const status = synchronize_(stream);
        if (status != nk_success_k) return status;
        std::memset(destination, 0, bytes);
        return nk_success_k;
    }

    /** Launches @p kernel with @p arguments on the stream. */
    template <typename kernel_type_, typename... arguments_types_>
    nk_status_t call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        if (owner->status != nk_success_k) return owner->status;
        return kernel(arguments..., stream);
    }

    /** Calls @p kernel on operands it must refuse, reporting whether it returned
     *  @c nk_misaligned_k unlaunched. */
    template <typename kernel_type_, typename... arguments_types_>
    bool refuses_misaligned(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., stream) == nk_misaligned_k;
    }

    /** Waits for submitted work. */
    nk_status_t synchronize() noexcept {
        if (owner->status != nk_success_k) return owner->status;
        return synchronize_(stream);
    }
};

template <typename value_type_, auto synchronize_>
auto allocator_of(device_backend<synchronize_> const &backend) noexcept {
    return nk::allocator<value_type_>(backend.owner->memory);
}

#if NUMKONG_ARCH_ROCM_
using rocm_backend_t = device_backend<nk_stream_synchronize_rocm>;
#endif // NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_CUDA_
using cuda_backend_t = device_backend<nk_stream_synchronize_cuda>;
#endif // NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_METAL_
using metal_backend_t = device_backend<nk_stream_synchronize_metal>;
#endif // NUMKONG_ARCH_METAL_

template <typename backend_type_>
backend_type_ make_backend(settings_t const &settings) {
    if constexpr (std::is_same_v<backend_type_, host_backend_t>) return {};
    else return backend_type_(settings.device);
}

/** The allocator @p backend hands out @p value_type_ from: stateless, unless the backend's memory
 *  belongs to a context it holds and it overloads this. */
template <typename value_type_, typename backend_type_>
typename backend_type_::template allocator<value_type_> allocator_of(backend_type_ const &) noexcept {
    return {};
}

#pragma endregion Host Backend

/** Forward declarations for test modules. */
void test_casts(error_stats_section_t &check);
void test_reduce(error_stats_section_t &check);
void test_dot(error_stats_section_t &check);
void test_spatial(error_stats_section_t &check);
void test_set(error_stats_section_t &check);
void test_curved(error_stats_section_t &check);
void test_probability(error_stats_section_t &check);
void test_each(error_stats_section_t &check);
void test_trigonometry(error_stats_section_t &check);
void test_geospatial(error_stats_section_t &check);
void test_mesh(error_stats_section_t &check);
void test_sparse(error_stats_section_t &check);
void test_vector_types(error_stats_section_t &check);
void test_tensor_ops(error_stats_section_t &check);
void test_maxsim(error_stats_section_t &check);
void test_each_cuda(error_stats_section_t &check);
void test_each_rocm(error_stats_section_t &check);
void test_cast_cuda(error_stats_section_t &check);
void test_reduce_cuda(error_stats_section_t &check);
void test_tensor_cuda(error_stats_section_t &check);

/** Forward declarations for cross/batch tests, ISA-family files. */
void test_cross_serial(error_stats_section_t &check);
void test_cross_x8664(error_stats_section_t &check);
void test_cross_arm64(error_stats_section_t &check);
void test_cross_blas(error_stats_section_t &check);
void test_cross_riscv64(error_stats_section_t &check);
void test_cross_ppc64(error_stats_section_t &check);
void test_cross_loongarch64(error_stats_section_t &check);
void test_cross_wasm(error_stats_section_t &check);
void test_cross_cuda(error_stats_section_t &check);
void test_cross_rocm(error_stats_section_t &check);
void test_cross_metal(error_stats_section_t &check);

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_HARNESS_HPP
