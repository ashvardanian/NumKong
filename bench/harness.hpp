/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief NumKong C++ benchmark suite, header file.
 *
 *  Comprehensive benchmarks for NumKong SIMD-optimized functions measuring throughput performance.
 *  This header holds measurement infrastructure and templates shared across many benchmark files.
 *
 *  Environment Variables, each overridden by its flag, like <tt>--time-limit 2s</tt>:
 *
 *  @verbatim
 *  Variable                      Flag                    Default  Meaning
 *  NUMWARS_DEVICES               --devices               auto     Comma-separated backend:ordinal, like cuda:0,rocm:1
 *  NUMWARS_FILTER                --filter                none     Regex over benchmark names, or a substring if not a
 * regex NUMWARS_SEED                  --seed                  42       Seed for random inputs, any unsigned 64-bit
 * value, or random NUMWARS_WARMUP                --warmup                1s       Untimed run per benchmark, like
 * 200ms, 1s or 2m NUMWARS_TIME_LIMIT            --time-limit            10s      Timed run per benchmark, like 200ms,
 * 10s or 2m NUMWARS_BATCH_PER_CORE        --batch-per-core        2048     Elements, pairs or points per call of
 * streaming kernels NUMWARS_DIMS                  --dims                  1536     Vector dimension for dot and spatial
 * benchmarks NUMWARS_DIMS_HEIGHT           --dims-height           1024     GEMM M dimension, like the dataset size for
 * kNN NUMWARS_DIMS_WIDTH            --dims-width            128      GEMM N dimension, like the query count for kNN
 *  NUMWARS_DIMS_DEPTH            --dims-depth            1536     GEMM K dimension, like the vector dimension for kNN
 *  NUMKONG_CURVED_DIMS           --curved-dims           64       Vector dimension for curved benchmarks
 *  NUMKONG_SHAPE_POINTS          --shape-points          32       Points per cloud for shape benchmarks
 *  NUMKONG_SPARSE_FIRST_LENGTH   --sparse-first-length   1024     Elements in the first sparse set
 *  NUMKONG_SPARSE_SECOND_LENGTH  --sparse-second-length  8192     Elements in the second sparse set
 *  NUMKONG_SPARSE_INTERSECTION   --sparse-intersection   0.5      Share of the smaller sparse set both sets hold, in
 * (0, 1] NUMKONG_MAX_COORD_ANGLE       --max-coord-angle       180      Max geospatial angular separation in degrees,
 * in [0, 180]
 *  @endverbatim
 */

#pragma once
#ifndef NUMKONG_BENCH_HARNESS_HPP
#define NUMKONG_BENCH_HARNESS_HPP

#include <cctype>  // `std::tolower`
#include <cmath>   // `std::isfinite`, `std::floor`
#include <cstddef> // `std::byte`
#include <cstdint> // `std::uint64_t`
#include <cstdio>  // `std::fprintf`, `std::snprintf`
#include <cstdlib> // `std::abort`, `std::exit`, `std::getenv`, `std::strtod`
#include <cstring> // `std::strcmp`, `std::strncmp`

#include <algorithm>    // `std::clamp`, `std::equal`
#include <array>        // `std::array`
#include <bit>          // `std::bit_floor`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <limits>       // `std::numeric_limits`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`
#include <regex>        // `std::regex`, `std::regex_search`
#include <string>       // `std::string`, `std::to_string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <type_traits>  // `std::is_function_v`, `std::is_null_pointer_v`
#include <utility>      // `std::move`
#include <vector>       // `std::vector`

#include <fmt/base.h> // `fmt::println`

#if !defined(NUMKONG_ALLOW_ISA_REDIRECT)
#define NUMKONG_ALLOW_ISA_REDIRECT 0
#endif

#if !defined(NUMKONG_COMPARE_TO_MKL)
#define NUMKONG_COMPARE_TO_MKL 0
#endif
#if !defined(NUMKONG_COMPARE_TO_BLAS)
#define NUMKONG_COMPARE_TO_BLAS 0
#endif
#if !defined(NUMKONG_COMPARE_TO_ACCELERATE)
#define NUMKONG_COMPARE_TO_ACCELERATE 0
#endif

/*  MKL provides additional GEMM routines:
 *  - cblas_gemm_bf16bf16f32: BF16 inputs to F32 output
 *  - cblas_hgemm: F16 GEMM, if available */
#if NUMKONG_COMPARE_TO_MKL
#include <mkl.h>
#elif NUMKONG_COMPARE_TO_ACCELERATE
#include <Accelerate/Accelerate.h> // Apple Accelerate framework
#elif NUMKONG_COMPARE_TO_BLAS
#include <cblas.h> // Generic CBLAS (OpenBLAS, etc.)

/** OpenBLAS thread control, weak symbol to avoid link errors if not present. */
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
#endif             // NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE || NUMKONG_COMPARE_TO_BLAS

#include "numkong/capabilities.h" // `nk_capabilities_name`, `NUMKONG_VERSION_MAJOR`
#include "numkong/types.hpp"
#include "numkong/memory.hpp"
#include "numkong/tensor.hpp"
#include "numkong/random.hpp"

namespace nk = ashvardanian::numkong;

namespace ashvardanian::numkong::bench {

/** A size in bytes, kept apart from element counts. */
struct bytes_t {
    std::size_t value = 0;
};

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const *variable, char const *text, char const *expected) noexcept {
    std::fprintf(stderr, "%s=\"%s\" does not parse, expected %s\n", variable, text, expected);
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
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
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
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (error == std::errc {} && count != 0) {
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
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: a regex search when the pattern compiles, else a substring. */
struct filter_t {
    std::string pattern;
    std::optional<std::regex> regex;

    bool selects(std::string_view label) const noexcept {
        if (pattern.empty()) return true;
        if (regex) return std::regex_search(label.begin(), label.end(), *regex);
        return label.find(pattern) != std::string_view::npos;
    }
};

/** The filter a setting spells, compiled once; empty, selecting everything, when unset. */
inline filter_t read_filter_setting(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    filter_t filter;
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.pattern = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.emplace(filter.pattern);
    }
    catch (std::regex_error const &) {
    }
#else
    filter.regex.emplace(filter.pattern);
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    filter_t filter;
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
 *  @brief SplitMix64 over one 64-bit state, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t state = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : state(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(state += 0x9E3779B97F4A7C15ull); }

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

/** Spells @p duration as a user types it: whole seconds as "10s", anything else as "1500ms". */
inline std::string spell_duration(std::chrono::milliseconds duration) {
    auto const count = duration.count();
    return count % 1000 ? std::to_string(count) + "ms" : std::to_string(count / 1000) + "s";
}

/** Keeps @p value, and every write before it, from being optimized away. */
template <typename value_type_>
inline void do_not_optimize(value_type_ &&value) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    [[maybe_unused]] auto volatile *pointer = &value;
    _ReadWriteBarrier();
#elif defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

/** How @c print shows a counter: as it is, or per second in decimal units or in binary bytes. */
enum class counter_kind_t : unsigned char { value_k, decimal_rate_k, byte_rate_k };

/** A named amount per call, shown as its @c kind says. */
struct counter_t {
    char const *name = nullptr;
    double value = 0;
    counter_kind_t kind = counter_kind_t::value_k;
};

/** The width every row pads its name to, so the counters line up in one column. */
inline constexpr std::size_t row_name_width_k = 56;

/** A finished benchmark: its name, calls per second and counters, or why it was skipped. */
struct row_t {
    std::string_view name;
    double calls_per_second = 0;
    std::array<counter_t, 4> counters {};
    std::optional<std::string_view> skip_reason;
};

/** Prints @p row on one line: byte rates in binary units like "GB/s", other rates like "47.6/s" or
 *  "1.234 M/s", then the speed-up over @p baseline_calls_per_second when there is one. */
inline void print(row_t const &row, std::optional<double> baseline_calls_per_second = std::nullopt) {
    if (row.skip_reason) return fmt::println("{:<{}} skipped: {}", row.name, row_name_width_k, *row.skip_reason);
    auto const print_rate = [](char const *name, double per_second, counter_kind_t kind) {
        bool const binary = kind == counter_kind_t::byte_rate_k;
        double const step = binary ? 1024 : 1000;
        char const *prefix = "";
        for (char const *next : binary ? std::array {"K", "M", "G", "T", "P"} : std::array {"k", "M", "G", "T", "P"})
            if (per_second >= step) per_second /= step, prefix = next;
        // Significant digits rather than decimals, so a slow rate never rounds to zero
        fmt::print("  {} {:.4g}", name, per_second);
        if (binary || *prefix) fmt::print(" {}{}", prefix, binary ? "B" : "");
        fmt::print("/s");
    };
    fmt::print("{:<{}}", row.name, row_name_width_k);
    print_rate("calls", row.calls_per_second, counter_kind_t::decimal_rate_k);
    for (counter_t const &counter : row.counters)
        if (counter.kind != counter_kind_t::value_k)
            print_rate(counter.name, counter.value * row.calls_per_second, counter.kind);
        else if (counter.name) fmt::print("  {} {:.4g}", counter.name, counter.value);
    if (baseline_calls_per_second) fmt::print("  {:.2f}x", row.calls_per_second / *baseline_calls_per_second);
    fmt::print("\n");
}

/**
 *  @brief One benchmark's timed loop, iterated as `for (std::size_t call : loop)`.
 *
 *  Setup above the loop stays untimed, and the clock starts at @c begin. The loop runs untimed for
 *  the warm-up, then counts calls until the time limit. It reads the clock again only once the
 *  calls so far have grown by a 64th, so a short call doesn't time the clock itself. A backend that
 *  times windows of calls on its device reports each through @c add_window, and those then replace
 *  the loop's calls and wall time.
 */
class loop_t {
    using steady_clock_t = std::chrono::steady_clock;
    using seconds_t = std::chrono::duration<double>;

    /** Where the loop is: untimed, timed, or finished by the time limit or a skip. */
    enum class phase_t : unsigned char { warming_up_k, timed_k, done_k };

    /** The calls so far, divided by this, run between two reads of the clock. */
    static constexpr std::size_t clock_stride_k = 64;

    std::chrono::milliseconds warmup_, time_limit_;
    steady_clock_t::time_point start_ {};
    steady_clock_t::duration elapsed_ {};
    seconds_t window_elapsed_ {};
    std::size_t calls_ = 0, next_check_ = 0, window_calls_ = 0;
    phase_t phase_ = phase_t::warming_up_k;
    std::optional<std::string_view> skip_reason_;
    std::array<counter_t, 4> counters_ {};

    bool keep_running() noexcept {
        if (phase_ == phase_t::done_k) return false;
        if (calls_ < next_check_) return true;
        elapsed_ = steady_clock_t::now() - start_;
        if (phase_ == phase_t::warming_up_k && elapsed_ >= warmup_)
            phase_ = phase_t::timed_k, start_ = steady_clock_t::now(), elapsed_ = {}, calls_ = 0, window_elapsed_ = {},
            window_calls_ = 0;
        else if (phase_ == phase_t::timed_k && elapsed_ >= time_limit_) return phase_ = phase_t::done_k, false;
        next_check_ = calls_ + calls_ / clock_stride_k + 1;
        return true;
    }

    void add(counter_t counter) noexcept {
        for (counter_t &slot : counters_)
            if (!slot.name) return void(slot = counter);
    }

  public:
    struct end_t {};
    struct iterator_t {
        loop_t *loop;
        bool operator!=(end_t) noexcept { return loop->keep_running(); }
        std::size_t operator*() const noexcept { return loop->calls_; }
        void operator++() noexcept { ++loop->calls_; }
    };

    loop_t(std::chrono::milliseconds warmup, std::chrono::milliseconds time_limit) noexcept
        : warmup_(warmup), time_limit_(time_limit) {}

    iterator_t begin() noexcept { return start_ = steady_clock_t::now(), iterator_t {this}; }
    end_t end() const noexcept { return {}; }

    /** Stops the loop and reports @p reason instead of results. */
    void skip(std::string_view reason) noexcept { skip_reason_ = reason, phase_ = phase_t::done_k; }

    /** Adds @p calls the device ran back to back in @p elapsed. */
    void add_window(seconds_t elapsed, std::size_t calls) noexcept {
        window_elapsed_ += elapsed, window_calls_ += calls;
    }

    /** Reports @p per_call of @p name per second, in decimal units. */
    void rate(char const *name, double per_call) noexcept { add({name, per_call, counter_kind_t::decimal_rate_k}); }

    /** Reports @p bytes_per_call per second, in binary units. */
    void byte_rate(double bytes_per_call) noexcept { add({"bytes", bytes_per_call, counter_kind_t::byte_rate_k}); }

    /** Reports @p value as @p name, unscaled. */
    void counter(char const *name, double value) noexcept { add({name, value, counter_kind_t::value_k}); }

    /** The finished benchmark under @p name. */
    row_t row(std::string_view name) const noexcept {
        if (skip_reason_) return {name, 0, {}, skip_reason_};
        double const calls_per_second = window_calls_ ? window_calls_ / window_elapsed_.count()
                                                      : calls_ / seconds_t(elapsed_).count();
        return {name, calls_per_second, counters_, std::nullopt};
    }
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

/** Whether a kernel's @p status is a success; otherwise skips @p loop, naming the status. */
inline bool succeeded(loop_t &loop, nk_status_t status) noexcept {
    if (status == nk_success_k) return true;
    loop.skip(nk_status_name(status));
    return false;
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

/** Every benchmark setting, its default as the initializer, read once by @c read_bench_settings. */
struct bench_settings_t : common_settings_t {

    /** GPUs to time, or every one found when unset. */
    std::optional<std::vector<device_selection_t>> devices;

    /** Untimed run ahead of each benchmark. */
    std::chrono::milliseconds warmup = std::chrono::seconds(1);

    /** Timed run of each benchmark. */
    std::chrono::milliseconds time_limit = std::chrono::seconds(10);

    /** Elements, pairs or points per call of the streaming kernels, times one core on the CPU. */
    std::size_t batch_per_core = 2048;

    /** Vector dimension for dot products and spatial metrics. */
    std::size_t dense_dimensions = 1536;

    /** GEMM M dimension. */
    std::size_t matrix_height = 1024;

    /** GEMM N dimension. */
    std::size_t matrix_width = 128;

    /** GEMM K dimension. */
    std::size_t matrix_depth = 1536;

    /** Curved metric dimensions, quadratic in cost. */
    std::size_t curved_dimensions = 64;

    /** Points per cloud for the many-to-many shape distances. */
    std::size_t shape_points = 32;

    /** Elements in the first sparse set. */
    std::size_t sparse_first_count = 1024;

    /** Elements in the second sparse set. */
    std::size_t sparse_second_count = 8192;

    /** The share of the smaller sparse set both sets hold, in (0, 1]. */
    double sparse_intersection_share = 0.5;

    /** Max geospatial angular separation. */
    float max_coord_angle_degrees = 180.0f;
};

/** A number the bench shares with the test suite, hence a @c NUMKONG_* variable, not a @c NUMWARS_*
 *  one, @p fallback when unset; a whole one, for an integral @p value_type_ , between @p low and
 *  @p high . */
template <typename value_type_>
value_type_ read_library_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                 value_type_ fallback, double low, double high, char const *expected) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    char *end = nullptr;
    double const value = std::strtod(text, &end);
    bool const whole = !std::is_integral_v<value_type_> || value == std::floor(value);
    if (*end == '\0' && whole && value >= low && value <= high) return static_cast<value_type_>(value);
    exit_unparsed(variable, text, expected);
}

/** Reads every @c bench_settings_t setting. */
inline bench_settings_t read_bench_settings(int argc, char **argv) noexcept {
    bench_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "NUMWARS_SEED", "--seed");
    settings.filter = read_filter_setting(argc, argv, "NUMWARS_FILTER", "--filter");
    settings.program = argc != 0 ? argv[0] : "";
    if (char const *text = setting_text(argc, argv, "NUMWARS_DEVICES", "--devices")) {
        settings.devices = parse_devices(text);
        if (!settings.devices) exit_unparsed("NUMWARS_DEVICES", text, "comma-separated devices such as cuda:0,rocm:1");
    }
    settings.warmup = read_duration_setting(argc, argv, "NUMWARS_WARMUP", "--warmup", settings.warmup);
    settings.time_limit = read_duration_setting(argc, argv, "NUMWARS_TIME_LIMIT", "--time-limit", settings.time_limit);
    settings.batch_per_core = read_count_setting(argc, argv, "NUMWARS_BATCH_PER_CORE", "--batch-per-core",
                                                 settings.batch_per_core);
    settings.dense_dimensions = read_count_setting(argc, argv, "NUMWARS_DIMS", "--dims", settings.dense_dimensions);
    settings.matrix_height = read_count_setting(argc, argv, "NUMWARS_DIMS_HEIGHT", "--dims-height",
                                                settings.matrix_height);
    settings.matrix_width = read_count_setting(argc, argv, "NUMWARS_DIMS_WIDTH", "--dims-width", settings.matrix_width);
    settings.matrix_depth = read_count_setting(argc, argv, "NUMWARS_DIMS_DEPTH", "--dims-depth", settings.matrix_depth);
    double const counts = static_cast<double>((std::numeric_limits<std::size_t>::max)());
    settings.curved_dimensions = read_library_setting(argc, argv, "NUMKONG_CURVED_DIMS", "--curved-dims",
                                                      settings.curved_dimensions, 1, counts, "a positive count");
    settings.shape_points = read_library_setting(argc, argv, "NUMKONG_SHAPE_POINTS", "--shape-points",
                                                 settings.shape_points, 1, counts, "a positive count");
    settings.sparse_first_count = read_library_setting(argc, argv, "NUMKONG_SPARSE_FIRST_LENGTH",
                                                       "--sparse-first-length", settings.sparse_first_count, 1, counts,
                                                       "a positive count");
    settings.sparse_second_count = read_library_setting(argc, argv, "NUMKONG_SPARSE_SECOND_LENGTH",
                                                        "--sparse-second-length", settings.sparse_second_count, 1,
                                                        counts, "a positive count");
    settings.sparse_intersection_share = read_library_setting(
        argc, argv, "NUMKONG_SPARSE_INTERSECTION", "--sparse-intersection", settings.sparse_intersection_share,
        (std::numeric_limits<double>::min)(), 1, "a number in (0, 1]");
    settings.max_coord_angle_degrees = read_library_setting(argc, argv, "NUMKONG_MAX_COORD_ANGLE", "--max-coord-angle",
                                                            settings.max_coord_angle_degrees, 0, 180,
                                                            "a number in [0, 180]");
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(bench_settings_t const &settings) {
    if (settings.devices) {
        for (device_selection_t const &device : *settings.devices)
            fmt::println("- Device: {}:{}", device_name(device.backend), device.ordinal);
    }
    else fmt::println("- Devices: auto");
    fmt::println("- Seed: {}", settings.seed);
    fmt::println("- Filter: {}", settings.filter.pattern.empty() ? std::string_view("none") : settings.filter.pattern);
    fmt::println("- Warm-up: {}", spell_duration(settings.warmup));
    fmt::println("- Time limit: {}", spell_duration(settings.time_limit));
    fmt::println("- Batch per core: {}", settings.batch_per_core);
    fmt::println("- Dims: {}", settings.dense_dimensions);
    fmt::println("- Dims height: {}", settings.matrix_height);
    fmt::println("- Dims width: {}", settings.matrix_width);
    fmt::println("- Dims depth: {}", settings.matrix_depth);
    fmt::println("- Curved dims: {}", settings.curved_dimensions);
    fmt::println("- Shape points: {}", settings.shape_points);
    fmt::println("- Sparse first length: {}", settings.sparse_first_count);
    fmt::println("- Sparse second length: {}", settings.sparse_second_count);
    fmt::println("- Sparse intersection: {}", settings.sparse_intersection_share);
    fmt::println("- Max coord angle: {}", settings.max_coord_angle_degrees);
}

/** The facts this binary and this machine report: the library version, and the capabilities
 *  compiled in and detected. */
struct machine_t {
    std::array<unsigned, 3> version {NUMKONG_VERSION_MAJOR, NUMKONG_VERSION_MINOR, NUMKONG_VERSION_PATCH};
    nk_capability_t compiled = 0;
    nk_capability_t detected = 0;
};

/** Probes the capabilities @c machine_t reports. */
inline machine_t probe_machine() noexcept {
    machine_t machine;
    machine.compiled = capabilities_compiled_cpu();
    machine.detected = capabilities_detected_cpu();
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

/** One timed attention segment: @c label appends to the row name when set, the rest are per-token
 *  counts, queries at the end of a cache of @c keys with heads grouped over K and V. */
struct attention_shape_t {
    char const *label;
    std::size_t head_count;
    std::size_t key_value_head_count;
    std::size_t depth;
    std::size_t queries;
    std::size_t keys;
};

/** The prefill and decode segments every backend times: 32 query heads over 8 key-value heads of
 *  @c NUMWARS_DIMS_WIDTH dimensions, @c NUMWARS_DIMS_DEPTH queries or one, against
 *  @c NUMWARS_DIMS_HEIGHT keys. */
inline std::vector<attention_shape_t> attention_shapes(bench_settings_t const &env) {
    std::size_t const depth = env.matrix_width, keys = env.matrix_height;
    return {{"prefill", 32, 8, depth, env.matrix_depth, keys}, {"decode", 32, 8, depth, 1, keys}};
}

struct device_backend_t {
    nk::device_t device;
    nk_capability_t capabilities;
    nk::allocator<char> memory;

    template <typename value_type_>
    using allocator = nk::allocator<value_type_>;
};

template <auto get_device_, auto set_device_>
struct device_scope {
    int caller = 0;
    decltype(get_device_(&caller)) status;

    explicit device_scope(int ordinal) noexcept : status(get_device_(&caller)) {
        if (status == 0) status = set_device_(ordinal);
    }
    device_scope(device_scope const &) = delete;
    device_scope &operator=(device_scope const &) = delete;
    ~device_scope() noexcept {
        if (status == 0) set_device_(caller);
    }
};

/** Prints @p title, and returns whether this machine runs @p capabilities; otherwise prints why
 *  it skips. */
inline bool section(bench_settings_t const &, std::string_view title, nk_capability_t capabilities) noexcept {
    nk_capability_t const missing = capabilities & ~capabilities_detected_cpu();
    if (!missing) {
        fmt::println("\n{}:", title);
        return true;
    }
    char missing_names[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_capabilities_name(missing, missing_names, sizeof(missing_names));
    fmt::println("\n{}: skipped, {} not detected", title, missing_names);
    return false;
}

/** Bytes in @p count elements of @p dtype, rounding sub-byte types up to whole bytes. */
inline bytes_t dtype_bytes(nk_dtype_t dtype, std::size_t count) {
    return {nk::divide_round_up(count * nk_dtype_bits(dtype), std::size_t(NUMKONG_BITS_PER_BYTE))};
}

/** The most input sets a benchmark rotates through, a power of two so the ring index is a mask. */
inline constexpr std::size_t input_sets_limit_k = 1024;

/** The number of input sets of @p per_set each to preallocate: @c input_sets_limit_k, fewer past
 *  1 GB, or past 32 MB when 32-bit, and always a power of two. */
inline std::size_t input_sets_count(bytes_t per_set) {
    constexpr std::size_t footprint_bytes = sizeof(std::size_t) == 4 ? std::size_t(32) << 20 : std::size_t(1) << 30;
    return std::bit_floor(
        std::clamp(footprint_bytes / nk::max_of(per_set.value, std::size_t(1)), std::size_t(1), input_sets_limit_k));
}

/** Whether @p argument is a kernel this capability lacks: a null function pointer. */
template <typename argument_type_>
constexpr bool is_missing_kernel(argument_type_ const &argument) noexcept {
    if constexpr (std::is_pointer_v<argument_type_> && std::is_function_v<std::remove_pointer_t<argument_type_>>)
        return argument == nullptr;
    else return std::is_null_pointer_v<argument_type_>;
}

/** Runs @p measure over @p arguments when @p name passes the filter and no kernel among them is
 *  missing, then prints its row. */
template <typename measure_type_, typename... arguments_types_>
void run_benchmark(bench_settings_t const &env, std::string const &name, measure_type_ measure,
                   arguments_types_... arguments) {
    if (!env.filter.selects(name) || (is_missing_kernel(arguments) || ...)) return;
    loop_t loop(env.warmup, env.time_limit);
    measure(loop, env, arguments...);
    print(loop.row(name));
}

/** Factory function to allocate vectors, potentially raising bad-allocs. */
template <typename type_>
[[nodiscard]] nk::vector<type_> make_vector(std::size_t count) {
    auto result = nk::vector<type_>::zeros(count);
#if defined(__cpp_exceptions) && __cpp_exceptions
    if (!result) throw std::bad_alloc();
#else
    if (!result) std::abort();
#endif
    return std::move(result.value);
}

/**
 *  @brief Measures the performance of a @b dense kernel function.
 *
 *  Used by: dot.cpp, spatial.cpp, set.cpp, probability.cpp
 *
 *  @param[in,out] loop The timed loop, which takes the counters.
 *  @param[in] kernel The kernel function to benchmark.
 *  @param[in] dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_dense(loop_t &loop, bench_settings_t const &env, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate inputs: as many vector pairs as `input_sets_count` allows
    std::size_t const vector_count = input_sets_count(dtype_bytes(input_dtype_, 2 * dimensions));
    std::vector<input_vector_t> first_vectors(vector_count), second_vectors(vector_count);
    split_mix_generator_t generator(env.seed);
    for (std::size_t index = 0; index != vector_count; ++index) {
        first_vectors[index] = make_vector<input_t>(dimensions);
        second_vectors[index] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, first_vectors[index].values_data(), first_vectors[index].size_values());
        nk::fill_uniform(generator, second_vectors[index].values_data(), second_vectors[index].size_values());
    }

    // Benchmark loop
    for (std::size_t call : loop) {
        output_t output;
        std::size_t const index = call & (vector_count - 1);
        if (!succeeded(loop, kernel(first_vectors[index].raw_values_data(), second_vectors[index].raw_values_data(),
                                    dimensions, &output.raw_, nullptr)))
            break;
        do_not_optimize(output);
    }

    loop.byte_rate(2.0 * first_vectors[0].size_bytes());
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void run_dense(bench_settings_t const &env, std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(env.dense_dimensions) + ">";
    run_benchmark(env, bench_name, measure_dense<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                  env.dense_dimensions);
}

/** Measures packed Hamming distance computation, used by every bench/cross_*.cpp file. */
template <nk_dtype_t input_dtype_>
void measure_hammings_packed(                                                              //
    loop_t &loop, bench_settings_t const &env,                                             //
    typename nk::type_for<input_dtype_>::type::hammings_pack_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::hammings_pack_kernel_t pack_fn,             //
    typename nk::type_for<input_dtype_>::type::hammings_packed_kernel_t kernel,            //
    std::size_t m, std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::hamming_result_t;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    // Calculate correct strides for binary data (k is in bits)
    nk_size_t values_per_row = nk::divide_round_up(k, 8);
    nk_size_t a_stride = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t b_stride = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t packed_bytes = 0;
    if (!succeeded(loop, packed_size_fn(n, k, &packed_bytes))) return;

    // Preallocate as many input sets as `input_sets_count` allows
    bytes_t const per_set {m * a_stride + n * b_stride + packed_bytes + m * n * sizeof(raw_output_t)};
    std::size_t const sets_count = input_sets_count(per_set);

    struct hamming_set_t {
        nk::vector<input_t> a, b;
        std::vector<char> b_packed;
        nk::vector<output_t> c;
    };
    std::vector<hamming_set_t> sets(sets_count);
    split_mix_generator_t generator(env.seed);
    for (auto &s : sets) {
        s.a = make_vector<input_t>(m * k);
        s.b = make_vector<input_t>(n * k);
        s.b_packed.resize(packed_bytes, 0);
        s.c = make_vector<output_t>(m * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
        nk::fill_uniform(generator, s.b.values_data(), s.b.size_values());
        if (!succeeded(loop, pack_fn(s.b.raw_values_data(), n, k, b_stride, s.b_packed.data(), 0, n, nullptr))) return;
    }

    for (std::size_t call : loop) {
        auto &s = sets[call & (sets_count - 1)];
        do_not_optimize(s.c.raw_values_data());
        if (!succeeded(loop, kernel(s.a.raw_values_data(), s.b_packed.data(), s.c.raw_values_data(), //
                                    m, n, k, a_stride, n * sizeof(raw_output_t), nullptr, 0, nullptr)))
            break;
    }

    loop.rate("scalar-ops", 1.0 * m * n * k);
}

/** Measure symmetric Hamming distance matrix computation. */
template <nk_dtype_t input_dtype_>
void measure_hammings_symmetric(                                                   //
    loop_t &loop, bench_settings_t const &env,                                     //
    typename nk::type_for<input_dtype_>::type::hammings_symmetric_kernel_t kernel, //
    std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::hamming_result_t;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    // Calculate correct strides for binary data (k is in bits)
    nk_size_t input_values_per_row = nk::divide_round_up(k, 8);
    nk_size_t input_stride = input_values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t output_stride = n * sizeof(raw_output_t);

    // Preallocate as many input sets as `input_sets_count` allows
    bytes_t const per_set {n * input_stride + n * n * sizeof(raw_output_t)};
    std::size_t const sets_count = input_sets_count(per_set);

    struct hamming_sym_set_t {
        nk::vector<input_t> a;
        nk::vector<output_t> c;
    };
    std::vector<hamming_sym_set_t> sets(sets_count);
    split_mix_generator_t generator(env.seed);
    for (auto &s : sets) {
        s.a = make_vector<input_t>(n * k);
        s.c = make_vector<output_t>(n * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
    }

    for (std::size_t call : loop) {
        auto &s = sets[call & (sets_count - 1)];
        do_not_optimize(s.c.raw_values_data());
        if (!succeeded(loop, kernel(s.a.raw_values_data(), n, k, input_stride, //
                                    s.c.raw_values_data(), output_stride, 0, n, nullptr, 0, nullptr)))
            break;
    }

    loop.rate("scalar-ops", 1.0 * n * (n + 1) * k / 2);
}

template <nk_dtype_t input_dtype_>
void run_hammings_packed(bench_settings_t const &env, std::string name, //
                         typename nk::type_for<input_dtype_>::type::hammings_pack_size_kernel_t packed_size_fn,
                         typename nk::type_for<input_dtype_>::type::hammings_pack_kernel_t pack_fn,
                         typename nk::type_for<input_dtype_>::type::hammings_packed_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(env.matrix_height) + "x" + std::to_string(env.matrix_width) +
                             "x" + std::to_string(env.matrix_depth) + ">";
    run_benchmark(env, bench_name, measure_hammings_packed<input_dtype_>, packed_size_fn, pack_fn, kernel,
                  env.matrix_height, env.matrix_width, env.matrix_depth);
}

template <nk_dtype_t input_dtype_>
void run_hammings_symmetric(bench_settings_t const &env, std::string name, //
                            typename nk::type_for<input_dtype_>::type::hammings_symmetric_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(env.matrix_height) + "x" + std::to_string(env.matrix_depth) +
                             ">";
    run_benchmark(env, bench_name, measure_hammings_symmetric<input_dtype_>, //
                  kernel, env.matrix_height, env.matrix_depth);
}

/** Measure packed Jaccard distance matrix computation. */
template <nk_dtype_t input_dtype_>
void measure_jaccards_packed(                                                              //
    loop_t &loop, bench_settings_t const &env,                                             //
    typename nk::type_for<input_dtype_>::type::jaccards_pack_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::jaccards_pack_kernel_t pack_fn,             //
    typename nk::type_for<input_dtype_>::type::jaccards_packed_kernel_t kernel,            //
    std::size_t m, std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;

    nk_size_t values_per_row = nk::divide_round_up(k, 8);
    nk_size_t a_stride = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t b_stride = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t packed_bytes = 0;
    if (!succeeded(loop, packed_size_fn(n, k, &packed_bytes))) return;

    bytes_t const per_set {m * a_stride + n * b_stride + packed_bytes + m * n * sizeof(nk_f32_t)};
    std::size_t const sets_count = input_sets_count(per_set);

    struct jaccards_set_t {
        nk::vector<input_t> a, b;
        std::vector<char> b_packed;
        std::vector<nk_f32_t> c;
    };
    std::vector<jaccards_set_t> sets(sets_count);
    split_mix_generator_t generator(env.seed);
    for (auto &s : sets) {
        s.a = make_vector<input_t>(m * k);
        s.b = make_vector<input_t>(n * k);
        s.b_packed.resize(packed_bytes, 0);
        s.c.resize(m * n, 0);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
        nk::fill_uniform(generator, s.b.values_data(), s.b.size_values());
        if (!succeeded(loop, pack_fn(s.b.raw_values_data(), n, k, b_stride, s.b_packed.data(), 0, n, nullptr))) return;
    }

    for (std::size_t call : loop) {
        auto &s = sets[call & (sets_count - 1)];
        do_not_optimize(s.c.data());
        if (!succeeded(loop, kernel(s.a.raw_values_data(), s.b_packed.data(), s.c.data(), //
                                    m, n, k, a_stride, n * sizeof(nk_f32_t), nullptr, 0, nullptr)))
            break;
    }

    loop.rate("scalar-ops", 1.0 * m * n * k);
}

/** Measure symmetric Jaccard distance matrix computation. */
template <nk_dtype_t input_dtype_>
void measure_jaccards_symmetric(                                                   //
    loop_t &loop, bench_settings_t const &env,                                     //
    typename nk::type_for<input_dtype_>::type::jaccards_symmetric_kernel_t kernel, //
    std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;

    nk_size_t input_values_per_row = nk::divide_round_up(k, 8);
    nk_size_t input_stride = input_values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t output_stride = n * sizeof(nk_f32_t);

    bytes_t const per_set {n * input_stride + n * n * sizeof(nk_f32_t)};
    std::size_t const sets_count = input_sets_count(per_set);

    struct jaccards_sym_set_t {
        nk::vector<input_t> a;
        std::vector<nk_f32_t> c;
    };
    std::vector<jaccards_sym_set_t> sets(sets_count);
    split_mix_generator_t generator(env.seed);
    for (auto &s : sets) {
        s.a = make_vector<input_t>(n * k);
        s.c.resize(n * n, 0);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
    }

    for (std::size_t call : loop) {
        auto &s = sets[call & (sets_count - 1)];
        do_not_optimize(s.c.data());
        if (!succeeded(loop, kernel(s.a.raw_values_data(), n, k, input_stride, //
                                    s.c.data(), output_stride, 0, n, nullptr, 0, nullptr)))
            break;
    }

    loop.rate("scalar-ops", 1.0 * n * (n + 1) * k / 2);
}

template <nk_dtype_t input_dtype_>
void run_jaccards_packed(bench_settings_t const &env, std::string name, //
                         typename nk::type_for<input_dtype_>::type::jaccards_pack_size_kernel_t packed_size_fn,
                         typename nk::type_for<input_dtype_>::type::jaccards_pack_kernel_t pack_fn,
                         typename nk::type_for<input_dtype_>::type::jaccards_packed_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(env.matrix_height) + "x" + std::to_string(env.matrix_width) +
                             "x" + std::to_string(env.matrix_depth) + ">";
    run_benchmark(env, bench_name, measure_jaccards_packed<input_dtype_>, packed_size_fn, pack_fn, kernel,
                  env.matrix_height, env.matrix_width, env.matrix_depth);
}

template <nk_dtype_t input_dtype_>
void run_jaccards_symmetric(bench_settings_t const &env, std::string name, //
                            typename nk::type_for<input_dtype_>::type::jaccards_symmetric_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(env.matrix_height) + "x" + std::to_string(env.matrix_depth) +
                             ">";
    run_benchmark(env, bench_name, measure_jaccards_symmetric<input_dtype_>, //
                  kernel, env.matrix_height, env.matrix_depth);
}

/** Benchmark families, defined in separate files. */
void bench_dot(bench_settings_t const &env);
void bench_spatial(bench_settings_t const &env);
void bench_set(bench_settings_t const &env);
void bench_curved(bench_settings_t const &env);
void bench_probability(bench_settings_t const &env);
void bench_each(bench_settings_t const &env);
void bench_trigonometry(bench_settings_t const &env);
void bench_geospatial(bench_settings_t const &env);
void bench_shapes(bench_settings_t const &env);
void bench_sparse(bench_settings_t const &env);
void bench_sparse_dot(bench_settings_t const &env);
void bench_cast(bench_settings_t const &env);
void bench_reduce(bench_settings_t const &env);
void bench_maxsim(bench_settings_t const &env);

/** Forward declarations for cross/batch operations, ISA-family files. */
void bench_cross_serial(bench_settings_t const &env);
void bench_cross_x8664(bench_settings_t const &env);
void bench_cross_arm64(bench_settings_t const &env);
void bench_cross_blas(bench_settings_t const &env);
void bench_cross_riscv64(bench_settings_t const &env);
void bench_cross_ppc64(bench_settings_t const &env);
void bench_cross_wasm(bench_settings_t const &env);
void bench_cross_loongarch64(bench_settings_t const &env);
nk::status_t bench_cross_cuda(bench_settings_t const &env, device_backend_t const &runtime);
nk::status_t bench_cross_rocm(bench_settings_t const &env, device_backend_t const &runtime);
nk::status_t bench_cross_metal(bench_settings_t const &env, device_backend_t const &runtime);

} // namespace ashvardanian::numkong::bench

#endif // NUMKONG_BENCH_HARNESS_HPP
