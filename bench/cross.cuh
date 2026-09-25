/**
 *  @file bench/cross.cuh
 *  @author Ash Vardanian
 *  @date September 23, 2026
 *  @brief Backend-neutral cross-kernel benchmarks: batched dots, angular and euclidean distances,
 *      and ragged attention.
 *
 *  Every driver is a template over the input dtype, its kernels, and a backend owning where
 *  operands live, how a kernel is called, and when its results become readable. The benchmarks
 *  extend the backends of `test/` with how a window of calls is timed: @c host_backend_t under
 *  Google Benchmark's wall time, the CUDA benchmark's under CUDA events.
 *
 *  Input sets rotate, as many as the backend asks for, so a small problem does not time a
 *  cache-resident replay. Matrix rows report `scalar-ops` and, against a double-double reference
 *  over up to 4096 sampled entries of the first set, @c ulp in the output's own precision for
 *  floats or @c exact as the share of exact integer results. Attention rows report @c tokens as
 *  queries and @c flops as the 4 · depth operations per visible query-key pair and head, so a
 *  masked row is not credited for the keys it skips.
 */
#pragma once
#ifndef NUMKONG_BENCH_CROSS_CUH
#define NUMKONG_BENCH_CROSS_CUH

#include <cmath>   // `std::fma`, `std::sqrt`
#include <cstdint> // `std::int64_t`, `std::uint64_t`
#include <cstring> // `std::memcpy`

#include <algorithm>   // `std::min`, `std::max`
#include <array>       // `std::array`
#include <random>      // `std::uniform_int_distribution`
#include <string>      // `std::string`, `std::to_string`
#include <type_traits> // `std::is_integral_v`, `std::is_same_v`
#include <vector>      // `std::vector`

#include "numkong/cast.h" // `nk_cast_serial`

#include "../test/harness.hpp" // `test::host_backend_t`
#include "harness.hpp"

namespace ashvardanian::numkong::bench {

#pragma region Backend Policy

/** Attention visibility: every key, the keys up to the query's own position, or the last 1024 of
 *  those. */
enum class attention_visibility_t { bidirectional_k, causal_k, causal_window_1024_k };

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

/** The shared host backend, timed by Google Benchmark's wall clock. */
struct host_backend_t : test::host_backend_t {

    /** Rotation sets of @p bytes_per_set each: as many as the memory budget holds, up to 1024. */
    std::size_t input_sets(std::size_t bytes_per_set) const noexcept { return bench_input_count(bytes_per_set); }

    /** One shape from the matrix config: keys from its height, head depth from its width, queries
     *  from its depth. */
    static std::vector<attention_shape_t> attention_shapes() {
        return {{"", 8, 8, bench_config.matrix_width, bench_config.matrix_depth, bench_config.matrix_height}};
    }

    /** Calls @p launch once per iteration over rotating sets, returning the call count. */
    template <typename launch_type_>
    std::size_t time(bm::State &state, std::size_t sets_count, launch_type_ &launch) {
        std::size_t calls = 0;
        for (auto _ : state) {
            launch(calls & (sets_count - 1));
            bm::ClobberMemory();
            ++calls;
        }
        return calls;
    }

    /** Wall time needs no registration options. */
    static void configure(bm::internal::Benchmark *) noexcept {}
};

/** A backend copy of @p count values at @p source, left empty when the allocation fails. */
template <typename backend_type_, typename value_type_>
nk::vector<value_type_, typename backend_type_::template allocator<value_type_>> upload(backend_type_ &backend,
                                                                                        value_type_ const *source,
                                                                                        std::size_t count) {
    auto destination = nk::vector<value_type_, typename backend_type_::template allocator<value_type_>>::try_empty(
        count);
    if (!destination.empty()) backend.copy(destination.raw_values_data(), source, destination.size_bytes());
    return destination;
}

/** Launches set 0 once, then times @p launch over rotating sets; returns the call count, or zero
 *  once skipped. */
template <typename backend_type_, typename launch_type_>
std::size_t time_rotating(bm::State &state, backend_type_ &backend, std::size_t sets_count, launch_type_ launch) {
    launch(std::size_t(0));
    if (char const *failure = backend.synchronize()) return state.SkipWithError(failure), 0;
    std::size_t const calls = backend.time(state, sets_count, launch);
    if (char const *failure = backend.synchronize()) return state.SkipWithError(failure), 0;
    return calls;
}

#pragma endregion Backend Policy

#pragma region Accuracy

/** Distance in representable F32 values, with both signs mapped onto one ordered integer line. */
inline std::uint64_t ulp_distance_f32(float first, float second) noexcept {
    if (first == second) return 0;
    std::int32_t first_bits, second_bits;
    std::memcpy(&first_bits, &first, sizeof(first_bits));
    std::memcpy(&second_bits, &second, sizeof(second_bits));
    if (first_bits < 0) first_bits ^= 0x7FFFFFFF;
    if (second_bits < 0) second_bits ^= 0x7FFFFFFF;
    std::int64_t const difference = std::int64_t(first_bits) - std::int64_t(second_bits);
    return std::uint64_t(difference < 0 ? -difference : difference);
}

/** Distance in representable F64 values, with both signs mapped onto one ordered integer line. */
inline std::uint64_t ulp_distance_f64(double first, double second) noexcept {
    if (first == second) return 0;
    std::int64_t first_bits, second_bits;
    std::memcpy(&first_bits, &first, sizeof(first_bits));
    std::memcpy(&second_bits, &second, sizeof(second_bits));
    if (first_bits < 0) first_bits ^= 0x7FFFFFFFFFFFFFFFLL;
    if (second_bits < 0) second_bits ^= 0x7FFFFFFFFFFFFFFFLL;
    return first_bits >= second_bits ? std::uint64_t(first_bits) - std::uint64_t(second_bits)
                                     : std::uint64_t(second_bits) - std::uint64_t(first_bits);
}

/** Σ first[i] × second[i] in double-double: exact products through FMA, error-free sums, one final
 *  rounding. */
inline double dot_double_double(double const *first, double const *second, std::size_t count) noexcept {
    double high = 0, low = 0;
    for (std::size_t index = 0; index != count; ++index) {
        double const product = first[index] * second[index];
        double const product_error = std::fma(first[index], second[index], -product);
        double const sum = high + product, bent = sum - high;
        double const sum_error = (high - (sum - bent)) + (product - bent);
        high = sum, low += sum_error + product_error;
    }
    return high + low;
}

/** What a matrix row computes between two rows, and so which reference it's judged against. */
enum class reference_metric_t { dot_k, angular_k, euclidean_k };

/** Which C entries a kernel writes, and so which ones its accuracy is measured on. */
enum class written_entries_t { full_k, upper_triangle_k, strict_upper_triangle_k };

/** The reference between two decoded rows in double-double, with the serial backends' zero-norm
 *  rule and clamps. */
inline double reference_distance(reference_metric_t metric, double const *first, double const *second,
                                 std::size_t count) noexcept {
    double const dot = dot_double_double(first, second, count);
    if (metric == reference_metric_t::dot_k) return dot;
    double const first_norm = dot_double_double(first, first, count);
    double const second_norm = dot_double_double(second, second, count);
    if (metric == reference_metric_t::euclidean_k) return std::sqrt(std::max(0.0, first_norm + second_norm - 2 * dot));
    if (!(first_norm > 0 && second_norm > 0)) return dot == 0 ? 0 : 1;
    return std::max(0.0, 1 - dot / std::sqrt(first_norm) / std::sqrt(second_norm));
}

/** Random A and B of @p height and @p width rows of @p depth dimensions under one seed: A at @p
 *  backend_type_'s row stride and zero past each row's end, B dense. A matrix whose allocation
 *  failed comes back empty. */
template <nk_dtype_t input_dtype_, typename backend_type_>
std::array<nk::tensor<typename nk::type_for<input_dtype_>::type,
                      nk::aligned_allocator<typename nk::type_for<input_dtype_>::type>, 2>,
           2>
random_matrices(std::size_t height, std::size_t width, std::size_t depth) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using matrix_t = nk::tensor<input_t, nk::aligned_allocator<input_t>, 2>;
    std::size_t const dimensions_per_value = nk::dimensions_per_value<input_t>();
    std::size_t const row_values = nk::divide_round_up(depth, dimensions_per_value);
    std::size_t const a_stride_values = backend_type_::row_stride(row_values * sizeof(input_t)) / sizeof(input_t);
    std::array<matrix_t, 2> matrices {matrix_t::try_zeros({height, a_stride_values * dimensions_per_value}),
                                      matrix_t::try_zeros({width, row_values * dimensions_per_value})};
    auto generator = make_random_engine();
    for (matrix_t &matrix : matrices) {
        if (matrix.empty()) continue;
        std::size_t const stride_values = matrix.stride_bytes(0) / sizeof(input_t);
        for (std::size_t row = 0; row != matrix.extent(0); ++row)
            nk::fill_uniform(generator, matrix.data() + row * stride_values, row_values);
    }
    return matrices;
}

/** Accuracy of @p c against the reference over its written entries, or 4096 sampled ones: mean ULP
 *  in the output's precision for floats, share of exact matches for integers. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_>
double sampled_accuracy(backend_type_ &backend,
                        nk::vector<output_type_, typename backend_type_::template allocator<output_type_>> const &c,
                        nk::tensor_view<typename nk::type_for<input_dtype_>::type, 2> first,
                        nk::tensor_view<typename nk::type_for<input_dtype_>::type, 2> second, std::size_t depth,
                        reference_metric_t metric, written_entries_t written) {
    using output_raw_t = typename output_type_::raw_t;
    std::size_t const entries = first.extent(0) * second.extent(0), samples = std::min(entries, std::size_t(4096));
    std::vector<output_raw_t> result(entries);
    backend.copy(result.data(), c.raw_values_data(), entries * sizeof(output_raw_t));
    std::vector<double> first_decoded(depth), second_decoded(depth);
    std::mt19937 generator(bench_config.seed);
    std::uniform_int_distribution<std::size_t> entry_distribution(0, entries - 1);
    double score_sum = 0;
    std::size_t measured = 0;
    for (std::size_t sample = 0; sample != samples; ++sample) {
        std::size_t const entry = samples == entries ? sample : entry_distribution(generator);
        std::size_t const row = entry / second.extent(0), column = entry % second.extent(0);
        if (written == written_entries_t::upper_triangle_k && column < row) continue;
        if (written == written_entries_t::strict_upper_triangle_k && column <= row) continue;
        nk_cast_serial(first.byte_data() + row * first.stride_bytes(0), input_dtype_, depth, first_decoded.data(),
                       nk_f64_k);
        nk_cast_serial(second.byte_data() + column * second.stride_bytes(0), input_dtype_, depth, second_decoded.data(),
                       nk_f64_k);
        double const expected = reference_distance(metric, first_decoded.data(), second_decoded.data(), depth);
        if constexpr (std::is_integral_v<output_raw_t>) score_sum += double(double(result[entry]) == expected);
        else if constexpr (sizeof(output_raw_t) == 8) score_sum += double(ulp_distance_f64(result[entry], expected));
        else score_sum += double(ulp_distance_f32(result[entry], float(expected)));
        ++measured;
    }
    return score_sum / double(std::max(measured, std::size_t(1)));
}

/** Fills `scalar-ops` and @p calls, and @c exact for integer outputs or @c ulp for floats. */
template <typename output_type_>
void report_matrix(bm::State &state, std::size_t calls, double scalar_ops_per_call, double score) {
    state.counters["scalar-ops"] = bm::Counter(double(calls) * scalar_ops_per_call, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(double(calls), bm::Counter::kIsRate);
    state.counters[std::is_integral_v<typename output_type_::raw_t> ? "exact" : "ulp"] = bm::Counter(score);
}

#pragma endregion Accuracy

#pragma region Matrices

/** A at the stride it was generated with, B as the pack kernel laid it out or dense rows, and C as
 *  the output, for one rotation slot in @p backend_type_ memory. */
template <typename backend_type_, typename output_type_>
struct matrix_set {

    /** Bytes in backend memory. */
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    /** Outputs in backend memory. */
    using outputs_t = nk::vector<output_type_, typename backend_type_::template allocator<output_type_>>;

    bytes_t a;
    bytes_t b;
    outputs_t c;
};

/** Times a packed-B kernel, C = A × Bᵀ or a distance over the same tile, against its @p metric
 *  reference. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void measure_packed(bm::State &state, reference_metric_t metric, pack_size_kernel_type_ packed_size_fn,
                    pack_kernel_type_ pack_fn, kernel_type_ kernel, std::size_t height, std::size_t width,
                    std::size_t depth) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using set_t = matrix_set<backend_type_, output_type_>;
    backend_type_ backend;
    auto const [a, b] = random_matrices<input_dtype_, backend_type_>(height, width, depth);
    if (a.empty() || b.empty()) return state.SkipWithError("input allocation failed");
    std::size_t const a_stride = a.stride_bytes(0), row_bytes = b.stride_bytes(0), a_bytes = height * a_stride;
    std::size_t const packed_bytes = packed_size_fn(width, depth);
    auto const b_uploaded = upload(backend, b.data(), b.numel());
    if (b_uploaded.empty()) return state.SkipWithError("B allocation failed");

    // One B upload, packed into every set, since the sets differ only in where they live.
    std::vector<set_t> sets(backend.input_sets(a_bytes + packed_bytes + height * width * sizeof(output_type_)));
    for (set_t &set : sets) {
        set = {set_t::bytes_t::try_empty(a_bytes), set_t::bytes_t::try_empty(packed_bytes),
               set_t::outputs_t::try_empty(height * width)};
        if (set.a.empty() || set.b.empty() || set.c.empty()) return state.SkipWithError("set allocation failed");
        backend.copy(set.a.raw_values_data(), a.data(), a_bytes);
        backend.zero(set.b.raw_values_data(), packed_bytes), backend.zero(set.c.raw_values_data(), set.c.size_bytes());
        backend.call(pack_fn, b_uploaded.raw_values_data(), width, depth, row_bytes, set.b.raw_values_data(),
                     std::size_t(0), width);
    }
    std::size_t const calls = time_rotating(state, backend, sets.size(), [&](std::size_t index) {
        set_t &set = sets[index];
        backend.call(kernel, reinterpret_cast<typename input_t::raw_t const *>(set.a.raw_values_data()),
                     static_cast<void const *>(set.b.raw_values_data()), set.c.raw_values_data(), height, width, depth,
                     a_stride, width * sizeof(output_type_));
    });
    if (!calls) return;
    double const score = sampled_accuracy<input_dtype_, output_type_>(backend, sets[0].c, a.view(), b.view(), depth,
                                                                      metric, written_entries_t::full_k);
    report_matrix<output_type_>(state, calls, 2.0 * height * width * depth, score);
}

/** Times a symmetric kernel over A × Aᵀ, judged on the upper triangle: with the diagonal for dots,
 *  without it for distances. `scalar-ops` counts the triangle's height · (height + 1) · depth. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_, typename kernel_type_>
void measure_symmetric(bm::State &state, reference_metric_t metric, kernel_type_ kernel, std::size_t height,
                       std::size_t depth) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using set_t = matrix_set<backend_type_, output_type_>;
    backend_type_ backend;
    auto const matrices = random_matrices<input_dtype_, backend_type_>(height, 0, depth);
    auto const &a = matrices[0];
    if (a.empty()) return state.SkipWithError("input allocation failed");
    std::size_t const a_stride = a.stride_bytes(0), a_bytes = height * a_stride;
    std::vector<set_t> sets(backend.input_sets(a_bytes + height * height * sizeof(output_type_)));
    for (set_t &set : sets) {
        set.a = set_t::bytes_t::try_empty(a_bytes), set.c = set_t::outputs_t::try_empty(height * height);
        if (set.a.empty() || set.c.empty()) return state.SkipWithError("set allocation failed");
        backend.copy(set.a.raw_values_data(), a.data(), a_bytes);
        backend.zero(set.c.raw_values_data(), set.c.size_bytes());
    }
    std::size_t const calls = time_rotating(state, backend, sets.size(), [&](std::size_t index) {
        set_t &set = sets[index];
        backend.call(kernel, reinterpret_cast<typename input_t::raw_t const *>(set.a.raw_values_data()), height, depth,
                     a_stride, set.c.raw_values_data(), height * sizeof(output_type_), std::size_t(0), height);
    });
    if (!calls) return;
    written_entries_t const written = metric == reference_metric_t::dot_k ? written_entries_t::upper_triangle_k
                                                                          : written_entries_t::strict_upper_triangle_k;
    double const score = sampled_accuracy<input_dtype_, output_type_>(backend, sets[0].c, a.view(), a.view(), depth,
                                                                      metric, written);
    report_matrix<output_type_>(state, calls, 1.0 * height * (height + 1) * depth, score);
}

/** The `<height x width x depth>` suffix of a matrix row's name. */
inline std::string matrix_row_name(std::string const &name, std::size_t height, std::size_t width, std::size_t depth) {
    return name + "<" + std::to_string(height) + "x" + std::to_string(width) + "x" + std::to_string(depth) + ">";
}

/** Registers a packed-B row over the configured matrix shape. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void register_packed(std::string const &name, reference_metric_t metric, pack_size_kernel_type_ packed_size_fn,
                     pack_kernel_type_ pack_fn, kernel_type_ kernel) {
    std::size_t const height = bench_config.matrix_height, width = bench_config.matrix_width,
                      depth = bench_config.matrix_depth;
    backend_type_::configure(
        bm::RegisterBenchmark(matrix_row_name(name, height, width, depth).c_str(),
                              measure_packed<input_dtype_, output_type_, backend_type_, pack_size_kernel_type_,
                                             pack_kernel_type_, kernel_type_>,
                              metric, packed_size_fn, pack_fn, kernel, height, width, depth));
}

/** Registers a symmetric row over @c NUMWARS_DIMS_HEIGHT vectors of @c NUMWARS_DIMS_DEPTH
 *  dimensions. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_, typename kernel_type_>
void register_symmetric(std::string const &name, reference_metric_t metric, kernel_type_ kernel) {
    std::size_t const height = bench_config.matrix_height, depth = bench_config.matrix_depth;
    std::string const row_name = name + "<" + std::to_string(height) + "x" + std::to_string(depth) + ">";
    backend_type_::configure(bm::RegisterBenchmark(
        row_name.c_str(), measure_symmetric<input_dtype_, output_type_, backend_type_, kernel_type_>, metric, kernel,
        height, depth));
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void run_dots_packed(std::string const &name, pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                     kernel_type_ kernel) {
    register_packed<input_dtype_, typename nk::type_for<input_dtype_>::type::dot_result_t, backend_type_>(
        name, reference_metric_t::dot_k, packed_size_fn, pack_fn, kernel);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void run_angulars_packed(std::string const &name, pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                         kernel_type_ kernel) {
    register_packed<input_dtype_, typename nk::type_for<input_dtype_>::type::angular_result_t, backend_type_>(
        name, reference_metric_t::angular_k, packed_size_fn, pack_fn, kernel);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void run_euclideans_packed(std::string const &name, pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                           kernel_type_ kernel) {
    register_packed<input_dtype_, typename nk::type_for<input_dtype_>::type::euclidean_result_t, backend_type_>(
        name, reference_metric_t::euclidean_k, packed_size_fn, pack_fn, kernel);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename kernel_type_>
void run_dots_symmetric(std::string const &name, kernel_type_ kernel) {
    register_symmetric<input_dtype_, typename nk::type_for<input_dtype_>::type::dot_result_t, backend_type_>(
        name, reference_metric_t::dot_k, kernel);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename kernel_type_>
void run_angulars_symmetric(std::string const &name, kernel_type_ kernel) {
    register_symmetric<input_dtype_, typename nk::type_for<input_dtype_>::type::angular_result_t, backend_type_>(
        name, reference_metric_t::angular_k, kernel);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename kernel_type_>
void run_euclideans_symmetric(std::string const &name, kernel_type_ kernel) {
    register_symmetric<input_dtype_, typename nk::type_for<input_dtype_>::type::euclidean_result_t, backend_type_>(
        name, reference_metric_t::euclidean_k, kernel);
}

#pragma endregion Matrices

#pragma region Attention

/** Visible keys including the query itself, @c NUMKONG_SIZE_MAX when unbounded. */
inline nk_size_t attention_window(attention_visibility_t visibility) noexcept {
    return visibility == attention_visibility_t::causal_window_1024_k ? 1024 : NUMKONG_SIZE_MAX;
}

/** Whether the 1024-key window hides keys a plain causal row of @p shape would see. */
inline bool attention_window_clips(attention_shape_t shape) noexcept {
    return shape.keys > attention_window(attention_visibility_t::causal_window_1024_k);
}

/** A row's name: backend, window, shape label, then heads, queries, keys and depth. */
inline std::string attention_row_name(std::string const &name, attention_visibility_t visibility,
                                      attention_shape_t shape) {
    std::string row = name;
    if (visibility == attention_visibility_t::causal_window_1024_k) row += "_window1024";
    if (*shape.label) row += std::string("_") + shape.label;
    return row + "<" + std::to_string(shape.head_count) + "h_" + std::to_string(shape.queries) + "q_" +
           std::to_string(shape.keys) + "kv_" + std::to_string(shape.depth) + "d>";
}

/** Query-key pairs per head that the queries at the end of the keys of @p shape see. */
inline double attention_visible_pairs(attention_visibility_t visibility, attention_shape_t shape) noexcept {
    if (visibility == attention_visibility_t::bidirectional_k) return double(shape.queries) * double(shape.keys);
    std::int64_t const diagonal_offset = std::int64_t(shape.keys) - std::int64_t(shape.queries);
    std::int64_t const window = std::int64_t(std::min<nk_size_t>(attention_window(visibility), shape.keys));
    double pairs = 0;
    for (std::size_t row = 0; row != shape.queries; ++row) {
        std::int64_t const position = std::int64_t(row) + diagonal_offset;
        std::int64_t const first = std::max<std::int64_t>(0, position - window + 1);
        pairs += double(std::max<std::int64_t>(0, std::min<std::int64_t>(position, shape.keys - 1) - first + 1));
    }
    return pairs;
}

/** Fills @c tokens with queries per second, @c flops with the visible 4 · depth work per pair and
 *  head, and @p calls. */
inline void report_attention(bm::State &state, std::size_t calls, attention_visibility_t visibility,
                             attention_shape_t shape) {
    double const flops_per_call = 4.0 * double(shape.depth * shape.head_count) *
                                  attention_visible_pairs(visibility, shape);
    state.counters["tokens"] = bm::Counter(double(calls) * double(shape.queries), bm::Counter::kIsRate);
    state.counters["flops"] = bm::Counter(double(calls) * flops_per_call, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(double(calls), bm::Counter::kIsRate);
}

/** Queries, keys and values of @p shape in [-1, 1], or [-32, 32] for integers, identical for every
 *  backend. */
template <nk_dtype_t input_dtype_>
std::array<nk::vector<typename nk::type_for<input_dtype_>::type>, 3> random_attention(attention_shape_t shape) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    auto generator = make_random_engine();
    int const range = nk::is_integral_dtype<input_t>() ? 32 : 1;
    std::size_t const key_values = shape.keys * shape.key_value_head_count * shape.depth;
    std::array<nk::vector<input_t>, 3> inputs {make_vector<input_t>(shape.queries * shape.head_count * shape.depth),
                                               make_vector<input_t>(key_values), make_vector<input_t>(key_values)};
    for (nk::vector<input_t> &rows : inputs)
        nk::fill_uniform(generator, rows.values_data(), rows.size_values(), -range, range);
    return inputs;
}

/** Packs the keys and values of the one segment of @p shape, whose @p directory holds key offsets
 *  then lengths. */
template <typename backend_type_, typename pack_kernel_type_, typename raw_type_>
void attention_pack(backend_type_ &backend, pack_kernel_type_ pack_fn, attention_shape_t shape, raw_type_ const *keys,
                    raw_type_ const *values, nk_u32_t const *directory, void *packed) {
    std::size_t const key_stride = shape.key_value_head_count * shape.depth * sizeof(raw_type_);
    backend.call(pack_fn, keys, values, shape.key_value_head_count, shape.depth, directory, directory + 1,
                 std::size_t(1), key_stride, key_stride, packed, std::size_t(0), shape.key_value_head_count);
}

/** Runs @p attention_fn over the one segment of @p shape under @p visibility_, queries aligned to
 *  the keys' end. */
template <attention_visibility_t visibility_, typename backend_type_, typename attention_kernel_type_,
          typename raw_type_>
void attend(backend_type_ &backend, attention_kernel_type_ attention_fn, attention_shape_t shape,
            nk_u32_t const *directory, raw_type_ const *queries, void const *packed, nk_f32_t *output) {
    std::size_t const query_stride = shape.head_count * shape.depth * sizeof(raw_type_);
    std::size_t const output_stride = shape.head_count * shape.depth * sizeof(nk_f32_t);
    nk_f32_t const scale = 1.0f / std::sqrt(float(shape.depth));
    if constexpr (visibility_ == attention_visibility_t::bidirectional_k)
        backend.call(attention_fn, queries, packed, output, shape.head_count, shape.key_value_head_count, shape.depth,
                     directory + 2, query_stride, output_stride, scale, std::size_t(0), shape.head_count);
    else
        backend.call(attention_fn, queries, packed, output, shape.head_count, shape.key_value_head_count, shape.depth,
                     directory + 2, query_stride, output_stride, scale, nk_i64_t(shape.keys) - nk_i64_t(shape.queries),
                     attention_window(visibility_), std::size_t(0), shape.head_count);
}

/** Queries, the packed key/value cache, and one F32-row-per-query output, for one rotation slot in
 *  @p backend_type_ memory. */
template <typename backend_type_, typename input_type_>
struct attention_set {

    /** Queries in backend memory. */
    using queries_t = nk::vector<input_type_, typename backend_type_::template allocator<input_type_>>;

    /** Bytes in backend memory. */
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    /** Outputs in backend memory. */
    using outputs_t = nk::vector<nk::f32_t, typename backend_type_::template allocator<nk::f32_t>>;

    queries_t queries;
    bytes_t packed;
    outputs_t output;
};

/** Times one segment of @p shape under @p visibility_: pack once per set, then attention calls over
 *  rotating sets. */
template <nk_dtype_t input_dtype_, attention_visibility_t visibility_, typename backend_type_,
          typename pack_size_kernel_type_, typename pack_kernel_type_, typename attention_kernel_type_>
void measure_attention(bm::State &state, pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                       attention_kernel_type_ attention_fn, attention_shape_t shape) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using set_t = attention_set<backend_type_, input_t>;
    backend_type_ backend;
    auto const [queries, keys, values] = random_attention<input_dtype_>(shape);
    auto const keys_uploaded = upload(backend, keys.values_data(), keys.size());
    auto const values_uploaded = upload(backend, values.values_data(), values.size());
    nk_u32_t const offsets[4] = {0, nk_u32_t(shape.keys), 0, nk_u32_t(shape.queries)};
    auto const directory = upload(backend, offsets, 4);
    if (keys_uploaded.empty() || values_uploaded.empty() || directory.empty())
        return state.SkipWithError("input allocation failed");
    nk_u32_t const lengths[1] = {nk_u32_t(shape.keys)};
    std::size_t const packed_bytes = packed_size_fn(shape.key_value_head_count, shape.depth, lengths, 1);
    std::size_t const output_count = shape.queries * shape.head_count * shape.depth;
    std::vector<set_t> sets(backend.input_sets(queries.size_bytes() + packed_bytes + output_count * sizeof(nk_f32_t)));
    for (set_t &set : sets) {
        set = {upload(backend, queries.values_data(), queries.size()), set_t::bytes_t::try_empty(packed_bytes),
               set_t::outputs_t::try_empty(output_count)};
        if (set.queries.empty() || set.packed.empty() || set.output.empty())
            return state.SkipWithError("set allocation failed");
        backend.zero(set.packed.raw_values_data(), packed_bytes);
        backend.zero(set.output.raw_values_data(), set.output.size_bytes());
        attention_pack(backend, pack_fn, shape, keys_uploaded.raw_values_data(), values_uploaded.raw_values_data(),
                       directory.raw_values_data(), set.packed.raw_values_data());
    }
    std::size_t const calls = time_rotating(state, backend, sets.size(), [&](std::size_t index) {
        set_t &set = sets[index];
        attend<visibility_>(backend, attention_fn, shape, directory.raw_values_data(), set.queries.raw_values_data(),
                            set.packed.raw_values_data(), set.output.raw_values_data());
    });
    if (calls) report_attention(state, calls, visibility_, shape);
}

/** Registers one attention row of @p shape under @p visibility_. */
template <nk_dtype_t input_dtype_, attention_visibility_t visibility_, typename backend_type_,
          typename pack_size_kernel_type_, typename pack_kernel_type_, typename attention_kernel_type_>
void register_attention(std::string const &name, pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                        attention_kernel_type_ attention_fn, attention_shape_t shape) {
    backend_type_::configure(
        bm::RegisterBenchmark(attention_row_name(name, visibility_, shape).c_str(),
                              measure_attention<input_dtype_, visibility_, backend_type_, pack_size_kernel_type_,
                                                pack_kernel_type_, attention_kernel_type_>,
                              packed_size_fn, pack_fn, attention_fn, shape));
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename attention_kernel_type_>
void run_attention_bidirectional(std::string const &name, pack_size_kernel_type_ packed_size_fn,
                                 pack_kernel_type_ pack_fn, attention_kernel_type_ attention_fn) {
    for (attention_shape_t const shape : backend_type_::attention_shapes())
        register_attention<input_dtype_, attention_visibility_t::bidirectional_k, backend_type_>(
            name, packed_size_fn, pack_fn, attention_fn, shape);
}

/** Causal rows per backend shape, plus a 1024-key window wherever it hides keys. */
template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename attention_kernel_type_>
void run_attention_causal(std::string const &name, pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                          attention_kernel_type_ attention_fn) {
    for (attention_shape_t const shape : backend_type_::attention_shapes()) {
        register_attention<input_dtype_, attention_visibility_t::causal_k, backend_type_>(name, packed_size_fn, pack_fn,
                                                                                          attention_fn, shape);
        if (attention_window_clips(shape))
            register_attention<input_dtype_, attention_visibility_t::causal_window_1024_k, backend_type_>(
                name, packed_size_fn, pack_fn, attention_fn, shape);
    }
}

#pragma endregion Attention

} // namespace ashvardanian::numkong::bench

#endif // NUMKONG_BENCH_CROSS_CUH
