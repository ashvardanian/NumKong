/**
 *  @file bench/cross.hpp
 *  @author Ash Vardanian
 *  @date September 23, 2026
 *  @brief Backend-neutral cross-kernel benchmarks: batched dots, angular and euclidean distances,
 *      ragged attention, and the element-wise kernels over token rows.
 *
 *  Every driver is a template over the input dtype, its kernels, and a backend value owning where
 *  operands live, how a kernel is called, when its results become readable, and how a window of
 *  calls is timed: @c host_backend_t under the loop's wall time, CUDA under events around each
 *  window of launches, and Metal under a wall clock around each window's synchronization.
 *
 *  Input sets rotate, as many as the backend asks for, so a small problem does not time a
 *  cache-resident replay. Matrix rows report `scalar-ops` and, against compensated F64 calculations
 *  over up to 4096 sampled entries of the first set, @c ulp in the output's own precision for
 *  floats or @c exact as the share of exact integer results. Attention rows report @c tokens as
 *  queries and `scalar-ops` as the 4 · depth operations per visible query-key pair and head, so a
 *  masked row is not credited for the keys it skips. Token rows report @c bytes as the inputs a
 *  single pass reads.
 */
#pragma once
#ifndef NUMKONG_BENCH_CROSS_HPP
#define NUMKONG_BENCH_CROSS_HPP

#include <cmath>   // `std::fma`, `std::sqrt`
#include <cstdint> // `std::int64_t`, `std::uint64_t`
#include <cstring> // `std::memcpy`, `std::memset`

#include <algorithm>   // `std::min`, `std::max`
#include <array>       // `std::array`
#include <random>      // `std::uniform_int_distribution`
#include <string>      // `std::string`, `std::to_string`
#include <type_traits> // `std::is_integral_v`, `std::is_same_v`
#include <vector>      // `std::vector`

#include "numkong/cast.h" // `nk_cast_serial`

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

/** Runs CPU kernels in place over host memory, timed by the loop's wall clock. */
struct host_backend_t {

    /** The allocator every kernel operand comes from. */
    template <typename value_type_>
    using allocator = nk::aligned_allocator<value_type_>;

    /** Row stride for @p row_bytes: exactly one row, keeping the tightest stride covered. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return row_bytes; }

    /** Rotation sets of @p per_set each, as many as @c input_sets_count allows. */
    std::size_t input_sets(bytes_t per_set) const noexcept { return input_sets_count(per_set); }

    /** Rows a token-row benchmark batches: one, as a row of the configured length already fills the
     *  caches. */
    static std::size_t token_rows(environment_t const &) noexcept { return 1; }

    /** One shape from the matrix config: keys from its height, head depth from its width, queries
     *  from its depth. */
    static std::vector<attention_shape_t> attention_shapes(environment_t const &env) {
        return {{"", 8, 8, env.settings.matrix_width, env.settings.matrix_depth, env.settings.matrix_height}};
    }

    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        std::memcpy(destination, source, bytes);
        return nk_success_k;
    }

    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        std::memset(destination, 0, bytes);
        return nk_success_k;
    }

    template <typename kernel_type_, typename... arguments_types_>
    nk_status_t call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        return kernel(arguments..., nullptr);
    }

    nk_status_t synchronize() noexcept { return nk_success_k; }

    template <typename launch_type_>
    nk_status_t time(loop_t &loop, std::size_t sets_count, launch_type_ &launch) {
        for (std::size_t call : loop)
            if (nk_status_t const status = launch(call & (sets_count - 1)); status != nk_success_k) return status;
        return nk_success_k;
    }
};

/** The allocator @p backend hands out @p value_type_ from: stateless, unless the backend's memory
 *  belongs to a context it holds and it overloads this. */
template <typename value_type_, typename backend_type_>
typename backend_type_::template allocator<value_type_> allocator_of(backend_type_ const &backend) noexcept {
    if constexpr (std::is_base_of_v<device_backend_t, backend_type_>) return nk::allocator<value_type_>(backend.memory);
    else return {};
}

/** A backend copy of @p count values at @p source, returning the allocation or copy status. */
template <typename backend_type_, typename value_type_>
nk::expected<nk::vector<value_type_, typename backend_type_::template allocator<value_type_>>> upload(
    backend_type_ &backend, value_type_ const *source, std::size_t count) {
    auto destination = nk::vector<value_type_, typename backend_type_::template allocator<value_type_>>::uninitialized(
        count, allocator_of<value_type_>(backend));
    if (nk::failed(destination.status)) return destination;
    if (nk_status_t const status = backend.copy(destination.value.raw_values_data(), source,
                                                destination.value.size_bytes());
        status != nk_success_k)
        return {{}, static_cast<nk::status_t>(status)};
    return destination;
}

/** Launches set 0 once, then times @p launch over rotating sets; returns false once skipped. */
template <typename backend_type_, typename launch_type_>
bool time_rotating(loop_t &loop, backend_type_ &backend, std::size_t sets_count, launch_type_ launch) {
    nk_status_t const launch_status = launch(std::size_t(0));
    nk_status_t const ready_status = backend.synchronize();
    if (!succeeded(loop, launch_status) || !succeeded(loop, ready_status)) return false;
    nk_status_t const time_status = backend.time(loop, sets_count, launch);
    if (time_status != nk_success_k) { [[maybe_unused]] nk_status_t const drained = backend.synchronize(); }
    return succeeded(loop, time_status);
}

/** The `<columns>` suffix of a one-row benchmark name, `<rows x columns>` for more rows. */
inline std::string token_rows_name(std::string const &name, std::size_t rows, std::size_t columns) {
    return name + "<" + (rows == 1 ? "" : std::to_string(rows) + "x") + std::to_string(columns) + ">";
}

/** Uploads seeded random values, returning the allocation or copy status. */
template <nk_dtype_t input_dtype_, typename backend_type_>
nk::expected<nk::vector<typename nk::type_for<input_dtype_>::type,
                        typename backend_type_::template allocator<typename nk::type_for<input_dtype_>::type>>>
random_upload(backend_type_ &backend, std::size_t count, seed_t seed) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    auto host = make_vector<input_t>(count);
    std::mt19937 generator(seed.value);
    nk::fill_uniform(generator, host.values_data(), host.size_values());
    return upload(backend, host.values_data(), count);
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

/** Dot product with FMA residuals and compensated summation for accuracy checks. */
inline double dot_compensated_f64(double const *first, double const *second, std::size_t count) noexcept {
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

using expected_metric_t = double (*)(double const *, double const *, std::size_t) noexcept;

/** Which C entries a kernel writes, and so which ones its accuracy is measured on. */
enum class written_entries_t { full_k, upper_triangle_k, strict_upper_triangle_k };

inline double angular_compensated_f64(double const *first, double const *second, std::size_t count) noexcept {
    double const dot = dot_compensated_f64(first, second, count);
    double const first_norm = dot_compensated_f64(first, first, count);
    double const second_norm = dot_compensated_f64(second, second, count);
    if (!(first_norm > 0 && second_norm > 0)) return dot == 0 ? 0 : 1;
    return std::max(0.0, 1 - dot / std::sqrt(first_norm) / std::sqrt(second_norm));
}

inline double euclidean_compensated_f64(double const *first, double const *second, std::size_t count) noexcept {
    double const dot = dot_compensated_f64(first, second, count);
    double const first_norm = dot_compensated_f64(first, first, count);
    double const second_norm = dot_compensated_f64(second, second, count);
    return std::sqrt(std::max(0.0, first_norm + second_norm - 2 * dot));
}

/** Random A of @p rows and B of @p columns rows of @p depth dimensions under @p seed: A at @p
 *  backend_type_'s row stride and zero past each row's end, B dense. A matrix whose allocation
 *  failed comes back empty. */
template <nk_dtype_t input_dtype_, typename backend_type_>
std::array<nk::tensor<typename nk::type_for<input_dtype_>::type,
                      nk::aligned_allocator<typename nk::type_for<input_dtype_>::type>, 2>,
           2>
random_matrices(seed_t seed, std::size_t rows, std::size_t columns, std::size_t depth) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using matrix_t = nk::tensor<input_t, nk::aligned_allocator<input_t>, 2>;
    std::size_t const dimensions_per_value = nk::dimensions_per_value<input_t>();
    std::size_t const row_values = nk::divide_round_up(depth, dimensions_per_value);
    std::size_t const a_stride_values = backend_type_::row_stride(row_values * sizeof(input_t)) / sizeof(input_t);
    std::array<matrix_t, 2> matrices {matrix_t::zeros({rows, a_stride_values * dimensions_per_value}).value,
                                      matrix_t::zeros({columns, row_values * dimensions_per_value}).value};
    std::mt19937 generator(seed.value);
    for (matrix_t &matrix : matrices) {
        if (matrix.empty()) continue;
        std::size_t const stride_values = matrix.stride_bytes(0) / sizeof(input_t);
        for (std::size_t row = 0; row != matrix.extent(0); ++row)
            nk::fill_uniform(generator, matrix.data() + row * stride_values, row_values);
    }
    return matrices;
}

/** Accuracy of @p c against the reference over its written entries, or 4096 sampled under @p seed:
 *  mean ULP in the output's precision for floats, share of exact matches for integers. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_>
nk::expected<double> sampled_accuracy(
    backend_type_ &backend, nk::vector<output_type_, typename backend_type_::template allocator<output_type_>> const &c,
    nk::tensor_view<typename nk::type_for<input_dtype_>::type, 2> first,
    nk::tensor_view<typename nk::type_for<input_dtype_>::type, 2> second, std::size_t depth,
    expected_metric_t compute_expected, written_entries_t written, seed_t seed) {
    using output_raw_t = typename output_type_::raw_t;
    std::size_t const entries = first.extent(0) * second.extent(0), samples = std::min(entries, std::size_t(4096));
    std::vector<output_raw_t> result(entries);
    if (nk_status_t const status = backend.copy(result.data(), c.raw_values_data(), entries * sizeof(output_raw_t));
        status != nk_success_k)
        return {0, static_cast<nk::status_t>(status)};
    std::vector<double> first_decoded(depth), second_decoded(depth);
    std::mt19937 generator(seed.value);
    std::uniform_int_distribution<std::size_t> entry_distribution(0, entries - 1);
    double score_sum = 0;
    std::size_t measured = 0;
    for (std::size_t sample = 0; sample != samples; ++sample) {
        std::size_t const entry = samples == entries ? sample : entry_distribution(generator);
        std::size_t const row = entry / second.extent(0), column = entry % second.extent(0);
        if (written == written_entries_t::upper_triangle_k && column < row) continue;
        if (written == written_entries_t::strict_upper_triangle_k && column <= row) continue;
        if (nk_cast_serial(first.byte_data() + row * first.stride_bytes(0), input_dtype_, first_decoded.data(),
                           nk_f64_k, depth, nullptr) != nk_success_k ||
            nk_cast_serial(second.byte_data() + column * second.stride_bytes(0), input_dtype_, second_decoded.data(),
                           nk_f64_k, depth, nullptr) != nk_success_k)
            continue;
        double const expected = compute_expected(first_decoded.data(), second_decoded.data(), depth);
        if constexpr (std::is_integral_v<output_raw_t>) score_sum += double(double(result[entry]) == expected);
        else if constexpr (sizeof(output_raw_t) == 8) score_sum += double(ulp_distance_f64(result[entry], expected));
        else score_sum += double(ulp_distance_f32(result[entry], float(expected)));
        ++measured;
    }
    return {score_sum / double(std::max(measured, std::size_t(1))), nk::status_t::success_k};
}

/** Fills `scalar-ops`, and @c exact for integer outputs or @c ulp for floats. */
template <typename output_type_>
void report_matrix(loop_t &loop, double scalar_ops_per_call, double score) {
    loop.rate("scalar-ops", scalar_ops_per_call);
    loop.counter(std::is_integral_v<typename output_type_::raw_t> ? "exact" : "ulp", score);
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

template <typename input_type_>
struct cref_of_ {
    struct none {};
    using type = none;
};

template <typename input_type_>
    requires requires { typename input_type_::cref_t; }
struct cref_of_<input_type_> {
    using type = typename input_type_::cref_t;
};

/** The scales of one operand in @p backend_type_'s memory, all one with no tensor scale, so a
 *  block-scaled kernel multiplies the same values as its element dtype; @c operand wraps codes into
 *  what kernels take: the codes themselves for plain dtypes, else a reference holding the codes and
 *  their scales, as every kernel of the family expects. */
template <nk_dtype_t input_dtype_, typename backend_type_>
struct unit_block_scales {
    static constexpr nk_block_scaled_format_t format = nk_block_scaled_format_of_dtype(input_dtype_);
    using scale_t = typename nk::type_for<format.block_size ? format.scale_dtype : nk_u8_k>::type;
    using blocks_t = nk::tensor<scale_t, typename backend_type_::template allocator<scale_t>, 2>;

    blocks_t blocks;
    mutable typename cref_of_<typename nk::type_for<input_dtype_>::type>::type reference {};

    nk_status_t initialize(backend_type_ &backend, std::size_t rows, std::size_t stride) {
        if constexpr (format.block_size) {
            std::size_t const scale_stride = stride / format.block_bytes;
            blocks = blocks_t::uninitialized({rows, scale_stride}, allocator_of<scale_t>(backend)).value;
            std::vector<scale_t> const host(rows * scale_stride, scale_t(1.0f));
            if (blocks.empty()) return nk_bad_alloc_k;
            return backend.copy(blocks.data(), host.data(), host.size() * sizeof(scale_t));
        }
        return nk_success_k;
    }

    template <typename codes_pointer_type_>
    auto operand(codes_pointer_type_ codes) const noexcept {
        if constexpr (!format.block_size) return codes;
        else {
            reference.elements = reinterpret_cast<decltype(reference.elements)>(codes);
            reference.scales = reinterpret_cast<decltype(reference.scales)>(blocks.data());
            return &reference;
        }
    }
};

/** Times a packed-B kernel and checks its output through @p compute_expected. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void measure_packed(loop_t &loop, environment_t const &env, backend_type_ backend, expected_metric_t compute_expected,
                    pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn, kernel_type_ kernel,
                    std::size_t rows, std::size_t columns, std::size_t depth) {
    constexpr nk_dtype_t element_dtype_ = nk_block_scaled_format_of_dtype(input_dtype_).element_dtype;
    using input_t = typename nk::type_for<element_dtype_>::type;
    using set_t = matrix_set<backend_type_, output_type_>;
    auto const [a, b] = random_matrices<element_dtype_, backend_type_>(env.settings.seed, rows, columns, depth);
    if (a.empty() || b.empty()) return loop.skip("input allocation failed");
    std::size_t const a_stride = a.stride_bytes(0), row_bytes = b.stride_bytes(0), a_bytes = rows * a_stride;
    unit_block_scales<input_dtype_, backend_type_> a_scales, b_scales;
    if (!succeeded(loop, a_scales.initialize(backend, rows, a_stride)) ||
        !succeeded(loop, b_scales.initialize(backend, columns, row_bytes)))
        return;
    nk_size_t packed_bytes = 0;
    if (!succeeded(loop, packed_size_fn(columns, depth, &packed_bytes))) return;
    auto const [b_uploaded, b_uploaded_status] = upload(backend, b.data(), b.numel());
    if (!nk::succeeded(b_uploaded_status)) return loop.skip(nk::status_name(b_uploaded_status));
    if (b_uploaded.empty()) return loop.skip("B allocation failed");

    // One B upload, packed into every set, since the sets differ only in where they live.
    std::vector<set_t> sets(
        backend.input_sets(bytes_t {a_bytes + packed_bytes + rows * columns * sizeof(output_type_)}));
    for (set_t &set : sets) {
        set = {set_t::bytes_t::uninitialized(a_bytes, allocator_of<char>(backend)).value,
               set_t::bytes_t::uninitialized(packed_bytes, allocator_of<char>(backend)).value,
               set_t::outputs_t::uninitialized(rows * columns, allocator_of<output_type_>(backend)).value};
        if (set.a.empty() || set.b.empty() || set.c.empty()) return loop.skip("set allocation failed");
        if (!succeeded(loop, backend.copy(set.a.raw_values_data(), a.data(), a_bytes))) return;
        if (!succeeded(loop, backend.zero(set.b.raw_values_data(), packed_bytes))) return;
        if (!succeeded(loop, backend.zero(set.c.raw_values_data(), set.c.size_bytes()))) return;
        nk_status_t const submission_status = backend.call(pack_fn, b_scales.operand(b_uploaded.raw_values_data()),
                                                           columns, depth, row_bytes, set.b.raw_values_data(),
                                                           std::size_t(0), columns);
        nk_status_t const completion_status = backend.synchronize();
        if (!succeeded(loop, submission_status) || !succeeded(loop, completion_status)) return;
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        set_t &set = sets[index];
        return backend.call(
            kernel, a_scales.operand(reinterpret_cast<typename input_t::raw_t const *>(set.a.raw_values_data())),
            static_cast<void const *>(set.b.raw_values_data()), set.c.raw_values_data(), rows, columns, depth, a_stride,
            columns * sizeof(output_type_));
    });
    if (!timed) return;
    auto const [score, score_status] = sampled_accuracy<element_dtype_, output_type_>(
        backend, sets[0].c, a.view(), b.view(), depth, compute_expected, written_entries_t::full_k, env.settings.seed);
    if (!nk::succeeded(score_status)) return loop.skip(nk::status_name(score_status));
    report_matrix<output_type_>(loop, 2.0 * rows * columns * depth, score);
}

/** Times a symmetric kernel over A × Aᵀ, judged on the upper triangle: with the diagonal for dots,
 *  without it for distances. `scalar-ops` counts the triangle's rows · (rows + 1) · depth. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_, typename kernel_type_>
void measure_symmetric(loop_t &loop, environment_t const &env, backend_type_ backend,
                       expected_metric_t compute_expected, written_entries_t written, kernel_type_ kernel,
                       std::size_t rows, std::size_t depth) {
    constexpr nk_dtype_t element_dtype_ = nk_block_scaled_format_of_dtype(input_dtype_).element_dtype;
    using input_t = typename nk::type_for<element_dtype_>::type;
    using set_t = matrix_set<backend_type_, output_type_>;
    auto const matrices = random_matrices<element_dtype_, backend_type_>(env.settings.seed, rows, 0, depth);
    auto const &a = matrices[0];
    if (a.empty()) return loop.skip("input allocation failed");
    std::size_t const a_stride = a.stride_bytes(0), a_bytes = rows * a_stride;
    unit_block_scales<input_dtype_, backend_type_> scales;
    if (!succeeded(loop, scales.initialize(backend, rows, a_stride))) return;
    std::vector<set_t> sets(backend.input_sets(bytes_t {a_bytes + rows * rows * sizeof(output_type_)}));
    for (set_t &set : sets) {
        set.a = set_t::bytes_t::uninitialized(a_bytes, allocator_of<char>(backend)).value;
        set.c = set_t::outputs_t::uninitialized(rows * rows, allocator_of<output_type_>(backend)).value;
        if (set.a.empty() || set.c.empty()) return loop.skip("set allocation failed");
        if (!succeeded(loop, backend.copy(set.a.raw_values_data(), a.data(), a_bytes))) return;
        if (!succeeded(loop, backend.zero(set.c.raw_values_data(), set.c.size_bytes()))) return;
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        set_t &set = sets[index];
        return backend.call(
            kernel, scales.operand(reinterpret_cast<typename input_t::raw_t const *>(set.a.raw_values_data())), rows,
            depth, a_stride, set.c.raw_values_data(), rows * sizeof(output_type_), std::size_t(0), rows);
    });
    if (!timed) return;
    auto const [score, score_status] = sampled_accuracy<element_dtype_, output_type_>(
        backend, sets[0].c, a.view(), a.view(), depth, compute_expected, written, env.settings.seed);
    if (!nk::succeeded(score_status)) return loop.skip(nk::status_name(score_status));
    report_matrix<output_type_>(loop, 1.0 * rows * (rows + 1) * depth, score);
}

/** The `<rows x columns x depth>` suffix of a matrix row's name. */
inline std::string matrix_row_name(std::string const &name, std::size_t rows, std::size_t columns, std::size_t depth) {
    return name + "<" + std::to_string(rows) + "x" + std::to_string(columns) + "x" + std::to_string(depth) + ">";
}

/** Runs a packed-B row over the configured matrix shape on @p backend. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void run_packed(environment_t const &env, std::string const &name, expected_metric_t compute_expected,
                pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn, kernel_type_ kernel,
                backend_type_ backend) {
    std::size_t const rows = env.settings.matrix_height, columns = env.settings.matrix_width,
                      depth = env.settings.matrix_depth;
    run_benchmark(env, matrix_row_name(name, rows, columns, depth),
                  measure_packed<input_dtype_, output_type_, backend_type_, pack_size_kernel_type_, pack_kernel_type_,
                                 kernel_type_>,
                  backend, compute_expected, packed_size_fn, pack_fn, kernel, rows, columns, depth);
}

/** Runs a symmetric row over @c NUMWARS_DIMS_HEIGHT vectors of @c NUMWARS_DIMS_DEPTH dimensions on
 *  @p backend. */
template <nk_dtype_t input_dtype_, typename output_type_, typename backend_type_, typename kernel_type_>
void run_symmetric(environment_t const &env, std::string const &name, expected_metric_t compute_expected,
                   written_entries_t written, kernel_type_ kernel, backend_type_ backend) {
    std::size_t const rows = env.settings.matrix_height, depth = env.settings.matrix_depth;
    std::string const row_name = name + "<" + std::to_string(rows) + "x" + std::to_string(depth) + ">";
    run_benchmark(env, row_name, measure_symmetric<input_dtype_, output_type_, backend_type_, kernel_type_>, backend,
                  compute_expected, written, kernel, rows, depth);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void run_dots_packed(environment_t const &env, std::string const &name, pack_size_kernel_type_ packed_size_fn,
                     pack_kernel_type_ pack_fn, kernel_type_ kernel, backend_type_ backend = {}) {
    run_packed<input_dtype_, typename nk::type_for<input_dtype_>::type::dot_result_t, backend_type_>(
        env, name, dot_compensated_f64, packed_size_fn, pack_fn, kernel, backend);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void run_angulars_packed(environment_t const &env, std::string const &name, pack_size_kernel_type_ packed_size_fn,
                         pack_kernel_type_ pack_fn, kernel_type_ kernel, backend_type_ backend = {}) {
    run_packed<input_dtype_, typename nk::type_for<input_dtype_>::type::angular_result_t, backend_type_>(
        env, name, angular_compensated_f64, packed_size_fn, pack_fn, kernel, backend);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename kernel_type_>
void run_euclideans_packed(environment_t const &env, std::string const &name, pack_size_kernel_type_ packed_size_fn,
                           pack_kernel_type_ pack_fn, kernel_type_ kernel, backend_type_ backend = {}) {
    run_packed<input_dtype_, typename nk::type_for<input_dtype_>::type::euclidean_result_t, backend_type_>(
        env, name, euclidean_compensated_f64, packed_size_fn, pack_fn, kernel, backend);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename kernel_type_>
void run_dots_symmetric(environment_t const &env, std::string const &name, kernel_type_ kernel,
                        backend_type_ backend = {}) {
    run_symmetric<input_dtype_, typename nk::type_for<input_dtype_>::type::dot_result_t, backend_type_>(
        env, name, dot_compensated_f64, written_entries_t::upper_triangle_k, kernel, backend);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename kernel_type_>
void run_angulars_symmetric(environment_t const &env, std::string const &name, kernel_type_ kernel,
                            backend_type_ backend = {}) {
    run_symmetric<input_dtype_, typename nk::type_for<input_dtype_>::type::angular_result_t, backend_type_>(
        env, name, angular_compensated_f64, written_entries_t::strict_upper_triangle_k, kernel, backend);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename kernel_type_>
void run_euclideans_symmetric(environment_t const &env, std::string const &name, kernel_type_ kernel,
                              backend_type_ backend = {}) {
    run_symmetric<input_dtype_, typename nk::type_for<input_dtype_>::type::euclidean_result_t, backend_type_>(
        env, name, euclidean_compensated_f64, written_entries_t::strict_upper_triangle_k, kernel, backend);
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

/** Fills @c tokens with queries per second, and `scalar-ops` with the visible 4 · depth work per
 *  pair and head. */
inline void report_attention(loop_t &loop, attention_visibility_t visibility, attention_shape_t shape) {
    double const scalar_ops_per_call = 4.0 * double(shape.depth * shape.head_count) *
                                       attention_visible_pairs(visibility, shape);
    loop.rate("tokens", double(shape.queries));
    loop.rate("scalar-ops", scalar_ops_per_call);
}

/** Queries, keys and values of @p shape in [-1, 1], or [-32, 32] for integers, identical for every
 *  backend under one @p seed. */
template <nk_dtype_t input_dtype_>
std::array<nk::vector<typename nk::type_for<input_dtype_>::type>, 3> random_attention(attention_shape_t shape,
                                                                                      seed_t seed) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    std::mt19937 generator(seed.value);
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
nk_status_t attention_pack(backend_type_ &backend, pack_kernel_type_ pack_fn, attention_shape_t shape,
                           raw_type_ const *keys, raw_type_ const *values, nk_u32_t const *directory, void *packed) {
    std::size_t const key_stride = shape.key_value_head_count * shape.depth * sizeof(raw_type_);
    return backend.call(pack_fn, keys, values, shape.key_value_head_count, shape.depth, directory, directory + 1,
                        std::size_t(1), key_stride, key_stride, packed, std::size_t(0), shape.key_value_head_count);
}

/** Runs @p attention_fn over the one segment of @p shape under @p visibility_, queries aligned to
 *  the keys' end. */
template <attention_visibility_t visibility_, typename backend_type_, typename attention_kernel_type_,
          typename raw_type_>
nk_status_t attend(backend_type_ &backend, attention_kernel_type_ attention_fn, attention_shape_t shape,
                   nk_u32_t const *directory, raw_type_ const *queries, void const *packed, nk_f32_t *output) {
    std::size_t const query_stride = shape.head_count * shape.depth * sizeof(raw_type_);
    std::size_t const output_stride = shape.head_count * shape.depth * sizeof(nk_f32_t);
    nk_f32_t const scale = 1.0f / std::sqrt(float(shape.depth));
    if constexpr (visibility_ == attention_visibility_t::bidirectional_k)
        return backend.call(attention_fn, queries, packed, output, shape.head_count, shape.key_value_head_count,
                            shape.depth, directory + 2, query_stride, output_stride, scale, std::size_t(0),
                            shape.head_count);
    else
        return backend.call(attention_fn, queries, packed, output, shape.head_count, shape.key_value_head_count,
                            shape.depth, directory + 2, query_stride, output_stride, scale,
                            nk_i64_t(shape.keys) - nk_i64_t(shape.queries), attention_window(visibility_),
                            std::size_t(0), shape.head_count);
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
void measure_attention(loop_t &loop, environment_t const &env, backend_type_ backend,
                       pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                       attention_kernel_type_ attention_fn, attention_shape_t shape) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using set_t = attention_set<backend_type_, input_t>;
    auto const [queries, keys, values] = random_attention<input_dtype_>(shape, env.settings.seed);
    auto const [keys_uploaded, keys_uploaded_status] = upload(backend, keys.values_data(), keys.size());
    if (!nk::succeeded(keys_uploaded_status)) return loop.skip(nk::status_name(keys_uploaded_status));
    auto const [values_uploaded, values_uploaded_status] = upload(backend, values.values_data(), values.size());
    if (!nk::succeeded(values_uploaded_status)) return loop.skip(nk::status_name(values_uploaded_status));
    nk_u32_t const offsets[4] = {0, nk_u32_t(shape.keys), 0, nk_u32_t(shape.queries)};
    auto const [directory, directory_status] = upload(backend, offsets, 4);
    if (!nk::succeeded(directory_status)) return loop.skip(nk::status_name(directory_status));
    if (keys_uploaded.empty() || values_uploaded.empty() || directory.empty())
        return loop.skip("input allocation failed");
    nk_u32_t const lengths[1] = {nk_u32_t(shape.keys)};
    nk_size_t packed_bytes = 0;
    if (!succeeded(loop, packed_size_fn(shape.key_value_head_count, shape.depth, lengths, 1, &packed_bytes))) return;
    std::size_t const output_count = shape.queries * shape.head_count * shape.depth;
    std::vector<set_t> sets(
        backend.input_sets(bytes_t {queries.size_bytes() + packed_bytes + output_count * sizeof(nk_f32_t)}));
    for (set_t &set : sets) {
        auto [uploaded, upload_status] = upload(backend, queries.values_data(), queries.size());
        if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
        set = {std::move(uploaded), set_t::bytes_t::uninitialized(packed_bytes, allocator_of<char>(backend)).value,
               set_t::outputs_t::uninitialized(output_count, allocator_of<nk::f32_t>(backend)).value};
        if (set.queries.empty() || set.packed.empty() || set.output.empty()) return loop.skip("set allocation failed");
        if (!succeeded(loop, backend.zero(set.packed.raw_values_data(), packed_bytes))) return;
        if (!succeeded(loop, backend.zero(set.output.raw_values_data(), set.output.size_bytes()))) return;
        nk_status_t const submission_status = attention_pack(backend, pack_fn, shape, keys_uploaded.raw_values_data(),
                                                             values_uploaded.raw_values_data(),
                                                             directory.raw_values_data(), set.packed.raw_values_data());
        nk_status_t const completion_status = backend.synchronize();
        if (!succeeded(loop, submission_status) || !succeeded(loop, completion_status)) return;
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        set_t &set = sets[index];
        return attend<visibility_>(backend, attention_fn, shape, directory.raw_values_data(),
                                   set.queries.raw_values_data(), set.packed.raw_values_data(),
                                   set.output.raw_values_data());
    });
    if (timed) report_attention(loop, visibility_, shape);
}

/** Runs one attention row of @p shape under @p visibility_ on @p backend. */
template <nk_dtype_t input_dtype_, attention_visibility_t visibility_, typename backend_type_,
          typename pack_size_kernel_type_, typename pack_kernel_type_, typename attention_kernel_type_>
void run_attention_row(environment_t const &env, std::string const &name, pack_size_kernel_type_ packed_size_fn,
                       pack_kernel_type_ pack_fn, attention_kernel_type_ attention_fn, attention_shape_t shape,
                       backend_type_ backend) {
    run_benchmark(env, attention_row_name(name, visibility_, shape),
                  measure_attention<input_dtype_, visibility_, backend_type_, pack_size_kernel_type_, pack_kernel_type_,
                                    attention_kernel_type_>,
                  backend, packed_size_fn, pack_fn, attention_fn, shape);
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename attention_kernel_type_>
void run_attention_bidirectional(environment_t const &env, std::string const &name,
                                 pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                                 attention_kernel_type_ attention_fn, backend_type_ backend = {}) {
    for (attention_shape_t const shape : backend.attention_shapes(env))
        run_attention_row<input_dtype_, attention_visibility_t::bidirectional_k, backend_type_>(
            env, name, packed_size_fn, pack_fn, attention_fn, shape, backend);
}

/** Causal rows per backend shape, plus a 1024-key window wherever it hides keys. */
template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename attention_kernel_type_>
void run_attention_causal(environment_t const &env, std::string const &name, pack_size_kernel_type_ packed_size_fn,
                          pack_kernel_type_ pack_fn, attention_kernel_type_ attention_fn, backend_type_ backend = {}) {
    for (attention_shape_t const shape : backend.attention_shapes(env)) {
        run_attention_row<input_dtype_, attention_visibility_t::causal_k, backend_type_>(
            env, name, packed_size_fn, pack_fn, attention_fn, shape, backend);
        if (attention_window_clips(shape))
            run_attention_row<input_dtype_, attention_visibility_t::causal_window_1024_k, backend_type_>(
                env, name, packed_size_fn, pack_fn, attention_fn, shape, backend);
    }
}

/** Measures an in-place NeoX split-half RoPE kernel over @p rows single-head rows, all pairs
 *  rotated. */
template <nk_dtype_t input_dtype_, typename backend_type_, typename kernel_type_>
void measure_attention_rope(loop_t &loop, environment_t const &env, backend_type_ backend, kernel_type_ kernel,
                            std::size_t rows, std::size_t dimensions) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using values_t = nk::vector<input_t, typename backend_type_::template allocator<input_t>>;
    std::size_t const count = rows * dimensions, angles = rows * dimensions / 2;
    std::size_t const stride = dimensions * sizeof(typename input_t::raw_t);
    // An exact quarter-turn preserves values across repeated in-place rounds.
    std::vector<nk::f32_t> const cosines_host(angles, nk::f32_t(0)), sines_host(angles, nk::f32_t(1));
    auto const [cosines, cosines_status] = upload(backend, cosines_host.data(), angles);
    if (!nk::succeeded(cosines_status)) return loop.skip(nk::status_name(cosines_status));
    auto const [sines, sines_status] = upload(backend, sines_host.data(), angles);
    if (!nk::succeeded(sines_status)) return loop.skip(nk::status_name(sines_status));
    if (cosines.empty() || sines.empty()) return loop.skip("angle allocation failed");
    std::vector<values_t> sets(backend.input_sets(dtype_bytes(input_dtype_, count)));
    for (values_t &set : sets) {
        auto [uploaded, upload_status] = random_upload<input_dtype_>(backend, count, env.settings.seed);
        if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
        set = std::move(uploaded);
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        auto *tokens = sets[index].raw_values_data();
        return backend.call(kernel, tokens, cosines.raw_values_data(), sines.raw_values_data(), tokens, rows,
                            std::size_t(1), dimensions, stride, stride);
    });
    if (timed) loop.byte_rate(double(dtype_bytes(input_dtype_, count).value));
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename kernel_type_ = void>
void run_attention_rope(environment_t const &env, std::string const &name, kernel_type_ *kernel,
                        backend_type_ backend = {}) {
    std::size_t const rows = backend.token_rows(env), dimensions = env.settings.dense_dimensions;
    run_benchmark(env, token_rows_name(name, rows, dimensions),
                  measure_attention_rope<input_dtype_, backend_type_, kernel_type_ *>, backend, kernel, rows,
                  dimensions);
}

#pragma endregion Attention

#pragma region Token Rows

/** Times the element-wise @p kernel_kind_, a sum, scale, blend or FMA, over @p count values with
 *  α = 0.2 and β = 0.3; @c bytes counts the inputs. */
template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_, nk_dtype_t alpha_dtype_, typename backend_type_,
          typename kernel_type_>
void measure_each(loop_t &loop, environment_t const &env, backend_type_ backend, kernel_type_ kernel,
                  std::size_t count) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using alpha_t = typename nk::type_for<alpha_dtype_>::type;
    using values_t = nk::vector<input_t, typename backend_type_::template allocator<input_t>>;
    std::size_t const inputs = kernel_kind_ == nk_kernel_each_fma_k     ? 3
                               : kernel_kind_ == nk_kernel_each_scale_k ? 1
                                                                        : 2;
    alpha_t const host_coefficients[2] = {alpha_t(0.2f), alpha_t(0.3f)};
    auto const [coefficients, coefficients_status] = upload(backend, host_coefficients, 2);
    if (!nk::succeeded(coefficients_status)) return loop.skip(nk::status_name(coefficients_status));
    if (coefficients.empty()) return loop.skip("coefficient allocation failed");
    auto const *alpha = coefficients.raw_values_data(), *beta = alpha + 1;
    std::vector<std::array<values_t, 4>> sets(backend.input_sets(dtype_bytes(input_dtype_, (inputs + 1) * count)));
    for (std::array<values_t, 4> &set : sets) {
        for (std::size_t input = 0; input != inputs; ++input) {
            auto [uploaded, upload_status] = random_upload<input_dtype_>(backend, count, env.settings.seed);
            if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
            set[input] = std::move(uploaded);
        }
        set[3] = values_t::uninitialized(count, allocator_of<input_t>(backend)).value;
        if (set[3].empty()) return loop.skip("set allocation failed");
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        auto &set = sets[index];
        if constexpr (kernel_kind_ == nk_kernel_each_sum_k)
            return backend.call(kernel, set[0].raw_values_data(), set[1].raw_values_data(), count,
                                set[3].raw_values_data());
        else if constexpr (kernel_kind_ == nk_kernel_each_scale_k)
            return backend.call(kernel, set[0].raw_values_data(), count, alpha, beta, set[3].raw_values_data());
        else if constexpr (kernel_kind_ == nk_kernel_each_blend_k)
            return backend.call(kernel, set[0].raw_values_data(), set[1].raw_values_data(), count, alpha, beta,
                                set[3].raw_values_data());
        else
            return backend.call(kernel, set[0].raw_values_data(), set[1].raw_values_data(), set[2].raw_values_data(),
                                count, alpha, beta, set[3].raw_values_data());
    });
    if (timed) loop.byte_rate(double(dtype_bytes(input_dtype_, inputs * count).value));
}

template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_ = nk_kernel_unknown_k,
          nk_dtype_t alpha_dtype_ = nk_dtype_unknown_k, typename backend_type_ = host_backend_t,
          typename kernel_type_ = void>
void run_each(environment_t const &env, std::string const &name, kernel_type_ *kernel, backend_type_ backend = {}) {
    std::size_t const rows = backend.token_rows(env), columns = env.settings.batch_per_core;
    run_benchmark(env, token_rows_name(name, rows, columns),
                  measure_each<input_dtype_, kernel_kind_, alpha_dtype_, backend_type_, kernel_type_ *>, backend,
                  kernel, rows * columns);
}

/** Times SwiGLU over @p rows rows of @p columns gate and up values; @c bytes counts both inputs. */
template <nk_dtype_t input_dtype_, typename backend_type_, typename kernel_type_>
void measure_swiglu(loop_t &loop, environment_t const &env, backend_type_ backend, kernel_type_ kernel,
                    std::size_t rows, std::size_t columns) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using values_t = nk::vector<input_t, typename backend_type_::template allocator<input_t>>;
    std::size_t const count = rows * columns, stride = columns * sizeof(typename input_t::raw_t);
    std::vector<std::array<values_t, 3>> sets(backend.input_sets(dtype_bytes(input_dtype_, 3 * count)));
    for (std::array<values_t, 3> &set : sets) {
        {
            auto [uploaded, upload_status] = random_upload<input_dtype_>(backend, count, env.settings.seed);
            if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
            set[0] = std::move(uploaded);
        }
        {
            auto [uploaded, upload_status] = random_upload<input_dtype_>(backend, count, env.settings.seed);
            if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
            set[1] = std::move(uploaded);
        }
        set[2] = values_t::uninitialized(count, allocator_of<input_t>(backend)).value;
        if (set[0].empty() || set[1].empty() || set[2].empty()) return loop.skip("set allocation failed");
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        auto &set = sets[index];
        return backend.call(kernel, set[0].raw_values_data(), set[1].raw_values_data(), set[2].raw_values_data(), rows,
                            columns, stride, stride, stride, 1.0f, 1.0f);
    });
    if (timed) loop.byte_rate(double(dtype_bytes(input_dtype_, 2 * count).value));
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename kernel_type_ = void>
void run_swiglu(environment_t const &env, std::string const &name, kernel_type_ *kernel, backend_type_ backend = {}) {
    std::size_t const rows = backend.token_rows(env), columns = env.settings.batch_per_core;
    run_benchmark(env, token_rows_name(name, rows, columns),
                  measure_swiglu<input_dtype_, backend_type_, kernel_type_ *>, backend, kernel, rows, columns);
}

/** Times RMSNorm over @p rows rows of @p columns values, one group with a unit γ each; @c bytes
 *  counts the input. */
template <nk_dtype_t input_dtype_, typename backend_type_, typename kernel_type_>
void measure_rmsnorm(loop_t &loop, environment_t const &env, backend_type_ backend, kernel_type_ kernel,
                     std::size_t rows, std::size_t columns) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using values_t = nk::vector<input_t, typename backend_type_::template allocator<input_t>>;
    std::size_t const count = rows * columns, stride = columns * sizeof(typename input_t::raw_t);
    std::vector<nk::f32_t> const ones(columns, nk::f32_t(1.0f));
    auto const [gamma, gamma_status] = upload(backend, ones.data(), columns);
    if (!nk::succeeded(gamma_status)) return loop.skip(nk::status_name(gamma_status));
    if (gamma.empty()) return loop.skip("gamma allocation failed");
    std::vector<std::array<values_t, 2>> sets(backend.input_sets(dtype_bytes(input_dtype_, 2 * count)));
    for (std::array<values_t, 2> &set : sets) {
        {
            auto [uploaded, upload_status] = random_upload<input_dtype_>(backend, count, env.settings.seed);
            if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
            set[0] = std::move(uploaded);
        }
        set[1] = values_t::uninitialized(count, allocator_of<input_t>(backend)).value;
        if (set[0].empty() || set[1].empty()) return loop.skip("set allocation failed");
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        return backend.call(kernel, sets[index][0].raw_values_data(), gamma.raw_values_data(),
                            sets[index][1].raw_values_data(), rows, std::size_t(1), columns, stride, stride, 1e-6f);
    });
    if (timed) loop.byte_rate(double(dtype_bytes(input_dtype_, count).value));
}

template <nk_dtype_t input_dtype_, typename backend_type_ = host_backend_t, typename kernel_type_ = void>
void run_rmsnorm(environment_t const &env, std::string const &name, kernel_type_ *kernel, backend_type_ backend = {}) {
    std::size_t const rows = backend.token_rows(env), columns = env.settings.batch_per_core;
    run_benchmark(env, token_rows_name(name, rows, columns),
                  measure_rmsnorm<input_dtype_, backend_type_, kernel_type_ *>, backend, kernel, rows, columns);
}

/** Times a bulk cast of @p count random @p from_dtype_ values into @p to_dtype_. @c bytes counts
 *  the input and the output once. */
template <nk_dtype_t from_dtype_, nk_dtype_t to_dtype_, typename backend_type_, typename kernel_type_>
void measure_cast_rows(loop_t &loop, environment_t const &env, backend_type_ backend, kernel_type_ kernel,
                       std::size_t count) {
    using from_t = typename nk::type_for<from_dtype_>::type;
    using to_t = typename nk::type_for<to_dtype_>::type;
    using sources_t = nk::vector<from_t, typename backend_type_::template allocator<from_t>>;
    using targets_t = nk::vector<to_t, typename backend_type_::template allocator<to_t>>;
    bytes_t const per_set {dtype_bytes(from_dtype_, count).value + dtype_bytes(to_dtype_, count).value};
    std::size_t const sets_count = backend.input_sets(per_set);
    std::vector<sources_t> sources(sets_count);
    std::vector<targets_t> targets(sets_count);
    for (std::size_t set = 0; set != sets_count; ++set) {
        {
            auto [uploaded, upload_status] = random_upload<from_dtype_>(backend, count, env.settings.seed);
            if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
            sources[set] = std::move(uploaded);
        }
        targets[set] = targets_t::uninitialized(count, allocator_of<to_t>(backend)).value;
        if (sources[set].empty() || targets[set].empty()) return loop.skip("set allocation failed");
    }
    bool const timed = time_rotating(loop, backend, sets_count, [&](std::size_t index) {
        return backend.call(kernel, static_cast<void const *>(sources[index].raw_values_data()), from_dtype_,
                            static_cast<void *>(targets[index].raw_values_data()), to_dtype_, count);
    });
    if (timed) loop.byte_rate(double(per_set.value));
}

/** Whether a block-scaled row times encoding F32 into its format or decoding the format to F32. */
enum class block_scaled_direction_t { encode_k, decode_k };

/** Times encoding @p count random F32 values into @p dtype_, its tensor scale set to one, or
 *  decoding that encoding back. @c bytes counts the F32 values, elements and scales once. */
template <block_scaled_direction_t direction_, nk_dtype_t dtype_, typename backend_type_, typename kernel_type_>
void measure_block_scaled_rows(loop_t &loop, environment_t const &env, backend_type_ backend, kernel_type_ kernel,
                               std::size_t count) {
    using values_t = nk::vector<nk::f32_t, typename backend_type_::template allocator<nk::f32_t>>;
    using codes_t = nk::vector<char, typename backend_type_::template allocator<char>>;
    using block_t = typename nk::type_for<dtype_>::type;
    using cref_t = typename block_t::cref_t;
    using ref_t = typename block_t::ref_t;
    constexpr nk_block_scaled_format_t format = nk_block_scaled_format_of_dtype(dtype_);
    std::size_t const elements_bytes = nk_block_scaled_elements_size(count, format);
    std::size_t const scales_bytes = nk_block_scaled_scales_size(count, format);
    nk_f32_t const unit_scale = 1.0f;
    auto [tensor_scale_bytes, tensor_scale_bytes_status] = upload(backend, reinterpret_cast<char const *>(&unit_scale),
                                                                  sizeof(unit_scale));
    if (!nk::succeeded(tensor_scale_bytes_status)) return loop.skip(nk::status_name(tensor_scale_bytes_status));
    if (tensor_scale_bytes.empty()) return loop.skip("tensor scale allocation failed");
    auto *tensor_scale = reinterpret_cast<nk_f32_t *>(tensor_scale_bytes.raw_values_data());
    bytes_t const per_set {count * sizeof(nk_f32_t) + elements_bytes + scales_bytes};
    std::size_t const sets_count = backend.input_sets(per_set);
    std::vector<values_t> values(sets_count);
    std::vector<codes_t> elements(sets_count), scales(sets_count);
    std::vector<ref_t> refs(sets_count);
    std::vector<cref_t> crefs(sets_count);
    for (std::size_t set = 0; set != sets_count; ++set) {
        {
            auto [uploaded, upload_status] = random_upload<nk_f32_k>(backend, count, env.settings.seed);
            if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
            values[set] = std::move(uploaded);
        }
        elements[set] = codes_t::uninitialized(elements_bytes, allocator_of<char>(backend)).value;
        scales[set] = codes_t::uninitialized(scales_bytes, allocator_of<char>(backend)).value;
        if (values[set].empty() || elements[set].empty() || scales[set].empty())
            return loop.skip("set allocation failed");
        refs[set].elements = reinterpret_cast<decltype(ref_t::elements)>(elements[set].raw_values_data());
        refs[set].scales = reinterpret_cast<decltype(ref_t::scales)>(scales[set].raw_values_data());
        crefs[set].elements = refs[set].elements;
        crefs[set].scales = refs[set].scales;
        if constexpr (format.tensor_scale_dtype == nk_f32_k) refs[set].tensor_scale = tensor_scale;
        if constexpr (format.tensor_scale_dtype == nk_f32_k) crefs[set].tensor_scale = tensor_scale;
        nk_status_t const submission_status = backend.call(kernel,
                                                           static_cast<void const *>(values[set].raw_values_data()),
                                                           nk_f32_k, static_cast<void *>(&refs[set]), dtype_, count);
        nk_status_t const completion_status = backend.synchronize();
        if (!succeeded(loop, submission_status) || !succeeded(loop, completion_status)) return;
    }
    bool const timed = time_rotating(loop, backend, sets_count, [&](std::size_t index) {
        if constexpr (direction_ == block_scaled_direction_t::encode_k)
            return backend.call(kernel, static_cast<void const *>(values[index].raw_values_data()), nk_f32_k,
                                static_cast<void *>(&refs[index]), dtype_, count);
        else
            return backend.call(kernel, static_cast<void const *>(&crefs[index]), dtype_,
                                static_cast<void *>(values[index].raw_values_data()), nk_f32_k, count);
    });
    if (timed) loop.byte_rate(double(per_set.value));
}

/** Times the sum and sum of squares of @p count random @p input_dtype_ values. @c bytes counts the
 *  input once. */
template <nk_dtype_t input_dtype_, typename backend_type_, typename kernel_type_>
void measure_moments_rows(loop_t &loop, environment_t const &env, backend_type_ backend, kernel_type_ kernel,
                          std::size_t count) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using sum_t = typename input_t::reduce_moments_sum_t;
    using sumsq_t = typename input_t::reduce_moments_sumsq_t;
    using values_t = nk::vector<input_t, typename backend_type_::template allocator<input_t>>;
    auto sum = nk::vector<sum_t, typename backend_type_::template allocator<sum_t>>::uninitialized(
                   1, allocator_of<sum_t>(backend))
                   .value;
    auto sumsq = nk::vector<sumsq_t, typename backend_type_::template allocator<sumsq_t>>::uninitialized(
                     1, allocator_of<sumsq_t>(backend))
                     .value;
    if (sum.empty() || sumsq.empty()) return loop.skip("output allocation failed");
    std::vector<values_t> sets(backend.input_sets(dtype_bytes(input_dtype_, count)));
    for (values_t &set : sets) {
        auto [uploaded, upload_status] = random_upload<input_dtype_>(backend, count, env.settings.seed);
        if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
        set = std::move(uploaded);
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        return backend.call(kernel, sets[index].raw_values_data(), count, sizeof(typename input_t::raw_t),
                            sum.raw_values_data(), sumsq.raw_values_data());
    });
    if (timed) loop.byte_rate(double(dtype_bytes(input_dtype_, count).value));
}

/** Times the first minimum and maximum of @p count random @p input_dtype_ values with their
 *  indices. @c bytes counts the input once. */
template <nk_dtype_t input_dtype_, typename backend_type_, typename kernel_type_>
void measure_minmax_rows(loop_t &loop, environment_t const &env, backend_type_ backend, kernel_type_ kernel,
                         std::size_t count) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using value_t = typename input_t::reduce_minmax_value_t;
    using values_t = nk::vector<input_t, typename backend_type_::template allocator<input_t>>;
    auto extrema = nk::vector<value_t, typename backend_type_::template allocator<value_t>>::uninitialized(
                       2, allocator_of<value_t>(backend))
                       .value;
    auto indices = nk::vector<nk::u64_t, typename backend_type_::template allocator<nk::u64_t>>::uninitialized(
                       2, allocator_of<nk::u64_t>(backend))
                       .value;
    if (extrema.empty() || indices.empty()) return loop.skip("output allocation failed");
    auto *index_values = reinterpret_cast<nk_size_t *>(indices.raw_values_data());
    std::vector<values_t> sets(backend.input_sets(dtype_bytes(input_dtype_, count)));
    for (values_t &set : sets) {
        auto [uploaded, upload_status] = random_upload<input_dtype_>(backend, count, env.settings.seed);
        if (!nk::succeeded(upload_status)) return loop.skip(nk::status_name(upload_status));
        set = std::move(uploaded);
    }
    bool const timed = time_rotating(loop, backend, sets.size(), [&](std::size_t index) {
        return backend.call(kernel, sets[index].raw_values_data(), count, sizeof(typename input_t::raw_t),
                            extrema.raw_values_data(), index_values, extrema.raw_values_data() + 1, index_values + 1);
    });
    if (timed) loop.byte_rate(double(dtype_bytes(input_dtype_, count).value));
}

template <nk_dtype_t from_dtype_, nk_dtype_t to_dtype_, typename backend_type_, typename kernel_type_>
void run_cast_rows(environment_t const &env, std::string const &name, kernel_type_ kernel, std::size_t rows,
                   std::size_t columns, backend_type_ backend) {
    run_benchmark(env, token_rows_name(name, rows, columns),
                  measure_cast_rows<from_dtype_, to_dtype_, backend_type_, kernel_type_>, backend, kernel,
                  rows * columns);
}

template <block_scaled_direction_t direction_, nk_dtype_t dtype_, typename backend_type_, typename kernel_type_>
void run_block_scaled_rows(environment_t const &env, std::string const &name, kernel_type_ kernel, std::size_t rows,
                           std::size_t columns, backend_type_ backend) {
    run_benchmark(env, token_rows_name(name, rows, columns),
                  measure_block_scaled_rows<direction_, dtype_, backend_type_, kernel_type_>, backend, kernel,
                  rows * columns);
}

template <nk_dtype_t input_dtype_, typename backend_type_, typename kernel_type_>
void run_moments_rows(environment_t const &env, std::string const &name, kernel_type_ kernel, std::size_t rows,
                      std::size_t columns, backend_type_ backend) {
    run_benchmark(env, token_rows_name(name, rows, columns),
                  measure_moments_rows<input_dtype_, backend_type_, kernel_type_>, backend, kernel, rows * columns);
}

template <nk_dtype_t input_dtype_, typename backend_type_, typename kernel_type_>
void run_minmax_rows(environment_t const &env, std::string const &name, kernel_type_ kernel, std::size_t rows,
                     std::size_t columns, backend_type_ backend) {
    run_benchmark(env, token_rows_name(name, rows, columns),
                  measure_minmax_rows<input_dtype_, backend_type_, kernel_type_>, backend, kernel, rows * columns);
}

#pragma endregion Token Rows

} // namespace ashvardanian::numkong::bench

#endif // NUMKONG_BENCH_CROSS_HPP
