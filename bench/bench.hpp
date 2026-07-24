/**
 *  @brief NumKong C++ Benchmark Suite using Google Benchmark - Header file.
 *  @file bench/bench.hpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *
 *  Comprehensive benchmarks for NumKong SIMD-optimized functions measuring
 *  throughput performance. This header contains measurement infrastructure
 *  and helper templates used across multiple benchmark files.
 */

#pragma once

#include <algorithm> // `std::min`, `std::max`
#include <bit>       // `std::bit_floor`
#include <random>    // `std::mt19937`
#include <string>    // `std::string`, `std::to_string`
#include <vector>    // `std::vector`

#include <benchmark/benchmark.h>

#if !defined(NK_ALLOW_ISA_REDIRECT)
#define NK_ALLOW_ISA_REDIRECT 0
#endif

#if !defined(NK_COMPARE_TO_MKL)
#define NK_COMPARE_TO_MKL 0
#endif
#if !defined(NK_COMPARE_TO_BLAS)
#define NK_COMPARE_TO_BLAS 0
#endif
#if !defined(NK_COMPARE_TO_ACCELERATE)
#define NK_COMPARE_TO_ACCELERATE 0
#endif

#if NK_COMPARE_TO_MKL
#include <mkl.h>
// MKL provides additional GEMM routines:
// - cblas_gemm_bf16bf16f32: BF16 inputs -> F32 output
// - cblas_hgemm: F16 GEMM (if available)
#elif NK_COMPARE_TO_ACCELERATE
#include <Accelerate/Accelerate.h> // Apple Accelerate framework
#elif NK_COMPARE_TO_BLAS
#include <cblas.h> // Generic CBLAS (OpenBLAS, etc.)
// OpenBLAS thread control (weak symbol to avoid link errors if not present)
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
#endif

#include "numkong/types.hpp"
#include "numkong/tensor.hpp"
#include "numkong/random.hpp"

namespace bm = benchmark;
namespace nk = ashvardanian::numkong;

struct bench_config_t {
    /** Vector dimension for dot products and spatial metrics. Override: `NK_DENSE_DIMENSIONS`. */
    std::size_t dense_dimensions = 1536;
    /** Curved metric dimensions (quadratic impact). Override: `NK_CURVED_DIMENSIONS`. */
    std::size_t curved_dimensions = 64;
    /** Number of 3D points for mesh metrics (RMSD, Kabsch). Override: `NK_MESH_POINTS`. */
    std::size_t mesh_points = 1000;
    /** GEMM M dimension. Override: `NK_MATRIX_HEIGHT`. */
    std::size_t matrix_height = 1024;
    /** GEMM N dimension. Override: `NK_MATRIX_WIDTH`. */
    std::size_t matrix_width = 128;
    /** GEMM K dimension. Override: `NK_MATRIX_DEPTH`. */
    std::size_t matrix_depth = 1536;
    /** Random seed for reproducible benchmarks. Override: `NK_SEED`. */
    std::uint32_t seed = 42;
    /** First sparse set size. Override: `NK_SPARSE_FIRST_LENGTH`. */
    std::size_t sparse_first_length = 1024;
    /** Second sparse set size. Override: `NK_SPARSE_SECOND_LENGTH`. */
    std::size_t sparse_second_length = 8192;
    /** Sparse intersection share [0.0, 1.0]. Override: `NK_SPARSE_INTERSECTION`. */
    double sparse_intersection_share = 0.5;
    /** Max angular separation in degrees for geospatial benchmarks. Override: `NK_MAX_COORD_ANGLE`. */
    float max_coord_angle = 180.0f;
    /** Memory budget in bytes for pre-allocated inputs. Override: `NK_BUDGET_MB`. */
#if defined(__wasi__)
    std::size_t budget_bytes = std::size_t(32) * 1024 * 1024;
#else
    std::size_t budget_bytes = std::size_t(1024) * 1024 * 1024;
#endif
};

extern bench_config_t bench_config;

inline std::mt19937 make_random_engine() { return std::mt19937(bench_config.seed); }

/**
 *  @brief Compute byte count for `count` elements of `dtype`, handling sub-byte and complex types.
 *
 *  Uses `nk_dtype_bits` to get the correct bits per element, then rounds up to whole bytes.
 */
inline std::size_t bench_dtype_bytes(nk_dtype_t dtype, std::size_t count) {
    return nk::divide_round_up(count * nk_dtype_bits(dtype), std::size_t(NK_BITS_PER_BYTE));
}

/**
 *  @brief Compute the number of pre-allocated input sets that fit within `bench_budget`.
 *
 *  Returns a power-of-two count in [1, 1024] so the benchmark loop can use
 *  `iterations & (count - 1)` as a fast modulo for input cycling.
 */
inline std::size_t bench_input_count(std::size_t bytes_per_set) {
    std::size_t count = bench_config.budget_bytes / std::max(bytes_per_set, std::size_t(1));
    count = std::min(count, std::size_t(1024));
    count = std::max(std::bit_floor(count), std::size_t(1));
    return count;
}

/** @brief Factory function to allocate vectors, potentially raising bad-allocs. */
template <typename type_>
[[nodiscard]] nk::vector<type_> make_vector(std::size_t count) {
    auto result = nk::vector<type_>::try_zeros(count);
#if defined(__cpp_exceptions) && __cpp_exceptions
    if (result.empty() && count > 0) throw std::bad_alloc();
#else
    if (result.empty() && count > 0) std::abort();
#endif
    return result;
}

/**
 *  @brief Factory function to allocate matrix buffers with correct size for sub-byte types.
 *
 *  For sub-byte types (u4, i4, u1), calculates the actual number of raw_t elements needed
 *  to store a matrix of logical dimensions rows x cols.
 *
 *  @tparam dtype_ The NumKong dtype (e.g., nk_u4_k, nk_i4_k)
 *  @param rows Number of logical rows
 *  @param cols Number of logical columns per row
 *  @return Vector with correct byte capacity for the matrix
 */
template <nk_dtype_t dtype_>
[[nodiscard]] nk::vector<typename nk::type_for<dtype_>::type> make_vector_for_matrix(std::size_t rows,
                                                                                     std::size_t cols) {
    using type_ = typename nk::type_for<dtype_>::type;
    using raw_t = typename type_::raw_t;

    nk_size_t bits_per_element = nk_dtype_bits(dtype_);
    nk_size_t bits_per_row = cols * bits_per_element;
    nk_size_t bytes_per_row = nk::divide_round_up(bits_per_row, NK_BITS_PER_BYTE);
    nk_size_t total_values = rows * bytes_per_row / sizeof(raw_t);

    std::size_t total_dims = total_values * nk::dimensions_per_value<type_>();
    auto result = nk::vector<type_>::try_zeros(total_dims);
#if defined(__cpp_exceptions) && __cpp_exceptions
    if (result.empty() && total_dims > 0) throw std::bad_alloc();
#else
    if (result.empty() && total_dims > 0) std::abort();
#endif
    return result;
}

/**
 *  @brief Measures the performance of a @b dense kernel function using Google Benchmark.
 *
 *  Used by: bench_dot.cpp, bench_spatial.cpp, bench_set.cpp, bench_probability.cpp
 *
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_dense(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate inputs: enough vector pairs to fit within bench_budget
    std::size_t const vectors_count = bench_input_count(2 * bench_dtype_bytes(input_dtype_, dimensions));
    std::vector<input_vector_t> first_vectors(vectors_count), second_vectors(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != vectors_count; ++index) {
        first_vectors[index] = make_vector<input_t>(dimensions);
        second_vectors[index] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, first_vectors[index].values_data(), first_vectors[index].size_values());
        nk::fill_uniform(generator, second_vectors[index].values_data(), second_vectors[index].size_values());
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        output_t output;
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(first_vectors[index].raw_values_data(), second_vectors[index].raw_values_data(), dimensions,
               &output.raw_);
        bm::DoNotOptimize(output);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(2.0 * iterations * first_vectors[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void run_dense(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_dense<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          bench_config.dense_dimensions);
}

//  Batched dot products measurement for packed B matrix API
//  Used by: all bench_cross_*.cpp files
template <nk_dtype_t input_dtype_>
void measure_dots_packed(                                                              //
    bm::State &state,                                                                  //
    typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::dots_pack_kernel_t pack_fn,             //
    typename nk::type_for<input_dtype_>::type::dots_packed_kernel_t kernel,            //
    std::size_t m, std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::dot_result_t;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    // Calculate correct strides for sub-byte types (u4, i4, u1, etc.)
    nk_size_t values_per_row = nk::divide_round_up(k, nk::dimensions_per_value<input_t>());
    nk_size_t a_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t b_stride_bytes = values_per_row * sizeof(typename input_t::raw_t); // B is n x k, so k columns per row
    nk_size_t packed_bytes = packed_size_fn(n, k);

    // Preallocate multiple input sets within bench_budget for input diversity
    std::size_t bytes_per_set = m * a_stride_bytes + n * b_stride_bytes + packed_bytes + m * n * sizeof(raw_output_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct gemm_set_t {
        nk::vector<input_t> a, b;
        std::vector<char> b_packed;
        nk::vector<output_t> c;
    };
    std::vector<gemm_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector_for_matrix<input_dtype_>(m, k);
        s.b = make_vector_for_matrix<input_dtype_>(n, k);
        s.b_packed.resize(packed_bytes, 0);
        s.c = make_vector<output_t>(m * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
        nk::fill_uniform(generator, s.b.values_data(), s.b.size_values());
        pack_fn(s.b.raw_values_data(), n, k, b_stride_bytes, s.b_packed.data(), 0, n);
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.raw_values_data());
        kernel(s.a.raw_values_data(), s.b_packed.data(), s.c.raw_values_data(), //
               m, n, k, a_stride_bytes, n * sizeof(raw_output_t));
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(2.0 * iterations * m * n * k, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_dots_packed(std::string name, //
                     typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn,
                     typename nk::type_for<input_dtype_>::type::dots_pack_kernel_t pack_fn,
                     typename nk::type_for<input_dtype_>::type::dots_packed_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_width) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_dots_packed<input_dtype_>, packed_size_fn, pack_fn, kernel,
                          bench_config.matrix_height, bench_config.matrix_width, bench_config.matrix_depth);
}

/**
 *  @brief Ragged attention benchmark: pack KV once, then time the attention kernel on a
 *         single self-attention segment. Reports scalar-ops as 4 · heads · seq² · dim.
 */
template <nk_dtype_t input_dtype_>
void measure_attention(                                                                  //
    bm::State &state,                                                                    //
    nk_size_t (*packed_size_fn)(nk_size_t, nk_size_t, nk_u32_t const *, nk_size_t),      //
    void (*pack_fn)(typename nk::type_for<input_dtype_>::type::raw_t const *,            //
                    typename nk::type_for<input_dtype_>::type::raw_t const *, nk_size_t, //
                    nk_size_t, nk_u32_t const *, nk_u32_t const *, nk_size_t, nk_size_t, //
                    nk_size_t, void *, nk_size_t, nk_size_t),                            //
    void (*attention_fn)(typename nk::type_for<input_dtype_>::type::raw_t const *,       //
                         void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,      //
                         nk_u32_t const *, nk_size_t, nk_size_t, nk_f32_t, nk_size_t,    //
                         nk_size_t),                                                     //
    nk_size_t head_count, nk_size_t query_length, nk_size_t key_value_length, nk_size_t depth) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    nk_size_t const row_width = head_count * depth;
    nk_size_t const row_stride_bytes = row_width * sizeof(input_t);
    nk_f32_t const scale = 1.0f / std::sqrt((nk_f32_t)depth);
    nk_u32_t const key_value_offsets[2] = {0, (nk_u32_t)key_value_length};
    nk_u32_t const key_value_lengths[1] = {(nk_u32_t)key_value_length};
    nk_u32_t const query_offsets[2] = {0, (nk_u32_t)query_length};

    auto queries = make_vector<input_t>(query_length * row_width);
    auto keys = make_vector<input_t>(key_value_length * row_width);
    auto values = make_vector<input_t>(key_value_length * row_width);
    auto output = make_vector<nk_f32_t>(query_length * row_width);
    auto generator = make_random_engine();
    nk::fill_uniform(generator, queries.values_data(), queries.size_values());
    nk::fill_uniform(generator, keys.values_data(), keys.size_values());
    nk::fill_uniform(generator, values.values_data(), values.size_values());

    auto key_value_packed = make_vector<char>(packed_size_fn(head_count, depth, key_value_lengths, 1));
    pack_fn(keys.raw_values_data(), values.raw_values_data(), head_count, depth, key_value_offsets, key_value_lengths,
            1, row_stride_bytes, row_stride_bytes, key_value_packed.raw_values_data(), 0, head_count);

    std::size_t iterations = 0;
    for (auto _ : state) {
        attention_fn(queries.raw_values_data(), key_value_packed.raw_values_data(), output.raw_values_data(),
                     head_count, head_count, depth, query_offsets, row_stride_bytes, row_width * sizeof(nk_f32_t),
                     scale, 0, 0);
        ++iterations;
        bm::DoNotOptimize(output.raw_values_data());
    }
    state.counters["scalar-ops"] = bm::Counter(4.0 * iterations * head_count * query_length * key_value_length * depth,
                                               bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_attention(                                                                      //
    std::string name,                                                                    //
    nk_size_t (*packed_size_fn)(nk_size_t, nk_size_t, nk_u32_t const *, nk_size_t),      //
    void (*pack_fn)(typename nk::type_for<input_dtype_>::type::raw_t const *,            //
                    typename nk::type_for<input_dtype_>::type::raw_t const *, nk_size_t, //
                    nk_size_t, nk_u32_t const *, nk_u32_t const *, nk_size_t, nk_size_t, //
                    nk_size_t, void *, nk_size_t, nk_size_t),                            //
    void (*attention_fn)(typename nk::type_for<input_dtype_>::type::raw_t const *,       //
                         void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,      //
                         nk_u32_t const *, nk_size_t, nk_size_t, nk_f32_t, nk_size_t,    //
                         nk_size_t)) {
    // Attention is a soft-kNN: each query attends over the KV "database". Mapping the shared matrix
    // dimensions accordingly — height = KV context length, width = head dimension (its 128 default
    // matches a real head), depth = query length (set to 1 for the decode shape).

    nk_size_t const head_count = 8;
    nk_size_t const head_dim = bench_config.matrix_width > 0 ? bench_config.matrix_width : 128;
    nk_size_t const key_value_length = bench_config.matrix_height > 0 ? bench_config.matrix_height : 1024;
    nk_size_t const query_length = bench_config.matrix_depth > 0 ? bench_config.matrix_depth : 1024;
    std::string bench_name = name + "<" + std::to_string(head_count) + "h_" + std::to_string(query_length) + "q_" +
                             std::to_string(key_value_length) + "kv_" + std::to_string(head_dim) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_attention<input_dtype_>, packed_size_fn, pack_fn, attention_fn,
                          head_count, query_length, key_value_length, head_dim);
}

template <nk_dtype_t input_dtype_>
void measure_angulars_packed(                                                          //
    bm::State &state,                                                                  //
    typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::dots_pack_kernel_t pack_fn,             //
    typename nk::type_for<input_dtype_>::type::angulars_packed_kernel_t kernel,        //
    std::size_t m, std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::angular_result_t;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    nk_size_t values_per_row = nk::divide_round_up(k, nk::dimensions_per_value<input_t>());
    nk_size_t a_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t b_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t packed_bytes = packed_size_fn(n, k);

    std::size_t bytes_per_set = m * a_stride_bytes + n * b_stride_bytes + packed_bytes + m * n * sizeof(raw_output_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct gemm_set_t {
        nk::vector<input_t> a, b;
        std::vector<char> b_packed;
        nk::vector<output_t> c;
    };
    std::vector<gemm_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector_for_matrix<input_dtype_>(m, k);
        s.b = make_vector_for_matrix<input_dtype_>(n, k);
        s.b_packed.resize(packed_bytes, 0);
        s.c = make_vector<output_t>(m * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
        nk::fill_uniform(generator, s.b.values_data(), s.b.size_values());
        pack_fn(s.b.raw_values_data(), n, k, b_stride_bytes, s.b_packed.data(), 0, n);
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.raw_values_data());
        kernel(s.a.raw_values_data(), s.b_packed.data(), s.c.raw_values_data(), //
               m, n, k, a_stride_bytes, n * sizeof(raw_output_t));
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(2.0 * iterations * m * n * k, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_angulars_packed(std::string name, //
                         typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn,
                         typename nk::type_for<input_dtype_>::type::dots_pack_kernel_t pack_fn,
                         typename nk::type_for<input_dtype_>::type::angulars_packed_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_width) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_angulars_packed<input_dtype_>, packed_size_fn, pack_fn, kernel,
                          bench_config.matrix_height, bench_config.matrix_width, bench_config.matrix_depth);
}

template <nk_dtype_t input_dtype_>
void measure_euclideans_packed(                                                        //
    bm::State &state,                                                                  //
    typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::dots_pack_kernel_t pack_fn,             //
    typename nk::type_for<input_dtype_>::type::euclideans_packed_kernel_t kernel,      //
    std::size_t m, std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::euclidean_result_t;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    nk_size_t values_per_row = nk::divide_round_up(k, nk::dimensions_per_value<input_t>());
    nk_size_t a_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t b_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t packed_bytes = packed_size_fn(n, k);

    std::size_t bytes_per_set = m * a_stride_bytes + n * b_stride_bytes + packed_bytes + m * n * sizeof(raw_output_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct gemm_set_t {
        nk::vector<input_t> a, b;
        std::vector<char> b_packed;
        nk::vector<output_t> c;
    };
    std::vector<gemm_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector_for_matrix<input_dtype_>(m, k);
        s.b = make_vector_for_matrix<input_dtype_>(n, k);
        s.b_packed.resize(packed_bytes, 0);
        s.c = make_vector<output_t>(m * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
        nk::fill_uniform(generator, s.b.values_data(), s.b.size_values());
        pack_fn(s.b.raw_values_data(), n, k, b_stride_bytes, s.b_packed.data(), 0, n);
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.raw_values_data());
        kernel(s.a.raw_values_data(), s.b_packed.data(), s.c.raw_values_data(), //
               m, n, k, a_stride_bytes, n * sizeof(raw_output_t));
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(2.0 * iterations * m * n * k, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_euclideans_packed(std::string name, //
                           typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn,
                           typename nk::type_for<input_dtype_>::type::dots_pack_kernel_t pack_fn,
                           typename nk::type_for<input_dtype_>::type::euclideans_packed_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_width) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_euclideans_packed<input_dtype_>, packed_size_fn, pack_fn, kernel,
                          bench_config.matrix_height, bench_config.matrix_width, bench_config.matrix_depth);
}

template <nk_dtype_t input_dtype_>
void measure_dots_symmetric(                                                   //
    bm::State &state,                                                          //
    typename nk::type_for<input_dtype_>::type::dots_symmetric_kernel_t kernel, //
    std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::dot_result_t;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    // Calculate correct strides for sub-byte types (u4, i4, u1, etc.)
    nk_size_t input_values_per_row = nk::divide_round_up(k, nk::dimensions_per_value<input_t>());
    nk_size_t input_stride_bytes = input_values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t output_stride_bytes = n * sizeof(raw_output_t);

    // Preallocate multiple input sets within bench_budget
    std::size_t bytes_per_set = n * input_stride_bytes + n * n * sizeof(raw_output_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct syrk_set_t {
        nk::vector<input_t> a;
        nk::vector<output_t> c;
    };
    std::vector<syrk_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector_for_matrix<input_dtype_>(n, k);
        s.c = make_vector<output_t>(n * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.raw_values_data());
        kernel(s.a.raw_values_data(), n, k, input_stride_bytes, //
               s.c.raw_values_data(), output_stride_bytes, 0, n);
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(1.0 * iterations * n * (n + 1) * k, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_dots_symmetric(std::string name, //
                        typename nk::type_for<input_dtype_>::type::dots_symmetric_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_dots_symmetric<input_dtype_>, //
                          kernel, bench_config.matrix_height, bench_config.matrix_depth);
}

template <nk_dtype_t input_dtype_>
void measure_angulars_symmetric(                                                   //
    bm::State &state,                                                              //
    typename nk::type_for<input_dtype_>::type::angulars_symmetric_kernel_t kernel, //
    std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::angular_result_t;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    nk_size_t input_values_per_row = nk::divide_round_up(k, nk::dimensions_per_value<input_t>());
    nk_size_t input_stride_bytes = input_values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t output_stride_bytes = n * sizeof(raw_output_t);

    std::size_t bytes_per_set = n * input_stride_bytes + n * n * sizeof(raw_output_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct syrk_set_t {
        nk::vector<input_t> a;
        nk::vector<output_t> c;
    };
    std::vector<syrk_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector_for_matrix<input_dtype_>(n, k);
        s.c = make_vector<output_t>(n * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.raw_values_data());
        kernel(s.a.raw_values_data(), n, k, input_stride_bytes, //
               s.c.raw_values_data(), output_stride_bytes, 0, n);
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(1.0 * iterations * n * (n + 1) * k, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_angulars_symmetric(std::string name, //
                            typename nk::type_for<input_dtype_>::type::angulars_symmetric_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_angulars_symmetric<input_dtype_>, //
                          kernel, bench_config.matrix_height, bench_config.matrix_depth);
}

template <nk_dtype_t input_dtype_>
void measure_euclideans_symmetric(                                                   //
    bm::State &state,                                                                //
    typename nk::type_for<input_dtype_>::type::euclideans_symmetric_kernel_t kernel, //
    std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::euclidean_result_t;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    nk_size_t input_values_per_row = nk::divide_round_up(k, nk::dimensions_per_value<input_t>());
    nk_size_t input_stride_bytes = input_values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t output_stride_bytes = n * sizeof(raw_output_t);

    std::size_t bytes_per_set = n * input_stride_bytes + n * n * sizeof(raw_output_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct syrk_set_t {
        nk::vector<input_t> a;
        nk::vector<output_t> c;
    };
    std::vector<syrk_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector_for_matrix<input_dtype_>(n, k);
        s.c = make_vector<output_t>(n * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.raw_values_data());
        kernel(s.a.raw_values_data(), n, k, input_stride_bytes, //
               s.c.raw_values_data(), output_stride_bytes, 0, n);
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(1.0 * iterations * n * (n + 1) * k, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_euclideans_symmetric(std::string name, //
                              typename nk::type_for<input_dtype_>::type::euclideans_symmetric_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_euclideans_symmetric<input_dtype_>, //
                          kernel, bench_config.matrix_height, bench_config.matrix_depth);
}

/**
 *  @brief Measure packed Hamming distance computation.
 *  Used by: all bench_cross_*.cpp files
 */
template <nk_dtype_t input_dtype_>
void measure_hammings_packed(                                                              //
    bm::State &state,                                                                      //
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
    nk_size_t a_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t b_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t packed_bytes = packed_size_fn(n, k);

    // Preallocate multiple input sets within bench_budget
    std::size_t bytes_per_set = m * a_stride_bytes + n * b_stride_bytes + packed_bytes + m * n * sizeof(raw_output_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct hamming_set_t {
        nk::vector<input_t> a, b;
        std::vector<char> b_packed;
        nk::vector<output_t> c;
    };
    std::vector<hamming_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector<input_t>(m * k);
        s.b = make_vector<input_t>(n * k);
        s.b_packed.resize(packed_bytes, 0);
        s.c = make_vector<output_t>(m * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
        nk::fill_uniform(generator, s.b.values_data(), s.b.size_values());
        pack_fn(s.b.raw_values_data(), n, k, b_stride_bytes, s.b_packed.data(), 0, n);
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.raw_values_data());
        kernel(s.a.raw_values_data(), s.b_packed.data(), s.c.raw_values_data(), //
               m, n, k, a_stride_bytes, n * sizeof(raw_output_t));
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(1.0 * iterations * m * n * k, bm::Counter::kIsRate);
}

/**
 *  @brief Measure symmetric Hamming distance matrix computation.
 */
template <nk_dtype_t input_dtype_>
void measure_hammings_symmetric(                                                   //
    bm::State &state,                                                              //
    typename nk::type_for<input_dtype_>::type::hammings_symmetric_kernel_t kernel, //
    std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::hamming_result_t;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    // Calculate correct strides for binary data (k is in bits)
    nk_size_t input_values_per_row = nk::divide_round_up(k, 8);
    nk_size_t input_stride_bytes = input_values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t output_stride_bytes = n * sizeof(raw_output_t);

    // Preallocate multiple input sets within bench_budget
    std::size_t bytes_per_set = n * input_stride_bytes + n * n * sizeof(raw_output_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct hamming_sym_set_t {
        nk::vector<input_t> a;
        nk::vector<output_t> c;
    };
    std::vector<hamming_sym_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector<input_t>(n * k);
        s.c = make_vector<output_t>(n * n);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.raw_values_data());
        kernel(s.a.raw_values_data(), n, k, input_stride_bytes, //
               s.c.raw_values_data(), output_stride_bytes, 0, n);
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(1.0 * iterations * n * (n + 1) * k / 2.0, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_hammings_packed(std::string name, //
                         typename nk::type_for<input_dtype_>::type::hammings_pack_size_kernel_t packed_size_fn,
                         typename nk::type_for<input_dtype_>::type::hammings_pack_kernel_t pack_fn,
                         typename nk::type_for<input_dtype_>::type::hammings_packed_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_width) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_hammings_packed<input_dtype_>, packed_size_fn, pack_fn, kernel,
                          bench_config.matrix_height, bench_config.matrix_width, bench_config.matrix_depth);
}

template <nk_dtype_t input_dtype_>
void run_hammings_symmetric(std::string name, //
                            typename nk::type_for<input_dtype_>::type::hammings_symmetric_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_hammings_symmetric<input_dtype_>, //
                          kernel, bench_config.matrix_height, bench_config.matrix_depth);
}

/**
 *  @brief Measure packed Jaccard distance matrix computation.
 */
template <nk_dtype_t input_dtype_>
void measure_jaccards_packed(                                                              //
    bm::State &state,                                                                      //
    typename nk::type_for<input_dtype_>::type::jaccards_pack_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::jaccards_pack_kernel_t pack_fn,             //
    typename nk::type_for<input_dtype_>::type::jaccards_packed_kernel_t kernel,            //
    std::size_t m, std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;

    nk_size_t values_per_row = nk::divide_round_up(k, 8);
    nk_size_t a_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t b_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t packed_bytes = packed_size_fn(n, k);

    std::size_t bytes_per_set = m * a_stride_bytes + n * b_stride_bytes + packed_bytes + m * n * sizeof(nk_f32_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct jaccards_set_t {
        nk::vector<input_t> a, b;
        std::vector<char> b_packed;
        std::vector<nk_f32_t> c;
    };
    std::vector<jaccards_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector<input_t>(m * k);
        s.b = make_vector<input_t>(n * k);
        s.b_packed.resize(packed_bytes, 0);
        s.c.resize(m * n, 0);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
        nk::fill_uniform(generator, s.b.values_data(), s.b.size_values());
        pack_fn(s.b.raw_values_data(), n, k, b_stride_bytes, s.b_packed.data(), 0, n);
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.data());
        kernel(s.a.raw_values_data(), s.b_packed.data(), s.c.data(), //
               m, n, k, a_stride_bytes, n * sizeof(nk_f32_t));
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(1.0 * iterations * m * n * k, bm::Counter::kIsRate);
}

/**
 *  @brief Measure symmetric Jaccard distance matrix computation.
 */
template <nk_dtype_t input_dtype_>
void measure_jaccards_symmetric(                                                   //
    bm::State &state,                                                              //
    typename nk::type_for<input_dtype_>::type::jaccards_symmetric_kernel_t kernel, //
    std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;

    nk_size_t input_values_per_row = nk::divide_round_up(k, 8);
    nk_size_t input_stride_bytes = input_values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t output_stride_bytes = n * sizeof(nk_f32_t);

    std::size_t bytes_per_set = n * input_stride_bytes + n * n * sizeof(nk_f32_t);
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct jaccards_sym_set_t {
        nk::vector<input_t> a;
        std::vector<nk_f32_t> c;
    };
    std::vector<jaccards_sym_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.a = make_vector<input_t>(n * k);
        s.c.resize(n * n, 0);
        nk::fill_uniform(generator, s.a.values_data(), s.a.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        bm::DoNotOptimize(s.c.data());
        kernel(s.a.raw_values_data(), n, k, input_stride_bytes, //
               s.c.data(), output_stride_bytes, 0, n);
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(1.0 * iterations * n * (n + 1) * k / 2.0, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_jaccards_packed(std::string name, //
                         typename nk::type_for<input_dtype_>::type::jaccards_pack_size_kernel_t packed_size_fn,
                         typename nk::type_for<input_dtype_>::type::jaccards_pack_kernel_t pack_fn,
                         typename nk::type_for<input_dtype_>::type::jaccards_packed_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_width) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_jaccards_packed<input_dtype_>, packed_size_fn, pack_fn, kernel,
                          bench_config.matrix_height, bench_config.matrix_width, bench_config.matrix_depth);
}

template <nk_dtype_t input_dtype_>
void run_jaccards_symmetric(std::string name, //
                            typename nk::type_for<input_dtype_>::type::jaccards_symmetric_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.matrix_height) + "x" +
                             std::to_string(bench_config.matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_jaccards_symmetric<input_dtype_>, //
                          kernel, bench_config.matrix_height, bench_config.matrix_depth);
}

// Forward declarations for benchmark registration functions (defined in separate files)
void bench_dot();
void bench_spatial();
void bench_set();
void bench_curved();
void bench_probability();
void bench_each();
void bench_trigonometry();
void bench_geospatial();
void bench_mesh();
void bench_sparse();
void bench_sparse_dot();
void bench_cast();
void bench_reduce();
void bench_maxsim();

// Forward declarations for cross/batch operations (ISA-family files)
void bench_cross_serial();
void bench_cross_x86();
void bench_cross_amx();
void bench_cross_arm();
void bench_cross_sme();
void bench_cross_blas();
void bench_cross_rvv();
void bench_cross_power();
void bench_cross_wasm();
void bench_cross_loongarch();
