/**
 *  @brief Sparse operations benchmarks (sparse_intersect).
 *  @file bench/bench_sparse.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/sparse.h"

#include "bench.hpp"

/**
 *  @brief Measures the performance of a @b sparse (set intersection) kernel function using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param first_size The number of elements in the first (smaller) set.
 *  @param second_size The number of elements in the second (larger) set.
 *  @param intersection_size The expected number of common elements between the sets.
 */
template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_sparse(bm::State &state, kernel_type_ kernel, std::size_t first_size, std::size_t second_size,
                    std::size_t intersection_size) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate sorted unique set vectors
    std::size_t bytes_per_set = bench_dtype_bytes(input_dtype_, first_size + second_size);
    std::size_t const vectors_count = bench_input_count(bytes_per_set);
    std::vector<input_vector_t> first_vectors(vectors_count), second_vectors(vectors_count);
    auto generator = make_random_engine();

    auto max_val = input_t(first_size * second_size / intersection_size);
    for (std::size_t index = 0; index != vectors_count; ++index) {
        first_vectors[index] = make_vector<input_t>(first_size);
        second_vectors[index] = make_vector<input_t>(second_size);
        nk::fill_sorted_unique(generator, first_vectors[index].values_data(), first_size, max_val);
        nk::fill_sorted_unique(generator, second_vectors[index].values_data(), second_size, max_val);
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        nk_size_t count;
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(first_vectors[index].raw_values_data(), second_vectors[index].raw_values_data(), first_size, second_size,
               nullptr, &count);
        bm::DoNotOptimize(count);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(
        1.0 * iterations * (first_vectors[0].size_bytes() + second_vectors[0].size_bytes()), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_sparse(std::string name, kernel_type_ *kernel) {
    std::size_t intersection_size = static_cast<std::size_t>(
        std::min(bench_config.sparse_first_length, bench_config.sparse_second_length) *
        bench_config.sparse_intersection_share);
    std::string bench_name = name + "<|A|=" + std::to_string(bench_config.sparse_first_length) +
                             ",|B|=" + std::to_string(bench_config.sparse_second_length) +
                             ",|A^B|=" + std::to_string(intersection_size) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_sparse<input_dtype_, kernel_type_ *>, kernel,
                          bench_config.sparse_first_length, bench_config.sparse_second_length, intersection_size);
}

void bench_sparse() {
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t u64_k = nk_u64_k;

#if NK_TARGET_NEON
    run_sparse<u16_k>("sparse_intersect_u16_neon", nk_sparse_intersect_u16_neon);
    run_sparse<u32_k>("sparse_intersect_u32_neon", nk_sparse_intersect_u32_neon);
    run_sparse<u64_k>("sparse_intersect_u64_neon", nk_sparse_intersect_u64_neon);
#endif

#if NK_TARGET_SVE2
    run_sparse<u16_k>("sparse_intersect_u16_sve2", nk_sparse_intersect_u16_sve2);
    run_sparse<u32_k>("sparse_intersect_u32_sve2", nk_sparse_intersect_u32_sve2);
    run_sparse<u64_k>("sparse_intersect_u64_sve2", nk_sparse_intersect_u64_sve2);
#endif

#if NK_TARGET_ICELAKE
    run_sparse<u16_k>("sparse_intersect_u16_icelake", nk_sparse_intersect_u16_icelake);
    run_sparse<u32_k>("sparse_intersect_u32_icelake", nk_sparse_intersect_u32_icelake);
    run_sparse<u64_k>("sparse_intersect_u64_icelake", nk_sparse_intersect_u64_icelake);
#endif

#if NK_TARGET_TURIN
    run_sparse<u16_k>("sparse_intersect_u16_turin", nk_sparse_intersect_u16_turin);
    run_sparse<u32_k>("sparse_intersect_u32_turin", nk_sparse_intersect_u32_turin);
    run_sparse<u64_k>("sparse_intersect_u64_turin", nk_sparse_intersect_u64_turin);
#endif

    // Serial fallbacks
    run_sparse<u16_k>("sparse_intersect_u16_serial", nk_sparse_intersect_u16_serial);
    run_sparse<u32_k>("sparse_intersect_u32_serial", nk_sparse_intersect_u32_serial);
    run_sparse<u64_k>("sparse_intersect_u64_serial", nk_sparse_intersect_u64_serial);
}

/**
 *  @brief Measures the performance of a @b sparse dot-product kernel function using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param first_size The number of elements in the first (smaller) set.
 *  @param second_size The number of elements in the second (larger) set.
 *  @param intersection_size The expected number of common elements between the sets.
 */
template <nk_dtype_t index_dtype_, nk_dtype_t weight_dtype_, typename kernel_type_ = void>
void measure_sparse_dot(bm::State &state, kernel_type_ kernel, std::size_t first_size, std::size_t second_size,
                        std::size_t intersection_size) {

    using index_t = typename nk::type_for<index_dtype_>::type;
    using weight_t = typename nk::type_for<weight_dtype_>::type;
    using product_t = typename weight_t::dot_result_t;
    using index_vector_t = nk::vector<index_t>;
    using weight_vector_t = nk::vector<weight_t>;

    std::size_t bytes_per_set = bench_dtype_bytes(index_dtype_, first_size + second_size);
    std::size_t const vectors_count = bench_input_count(bytes_per_set);
    std::vector<index_vector_t> first_indices(vectors_count), second_indices(vectors_count);
    std::vector<weight_vector_t> first_weights(vectors_count), second_weights(vectors_count);
    auto generator = make_random_engine();
    auto max_val = index_t(first_size * second_size / intersection_size);

    for (std::size_t index = 0; index != vectors_count; ++index) {
        first_indices[index] = make_vector<index_t>(first_size);
        second_indices[index] = make_vector<index_t>(second_size);
        first_weights[index] = make_vector<weight_t>(first_size);
        second_weights[index] = make_vector<weight_t>(second_size);

        nk::fill_sorted_unique(generator, first_indices[index].values_data(), first_size, max_val);
        nk::fill_sorted_unique(generator, second_indices[index].values_data(), second_size, max_val);

        // Fill weights with small random values
        nk::fill_uniform(generator, first_weights[index].values_data(), first_size);
        nk::fill_uniform(generator, second_weights[index].values_data(), second_size);
    }

    product_t product;
    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const idx = iterations & (vectors_count - 1);
        kernel(first_indices[idx].raw_values_data(), second_indices[idx].raw_values_data(),
               first_weights[idx].raw_values_data(), second_weights[idx].raw_values_data(), first_size, second_size,
               &product.raw_);
        bm::DoNotOptimize(product);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(1.0 * iterations *
                                              (first_indices[0].size_bytes() + second_indices[0].size_bytes() +
                                               first_weights[0].size_bytes() + second_weights[0].size_bytes()),
                                          bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t index_dtype_, nk_dtype_t weight_dtype_, typename kernel_type_ = void>
void run_sparse_dot(std::string name, kernel_type_ *kernel) {
    std::size_t intersection_size = static_cast<std::size_t>(
        std::min(bench_config.sparse_first_length, bench_config.sparse_second_length) *
        bench_config.sparse_intersection_share);
    std::string bench_name = name + "<|A|=" + std::to_string(bench_config.sparse_first_length) +
                             ",|B|=" + std::to_string(bench_config.sparse_second_length) +
                             ",|A^B|=" + std::to_string(intersection_size) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_sparse_dot<index_dtype_, weight_dtype_, kernel_type_ *>, kernel,
                          bench_config.sparse_first_length, bench_config.sparse_second_length, intersection_size);
}

void bench_sparse_dot() {
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;

#if NK_TARGET_SVE2
    run_sparse_dot<u32_k, f32_k>("sparse_dot_u32f32_sve2", nk_sparse_dot_u32f32_sve2);
#if NK_TARGET_SVEBFDOT
    run_sparse_dot<u16_k, bf16_k>("sparse_dot_u16bf16_sve2", nk_sparse_dot_u16bf16_sve2);
#endif
#endif

#if NK_TARGET_ICELAKE
    run_sparse_dot<u32_k, f32_k>("sparse_dot_u32f32_icelake", nk_sparse_dot_u32f32_icelake);
#endif

#if NK_TARGET_HASWELL
    run_sparse_dot<u32_k, f32_k>("sparse_dot_u32f32_haswell", nk_sparse_dot_u32f32_haswell);
#endif

#if NK_TARGET_TURIN
    run_sparse_dot<u16_k, bf16_k>("sparse_dot_u16bf16_turin", nk_sparse_dot_u16bf16_turin);
    run_sparse_dot<u32_k, f32_k>("sparse_dot_u32f32_turin", nk_sparse_dot_u32f32_turin);
#endif

    // Serial fallbacks
    run_sparse_dot<u16_k, bf16_k>("sparse_dot_u16bf16_serial", nk_sparse_dot_u16bf16_serial);
    run_sparse_dot<u32_k, f32_k>("sparse_dot_u32f32_serial", nk_sparse_dot_u32f32_serial);
}
