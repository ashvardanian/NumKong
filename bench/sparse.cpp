/**
 *  @file bench/sparse.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Sparse operations benchmarks, sparse_intersect.
 */

#include "numkong/sparse.h"

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

/**
 *  @brief Measures a @b sparse kernel, set intersection.
 *  @param[inout] loop The timed loop, which takes the counters.
 *  @param[in] kernel The kernel function to benchmark.
 *  @param[in] first_size The number of elements in the first, smaller, set.
 *  @param[in] second_size The number of elements in the second, larger, set.
 *  @param[in] intersection_size The expected number of common elements between the sets.
 */
template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_sparse(loop_t &loop, environment_t const &env, kernel_type_ kernel, std::size_t first_size,
                    std::size_t second_size, std::size_t intersection_size) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate sorted unique set vectors
    std::size_t const vectors_count = input_sets_count(dtype_bytes(input_dtype_, first_size + second_size));
    std::vector<input_vector_t> first_vectors(vectors_count), second_vectors(vectors_count);
    std::mt19937 generator(env.settings.seed.value);

    auto max_val = input_t(first_size * second_size / intersection_size);
    for (std::size_t index = 0; index != vectors_count; ++index) {
        first_vectors[index] = make_vector<input_t>(first_size);
        second_vectors[index] = make_vector<input_t>(second_size);
        nk::fill_sorted_unique(generator, first_vectors[index].values_data(), first_size, max_val);
        nk::fill_sorted_unique(generator, second_vectors[index].values_data(), second_size, max_val);
    }

    // Benchmark loop
    for (std::size_t call : loop) {
        nk_size_t count;
        std::size_t const index = call & (vectors_count - 1);
        if (!succeeded(loop, kernel(first_vectors[index].raw_values_data(), second_vectors[index].raw_values_data(),
                                    first_size, second_size, nullptr, &count, nullptr)))
            break;
        do_not_optimize(count);
    }

    loop.byte_rate(first_vectors[0].size_bytes() + second_vectors[0].size_bytes());
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_sparse(environment_t const &env, std::string name, kernel_type_ *kernel) {
    std::size_t const intersection_size = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::min(env.settings.sparse_first_count, env.settings.sparse_second_count) *
                                    env.settings.sparse_intersection_share));
    std::string bench_name = name + "<|A|=" + std::to_string(env.settings.sparse_first_count) +
                             ",|B|=" + std::to_string(env.settings.sparse_second_count) +
                             ",|A^B|=" + std::to_string(intersection_size) + ">";
    run_benchmark(env, bench_name, measure_sparse<input_dtype_, kernel_type_ *>, kernel,
                  env.settings.sparse_first_count, env.settings.sparse_second_count, intersection_size);
}

void bench_sparse(environment_t const &env) {
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t u64_k = nk_u64_k;

#if NUMKONG_TARGET_NEON
    if (section(env, "Sparse Operations NEON", nk_cap_neon_k)) {
        run_sparse<u16_k>(env, "sparse_intersect_u16_neon", nk_sparse_intersect_u16_neon);
        run_sparse<u32_k>(env, "sparse_intersect_u32_neon", nk_sparse_intersect_u32_neon);
        run_sparse<u64_k>(env, "sparse_intersect_u64_neon", nk_sparse_intersect_u64_neon);
    }
#endif

#if NUMKONG_TARGET_SVE2
    if (section(env, "Sparse Operations SVE2", nk_cap_sve2_k)) {
        run_sparse<u16_k>(env, "sparse_intersect_u16_sve2", nk_sparse_intersect_u16_sve2);
        run_sparse<u32_k>(env, "sparse_intersect_u32_sve2", nk_sparse_intersect_u32_sve2);
        run_sparse<u64_k>(env, "sparse_intersect_u64_sve2", nk_sparse_intersect_u64_sve2);
    }
#endif

#if NUMKONG_TARGET_ICELAKE
    if (section(env, "Sparse Operations Ice Lake", nk_cap_icelake_k)) {
        run_sparse<u16_k>(env, "sparse_intersect_u16_icelake", nk_sparse_intersect_u16_icelake);
        run_sparse<u32_k>(env, "sparse_intersect_u32_icelake", nk_sparse_intersect_u32_icelake);
        run_sparse<u64_k>(env, "sparse_intersect_u64_icelake", nk_sparse_intersect_u64_icelake);
    }
#endif

#if NUMKONG_TARGET_TURIN
    if (section(env, "Sparse Operations Turin", nk_cap_turin_k)) {
        run_sparse<u16_k>(env, "sparse_intersect_u16_turin", nk_sparse_intersect_u16_turin);
        run_sparse<u32_k>(env, "sparse_intersect_u32_turin", nk_sparse_intersect_u32_turin);
        run_sparse<u64_k>(env, "sparse_intersect_u64_turin", nk_sparse_intersect_u64_turin);
    }
#endif

    // Serial fallbacks
    section(env, "Sparse Operations Serial", nk_cap_serial_k);
    run_sparse<u16_k>(env, "sparse_intersect_u16_serial", nk_sparse_intersect_u16_serial);
    run_sparse<u32_k>(env, "sparse_intersect_u32_serial", nk_sparse_intersect_u32_serial);
    run_sparse<u64_k>(env, "sparse_intersect_u64_serial", nk_sparse_intersect_u64_serial);
}

/**
 *  @brief Measures a @b sparse dot-product kernel.
 *  @param[inout] loop The timed loop, which takes the counters.
 *  @param[in] kernel The kernel function to benchmark.
 *  @param[in] first_size The number of elements in the first, smaller, set.
 *  @param[in] second_size The number of elements in the second, larger, set.
 *  @param[in] intersection_size The expected number of common elements between the sets.
 */
template <nk_dtype_t index_dtype_, nk_dtype_t weight_dtype_, typename kernel_type_ = void>
void measure_sparse_dot(loop_t &loop, environment_t const &env, kernel_type_ kernel, std::size_t first_size,
                        std::size_t second_size, std::size_t intersection_size) {

    using index_t = typename nk::type_for<index_dtype_>::type;
    using weight_t = typename nk::type_for<weight_dtype_>::type;
    using product_t = typename weight_t::dot_result_t;
    using index_vector_t = nk::vector<index_t>;
    using weight_vector_t = nk::vector<weight_t>;

    std::size_t const vectors_count = input_sets_count(dtype_bytes(index_dtype_, first_size + second_size));
    std::vector<index_vector_t> first_indices(vectors_count), second_indices(vectors_count);
    std::vector<weight_vector_t> first_weights(vectors_count), second_weights(vectors_count);
    std::mt19937 generator(env.settings.seed.value);
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
    for (std::size_t call : loop) {
        std::size_t const idx = call & (vectors_count - 1);
        if (!succeeded(loop, kernel(first_indices[idx].raw_values_data(), second_indices[idx].raw_values_data(),
                                    first_weights[idx].raw_values_data(), second_weights[idx].raw_values_data(),
                                    first_size, second_size, &product.raw_, nullptr)))
            break;
        do_not_optimize(product);
    }

    loop.byte_rate(first_indices[0].size_bytes() + second_indices[0].size_bytes() + first_weights[0].size_bytes() +
                   second_weights[0].size_bytes());
}

template <nk_dtype_t index_dtype_, nk_dtype_t weight_dtype_, typename kernel_type_ = void>
void run_sparse_dot(environment_t const &env, std::string name, kernel_type_ *kernel) {
    std::size_t const intersection_size = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::min(env.settings.sparse_first_count, env.settings.sparse_second_count) *
                                    env.settings.sparse_intersection_share));
    std::string bench_name = name + "<|A|=" + std::to_string(env.settings.sparse_first_count) +
                             ",|B|=" + std::to_string(env.settings.sparse_second_count) +
                             ",|A^B|=" + std::to_string(intersection_size) + ">";
    run_benchmark(env, bench_name, measure_sparse_dot<index_dtype_, weight_dtype_, kernel_type_ *>, kernel,
                  env.settings.sparse_first_count, env.settings.sparse_second_count, intersection_size);
}

void bench_sparse_dot(environment_t const &env) {
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;

#if NUMKONG_TARGET_SVE2
    if (section(env, "Sparse Operations SVE2", nk_cap_sve2_k)) {
        run_sparse_dot<u32_k, f32_k>(env, "sparse_dot_u32f32_sve2", nk_sparse_dot_u32f32_sve2);
    }
#endif

#if NUMKONG_TARGET_SVE2 && NUMKONG_TARGET_SVEBFDOT
    if (section(env, "Sparse Operations SVE2 BF16", nk_cap_sve2_k | nk_cap_svebfdot_k)) {
        run_sparse_dot<u16_k, bf16_k>(env, "sparse_dot_u16bf16_sve2", nk_sparse_dot_u16bf16_sve2);
    }
#endif

#if NUMKONG_TARGET_ICELAKE
    if (section(env, "Sparse Operations Ice Lake", nk_cap_icelake_k)) {
        run_sparse_dot<u32_k, f32_k>(env, "sparse_dot_u32f32_icelake", nk_sparse_dot_u32f32_icelake);
    }
#endif

#if NUMKONG_TARGET_HASWELL
    if (section(env, "Sparse Operations Haswell", nk_cap_haswell_k)) {
        run_sparse_dot<u32_k, f32_k>(env, "sparse_dot_u32f32_haswell", nk_sparse_dot_u32f32_haswell);
    }
#endif

#if NUMKONG_TARGET_TURIN
    if (section(env, "Sparse Operations Turin", nk_cap_turin_k)) {
        run_sparse_dot<u16_k, bf16_k>(env, "sparse_dot_u16bf16_turin", nk_sparse_dot_u16bf16_turin);
        run_sparse_dot<u32_k, f32_k>(env, "sparse_dot_u32f32_turin", nk_sparse_dot_u32f32_turin);
    }
#endif

    // Serial fallbacks
    section(env, "Sparse Operations Serial", nk_cap_serial_k);
    run_sparse_dot<u16_k, bf16_k>(env, "sparse_dot_u16bf16_serial", nk_sparse_dot_u16bf16_serial);
    run_sparse_dot<u32_k, f32_k>(env, "sparse_dot_u32f32_serial", nk_sparse_dot_u32f32_serial);
}

} // namespace ashvardanian::numkong::bench
