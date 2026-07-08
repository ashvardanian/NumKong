/**
 *  @brief Trigonometry benchmarks (sin, cos, atan).
 *  @file bench/bench_trigonometry.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/trigonometry.h"

#include "bench.hpp"

template <typename scalar_type_>
struct sin_with_stl {
    scalar_type_ operator()(scalar_type_ x) const { return std::sin(x); }
};

template <typename scalar_type_>
struct cos_with_stl {
    scalar_type_ operator()(scalar_type_ x) const { return std::cos(x); }
};

template <typename scalar_type_>
struct atan_with_stl {
    scalar_type_ operator()(scalar_type_ x) const { return std::atan(x); }
};

template <typename scalar_type_, typename kernel_type_>
void elementwise_with_stl(scalar_type_ const *ins, nk_size_t n, scalar_type_ *outs) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = kernel_type_ {}(ins[i]);
}

/**
 *  @brief Measures the performance of trigonometric operations (sin, cos, atan) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_trigonometry(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate vectors for trigonometric kernels (unary: input + output)
    std::size_t bytes_per_set = bench_dtype_bytes(input_dtype_, 2 * dimensions);
    std::size_t const vectors_count = bench_input_count(bytes_per_set);
    std::vector<input_vector_t> input_a(vectors_count), output(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != vectors_count; ++index) {
        input_a[index] = make_vector<input_t>(dimensions);
        output[index] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, input_a[index].values_data(), dimensions);
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(input_a[index].raw_values_data(), dimensions, output[index].raw_values_data());
        bm::ClobberMemory();
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(1.0 * iterations * input_a[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_trigonometry(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_trigonometry<input_dtype_, kernel_type_ *>, kernel,
                          bench_config.dense_dimensions);
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_rope(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    std::size_t const half_dim = dimensions / 2;
    std::size_t const vectors_count = bench_input_count(bench_dtype_bytes(input_dtype_, dimensions));
    std::vector<nk::vector<input_t>> tokens(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t i = 0; i < vectors_count; ++i) {
        tokens[i] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, tokens[i].values_data(), tokens[i].size_values());
    }
    std::vector<nk_f32_t> cos_grid(half_dim, 0.5f), sin_grid(half_dim, 0.5f); // [rows=1, half_dim]
    std::size_t const stride = dimensions * sizeof(typename input_t::raw_t);

    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (vectors_count - 1);
        auto *ptr = tokens[index].raw_values_data();
        kernel(ptr, ptr, cos_grid.data(), sin_grid.data(), 1, 1, half_dim, stride, stride, 1.0f);
        bm::ClobberMemory();
        ++iterations;
    }
    state.counters["bytes"] = bm::Counter(1.0 * iterations * tokens[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_rope(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_rope<input_dtype_, kernel_type_ *>, kernel,
                          bench_config.dense_dimensions);
}

void bench_trigonometry() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;

#if NK_TARGET_NEON
    run_trigonometry<f32_k>("trig_sin_f32_neon", nk_trig_sin_f32_neon);
    run_trigonometry<f32_k>("trig_cos_f32_neon", nk_trig_cos_f32_neon);
    run_trigonometry<f32_k>("trig_atan_f32_neon", nk_trig_atan_f32_neon);
    run_trigonometry<f64_k>("trig_sin_f64_neon", nk_trig_sin_f64_neon);
    run_trigonometry<f64_k>("trig_cos_f64_neon", nk_trig_cos_f64_neon);
    run_trigonometry<f64_k>("trig_atan_f64_neon", nk_trig_atan_f64_neon);
#endif

#if NK_TARGET_HASWELL
    run_trigonometry<f32_k>("trig_sin_f32_haswell", nk_trig_sin_f32_haswell);
    run_trigonometry<f32_k>("trig_cos_f32_haswell", nk_trig_cos_f32_haswell);
    run_trigonometry<f32_k>("trig_atan_f32_haswell", nk_trig_atan_f32_haswell);
    run_trigonometry<f64_k>("trig_sin_f64_haswell", nk_trig_sin_f64_haswell);
    run_trigonometry<f64_k>("trig_cos_f64_haswell", nk_trig_cos_f64_haswell);
    run_trigonometry<f64_k>("trig_atan_f64_haswell", nk_trig_atan_f64_haswell);
    run_rope<f32_k>("trig_rope_f32_haswell", nk_trig_rope_f32_haswell);
    run_rope<bf16_k>("trig_rope_bf16_haswell", nk_trig_rope_bf16_haswell);
    run_rope<e4m3_k>("trig_rope_e4m3_haswell", nk_trig_rope_e4m3_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_trigonometry<f32_k>("trig_sin_f32_skylake", nk_trig_sin_f32_skylake);
    run_trigonometry<f32_k>("trig_cos_f32_skylake", nk_trig_cos_f32_skylake);
    run_trigonometry<f32_k>("trig_atan_f32_skylake", nk_trig_atan_f32_skylake);
    run_trigonometry<f64_k>("trig_sin_f64_skylake", nk_trig_sin_f64_skylake);
    run_trigonometry<f64_k>("trig_cos_f64_skylake", nk_trig_cos_f64_skylake);
    run_trigonometry<f64_k>("trig_atan_f64_skylake", nk_trig_atan_f64_skylake);
    run_trigonometry<f16_k>("trig_sin_f16_skylake", nk_trig_sin_f16_skylake);
    run_trigonometry<f16_k>("trig_cos_f16_skylake", nk_trig_cos_f16_skylake);
    run_trigonometry<f16_k>("trig_atan_f16_skylake", nk_trig_atan_f16_skylake);
    run_rope<f32_k>("trig_rope_f32_skylake", nk_trig_rope_f32_skylake);
    run_rope<bf16_k>("trig_rope_bf16_skylake", nk_trig_rope_bf16_skylake);
    run_rope<e4m3_k>("trig_rope_e4m3_skylake", nk_trig_rope_e4m3_skylake);
#endif

#if NK_TARGET_V128RELAXED
    run_trigonometry<f32_k>("trig_sin_f32_v128relaxed", nk_trig_sin_f32_v128relaxed);
    run_trigonometry<f32_k>("trig_cos_f32_v128relaxed", nk_trig_cos_f32_v128relaxed);
    run_trigonometry<f32_k>("trig_atan_f32_v128relaxed", nk_trig_atan_f32_v128relaxed);
    run_trigonometry<f64_k>("trig_sin_f64_v128relaxed", nk_trig_sin_f64_v128relaxed);
    run_trigonometry<f64_k>("trig_cos_f64_v128relaxed", nk_trig_cos_f64_v128relaxed);
    run_trigonometry<f64_k>("trig_atan_f64_v128relaxed", nk_trig_atan_f64_v128relaxed);
#endif

    // STL baselines
    run_trigonometry<f32_k>("trig_sin_f32_stl", elementwise_with_stl<nk_f32_t, sin_with_stl<nk_f32_t>>);
    run_trigonometry<f32_k>("trig_cos_f32_stl", elementwise_with_stl<nk_f32_t, cos_with_stl<nk_f32_t>>);
    run_trigonometry<f32_k>("trig_atan_f32_stl", elementwise_with_stl<nk_f32_t, atan_with_stl<nk_f32_t>>);
    run_trigonometry<f64_k>("trig_sin_f64_stl", elementwise_with_stl<nk_f64_t, sin_with_stl<nk_f64_t>>);
    run_trigonometry<f64_k>("trig_cos_f64_stl", elementwise_with_stl<nk_f64_t, cos_with_stl<nk_f64_t>>);
    run_trigonometry<f64_k>("trig_atan_f64_stl", elementwise_with_stl<nk_f64_t, atan_with_stl<nk_f64_t>>);

    // Serial fallbacks
    run_trigonometry<f32_k>("trig_sin_f32_serial", nk_trig_sin_f32_serial);
    run_trigonometry<f32_k>("trig_cos_f32_serial", nk_trig_cos_f32_serial);
    run_trigonometry<f32_k>("trig_atan_f32_serial", nk_trig_atan_f32_serial);
    run_trigonometry<f64_k>("trig_sin_f64_serial", nk_trig_sin_f64_serial);
    run_trigonometry<f64_k>("trig_cos_f64_serial", nk_trig_cos_f64_serial);
    run_trigonometry<f64_k>("trig_atan_f64_serial", nk_trig_atan_f64_serial);
    run_trigonometry<f16_k>("trig_sin_f16_serial", nk_trig_sin_f16_serial);
    run_trigonometry<f16_k>("trig_cos_f16_serial", nk_trig_cos_f16_serial);
    run_trigonometry<f16_k>("trig_atan_f16_serial", nk_trig_atan_f16_serial);
    run_rope<f32_k>("trig_rope_f32_serial", nk_trig_rope_f32_serial);
    run_rope<bf16_k>("trig_rope_bf16_serial", nk_trig_rope_bf16_serial);
    run_rope<e4m3_k>("trig_rope_e4m3_serial", nk_trig_rope_e4m3_serial);
}
