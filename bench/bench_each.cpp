/**
 *  @brief Elementwise operations benchmarks (fma, blend, sum, scale).
 *  @file bench/bench_each.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include <cstring> // std::memset

#include "numkong/capabilities.h" // nk_kernel_kind_t
#include "numkong/each.h"         // nk_each_swiglu_*
#include "numkong/trigonometry.h" // nk_each_rope_*

#include "bench.hpp"

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void sum_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    int const ni = static_cast<int>(n);
    cblas_scopy(ni, a, 1, result, 1);
    cblas_saxpy(ni, 1.0f, b, 1, result, 1);
}

void sum_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    int const ni = static_cast<int>(n);
    cblas_dcopy(ni, a, 1, result, 1);
    cblas_daxpy(ni, 1.0, b, 1, result, 1);
}

void blend_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                         nk_f32_t *result) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(nk_f32_t));
    if (*alpha != 0) cblas_saxpy(ni, *alpha, a, 1, result, 1);
    if (*beta != 0) cblas_saxpy(ni, *beta, b, 1, result, 1);
}

void blend_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                         nk_f64_t *result) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(nk_f64_t));
    if (*alpha != 0) cblas_daxpy(ni, *alpha, a, 1, result, 1);
    if (*beta != 0) cblas_daxpy(ni, *beta, b, 1, result, 1);
}

#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

/**
 *  @brief Measures the performance of elementwise operations (sum, blend, fma, scale) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_, nk_dtype_t alpha_dtype_, typename kernel_type_ = void>
void measure_each(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using alpha_t = typename nk::type_for<alpha_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Scaling parameters for FMA/blend/scale kernels
    alpha_t alpha = alpha_t(0.2f);
    alpha_t beta = alpha_t(0.3f);

    // Preallocate vectors for different kernel types:
    // - sum: input_a, input_c -> output
    // - blend: input_a, input_c + alpha, beta -> output
    // - fma: input_a, input_b, input_c + alpha, beta -> output
    // - scale: input_a + alpha, beta -> output
    std::size_t bytes_per_set = bench_dtype_bytes(input_dtype_, 4 * dimensions);
    std::size_t const vectors_count = bench_input_count(bytes_per_set);
    std::vector<input_vector_t> input_a(vectors_count), input_b(vectors_count);
    std::vector<input_vector_t> input_c(vectors_count), output(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != vectors_count; ++index) {
        input_a[index] = make_vector<input_t>(dimensions);
        input_b[index] = make_vector<input_t>(dimensions);
        input_c[index] = make_vector<input_t>(dimensions);
        output[index] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, input_a[index].values_data(), dimensions);
        std::fill(input_b[index].values_data(), input_b[index].values_data() + dimensions,
                  input_t(2)); // Small constant
        nk::fill_uniform(generator, input_c[index].values_data(), dimensions);
    }

    // Benchmark loop with kernel dispatch
    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (vectors_count - 1);
        if constexpr (kernel_kind_ == nk_kernel_each_blend_k) {
            kernel(input_a[index].raw_values_data(), input_c[index].raw_values_data(), dimensions, &alpha.raw_,
                   &beta.raw_, output[index].raw_values_data());
        }
        else if constexpr (kernel_kind_ == nk_kernel_each_fma_k) {
            kernel(input_a[index].raw_values_data(), input_b[index].raw_values_data(), input_c[index].raw_values_data(),
                   dimensions, &alpha.raw_, &beta.raw_, output[index].raw_values_data());
        }
        else if constexpr (kernel_kind_ == nk_kernel_each_sum_k) {
            kernel(input_a[index].raw_values_data(), input_c[index].raw_values_data(), dimensions,
                   output[index].raw_values_data());
        }
        else if constexpr (kernel_kind_ == nk_kernel_each_scale_k) {
            kernel(input_a[index].raw_values_data(), dimensions, &alpha.raw_, &beta.raw_,
                   output[index].raw_values_data());
        }
        bm::ClobberMemory();
        iterations++;
    }

    std::size_t bytes_per_call = input_a[0].size_bytes();
    if constexpr (kernel_kind_ == nk_kernel_each_blend_k) bytes_per_call *= 2;
    else if constexpr (kernel_kind_ == nk_kernel_each_fma_k) bytes_per_call *= 3;
    else if constexpr (kernel_kind_ == nk_kernel_each_sum_k) bytes_per_call *= 2;
    else if constexpr (kernel_kind_ == nk_kernel_each_scale_k) bytes_per_call *= 1;

    state.counters["bytes"] = bm::Counter(1.0 * iterations * bytes_per_call, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_ = nk_kernel_unknown_k,
          nk_dtype_t alpha_dtype_ = nk_dtype_unknown_k, typename kernel_type_ = void>
void run_each(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_each<input_dtype_, kernel_kind_, alpha_dtype_, kernel_type_ *>,
                          kernel, bench_config.dense_dimensions);
}

/**
 *  @brief Measures a fused SwiGLU kernel (single row, gate ⊙ up).
 */
template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_swiglu(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    std::size_t const vectors_count = bench_input_count(bench_dtype_bytes(input_dtype_, 3 * dimensions));
    std::vector<nk::vector<input_t>> gate(vectors_count), up(vectors_count), output(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t i = 0; i < vectors_count; ++i) {
        gate[i] = make_vector<input_t>(dimensions);
        up[i] = make_vector<input_t>(dimensions);
        output[i] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, gate[i].values_data(), gate[i].size_values());
        nk::fill_uniform(generator, up[i].values_data(), up[i].size_values());
    }
    std::size_t const stride = dimensions * sizeof(typename input_t::raw_t);

    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(gate[index].raw_values_data(), up[index].raw_values_data(), output[index].raw_values_data(), 1,
               dimensions, stride, stride, stride, 1.0f);
        bm::ClobberMemory();
        ++iterations;
    }
    state.counters["bytes"] = bm::Counter(2.0 * iterations * gate[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_swiglu(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_swiglu<input_dtype_, kernel_type_ *>, kernel,
                          bench_config.dense_dimensions);
}

/**
 *  @brief Measures an in-place NeoX split-half RoPE kernel (single head, all pairs rotated).
 */
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

void bench_each() {
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t i16_k = nk_i16_k;
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t i64_k = nk_i64_k;
    constexpr nk_dtype_t u64_k = nk_u64_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t f32c_k = nk_f32c_k;
    constexpr nk_dtype_t f64c_k = nk_f64c_k;

    constexpr nk_kernel_kind_t fma_k = nk_kernel_each_fma_k;
    constexpr nk_kernel_kind_t blend_k = nk_kernel_each_blend_k;
    constexpr nk_kernel_kind_t sum_k = nk_kernel_each_sum_k;
    constexpr nk_kernel_kind_t scale_k = nk_kernel_each_scale_k;

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
    run_each<f32_k, sum_k, f32_k>("sum_f32_with_blas", sum_f32_with_blas);
    run_each<f32_k, blend_k, f32_k>("each_blend_f32_with_blas", blend_f32_with_blas);
    run_each<f64_k, sum_k, f64_k>("sum_f64_with_blas", sum_f64_with_blas);
    run_each<f64_k, blend_k, f64_k>("each_blend_f64_with_blas", blend_f64_with_blas);
#endif

#if NK_TARGET_NEON
    // f64
    run_each<f64_k, sum_k, f64_k>("each_sum_f64_neon", nk_each_sum_f64_neon);
    run_each<f64_k, scale_k, f64_k>("each_scale_f64_neon", nk_each_scale_f64_neon);
    run_each<f64_k, blend_k, f64_k>("each_blend_f64_neon", nk_each_blend_f64_neon);
    run_each<f64_k, fma_k, f64_k>("each_fma_f64_neon", nk_each_fma_f64_neon);
    // f32
    run_each<f32_k, sum_k, f32_k>("each_sum_f32_neon", nk_each_sum_f32_neon);
    run_each<f32_k, scale_k, f32_k>("each_scale_f32_neon", nk_each_scale_f32_neon);
    run_each<f32_k, blend_k, f32_k>("each_blend_f32_neon", nk_each_blend_f32_neon);
    run_each<f32_k, fma_k, f32_k>("each_fma_f32_neon", nk_each_fma_f32_neon);
    // e4m3
    run_each<e4m3_k, sum_k, f32_k>("each_sum_e4m3_neon", nk_each_sum_e4m3_neon);
    run_each<e4m3_k, scale_k, f32_k>("each_scale_e4m3_neon", nk_each_scale_e4m3_neon);
    run_each<e4m3_k, blend_k, f32_k>("each_blend_e4m3_neon", nk_each_blend_e4m3_neon);
    run_each<e4m3_k, fma_k, f32_k>("each_fma_e4m3_neon", nk_each_fma_e4m3_neon);
    // e5m2
    run_each<e5m2_k, sum_k, f32_k>("each_sum_e5m2_neon", nk_each_sum_e5m2_neon);
    run_each<e5m2_k, scale_k, f32_k>("each_scale_e5m2_neon", nk_each_scale_e5m2_neon);
    run_each<e5m2_k, blend_k, f32_k>("each_blend_e5m2_neon", nk_each_blend_e5m2_neon);
    run_each<e5m2_k, fma_k, f32_k>("each_fma_e5m2_neon", nk_each_fma_e5m2_neon);
    // u8, i8
    run_each<u8_k, sum_k, f32_k>("each_sum_u8_neon", nk_each_sum_u8_neon);
    run_each<i8_k, sum_k, f32_k>("each_sum_i8_neon", nk_each_sum_i8_neon);
    // i16, u16
    run_each<i16_k, sum_k, f32_k>("each_sum_i16_neon", nk_each_sum_i16_neon);
    run_each<i16_k, scale_k, f32_k>("each_scale_i16_neon", nk_each_scale_i16_neon);
    run_each<i16_k, fma_k, f32_k>("each_fma_i16_neon", nk_each_fma_i16_neon);
    run_each<u16_k, sum_k, f32_k>("each_sum_u16_neon", nk_each_sum_u16_neon);
    run_each<u16_k, scale_k, f32_k>("each_scale_u16_neon", nk_each_scale_u16_neon);
    run_each<u16_k, fma_k, f32_k>("each_fma_u16_neon", nk_each_fma_u16_neon);
    // i32, u32
    run_each<i32_k, sum_k, f64_k>("each_sum_i32_neon", nk_each_sum_i32_neon);
    run_each<i32_k, scale_k, f64_k>("each_scale_i32_neon", nk_each_scale_i32_neon);
    run_each<i32_k, fma_k, f64_k>("each_fma_i32_neon", nk_each_fma_i32_neon);
    run_each<u32_k, sum_k, f64_k>("each_sum_u32_neon", nk_each_sum_u32_neon);
    run_each<u32_k, scale_k, f64_k>("each_scale_u32_neon", nk_each_scale_u32_neon);
    run_each<u32_k, fma_k, f64_k>("each_fma_u32_neon", nk_each_fma_u32_neon);
    // i64, u64
    run_each<i64_k, sum_k, f64_k>("each_sum_i64_neon", nk_each_sum_i64_neon);
    run_each<i64_k, scale_k, f64_k>("each_scale_i64_neon", nk_each_scale_i64_neon);
    run_each<i64_k, fma_k, f64_k>("each_fma_i64_neon", nk_each_fma_i64_neon);
    run_each<u64_k, sum_k, f64_k>("each_sum_u64_neon", nk_each_sum_u64_neon);
    run_each<u64_k, scale_k, f64_k>("each_scale_u64_neon", nk_each_scale_u64_neon);
    run_each<u64_k, fma_k, f64_k>("each_fma_u64_neon", nk_each_fma_u64_neon);
    // complex
    run_each<f32c_k, scale_k, f32c_k>("each_scale_f32c_neon", nk_each_scale_f32c_neon);
    run_each<f32c_k, blend_k, f32c_k>("each_blend_f32c_neon", nk_each_blend_f32c_neon);
    run_each<f32c_k, fma_k, f32c_k>("each_fma_f32c_neon", nk_each_fma_f32c_neon);
    run_each<f64c_k, scale_k, f64c_k>("each_scale_f64c_neon", nk_each_scale_f64c_neon);
    run_each<f64c_k, blend_k, f64c_k>("each_blend_f64c_neon", nk_each_blend_f64c_neon);
    run_each<f64c_k, fma_k, f64c_k>("each_fma_f64c_neon", nk_each_fma_f64c_neon);
#endif

#if NK_TARGET_NEONHALF
    // f16
    run_each<f16_k, sum_k, f32_k>("each_sum_f16_neonhalf", nk_each_sum_f16_neonhalf);
    run_each<f16_k, scale_k, f32_k>("each_scale_f16_neonhalf", nk_each_scale_f16_neonhalf);
    run_each<f16_k, blend_k, f32_k>("each_blend_f16_neonhalf", nk_each_blend_f16_neonhalf);
    run_each<f16_k, fma_k, f32_k>("each_fma_f16_neonhalf", nk_each_fma_f16_neonhalf);
    // u8
    run_each<u8_k, scale_k, f32_k>("each_scale_u8_neonhalf", nk_each_scale_u8_neonhalf);
    run_each<u8_k, blend_k, f32_k>("each_blend_u8_neonhalf", nk_each_blend_u8_neonhalf);
    // i8
    run_each<i8_k, scale_k, f32_k>("each_scale_i8_neonhalf", nk_each_scale_i8_neonhalf);
    run_each<i8_k, blend_k, f32_k>("each_blend_i8_neonhalf", nk_each_blend_i8_neonhalf);
#endif

#if NK_TARGET_NEONBFDOT
    run_each<bf16_k, sum_k, f32_k>("each_sum_bf16_neonbfdot", nk_each_sum_bf16_neonbfdot);
    run_each<bf16_k, scale_k, f32_k>("each_scale_bf16_neonbfdot", nk_each_scale_bf16_neonbfdot);
    run_each<bf16_k, blend_k, f32_k>("each_blend_bf16_neonbfdot", nk_each_blend_bf16_neonbfdot);
    run_each<bf16_k, fma_k, f32_k>("each_fma_bf16_neonbfdot", nk_each_fma_bf16_neonbfdot);
#endif

#if NK_TARGET_HASWELL
    // f64
    run_each<f64_k, sum_k, f64_k>("each_sum_f64_haswell", nk_each_sum_f64_haswell);
    run_each<f64_k, scale_k, f64_k>("each_scale_f64_haswell", nk_each_scale_f64_haswell);
    run_each<f64_k, blend_k, f64_k>("each_blend_f64_haswell", nk_each_blend_f64_haswell);
    run_each<f64_k, fma_k, f64_k>("each_fma_f64_haswell", nk_each_fma_f64_haswell);
    // f32
    run_each<f32_k, sum_k, f32_k>("each_sum_f32_haswell", nk_each_sum_f32_haswell);
    run_each<f32_k, scale_k, f32_k>("each_scale_f32_haswell", nk_each_scale_f32_haswell);
    run_each<f32_k, blend_k, f32_k>("each_blend_f32_haswell", nk_each_blend_f32_haswell);
    run_each<f32_k, fma_k, f32_k>("each_fma_f32_haswell", nk_each_fma_f32_haswell);
    // f16
    run_each<f16_k, sum_k, f32_k>("each_sum_f16_haswell", nk_each_sum_f16_haswell);
    run_each<f16_k, scale_k, f32_k>("each_scale_f16_haswell", nk_each_scale_f16_haswell);
    run_each<f16_k, blend_k, f32_k>("each_blend_f16_haswell", nk_each_blend_f16_haswell);
    run_each<f16_k, fma_k, f32_k>("each_fma_f16_haswell", nk_each_fma_f16_haswell);
    // bf16
    run_each<bf16_k, sum_k, f32_k>("each_sum_bf16_haswell", nk_each_sum_bf16_haswell);
    run_each<bf16_k, scale_k, f32_k>("each_scale_bf16_haswell", nk_each_scale_bf16_haswell);
    run_swiglu<f32_k>("swiglu_f32_haswell", nk_each_swiglu_f32_haswell);
    run_swiglu<bf16_k>("swiglu_bf16_haswell", nk_each_swiglu_bf16_haswell);
    run_swiglu<e4m3_k>("swiglu_e4m3_haswell", nk_each_swiglu_e4m3_haswell);
    run_rope<f32_k>("rope_f32_haswell", nk_each_rope_f32_haswell);
    run_rope<bf16_k>("rope_bf16_haswell", nk_each_rope_bf16_haswell);
    run_rope<e4m3_k>("rope_e4m3_haswell", nk_each_rope_e4m3_haswell);
    run_each<bf16_k, blend_k, f32_k>("each_blend_bf16_haswell", nk_each_blend_bf16_haswell);
    run_each<bf16_k, fma_k, f32_k>("each_fma_bf16_haswell", nk_each_fma_bf16_haswell);
    // e4m3
    run_each<e4m3_k, sum_k, f32_k>("each_sum_e4m3_haswell", nk_each_sum_e4m3_haswell);
    run_each<e4m3_k, scale_k, f32_k>("each_scale_e4m3_haswell", nk_each_scale_e4m3_haswell);
    run_each<e4m3_k, blend_k, f32_k>("each_blend_e4m3_haswell", nk_each_blend_e4m3_haswell);
    run_each<e4m3_k, fma_k, f32_k>("each_fma_e4m3_haswell", nk_each_fma_e4m3_haswell);
    // e5m2
    run_each<e5m2_k, sum_k, f32_k>("each_sum_e5m2_haswell", nk_each_sum_e5m2_haswell);
    run_each<e5m2_k, scale_k, f32_k>("each_scale_e5m2_haswell", nk_each_scale_e5m2_haswell);
    run_each<e5m2_k, blend_k, f32_k>("each_blend_e5m2_haswell", nk_each_blend_e5m2_haswell);
    run_each<e5m2_k, fma_k, f32_k>("each_fma_e5m2_haswell", nk_each_fma_e5m2_haswell);
    // i8, u8
    run_each<i8_k, sum_k, f32_k>("each_sum_i8_haswell", nk_each_sum_i8_haswell);
    run_each<i8_k, scale_k, f32_k>("each_scale_i8_haswell", nk_each_scale_i8_haswell);
    run_each<i8_k, blend_k, f32_k>("each_blend_i8_haswell", nk_each_blend_i8_haswell);
    run_each<i8_k, fma_k, f32_k>("each_fma_i8_haswell", nk_each_fma_i8_haswell);
    run_each<u8_k, sum_k, f32_k>("each_sum_u8_haswell", nk_each_sum_u8_haswell);
    run_each<u8_k, scale_k, f32_k>("each_scale_u8_haswell", nk_each_scale_u8_haswell);
    run_each<u8_k, blend_k, f32_k>("each_blend_u8_haswell", nk_each_blend_u8_haswell);
    run_each<u8_k, fma_k, f32_k>("each_fma_u8_haswell", nk_each_fma_u8_haswell);
    // i16, u16
    run_each<i16_k, sum_k, f32_k>("each_sum_i16_haswell", nk_each_sum_i16_haswell);
    run_each<i16_k, scale_k, f32_k>("each_scale_i16_haswell", nk_each_scale_i16_haswell);
    run_each<i16_k, fma_k, f32_k>("each_fma_i16_haswell", nk_each_fma_i16_haswell);
    run_each<u16_k, sum_k, f32_k>("each_sum_u16_haswell", nk_each_sum_u16_haswell);
    run_each<u16_k, scale_k, f32_k>("each_scale_u16_haswell", nk_each_scale_u16_haswell);
    run_each<u16_k, fma_k, f32_k>("each_fma_u16_haswell", nk_each_fma_u16_haswell);
    // i32, u32
    run_each<i32_k, sum_k, f64_k>("each_sum_i32_haswell", nk_each_sum_i32_haswell);
    run_each<i32_k, scale_k, f64_k>("each_scale_i32_haswell", nk_each_scale_i32_haswell);
    run_each<i32_k, fma_k, f64_k>("each_fma_i32_haswell", nk_each_fma_i32_haswell);
    run_each<u32_k, sum_k, f64_k>("each_sum_u32_haswell", nk_each_sum_u32_haswell);
    run_each<u32_k, scale_k, f64_k>("each_scale_u32_haswell", nk_each_scale_u32_haswell);
    run_each<u32_k, fma_k, f64_k>("each_fma_u32_haswell", nk_each_fma_u32_haswell);
    // complex
    run_each<f32c_k, scale_k, f32c_k>("each_scale_f32c_haswell", nk_each_scale_f32c_haswell);
    run_each<f32c_k, blend_k, f32c_k>("each_blend_f32c_haswell", nk_each_blend_f32c_haswell);
    run_each<f32c_k, fma_k, f32c_k>("each_fma_f32c_haswell", nk_each_fma_f32c_haswell);
    run_each<f64c_k, scale_k, f64c_k>("each_scale_f64c_haswell", nk_each_scale_f64c_haswell);
    run_each<f64c_k, blend_k, f64c_k>("each_blend_f64c_haswell", nk_each_blend_f64c_haswell);
    run_each<f64c_k, fma_k, f64c_k>("each_fma_f64c_haswell", nk_each_fma_f64c_haswell);
#endif

#if NK_TARGET_SKYLAKE
    // f64
    run_each<f64_k, sum_k, f64_k>("each_sum_f64_skylake", nk_each_sum_f64_skylake);
    run_each<f64_k, scale_k, f64_k>("each_scale_f64_skylake", nk_each_scale_f64_skylake);
    run_each<f64_k, blend_k, f64_k>("each_blend_f64_skylake", nk_each_blend_f64_skylake);
    run_each<f64_k, fma_k, f64_k>("each_fma_f64_skylake", nk_each_fma_f64_skylake);
    // f32
    run_each<f32_k, sum_k, f32_k>("each_sum_f32_skylake", nk_each_sum_f32_skylake);
    run_each<f32_k, scale_k, f32_k>("each_scale_f32_skylake", nk_each_scale_f32_skylake);
    run_each<f32_k, blend_k, f32_k>("each_blend_f32_skylake", nk_each_blend_f32_skylake);
    run_each<f32_k, fma_k, f32_k>("each_fma_f32_skylake", nk_each_fma_f32_skylake);
    // f16
    run_each<f16_k, scale_k, f32_k>("each_scale_f16_skylake", nk_each_scale_f16_skylake);
    run_each<f16_k, blend_k, f32_k>("each_blend_f16_skylake", nk_each_blend_f16_skylake);
    run_each<f16_k, fma_k, f32_k>("each_fma_f16_skylake", nk_each_fma_f16_skylake);
    // bf16
    run_each<bf16_k, sum_k, f32_k>("each_sum_bf16_skylake", nk_each_sum_bf16_skylake);
    run_each<bf16_k, scale_k, f32_k>("each_scale_bf16_skylake", nk_each_scale_bf16_skylake);
    run_swiglu<f32_k>("swiglu_f32_skylake", nk_each_swiglu_f32_skylake);
    run_swiglu<bf16_k>("swiglu_bf16_skylake", nk_each_swiglu_bf16_skylake);
    run_swiglu<e4m3_k>("swiglu_e4m3_skylake", nk_each_swiglu_e4m3_skylake);
    run_rope<f32_k>("rope_f32_skylake", nk_each_rope_f32_skylake);
    run_rope<bf16_k>("rope_bf16_skylake", nk_each_rope_bf16_skylake);
    run_rope<e4m3_k>("rope_e4m3_skylake", nk_each_rope_e4m3_skylake);
    run_each<bf16_k, blend_k, f32_k>("each_blend_bf16_skylake", nk_each_blend_bf16_skylake);
    run_each<bf16_k, fma_k, f32_k>("each_fma_bf16_skylake", nk_each_fma_bf16_skylake);
    // e4m3
    run_each<e4m3_k, sum_k, f32_k>("each_sum_e4m3_skylake", nk_each_sum_e4m3_skylake);
    run_each<e4m3_k, scale_k, f32_k>("each_scale_e4m3_skylake", nk_each_scale_e4m3_skylake);
    run_each<e4m3_k, blend_k, f32_k>("each_blend_e4m3_skylake", nk_each_blend_e4m3_skylake);
    run_each<e4m3_k, fma_k, f32_k>("each_fma_e4m3_skylake", nk_each_fma_e4m3_skylake);
    // e5m2
    run_each<e5m2_k, sum_k, f32_k>("each_sum_e5m2_skylake", nk_each_sum_e5m2_skylake);
    run_each<e5m2_k, scale_k, f32_k>("each_scale_e5m2_skylake", nk_each_scale_e5m2_skylake);
    run_each<e5m2_k, blend_k, f32_k>("each_blend_e5m2_skylake", nk_each_blend_e5m2_skylake);
    run_each<e5m2_k, fma_k, f32_k>("each_fma_e5m2_skylake", nk_each_fma_e5m2_skylake);
    // i8, u8
    run_each<i8_k, scale_k, f32_k>("each_scale_i8_skylake", nk_each_scale_i8_skylake);
    run_each<i8_k, fma_k, f32_k>("each_fma_i8_skylake", nk_each_fma_i8_skylake);
    run_each<u8_k, scale_k, f32_k>("each_scale_u8_skylake", nk_each_scale_u8_skylake);
    run_each<u8_k, fma_k, f32_k>("each_fma_u8_skylake", nk_each_fma_u8_skylake);
    // i16, u16
    run_each<i16_k, scale_k, f32_k>("each_scale_i16_skylake", nk_each_scale_i16_skylake);
    run_each<i16_k, fma_k, f32_k>("each_fma_i16_skylake", nk_each_fma_i16_skylake);
    run_each<u16_k, scale_k, f32_k>("each_scale_u16_skylake", nk_each_scale_u16_skylake);
    run_each<u16_k, fma_k, f32_k>("each_fma_u16_skylake", nk_each_fma_u16_skylake);
    // i32, u32
    run_each<i32_k, scale_k, f64_k>("each_scale_i32_skylake", nk_each_scale_i32_skylake);
    run_each<i32_k, fma_k, f64_k>("each_fma_i32_skylake", nk_each_fma_i32_skylake);
    run_each<u32_k, scale_k, f64_k>("each_scale_u32_skylake", nk_each_scale_u32_skylake);
    run_each<u32_k, fma_k, f64_k>("each_fma_u32_skylake", nk_each_fma_u32_skylake);
    // i64, u64
    run_each<i64_k, scale_k, f64_k>("each_scale_i64_skylake", nk_each_scale_i64_skylake);
    run_each<i64_k, fma_k, f64_k>("each_fma_i64_skylake", nk_each_fma_i64_skylake);
    run_each<u64_k, scale_k, f64_k>("each_scale_u64_skylake", nk_each_scale_u64_skylake);
    run_each<u64_k, fma_k, f64_k>("each_fma_u64_skylake", nk_each_fma_u64_skylake);
    // complex
    run_each<f32c_k, scale_k, f32c_k>("each_scale_f32c_skylake", nk_each_scale_f32c_skylake);
    run_each<f32c_k, blend_k, f32c_k>("each_blend_f32c_skylake", nk_each_blend_f32c_skylake);
    run_each<f32c_k, fma_k, f32c_k>("each_fma_f32c_skylake", nk_each_fma_f32c_skylake);
    run_each<f64c_k, scale_k, f64c_k>("each_scale_f64c_skylake", nk_each_scale_f64c_skylake);
    run_each<f64c_k, blend_k, f64c_k>("each_blend_f64c_skylake", nk_each_blend_f64c_skylake);
    run_each<f64c_k, fma_k, f64c_k>("each_fma_f64c_skylake", nk_each_fma_f64c_skylake);
#endif

#if NK_TARGET_ICELAKE
    run_each<i8_k, sum_k, f32_k>("each_sum_i8_icelake", nk_each_sum_i8_icelake);
    run_each<u8_k, sum_k, f32_k>("each_sum_u8_icelake", nk_each_sum_u8_icelake);
    run_each<i16_k, sum_k, f32_k>("each_sum_i16_icelake", nk_each_sum_i16_icelake);
    run_each<u16_k, sum_k, f32_k>("each_sum_u16_icelake", nk_each_sum_u16_icelake);
    run_each<i32_k, sum_k, f32_k>("each_sum_i32_icelake", nk_each_sum_i32_icelake);
    run_each<u32_k, sum_k, f32_k>("each_sum_u32_icelake", nk_each_sum_u32_icelake);
    run_each<i64_k, sum_k, f64_k>("each_sum_i64_icelake", nk_each_sum_i64_icelake);
    run_each<u64_k, sum_k, f64_k>("each_sum_u64_icelake", nk_each_sum_u64_icelake);
#endif

#if NK_TARGET_SAPPHIRE
    // i8, u8
    run_each<i8_k, scale_k, f32_k>("each_scale_i8_sapphire", nk_each_scale_i8_sapphire);
    run_each<i8_k, blend_k, f32_k>("each_blend_i8_sapphire", nk_each_blend_i8_sapphire);
    run_each<u8_k, scale_k, f32_k>("each_scale_u8_sapphire", nk_each_scale_u8_sapphire);
    run_each<u8_k, blend_k, f32_k>("each_blend_u8_sapphire", nk_each_blend_u8_sapphire);
    // f16, e4m3
    run_each<f16_k, sum_k, f32_k>("each_sum_f16_sapphire", nk_each_sum_f16_sapphire);
    run_each<e4m3_k, sum_k, f32_k>("each_sum_e4m3_sapphire", nk_each_sum_e4m3_sapphire);
#endif

#if NK_TARGET_RVV
    run_each<f64_k, scale_k, f64_k>("each_scale_f64_rvv", nk_each_scale_f64_rvv);
    run_each<f64_k, fma_k, f64_k>("each_fma_f64_rvv", nk_each_fma_f64_rvv);
    run_each<f64_k, blend_k, f64_k>("each_blend_f64_rvv", nk_each_blend_f64_rvv);
    run_each<f32_k, scale_k, f32_k>("each_scale_f32_rvv", nk_each_scale_f32_rvv);
    run_each<f32_k, fma_k, f32_k>("each_fma_f32_rvv", nk_each_fma_f32_rvv);
    run_each<f32_k, blend_k, f32_k>("each_blend_f32_rvv", nk_each_blend_f32_rvv);
    run_each<f16_k, scale_k, f32_k>("each_scale_f16_rvv", nk_each_scale_f16_rvv);
    run_each<f16_k, fma_k, f32_k>("each_fma_f16_rvv", nk_each_fma_f16_rvv);
    run_each<f16_k, blend_k, f32_k>("each_blend_f16_rvv", nk_each_blend_f16_rvv);
    run_each<bf16_k, scale_k, f32_k>("each_scale_bf16_rvv", nk_each_scale_bf16_rvv);
    run_each<bf16_k, fma_k, f32_k>("each_fma_bf16_rvv", nk_each_fma_bf16_rvv);
    run_each<bf16_k, blend_k, f32_k>("each_blend_bf16_rvv", nk_each_blend_bf16_rvv);
    run_each<i8_k, scale_k, f32_k>("each_scale_i8_rvv", nk_each_scale_i8_rvv);
    run_each<i8_k, fma_k, f32_k>("each_fma_i8_rvv", nk_each_fma_i8_rvv);
    run_each<i8_k, blend_k, f32_k>("each_blend_i8_rvv", nk_each_blend_i8_rvv);
    run_each<u8_k, scale_k, f32_k>("each_scale_u8_rvv", nk_each_scale_u8_rvv);
    run_each<u8_k, fma_k, f32_k>("each_fma_u8_rvv", nk_each_fma_u8_rvv);
    run_each<u8_k, blend_k, f32_k>("each_blend_u8_rvv", nk_each_blend_u8_rvv);
#endif

#if NK_TARGET_V128RELAXED
    run_each<f32_k, sum_k, f32_k>("each_sum_f32_v128relaxed", nk_each_sum_f32_v128relaxed);
    run_each<f32_k, scale_k, f32_k>("each_scale_f32_v128relaxed", nk_each_scale_f32_v128relaxed);
    run_each<f32_k, blend_k, f32_k>("each_blend_f32_v128relaxed", nk_each_blend_f32_v128relaxed);
    run_each<f32_k, fma_k, f32_k>("each_fma_f32_v128relaxed", nk_each_fma_f32_v128relaxed);
    run_each<f16_k, sum_k, f32_k>("each_sum_f16_v128relaxed", nk_each_sum_f16_v128relaxed);
    run_each<f16_k, scale_k, f32_k>("each_scale_f16_v128relaxed", nk_each_scale_f16_v128relaxed);
    run_each<f16_k, blend_k, f32_k>("each_blend_f16_v128relaxed", nk_each_blend_f16_v128relaxed);
    run_each<f16_k, fma_k, f32_k>("each_fma_f16_v128relaxed", nk_each_fma_f16_v128relaxed);
    run_each<bf16_k, sum_k, f32_k>("each_sum_bf16_v128relaxed", nk_each_sum_bf16_v128relaxed);
    run_each<bf16_k, scale_k, f32_k>("each_scale_bf16_v128relaxed", nk_each_scale_bf16_v128relaxed);
    run_each<bf16_k, blend_k, f32_k>("each_blend_bf16_v128relaxed", nk_each_blend_bf16_v128relaxed);
    run_each<bf16_k, fma_k, f32_k>("each_fma_bf16_v128relaxed", nk_each_fma_bf16_v128relaxed);
    run_each<i8_k, sum_k, f32_k>("each_sum_i8_v128relaxed", nk_each_sum_i8_v128relaxed);
    run_each<i8_k, scale_k, f32_k>("each_scale_i8_v128relaxed", nk_each_scale_i8_v128relaxed);
    run_each<i8_k, blend_k, f32_k>("each_blend_i8_v128relaxed", nk_each_blend_i8_v128relaxed);
    run_each<i8_k, fma_k, f32_k>("each_fma_i8_v128relaxed", nk_each_fma_i8_v128relaxed);
    run_each<u8_k, sum_k, f32_k>("each_sum_u8_v128relaxed", nk_each_sum_u8_v128relaxed);
    run_each<u8_k, scale_k, f32_k>("each_scale_u8_v128relaxed", nk_each_scale_u8_v128relaxed);
    run_each<u8_k, blend_k, f32_k>("each_blend_u8_v128relaxed", nk_each_blend_u8_v128relaxed);
    run_each<u8_k, fma_k, f32_k>("each_fma_u8_v128relaxed", nk_each_fma_u8_v128relaxed);
#endif

    // Serial fallbacks — f64
    run_each<f64_k, sum_k, f64_k>("each_sum_f64_serial", nk_each_sum_f64_serial);
    run_each<f64_k, scale_k, f64_k>("each_scale_f64_serial", nk_each_scale_f64_serial);
    run_each<f64_k, blend_k, f64_k>("each_blend_f64_serial", nk_each_blend_f64_serial);
    run_each<f64_k, fma_k, f64_k>("each_fma_f64_serial", nk_each_fma_f64_serial);
    // Serial fallbacks — f32
    run_each<f32_k, sum_k, f32_k>("each_sum_f32_serial", nk_each_sum_f32_serial);
    run_each<f32_k, scale_k, f32_k>("each_scale_f32_serial", nk_each_scale_f32_serial);
    run_each<f32_k, blend_k, f32_k>("each_blend_f32_serial", nk_each_blend_f32_serial);
    run_each<f32_k, fma_k, f32_k>("each_fma_f32_serial", nk_each_fma_f32_serial);
    // Serial fallbacks — f16, bf16
    run_each<f16_k, sum_k, f32_k>("each_sum_f16_serial", nk_each_sum_f16_serial);
    run_each<f16_k, scale_k, f32_k>("each_scale_f16_serial", nk_each_scale_f16_serial);
    run_each<f16_k, blend_k, f32_k>("each_blend_f16_serial", nk_each_blend_f16_serial);
    run_each<f16_k, fma_k, f32_k>("each_fma_f16_serial", nk_each_fma_f16_serial);
    run_each<bf16_k, sum_k, f32_k>("each_sum_bf16_serial", nk_each_sum_bf16_serial);
    run_each<bf16_k, scale_k, f32_k>("each_scale_bf16_serial", nk_each_scale_bf16_serial);
    run_swiglu<f32_k>("swiglu_f32_serial", nk_each_swiglu_f32_serial);
    run_swiglu<bf16_k>("swiglu_bf16_serial", nk_each_swiglu_bf16_serial);
    run_swiglu<e4m3_k>("swiglu_e4m3_serial", nk_each_swiglu_e4m3_serial);
    run_rope<f32_k>("rope_f32_serial", nk_each_rope_f32_serial);
    run_rope<bf16_k>("rope_bf16_serial", nk_each_rope_bf16_serial);
    run_rope<e4m3_k>("rope_e4m3_serial", nk_each_rope_e4m3_serial);
    run_each<bf16_k, blend_k, f32_k>("each_blend_bf16_serial", nk_each_blend_bf16_serial);
    run_each<bf16_k, fma_k, f32_k>("each_fma_bf16_serial", nk_each_fma_bf16_serial);
    // Serial fallbacks — e4m3, e5m2, e2m3, e3m2
    run_each<e4m3_k, sum_k, f32_k>("each_sum_e4m3_serial", nk_each_sum_e4m3_serial);
    run_each<e4m3_k, scale_k, f32_k>("each_scale_e4m3_serial", nk_each_scale_e4m3_serial);
    run_each<e4m3_k, blend_k, f32_k>("each_blend_e4m3_serial", nk_each_blend_e4m3_serial);
    run_each<e4m3_k, fma_k, f32_k>("each_fma_e4m3_serial", nk_each_fma_e4m3_serial);
    run_each<e5m2_k, sum_k, f32_k>("each_sum_e5m2_serial", nk_each_sum_e5m2_serial);
    run_each<e5m2_k, scale_k, f32_k>("each_scale_e5m2_serial", nk_each_scale_e5m2_serial);
    run_each<e5m2_k, blend_k, f32_k>("each_blend_e5m2_serial", nk_each_blend_e5m2_serial);
    run_each<e5m2_k, fma_k, f32_k>("each_fma_e5m2_serial", nk_each_fma_e5m2_serial);
    run_each<e2m3_k, sum_k, f32_k>("each_sum_e2m3_serial", nk_each_sum_e2m3_serial);
    run_each<e2m3_k, scale_k, f32_k>("each_scale_e2m3_serial", nk_each_scale_e2m3_serial);
    run_each<e2m3_k, blend_k, f32_k>("each_blend_e2m3_serial", nk_each_blend_e2m3_serial);
    run_each<e2m3_k, fma_k, f32_k>("each_fma_e2m3_serial", nk_each_fma_e2m3_serial);
    run_each<e3m2_k, sum_k, f32_k>("each_sum_e3m2_serial", nk_each_sum_e3m2_serial);
    run_each<e3m2_k, scale_k, f32_k>("each_scale_e3m2_serial", nk_each_scale_e3m2_serial);
    run_each<e3m2_k, blend_k, f32_k>("each_blend_e3m2_serial", nk_each_blend_e3m2_serial);
    run_each<e3m2_k, fma_k, f32_k>("each_fma_e3m2_serial", nk_each_fma_e3m2_serial);
    // Serial fallbacks — i8, u8
    run_each<i8_k, sum_k, f32_k>("each_sum_i8_serial", nk_each_sum_i8_serial);
    run_each<i8_k, scale_k, f32_k>("each_scale_i8_serial", nk_each_scale_i8_serial);
    run_each<i8_k, blend_k, f32_k>("each_blend_i8_serial", nk_each_blend_i8_serial);
    run_each<i8_k, fma_k, f32_k>("each_fma_i8_serial", nk_each_fma_i8_serial);
    run_each<u8_k, sum_k, f32_k>("each_sum_u8_serial", nk_each_sum_u8_serial);
    run_each<u8_k, scale_k, f32_k>("each_scale_u8_serial", nk_each_scale_u8_serial);
    run_each<u8_k, blend_k, f32_k>("each_blend_u8_serial", nk_each_blend_u8_serial);
    run_each<u8_k, fma_k, f32_k>("each_fma_u8_serial", nk_each_fma_u8_serial);
    // Serial fallbacks — i16, u16
    run_each<i16_k, sum_k, f32_k>("each_sum_i16_serial", nk_each_sum_i16_serial);
    run_each<i16_k, scale_k, f32_k>("each_scale_i16_serial", nk_each_scale_i16_serial);
    run_each<i16_k, blend_k, f32_k>("each_blend_i16_serial", nk_each_blend_i16_serial);
    run_each<i16_k, fma_k, f32_k>("each_fma_i16_serial", nk_each_fma_i16_serial);
    run_each<u16_k, sum_k, f32_k>("each_sum_u16_serial", nk_each_sum_u16_serial);
    run_each<u16_k, scale_k, f32_k>("each_scale_u16_serial", nk_each_scale_u16_serial);
    run_each<u16_k, blend_k, f32_k>("each_blend_u16_serial", nk_each_blend_u16_serial);
    run_each<u16_k, fma_k, f32_k>("each_fma_u16_serial", nk_each_fma_u16_serial);
    // Serial fallbacks — i32, u32
    run_each<i32_k, sum_k, f64_k>("each_sum_i32_serial", nk_each_sum_i32_serial);
    run_each<i32_k, scale_k, f64_k>("each_scale_i32_serial", nk_each_scale_i32_serial);
    run_each<i32_k, blend_k, f64_k>("each_blend_i32_serial", nk_each_blend_i32_serial);
    run_each<i32_k, fma_k, f64_k>("each_fma_i32_serial", nk_each_fma_i32_serial);
    run_each<u32_k, sum_k, f64_k>("each_sum_u32_serial", nk_each_sum_u32_serial);
    run_each<u32_k, scale_k, f64_k>("each_scale_u32_serial", nk_each_scale_u32_serial);
    run_each<u32_k, blend_k, f64_k>("each_blend_u32_serial", nk_each_blend_u32_serial);
    run_each<u32_k, fma_k, f64_k>("each_fma_u32_serial", nk_each_fma_u32_serial);
    // Serial fallbacks — i64, u64
    run_each<i64_k, sum_k, f64_k>("each_sum_i64_serial", nk_each_sum_i64_serial);
    run_each<i64_k, scale_k, f64_k>("each_scale_i64_serial", nk_each_scale_i64_serial);
    run_each<i64_k, blend_k, f64_k>("each_blend_i64_serial", nk_each_blend_i64_serial);
    run_each<i64_k, fma_k, f64_k>("each_fma_i64_serial", nk_each_fma_i64_serial);
    run_each<u64_k, sum_k, f64_k>("each_sum_u64_serial", nk_each_sum_u64_serial);
    run_each<u64_k, scale_k, f64_k>("each_scale_u64_serial", nk_each_scale_u64_serial);
    run_each<u64_k, blend_k, f64_k>("each_blend_u64_serial", nk_each_blend_u64_serial);
    run_each<u64_k, fma_k, f64_k>("each_fma_u64_serial", nk_each_fma_u64_serial);
    // Serial fallbacks — complex
    run_each<f32c_k, sum_k, f32c_k>("each_sum_f32c_serial", nk_each_sum_f32c_serial);
    run_each<f32c_k, scale_k, f32c_k>("each_scale_f32c_serial", nk_each_scale_f32c_serial);
    run_each<f32c_k, blend_k, f32c_k>("each_blend_f32c_serial", nk_each_blend_f32c_serial);
    run_each<f32c_k, fma_k, f32c_k>("each_fma_f32c_serial", nk_each_fma_f32c_serial);
    run_each<f64c_k, sum_k, f64c_k>("each_sum_f64c_serial", nk_each_sum_f64c_serial);
    run_each<f64c_k, scale_k, f64c_k>("each_scale_f64c_serial", nk_each_scale_f64c_serial);
    run_each<f64c_k, blend_k, f64c_k>("each_blend_f64c_serial", nk_each_blend_f64c_serial);
    run_each<f64c_k, fma_k, f64c_k>("each_fma_f64c_serial", nk_each_fma_f64c_serial);
}
