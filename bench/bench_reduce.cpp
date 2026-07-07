/**
 *  @brief Reduction benchmarks (reduce_moments, reduce_minmax).
 *  @file bench/bench_reduce.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "numkong/reduce.h"
#include "numkong/reduce/serial.h"

#include "bench.hpp"

/**
 *  @brief Measures the performance of a reduce_moments kernel (sum + sumsq).
 */
template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_reduce_moments(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::reduce_moments_sum_t;
    using sumsq_t = typename input_t::reduce_moments_sumsq_t;

    std::size_t const vectors_count = bench_input_count(bench_dtype_bytes(input_dtype_, dimensions));
    std::vector<nk::vector<input_t>> vectors(vectors_count);
    auto generator = make_random_engine();
    for (auto &v : vectors) {
        v = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, v.values_data(), v.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        typename output_t::raw_t sum;
        typename sumsq_t::raw_t sumsq;
        auto const &v = vectors[iterations & (vectors_count - 1)];
        kernel(v.raw_values_data(), dimensions, sizeof(typename input_t::raw_t), &sum, &sumsq);
        bm::DoNotOptimize(sum);
        bm::DoNotOptimize(sumsq);
        ++iterations;
    }

    state.counters["bytes"] = bm::Counter(1.0 * iterations * vectors[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_reduce_moments(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_reduce_moments<input_dtype_, kernel_type_ *>, kernel,
                          bench_config.dense_dimensions);
}

/**
 *  @brief Measures the performance of a grouped RMSNorm kernel (single row, single group).
 */
template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_rmsnorm(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    // Each set holds two buffers (input + output), so size the ring against 2x the per-vector bytes.
    std::size_t const vectors_count = bench_input_count(bench_dtype_bytes(input_dtype_, 2 * dimensions));
    std::vector<nk::vector<input_t>> inputs(vectors_count), outputs(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t i = 0; i < vectors_count; ++i) {
        inputs[i] = make_vector<input_t>(dimensions);
        outputs[i] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, inputs[i].values_data(), inputs[i].size_values());
    }
    std::vector<nk_f32_t> gamma(dimensions, 1.0f);
    std::size_t const stride = dimensions * sizeof(typename input_t::raw_t);

    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(inputs[index].raw_values_data(), gamma.data(), outputs[index].raw_values_data(), 1, 1, dimensions,
               stride, stride, 1e-6f, 1.0f);
        bm::ClobberMemory();
        ++iterations;
    }
    state.counters["bytes"] = bm::Counter(1.0 * iterations * inputs[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_rmsnorm(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_rmsnorm<input_dtype_, kernel_type_ *>, kernel,
                          bench_config.dense_dimensions);
}

/**
 *  @brief Measures the performance of a reduce_minmax kernel (min + max with indices).
 */
template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_reduce_minmax(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::reduce_minmax_value_t;

    std::size_t const vectors_count = bench_input_count(bench_dtype_bytes(input_dtype_, dimensions));
    std::vector<nk::vector<input_t>> vectors(vectors_count);
    auto generator = make_random_engine();
    for (auto &v : vectors) {
        v = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, v.values_data(), v.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        typename output_t::raw_t min_val, max_val;
        nk_size_t min_idx, max_idx;
        auto const &v = vectors[iterations & (vectors_count - 1)];
        kernel(v.raw_values_data(), dimensions, sizeof(typename input_t::raw_t), &min_val, &min_idx, &max_val,
               &max_idx);
        bm::DoNotOptimize(min_val);
        bm::DoNotOptimize(max_val);
        ++iterations;
    }

    state.counters["bytes"] = bm::Counter(1.0 * iterations * vectors[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_reduce_minmax(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_reduce_minmax<input_dtype_, kernel_type_ *>, kernel,
                          bench_config.dense_dimensions);
}

void bench_reduce() {
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t i16_k = nk_i16_k;
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t i64_k = nk_i64_k;
    constexpr nk_dtype_t u64_k = nk_u64_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
    constexpr nk_dtype_t u1_k = nk_u1_k;

#if NK_TARGET_NEON
    run_reduce_moments<f64_k>("reduce_moments_f64_neon", nk_reduce_moments_f64_neon);
    run_reduce_moments<f32_k>("reduce_moments_f32_neon", nk_reduce_moments_f32_neon);
    run_reduce_moments<f16_k>("reduce_moments_f16_neon", nk_reduce_moments_f16_neon);
    run_reduce_moments<i8_k>("reduce_moments_i8_neon", nk_reduce_moments_i8_neon);
    run_reduce_moments<u8_k>("reduce_moments_u8_neon", nk_reduce_moments_u8_neon);
    run_reduce_moments<i16_k>("reduce_moments_i16_neon", nk_reduce_moments_i16_neon);
    run_reduce_moments<u16_k>("reduce_moments_u16_neon", nk_reduce_moments_u16_neon);
    run_reduce_moments<i32_k>("reduce_moments_i32_neon", nk_reduce_moments_i32_neon);
    run_reduce_moments<u32_k>("reduce_moments_u32_neon", nk_reduce_moments_u32_neon);
    run_reduce_moments<i64_k>("reduce_moments_i64_neon", nk_reduce_moments_i64_neon);
    run_reduce_moments<u64_k>("reduce_moments_u64_neon", nk_reduce_moments_u64_neon);
    run_reduce_moments<e4m3_k>("reduce_moments_e4m3_neon", nk_reduce_moments_e4m3_neon);
    run_reduce_moments<e5m2_k>("reduce_moments_e5m2_neon", nk_reduce_moments_e5m2_neon);
    run_reduce_moments<e2m3_k>("reduce_moments_e2m3_neon", nk_reduce_moments_e2m3_neon);
    run_reduce_moments<e3m2_k>("reduce_moments_e3m2_neon", nk_reduce_moments_e3m2_neon);
    run_reduce_moments<u1_k>("reduce_moments_u1_neon", nk_reduce_moments_u1_neon);
    run_reduce_minmax<f64_k>("reduce_minmax_f64_neon", nk_reduce_minmax_f64_neon);
    run_reduce_minmax<f32_k>("reduce_minmax_f32_neon", nk_reduce_minmax_f32_neon);
    run_reduce_minmax<i8_k>("reduce_minmax_i8_neon", nk_reduce_minmax_i8_neon);
    run_reduce_minmax<u8_k>("reduce_minmax_u8_neon", nk_reduce_minmax_u8_neon);
    run_reduce_minmax<i16_k>("reduce_minmax_i16_neon", nk_reduce_minmax_i16_neon);
    run_reduce_minmax<u16_k>("reduce_minmax_u16_neon", nk_reduce_minmax_u16_neon);
    run_reduce_minmax<i32_k>("reduce_minmax_i32_neon", nk_reduce_minmax_i32_neon);
    run_reduce_minmax<u32_k>("reduce_minmax_u32_neon", nk_reduce_minmax_u32_neon);
    run_reduce_minmax<i64_k>("reduce_minmax_i64_neon", nk_reduce_minmax_i64_neon);
    run_reduce_minmax<u64_k>("reduce_minmax_u64_neon", nk_reduce_minmax_u64_neon);
    run_reduce_minmax<e4m3_k>("reduce_minmax_e4m3_neon", nk_reduce_minmax_e4m3_neon);
    run_reduce_minmax<e5m2_k>("reduce_minmax_e5m2_neon", nk_reduce_minmax_e5m2_neon);
    run_reduce_minmax<e2m3_k>("reduce_minmax_e2m3_neon", nk_reduce_minmax_e2m3_neon);
    run_reduce_minmax<e3m2_k>("reduce_minmax_e3m2_neon", nk_reduce_minmax_e3m2_neon);
#endif

#if NK_TARGET_NEONBFDOT
    run_reduce_moments<bf16_k>("reduce_moments_bf16_neonbfdot", nk_reduce_moments_bf16_neonbfdot);
#endif

#if NK_TARGET_NEONSDOT
    run_reduce_moments<i8_k>("reduce_moments_i8_neonsdot", nk_reduce_moments_i8_neonsdot);
    run_reduce_moments<u8_k>("reduce_moments_u8_neonsdot", nk_reduce_moments_u8_neonsdot);
    run_reduce_moments<e2m3_k>("reduce_moments_e2m3_neonsdot", nk_reduce_moments_e2m3_neonsdot);
#endif

#if NK_TARGET_NEONFHM
    run_reduce_moments<e4m3_k>("reduce_moments_e4m3_neonfhm", nk_reduce_moments_e4m3_neonfhm);
    run_reduce_moments<e5m2_k>("reduce_moments_e5m2_neonfhm", nk_reduce_moments_e5m2_neonfhm);
#endif

#if NK_TARGET_HASWELL
    run_reduce_moments<f32_k>("reduce_moments_f32_haswell", nk_reduce_moments_f32_haswell);
    run_reduce_moments<f64_k>("reduce_moments_f64_haswell", nk_reduce_moments_f64_haswell);
    run_reduce_moments<i8_k>("reduce_moments_i8_haswell", nk_reduce_moments_i8_haswell);
    run_reduce_moments<u8_k>("reduce_moments_u8_haswell", nk_reduce_moments_u8_haswell);
    run_reduce_moments<i16_k>("reduce_moments_i16_haswell", nk_reduce_moments_i16_haswell);
    run_reduce_moments<u16_k>("reduce_moments_u16_haswell", nk_reduce_moments_u16_haswell);
    run_reduce_moments<i32_k>("reduce_moments_i32_haswell", nk_reduce_moments_i32_haswell);
    run_reduce_moments<u32_k>("reduce_moments_u32_haswell", nk_reduce_moments_u32_haswell);
    run_reduce_moments<i64_k>("reduce_moments_i64_haswell", nk_reduce_moments_i64_haswell);
    run_reduce_moments<u64_k>("reduce_moments_u64_haswell", nk_reduce_moments_u64_haswell);
    run_reduce_moments<e4m3_k>("reduce_moments_e4m3_haswell", nk_reduce_moments_e4m3_haswell);
    run_reduce_moments<e5m2_k>("reduce_moments_e5m2_haswell", nk_reduce_moments_e5m2_haswell);
    run_reduce_moments<e2m3_k>("reduce_moments_e2m3_haswell", nk_reduce_moments_e2m3_haswell);
    run_reduce_moments<e3m2_k>("reduce_moments_e3m2_haswell", nk_reduce_moments_e3m2_haswell);
    run_reduce_moments<i4_k>("reduce_moments_i4_haswell", nk_reduce_moments_i4_haswell);
    run_reduce_moments<u4_k>("reduce_moments_u4_haswell", nk_reduce_moments_u4_haswell);
    run_reduce_moments<u1_k>("reduce_moments_u1_haswell", nk_reduce_moments_u1_haswell);
    run_reduce_moments<bf16_k>("reduce_moments_bf16_haswell", nk_reduce_moments_bf16_haswell);
    run_rmsnorm<f32_k>("rmsnorm_f32_haswell", nk_reduce_rmsnorm_f32_haswell);
    run_rmsnorm<bf16_k>("rmsnorm_bf16_haswell", nk_reduce_rmsnorm_bf16_haswell);
    run_rmsnorm<e4m3_k>("rmsnorm_e4m3_haswell", nk_reduce_rmsnorm_e4m3_haswell);
    run_reduce_moments<f16_k>("reduce_moments_f16_haswell", nk_reduce_moments_f16_haswell);
    run_reduce_minmax<f32_k>("reduce_minmax_f32_haswell", nk_reduce_minmax_f32_haswell);
    run_reduce_minmax<f64_k>("reduce_minmax_f64_haswell", nk_reduce_minmax_f64_haswell);
    run_reduce_minmax<i8_k>("reduce_minmax_i8_haswell", nk_reduce_minmax_i8_haswell);
    run_reduce_minmax<u8_k>("reduce_minmax_u8_haswell", nk_reduce_minmax_u8_haswell);
    run_reduce_minmax<i16_k>("reduce_minmax_i16_haswell", nk_reduce_minmax_i16_haswell);
    run_reduce_minmax<u16_k>("reduce_minmax_u16_haswell", nk_reduce_minmax_u16_haswell);
    run_reduce_minmax<i32_k>("reduce_minmax_i32_haswell", nk_reduce_minmax_i32_haswell);
    run_reduce_minmax<u32_k>("reduce_minmax_u32_haswell", nk_reduce_minmax_u32_haswell);
    run_reduce_minmax<i64_k>("reduce_minmax_i64_haswell", nk_reduce_minmax_i64_haswell);
    run_reduce_minmax<u64_k>("reduce_minmax_u64_haswell", nk_reduce_minmax_u64_haswell);
    run_reduce_minmax<e4m3_k>("reduce_minmax_e4m3_haswell", nk_reduce_minmax_e4m3_haswell);
    run_reduce_minmax<e5m2_k>("reduce_minmax_e5m2_haswell", nk_reduce_minmax_e5m2_haswell);
    run_reduce_minmax<e2m3_k>("reduce_minmax_e2m3_haswell", nk_reduce_minmax_e2m3_haswell);
    run_reduce_minmax<e3m2_k>("reduce_minmax_e3m2_haswell", nk_reduce_minmax_e3m2_haswell);
    run_reduce_minmax<bf16_k>("reduce_minmax_bf16_haswell", nk_reduce_minmax_bf16_haswell);
    run_reduce_minmax<f16_k>("reduce_minmax_f16_haswell", nk_reduce_minmax_f16_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_reduce_moments<f32_k>("reduce_moments_f32_skylake", nk_reduce_moments_f32_skylake);
    run_reduce_moments<f64_k>("reduce_moments_f64_skylake", nk_reduce_moments_f64_skylake);
    run_reduce_moments<i8_k>("reduce_moments_i8_skylake", nk_reduce_moments_i8_skylake);
    run_reduce_moments<u8_k>("reduce_moments_u8_skylake", nk_reduce_moments_u8_skylake);
    run_reduce_moments<i16_k>("reduce_moments_i16_skylake", nk_reduce_moments_i16_skylake);
    run_reduce_moments<u16_k>("reduce_moments_u16_skylake", nk_reduce_moments_u16_skylake);
    run_reduce_moments<i32_k>("reduce_moments_i32_skylake", nk_reduce_moments_i32_skylake);
    run_reduce_moments<u32_k>("reduce_moments_u32_skylake", nk_reduce_moments_u32_skylake);
    run_reduce_moments<i64_k>("reduce_moments_i64_skylake", nk_reduce_moments_i64_skylake);
    run_reduce_moments<u64_k>("reduce_moments_u64_skylake", nk_reduce_moments_u64_skylake);
    run_reduce_moments<e4m3_k>("reduce_moments_e4m3_skylake", nk_reduce_moments_e4m3_skylake);
    run_reduce_moments<e5m2_k>("reduce_moments_e5m2_skylake", nk_reduce_moments_e5m2_skylake);
    run_reduce_moments<e2m3_k>("reduce_moments_e2m3_skylake", nk_reduce_moments_e2m3_skylake);
    run_reduce_moments<e3m2_k>("reduce_moments_e3m2_skylake", nk_reduce_moments_e3m2_skylake);
    run_reduce_moments<i4_k>("reduce_moments_i4_skylake", nk_reduce_moments_i4_skylake);
    run_reduce_moments<u4_k>("reduce_moments_u4_skylake", nk_reduce_moments_u4_skylake);
    run_reduce_moments<u1_k>("reduce_moments_u1_skylake", nk_reduce_moments_u1_skylake);
    run_reduce_minmax<f32_k>("reduce_minmax_f32_skylake", nk_reduce_minmax_f32_skylake);
    run_reduce_minmax<f64_k>("reduce_minmax_f64_skylake", nk_reduce_minmax_f64_skylake);
    run_reduce_minmax<i8_k>("reduce_minmax_i8_skylake", nk_reduce_minmax_i8_skylake);
    run_reduce_minmax<u8_k>("reduce_minmax_u8_skylake", nk_reduce_minmax_u8_skylake);
    run_reduce_minmax<i16_k>("reduce_minmax_i16_skylake", nk_reduce_minmax_i16_skylake);
    run_reduce_minmax<u16_k>("reduce_minmax_u16_skylake", nk_reduce_minmax_u16_skylake);
    run_reduce_minmax<i32_k>("reduce_minmax_i32_skylake", nk_reduce_minmax_i32_skylake);
    run_reduce_minmax<u32_k>("reduce_minmax_u32_skylake", nk_reduce_minmax_u32_skylake);
    run_reduce_minmax<i64_k>("reduce_minmax_i64_skylake", nk_reduce_minmax_i64_skylake);
    run_reduce_minmax<u64_k>("reduce_minmax_u64_skylake", nk_reduce_minmax_u64_skylake);
    run_reduce_minmax<e4m3_k>("reduce_minmax_e4m3_skylake", nk_reduce_minmax_e4m3_skylake);
    run_reduce_minmax<e5m2_k>("reduce_minmax_e5m2_skylake", nk_reduce_minmax_e5m2_skylake);
    run_reduce_minmax<e2m3_k>("reduce_minmax_e2m3_skylake", nk_reduce_minmax_e2m3_skylake);
    run_reduce_minmax<e3m2_k>("reduce_minmax_e3m2_skylake", nk_reduce_minmax_e3m2_skylake);
    run_reduce_moments<bf16_k>("reduce_moments_bf16_skylake", nk_reduce_moments_bf16_skylake);
    run_rmsnorm<f32_k>("rmsnorm_f32_skylake", nk_reduce_rmsnorm_f32_skylake);
    run_rmsnorm<bf16_k>("rmsnorm_bf16_skylake", nk_reduce_rmsnorm_bf16_skylake);
    run_rmsnorm<e4m3_k>("rmsnorm_e4m3_skylake", nk_reduce_rmsnorm_e4m3_skylake);
    run_reduce_minmax<bf16_k>("reduce_minmax_bf16_skylake", nk_reduce_minmax_bf16_skylake);
    run_reduce_moments<f16_k>("reduce_moments_f16_skylake", nk_reduce_moments_f16_skylake);
    run_reduce_minmax<f16_k>("reduce_minmax_f16_skylake", nk_reduce_minmax_f16_skylake);
#endif

#if NK_TARGET_ICELAKE
    run_reduce_moments<i8_k>("reduce_moments_i8_icelake", nk_reduce_moments_i8_icelake);
    run_reduce_moments<u8_k>("reduce_moments_u8_icelake", nk_reduce_moments_u8_icelake);
    run_reduce_moments<i16_k>("reduce_moments_i16_icelake", nk_reduce_moments_i16_icelake);
    run_reduce_moments<e2m3_k>("reduce_moments_e2m3_icelake", nk_reduce_moments_e2m3_icelake);
    run_reduce_moments<e3m2_k>("reduce_moments_e3m2_icelake", nk_reduce_moments_e3m2_icelake);
#endif

#if NK_TARGET_GENOA
    run_reduce_moments<bf16_k>("reduce_moments_bf16_genoa", nk_reduce_moments_bf16_genoa);
    run_rmsnorm<bf16_k>("rmsnorm_bf16_genoa", nk_reduce_rmsnorm_bf16_genoa);
    run_rmsnorm<e4m3_k>("rmsnorm_e4m3_genoa", nk_reduce_rmsnorm_e4m3_genoa);
    run_reduce_moments<e4m3_k>("reduce_moments_e4m3_genoa", nk_reduce_moments_e4m3_genoa);
    run_reduce_moments<e5m2_k>("reduce_moments_e5m2_genoa", nk_reduce_moments_e5m2_genoa);
#endif

#if NK_TARGET_ALDER
    run_reduce_moments<u8_k>("reduce_moments_u8_alder", nk_reduce_moments_u8_alder);
    run_reduce_moments<i16_k>("reduce_moments_i16_alder", nk_reduce_moments_i16_alder);
    run_reduce_moments<u16_k>("reduce_moments_u16_alder", nk_reduce_moments_u16_alder);
    run_reduce_moments<e3m2_k>("reduce_moments_e3m2_alder", nk_reduce_moments_e3m2_alder);
#endif

#if NK_TARGET_SIERRA
    run_reduce_moments<i8_k>("reduce_moments_i8_sierra", nk_reduce_moments_i8_sierra);
    run_reduce_moments<u8_k>("reduce_moments_u8_sierra", nk_reduce_moments_u8_sierra);
    run_reduce_moments<e2m3_k>("reduce_moments_e2m3_sierra", nk_reduce_moments_e2m3_sierra);
#endif

#if NK_TARGET_RVV
    run_reduce_moments<f32_k>("reduce_moments_f32_rvv", nk_reduce_moments_f32_rvv);
    run_reduce_moments<f64_k>("reduce_moments_f64_rvv", nk_reduce_moments_f64_rvv);
    run_reduce_moments<i8_k>("reduce_moments_i8_rvv", nk_reduce_moments_i8_rvv);
    run_reduce_moments<u8_k>("reduce_moments_u8_rvv", nk_reduce_moments_u8_rvv);
    run_reduce_moments<i16_k>("reduce_moments_i16_rvv", nk_reduce_moments_i16_rvv);
    run_reduce_moments<u16_k>("reduce_moments_u16_rvv", nk_reduce_moments_u16_rvv);
    run_reduce_moments<i32_k>("reduce_moments_i32_rvv", nk_reduce_moments_i32_rvv);
    run_reduce_moments<u32_k>("reduce_moments_u32_rvv", nk_reduce_moments_u32_rvv);
    run_reduce_moments<i64_k>("reduce_moments_i64_rvv", nk_reduce_moments_i64_rvv);
    run_reduce_moments<u64_k>("reduce_moments_u64_rvv", nk_reduce_moments_u64_rvv);
    run_reduce_moments<bf16_k>("reduce_moments_bf16_rvv", nk_reduce_moments_bf16_rvv);
    run_reduce_moments<f16_k>("reduce_moments_f16_rvv", nk_reduce_moments_f16_rvv);
    run_reduce_moments<e4m3_k>("reduce_moments_e4m3_rvv", nk_reduce_moments_e4m3_rvv);
    run_reduce_moments<e5m2_k>("reduce_moments_e5m2_rvv", nk_reduce_moments_e5m2_rvv);
    run_reduce_moments<e2m3_k>("reduce_moments_e2m3_rvv", nk_reduce_moments_e2m3_rvv);
    run_reduce_moments<e3m2_k>("reduce_moments_e3m2_rvv", nk_reduce_moments_e3m2_rvv);
    run_reduce_minmax<f32_k>("reduce_minmax_f32_rvv", nk_reduce_minmax_f32_rvv);
    run_reduce_minmax<f64_k>("reduce_minmax_f64_rvv", nk_reduce_minmax_f64_rvv);
    run_reduce_minmax<i8_k>("reduce_minmax_i8_rvv", nk_reduce_minmax_i8_rvv);
    run_reduce_minmax<u8_k>("reduce_minmax_u8_rvv", nk_reduce_minmax_u8_rvv);
    run_reduce_minmax<i16_k>("reduce_minmax_i16_rvv", nk_reduce_minmax_i16_rvv);
    run_reduce_minmax<u16_k>("reduce_minmax_u16_rvv", nk_reduce_minmax_u16_rvv);
    run_reduce_minmax<i32_k>("reduce_minmax_i32_rvv", nk_reduce_minmax_i32_rvv);
    run_reduce_minmax<u32_k>("reduce_minmax_u32_rvv", nk_reduce_minmax_u32_rvv);
    run_reduce_minmax<i64_k>("reduce_minmax_i64_rvv", nk_reduce_minmax_i64_rvv);
    run_reduce_minmax<u64_k>("reduce_minmax_u64_rvv", nk_reduce_minmax_u64_rvv);
    run_reduce_minmax<bf16_k>("reduce_minmax_bf16_rvv", nk_reduce_minmax_bf16_rvv);
    run_reduce_minmax<f16_k>("reduce_minmax_f16_rvv", nk_reduce_minmax_f16_rvv);
    run_reduce_minmax<e4m3_k>("reduce_minmax_e4m3_rvv", nk_reduce_minmax_e4m3_rvv);
    run_reduce_minmax<e5m2_k>("reduce_minmax_e5m2_rvv", nk_reduce_minmax_e5m2_rvv);
    run_reduce_minmax<e2m3_k>("reduce_minmax_e2m3_rvv", nk_reduce_minmax_e2m3_rvv);
    run_reduce_minmax<e3m2_k>("reduce_minmax_e3m2_rvv", nk_reduce_minmax_e3m2_rvv);
#endif

#if NK_TARGET_V128RELAXED
    run_reduce_moments<f32_k>("reduce_moments_f32_v128relaxed", nk_reduce_moments_f32_v128relaxed);
    run_reduce_moments<f64_k>("reduce_moments_f64_v128relaxed", nk_reduce_moments_f64_v128relaxed);
    run_reduce_moments<i8_k>("reduce_moments_i8_v128relaxed", nk_reduce_moments_i8_v128relaxed);
    run_reduce_moments<u8_k>("reduce_moments_u8_v128relaxed", nk_reduce_moments_u8_v128relaxed);
    run_reduce_moments<i16_k>("reduce_moments_i16_v128relaxed", nk_reduce_moments_i16_v128relaxed);
    run_reduce_moments<u16_k>("reduce_moments_u16_v128relaxed", nk_reduce_moments_u16_v128relaxed);
    run_reduce_moments<i32_k>("reduce_moments_i32_v128relaxed", nk_reduce_moments_i32_v128relaxed);
    run_reduce_moments<u32_k>("reduce_moments_u32_v128relaxed", nk_reduce_moments_u32_v128relaxed);
    run_reduce_moments<i64_k>("reduce_moments_i64_v128relaxed", nk_reduce_moments_i64_v128relaxed);
    run_reduce_moments<u64_k>("reduce_moments_u64_v128relaxed", nk_reduce_moments_u64_v128relaxed);
    run_reduce_moments<e4m3_k>("reduce_moments_e4m3_v128relaxed", nk_reduce_moments_e4m3_v128relaxed);
    run_reduce_moments<e5m2_k>("reduce_moments_e5m2_v128relaxed", nk_reduce_moments_e5m2_v128relaxed);
    run_reduce_moments<e2m3_k>("reduce_moments_e2m3_v128relaxed", nk_reduce_moments_e2m3_v128relaxed);
    run_reduce_moments<e3m2_k>("reduce_moments_e3m2_v128relaxed", nk_reduce_moments_e3m2_v128relaxed);
    run_reduce_moments<bf16_k>("reduce_moments_bf16_v128relaxed", nk_reduce_moments_bf16_v128relaxed);
    run_reduce_moments<f16_k>("reduce_moments_f16_v128relaxed", nk_reduce_moments_f16_v128relaxed);
    run_reduce_minmax<f32_k>("reduce_minmax_f32_v128relaxed", nk_reduce_minmax_f32_v128relaxed);
    run_reduce_minmax<f64_k>("reduce_minmax_f64_v128relaxed", nk_reduce_minmax_f64_v128relaxed);
    run_reduce_minmax<i8_k>("reduce_minmax_i8_v128relaxed", nk_reduce_minmax_i8_v128relaxed);
    run_reduce_minmax<u8_k>("reduce_minmax_u8_v128relaxed", nk_reduce_minmax_u8_v128relaxed);
    run_reduce_minmax<i16_k>("reduce_minmax_i16_v128relaxed", nk_reduce_minmax_i16_v128relaxed);
    run_reduce_minmax<u16_k>("reduce_minmax_u16_v128relaxed", nk_reduce_minmax_u16_v128relaxed);
    run_reduce_minmax<i32_k>("reduce_minmax_i32_v128relaxed", nk_reduce_minmax_i32_v128relaxed);
    run_reduce_minmax<u32_k>("reduce_minmax_u32_v128relaxed", nk_reduce_minmax_u32_v128relaxed);
    run_reduce_minmax<i64_k>("reduce_minmax_i64_v128relaxed", nk_reduce_minmax_i64_v128relaxed);
    run_reduce_minmax<u64_k>("reduce_minmax_u64_v128relaxed", nk_reduce_minmax_u64_v128relaxed);
    run_reduce_minmax<e4m3_k>("reduce_minmax_e4m3_v128relaxed", nk_reduce_minmax_e4m3_v128relaxed);
    run_reduce_minmax<e5m2_k>("reduce_minmax_e5m2_v128relaxed", nk_reduce_minmax_e5m2_v128relaxed);
    run_reduce_minmax<e2m3_k>("reduce_minmax_e2m3_v128relaxed", nk_reduce_minmax_e2m3_v128relaxed);
    run_reduce_minmax<e3m2_k>("reduce_minmax_e3m2_v128relaxed", nk_reduce_minmax_e3m2_v128relaxed);
    run_reduce_minmax<bf16_k>("reduce_minmax_bf16_v128relaxed", nk_reduce_minmax_bf16_v128relaxed);
    run_reduce_minmax<f16_k>("reduce_minmax_f16_v128relaxed", nk_reduce_minmax_f16_v128relaxed);
#endif

    run_reduce_moments<f32_k>("reduce_moments_f32_serial", nk_reduce_moments_f32_serial);
    run_reduce_moments<f64_k>("reduce_moments_f64_serial", nk_reduce_moments_f64_serial);
    run_reduce_moments<i8_k>("reduce_moments_i8_serial", nk_reduce_moments_i8_serial);
    run_reduce_moments<u8_k>("reduce_moments_u8_serial", nk_reduce_moments_u8_serial);
    run_reduce_moments<i16_k>("reduce_moments_i16_serial", nk_reduce_moments_i16_serial);
    run_reduce_moments<u16_k>("reduce_moments_u16_serial", nk_reduce_moments_u16_serial);
    run_reduce_moments<i32_k>("reduce_moments_i32_serial", nk_reduce_moments_i32_serial);
    run_reduce_moments<u32_k>("reduce_moments_u32_serial", nk_reduce_moments_u32_serial);
    run_reduce_moments<i64_k>("reduce_moments_i64_serial", nk_reduce_moments_i64_serial);
    run_reduce_moments<u64_k>("reduce_moments_u64_serial", nk_reduce_moments_u64_serial);
    run_reduce_moments<f16_k>("reduce_moments_f16_serial", nk_reduce_moments_f16_serial);
    run_reduce_moments<bf16_k>("reduce_moments_bf16_serial", nk_reduce_moments_bf16_serial);
    run_rmsnorm<f32_k>("rmsnorm_f32_serial", nk_reduce_rmsnorm_f32_serial);
    run_rmsnorm<bf16_k>("rmsnorm_bf16_serial", nk_reduce_rmsnorm_bf16_serial);
    run_rmsnorm<e4m3_k>("rmsnorm_e4m3_serial", nk_reduce_rmsnorm_e4m3_serial);
    run_reduce_moments<e4m3_k>("reduce_moments_e4m3_serial", nk_reduce_moments_e4m3_serial);
    run_reduce_moments<e5m2_k>("reduce_moments_e5m2_serial", nk_reduce_moments_e5m2_serial);
    run_reduce_moments<e2m3_k>("reduce_moments_e2m3_serial", nk_reduce_moments_e2m3_serial);
    run_reduce_moments<e3m2_k>("reduce_moments_e3m2_serial", nk_reduce_moments_e3m2_serial);
    run_reduce_moments<i4_k>("reduce_moments_i4_serial", nk_reduce_moments_i4_serial);
    run_reduce_moments<u4_k>("reduce_moments_u4_serial", nk_reduce_moments_u4_serial);
    run_reduce_moments<u1_k>("reduce_moments_u1_serial", nk_reduce_moments_u1_serial);
    run_reduce_minmax<f32_k>("reduce_minmax_f32_serial", nk_reduce_minmax_f32_serial);
    run_reduce_minmax<f64_k>("reduce_minmax_f64_serial", nk_reduce_minmax_f64_serial);
    run_reduce_minmax<i8_k>("reduce_minmax_i8_serial", nk_reduce_minmax_i8_serial);
    run_reduce_minmax<u8_k>("reduce_minmax_u8_serial", nk_reduce_minmax_u8_serial);
    run_reduce_minmax<i16_k>("reduce_minmax_i16_serial", nk_reduce_minmax_i16_serial);
    run_reduce_minmax<u16_k>("reduce_minmax_u16_serial", nk_reduce_minmax_u16_serial);
    run_reduce_minmax<i32_k>("reduce_minmax_i32_serial", nk_reduce_minmax_i32_serial);
    run_reduce_minmax<u32_k>("reduce_minmax_u32_serial", nk_reduce_minmax_u32_serial);
    run_reduce_minmax<i64_k>("reduce_minmax_i64_serial", nk_reduce_minmax_i64_serial);
    run_reduce_minmax<u64_k>("reduce_minmax_u64_serial", nk_reduce_minmax_u64_serial);
    run_reduce_minmax<f16_k>("reduce_minmax_f16_serial", nk_reduce_minmax_f16_serial);
    run_reduce_minmax<bf16_k>("reduce_minmax_bf16_serial", nk_reduce_minmax_bf16_serial);
    run_reduce_minmax<e4m3_k>("reduce_minmax_e4m3_serial", nk_reduce_minmax_e4m3_serial);
    run_reduce_minmax<e5m2_k>("reduce_minmax_e5m2_serial", nk_reduce_minmax_e5m2_serial);
    run_reduce_minmax<e2m3_k>("reduce_minmax_e2m3_serial", nk_reduce_minmax_e2m3_serial);
    run_reduce_minmax<e3m2_k>("reduce_minmax_e3m2_serial", nk_reduce_minmax_e3m2_serial);
    run_reduce_minmax<i4_k>("reduce_minmax_i4_serial", nk_reduce_minmax_i4_serial);
    run_reduce_minmax<u4_k>("reduce_minmax_u4_serial", nk_reduce_minmax_u4_serial);
    run_reduce_minmax<u1_k>("reduce_minmax_u1_serial", nk_reduce_minmax_u1_serial);
}
