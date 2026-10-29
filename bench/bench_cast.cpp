/**
 *  @brief Type casting benchmarks.
 *  @file bench/bench_cast.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/cast.h"

#include "bench.hpp"

using cast_kernel_t = void (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t);

/**
 *  @brief Measures the performance of type casting operations using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The cast kernel function to benchmark.
 *  @param count The number of elements to cast.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void measure_cast(bm::State &state, cast_kernel_t kernel, std::size_t count) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;

    auto input = make_vector<input_t>(count);
    auto output = make_vector<output_t>(count);

    // Initialize input with random values
    auto generator = make_random_engine();
    nk::fill_uniform(generator, input.values_data(), count);

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        kernel(input.values_data(), input_dtype_, count, output.values_data(), output_dtype_);
        bm::ClobberMemory();
        iterations++;
    }

    std::size_t const bytes_per_call = count * (sizeof(input_t) + sizeof(output_t));
    state.counters["bytes"] = bm::Counter(1.0 * iterations * bytes_per_call, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void run_cast(std::string name, cast_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_cast<input_dtype_, output_dtype_>, kernel,
                          bench_config.dense_dimensions);
}

void bench_cast() {
    run_cast<nk_f32_k, nk_u8_k>("cast_truncate_f32_to_u8_serial", nk_cast_truncate_serial);
    run_cast<nk_f64_k, nk_i64_k>("cast_truncate_f64_to_i64_serial", nk_cast_truncate_serial);
    run_cast<nk_f32_k, nk_u8_k>("cast_truncate_f32_to_u8", nk_cast_truncate);
    run_cast<nk_f64_k, nk_i64_k>("cast_truncate_f64_to_i64", nk_cast_truncate);

#if NK_TARGET_HASWELL
    run_cast<nk_f32_k, nk_f16_k>("cast_f32_to_f16_haswell", nk_cast_haswell);
    run_cast<nk_f16_k, nk_f32_k>("cast_f16_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_haswell", nk_cast_haswell);
    run_cast<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_haswell", nk_cast_haswell);
    run_cast<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_haswell", nk_cast_haswell);
    run_cast<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_f32_k, nk_e2m3_k>("cast_f32_to_e2m3_haswell", nk_cast_haswell);
    run_cast<nk_e2m3_k, nk_f32_k>("cast_e2m3_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_f32_k, nk_e3m2_k>("cast_f32_to_e3m2_haswell", nk_cast_haswell);
    run_cast<nk_e3m2_k, nk_f32_k>("cast_e3m2_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_i8_k, nk_f32_k>("cast_i8_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_f32_k, nk_i8_k>("cast_f32_to_i8_haswell", nk_cast_haswell);
    run_cast<nk_i16_k, nk_f32_k>("cast_i16_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_f32_k, nk_i16_k>("cast_f32_to_i16_haswell", nk_cast_haswell);
    run_cast<nk_u16_k, nk_f32_k>("cast_u16_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_f32_k, nk_u16_k>("cast_f32_to_u16_haswell", nk_cast_haswell);
    run_cast<nk_u8_k, nk_f32_k>("cast_u8_to_f32_haswell", nk_cast_haswell);
    run_cast<nk_f32_k, nk_u8_k>("cast_f32_to_u8_haswell", nk_cast_haswell);
#endif

#if NK_TARGET_SKYLAKE
    // float ↔ half/brain/MX
    run_cast<nk_f32_k, nk_f16_k>("cast_f32_to_f16_skylake", nk_cast_skylake);
    run_cast<nk_f16_k, nk_f32_k>("cast_f16_to_f32_skylake", nk_cast_skylake);
    run_cast<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_skylake", nk_cast_skylake);
    run_cast<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_skylake", nk_cast_skylake);
    run_cast<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_skylake", nk_cast_skylake);
    run_cast<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_skylake", nk_cast_skylake);
    run_cast<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_skylake", nk_cast_skylake);
    run_cast<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_skylake", nk_cast_skylake);
    run_cast<nk_f32_k, nk_e2m3_k>("cast_f32_to_e2m3_skylake", nk_cast_skylake);
    run_cast<nk_e2m3_k, nk_f32_k>("cast_e2m3_to_f32_skylake", nk_cast_skylake);
    run_cast<nk_f32_k, nk_e3m2_k>("cast_f32_to_e3m2_skylake", nk_cast_skylake);
    run_cast<nk_e3m2_k, nk_f32_k>("cast_e3m2_to_f32_skylake", nk_cast_skylake);
    // float ↔ double, integer ↔ float
    run_cast<nk_f64_k, nk_f32_k>("cast_f64_to_f32_skylake", nk_cast_skylake);
    run_cast<nk_f32_k, nk_f64_k>("cast_f32_to_f64_skylake", nk_cast_skylake);
    run_cast<nk_i32_k, nk_f64_k>("cast_i32_to_f64_skylake", nk_cast_skylake);
    run_cast<nk_f64_k, nk_i32_k>("cast_f64_to_i32_skylake", nk_cast_skylake);
    run_cast<nk_i8_k, nk_i32_k>("cast_i8_to_i32_skylake", nk_cast_skylake);
    run_cast<nk_i32_k, nk_i8_k>("cast_i32_to_i8_skylake", nk_cast_skylake);
    run_cast<nk_i16_k, nk_f32_k>("cast_i16_to_f32_skylake", nk_cast_skylake);
    run_cast<nk_f32_k, nk_i16_k>("cast_f32_to_i16_skylake", nk_cast_skylake);
    run_cast<nk_u16_k, nk_f32_k>("cast_u16_to_f32_skylake", nk_cast_skylake);
    run_cast<nk_f32_k, nk_u16_k>("cast_f32_to_u16_skylake", nk_cast_skylake);
    run_cast<nk_u8_k, nk_f32_k>("cast_u8_to_f32_skylake", nk_cast_skylake);
    run_cast<nk_f32_k, nk_u8_k>("cast_f32_to_u8_skylake", nk_cast_skylake);
    run_cast<nk_i64_k, nk_f64_k>("cast_i64_to_f64_skylake", nk_cast_skylake);
    run_cast<nk_f64_k, nk_i64_k>("cast_f64_to_i64_skylake", nk_cast_skylake);
    run_cast<nk_u64_k, nk_f64_k>("cast_u64_to_f64_skylake", nk_cast_skylake);
    run_cast<nk_f64_k, nk_u64_k>("cast_f64_to_u64_skylake", nk_cast_skylake);
    run_cast<nk_u32_k, nk_f64_k>("cast_u32_to_f64_skylake", nk_cast_skylake);
    run_cast<nk_f64_k, nk_u32_k>("cast_f64_to_u32_skylake", nk_cast_skylake);
#endif

#if NK_TARGET_ICELAKE
    // float ↔ half/brain/MX
    run_cast<nk_f32_k, nk_f16_k>("cast_f32_to_f16_icelake", nk_cast_icelake);
    run_cast<nk_f16_k, nk_f32_k>("cast_f16_to_f32_icelake", nk_cast_icelake);
    run_cast<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_icelake", nk_cast_icelake);
    run_cast<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_icelake", nk_cast_icelake);
    run_cast<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_icelake", nk_cast_icelake);
    run_cast<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_icelake", nk_cast_icelake);
    run_cast<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_icelake", nk_cast_icelake);
    run_cast<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_icelake", nk_cast_icelake);
    run_cast<nk_f32_k, nk_e2m3_k>("cast_f32_to_e2m3_icelake", nk_cast_icelake);
    run_cast<nk_e2m3_k, nk_f32_k>("cast_e2m3_to_f32_icelake", nk_cast_icelake);
    run_cast<nk_f32_k, nk_e3m2_k>("cast_f32_to_e3m2_icelake", nk_cast_icelake);
    run_cast<nk_e3m2_k, nk_f32_k>("cast_e3m2_to_f32_icelake", nk_cast_icelake);
    // integer ↔ float
    run_cast<nk_i8_k, nk_f32_k>("cast_i8_to_f32_icelake", nk_cast_icelake);
    run_cast<nk_f32_k, nk_i8_k>("cast_f32_to_i8_icelake", nk_cast_icelake);
#endif

#if NK_TARGET_SAPPHIRE
    // float ↔ half/brain/MX
    run_cast<nk_f32_k, nk_f16_k>("cast_f32_to_f16_sapphire", nk_cast_sapphire);
    run_cast<nk_f16_k, nk_f32_k>("cast_f16_to_f32_sapphire", nk_cast_sapphire);
    run_cast<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_sapphire", nk_cast_sapphire);
    run_cast<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_sapphire", nk_cast_sapphire);
    run_cast<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_sapphire", nk_cast_sapphire);
    run_cast<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_sapphire", nk_cast_sapphire);
    run_cast<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_sapphire", nk_cast_sapphire);
    run_cast<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_sapphire", nk_cast_sapphire);
    run_cast<nk_f32_k, nk_e2m3_k>("cast_f32_to_e2m3_sapphire", nk_cast_sapphire);
    run_cast<nk_e2m3_k, nk_f32_k>("cast_e2m3_to_f32_sapphire", nk_cast_sapphire);
    run_cast<nk_f32_k, nk_e3m2_k>("cast_f32_to_e3m2_sapphire", nk_cast_sapphire);
    run_cast<nk_e3m2_k, nk_f32_k>("cast_e3m2_to_f32_sapphire", nk_cast_sapphire);
    // integer ↔ float
    run_cast<nk_i8_k, nk_f32_k>("cast_i8_to_f32_sapphire", nk_cast_sapphire);
    run_cast<nk_f32_k, nk_i8_k>("cast_f32_to_i8_sapphire", nk_cast_sapphire);
#endif

#if NK_TARGET_NEON
    // NEON — float ↔ half/brain/MX
    run_cast<nk_f32_k, nk_f16_k>("cast_f32_to_f16_neon", nk_cast_neon);
    run_cast<nk_f16_k, nk_f32_k>("cast_f16_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_neon", nk_cast_neon);
    run_cast<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_neon", nk_cast_neon);
    run_cast<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_neon", nk_cast_neon);
    run_cast<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_e2m3_k>("cast_f32_to_e2m3_neon", nk_cast_neon);
    run_cast<nk_e2m3_k, nk_f32_k>("cast_e2m3_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_e3m2_k>("cast_f32_to_e3m2_neon", nk_cast_neon);
    run_cast<nk_e3m2_k, nk_f32_k>("cast_e3m2_to_f32_neon", nk_cast_neon);
    // NEON — float ↔ double
    run_cast<nk_f64_k, nk_f32_k>("cast_f64_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_f64_k>("cast_f32_to_f64_neon", nk_cast_neon);
    // NEON — integer ↔ float
    run_cast<nk_i8_k, nk_f32_k>("cast_i8_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_i8_k>("cast_f32_to_i8_neon", nk_cast_neon);
    run_cast<nk_u8_k, nk_f32_k>("cast_u8_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_u8_k>("cast_f32_to_u8_neon", nk_cast_neon);
    run_cast<nk_i16_k, nk_f32_k>("cast_i16_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_i16_k>("cast_f32_to_i16_neon", nk_cast_neon);
    run_cast<nk_u16_k, nk_f32_k>("cast_u16_to_f32_neon", nk_cast_neon);
    run_cast<nk_f32_k, nk_u16_k>("cast_f32_to_u16_neon", nk_cast_neon);
    // NEON — integer ↔ double
    run_cast<nk_i32_k, nk_f64_k>("cast_i32_to_f64_neon", nk_cast_neon);
    run_cast<nk_f64_k, nk_i32_k>("cast_f64_to_i32_neon", nk_cast_neon);
    run_cast<nk_u32_k, nk_f64_k>("cast_u32_to_f64_neon", nk_cast_neon);
    run_cast<nk_f64_k, nk_u32_k>("cast_f64_to_u32_neon", nk_cast_neon);
    run_cast<nk_i64_k, nk_f64_k>("cast_i64_to_f64_neon", nk_cast_neon);
    run_cast<nk_f64_k, nk_i64_k>("cast_f64_to_i64_neon", nk_cast_neon);
    run_cast<nk_u64_k, nk_f64_k>("cast_u64_to_f64_neon", nk_cast_neon);
    run_cast<nk_f64_k, nk_u64_k>("cast_f64_to_u64_neon", nk_cast_neon);
#endif

#if NK_TARGET_POWERVSX
    run_cast<nk_f32_k, nk_f16_k>("cast_f32_to_f16_powervsx", nk_cast_powervsx);
    run_cast<nk_f16_k, nk_f32_k>("cast_f16_to_f32_powervsx", nk_cast_powervsx);
    run_cast<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_powervsx", nk_cast_powervsx);
    run_cast<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_powervsx", nk_cast_powervsx);
    run_cast<nk_i8_k, nk_f32_k>("cast_i8_to_f32_powervsx", nk_cast_powervsx);
    run_cast<nk_f32_k, nk_i8_k>("cast_f32_to_i8_powervsx", nk_cast_powervsx);
    run_cast<nk_u8_k, nk_f32_k>("cast_u8_to_f32_powervsx", nk_cast_powervsx);
    run_cast<nk_f32_k, nk_u8_k>("cast_f32_to_u8_powervsx", nk_cast_powervsx);
    run_cast<nk_i16_k, nk_f32_k>("cast_i16_to_f32_powervsx", nk_cast_powervsx);
    run_cast<nk_f32_k, nk_i16_k>("cast_f32_to_i16_powervsx", nk_cast_powervsx);
    run_cast<nk_u16_k, nk_f32_k>("cast_u16_to_f32_powervsx", nk_cast_powervsx);
    run_cast<nk_f32_k, nk_u16_k>("cast_f32_to_u16_powervsx", nk_cast_powervsx);
#endif

#if NK_TARGET_V128RELAXED
    run_cast<nk_f32_k, nk_f16_k>("cast_f32_to_f16_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_f16_k, nk_f32_k>("cast_f16_to_f32_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_f32_k, nk_e2m3_k>("cast_f32_to_e2m3_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_e2m3_k, nk_f32_k>("cast_e2m3_to_f32_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_f32_k, nk_e3m2_k>("cast_f32_to_e3m2_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_e3m2_k, nk_f32_k>("cast_e3m2_to_f32_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_i8_k, nk_f32_k>("cast_i8_to_f32_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_f32_k, nk_i8_k>("cast_f32_to_i8_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_u8_k, nk_f32_k>("cast_u8_to_f32_v128relaxed", nk_cast_v128relaxed);
    run_cast<nk_f32_k, nk_u8_k>("cast_f32_to_u8_v128relaxed", nk_cast_v128relaxed);
#endif

    // Serial — float ↔ half/brain/MX
    run_cast<nk_f32_k, nk_f16_k>("cast_f32_to_f16_serial", nk_cast_serial);
    run_cast<nk_f16_k, nk_f32_k>("cast_f16_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_serial", nk_cast_serial);
    run_cast<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_serial", nk_cast_serial);
    run_cast<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_serial", nk_cast_serial);
    run_cast<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_e2m3_k>("cast_f32_to_e2m3_serial", nk_cast_serial);
    run_cast<nk_e2m3_k, nk_f32_k>("cast_e2m3_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_e3m2_k>("cast_f32_to_e3m2_serial", nk_cast_serial);
    run_cast<nk_e3m2_k, nk_f32_k>("cast_e3m2_to_f32_serial", nk_cast_serial);
    // Serial — float ↔ double
    run_cast<nk_f64_k, nk_f32_k>("cast_f64_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_f64_k>("cast_f32_to_f64_serial", nk_cast_serial);
    // Serial — integer ↔ float
    run_cast<nk_i8_k, nk_f32_k>("cast_i8_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_i8_k>("cast_f32_to_i8_serial", nk_cast_serial);
    run_cast<nk_u8_k, nk_f32_k>("cast_u8_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_u8_k>("cast_f32_to_u8_serial", nk_cast_serial);
    run_cast<nk_i16_k, nk_f32_k>("cast_i16_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_i16_k>("cast_f32_to_i16_serial", nk_cast_serial);
    run_cast<nk_u16_k, nk_f32_k>("cast_u16_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_u16_k>("cast_f32_to_u16_serial", nk_cast_serial);
    run_cast<nk_i32_k, nk_f32_k>("cast_i32_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_i32_k>("cast_f32_to_i32_serial", nk_cast_serial);
    // Serial — integer ↔ double
    run_cast<nk_i8_k, nk_f64_k>("cast_i8_to_f64_serial", nk_cast_serial);
    run_cast<nk_u8_k, nk_f64_k>("cast_u8_to_f64_serial", nk_cast_serial);
    run_cast<nk_i32_k, nk_f64_k>("cast_i32_to_f64_serial", nk_cast_serial);
    run_cast<nk_f64_k, nk_i32_k>("cast_f64_to_i32_serial", nk_cast_serial);
    run_cast<nk_u32_k, nk_f64_k>("cast_u32_to_f64_serial", nk_cast_serial);
    run_cast<nk_f64_k, nk_u32_k>("cast_f64_to_u32_serial", nk_cast_serial);
    run_cast<nk_i64_k, nk_f64_k>("cast_i64_to_f64_serial", nk_cast_serial);
    run_cast<nk_f64_k, nk_i64_k>("cast_f64_to_i64_serial", nk_cast_serial);
    run_cast<nk_u64_k, nk_f64_k>("cast_u64_to_f64_serial", nk_cast_serial);
    run_cast<nk_f64_k, nk_u64_k>("cast_f64_to_u64_serial", nk_cast_serial);
    // Serial — integer ↔ integer
    run_cast<nk_i8_k, nk_i32_k>("cast_i8_to_i32_serial", nk_cast_serial);
    run_cast<nk_i32_k, nk_i8_k>("cast_i32_to_i8_serial", nk_cast_serial);
    run_cast<nk_i16_k, nk_i64_k>("cast_i16_to_i64_serial", nk_cast_serial);
}
