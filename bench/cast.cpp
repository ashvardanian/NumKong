/**
 *  @file bench/cast.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Type casting benchmarks.
 */

#include "numkong/cast.h"

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

using cast_kernel_t = nk_status_t (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t, void *);

/**
 *  @brief Measures the performance of type casting operations.
 *  @param[inout] loop The timed loop, which takes the counters.
 *  @param[in] kernel The cast kernel function to benchmark.
 *  @param[in] count The number of elements to cast.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void measure_cast(loop_t &loop, environment_t const &env, cast_kernel_t kernel, std::size_t count) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;

    auto input = make_vector<input_t>(count);
    auto output = make_vector<output_t>(count);

    // Initialize input with random values
    std::mt19937 generator(env.settings.seed.value);
    nk::fill_uniform(generator, input.values_data(), count);

    // Benchmark loop
    for ([[maybe_unused]] std::size_t call : loop) {
        if (!succeeded(loop,
                       kernel(input.values_data(), input_dtype_, count, output.values_data(), output_dtype_, nullptr)))
            break;
        do_not_optimize(output.values_data());
    }

    std::size_t const bytes_per_call = count * (sizeof(input_t) + sizeof(output_t));
    loop.byte_rate(bytes_per_call);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void run_cast(environment_t const &env, std::string name, cast_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(env.settings.batch_per_core) + ">";
    run_benchmark(env, bench_name, measure_cast<input_dtype_, output_dtype_>, kernel, env.settings.batch_per_core);
}

void bench_cast(environment_t const &env) {

#if NUMKONG_TARGET_HASWELL
    if (section(env, "Type Casts Haswell", nk_cap_haswell_k)) {
        run_cast<nk_f32_k, nk_f16_k>(env, "cast_f32_to_f16_haswell", nk_cast_haswell);
        run_cast<nk_f16_k, nk_f32_k>(env, "cast_f16_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_haswell", nk_cast_haswell);
        run_cast<nk_bf16_k, nk_f32_k>(env, "cast_bf16_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_f32_k, nk_e4m3_k>(env, "cast_f32_to_e4m3_haswell", nk_cast_haswell);
        run_cast<nk_e4m3_k, nk_f32_k>(env, "cast_e4m3_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_f32_k, nk_e5m2_k>(env, "cast_f32_to_e5m2_haswell", nk_cast_haswell);
        run_cast<nk_e5m2_k, nk_f32_k>(env, "cast_e5m2_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_f32_k, nk_e2m3_k>(env, "cast_f32_to_e2m3_haswell", nk_cast_haswell);
        run_cast<nk_e2m3_k, nk_f32_k>(env, "cast_e2m3_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_f32_k, nk_e3m2_k>(env, "cast_f32_to_e3m2_haswell", nk_cast_haswell);
        run_cast<nk_e3m2_k, nk_f32_k>(env, "cast_e3m2_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_i8_k, nk_f32_k>(env, "cast_i8_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_f32_k, nk_i8_k>(env, "cast_f32_to_i8_haswell", nk_cast_haswell);
        run_cast<nk_i16_k, nk_f32_k>(env, "cast_i16_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_f32_k, nk_i16_k>(env, "cast_f32_to_i16_haswell", nk_cast_haswell);
        run_cast<nk_u16_k, nk_f32_k>(env, "cast_u16_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_f32_k, nk_u16_k>(env, "cast_f32_to_u16_haswell", nk_cast_haswell);
        run_cast<nk_u8_k, nk_f32_k>(env, "cast_u8_to_f32_haswell", nk_cast_haswell);
        run_cast<nk_f32_k, nk_u8_k>(env, "cast_f32_to_u8_haswell", nk_cast_haswell);
    }
#endif

#if NUMKONG_TARGET_SKYLAKE
    if (section(env, "Type Casts Skylake", nk_cap_skylake_k)) {
        // float ↔ half/brain/MX
        run_cast<nk_f32_k, nk_f16_k>(env, "cast_f32_to_f16_skylake", nk_cast_skylake);
        run_cast<nk_f16_k, nk_f32_k>(env, "cast_f16_to_f32_skylake", nk_cast_skylake);
        run_cast<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_skylake", nk_cast_skylake);
        run_cast<nk_bf16_k, nk_f32_k>(env, "cast_bf16_to_f32_skylake", nk_cast_skylake);
        run_cast<nk_f32_k, nk_e4m3_k>(env, "cast_f32_to_e4m3_skylake", nk_cast_skylake);
        run_cast<nk_e4m3_k, nk_f32_k>(env, "cast_e4m3_to_f32_skylake", nk_cast_skylake);
        run_cast<nk_f32_k, nk_e5m2_k>(env, "cast_f32_to_e5m2_skylake", nk_cast_skylake);
        run_cast<nk_e5m2_k, nk_f32_k>(env, "cast_e5m2_to_f32_skylake", nk_cast_skylake);
        run_cast<nk_f32_k, nk_e2m3_k>(env, "cast_f32_to_e2m3_skylake", nk_cast_skylake);
        run_cast<nk_e2m3_k, nk_f32_k>(env, "cast_e2m3_to_f32_skylake", nk_cast_skylake);
        run_cast<nk_f32_k, nk_e3m2_k>(env, "cast_f32_to_e3m2_skylake", nk_cast_skylake);
        run_cast<nk_e3m2_k, nk_f32_k>(env, "cast_e3m2_to_f32_skylake", nk_cast_skylake);
        // float ↔ double, integer ↔ float
        run_cast<nk_f64_k, nk_f32_k>(env, "cast_f64_to_f32_skylake", nk_cast_skylake);
        run_cast<nk_f32_k, nk_f64_k>(env, "cast_f32_to_f64_skylake", nk_cast_skylake);
        run_cast<nk_i32_k, nk_f64_k>(env, "cast_i32_to_f64_skylake", nk_cast_skylake);
        run_cast<nk_f64_k, nk_i32_k>(env, "cast_f64_to_i32_skylake", nk_cast_skylake);
        run_cast<nk_i8_k, nk_i32_k>(env, "cast_i8_to_i32_skylake", nk_cast_skylake);
        run_cast<nk_i32_k, nk_i8_k>(env, "cast_i32_to_i8_skylake", nk_cast_skylake);
        run_cast<nk_i16_k, nk_f32_k>(env, "cast_i16_to_f32_skylake", nk_cast_skylake);
        run_cast<nk_f32_k, nk_i16_k>(env, "cast_f32_to_i16_skylake", nk_cast_skylake);
        run_cast<nk_u16_k, nk_f32_k>(env, "cast_u16_to_f32_skylake", nk_cast_skylake);
        run_cast<nk_f32_k, nk_u16_k>(env, "cast_f32_to_u16_skylake", nk_cast_skylake);
        run_cast<nk_u8_k, nk_f32_k>(env, "cast_u8_to_f32_skylake", nk_cast_skylake);
        run_cast<nk_f32_k, nk_u8_k>(env, "cast_f32_to_u8_skylake", nk_cast_skylake);
        run_cast<nk_i64_k, nk_f64_k>(env, "cast_i64_to_f64_skylake", nk_cast_skylake);
        run_cast<nk_f64_k, nk_i64_k>(env, "cast_f64_to_i64_skylake", nk_cast_skylake);
        run_cast<nk_u64_k, nk_f64_k>(env, "cast_u64_to_f64_skylake", nk_cast_skylake);
        run_cast<nk_f64_k, nk_u64_k>(env, "cast_f64_to_u64_skylake", nk_cast_skylake);
        run_cast<nk_u32_k, nk_f64_k>(env, "cast_u32_to_f64_skylake", nk_cast_skylake);
        run_cast<nk_f64_k, nk_u32_k>(env, "cast_f64_to_u32_skylake", nk_cast_skylake);
    }
#endif

#if NUMKONG_TARGET_ICELAKE
    if (section(env, "Type Casts Ice Lake", nk_cap_icelake_k)) {
        // float ↔ half/brain/MX
        run_cast<nk_f32_k, nk_f16_k>(env, "cast_f32_to_f16_icelake", nk_cast_icelake);
        run_cast<nk_f16_k, nk_f32_k>(env, "cast_f16_to_f32_icelake", nk_cast_icelake);
        run_cast<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_icelake", nk_cast_icelake);
        run_cast<nk_bf16_k, nk_f32_k>(env, "cast_bf16_to_f32_icelake", nk_cast_icelake);
        run_cast<nk_f32_k, nk_e4m3_k>(env, "cast_f32_to_e4m3_icelake", nk_cast_icelake);
        run_cast<nk_e4m3_k, nk_f32_k>(env, "cast_e4m3_to_f32_icelake", nk_cast_icelake);
        run_cast<nk_f32_k, nk_e5m2_k>(env, "cast_f32_to_e5m2_icelake", nk_cast_icelake);
        run_cast<nk_e5m2_k, nk_f32_k>(env, "cast_e5m2_to_f32_icelake", nk_cast_icelake);
        run_cast<nk_f32_k, nk_e2m3_k>(env, "cast_f32_to_e2m3_icelake", nk_cast_icelake);
        run_cast<nk_e2m3_k, nk_f32_k>(env, "cast_e2m3_to_f32_icelake", nk_cast_icelake);
        run_cast<nk_f32_k, nk_e3m2_k>(env, "cast_f32_to_e3m2_icelake", nk_cast_icelake);
        run_cast<nk_e3m2_k, nk_f32_k>(env, "cast_e3m2_to_f32_icelake", nk_cast_icelake);
        // integer ↔ float
        run_cast<nk_i8_k, nk_f32_k>(env, "cast_i8_to_f32_icelake", nk_cast_icelake);
        run_cast<nk_f32_k, nk_i8_k>(env, "cast_f32_to_i8_icelake", nk_cast_icelake);
    }
#endif

#if NUMKONG_TARGET_SAPPHIRE
    if (section(env, "Type Casts Sapphire", nk_cap_sapphire_k)) {
        // float ↔ half/brain/MX
        run_cast<nk_f32_k, nk_f16_k>(env, "cast_f32_to_f16_sapphire", nk_cast_sapphire);
        run_cast<nk_f16_k, nk_f32_k>(env, "cast_f16_to_f32_sapphire", nk_cast_sapphire);
        run_cast<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_sapphire", nk_cast_sapphire);
        run_cast<nk_bf16_k, nk_f32_k>(env, "cast_bf16_to_f32_sapphire", nk_cast_sapphire);
        run_cast<nk_f32_k, nk_e4m3_k>(env, "cast_f32_to_e4m3_sapphire", nk_cast_sapphire);
        run_cast<nk_e4m3_k, nk_f32_k>(env, "cast_e4m3_to_f32_sapphire", nk_cast_sapphire);
        run_cast<nk_f32_k, nk_e5m2_k>(env, "cast_f32_to_e5m2_sapphire", nk_cast_sapphire);
        run_cast<nk_e5m2_k, nk_f32_k>(env, "cast_e5m2_to_f32_sapphire", nk_cast_sapphire);
        run_cast<nk_f32_k, nk_e2m3_k>(env, "cast_f32_to_e2m3_sapphire", nk_cast_sapphire);
        run_cast<nk_e2m3_k, nk_f32_k>(env, "cast_e2m3_to_f32_sapphire", nk_cast_sapphire);
        run_cast<nk_f32_k, nk_e3m2_k>(env, "cast_f32_to_e3m2_sapphire", nk_cast_sapphire);
        run_cast<nk_e3m2_k, nk_f32_k>(env, "cast_e3m2_to_f32_sapphire", nk_cast_sapphire);
        // integer ↔ float
        run_cast<nk_i8_k, nk_f32_k>(env, "cast_i8_to_f32_sapphire", nk_cast_sapphire);
        run_cast<nk_f32_k, nk_i8_k>(env, "cast_f32_to_i8_sapphire", nk_cast_sapphire);
    }
#endif

#if NUMKONG_TARGET_NEON
    if (section(env, "Type Casts NEON", nk_cap_neon_k)) {
        // NEON — float ↔ half/brain/MX
        run_cast<nk_f32_k, nk_f16_k>(env, "cast_f32_to_f16_neon", nk_cast_neon);
        run_cast<nk_f16_k, nk_f32_k>(env, "cast_f16_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_neon", nk_cast_neon);
        run_cast<nk_bf16_k, nk_f32_k>(env, "cast_bf16_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_e4m3_k>(env, "cast_f32_to_e4m3_neon", nk_cast_neon);
        run_cast<nk_e4m3_k, nk_f32_k>(env, "cast_e4m3_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_e5m2_k>(env, "cast_f32_to_e5m2_neon", nk_cast_neon);
        run_cast<nk_e5m2_k, nk_f32_k>(env, "cast_e5m2_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_e2m3_k>(env, "cast_f32_to_e2m3_neon", nk_cast_neon);
        run_cast<nk_e2m3_k, nk_f32_k>(env, "cast_e2m3_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_e3m2_k>(env, "cast_f32_to_e3m2_neon", nk_cast_neon);
        run_cast<nk_e3m2_k, nk_f32_k>(env, "cast_e3m2_to_f32_neon", nk_cast_neon);
        // NEON — float ↔ double
        run_cast<nk_f64_k, nk_f32_k>(env, "cast_f64_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_f64_k>(env, "cast_f32_to_f64_neon", nk_cast_neon);
        // NEON — integer ↔ float
        run_cast<nk_i8_k, nk_f32_k>(env, "cast_i8_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_i8_k>(env, "cast_f32_to_i8_neon", nk_cast_neon);
        run_cast<nk_u8_k, nk_f32_k>(env, "cast_u8_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_u8_k>(env, "cast_f32_to_u8_neon", nk_cast_neon);
        run_cast<nk_i16_k, nk_f32_k>(env, "cast_i16_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_i16_k>(env, "cast_f32_to_i16_neon", nk_cast_neon);
        run_cast<nk_u16_k, nk_f32_k>(env, "cast_u16_to_f32_neon", nk_cast_neon);
        run_cast<nk_f32_k, nk_u16_k>(env, "cast_f32_to_u16_neon", nk_cast_neon);
        // NEON — integer ↔ double
        run_cast<nk_i32_k, nk_f64_k>(env, "cast_i32_to_f64_neon", nk_cast_neon);
        run_cast<nk_f64_k, nk_i32_k>(env, "cast_f64_to_i32_neon", nk_cast_neon);
        run_cast<nk_u32_k, nk_f64_k>(env, "cast_u32_to_f64_neon", nk_cast_neon);
        run_cast<nk_f64_k, nk_u32_k>(env, "cast_f64_to_u32_neon", nk_cast_neon);
        run_cast<nk_i64_k, nk_f64_k>(env, "cast_i64_to_f64_neon", nk_cast_neon);
        run_cast<nk_f64_k, nk_i64_k>(env, "cast_f64_to_i64_neon", nk_cast_neon);
        run_cast<nk_u64_k, nk_f64_k>(env, "cast_u64_to_f64_neon", nk_cast_neon);
        run_cast<nk_f64_k, nk_u64_k>(env, "cast_f64_to_u64_neon", nk_cast_neon);
    }
#endif

#if NUMKONG_TARGET_POWERVSX
    if (section(env, "Type Casts Power VSX", nk_cap_powervsx_k)) {
        run_cast<nk_f32_k, nk_f16_k>(env, "cast_f32_to_f16_powervsx", nk_cast_powervsx);
        run_cast<nk_f16_k, nk_f32_k>(env, "cast_f16_to_f32_powervsx", nk_cast_powervsx);
        run_cast<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_powervsx", nk_cast_powervsx);
        run_cast<nk_bf16_k, nk_f32_k>(env, "cast_bf16_to_f32_powervsx", nk_cast_powervsx);
        run_cast<nk_i8_k, nk_f32_k>(env, "cast_i8_to_f32_powervsx", nk_cast_powervsx);
        run_cast<nk_f32_k, nk_i8_k>(env, "cast_f32_to_i8_powervsx", nk_cast_powervsx);
        run_cast<nk_u8_k, nk_f32_k>(env, "cast_u8_to_f32_powervsx", nk_cast_powervsx);
        run_cast<nk_f32_k, nk_u8_k>(env, "cast_f32_to_u8_powervsx", nk_cast_powervsx);
        run_cast<nk_i16_k, nk_f32_k>(env, "cast_i16_to_f32_powervsx", nk_cast_powervsx);
        run_cast<nk_f32_k, nk_i16_k>(env, "cast_f32_to_i16_powervsx", nk_cast_powervsx);
        run_cast<nk_u16_k, nk_f32_k>(env, "cast_u16_to_f32_powervsx", nk_cast_powervsx);
        run_cast<nk_f32_k, nk_u16_k>(env, "cast_f32_to_u16_powervsx", nk_cast_powervsx);
    }
#endif

#if NUMKONG_TARGET_V128RELAXED
    if (section(env, "Type Casts V128 Relaxed", nk_cap_v128relaxed_k)) {
        run_cast<nk_f32_k, nk_f16_k>(env, "cast_f32_to_f16_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_f16_k, nk_f32_k>(env, "cast_f16_to_f32_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_bf16_k, nk_f32_k>(env, "cast_bf16_to_f32_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_f32_k, nk_e4m3_k>(env, "cast_f32_to_e4m3_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_e4m3_k, nk_f32_k>(env, "cast_e4m3_to_f32_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_f32_k, nk_e5m2_k>(env, "cast_f32_to_e5m2_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_e5m2_k, nk_f32_k>(env, "cast_e5m2_to_f32_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_f32_k, nk_e2m3_k>(env, "cast_f32_to_e2m3_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_e2m3_k, nk_f32_k>(env, "cast_e2m3_to_f32_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_f32_k, nk_e3m2_k>(env, "cast_f32_to_e3m2_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_e3m2_k, nk_f32_k>(env, "cast_e3m2_to_f32_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_i8_k, nk_f32_k>(env, "cast_i8_to_f32_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_f32_k, nk_i8_k>(env, "cast_f32_to_i8_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_u8_k, nk_f32_k>(env, "cast_u8_to_f32_v128relaxed", nk_cast_v128relaxed);
        run_cast<nk_f32_k, nk_u8_k>(env, "cast_f32_to_u8_v128relaxed", nk_cast_v128relaxed);
    }
#endif

    // Serial — float ↔ half/brain/MX
    section(env, "Type Casts Serial", nk_cap_serial_k);
    run_cast<nk_f32_k, nk_f16_k>(env, "cast_f32_to_f16_serial", nk_cast_serial);
    run_cast<nk_f16_k, nk_f32_k>(env, "cast_f16_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_serial", nk_cast_serial);
    run_cast<nk_bf16_k, nk_f32_k>(env, "cast_bf16_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_e4m3_k>(env, "cast_f32_to_e4m3_serial", nk_cast_serial);
    run_cast<nk_e4m3_k, nk_f32_k>(env, "cast_e4m3_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_e5m2_k>(env, "cast_f32_to_e5m2_serial", nk_cast_serial);
    run_cast<nk_e5m2_k, nk_f32_k>(env, "cast_e5m2_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_e2m3_k>(env, "cast_f32_to_e2m3_serial", nk_cast_serial);
    run_cast<nk_e2m3_k, nk_f32_k>(env, "cast_e2m3_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_e3m2_k>(env, "cast_f32_to_e3m2_serial", nk_cast_serial);
    run_cast<nk_e3m2_k, nk_f32_k>(env, "cast_e3m2_to_f32_serial", nk_cast_serial);
    // Serial — float ↔ double
    run_cast<nk_f64_k, nk_f32_k>(env, "cast_f64_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_f64_k>(env, "cast_f32_to_f64_serial", nk_cast_serial);
    // Serial — integer ↔ float
    run_cast<nk_i8_k, nk_f32_k>(env, "cast_i8_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_i8_k>(env, "cast_f32_to_i8_serial", nk_cast_serial);
    run_cast<nk_u8_k, nk_f32_k>(env, "cast_u8_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_u8_k>(env, "cast_f32_to_u8_serial", nk_cast_serial);
    run_cast<nk_i16_k, nk_f32_k>(env, "cast_i16_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_i16_k>(env, "cast_f32_to_i16_serial", nk_cast_serial);
    run_cast<nk_u16_k, nk_f32_k>(env, "cast_u16_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_u16_k>(env, "cast_f32_to_u16_serial", nk_cast_serial);
    run_cast<nk_i32_k, nk_f32_k>(env, "cast_i32_to_f32_serial", nk_cast_serial);
    run_cast<nk_f32_k, nk_i32_k>(env, "cast_f32_to_i32_serial", nk_cast_serial);
    // Serial — integer ↔ double
    run_cast<nk_i8_k, nk_f64_k>(env, "cast_i8_to_f64_serial", nk_cast_serial);
    run_cast<nk_u8_k, nk_f64_k>(env, "cast_u8_to_f64_serial", nk_cast_serial);
    run_cast<nk_i32_k, nk_f64_k>(env, "cast_i32_to_f64_serial", nk_cast_serial);
    run_cast<nk_f64_k, nk_i32_k>(env, "cast_f64_to_i32_serial", nk_cast_serial);
    run_cast<nk_u32_k, nk_f64_k>(env, "cast_u32_to_f64_serial", nk_cast_serial);
    run_cast<nk_f64_k, nk_u32_k>(env, "cast_f64_to_u32_serial", nk_cast_serial);
    run_cast<nk_i64_k, nk_f64_k>(env, "cast_i64_to_f64_serial", nk_cast_serial);
    run_cast<nk_f64_k, nk_i64_k>(env, "cast_f64_to_i64_serial", nk_cast_serial);
    run_cast<nk_u64_k, nk_f64_k>(env, "cast_u64_to_f64_serial", nk_cast_serial);
    run_cast<nk_f64_k, nk_u64_k>(env, "cast_f64_to_u64_serial", nk_cast_serial);
    // Serial — integer ↔ integer
    run_cast<nk_i8_k, nk_i32_k>(env, "cast_i8_to_i32_serial", nk_cast_serial);
    run_cast<nk_i32_k, nk_i8_k>(env, "cast_i32_to_i8_serial", nk_cast_serial);
    run_cast<nk_i16_k, nk_i64_k>(env, "cast_i16_to_i64_serial", nk_cast_serial);
}

} // namespace ashvardanian::numkong::bench
