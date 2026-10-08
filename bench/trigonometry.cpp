/**
 *  @file bench/trigonometry.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Trigonometry benchmarks, sin, cos, atan.
 */

#include "numkong/trigonometry.h"

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

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
nk_status_t elementwise_with_stl(scalar_type_ const *ins, nk_size_t n, scalar_type_ *outs, nk_stream_t) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = kernel_type_ {}(ins[i]);
    return nk_success_k;
}

/**
 *  @brief Measures trigonometric operations, sin, cos, atan.
 *  @param[inout] loop The timed loop, which takes the counters.
 *  @param[in] kernel The kernel function to benchmark.
 *  @param[in] dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_trigonometry(loop_t &loop, environment_t const &env, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate vectors for trigonometric kernels (unary: input + output)
    std::size_t const vector_count = input_sets_count(dtype_bytes(input_dtype_, 2 * dimensions));
    std::vector<input_vector_t> input_a(vector_count), output(vector_count);
    std::mt19937 generator(env.settings.seed.value);
    for (std::size_t index = 0; index != vector_count; ++index) {
        input_a[index] = make_vector<input_t>(dimensions);
        output[index] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, input_a[index].values_data(), dimensions);
    }

    // Benchmark loop
    for (std::size_t call : loop) {
        std::size_t const index = call & (vector_count - 1);
        if (!succeeded(loop,
                       kernel(input_a[index].raw_values_data(), dimensions, output[index].raw_values_data(), nullptr)))
            break;
        do_not_optimize(output[index].raw_values_data());
    }

    loop.byte_rate(input_a[0].size_bytes());
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_trigonometry(environment_t const &env, std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(env.settings.batch_per_core) + ">";
    run_benchmark(env, bench_name, measure_trigonometry<input_dtype_, kernel_type_ *>, kernel,
                  env.settings.batch_per_core);
}

void bench_trigonometry(environment_t const &env) {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;

#if NUMKONG_TARGET_NEON
    if (section(env, "Trigonometry NEON", nk_cap_neon_k)) {
        run_trigonometry<f32_k>(env, "trig_sin_f32_neon", nk_trig_sin_f32_neon);
        run_trigonometry<f32_k>(env, "trig_cos_f32_neon", nk_trig_cos_f32_neon);
        run_trigonometry<f32_k>(env, "trig_atan_f32_neon", nk_trig_atan_f32_neon);
        run_trigonometry<f64_k>(env, "trig_sin_f64_neon", nk_trig_sin_f64_neon);
        run_trigonometry<f64_k>(env, "trig_cos_f64_neon", nk_trig_cos_f64_neon);
        run_trigonometry<f64_k>(env, "trig_atan_f64_neon", nk_trig_atan_f64_neon);
    }
#endif

#if NUMKONG_TARGET_NEONHALF
    if (section(env, "Trigonometry NEON HALF", nk_cap_neonhalf_k)) {
        run_trigonometry<f16_k>(env, "trig_sin_f16_neonhalf", nk_trig_sin_f16_neonhalf);
        run_trigonometry<f16_k>(env, "trig_cos_f16_neonhalf", nk_trig_cos_f16_neonhalf);
        run_trigonometry<f16_k>(env, "trig_atan_f16_neonhalf", nk_trig_atan_f16_neonhalf);
    }
#endif

#if NUMKONG_TARGET_SVEHALF
    if (section(env, "Trigonometry SVE HALF", nk_cap_svehalf_k)) {
        run_trigonometry<f16_k>(env, "trig_sin_f16_svehalf", nk_trig_sin_f16_svehalf);
        run_trigonometry<f16_k>(env, "trig_cos_f16_svehalf", nk_trig_cos_f16_svehalf);
        run_trigonometry<f16_k>(env, "trig_atan_f16_svehalf", nk_trig_atan_f16_svehalf);
    }
#endif

#if NUMKONG_TARGET_HASWELL
    if (section(env, "Trigonometry Haswell", nk_cap_haswell_k)) {
        run_trigonometry<f32_k>(env, "trig_sin_f32_haswell", nk_trig_sin_f32_haswell);
        run_trigonometry<f32_k>(env, "trig_cos_f32_haswell", nk_trig_cos_f32_haswell);
        run_trigonometry<f32_k>(env, "trig_atan_f32_haswell", nk_trig_atan_f32_haswell);
        run_trigonometry<f64_k>(env, "trig_sin_f64_haswell", nk_trig_sin_f64_haswell);
        run_trigonometry<f64_k>(env, "trig_cos_f64_haswell", nk_trig_cos_f64_haswell);
        run_trigonometry<f64_k>(env, "trig_atan_f64_haswell", nk_trig_atan_f64_haswell);
    }
#endif

#if NUMKONG_TARGET_SKYLAKE
    if (section(env, "Trigonometry Skylake", nk_cap_skylake_k)) {
        run_trigonometry<f32_k>(env, "trig_sin_f32_skylake", nk_trig_sin_f32_skylake);
        run_trigonometry<f32_k>(env, "trig_cos_f32_skylake", nk_trig_cos_f32_skylake);
        run_trigonometry<f32_k>(env, "trig_atan_f32_skylake", nk_trig_atan_f32_skylake);
        run_trigonometry<f64_k>(env, "trig_sin_f64_skylake", nk_trig_sin_f64_skylake);
        run_trigonometry<f64_k>(env, "trig_cos_f64_skylake", nk_trig_cos_f64_skylake);
        run_trigonometry<f64_k>(env, "trig_atan_f64_skylake", nk_trig_atan_f64_skylake);
        run_trigonometry<f16_k>(env, "trig_sin_f16_skylake", nk_trig_sin_f16_skylake);
        run_trigonometry<f16_k>(env, "trig_cos_f16_skylake", nk_trig_cos_f16_skylake);
        run_trigonometry<f16_k>(env, "trig_atan_f16_skylake", nk_trig_atan_f16_skylake);
    }
#endif

#if NUMKONG_TARGET_SAPPHIRE
    if (section(env, "Trigonometry Sapphire", nk_cap_sapphire_k)) {
        run_trigonometry<f16_k>(env, "trig_sin_f16_sapphire", nk_trig_sin_f16_sapphire);
        run_trigonometry<f16_k>(env, "trig_cos_f16_sapphire", nk_trig_cos_f16_sapphire);
        run_trigonometry<f16_k>(env, "trig_atan_f16_sapphire", nk_trig_atan_f16_sapphire);
    }
#endif

#if NUMKONG_TARGET_V128RELAXED
    if (section(env, "Trigonometry V128 Relaxed", nk_cap_v128relaxed_k)) {
        run_trigonometry<f32_k>(env, "trig_sin_f32_v128relaxed", nk_trig_sin_f32_v128relaxed);
        run_trigonometry<f32_k>(env, "trig_cos_f32_v128relaxed", nk_trig_cos_f32_v128relaxed);
        run_trigonometry<f32_k>(env, "trig_atan_f32_v128relaxed", nk_trig_atan_f32_v128relaxed);
        run_trigonometry<f64_k>(env, "trig_sin_f64_v128relaxed", nk_trig_sin_f64_v128relaxed);
        run_trigonometry<f64_k>(env, "trig_cos_f64_v128relaxed", nk_trig_cos_f64_v128relaxed);
        run_trigonometry<f64_k>(env, "trig_atan_f64_v128relaxed", nk_trig_atan_f64_v128relaxed);
    }
#endif

    // STL baselines
    section(env, "Trigonometry Serial", nk_cap_serial_k);
    run_trigonometry<f32_k>(env, "trig_sin_f32_stl", elementwise_with_stl<nk_f32_t, sin_with_stl<nk_f32_t>>);
    run_trigonometry<f32_k>(env, "trig_cos_f32_stl", elementwise_with_stl<nk_f32_t, cos_with_stl<nk_f32_t>>);
    run_trigonometry<f32_k>(env, "trig_atan_f32_stl", elementwise_with_stl<nk_f32_t, atan_with_stl<nk_f32_t>>);
    run_trigonometry<f64_k>(env, "trig_sin_f64_stl", elementwise_with_stl<nk_f64_t, sin_with_stl<nk_f64_t>>);
    run_trigonometry<f64_k>(env, "trig_cos_f64_stl", elementwise_with_stl<nk_f64_t, cos_with_stl<nk_f64_t>>);
    run_trigonometry<f64_k>(env, "trig_atan_f64_stl", elementwise_with_stl<nk_f64_t, atan_with_stl<nk_f64_t>>);

    // Serial fallbacks
    run_trigonometry<f32_k>(env, "trig_sin_f32_serial", nk_trig_sin_f32_serial);
    run_trigonometry<f32_k>(env, "trig_cos_f32_serial", nk_trig_cos_f32_serial);
    run_trigonometry<f32_k>(env, "trig_atan_f32_serial", nk_trig_atan_f32_serial);
    run_trigonometry<f64_k>(env, "trig_sin_f64_serial", nk_trig_sin_f64_serial);
    run_trigonometry<f64_k>(env, "trig_cos_f64_serial", nk_trig_cos_f64_serial);
    run_trigonometry<f64_k>(env, "trig_atan_f64_serial", nk_trig_atan_f64_serial);
    run_trigonometry<f16_k>(env, "trig_sin_f16_serial", nk_trig_sin_f16_serial);
    run_trigonometry<f16_k>(env, "trig_cos_f16_serial", nk_trig_cos_f16_serial);
    run_trigonometry<f16_k>(env, "trig_atan_f16_serial", nk_trig_atan_f16_serial);
}

} // namespace ashvardanian::numkong::bench
