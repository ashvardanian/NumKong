/**
 *  @brief Elementwise operations tests.
 *  @file test/test_each.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/each.hpp"          // `nk::sum`, `nk::scale`, `nk::blend`, `nk::fma`
#include "numkong/trigonometry.hpp"  // `nk::try_sin`, `nk::try_cos`, `nk::try_atan` wrappers

template <typename scalar_type_, typename generator_type_>
typename scalar_type_::scale_t random_coef(generator_type_ &gen) {
    using scale_t = typename scalar_type_::scale_t;
    if constexpr (scalar_type_::is_complex()) {
        using component_raw_t = typename scalar_type_::component_t::raw_t;
        std::uniform_real_distribution<component_raw_t> dist(-2, 2);
        scale_t coef;
        coef.real = dist(gen), coef.imag = dist(gen);
        return coef;
    }
    else {
        std::uniform_real_distribution<scale_t> dist(scale_t(-2), scale_t(2));
        return dist(gen);
    }
}

/**
 *  @brief Unified test for elementwise sum: result[i] = a[i] + b[i]
 */
template <typename scalar_type_>
error_stats_t test_sum(typename scalar_type_::sum_kernel_t kernel) {
    using scalar_t = scalar_type_;

    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::narrow_arithmetic_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, result.raw_values_data());
        nk::sum<scalar_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                         reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for scale: result[i] = alpha * x[i] + beta
 */
template <typename scalar_type_>
error_stats_t test_scale(typename scalar_type_::scale_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scale_t = typename scalar_t::scale_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::narrow_arithmetic_k);
    std::mt19937 generator(global_config.seed);
    auto input = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, input);
        scale_t alpha = random_coef<scalar_t>(generator);
        scale_t beta = random_coef<scalar_t>(generator);

        kernel(input.raw_values_data(), global_config.dense_dimensions, &alpha, &beta, result.raw_values_data());
        nk::scale<scalar_t, reference_t, nk::no_simd_k>(input.values_data(), global_config.dense_dimensions, &alpha,
                                                        &beta, reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for blend: result[i] = alpha * a[i] + beta * b[i]
 */
template <typename scalar_type_>
error_stats_t test_blend(typename scalar_type_::blend_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scale_t = typename scalar_t::scale_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::narrow_arithmetic_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        scale_t alpha = random_coef<scalar_t>(generator);
        scale_t beta = random_coef<scalar_t>(generator);

        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &alpha, &beta,
               result.raw_values_data());
        nk::blend<scalar_t, reference_t, nk::no_simd_k>(
            a.values_data(), b.values_data(), global_config.dense_dimensions, &alpha, &beta, reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for FMA: result[i] = alpha * a[i] * b[i] + beta * c[i]
 */
template <typename scalar_type_>
error_stats_t test_fma(typename scalar_type_::fma_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scale_t = typename scalar_t::scale_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::narrow_arithmetic_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto c = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, c);
        scale_t alpha = random_coef<scalar_t>(generator);
        scale_t beta = random_coef<scalar_t>(generator);

        kernel(a.raw_values_data(), b.raw_values_data(), c.raw_values_data(), global_config.dense_dimensions, &alpha,
               &beta, result.raw_values_data());
        nk::fma<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                      c.values_data(), &alpha, &beta, reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Smoke-test for the tensor-shaped trig wrappers (`nk::try_sin`/`cos`/`atan`).
 *  Runs allocating + into-span variants on a small zero tensor — just exercises the dispatch
 *  paths, not the numerical accuracy (the latter is covered by the kernel tests above).
 */
template <typename value_type_>
void test_tensor_trig_for_type() {
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto out = tensor_t::try_zeros({4, 8});
    auto av = a.view();

    { [[maybe_unused]] auto r = nk::try_sin<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::try_cos<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::try_atan<value_type_>(av); }
    { [[maybe_unused]] bool ok = nk::sin<value_type_>(av, out.span()); }
    { [[maybe_unused]] bool ok = nk::cos<value_type_>(av, out.span()); }
    { [[maybe_unused]] bool ok = nk::atan<value_type_>(av, out.span()); }
}

void test_each() {
    error_stats_section_t check("Elementwise Operations");

    // Tensor-shaped trig wrappers (float-capable types).
    test_tensor_trig_for_type<nk::f32_t>();
    test_tensor_trig_for_type<nk::f64_t>();
    test_tensor_trig_for_type<nk::f16_t>();
    test_tensor_trig_for_type<nk::bf16_t>();
    std::printf("  trig (4 types):               OK\n");


#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    check("each_scale_f32", test_scale<f32_t>, nk_each_scale_f32);
    check("each_sum_f32", test_sum<f32_t>, nk_each_sum_f32);
    check("each_blend_f32", test_blend<f32_t>, nk_each_blend_f32);
    check("each_fma_f32", test_fma<f32_t>, nk_each_fma_f32);
    check("each_scale_e4m3", test_scale<e4m3_t>, nk_each_scale_e4m3);
    check("each_scale_e5m2", test_scale<e5m2_t>, nk_each_scale_e5m2);
    check("each_sum_e4m3", test_sum<e4m3_t>, nk_each_sum_e4m3);
    check("each_sum_e5m2", test_sum<e5m2_t>, nk_each_sum_e5m2);
    check("each_blend_e4m3", test_blend<e4m3_t>, nk_each_blend_e4m3);
    check("each_blend_e5m2", test_blend<e5m2_t>, nk_each_blend_e5m2);
    check("each_fma_e4m3", test_fma<e4m3_t>, nk_each_fma_e4m3);
    check("each_fma_e5m2", test_fma<e5m2_t>, nk_each_fma_e5m2);
    check("each_sum_f32c", test_sum<f32c_t>, nk_each_sum_f32c);
    check("each_sum_f64c", test_sum<f64c_t>, nk_each_sum_f64c);
    check("each_scale_f32c", test_scale<f32c_t>, nk_each_scale_f32c);
    check("each_scale_f64c", test_scale<f64c_t>, nk_each_scale_f64c);
    check("each_blend_f32c", test_blend<f32c_t>, nk_each_blend_f32c);
    check("each_blend_f64c", test_blend<f64c_t>, nk_each_blend_f64c);
    check("each_fma_f32c", test_fma<f32c_t>, nk_each_fma_f32c);
    check("each_fma_f64c", test_fma<f64c_t>, nk_each_fma_f64c);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    // f64
    check("each_sum_f64_neon", test_sum<f64_t>, nk_each_sum_f64_neon);
    check("each_scale_f64_neon", test_scale<f64_t>, nk_each_scale_f64_neon);
    check("each_blend_f64_neon", test_blend<f64_t>, nk_each_blend_f64_neon);
    check("each_fma_f64_neon", test_fma<f64_t>, nk_each_fma_f64_neon);
    // f32
    check("each_sum_f32_neon", test_sum<f32_t>, nk_each_sum_f32_neon);
    check("each_scale_f32_neon", test_scale<f32_t>, nk_each_scale_f32_neon);
    check("each_blend_f32_neon", test_blend<f32_t>, nk_each_blend_f32_neon);
    check("each_fma_f32_neon", test_fma<f32_t>, nk_each_fma_f32_neon);
    // e4m3, e5m2
    check("each_sum_e4m3_neon", test_sum<e4m3_t>, nk_each_sum_e4m3_neon);
    check("each_scale_e4m3_neon", test_scale<e4m3_t>, nk_each_scale_e4m3_neon);
    check("each_blend_e4m3_neon", test_blend<e4m3_t>, nk_each_blend_e4m3_neon);
    check("each_fma_e4m3_neon", test_fma<e4m3_t>, nk_each_fma_e4m3_neon);
    check("each_sum_e5m2_neon", test_sum<e5m2_t>, nk_each_sum_e5m2_neon);
    check("each_scale_e5m2_neon", test_scale<e5m2_t>, nk_each_scale_e5m2_neon);
    check("each_blend_e5m2_neon", test_blend<e5m2_t>, nk_each_blend_e5m2_neon);
    check("each_fma_e5m2_neon", test_fma<e5m2_t>, nk_each_fma_e5m2_neon);
    // u8, i8
    check("each_sum_u8_neon", test_sum<u8_t>, nk_each_sum_u8_neon);
    check("each_sum_i8_neon", test_sum<i8_t>, nk_each_sum_i8_neon);
    // i16, u16
    check("each_sum_i16_neon", test_sum<i16_t>, nk_each_sum_i16_neon);
    check("each_scale_i16_neon", test_scale<i16_t>, nk_each_scale_i16_neon);
    check("each_fma_i16_neon", test_fma<i16_t>, nk_each_fma_i16_neon);
    check("each_sum_u16_neon", test_sum<u16_t>, nk_each_sum_u16_neon);
    check("each_scale_u16_neon", test_scale<u16_t>, nk_each_scale_u16_neon);
    check("each_fma_u16_neon", test_fma<u16_t>, nk_each_fma_u16_neon);
    // i32, u32
    check("each_sum_i32_neon", test_sum<i32_t>, nk_each_sum_i32_neon);
    check("each_scale_i32_neon", test_scale<i32_t>, nk_each_scale_i32_neon);
    check("each_fma_i32_neon", test_fma<i32_t>, nk_each_fma_i32_neon);
    check("each_sum_u32_neon", test_sum<u32_t>, nk_each_sum_u32_neon);
    check("each_scale_u32_neon", test_scale<u32_t>, nk_each_scale_u32_neon);
    check("each_fma_u32_neon", test_fma<u32_t>, nk_each_fma_u32_neon);
    // i64, u64
    check("each_sum_i64_neon", test_sum<i64_t>, nk_each_sum_i64_neon);
    check("each_scale_i64_neon", test_scale<i64_t>, nk_each_scale_i64_neon);
    check("each_fma_i64_neon", test_fma<i64_t>, nk_each_fma_i64_neon);
    check("each_sum_u64_neon", test_sum<u64_t>, nk_each_sum_u64_neon);
    check("each_scale_u64_neon", test_scale<u64_t>, nk_each_scale_u64_neon);
    check("each_fma_u64_neon", test_fma<u64_t>, nk_each_fma_u64_neon);
    // complex
    check("each_scale_f32c_neon", test_scale<f32c_t>, nk_each_scale_f32c_neon);
    check("each_blend_f32c_neon", test_blend<f32c_t>, nk_each_blend_f32c_neon);
    check("each_fma_f32c_neon", test_fma<f32c_t>, nk_each_fma_f32c_neon);
    check("each_scale_f64c_neon", test_scale<f64c_t>, nk_each_scale_f64c_neon);
    check("each_blend_f64c_neon", test_blend<f64c_t>, nk_each_blend_f64c_neon);
    check("each_fma_f64c_neon", test_fma<f64c_t>, nk_each_fma_f64c_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    check("each_scale_f16_neonhalf", test_scale<f16_t>, nk_each_scale_f16_neonhalf);
    check("each_sum_f16_neonhalf", test_sum<f16_t>, nk_each_sum_f16_neonhalf);
    check("each_blend_f16_neonhalf", test_blend<f16_t>, nk_each_blend_f16_neonhalf);
    check("each_fma_f16_neonhalf", test_fma<f16_t>, nk_each_fma_f16_neonhalf);
    check("each_scale_u8_neonhalf", test_scale<u8_t>, nk_each_scale_u8_neonhalf);
    check("each_blend_u8_neonhalf", test_blend<u8_t>, nk_each_blend_u8_neonhalf);
    check("each_scale_i8_neonhalf", test_scale<i8_t>, nk_each_scale_i8_neonhalf);
    check("each_blend_i8_neonhalf", test_blend<i8_t>, nk_each_blend_i8_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    check("each_scale_bf16_neonbfdot", test_scale<bf16_t>, nk_each_scale_bf16_neonbfdot);
    check("each_sum_bf16_neonbfdot", test_sum<bf16_t>, nk_each_sum_bf16_neonbfdot);
    check("each_blend_bf16_neonbfdot", test_blend<bf16_t>, nk_each_blend_bf16_neonbfdot);
    check("each_fma_bf16_neonbfdot", test_fma<bf16_t>, nk_each_fma_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_HASWELL
    check("each_scale_f32_haswell", test_scale<f32_t>, nk_each_scale_f32_haswell);
    check("each_sum_f32_haswell", test_sum<f32_t>, nk_each_sum_f32_haswell);
    check("each_blend_f32_haswell", test_blend<f32_t>, nk_each_blend_f32_haswell);
    check("each_fma_f32_haswell", test_fma<f32_t>, nk_each_fma_f32_haswell);
    check("each_scale_e4m3_haswell", test_scale<e4m3_t>, nk_each_scale_e4m3_haswell);
    check("each_scale_e5m2_haswell", test_scale<e5m2_t>, nk_each_scale_e5m2_haswell);
    check("each_sum_e4m3_haswell", test_sum<e4m3_t>, nk_each_sum_e4m3_haswell);
    check("each_sum_e5m2_haswell", test_sum<e5m2_t>, nk_each_sum_e5m2_haswell);
    check("each_blend_e4m3_haswell", test_blend<e4m3_t>, nk_each_blend_e4m3_haswell);
    check("each_blend_e5m2_haswell", test_blend<e5m2_t>, nk_each_blend_e5m2_haswell);
    check("each_fma_e4m3_haswell", test_fma<e4m3_t>, nk_each_fma_e4m3_haswell);
    check("each_fma_e5m2_haswell", test_fma<e5m2_t>, nk_each_fma_e5m2_haswell);
    check("each_scale_f32c_haswell", test_scale<f32c_t>, nk_each_scale_f32c_haswell);
    check("each_scale_f64c_haswell", test_scale<f64c_t>, nk_each_scale_f64c_haswell);
    check("each_blend_f32c_haswell", test_blend<f32c_t>, nk_each_blend_f32c_haswell);
    check("each_blend_f64c_haswell", test_blend<f64c_t>, nk_each_blend_f64c_haswell);
    check("each_fma_f32c_haswell", test_fma<f32c_t>, nk_each_fma_f32c_haswell);
    check("each_fma_f64c_haswell", test_fma<f64c_t>, nk_each_fma_f64c_haswell);
    check("each_blend_bf16_haswell", test_blend<bf16_t>, nk_each_blend_bf16_haswell);
    check("each_blend_f64_haswell", test_blend<f64_t>, nk_each_blend_f64_haswell);
    check("each_blend_i8_haswell", test_blend<i8_t>, nk_each_blend_i8_haswell);
    check("each_blend_u8_haswell", test_blend<u8_t>, nk_each_blend_u8_haswell);
    check("each_blend_f16_haswell", test_blend<f16_t>, nk_each_blend_f16_haswell);
    check("each_fma_bf16_haswell", test_fma<bf16_t>, nk_each_fma_bf16_haswell);
    check("each_fma_f64_haswell", test_fma<f64_t>, nk_each_fma_f64_haswell);
    check("each_fma_i16_haswell", test_fma<i16_t>, nk_each_fma_i16_haswell);
    check("each_fma_i8_haswell", test_fma<i8_t>, nk_each_fma_i8_haswell);
    check("each_fma_u16_haswell", test_fma<u16_t>, nk_each_fma_u16_haswell);
    check("each_fma_u8_haswell", test_fma<u8_t>, nk_each_fma_u8_haswell);
    check("each_fma_f16_haswell", test_fma<f16_t>, nk_each_fma_f16_haswell);
    check("each_scale_bf16_haswell", test_scale<bf16_t>, nk_each_scale_bf16_haswell);
    check("each_scale_f16_haswell", test_scale<f16_t>, nk_each_scale_f16_haswell);
    check("each_scale_f64_haswell", test_scale<f64_t>, nk_each_scale_f64_haswell);
    check("each_scale_i16_haswell", test_scale<i16_t>, nk_each_scale_i16_haswell);
    check("each_scale_i8_haswell", test_scale<i8_t>, nk_each_scale_i8_haswell);
    check("each_scale_u16_haswell", test_scale<u16_t>, nk_each_scale_u16_haswell);
    check("each_scale_u8_haswell", test_scale<u8_t>, nk_each_scale_u8_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check("each_scale_f32_skylake", test_scale<f32_t>, nk_each_scale_f32_skylake);
    check("each_sum_f32_skylake", test_sum<f32_t>, nk_each_sum_f32_skylake);
    check("each_blend_f32_skylake", test_blend<f32_t>, nk_each_blend_f32_skylake);
    check("each_fma_f32_skylake", test_fma<f32_t>, nk_each_fma_f32_skylake);
    check("each_scale_e4m3_skylake", test_scale<e4m3_t>, nk_each_scale_e4m3_skylake);
    check("each_scale_e5m2_skylake", test_scale<e5m2_t>, nk_each_scale_e5m2_skylake);
    check("each_sum_e4m3_skylake", test_sum<e4m3_t>, nk_each_sum_e4m3_skylake);
    check("each_sum_e5m2_skylake", test_sum<e5m2_t>, nk_each_sum_e5m2_skylake);
    check("each_blend_e4m3_skylake", test_blend<e4m3_t>, nk_each_blend_e4m3_skylake);
    check("each_blend_e5m2_skylake", test_blend<e5m2_t>, nk_each_blend_e5m2_skylake);
    check("each_fma_e4m3_skylake", test_fma<e4m3_t>, nk_each_fma_e4m3_skylake);
    check("each_fma_e5m2_skylake", test_fma<e5m2_t>, nk_each_fma_e5m2_skylake);
    check("each_scale_f32c_skylake", test_scale<f32c_t>, nk_each_scale_f32c_skylake);
    check("each_scale_f64c_skylake", test_scale<f64c_t>, nk_each_scale_f64c_skylake);
    check("each_blend_f32c_skylake", test_blend<f32c_t>, nk_each_blend_f32c_skylake);
    check("each_blend_f64c_skylake", test_blend<f64c_t>, nk_each_blend_f64c_skylake);
    check("each_fma_f32c_skylake", test_fma<f32c_t>, nk_each_fma_f32c_skylake);
    check("each_fma_f64c_skylake", test_fma<f64c_t>, nk_each_fma_f64c_skylake);
    check("each_scale_f16_skylake", test_scale<f16_t>, nk_each_scale_f16_skylake);
    check("each_blend_f16_skylake", test_blend<f16_t>, nk_each_blend_f16_skylake);
    check("each_fma_f16_skylake", test_fma<f16_t>, nk_each_fma_f16_skylake);
    check("each_blend_bf16_skylake", test_blend<bf16_t>, nk_each_blend_bf16_skylake);
    check("each_blend_f64_skylake", test_blend<f64_t>, nk_each_blend_f64_skylake);
    check("each_fma_bf16_skylake", test_fma<bf16_t>, nk_each_fma_bf16_skylake);
    check("each_fma_f64_skylake", test_fma<f64_t>, nk_each_fma_f64_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
    check("each_sum_i8_icelake", test_sum<i8_t>, nk_each_sum_i8_icelake);
    check("each_sum_u8_icelake", test_sum<u8_t>, nk_each_sum_u8_icelake);
    check("each_sum_i16_icelake", test_sum<i16_t>, nk_each_sum_i16_icelake);
    check("each_sum_u16_icelake", test_sum<u16_t>, nk_each_sum_u16_icelake);
    check("each_sum_i32_icelake", test_sum<i32_t>, nk_each_sum_i32_icelake);
    check("each_sum_u32_icelake", test_sum<u32_t>, nk_each_sum_u32_icelake);
    check("each_sum_i64_icelake", test_sum<i64_t>, nk_each_sum_i64_icelake);
    check("each_sum_u64_icelake", test_sum<u64_t>, nk_each_sum_u64_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_SAPPHIRE
    check("each_sum_f16_sapphire", test_sum<f16_t>, nk_each_sum_f16_sapphire);
    check("each_scale_u8_sapphire", test_scale<u8_t>, nk_each_scale_u8_sapphire);
    check("each_blend_u8_sapphire", test_blend<u8_t>, nk_each_blend_u8_sapphire);
    check("each_scale_i8_sapphire", test_scale<i8_t>, nk_each_scale_i8_sapphire);
    check("each_blend_i8_sapphire", test_blend<i8_t>, nk_each_blend_i8_sapphire);
    check("each_sum_e4m3_sapphire", test_sum<e4m3_t>, nk_each_sum_e4m3_sapphire);
#endif // NK_TARGET_SAPPHIRE

#if NK_TARGET_RVV
    check("each_sum_f64_rvv", test_sum<f64_t>, nk_each_sum_f64_rvv);
    check("each_scale_f64_rvv", test_scale<f64_t>, nk_each_scale_f64_rvv);
    check("each_blend_f64_rvv", test_blend<f64_t>, nk_each_blend_f64_rvv);
    check("each_fma_f64_rvv", test_fma<f64_t>, nk_each_fma_f64_rvv);
    check("each_sum_f32_rvv", test_sum<f32_t>, nk_each_sum_f32_rvv);
    check("each_scale_f32_rvv", test_scale<f32_t>, nk_each_scale_f32_rvv);
    check("each_blend_f32_rvv", test_blend<f32_t>, nk_each_blend_f32_rvv);
    check("each_fma_f32_rvv", test_fma<f32_t>, nk_each_fma_f32_rvv);
    check("each_sum_f16_rvv", test_sum<f16_t>, nk_each_sum_f16_rvv);
    check("each_scale_f16_rvv", test_scale<f16_t>, nk_each_scale_f16_rvv);
    check("each_blend_f16_rvv", test_blend<f16_t>, nk_each_blend_f16_rvv);
    check("each_fma_f16_rvv", test_fma<f16_t>, nk_each_fma_f16_rvv);
    check("each_sum_bf16_rvv", test_sum<bf16_t>, nk_each_sum_bf16_rvv);
    check("each_scale_bf16_rvv", test_scale<bf16_t>, nk_each_scale_bf16_rvv);
    check("each_blend_bf16_rvv", test_blend<bf16_t>, nk_each_blend_bf16_rvv);
    check("each_fma_bf16_rvv", test_fma<bf16_t>, nk_each_fma_bf16_rvv);
    check("each_sum_e4m3_rvv", test_sum<e4m3_t>, nk_each_sum_e4m3_rvv);
    check("each_scale_e4m3_rvv", test_scale<e4m3_t>, nk_each_scale_e4m3_rvv);
    check("each_blend_e4m3_rvv", test_blend<e4m3_t>, nk_each_blend_e4m3_rvv);
    check("each_fma_e4m3_rvv", test_fma<e4m3_t>, nk_each_fma_e4m3_rvv);
    check("each_sum_e5m2_rvv", test_sum<e5m2_t>, nk_each_sum_e5m2_rvv);
    check("each_scale_e5m2_rvv", test_scale<e5m2_t>, nk_each_scale_e5m2_rvv);
    check("each_blend_e5m2_rvv", test_blend<e5m2_t>, nk_each_blend_e5m2_rvv);
    check("each_fma_e5m2_rvv", test_fma<e5m2_t>, nk_each_fma_e5m2_rvv);
    check("each_sum_i8_rvv", test_sum<i8_t>, nk_each_sum_i8_rvv);
    check("each_scale_i8_rvv", test_scale<i8_t>, nk_each_scale_i8_rvv);
    check("each_blend_i8_rvv", test_blend<i8_t>, nk_each_blend_i8_rvv);
    check("each_fma_i8_rvv", test_fma<i8_t>, nk_each_fma_i8_rvv);
    check("each_sum_u8_rvv", test_sum<u8_t>, nk_each_sum_u8_rvv);
    check("each_scale_u8_rvv", test_scale<u8_t>, nk_each_scale_u8_rvv);
    check("each_blend_u8_rvv", test_blend<u8_t>, nk_each_blend_u8_rvv);
    check("each_fma_u8_rvv", test_fma<u8_t>, nk_each_fma_u8_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    check("each_sum_f32_v128relaxed", test_sum<f32_t>, nk_each_sum_f32_v128relaxed);
    check("each_scale_f32_v128relaxed", test_scale<f32_t>, nk_each_scale_f32_v128relaxed);
    check("each_blend_f32_v128relaxed", test_blend<f32_t>, nk_each_blend_f32_v128relaxed);
    check("each_fma_f32_v128relaxed", test_fma<f32_t>, nk_each_fma_f32_v128relaxed);
    check("each_sum_f16_v128relaxed", test_sum<f16_t>, nk_each_sum_f16_v128relaxed);
    check("each_scale_f16_v128relaxed", test_scale<f16_t>, nk_each_scale_f16_v128relaxed);
    check("each_blend_f16_v128relaxed", test_blend<f16_t>, nk_each_blend_f16_v128relaxed);
    check("each_fma_f16_v128relaxed", test_fma<f16_t>, nk_each_fma_f16_v128relaxed);
    check("each_sum_bf16_v128relaxed", test_sum<bf16_t>, nk_each_sum_bf16_v128relaxed);
    check("each_scale_bf16_v128relaxed", test_scale<bf16_t>, nk_each_scale_bf16_v128relaxed);
    check("each_blend_bf16_v128relaxed", test_blend<bf16_t>, nk_each_blend_bf16_v128relaxed);
    check("each_fma_bf16_v128relaxed", test_fma<bf16_t>, nk_each_fma_bf16_v128relaxed);
    check("each_sum_i8_v128relaxed", test_sum<i8_t>, nk_each_sum_i8_v128relaxed);
    check("each_scale_i8_v128relaxed", test_scale<i8_t>, nk_each_scale_i8_v128relaxed);
    check("each_blend_i8_v128relaxed", test_blend<i8_t>, nk_each_blend_i8_v128relaxed);
    check("each_fma_i8_v128relaxed", test_fma<i8_t>, nk_each_fma_i8_v128relaxed);
    check("each_sum_u8_v128relaxed", test_sum<u8_t>, nk_each_sum_u8_v128relaxed);
    check("each_scale_u8_v128relaxed", test_scale<u8_t>, nk_each_scale_u8_v128relaxed);
    check("each_blend_u8_v128relaxed", test_blend<u8_t>, nk_each_blend_u8_v128relaxed);
    check("each_fma_u8_v128relaxed", test_fma<u8_t>, nk_each_fma_u8_v128relaxed);
#endif // NK_TARGET_V128RELAXED

    // Serial always runs - baseline test
    check("each_scale_f32_serial", test_scale<f32_t>, nk_each_scale_f32_serial);
    check("each_sum_f32_serial", test_sum<f32_t>, nk_each_sum_f32_serial);
    check("each_blend_f32_serial", test_blend<f32_t>, nk_each_blend_f32_serial);
    check("each_fma_f32_serial", test_fma<f32_t>, nk_each_fma_f32_serial);
    check("each_scale_e4m3_serial", test_scale<e4m3_t>, nk_each_scale_e4m3_serial);
    check("each_scale_e5m2_serial", test_scale<e5m2_t>, nk_each_scale_e5m2_serial);
    check("each_sum_e4m3_serial", test_sum<e4m3_t>, nk_each_sum_e4m3_serial);
    check("each_sum_e5m2_serial", test_sum<e5m2_t>, nk_each_sum_e5m2_serial);
    check("each_blend_e4m3_serial", test_blend<e4m3_t>, nk_each_blend_e4m3_serial);
    check("each_blend_e5m2_serial", test_blend<e5m2_t>, nk_each_blend_e5m2_serial);
    check("each_fma_e4m3_serial", test_fma<e4m3_t>, nk_each_fma_e4m3_serial);
    check("each_fma_e5m2_serial", test_fma<e5m2_t>, nk_each_fma_e5m2_serial);
    check("each_sum_f32c_serial", test_sum<f32c_t>, nk_each_sum_f32c_serial);
    check("each_sum_f64c_serial", test_sum<f64c_t>, nk_each_sum_f64c_serial);
    check("each_scale_f32c_serial", test_scale<f32c_t>, nk_each_scale_f32c_serial);
    check("each_scale_f64c_serial", test_scale<f64c_t>, nk_each_scale_f64c_serial);
    check("each_blend_f32c_serial", test_blend<f32c_t>, nk_each_blend_f32c_serial);
    check("each_blend_f64c_serial", test_blend<f64c_t>, nk_each_blend_f64c_serial);
    check("each_fma_f32c_serial", test_fma<f32c_t>, nk_each_fma_f32c_serial);
    check("each_fma_f64c_serial", test_fma<f64c_t>, nk_each_fma_f64c_serial);
    check("each_blend_f16_serial", test_blend<f16_t>, nk_each_blend_f16_serial);
    check("each_blend_i8_serial", test_blend<i8_t>, nk_each_blend_i8_serial);
    check("each_blend_u8_serial", test_blend<u8_t>, nk_each_blend_u8_serial);
    check("each_fma_f16_serial", test_fma<f16_t>, nk_each_fma_f16_serial);
    check("each_fma_i8_serial", test_fma<i8_t>, nk_each_fma_i8_serial);
    check("each_fma_u8_serial", test_fma<u8_t>, nk_each_fma_u8_serial);
    check("each_sum_f64_serial", test_sum<f64_t>, nk_each_sum_f64_serial);
    check("each_scale_f64_serial", test_scale<f64_t>, nk_each_scale_f64_serial);
    check("each_blend_f64_serial", test_blend<f64_t>, nk_each_blend_f64_serial);
    check("each_fma_f64_serial", test_fma<f64_t>, nk_each_fma_f64_serial);
    check("each_sum_bf16_serial", test_sum<bf16_t>, nk_each_sum_bf16_serial);
    check("each_scale_bf16_serial", test_scale<bf16_t>, nk_each_scale_bf16_serial);
    check("each_blend_bf16_serial", test_blend<bf16_t>, nk_each_blend_bf16_serial);
    check("each_fma_bf16_serial", test_fma<bf16_t>, nk_each_fma_bf16_serial);
    check("each_sum_f16_serial", test_sum<f16_t>, nk_each_sum_f16_serial);
    check("each_scale_f16_serial", test_scale<f16_t>, nk_each_scale_f16_serial);

#endif // NK_DYNAMIC_DISPATCH
}
