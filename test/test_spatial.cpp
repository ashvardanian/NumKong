/**
 *  @brief Spatial distance tests.
 *  @file test/test_spatial.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/spatial.hpp"

/**
 *  @brief Unified squared Euclidean distance test for float types.
 *  Works with f32_t, f64_t, f16_t, bf16_t wrapper types.
 */
template <typename scalar_type_>
error_stats_t test_sqeuclidean(typename scalar_type_::sqeuclidean_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::sqeuclidean_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::mixed_precision_reduction_k);
    std::mt19937 generator(global_config.seed);
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const n = nk::divide_round_up(global_config.dense_dimensions, dims_per_value) * dims_per_value;
    auto a = make_vector<scalar_t>(n), b = make_vector<scalar_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_);

        reference_t reference;
        nk::sqeuclidean<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Unified angular (cosine) distance test for float types.
 *  Works with f32_t, f64_t, f16_t, bf16_t wrapper types.
 */
template <typename scalar_type_>
error_stats_t test_angular(typename scalar_type_::angular_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::angular_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::mixed_precision_reduction_k);
    std::mt19937 generator(global_config.seed);
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const n = nk::divide_round_up(global_config.dense_dimensions, dims_per_value) * dims_per_value;
    auto a = make_vector<scalar_t>(n), b = make_vector<scalar_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_);

        reference_t reference;
        nk::angular<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Unified Euclidean distance test for float types.
 *  Works with f32_t, f64_t, f16_t, bf16_t, e2m3_t, e3m2_t wrapper types.
 */
template <typename scalar_type_>
error_stats_t test_euclidean(typename scalar_type_::euclidean_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::euclidean_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::mixed_precision_reduction_k);
    std::mt19937 generator(global_config.seed);
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const n = nk::divide_round_up(global_config.dense_dimensions, dims_per_value) * dims_per_value;
    auto a = make_vector<scalar_t>(n), b = make_vector<scalar_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_);

        reference_t reference;
        nk::euclidean<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_spatial() {
    error_stats_section_t check("Spatial Distances");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    check("sqeuclidean_f32", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32);
    check("sqeuclidean_f64", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64);
    check("sqeuclidean_f16", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16);
    check("sqeuclidean_bf16", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16);
    check("sqeuclidean_e2m3", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3);
    check("sqeuclidean_e3m2", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2);
    check("euclidean_f32", test_euclidean<f32_t>, nk_euclidean_f32);
    check("euclidean_f64", test_euclidean<f64_t>, nk_euclidean_f64);
    check("euclidean_f16", test_euclidean<f16_t>, nk_euclidean_f16);
    check("euclidean_bf16", test_euclidean<bf16_t>, nk_euclidean_bf16);
    check("euclidean_e2m3", test_euclidean<e2m3_t>, nk_euclidean_e2m3);
    check("euclidean_e3m2", test_euclidean<e3m2_t>, nk_euclidean_e3m2);
    check("angular_f32", test_angular<f32_t>, nk_angular_f32);
    check("angular_f64", test_angular<f64_t>, nk_angular_f64);
    check("angular_f16", test_angular<f16_t>, nk_angular_f16);
    check("angular_bf16", test_angular<bf16_t>, nk_angular_bf16);
    check("angular_e2m3", test_angular<e2m3_t>, nk_angular_e2m3);
    check("angular_e3m2", test_angular<e3m2_t>, nk_angular_e3m2);
    check("sqeuclidean_i4", test_sqeuclidean<i4x2_t>, nk_sqeuclidean_i4);
    check("sqeuclidean_u4", test_sqeuclidean<u4x2_t>, nk_sqeuclidean_u4);
    check("angular_i4", test_angular<i4x2_t>, nk_angular_i4);
    check("angular_u4", test_angular<u4x2_t>, nk_angular_u4);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    check("sqeuclidean_f64_neon", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_neon);
    check("sqeuclidean_f32_neon", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_neon);
    check("sqeuclidean_bf16_neon", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_neon);
    check("sqeuclidean_f16_neon", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_neon);
    check("sqeuclidean_e5m2_neon", test_sqeuclidean<e5m2_t>, nk_sqeuclidean_e5m2_neon);
    check("sqeuclidean_e4m3_neon", test_sqeuclidean<e4m3_t>, nk_sqeuclidean_e4m3_neon);
    check("sqeuclidean_e3m2_neon", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2_neon);
    check("sqeuclidean_e2m3_neon", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3_neon);
    check("euclidean_f64_neon", test_euclidean<f64_t>, nk_euclidean_f64_neon);
    check("euclidean_f32_neon", test_euclidean<f32_t>, nk_euclidean_f32_neon);
    check("euclidean_bf16_neon", test_euclidean<bf16_t>, nk_euclidean_bf16_neon);
    check("euclidean_f16_neon", test_euclidean<f16_t>, nk_euclidean_f16_neon);
    check("euclidean_e5m2_neon", test_euclidean<e5m2_t>, nk_euclidean_e5m2_neon);
    check("euclidean_e4m3_neon", test_euclidean<e4m3_t>, nk_euclidean_e4m3_neon);
    check("euclidean_e3m2_neon", test_euclidean<e3m2_t>, nk_euclidean_e3m2_neon);
    check("euclidean_e2m3_neon", test_euclidean<e2m3_t>, nk_euclidean_e2m3_neon);
    check("angular_f64_neon", test_angular<f64_t>, nk_angular_f64_neon);
    check("angular_f32_neon", test_angular<f32_t>, nk_angular_f32_neon);
    check("angular_bf16_neon", test_angular<bf16_t>, nk_angular_bf16_neon);
    check("angular_f16_neon", test_angular<f16_t>, nk_angular_f16_neon);
    check("angular_e5m2_neon", test_angular<e5m2_t>, nk_angular_e5m2_neon);
    check("angular_e4m3_neon", test_angular<e4m3_t>, nk_angular_e4m3_neon);
    check("angular_e3m2_neon", test_angular<e3m2_t>, nk_angular_e3m2_neon);
    check("angular_e2m3_neon", test_angular<e2m3_t>, nk_angular_e2m3_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONBFDOT
    check("sqeuclidean_bf16_neonbfdot", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_neonbfdot);
    check("euclidean_bf16_neonbfdot", test_euclidean<bf16_t>, nk_euclidean_bf16_neonbfdot);
    check("angular_bf16_neonbfdot", test_angular<bf16_t>, nk_angular_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONSDOT
    check("angular_i8_neonsdot", test_angular<i8_t>, nk_angular_i8_neonsdot);
    check("sqeuclidean_i8_neonsdot", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_neonsdot);
    check("euclidean_i8_neonsdot", test_euclidean<i8_t>, nk_euclidean_i8_neonsdot);
    check("angular_u8_neonsdot", test_angular<u8_t>, nk_angular_u8_neonsdot);
    check("sqeuclidean_u8_neonsdot", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_neonsdot);
    check("euclidean_u8_neonsdot", test_euclidean<u8_t>, nk_euclidean_u8_neonsdot);
    check("angular_i4_neonsdot", test_angular<i4x2_t>, nk_angular_i4_neonsdot);
    check("sqeuclidean_i4_neonsdot", test_sqeuclidean<i4x2_t>, nk_sqeuclidean_i4_neonsdot);
    check("euclidean_i4_neonsdot", test_euclidean<i4x2_t>, nk_euclidean_i4_neonsdot);
    check("angular_u4_neonsdot", test_angular<u4x2_t>, nk_angular_u4_neonsdot);
    check("sqeuclidean_u4_neonsdot", test_sqeuclidean<u4x2_t>, nk_sqeuclidean_u4_neonsdot);
    check("euclidean_u4_neonsdot", test_euclidean<u4x2_t>, nk_euclidean_u4_neonsdot);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONFP8
    check("angular_e4m3_neonfp8", test_angular<e4m3_t>, nk_angular_e4m3_neonfp8);
    check("sqeuclidean_e4m3_neonfp8", test_sqeuclidean<e4m3_t>, nk_sqeuclidean_e4m3_neonfp8);
    check("euclidean_e4m3_neonfp8", test_euclidean<e4m3_t>, nk_euclidean_e4m3_neonfp8);
    check("angular_e5m2_neonfp8", test_angular<e5m2_t>, nk_angular_e5m2_neonfp8);
    check("sqeuclidean_e5m2_neonfp8", test_sqeuclidean<e5m2_t>, nk_sqeuclidean_e5m2_neonfp8);
    check("euclidean_e5m2_neonfp8", test_euclidean<e5m2_t>, nk_euclidean_e5m2_neonfp8);
    check("angular_e2m3_neonfp8", test_angular<e2m3_t>, nk_angular_e2m3_neonfp8);
    check("sqeuclidean_e2m3_neonfp8", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3_neonfp8);
    check("euclidean_e2m3_neonfp8", test_euclidean<e2m3_t>, nk_euclidean_e2m3_neonfp8);
    check("angular_e3m2_neonfp8", test_angular<e3m2_t>, nk_angular_e3m2_neonfp8);
    check("sqeuclidean_e3m2_neonfp8", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2_neonfp8);
    check("euclidean_e3m2_neonfp8", test_euclidean<e3m2_t>, nk_euclidean_e3m2_neonfp8);
#endif // NK_TARGET_NEONFP8

#if NK_TARGET_SVE
    check("angular_f64_sve", test_angular<f64_t>, nk_angular_f64_sve);
    check("sqeuclidean_f64_sve", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_sve);
    check("euclidean_f64_sve", test_euclidean<f64_t>, nk_euclidean_f64_sve);
    check("angular_f32_sve", test_angular<f32_t>, nk_angular_f32_sve);
    check("sqeuclidean_f32_sve", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_sve);
    check("euclidean_f32_sve", test_euclidean<f32_t>, nk_euclidean_f32_sve);
#endif // NK_TARGET_SVE

#if NK_TARGET_SVEHALF
    check("angular_f16_svehalf", test_angular<f16_t>, nk_angular_f16_svehalf);
    check("sqeuclidean_f16_svehalf", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_svehalf);
    check("euclidean_f16_svehalf", test_euclidean<f16_t>, nk_euclidean_f16_svehalf);
#endif // NK_TARGET_SVEHALF

#if NK_TARGET_SVEBFDOT
    check("angular_bf16_svebfdot", test_angular<bf16_t>, nk_angular_bf16_svebfdot);
    check("sqeuclidean_bf16_svebfdot", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_svebfdot);
    check("euclidean_bf16_svebfdot", test_euclidean<bf16_t>, nk_euclidean_bf16_svebfdot);
#endif // NK_TARGET_SVEBFDOT

#if NK_TARGET_SVESDOT
    check("angular_i8_svesdot", test_angular<i8_t>, nk_angular_i8_svesdot);
    check("sqeuclidean_i8_svesdot", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_svesdot);
    check("euclidean_i8_svesdot", test_euclidean<i8_t>, nk_euclidean_i8_svesdot);
    check("angular_u8_svesdot", test_angular<u8_t>, nk_angular_u8_svesdot);
    check("sqeuclidean_u8_svesdot", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_svesdot);
    check("euclidean_u8_svesdot", test_euclidean<u8_t>, nk_euclidean_u8_svesdot);
#endif // NK_TARGET_SVESDOT

#if NK_TARGET_HASWELL
    check("angular_f64_haswell", test_angular<f64_t>, nk_angular_f64_haswell);
    check("sqeuclidean_f64_haswell", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_haswell);
    check("euclidean_f64_haswell", test_euclidean<f64_t>, nk_euclidean_f64_haswell);
    check("angular_f32_haswell", test_angular<f32_t>, nk_angular_f32_haswell);
    check("sqeuclidean_f32_haswell", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_haswell);
    check("euclidean_f32_haswell", test_euclidean<f32_t>, nk_euclidean_f32_haswell);
    check("angular_bf16_haswell", test_angular<bf16_t>, nk_angular_bf16_haswell);
    check("sqeuclidean_bf16_haswell", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_haswell);
    check("euclidean_bf16_haswell", test_euclidean<bf16_t>, nk_euclidean_bf16_haswell);
    check("angular_f16_haswell", test_angular<f16_t>, nk_angular_f16_haswell);
    check("sqeuclidean_f16_haswell", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_haswell);
    check("euclidean_f16_haswell", test_euclidean<f16_t>, nk_euclidean_f16_haswell);
    check("angular_e5m2_haswell", test_angular<e5m2_t>, nk_angular_e5m2_haswell);
    check("sqeuclidean_e5m2_haswell", test_sqeuclidean<e5m2_t>, nk_sqeuclidean_e5m2_haswell);
    check("euclidean_e5m2_haswell", test_euclidean<e5m2_t>, nk_euclidean_e5m2_haswell);
    check("angular_e4m3_haswell", test_angular<e4m3_t>, nk_angular_e4m3_haswell);
    check("sqeuclidean_e4m3_haswell", test_sqeuclidean<e4m3_t>, nk_sqeuclidean_e4m3_haswell);
    check("euclidean_e4m3_haswell", test_euclidean<e4m3_t>, nk_euclidean_e4m3_haswell);
    check("angular_e3m2_haswell", test_angular<e3m2_t>, nk_angular_e3m2_haswell);
    check("sqeuclidean_e3m2_haswell", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2_haswell);
    check("euclidean_e3m2_haswell", test_euclidean<e3m2_t>, nk_euclidean_e3m2_haswell);
    check("angular_e2m3_haswell", test_angular<e2m3_t>, nk_angular_e2m3_haswell);
    check("sqeuclidean_e2m3_haswell", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3_haswell);
    check("euclidean_e2m3_haswell", test_euclidean<e2m3_t>, nk_euclidean_e2m3_haswell);
    check("angular_i8_haswell", test_angular<i8_t>, nk_angular_i8_haswell);
    check("sqeuclidean_i8_haswell", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_haswell);
    check("euclidean_i8_haswell", test_euclidean<i8_t>, nk_euclidean_i8_haswell);
    check("angular_u8_haswell", test_angular<u8_t>, nk_angular_u8_haswell);
    check("sqeuclidean_u8_haswell", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_haswell);
    check("euclidean_u8_haswell", test_euclidean<u8_t>, nk_euclidean_u8_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check("angular_f64_skylake", test_angular<f64_t>, nk_angular_f64_skylake);
    check("sqeuclidean_f64_skylake", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_skylake);
    check("euclidean_f64_skylake", test_euclidean<f64_t>, nk_euclidean_f64_skylake);
    check("angular_f32_skylake", test_angular<f32_t>, nk_angular_f32_skylake);
    check("sqeuclidean_f32_skylake", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_skylake);
    check("euclidean_f32_skylake", test_euclidean<f32_t>, nk_euclidean_f32_skylake);
    check("angular_f16_skylake", test_angular<f16_t>, nk_angular_f16_skylake);
    check("sqeuclidean_f16_skylake", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_skylake);
    check("euclidean_f16_skylake", test_euclidean<f16_t>, nk_euclidean_f16_skylake);
    check("angular_e5m2_skylake", test_angular<e5m2_t>, nk_angular_e5m2_skylake);
    check("sqeuclidean_e5m2_skylake", test_sqeuclidean<e5m2_t>, nk_sqeuclidean_e5m2_skylake);
    check("euclidean_e5m2_skylake", test_euclidean<e5m2_t>, nk_euclidean_e5m2_skylake);
    check("angular_e4m3_skylake", test_angular<e4m3_t>, nk_angular_e4m3_skylake);
    check("sqeuclidean_e4m3_skylake", test_sqeuclidean<e4m3_t>, nk_sqeuclidean_e4m3_skylake);
    check("euclidean_e4m3_skylake", test_euclidean<e4m3_t>, nk_euclidean_e4m3_skylake);
    check("angular_e3m2_skylake", test_angular<e3m2_t>, nk_angular_e3m2_skylake);
    check("sqeuclidean_e3m2_skylake", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2_skylake);
    check("euclidean_e3m2_skylake", test_euclidean<e3m2_t>, nk_euclidean_e3m2_skylake);
    check("angular_e2m3_skylake", test_angular<e2m3_t>, nk_angular_e2m3_skylake);
    check("sqeuclidean_e2m3_skylake", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3_skylake);
    check("euclidean_e2m3_skylake", test_euclidean<e2m3_t>, nk_euclidean_e2m3_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
    check("angular_i8_icelake", test_angular<i8_t>, nk_angular_i8_icelake);
    check("sqeuclidean_i8_icelake", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_icelake);
    check("euclidean_i8_icelake", test_euclidean<i8_t>, nk_euclidean_i8_icelake);
    check("angular_u8_icelake", test_angular<u8_t>, nk_angular_u8_icelake);
    check("sqeuclidean_u8_icelake", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_icelake);
    check("euclidean_u8_icelake", test_euclidean<u8_t>, nk_euclidean_u8_icelake);
    check("angular_i4_icelake", test_angular<i4x2_t>, nk_angular_i4_icelake);
    check("sqeuclidean_i4_icelake", test_sqeuclidean<i4x2_t>, nk_sqeuclidean_i4_icelake);
    check("euclidean_i4_icelake", test_euclidean<i4x2_t>, nk_euclidean_i4_icelake);
    check("angular_u4_icelake", test_angular<u4x2_t>, nk_angular_u4_icelake);
    check("sqeuclidean_u4_icelake", test_sqeuclidean<u4x2_t>, nk_sqeuclidean_u4_icelake);
    check("euclidean_u4_icelake", test_euclidean<u4x2_t>, nk_euclidean_u4_icelake);
    check("angular_e4m3_icelake", test_angular<e4m3_t>, nk_angular_e4m3_icelake);
    check("sqeuclidean_e4m3_icelake", test_sqeuclidean<e4m3_t>, nk_sqeuclidean_e4m3_icelake);
    check("euclidean_e4m3_icelake", test_euclidean<e4m3_t>, nk_euclidean_e4m3_icelake);
    check("angular_e2m3_icelake", test_angular<e2m3_t>, nk_angular_e2m3_icelake);
    check("sqeuclidean_e2m3_icelake", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3_icelake);
    check("euclidean_e2m3_icelake", test_euclidean<e2m3_t>, nk_euclidean_e2m3_icelake);
    check("angular_e3m2_icelake", test_angular<e3m2_t>, nk_angular_e3m2_icelake);
    check("sqeuclidean_e3m2_icelake", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2_icelake);
    check("euclidean_e3m2_icelake", test_euclidean<e3m2_t>, nk_euclidean_e3m2_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_GENOA
    check("angular_bf16_genoa", test_angular<bf16_t>, nk_angular_bf16_genoa);
    check("sqeuclidean_bf16_genoa", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_genoa);
    check("euclidean_bf16_genoa", test_euclidean<bf16_t>, nk_euclidean_bf16_genoa);
#endif // NK_TARGET_GENOA

#if NK_TARGET_DIAMOND
    check("angular_f16_diamond", test_angular<f16_t>, nk_angular_f16_diamond);
    check("sqeuclidean_f16_diamond", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_diamond);
    check("euclidean_f16_diamond", test_euclidean<f16_t>, nk_euclidean_f16_diamond);
    check("angular_e4m3_diamond", test_angular<e4m3_t>, nk_angular_e4m3_diamond);
    check("sqeuclidean_e4m3_diamond", test_sqeuclidean<e4m3_t>, nk_sqeuclidean_e4m3_diamond);
    check("euclidean_e4m3_diamond", test_euclidean<e4m3_t>, nk_euclidean_e4m3_diamond);
    check("angular_e5m2_diamond", test_angular<e5m2_t>, nk_angular_e5m2_diamond);
    check("sqeuclidean_e5m2_diamond", test_sqeuclidean<e5m2_t>, nk_sqeuclidean_e5m2_diamond);
    check("euclidean_e5m2_diamond", test_euclidean<e5m2_t>, nk_euclidean_e5m2_diamond);
#endif // NK_TARGET_DIAMOND

#if NK_TARGET_ALDER
    check("angular_e3m2_alder", test_angular<e3m2_t>, nk_angular_e3m2_alder);
    check("sqeuclidean_e3m2_alder", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2_alder);
    check("euclidean_e3m2_alder", test_euclidean<e3m2_t>, nk_euclidean_e3m2_alder);
    check("angular_e2m3_alder", test_angular<e2m3_t>, nk_angular_e2m3_alder);
    check("sqeuclidean_e2m3_alder", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3_alder);
    check("euclidean_e2m3_alder", test_euclidean<e2m3_t>, nk_euclidean_e2m3_alder);
    check("angular_i8_alder", test_angular<i8_t>, nk_angular_i8_alder);
    check("sqeuclidean_i8_alder", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_alder);
    check("euclidean_i8_alder", test_euclidean<i8_t>, nk_euclidean_i8_alder);
    check("angular_u8_alder", test_angular<u8_t>, nk_angular_u8_alder);
    check("sqeuclidean_u8_alder", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_alder);
    check("euclidean_u8_alder", test_euclidean<u8_t>, nk_euclidean_u8_alder);
#endif // NK_TARGET_ALDER

#if NK_TARGET_SIERRA
    check("angular_e2m3_sierra", test_angular<e2m3_t>, nk_angular_e2m3_sierra);
    check("sqeuclidean_e2m3_sierra", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3_sierra);
    check("euclidean_e2m3_sierra", test_euclidean<e2m3_t>, nk_euclidean_e2m3_sierra);
    check("angular_e3m2_sierra", test_angular<e3m2_t>, nk_angular_e3m2_sierra);
    check("sqeuclidean_e3m2_sierra", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2_sierra);
    check("euclidean_e3m2_sierra", test_euclidean<e3m2_t>, nk_euclidean_e3m2_sierra);
    check("angular_i8_sierra", test_angular<i8_t>, nk_angular_i8_sierra);
    check("sqeuclidean_i8_sierra", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_sierra);
    check("euclidean_i8_sierra", test_euclidean<i8_t>, nk_euclidean_i8_sierra);
    check("angular_u8_sierra", test_angular<u8_t>, nk_angular_u8_sierra);
    check("sqeuclidean_u8_sierra", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_sierra);
    check("euclidean_u8_sierra", test_euclidean<u8_t>, nk_euclidean_u8_sierra);
#endif // NK_TARGET_SIERRA

#if NK_TARGET_RVV
    check("angular_f64_rvv", test_angular<f64_t>, nk_angular_f64_rvv);
    check("sqeuclidean_f64_rvv", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_rvv);
    check("euclidean_f64_rvv", test_euclidean<f64_t>, nk_euclidean_f64_rvv);
    check("angular_f32_rvv", test_angular<f32_t>, nk_angular_f32_rvv);
    check("sqeuclidean_f32_rvv", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_rvv);
    check("euclidean_f32_rvv", test_euclidean<f32_t>, nk_euclidean_f32_rvv);
    check("angular_bf16_rvv", test_angular<bf16_t>, nk_angular_bf16_rvv);
    check("sqeuclidean_bf16_rvv", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_rvv);
    check("euclidean_bf16_rvv", test_euclidean<bf16_t>, nk_euclidean_bf16_rvv);
    check("angular_f16_rvv", test_angular<f16_t>, nk_angular_f16_rvv);
    check("sqeuclidean_f16_rvv", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_rvv);
    check("euclidean_f16_rvv", test_euclidean<f16_t>, nk_euclidean_f16_rvv);
    check("angular_e5m2_rvv", test_angular<e5m2_t>, nk_angular_e5m2_rvv);
    check("sqeuclidean_e5m2_rvv", test_sqeuclidean<e5m2_t>, nk_sqeuclidean_e5m2_rvv);
    check("euclidean_e5m2_rvv", test_euclidean<e5m2_t>, nk_euclidean_e5m2_rvv);
    check("angular_e4m3_rvv", test_angular<e4m3_t>, nk_angular_e4m3_rvv);
    check("sqeuclidean_e4m3_rvv", test_sqeuclidean<e4m3_t>, nk_sqeuclidean_e4m3_rvv);
    check("euclidean_e4m3_rvv", test_euclidean<e4m3_t>, nk_euclidean_e4m3_rvv);
    check("angular_i8_rvv", test_angular<i8_t>, nk_angular_i8_rvv);
    check("sqeuclidean_i8_rvv", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_rvv);
    check("euclidean_i8_rvv", test_euclidean<i8_t>, nk_euclidean_i8_rvv);
    check("angular_u8_rvv", test_angular<u8_t>, nk_angular_u8_rvv);
    check("sqeuclidean_u8_rvv", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_rvv);
    check("euclidean_u8_rvv", test_euclidean<u8_t>, nk_euclidean_u8_rvv);
    check("angular_i4_rvv", test_angular<i4x2_t>, nk_angular_i4_rvv);
    check("sqeuclidean_i4_rvv", test_sqeuclidean<i4x2_t>, nk_sqeuclidean_i4_rvv);
    check("euclidean_i4_rvv", test_euclidean<i4x2_t>, nk_euclidean_i4_rvv);
    check("angular_u4_rvv", test_angular<u4x2_t>, nk_angular_u4_rvv);
    check("sqeuclidean_u4_rvv", test_sqeuclidean<u4x2_t>, nk_sqeuclidean_u4_rvv);
    check("euclidean_u4_rvv", test_euclidean<u4x2_t>, nk_euclidean_u4_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    check("sqeuclidean_f32_v128relaxed", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_v128relaxed);
    check("sqeuclidean_f64_v128relaxed", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_v128relaxed);
    check("sqeuclidean_f16_v128relaxed", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_v128relaxed);
    check("sqeuclidean_bf16_v128relaxed", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_v128relaxed);
    check("euclidean_f32_v128relaxed", test_euclidean<f32_t>, nk_euclidean_f32_v128relaxed);
    check("euclidean_f64_v128relaxed", test_euclidean<f64_t>, nk_euclidean_f64_v128relaxed);
    check("euclidean_f16_v128relaxed", test_euclidean<f16_t>, nk_euclidean_f16_v128relaxed);
    check("euclidean_bf16_v128relaxed", test_euclidean<bf16_t>, nk_euclidean_bf16_v128relaxed);
    check("angular_f32_v128relaxed", test_angular<f32_t>, nk_angular_f32_v128relaxed);
    check("angular_f64_v128relaxed", test_angular<f64_t>, nk_angular_f64_v128relaxed);
    check("angular_f16_v128relaxed", test_angular<f16_t>, nk_angular_f16_v128relaxed);
    check("angular_bf16_v128relaxed", test_angular<bf16_t>, nk_angular_bf16_v128relaxed);
    check("sqeuclidean_u8_v128relaxed", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_v128relaxed);
    check("euclidean_u8_v128relaxed", test_euclidean<u8_t>, nk_euclidean_u8_v128relaxed);
    check("angular_u8_v128relaxed", test_angular<u8_t>, nk_angular_u8_v128relaxed);
    check("sqeuclidean_i8_v128relaxed", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_v128relaxed);
    check("euclidean_i8_v128relaxed", test_euclidean<i8_t>, nk_euclidean_i8_v128relaxed);
    check("angular_i8_v128relaxed", test_angular<i8_t>, nk_angular_i8_v128relaxed);
    check("sqeuclidean_e4m3_v128relaxed", test_sqeuclidean<e4m3_t>, nk_sqeuclidean_e4m3_v128relaxed);
    check("euclidean_e4m3_v128relaxed", test_euclidean<e4m3_t>, nk_euclidean_e4m3_v128relaxed);
    check("angular_e4m3_v128relaxed", test_angular<e4m3_t>, nk_angular_e4m3_v128relaxed);
    check("sqeuclidean_e5m2_v128relaxed", test_sqeuclidean<e5m2_t>, nk_sqeuclidean_e5m2_v128relaxed);
    check("euclidean_e5m2_v128relaxed", test_euclidean<e5m2_t>, nk_euclidean_e5m2_v128relaxed);
    check("angular_e5m2_v128relaxed", test_angular<e5m2_t>, nk_angular_e5m2_v128relaxed);
    check("sqeuclidean_e2m3_v128relaxed", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3_v128relaxed);
    check("euclidean_e2m3_v128relaxed", test_euclidean<e2m3_t>, nk_euclidean_e2m3_v128relaxed);
    check("angular_e2m3_v128relaxed", test_angular<e2m3_t>, nk_angular_e2m3_v128relaxed);
    check("sqeuclidean_e3m2_v128relaxed", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2_v128relaxed);
    check("euclidean_e3m2_v128relaxed", test_euclidean<e3m2_t>, nk_euclidean_e3m2_v128relaxed);
    check("angular_e3m2_v128relaxed", test_angular<e3m2_t>, nk_angular_e3m2_v128relaxed);
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_RVVHALF
    check("sqeuclidean_f16_rvvhalf", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_rvvhalf);
    check("euclidean_f16_rvvhalf", test_euclidean<f16_t>, nk_euclidean_f16_rvvhalf);
    check("angular_f16_rvvhalf", test_angular<f16_t>, nk_angular_f16_rvvhalf);
#endif // NK_TARGET_RVVHALF

#if NK_TARGET_RVVBF16
    check("sqeuclidean_bf16_rvvbf16", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_rvvbf16);
    check("euclidean_bf16_rvvbf16", test_euclidean<bf16_t>, nk_euclidean_bf16_rvvbf16);
    check("angular_bf16_rvvbf16", test_angular<bf16_t>, nk_angular_bf16_rvvbf16);
#endif // NK_TARGET_RVVBF16

#if NK_TARGET_LOONGSONASX
    check("angular_f64_loongsonasx", test_angular<f64_t>, nk_angular_f64_loongsonasx);
    check("sqeuclidean_f64_loongsonasx", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_loongsonasx);
    check("euclidean_f64_loongsonasx", test_euclidean<f64_t>, nk_euclidean_f64_loongsonasx);
    check("angular_f32_loongsonasx", test_angular<f32_t>, nk_angular_f32_loongsonasx);
    check("sqeuclidean_f32_loongsonasx", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_loongsonasx);
    check("euclidean_f32_loongsonasx", test_euclidean<f32_t>, nk_euclidean_f32_loongsonasx);
    check("angular_bf16_loongsonasx", test_angular<bf16_t>, nk_angular_bf16_loongsonasx);
    check("sqeuclidean_bf16_loongsonasx", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_loongsonasx);
    check("euclidean_bf16_loongsonasx", test_euclidean<bf16_t>, nk_euclidean_bf16_loongsonasx);
    check("angular_i8_loongsonasx", test_angular<i8_t>, nk_angular_i8_loongsonasx);
    check("sqeuclidean_i8_loongsonasx", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_loongsonasx);
    check("euclidean_i8_loongsonasx", test_euclidean<i8_t>, nk_euclidean_i8_loongsonasx);
    check("angular_u8_loongsonasx", test_angular<u8_t>, nk_angular_u8_loongsonasx);
    check("sqeuclidean_u8_loongsonasx", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_loongsonasx);
    check("euclidean_u8_loongsonasx", test_euclidean<u8_t>, nk_euclidean_u8_loongsonasx);
#endif // NK_TARGET_LOONGSONASX

#if NK_TARGET_POWERVSX
    check("sqeuclidean_f32_powervsx", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_powervsx);
    check("euclidean_f32_powervsx", test_euclidean<f32_t>, nk_euclidean_f32_powervsx);
    check("angular_f32_powervsx", test_angular<f32_t>, nk_angular_f32_powervsx);
    check("sqeuclidean_f64_powervsx", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_powervsx);
    check("euclidean_f64_powervsx", test_euclidean<f64_t>, nk_euclidean_f64_powervsx);
    check("angular_f64_powervsx", test_angular<f64_t>, nk_angular_f64_powervsx);
    check("sqeuclidean_f16_powervsx", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_powervsx);
    check("euclidean_f16_powervsx", test_euclidean<f16_t>, nk_euclidean_f16_powervsx);
    check("angular_f16_powervsx", test_angular<f16_t>, nk_angular_f16_powervsx);
    check("sqeuclidean_bf16_powervsx", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_powervsx);
    check("euclidean_bf16_powervsx", test_euclidean<bf16_t>, nk_euclidean_bf16_powervsx);
    check("angular_bf16_powervsx", test_angular<bf16_t>, nk_angular_bf16_powervsx);
    check("sqeuclidean_i8_powervsx", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_powervsx);
    check("euclidean_i8_powervsx", test_euclidean<i8_t>, nk_euclidean_i8_powervsx);
    check("angular_i8_powervsx", test_angular<i8_t>, nk_angular_i8_powervsx);
    check("sqeuclidean_u8_powervsx", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_powervsx);
    check("euclidean_u8_powervsx", test_euclidean<u8_t>, nk_euclidean_u8_powervsx);
    check("angular_u8_powervsx", test_angular<u8_t>, nk_angular_u8_powervsx);
#endif // NK_TARGET_POWERVSX

    // Serial always runs - baseline test
    check("sqeuclidean_f32_serial", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_serial);
    check("sqeuclidean_f64_serial", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_serial);
    check("sqeuclidean_f16_serial", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_serial);
    check("sqeuclidean_bf16_serial", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_serial);
    check("sqeuclidean_e2m3_serial", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3_serial);
    check("sqeuclidean_e3m2_serial", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2_serial);
    check("euclidean_f32_serial", test_euclidean<f32_t>, nk_euclidean_f32_serial);
    check("euclidean_f64_serial", test_euclidean<f64_t>, nk_euclidean_f64_serial);
    check("euclidean_f16_serial", test_euclidean<f16_t>, nk_euclidean_f16_serial);
    check("euclidean_bf16_serial", test_euclidean<bf16_t>, nk_euclidean_bf16_serial);
    check("euclidean_e2m3_serial", test_euclidean<e2m3_t>, nk_euclidean_e2m3_serial);
    check("euclidean_e3m2_serial", test_euclidean<e3m2_t>, nk_euclidean_e3m2_serial);
    check("angular_f32_serial", test_angular<f32_t>, nk_angular_f32_serial);
    check("angular_f64_serial", test_angular<f64_t>, nk_angular_f64_serial);
    check("angular_f16_serial", test_angular<f16_t>, nk_angular_f16_serial);
    check("angular_bf16_serial", test_angular<bf16_t>, nk_angular_bf16_serial);
    check("angular_e2m3_serial", test_angular<e2m3_t>, nk_angular_e2m3_serial);
    check("angular_e3m2_serial", test_angular<e3m2_t>, nk_angular_e3m2_serial);
    check("sqeuclidean_i4_serial", test_sqeuclidean<i4x2_t>, nk_sqeuclidean_i4_serial);
    check("sqeuclidean_u4_serial", test_sqeuclidean<u4x2_t>, nk_sqeuclidean_u4_serial);
    check("euclidean_i4_serial", test_euclidean<i4x2_t>, nk_euclidean_i4_serial);
    check("euclidean_u4_serial", test_euclidean<u4x2_t>, nk_euclidean_u4_serial);
    check("angular_i4_serial", test_angular<i4x2_t>, nk_angular_i4_serial);
    check("angular_u4_serial", test_angular<u4x2_t>, nk_angular_u4_serial);
    check("angular_e4m3_serial", test_angular<e4m3_t>, nk_angular_e4m3_serial);
    check("sqeuclidean_e4m3_serial", test_sqeuclidean<e4m3_t>, nk_sqeuclidean_e4m3_serial);
    check("euclidean_e4m3_serial", test_euclidean<e4m3_t>, nk_euclidean_e4m3_serial);
    check("angular_e5m2_serial", test_angular<e5m2_t>, nk_angular_e5m2_serial);
    check("sqeuclidean_e5m2_serial", test_sqeuclidean<e5m2_t>, nk_sqeuclidean_e5m2_serial);
    check("euclidean_e5m2_serial", test_euclidean<e5m2_t>, nk_euclidean_e5m2_serial);
    check("angular_i8_serial", test_angular<i8_t>, nk_angular_i8_serial);
    check("sqeuclidean_i8_serial", test_sqeuclidean<i8_t>, nk_sqeuclidean_i8_serial);
    check("euclidean_i8_serial", test_euclidean<i8_t>, nk_euclidean_i8_serial);
    check("angular_u8_serial", test_angular<u8_t>, nk_angular_u8_serial);
    check("sqeuclidean_u8_serial", test_sqeuclidean<u8_t>, nk_sqeuclidean_u8_serial);
    check("euclidean_u8_serial", test_euclidean<u8_t>, nk_euclidean_u8_serial);

#endif // NK_DYNAMIC_DISPATCH
}
