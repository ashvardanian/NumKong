/**
 *  @brief Dot product precision tests.
 *  @file test/test_dot.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/dot.hpp" // `nk::dot`

/**
 *  @brief Unified dot product test for all types: float, integer, and complex.
 *  Works with f32_t, f64_t, f16_t, bf16_t, e4m3_t, e5m2_t, i8_t, u8_t, f32c_t, f64c_t.
 */
template <typename scalar_type_>
error_stats_t test_dot(typename scalar_type_::dot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::dot_result_t;
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
        nk::dot<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Conjugate dot product test for complex types (vdot = conj(a) * b).
 */
template <typename scalar_type_>
error_stats_t test_vdot(typename scalar_type_::vdot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::vdot_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::mixed_precision_reduction_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        reference_t reference;
        nk::vdot<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                       &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_dot() {
    error_stats_section_t check("Dot Products");

#if NK_RUNTIME_DISPATCH
    check("dot_f32", test_dot<f32_t>, nk_dot_f32);
    check("dot_f64", test_dot<f64_t>, nk_dot_f64);
    check("dot_f16", test_dot<f16_t>, nk_dot_f16);
    check("dot_bf16", test_dot<bf16_t>, nk_dot_bf16);
    check("dot_e4m3", test_dot<e4m3_t>, nk_dot_e4m3);
    check("dot_e5m2", test_dot<e5m2_t>, nk_dot_e5m2);
    check("dot_e2m3", test_dot<e2m3_t>, nk_dot_e2m3);
    check("dot_e3m2", test_dot<e3m2_t>, nk_dot_e3m2);
    check("dot_i8", test_dot<i8_t>, nk_dot_i8);
    check("dot_u8", test_dot<u8_t>, nk_dot_u8);
    check("dot_i4", test_dot<i4x2_t>, nk_dot_i4);
    check("dot_u4", test_dot<u4x2_t>, nk_dot_u4);
    check("dot_u1", test_dot<u1x8_t>, nk_dot_u1);
    check("dot_f32c", test_dot<f32c_t>, nk_dot_f32c);
    check("vdot_f32c", test_vdot<f32c_t>, nk_vdot_f32c);
    check("dot_f64c", test_dot<f64c_t>, nk_dot_f64c);
    check("vdot_f64c", test_vdot<f64c_t>, nk_vdot_f64c);
    check("dot_f16c", test_dot<f16c_t>, nk_dot_f16c);
    check("vdot_f16c", test_vdot<f16c_t>, nk_vdot_f16c);
    check("dot_bf16c", test_dot<bf16c_t>, nk_dot_bf16c);
    check("vdot_bf16c", test_vdot<bf16c_t>, nk_vdot_bf16c);
#else

#if NK_TARGET_NEON
    check("dot_f32_neon", test_dot<f32_t>, nk_dot_f32_neon);
    check("dot_f64_neon", test_dot<f64_t>, nk_dot_f64_neon);
    check("dot_f32c_neon", test_dot<f32c_t>, nk_dot_f32c_neon);
    check("vdot_f32c_neon", test_vdot<f32c_t>, nk_vdot_f32c_neon);
    check("dot_f64c_neon", test_dot<f64c_t>, nk_dot_f64c_neon);
    check("vdot_f64c_neon", test_vdot<f64c_t>, nk_vdot_f64c_neon);
    check("dot_bf16_neon", test_dot<bf16_t>, nk_dot_bf16_neon);
    check("dot_e4m3_neon", test_dot<e4m3_t>, nk_dot_e4m3_neon);
    check("dot_e5m2_neon", test_dot<e5m2_t>, nk_dot_e5m2_neon);
    check("dot_e2m3_neon", test_dot<e2m3_t>, nk_dot_e2m3_neon);
    check("dot_e3m2_neon", test_dot<e3m2_t>, nk_dot_e3m2_neon);
    check("dot_f16_neon", test_dot<f16_t>, nk_dot_f16_neon);
    check("dot_f16c_neon", test_dot<f16c_t>, nk_dot_f16c_neon);
    check("vdot_f16c_neon", test_vdot<f16c_t>, nk_vdot_f16c_neon);
    check("dot_u1_neon", test_dot<u1x8_t>, nk_dot_u1_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONSDOT
    check("dot_i8_neonsdot", test_dot<i8_t>, nk_dot_i8_neonsdot);
    check("dot_u8_neonsdot", test_dot<u8_t>, nk_dot_u8_neonsdot);
    check("dot_i4_neonsdot", test_dot<i4x2_t>, nk_dot_i4_neonsdot);
    check("dot_u4_neonsdot", test_dot<u4x2_t>, nk_dot_u4_neonsdot);
    check("dot_e2m3_neonsdot", test_dot<e2m3_t>, nk_dot_e2m3_neonsdot);
    check("dot_e3m2_neonsdot", test_dot<e3m2_t>, nk_dot_e3m2_neonsdot);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONFHM
    check("dot_f16_neonfhm", test_dot<f16_t>, nk_dot_f16_neonfhm);
    check("dot_f16c_neonfhm", test_dot<f16c_t>, nk_dot_f16c_neonfhm);
    check("vdot_f16c_neonfhm", test_vdot<f16c_t>, nk_vdot_f16c_neonfhm);
    check("dot_e4m3_neonfhm", test_dot<e4m3_t>, nk_dot_e4m3_neonfhm);
    check("dot_e5m2_neonfhm", test_dot<e5m2_t>, nk_dot_e5m2_neonfhm);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_NEONBFDOT
    check("dot_bf16_neonbfdot", test_dot<bf16_t>, nk_dot_bf16_neonbfdot);
    check("dot_bf16c_neonbfdot", test_dot<bf16c_t>, nk_dot_bf16c_neonbfdot);
    check("vdot_bf16c_neonbfdot", test_vdot<bf16c_t>, nk_vdot_bf16c_neonbfdot);
    check("dot_e4m3_neonbfdot", test_dot<e4m3_t>, nk_dot_e4m3_neonbfdot);
    check("dot_e5m2_neonbfdot", test_dot<e5m2_t>, nk_dot_e5m2_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONFP8
    check("dot_e4m3_neonfp8", test_dot<e4m3_t>, nk_dot_e4m3_neonfp8);
    check("dot_e5m2_neonfp8", test_dot<e5m2_t>, nk_dot_e5m2_neonfp8);
    check("dot_e2m3_neonfp8", test_dot<e2m3_t>, nk_dot_e2m3_neonfp8);
    check("dot_e3m2_neonfp8", test_dot<e3m2_t>, nk_dot_e3m2_neonfp8);
#endif // NK_TARGET_NEONFP8

#if NK_TARGET_SVE
    check("dot_f32_sve", test_dot<f32_t>, nk_dot_f32_sve);
    check("dot_f64_sve", test_dot<f64_t>, nk_dot_f64_sve);
    check("dot_f32c_sve", test_dot<f32c_t>, nk_dot_f32c_sve);
    check("vdot_f32c_sve", test_vdot<f32c_t>, nk_vdot_f32c_sve);
    check("dot_f64c_sve", test_dot<f64c_t>, nk_dot_f64c_sve);
    check("vdot_f64c_sve", test_vdot<f64c_t>, nk_vdot_f64c_sve);
#endif // NK_TARGET_SVE

#if NK_TARGET_SVEHALF
    check("dot_f16_svehalf", test_dot<f16_t>, nk_dot_f16_svehalf);
    check("dot_f16c_svehalf", test_dot<f16c_t>, nk_dot_f16c_svehalf);
    check("vdot_f16c_svehalf", test_vdot<f16c_t>, nk_vdot_f16c_svehalf);
#endif // NK_TARGET_SVEHALF

#if NK_TARGET_SVEBFDOT
    check("dot_bf16_svebfdot", test_dot<bf16_t>, nk_dot_bf16_svebfdot);
#endif // NK_TARGET_SVEBFDOT

#if NK_TARGET_SVESDOT
    check("dot_i8_svesdot", test_dot<i8_t>, nk_dot_i8_svesdot);
    check("dot_u8_svesdot", test_dot<u8_t>, nk_dot_u8_svesdot);
#endif // NK_TARGET_SVESDOT

#if NK_TARGET_HASWELL
    check("dot_f64c_haswell", test_dot<f64c_t>, nk_dot_f64c_haswell);
    check("vdot_f64c_haswell", test_vdot<f64c_t>, nk_vdot_f64c_haswell);
    check("dot_f32c_haswell", test_dot<f32c_t>, nk_dot_f32c_haswell);
    check("vdot_f32c_haswell", test_vdot<f32c_t>, nk_vdot_f32c_haswell);
    check("dot_bf16c_haswell", test_dot<bf16c_t>, nk_dot_bf16c_haswell);
    check("vdot_bf16c_haswell", test_vdot<bf16c_t>, nk_vdot_bf16c_haswell);
    check("dot_f16c_haswell", test_dot<f16c_t>, nk_dot_f16c_haswell);
    check("vdot_f16c_haswell", test_vdot<f16c_t>, nk_vdot_f16c_haswell);
    check("dot_f64_haswell", test_dot<f64_t>, nk_dot_f64_haswell);
    check("dot_f32_haswell", test_dot<f32_t>, nk_dot_f32_haswell);
    check("dot_bf16_haswell", test_dot<bf16_t>, nk_dot_bf16_haswell);
    check("dot_f16_haswell", test_dot<f16_t>, nk_dot_f16_haswell);
    check("dot_e5m2_haswell", test_dot<e5m2_t>, nk_dot_e5m2_haswell);
    check("dot_e4m3_haswell", test_dot<e4m3_t>, nk_dot_e4m3_haswell);
    check("dot_e3m2_haswell", test_dot<e3m2_t>, nk_dot_e3m2_haswell);
    check("dot_e2m3_haswell", test_dot<e2m3_t>, nk_dot_e2m3_haswell);
    check("dot_i8_haswell", test_dot<i8_t>, nk_dot_i8_haswell);
    check("dot_u8_haswell", test_dot<u8_t>, nk_dot_u8_haswell);
    check("dot_i4_haswell", test_dot<i4x2_t>, nk_dot_i4_haswell);
    check("dot_u4_haswell", test_dot<u4x2_t>, nk_dot_u4_haswell);
    check("dot_u1_haswell", test_dot<u1x8_t>, nk_dot_u1_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check("dot_f32_skylake", test_dot<f32_t>, nk_dot_f32_skylake);
    check("dot_f64_skylake", test_dot<f64_t>, nk_dot_f64_skylake);
    check("dot_f16_skylake", test_dot<f16_t>, nk_dot_f16_skylake);
    check("dot_bf16_skylake", test_dot<bf16_t>, nk_dot_bf16_skylake);
    check("dot_e4m3_skylake", test_dot<e4m3_t>, nk_dot_e4m3_skylake);
    check("dot_e5m2_skylake", test_dot<e5m2_t>, nk_dot_e5m2_skylake);
    check("dot_e2m3_skylake", test_dot<e2m3_t>, nk_dot_e2m3_skylake);
    check("dot_e3m2_skylake", test_dot<e3m2_t>, nk_dot_e3m2_skylake);
    check("dot_i8_skylake", test_dot<i8_t>, nk_dot_i8_skylake);
    check("dot_u8_skylake", test_dot<u8_t>, nk_dot_u8_skylake);
    check("dot_f32c_skylake", test_dot<f32c_t>, nk_dot_f32c_skylake);
    check("vdot_f32c_skylake", test_vdot<f32c_t>, nk_vdot_f32c_skylake);
    check("dot_f64c_skylake", test_dot<f64c_t>, nk_dot_f64c_skylake);
    check("vdot_f64c_skylake", test_vdot<f64c_t>, nk_vdot_f64c_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
    check("dot_i8_icelake", test_dot<i8_t>, nk_dot_i8_icelake);
    check("dot_u8_icelake", test_dot<u8_t>, nk_dot_u8_icelake);
    check("dot_i4_icelake", test_dot<i4x2_t>, nk_dot_i4_icelake);
    check("dot_u4_icelake", test_dot<u4x2_t>, nk_dot_u4_icelake);
    check("dot_e2m3_icelake", test_dot<e2m3_t>, nk_dot_e2m3_icelake);
    check("dot_e3m2_icelake", test_dot<e3m2_t>, nk_dot_e3m2_icelake);
    check("dot_u1_icelake", test_dot<u1x8_t>, nk_dot_u1_icelake);
    check("dot_e4m3_icelake", test_dot<e4m3_t>, nk_dot_e4m3_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_ALDER
    check("dot_i8_alder", test_dot<i8_t>, nk_dot_i8_alder);
    check("dot_u8_alder", test_dot<u8_t>, nk_dot_u8_alder);
    check("dot_e2m3_alder", test_dot<e2m3_t>, nk_dot_e2m3_alder);
#endif // NK_TARGET_ALDER

#if NK_TARGET_SIERRA
    check("dot_i8_sierra", test_dot<i8_t>, nk_dot_i8_sierra);
    check("dot_u8_sierra", test_dot<u8_t>, nk_dot_u8_sierra);
    check("dot_e2m3_sierra", test_dot<e2m3_t>, nk_dot_e2m3_sierra);
#endif // NK_TARGET_SIERRA

#if NK_TARGET_GENOA
    check("dot_bf16_genoa", test_dot<bf16_t>, nk_dot_bf16_genoa);
    check("dot_e5m2_genoa", test_dot<e5m2_t>, nk_dot_e5m2_genoa);
    check("dot_bf16c_genoa", test_dot<bf16c_t>, nk_dot_bf16c_genoa);
    check("vdot_bf16c_genoa", test_vdot<bf16c_t>, nk_vdot_bf16c_genoa);
#endif // NK_TARGET_GENOA

#if NK_TARGET_DIAMOND
    check("dot_f16_diamond", test_dot<f16_t>, nk_dot_f16_diamond);
    check("dot_e4m3_diamond", test_dot<e4m3_t>, nk_dot_e4m3_diamond);
    check("dot_e5m2_diamond", test_dot<e5m2_t>, nk_dot_e5m2_diamond);
#endif // NK_TARGET_DIAMOND

#if NK_TARGET_RVV
    check("dot_f64c_rvv", test_dot<f64c_t>, nk_dot_f64c_rvv);
    check("vdot_f64c_rvv", test_vdot<f64c_t>, nk_vdot_f64c_rvv);
    check("dot_f32c_rvv", test_dot<f32c_t>, nk_dot_f32c_rvv);
    check("vdot_f32c_rvv", test_vdot<f32c_t>, nk_vdot_f32c_rvv);
    check("dot_f64_rvv", test_dot<f64_t>, nk_dot_f64_rvv);
    check("dot_f32_rvv", test_dot<f32_t>, nk_dot_f32_rvv);
    check("dot_bf16_rvv", test_dot<bf16_t>, nk_dot_bf16_rvv);
    check("dot_f16_rvv", test_dot<f16_t>, nk_dot_f16_rvv);
    check("dot_e5m2_rvv", test_dot<e5m2_t>, nk_dot_e5m2_rvv);
    check("dot_e4m3_rvv", test_dot<e4m3_t>, nk_dot_e4m3_rvv);
    check("dot_e3m2_rvv", test_dot<e3m2_t>, nk_dot_e3m2_rvv);
    check("dot_e2m3_rvv", test_dot<e2m3_t>, nk_dot_e2m3_rvv);
    check("dot_i8_rvv", test_dot<i8_t>, nk_dot_i8_rvv);
    check("dot_u8_rvv", test_dot<u8_t>, nk_dot_u8_rvv);
    check("dot_i4_rvv", test_dot<i4x2_t>, nk_dot_i4_rvv);
    check("dot_u4_rvv", test_dot<u4x2_t>, nk_dot_u4_rvv);
    check("dot_u1_rvv", test_dot<u1x8_t>, nk_dot_u1_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    check("dot_f32_v128relaxed", test_dot<f32_t>, nk_dot_f32_v128relaxed);
    check("dot_f64_v128relaxed", test_dot<f64_t>, nk_dot_f64_v128relaxed);
    check("dot_f16_v128relaxed", test_dot<f16_t>, nk_dot_f16_v128relaxed);
    check("dot_bf16_v128relaxed", test_dot<bf16_t>, nk_dot_bf16_v128relaxed);
    check("dot_i8_v128relaxed", test_dot<i8_t>, nk_dot_i8_v128relaxed);
    check("dot_u8_v128relaxed", test_dot<u8_t>, nk_dot_u8_v128relaxed);
    check("dot_e2m3_v128relaxed", test_dot<e2m3_t>, nk_dot_e2m3_v128relaxed);
    check("dot_e3m2_v128relaxed", test_dot<e3m2_t>, nk_dot_e3m2_v128relaxed);
    check("dot_u1_v128relaxed", test_dot<u1x8_t>, nk_dot_u1_v128relaxed);
    check("dot_e4m3_v128relaxed", test_dot<e4m3_t>, nk_dot_e4m3_v128relaxed);
    check("dot_e5m2_v128relaxed", test_dot<e5m2_t>, nk_dot_e5m2_v128relaxed);
    check("dot_i4_v128relaxed", test_dot<i4x2_t>, nk_dot_i4_v128relaxed);
    check("dot_u4_v128relaxed", test_dot<u4x2_t>, nk_dot_u4_v128relaxed);
    check("dot_f32c_v128relaxed", test_dot<f32c_t>, nk_dot_f32c_v128relaxed);
    check("vdot_f32c_v128relaxed", test_vdot<f32c_t>, nk_vdot_f32c_v128relaxed);
    check("dot_f64c_v128relaxed", test_dot<f64c_t>, nk_dot_f64c_v128relaxed);
    check("vdot_f64c_v128relaxed", test_vdot<f64c_t>, nk_vdot_f64c_v128relaxed);
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_RVVHALF
    check("dot_f16_rvvhalf", test_dot<f16_t>, nk_dot_f16_rvvhalf);
    check("dot_e4m3_rvvhalf", test_dot<e4m3_t>, nk_dot_e4m3_rvvhalf);
    check("dot_e5m2_rvvhalf", test_dot<e5m2_t>, nk_dot_e5m2_rvvhalf);
#endif // NK_TARGET_RVVHALF

#if NK_TARGET_RVVBF16
    check("dot_bf16_rvvbf16", test_dot<bf16_t>, nk_dot_bf16_rvvbf16);
    check("dot_e4m3_rvvbf16", test_dot<e4m3_t>, nk_dot_e4m3_rvvbf16);
    check("dot_e5m2_rvvbf16", test_dot<e5m2_t>, nk_dot_e5m2_rvvbf16);
#endif // NK_TARGET_RVVBF16

#if NK_TARGET_RVVBB
    check("dot_u1_rvvbb", test_dot<u1x8_t>, nk_dot_u1_rvvbb);
#endif // NK_TARGET_RVVBB

#if NK_TARGET_LOONGSONASX
    check("dot_f64_loongsonasx", test_dot<f64_t>, nk_dot_f64_loongsonasx);
    check("dot_f32_loongsonasx", test_dot<f32_t>, nk_dot_f32_loongsonasx);
    check("dot_bf16_loongsonasx", test_dot<bf16_t>, nk_dot_bf16_loongsonasx);
    check("dot_i8_loongsonasx", test_dot<i8_t>, nk_dot_i8_loongsonasx);
    check("dot_u8_loongsonasx", test_dot<u8_t>, nk_dot_u8_loongsonasx);
#endif // NK_TARGET_LOONGSONASX

#if NK_TARGET_POWERVSX
    check("dot_f64_powervsx", test_dot<f64_t>, nk_dot_f64_powervsx);
    check("dot_f32_powervsx", test_dot<f32_t>, nk_dot_f32_powervsx);
    check("dot_f16_powervsx", test_dot<f16_t>, nk_dot_f16_powervsx);
    check("dot_bf16_powervsx", test_dot<bf16_t>, nk_dot_bf16_powervsx);
    check("dot_i8_powervsx", test_dot<i8_t>, nk_dot_i8_powervsx);
    check("dot_u8_powervsx", test_dot<u8_t>, nk_dot_u8_powervsx);
    check("dot_u1_powervsx", test_dot<u1x8_t>, nk_dot_u1_powervsx);
#endif // NK_TARGET_POWERVSX

    // Serial always runs - baseline test
    check("dot_f32_serial", test_dot<f32_t>, nk_dot_f32_serial);
    check("dot_f64_serial", test_dot<f64_t>, nk_dot_f64_serial);
    check("dot_f16_serial", test_dot<f16_t>, nk_dot_f16_serial);
    check("dot_bf16_serial", test_dot<bf16_t>, nk_dot_bf16_serial);
    check("dot_e4m3_serial", test_dot<e4m3_t>, nk_dot_e4m3_serial);
    check("dot_e5m2_serial", test_dot<e5m2_t>, nk_dot_e5m2_serial);
    check("dot_e2m3_serial", test_dot<e2m3_t>, nk_dot_e2m3_serial);
    check("dot_e3m2_serial", test_dot<e3m2_t>, nk_dot_e3m2_serial);
    check("dot_i8_serial", test_dot<i8_t>, nk_dot_i8_serial);
    check("dot_u8_serial", test_dot<u8_t>, nk_dot_u8_serial);
    check("dot_i4_serial", test_dot<i4x2_t>, nk_dot_i4_serial);
    check("dot_u4_serial", test_dot<u4x2_t>, nk_dot_u4_serial);
    check("dot_u1_serial", test_dot<u1x8_t>, nk_dot_u1_serial);
    check("dot_f32c_serial", test_dot<f32c_t>, nk_dot_f32c_serial);
    check("vdot_f32c_serial", test_vdot<f32c_t>, nk_vdot_f32c_serial);
    check("dot_f64c_serial", test_dot<f64c_t>, nk_dot_f64c_serial);
    check("vdot_f64c_serial", test_vdot<f64c_t>, nk_vdot_f64c_serial);
    check("dot_f16c_serial", test_dot<f16c_t>, nk_dot_f16c_serial);
    check("vdot_f16c_serial", test_vdot<f16c_t>, nk_vdot_f16c_serial);
    check("dot_bf16c_serial", test_dot<bf16c_t>, nk_dot_bf16c_serial);
    check("vdot_bf16c_serial", test_vdot<bf16c_t>, nk_vdot_bf16c_serial);

#endif // NK_RUNTIME_DISPATCH
    // BLAS/MKL/Accelerate precision comparisons are in test_cross_blas.cpp
}
