/**
 *  @brief Reduction tests.
 *  @file test/test_reduce.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/reduce.hpp"
#include "numkong/reduce/serial.h"

constexpr std::size_t max_stride_k = 50;

template <typename input_type_>
error_stats_t test_reduce_moments(typename input_type_::reduce_moments_kernel_t kernel) {
    using sum_type_ = typename input_type_::reduce_moments_sum_t;
    using sumsq_type_ = typename input_type_::reduce_moments_sumsq_t;
    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    std::uniform_int_distribution<std::size_t> stride_bytes_distribution(1, max_stride_k);
    std::size_t const dims_per_value = nk::dimensions_per_value<input_type_>();
    std::size_t const n = nk::divide_round_up(global_config.dense_dimensions, dims_per_value) * dims_per_value;
    auto buffer = make_vector<input_type_>(n * (max_stride_k + sizeof(input_type_)));
    for (auto start = test_start_time(); within_time_budget(start);) {
        std::size_t stride_bytes = stride_bytes_distribution(generator);
        fill_random(generator, buffer);
        typename sum_type_::raw_t sum;
        typename sumsq_type_::raw_t sumsq;
        kernel(buffer.raw_values_data(), n, stride_bytes, &sum, &sumsq);
        sum_type_ ref_sum;
        sumsq_type_ ref_sumsq;
        nk::reduce_moments<input_type_, sum_type_, sumsq_type_, nk::no_simd_k>(buffer.values_data(), n, stride_bytes,
                                                                               &ref_sum, &ref_sumsq);
        stats.accumulate(sum_type_::from_raw(sum), ref_sum);
        stats.accumulate(sumsq_type_::from_raw(sumsq), ref_sumsq);
    }
    return stats;
}

template <typename input_type_>
error_stats_t test_reduce_minmax(typename input_type_::reduce_minmax_kernel_t kernel) {
    using output_type_ = typename input_type_::reduce_minmax_value_t;
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);
    std::uniform_int_distribution<std::size_t> stride_bytes_distribution(1, max_stride_k);
    std::size_t const dims_per_value = nk::dimensions_per_value<input_type_>();
    std::size_t const n = nk::divide_round_up(global_config.dense_dimensions, dims_per_value) * dims_per_value;
    auto buffer = make_vector<input_type_>(n * (max_stride_k + sizeof(input_type_)));
    for (auto start = test_start_time(); within_time_budget(start);) {
        std::size_t stride_bytes = stride_bytes_distribution(generator);
        fill_random(generator, buffer);
        typename output_type_::raw_t min_val, max_val;
        nk_size_t min_idx, max_idx;
        kernel(buffer.raw_values_data(), n, stride_bytes, &min_val, &min_idx, &max_val, &max_idx);
        output_type_ ref_min, ref_max;
        std::size_t ref_min_idx, ref_max_idx;
        nk::reduce_minmax<input_type_, output_type_, nk::no_simd_k>(buffer.values_data(), n, stride_bytes, &ref_min,
                                                                    &ref_min_idx, &ref_max, &ref_max_idx);
        stats.accumulate(output_type_::from_raw(min_val), ref_min);
        stats.accumulate(output_type_::from_raw(max_val), ref_max);
        stats.accumulate(static_cast<nk_size_t>(min_idx), static_cast<nk_size_t>(ref_min_idx));
        stats.accumulate(static_cast<nk_size_t>(max_idx), static_cast<nk_size_t>(ref_max_idx));
    }
    return stats;
}

/**
 *  @brief Known-value test for the vector-shaped reduction wrappers.
 */
inline error_stats_t test_vector_reductions() {
    // Known values, not `assert` — Release defines `NDEBUG`, which would delete the checks.
    error_stats_t stats(comparison_family_t::exact_k);
    nk::f32_t data[] = {nk::f32_t(1), nk::f32_t(2), nk::f32_t(3), nk::f32_t(4), nk::f32_t(5)};
    auto view = nk::vector_view<nk::f32_t>(data, std::size_t {5});

    auto m = nk::moments(view);
    stats.accumulate(m.sum, decltype(m.sum)(15));
    stats.accumulate(m.sumsq, decltype(m.sumsq)(55));

    auto mm = nk::minmax(view);
    stats.accumulate(mm.min_value, nk::f32_t(1));
    stats.accumulate(mm.max_value, nk::f32_t(5));
    stats.accumulate(static_cast<nk_size_t>(mm.min_index), static_cast<nk_size_t>(0));
    stats.accumulate(static_cast<nk_size_t>(mm.max_index), static_cast<nk_size_t>(4));
    return stats;
}

void test_reduce() {
    error_stats_section_t check;

    check.section("Reductions Serial", nk_cap_serial_k);
    check("vector_reductions", test_vector_reductions);
    check("reduce_moments_f32_serial", test_reduce_moments<f32_t>, nk_reduce_moments_f32_serial);
    check("reduce_moments_f64_serial", test_reduce_moments<f64_t>, nk_reduce_moments_f64_serial);
    check("reduce_moments_i8_serial", test_reduce_moments<i8_t>, nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_serial", test_reduce_moments<u8_t>, nk_reduce_moments_u8_serial);
    check("reduce_moments_i16_serial", test_reduce_moments<i16_t>, nk_reduce_moments_i16_serial);
    check("reduce_moments_u16_serial", test_reduce_moments<u16_t>, nk_reduce_moments_u16_serial);
    check("reduce_moments_i32_serial", test_reduce_moments<i32_t>, nk_reduce_moments_i32_serial);
    check("reduce_moments_u32_serial", test_reduce_moments<u32_t>, nk_reduce_moments_u32_serial);
    check("reduce_moments_i64_serial", test_reduce_moments<i64_t>, nk_reduce_moments_i64_serial);
    check("reduce_moments_u64_serial", test_reduce_moments<u64_t>, nk_reduce_moments_u64_serial);
    check("reduce_moments_f16_serial", test_reduce_moments<f16_t>, nk_reduce_moments_f16_serial);
    check("reduce_moments_bf16_serial", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_serial);
    check("reduce_moments_e4m3_serial", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2_serial", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_serial);
    check("reduce_moments_e2m3_serial", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_serial);
    check("reduce_moments_e3m2_serial", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_serial);
    check("reduce_moments_i4_serial", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4_serial);
    check("reduce_moments_u4_serial", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4_serial);
    check("reduce_moments_u1_serial", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1_serial);
    check("reduce_minmax_f32_serial", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_serial);
    check("reduce_minmax_f64_serial", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_serial);
    check("reduce_minmax_i8_serial", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_serial);
    check("reduce_minmax_u8_serial", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_serial);
    check("reduce_minmax_i16_serial", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_serial);
    check("reduce_minmax_u16_serial", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_serial);
    check("reduce_minmax_i32_serial", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_serial);
    check("reduce_minmax_u32_serial", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_serial);
    check("reduce_minmax_i64_serial", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_serial);
    check("reduce_minmax_u64_serial", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_serial);
    check("reduce_minmax_f16_serial", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_serial);
    check("reduce_minmax_bf16_serial", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_serial);
    check("reduce_minmax_e4m3_serial", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_serial);
    check("reduce_minmax_e5m2_serial", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_serial);
    check("reduce_minmax_e2m3_serial", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_serial);
    check("reduce_minmax_e3m2_serial", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_serial);
    check("reduce_minmax_i4_serial", test_reduce_minmax<i4x2_t>, nk_reduce_minmax_i4_serial);
    check("reduce_minmax_u4_serial", test_reduce_minmax<u4x2_t>, nk_reduce_minmax_u4_serial);
    check("reduce_minmax_u1_serial", test_reduce_minmax<u1x8_t>, nk_reduce_minmax_u1_serial);

#if NK_DYNAMIC_DISPATCH
    check.section("Reductions Dynamic", nk_cap_serial_k);
    check("reduce_moments_f32", test_reduce_moments<f32_t>, nk_reduce_moments_f32);
    check("reduce_moments_f64", test_reduce_moments<f64_t>, nk_reduce_moments_f64);
    check("reduce_moments_i8", test_reduce_moments<i8_t>, nk_reduce_moments_i8);
    check("reduce_moments_u8", test_reduce_moments<u8_t>, nk_reduce_moments_u8);
    check("reduce_moments_i16", test_reduce_moments<i16_t>, nk_reduce_moments_i16);
    check("reduce_moments_u16", test_reduce_moments<u16_t>, nk_reduce_moments_u16);
    check("reduce_moments_i32", test_reduce_moments<i32_t>, nk_reduce_moments_i32);
    check("reduce_moments_u32", test_reduce_moments<u32_t>, nk_reduce_moments_u32);
    check("reduce_moments_i64", test_reduce_moments<i64_t>, nk_reduce_moments_i64);
    check("reduce_moments_u64", test_reduce_moments<u64_t>, nk_reduce_moments_u64);
    check("reduce_moments_f16", test_reduce_moments<f16_t>, nk_reduce_moments_f16);
    check("reduce_moments_bf16", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16);
    check("reduce_moments_e4m3", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3);
    check("reduce_moments_e5m2", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2);
    check("reduce_moments_e2m3", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3);
    check("reduce_moments_e3m2", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2);
    check("reduce_moments_i4", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4);
    check("reduce_moments_u4", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4);
    check("reduce_moments_u1", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1);
    check("reduce_minmax_f32", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32);
    check("reduce_minmax_f64", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64);
    check("reduce_minmax_i8", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8);
    check("reduce_minmax_u8", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8);
    check("reduce_minmax_i16", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16);
    check("reduce_minmax_u16", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16);
    check("reduce_minmax_i32", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32);
    check("reduce_minmax_u32", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32);
    check("reduce_minmax_i64", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64);
    check("reduce_minmax_u64", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64);
    check("reduce_minmax_f16", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16);
    check("reduce_minmax_bf16", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16);
    check("reduce_minmax_e4m3", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3);
    check("reduce_minmax_e5m2", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2);
    check("reduce_minmax_e2m3", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3);
    check("reduce_minmax_e3m2", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2);
    check("reduce_minmax_i4", test_reduce_minmax<i4x2_t>, nk_reduce_minmax_i4);
    check("reduce_minmax_u4", test_reduce_minmax<u4x2_t>, nk_reduce_minmax_u4);
    check("reduce_minmax_u1", test_reduce_minmax<u1x8_t>, nk_reduce_minmax_u1);
#endif

#if NK_TARGET_NEON
    check.section("Reductions NEON", nk_cap_neon_k);
    check("reduce_moments_f32_neon", test_reduce_moments<f32_t>, nk_reduce_moments_f32_neon);
    check("reduce_moments_f64_neon", test_reduce_moments<f64_t>, nk_reduce_moments_f64_neon);
    check("reduce_moments_f16_neon", test_reduce_moments<f16_t>, nk_reduce_moments_f16_neon);
    check("reduce_moments_i8_neon", test_reduce_moments<i8_t>, nk_reduce_moments_i8_neon);
    check("reduce_moments_u8_neon", test_reduce_moments<u8_t>, nk_reduce_moments_u8_neon);
    check("reduce_moments_i16_neon", test_reduce_moments<i16_t>, nk_reduce_moments_i16_neon);
    check("reduce_moments_u16_neon", test_reduce_moments<u16_t>, nk_reduce_moments_u16_neon);
    check("reduce_moments_i32_neon", test_reduce_moments<i32_t>, nk_reduce_moments_i32_neon);
    check("reduce_moments_u32_neon", test_reduce_moments<u32_t>, nk_reduce_moments_u32_neon);
    check("reduce_moments_i64_neon", test_reduce_moments<i64_t>, nk_reduce_moments_i64_neon);
    check("reduce_moments_u64_neon", test_reduce_moments<u64_t>, nk_reduce_moments_u64_neon);
    check("reduce_moments_e4m3_neon", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_neon);
    check("reduce_moments_e5m2_neon", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_neon);
    check("reduce_moments_e2m3_neon", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_neon);
    check("reduce_moments_e3m2_neon", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_neon);
    check("reduce_minmax_f32_neon", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_neon);
    check("reduce_minmax_f64_neon", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_neon);
    check("reduce_minmax_i8_neon", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_neon);
    check("reduce_minmax_u8_neon", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_neon);
    check("reduce_minmax_i16_neon", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_neon);
    check("reduce_minmax_u16_neon", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_neon);
    check("reduce_minmax_i32_neon", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_neon);
    check("reduce_minmax_u32_neon", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_neon);
    check("reduce_minmax_i64_neon", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_neon);
    check("reduce_minmax_u64_neon", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_neon);
    check("reduce_minmax_e4m3_neon", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_neon);
    check("reduce_minmax_e5m2_neon", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_neon);
    check("reduce_minmax_e2m3_neon", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_neon);
    check("reduce_minmax_e3m2_neon", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONBFDOT
    check.section("Reductions NEON BF16", nk_cap_neonbfdot_k);
    check("reduce_moments_bf16_neonbfdot", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONSDOT
    check.section("Reductions NEON I8", nk_cap_neonsdot_k);
    check("reduce_moments_i8_neonsdot", test_reduce_moments<i8_t>, nk_reduce_moments_i8_neonsdot);
    check("reduce_moments_u8_neonsdot", test_reduce_moments<u8_t>, nk_reduce_moments_u8_neonsdot);
    check("reduce_moments_e2m3_neonsdot", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_neonsdot);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONFHM
    check.section("Reductions NEON FHM", nk_cap_neonfhm_k);
    check("reduce_moments_e4m3_neonfhm", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_neonfhm);
    check("reduce_moments_e5m2_neonfhm", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_neonfhm);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_HASWELL
    check.section("Reductions Haswell", nk_cap_haswell_k);
    check("reduce_moments_f32_haswell", test_reduce_moments<f32_t>, nk_reduce_moments_f32_haswell);
    check("reduce_moments_f64_haswell", test_reduce_moments<f64_t>, nk_reduce_moments_f64_haswell);
    check("reduce_moments_i8_haswell", test_reduce_moments<i8_t>, nk_reduce_moments_i8_haswell);
    check("reduce_moments_u8_haswell", test_reduce_moments<u8_t>, nk_reduce_moments_u8_haswell);
    check("reduce_moments_i16_haswell", test_reduce_moments<i16_t>, nk_reduce_moments_i16_haswell);
    check("reduce_moments_u16_haswell", test_reduce_moments<u16_t>, nk_reduce_moments_u16_haswell);
    check("reduce_moments_i32_haswell", test_reduce_moments<i32_t>, nk_reduce_moments_i32_haswell);
    check("reduce_moments_u32_haswell", test_reduce_moments<u32_t>, nk_reduce_moments_u32_haswell);
    check("reduce_moments_i64_haswell", test_reduce_moments<i64_t>, nk_reduce_moments_i64_haswell);
    check("reduce_moments_u64_haswell", test_reduce_moments<u64_t>, nk_reduce_moments_u64_haswell);
    check("reduce_moments_e4m3_haswell", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_haswell);
    check("reduce_moments_e5m2_haswell", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_haswell);
    check("reduce_moments_e2m3_haswell", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_haswell);
    check("reduce_moments_e3m2_haswell", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_haswell);
    check("reduce_moments_i4_haswell", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4_haswell);
    check("reduce_moments_u4_haswell", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4_haswell);
    check("reduce_moments_u1_haswell", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1_haswell);
    check("reduce_moments_bf16_haswell", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_haswell);
    check("reduce_moments_f16_haswell", test_reduce_moments<f16_t>, nk_reduce_moments_f16_haswell);
    check("reduce_minmax_f32_haswell", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_haswell);
    check("reduce_minmax_f64_haswell", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_haswell);
    check("reduce_minmax_i8_haswell", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_haswell);
    check("reduce_minmax_u8_haswell", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_haswell);
    check("reduce_minmax_i16_haswell", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_haswell);
    check("reduce_minmax_u16_haswell", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_haswell);
    check("reduce_minmax_i32_haswell", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_haswell);
    check("reduce_minmax_u32_haswell", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_haswell);
    check("reduce_minmax_i64_haswell", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_haswell);
    check("reduce_minmax_u64_haswell", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_haswell);
    check("reduce_minmax_e4m3_haswell", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_haswell);
    check("reduce_minmax_e5m2_haswell", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_haswell);
    check("reduce_minmax_e2m3_haswell", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_haswell);
    check("reduce_minmax_e3m2_haswell", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_haswell);
    check("reduce_minmax_bf16_haswell", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_haswell);
    check("reduce_minmax_f16_haswell", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check.section("Reductions Skylake", nk_cap_skylake_k);
    check("reduce_moments_f32_skylake", test_reduce_moments<f32_t>, nk_reduce_moments_f32_skylake);
    check("reduce_moments_f64_skylake", test_reduce_moments<f64_t>, nk_reduce_moments_f64_skylake);
    check("reduce_moments_i8_skylake", test_reduce_moments<i8_t>, nk_reduce_moments_i8_skylake);
    check("reduce_moments_u8_skylake", test_reduce_moments<u8_t>, nk_reduce_moments_u8_skylake);
    check("reduce_moments_i16_skylake", test_reduce_moments<i16_t>, nk_reduce_moments_i16_skylake);
    check("reduce_moments_u16_skylake", test_reduce_moments<u16_t>, nk_reduce_moments_u16_skylake);
    check("reduce_moments_i32_skylake", test_reduce_moments<i32_t>, nk_reduce_moments_i32_skylake);
    check("reduce_moments_u32_skylake", test_reduce_moments<u32_t>, nk_reduce_moments_u32_skylake);
    check("reduce_moments_i64_skylake", test_reduce_moments<i64_t>, nk_reduce_moments_i64_skylake);
    check("reduce_moments_u64_skylake", test_reduce_moments<u64_t>, nk_reduce_moments_u64_skylake);
    check("reduce_moments_e4m3_skylake", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_skylake);
    check("reduce_moments_e5m2_skylake", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_skylake);
    check("reduce_moments_e2m3_skylake", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_skylake);
    check("reduce_moments_e3m2_skylake", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_skylake);
    check("reduce_moments_i4_skylake", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4_skylake);
    check("reduce_moments_u4_skylake", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4_skylake);
    check("reduce_moments_u1_skylake", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1_skylake);
    check("reduce_minmax_f32_skylake", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_skylake);
    check("reduce_minmax_f64_skylake", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_skylake);
    check("reduce_minmax_i8_skylake", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_skylake);
    check("reduce_minmax_u8_skylake", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_skylake);
    check("reduce_minmax_i16_skylake", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_skylake);
    check("reduce_minmax_u16_skylake", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_skylake);
    check("reduce_minmax_i32_skylake", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_skylake);
    check("reduce_minmax_u32_skylake", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_skylake);
    check("reduce_minmax_i64_skylake", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_skylake);
    check("reduce_minmax_u64_skylake", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_skylake);
    check("reduce_minmax_e4m3_skylake", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_skylake);
    check("reduce_minmax_e5m2_skylake", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_skylake);
    check("reduce_minmax_e2m3_skylake", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_skylake);
    check("reduce_minmax_e3m2_skylake", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_skylake);
    check("reduce_moments_bf16_skylake", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_skylake);
    check("reduce_minmax_bf16_skylake", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_skylake);
    check("reduce_moments_f16_skylake", test_reduce_moments<f16_t>, nk_reduce_moments_f16_skylake);
    check("reduce_minmax_f16_skylake", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
    check.section("Reductions Ice Lake", nk_cap_icelake_k);
    check("reduce_moments_i8_icelake", test_reduce_moments<i8_t>, nk_reduce_moments_i8_icelake);
    check("reduce_moments_u8_icelake", test_reduce_moments<u8_t>, nk_reduce_moments_u8_icelake);
    check("reduce_moments_i16_icelake", test_reduce_moments<i16_t>, nk_reduce_moments_i16_icelake);
    check("reduce_moments_e2m3_icelake", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_icelake);
    check("reduce_moments_e3m2_icelake", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_GENOA
    check.section("Reductions Genoa", nk_cap_genoa_k);
    check("reduce_moments_bf16_genoa", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_genoa);
    check("reduce_moments_e4m3_genoa", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_genoa);
    check("reduce_moments_e5m2_genoa", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_genoa);
#endif // NK_TARGET_GENOA

#if NK_TARGET_ALDER
    check.section("Reductions Alder", nk_cap_alder_k);
    check("reduce_moments_u8_alder", test_reduce_moments<u8_t>, nk_reduce_moments_u8_alder);
    check("reduce_moments_i16_alder", test_reduce_moments<i16_t>, nk_reduce_moments_i16_alder);
    check("reduce_moments_u16_alder", test_reduce_moments<u16_t>, nk_reduce_moments_u16_alder);
    check("reduce_moments_e3m2_alder", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_alder);
#endif // NK_TARGET_ALDER

#if NK_TARGET_SIERRA
    check.section("Reductions Sierra", nk_cap_sierra_k);
    check("reduce_moments_i8_sierra", test_reduce_moments<i8_t>, nk_reduce_moments_i8_sierra);
    check("reduce_moments_u8_sierra", test_reduce_moments<u8_t>, nk_reduce_moments_u8_sierra);
    check("reduce_moments_e2m3_sierra", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_sierra);
#endif // NK_TARGET_SIERRA

#if NK_TARGET_RVV
    check.section("Reductions RVV", nk_cap_rvv_k);
    check("reduce_moments_f32_rvv", test_reduce_moments<f32_t>, nk_reduce_moments_f32_rvv);
    check("reduce_moments_f64_rvv", test_reduce_moments<f64_t>, nk_reduce_moments_f64_rvv);
    check("reduce_moments_i8_rvv", test_reduce_moments<i8_t>, nk_reduce_moments_i8_rvv);
    check("reduce_moments_u8_rvv", test_reduce_moments<u8_t>, nk_reduce_moments_u8_rvv);
    check("reduce_moments_i16_rvv", test_reduce_moments<i16_t>, nk_reduce_moments_i16_rvv);
    check("reduce_moments_u16_rvv", test_reduce_moments<u16_t>, nk_reduce_moments_u16_rvv);
    check("reduce_moments_i32_rvv", test_reduce_moments<i32_t>, nk_reduce_moments_i32_rvv);
    check("reduce_moments_u32_rvv", test_reduce_moments<u32_t>, nk_reduce_moments_u32_rvv);
    check("reduce_moments_i64_rvv", test_reduce_moments<i64_t>, nk_reduce_moments_i64_rvv);
    check("reduce_moments_u64_rvv", test_reduce_moments<u64_t>, nk_reduce_moments_u64_rvv);
    check("reduce_moments_bf16_rvv", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_rvv);
    check("reduce_moments_f16_rvv", test_reduce_moments<f16_t>, nk_reduce_moments_f16_rvv);
    check("reduce_moments_e4m3_rvv", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_rvv);
    check("reduce_moments_e5m2_rvv", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_rvv);
    check("reduce_moments_e2m3_rvv", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_rvv);
    check("reduce_moments_e3m2_rvv", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_rvv);
    check("reduce_minmax_f32_rvv", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_rvv);
    check("reduce_minmax_f64_rvv", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_rvv);
    check("reduce_minmax_i8_rvv", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_rvv);
    check("reduce_minmax_u8_rvv", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_rvv);
    check("reduce_minmax_i16_rvv", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_rvv);
    check("reduce_minmax_u16_rvv", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_rvv);
    check("reduce_minmax_i32_rvv", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_rvv);
    check("reduce_minmax_u32_rvv", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_rvv);
    check("reduce_minmax_i64_rvv", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_rvv);
    check("reduce_minmax_u64_rvv", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_rvv);
    check("reduce_minmax_bf16_rvv", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_rvv);
    check("reduce_minmax_f16_rvv", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_rvv);
    check("reduce_minmax_e4m3_rvv", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_rvv);
    check("reduce_minmax_e5m2_rvv", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_rvv);
    check("reduce_minmax_e2m3_rvv", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_rvv);
    check("reduce_minmax_e3m2_rvv", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    check.section("Reductions V128 Relaxed", nk_cap_v128relaxed_k);
    check("reduce_moments_f32_v128relaxed", test_reduce_moments<f32_t>, nk_reduce_moments_f32_v128relaxed);
    check("reduce_moments_f64_v128relaxed", test_reduce_moments<f64_t>, nk_reduce_moments_f64_v128relaxed);
    check("reduce_moments_i8_v128relaxed", test_reduce_moments<i8_t>, nk_reduce_moments_i8_v128relaxed);
    check("reduce_moments_u8_v128relaxed", test_reduce_moments<u8_t>, nk_reduce_moments_u8_v128relaxed);
    check("reduce_moments_i16_v128relaxed", test_reduce_moments<i16_t>, nk_reduce_moments_i16_v128relaxed);
    check("reduce_moments_u16_v128relaxed", test_reduce_moments<u16_t>, nk_reduce_moments_u16_v128relaxed);
    check("reduce_moments_i32_v128relaxed", test_reduce_moments<i32_t>, nk_reduce_moments_i32_v128relaxed);
    check("reduce_moments_u32_v128relaxed", test_reduce_moments<u32_t>, nk_reduce_moments_u32_v128relaxed);
    check("reduce_moments_i64_v128relaxed", test_reduce_moments<i64_t>, nk_reduce_moments_i64_v128relaxed);
    check("reduce_moments_u64_v128relaxed", test_reduce_moments<u64_t>, nk_reduce_moments_u64_v128relaxed);
    check("reduce_moments_e4m3_v128relaxed", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_v128relaxed);
    check("reduce_moments_e5m2_v128relaxed", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_v128relaxed);
    check("reduce_moments_e2m3_v128relaxed", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_v128relaxed);
    check("reduce_moments_e3m2_v128relaxed", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_v128relaxed);
    check("reduce_moments_bf16_v128relaxed", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_v128relaxed);
    check("reduce_moments_f16_v128relaxed", test_reduce_moments<f16_t>, nk_reduce_moments_f16_v128relaxed);
    check("reduce_minmax_f32_v128relaxed", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_v128relaxed);
    check("reduce_minmax_f64_v128relaxed", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_v128relaxed);
    check("reduce_minmax_i8_v128relaxed", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_v128relaxed);
    check("reduce_minmax_u8_v128relaxed", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_v128relaxed);
    check("reduce_minmax_i16_v128relaxed", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_v128relaxed);
    check("reduce_minmax_u16_v128relaxed", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_v128relaxed);
    check("reduce_minmax_i32_v128relaxed", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_v128relaxed);
    check("reduce_minmax_u32_v128relaxed", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_v128relaxed);
    check("reduce_minmax_i64_v128relaxed", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_v128relaxed);
    check("reduce_minmax_u64_v128relaxed", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_v128relaxed);
    check("reduce_minmax_e4m3_v128relaxed", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_v128relaxed);
    check("reduce_minmax_e5m2_v128relaxed", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_v128relaxed);
    check("reduce_minmax_e2m3_v128relaxed", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_v128relaxed);
    check("reduce_minmax_e3m2_v128relaxed", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_v128relaxed);
    check("reduce_minmax_bf16_v128relaxed", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_v128relaxed);
    check("reduce_minmax_f16_v128relaxed", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_v128relaxed);
#endif // NK_TARGET_V128RELAXED
}
