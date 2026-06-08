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
error_stats_t test_reduce_moments(typename input_type_::reduce_moments_kernel_t kernel,
                                  typename input_type_::reduce_moments_kernel_t reference) {
    using sum_type_ = typename input_type_::reduce_moments_sum_t;
    using sumsq_type_ = typename input_type_::reduce_moments_sumsq_t;
    error_stats_t stats(comparison_family_t::narrow_arithmetic_k);
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
        typename sum_type_::raw_t ref_sum;
        typename sumsq_type_::raw_t ref_sumsq;
        reference(buffer.raw_values_data(), n, stride_bytes, &ref_sum, &ref_sumsq);
        stats.accumulate(sum_type_::from_raw(sum), sum_type_::from_raw(ref_sum));
        stats.accumulate(sumsq_type_::from_raw(sumsq), sumsq_type_::from_raw(ref_sumsq));
    }
    return stats;
}

template <typename input_type_>
error_stats_t test_reduce_minmax(typename input_type_::reduce_minmax_kernel_t kernel,
                                 typename input_type_::reduce_minmax_kernel_t reference) {
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
        typename output_type_::raw_t ref_min, ref_max;
        nk_size_t ref_min_idx, ref_max_idx;
        reference(buffer.raw_values_data(), n, stride_bytes, &ref_min, &ref_min_idx, &ref_max, &ref_max_idx);
        stats.accumulate(output_type_::from_raw(min_val), output_type_::from_raw(ref_min));
        stats.accumulate(output_type_::from_raw(max_val), output_type_::from_raw(ref_max));
    }
    return stats;
}

/**
 *  @brief Smoke-test for the vector-shaped reduction wrappers (`nk::moments`/`minmax`/`sum`/...).
 *  Exercises every accessor variant on a small random vector. Numerical accuracy of the
 *  underlying kernels is validated by `test_reduce_moments` / `test_reduce_minmax` above.
 */
template <typename value_type_>
void test_vector_reductions_for_type() {
    auto v = make_vector<value_type_>(32);
    std::mt19937 generator(42);
    fill_random(generator, v);
    auto view = nk::vector_view<value_type_>(v.values_data(), static_cast<std::size_t>(v.size()));

    { [[maybe_unused]] auto r = nk::moments(view); }
    { [[maybe_unused]] auto r = nk::minmax(view); }
    { [[maybe_unused]] auto r = nk::sum(view); }
    { [[maybe_unused]] auto r = nk::min(view); }
    { [[maybe_unused]] auto r = nk::max(view); }
    { [[maybe_unused]] auto r = nk::argmin(view); }
    { [[maybe_unused]] auto r = nk::argmax(view); }
}

/**
 *  @brief Golden-value test for the vector-shaped reduction wrappers.
 */
inline void test_vector_reductions_correctness() {
    nk::f32_t data[] = {nk::f32_t(1), nk::f32_t(2), nk::f32_t(3), nk::f32_t(4), nk::f32_t(5)};
    auto view = nk::vector_view<nk::f32_t>(data, std::size_t {5});

    auto m = nk::moments(view);
    assert(m.sum.raw_ == 15.0 && "vector moments sum");
    assert(m.sumsq.raw_ == 55.0 && "vector moments sumsq");

    auto mm = nk::minmax(view);
    assert(mm.min_value.raw_ == 1.0f && "vector min");
    assert(mm.max_value.raw_ == 5.0f && "vector max");
    assert(mm.min_index == 0 && "vector argmin");
    assert(mm.max_index == 4 && "vector argmax");
}

void test_reduce() {
    error_stats_section_t check("Reductions");

    // Tensor- and vector-shaped wrapper smoke tests (independent of ISA dispatch).
    test_vector_reductions_for_type<nk::f32_t>();
    test_vector_reductions_for_type<nk::f64_t>();
    test_vector_reductions_for_type<nk::f16_t>();
    test_vector_reductions_for_type<nk::bf16_t>();
    test_vector_reductions_for_type<nk::i8_t>();
    test_vector_reductions_for_type<nk::u8_t>();
    std::printf("  vector reductions (6 types):  OK\n");

    test_vector_reductions_correctness();
    std::printf("  vector reductions correct:    OK\n");

#if NK_DYNAMIC_DISPATCH
    check("reduce_moments_f32", test_reduce_moments<f32_t>, nk_reduce_moments_f32, nk_reduce_moments_f32_serial);
    check("reduce_moments_f64", test_reduce_moments<f64_t>, nk_reduce_moments_f64, nk_reduce_moments_f64_serial);
    check("reduce_moments_i8", test_reduce_moments<i8_t>, nk_reduce_moments_i8, nk_reduce_moments_i8_serial);
    check("reduce_moments_u8", test_reduce_moments<u8_t>, nk_reduce_moments_u8, nk_reduce_moments_u8_serial);
    check("reduce_moments_i16", test_reduce_moments<i16_t>, nk_reduce_moments_i16, nk_reduce_moments_i16_serial);
    check("reduce_moments_u16", test_reduce_moments<u16_t>, nk_reduce_moments_u16, nk_reduce_moments_u16_serial);
    check("reduce_moments_i32", test_reduce_moments<i32_t>, nk_reduce_moments_i32, nk_reduce_moments_i32_serial);
    check("reduce_moments_u32", test_reduce_moments<u32_t>, nk_reduce_moments_u32, nk_reduce_moments_u32_serial);
    check("reduce_moments_i64", test_reduce_moments<i64_t>, nk_reduce_moments_i64, nk_reduce_moments_i64_serial);
    check("reduce_moments_u64", test_reduce_moments<u64_t>, nk_reduce_moments_u64, nk_reduce_moments_u64_serial);
    check("reduce_moments_f16", test_reduce_moments<f16_t>, nk_reduce_moments_f16, nk_reduce_moments_f16_serial);
    check("reduce_moments_bf16", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16, nk_reduce_moments_bf16_serial);
    check("reduce_moments_e4m3", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3, nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2, nk_reduce_moments_e5m2_serial);
    check("reduce_moments_e2m3", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3, nk_reduce_moments_e2m3_serial);
    check("reduce_moments_e3m2", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2, nk_reduce_moments_e3m2_serial);
    check("reduce_moments_i4", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4, nk_reduce_moments_i4_serial);
    check("reduce_moments_u4", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4, nk_reduce_moments_u4_serial);
    check("reduce_moments_u1", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1, nk_reduce_moments_u1_serial);
    check("reduce_minmax_f32", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32, nk_reduce_minmax_f32_serial);
    check("reduce_minmax_f64", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64, nk_reduce_minmax_f64_serial);
    check("reduce_minmax_i8", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8, nk_reduce_minmax_i8_serial);
    check("reduce_minmax_u8", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8, nk_reduce_minmax_u8_serial);
    check("reduce_minmax_i16", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16, nk_reduce_minmax_i16_serial);
    check("reduce_minmax_u16", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16, nk_reduce_minmax_u16_serial);
    check("reduce_minmax_i32", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32, nk_reduce_minmax_i32_serial);
    check("reduce_minmax_u32", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32, nk_reduce_minmax_u32_serial);
    check("reduce_minmax_i64", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64, nk_reduce_minmax_i64_serial);
    check("reduce_minmax_u64", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64, nk_reduce_minmax_u64_serial);
    check("reduce_minmax_f16", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16, nk_reduce_minmax_f16_serial);
    check("reduce_minmax_bf16", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16, nk_reduce_minmax_bf16_serial);
    check("reduce_minmax_e4m3", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3, nk_reduce_minmax_e4m3_serial);
    check("reduce_minmax_e5m2", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2, nk_reduce_minmax_e5m2_serial);
    check("reduce_minmax_e2m3", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3, nk_reduce_minmax_e2m3_serial);
    check("reduce_minmax_e3m2", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2, nk_reduce_minmax_e3m2_serial);
    check("reduce_minmax_i4", test_reduce_minmax<i4x2_t>, nk_reduce_minmax_i4, nk_reduce_minmax_i4_serial);
    check("reduce_minmax_u4", test_reduce_minmax<u4x2_t>, nk_reduce_minmax_u4, nk_reduce_minmax_u4_serial);
    check("reduce_minmax_u1", test_reduce_minmax<u1x8_t>, nk_reduce_minmax_u1, nk_reduce_minmax_u1_serial);
#else

#if NK_TARGET_NEON
    check("reduce_moments_f32_neon", test_reduce_moments<f32_t>, nk_reduce_moments_f32_neon,
          nk_reduce_moments_f32_serial);
    check("reduce_moments_f64_neon", test_reduce_moments<f64_t>, nk_reduce_moments_f64_neon,
          nk_reduce_moments_f64_serial);
    check("reduce_moments_f16_neon", test_reduce_moments<f16_t>, nk_reduce_moments_f16_neon,
          nk_reduce_moments_f16_serial);
    check("reduce_moments_i8_neon", test_reduce_moments<i8_t>, nk_reduce_moments_i8_neon, nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_neon", test_reduce_moments<u8_t>, nk_reduce_moments_u8_neon, nk_reduce_moments_u8_serial);
    check("reduce_moments_i16_neon", test_reduce_moments<i16_t>, nk_reduce_moments_i16_neon,
          nk_reduce_moments_i16_serial);
    check("reduce_moments_u16_neon", test_reduce_moments<u16_t>, nk_reduce_moments_u16_neon,
          nk_reduce_moments_u16_serial);
    check("reduce_moments_i32_neon", test_reduce_moments<i32_t>, nk_reduce_moments_i32_neon,
          nk_reduce_moments_i32_serial);
    check("reduce_moments_u32_neon", test_reduce_moments<u32_t>, nk_reduce_moments_u32_neon,
          nk_reduce_moments_u32_serial);
    check("reduce_moments_i64_neon", test_reduce_moments<i64_t>, nk_reduce_moments_i64_neon,
          nk_reduce_moments_i64_serial);
    check("reduce_moments_u64_neon", test_reduce_moments<u64_t>, nk_reduce_moments_u64_neon,
          nk_reduce_moments_u64_serial);
    check("reduce_moments_e4m3_neon", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_neon,
          nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2_neon", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_neon,
          nk_reduce_moments_e5m2_serial);
    check("reduce_moments_e2m3_neon", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_neon,
          nk_reduce_moments_e2m3_serial);
    check("reduce_moments_e3m2_neon", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_neon,
          nk_reduce_moments_e3m2_serial);
    check("reduce_minmax_f32_neon", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_neon, nk_reduce_minmax_f32_serial);
    check("reduce_minmax_f64_neon", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_neon, nk_reduce_minmax_f64_serial);
    check("reduce_minmax_i8_neon", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_neon, nk_reduce_minmax_i8_serial);
    check("reduce_minmax_u8_neon", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_neon, nk_reduce_minmax_u8_serial);
    check("reduce_minmax_i16_neon", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_neon, nk_reduce_minmax_i16_serial);
    check("reduce_minmax_u16_neon", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_neon, nk_reduce_minmax_u16_serial);
    check("reduce_minmax_i32_neon", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_neon, nk_reduce_minmax_i32_serial);
    check("reduce_minmax_u32_neon", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_neon, nk_reduce_minmax_u32_serial);
    check("reduce_minmax_i64_neon", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_neon, nk_reduce_minmax_i64_serial);
    check("reduce_minmax_u64_neon", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_neon, nk_reduce_minmax_u64_serial);
    check("reduce_minmax_e4m3_neon", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_neon,
          nk_reduce_minmax_e4m3_serial);
    check("reduce_minmax_e5m2_neon", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_neon,
          nk_reduce_minmax_e5m2_serial);
    check("reduce_minmax_e2m3_neon", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_neon,
          nk_reduce_minmax_e2m3_serial);
    check("reduce_minmax_e3m2_neon", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_neon,
          nk_reduce_minmax_e3m2_serial);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONBFDOT
    check("reduce_moments_bf16_neonbfdot", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_neonbfdot,
          nk_reduce_moments_bf16_serial);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONSDOT
    check("reduce_moments_i8_neonsdot", test_reduce_moments<i8_t>, nk_reduce_moments_i8_neonsdot,
          nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_neonsdot", test_reduce_moments<u8_t>, nk_reduce_moments_u8_neonsdot,
          nk_reduce_moments_u8_serial);
    check("reduce_moments_e2m3_neonsdot", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_neonsdot,
          nk_reduce_moments_e2m3_serial);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONFHM
    check("reduce_moments_e4m3_neonfhm", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_neonfhm,
          nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2_neonfhm", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_neonfhm,
          nk_reduce_moments_e5m2_serial);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_HASWELL
    check("reduce_moments_f32_haswell", test_reduce_moments<f32_t>, nk_reduce_moments_f32_haswell,
          nk_reduce_moments_f32_serial);
    check("reduce_moments_f64_haswell", test_reduce_moments<f64_t>, nk_reduce_moments_f64_haswell,
          nk_reduce_moments_f64_serial);
    check("reduce_moments_i8_haswell", test_reduce_moments<i8_t>, nk_reduce_moments_i8_haswell,
          nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_haswell", test_reduce_moments<u8_t>, nk_reduce_moments_u8_haswell,
          nk_reduce_moments_u8_serial);
    check("reduce_moments_i16_haswell", test_reduce_moments<i16_t>, nk_reduce_moments_i16_haswell,
          nk_reduce_moments_i16_serial);
    check("reduce_moments_u16_haswell", test_reduce_moments<u16_t>, nk_reduce_moments_u16_haswell,
          nk_reduce_moments_u16_serial);
    check("reduce_moments_i32_haswell", test_reduce_moments<i32_t>, nk_reduce_moments_i32_haswell,
          nk_reduce_moments_i32_serial);
    check("reduce_moments_u32_haswell", test_reduce_moments<u32_t>, nk_reduce_moments_u32_haswell,
          nk_reduce_moments_u32_serial);
    check("reduce_moments_i64_haswell", test_reduce_moments<i64_t>, nk_reduce_moments_i64_haswell,
          nk_reduce_moments_i64_serial);
    check("reduce_moments_u64_haswell", test_reduce_moments<u64_t>, nk_reduce_moments_u64_haswell,
          nk_reduce_moments_u64_serial);
    check("reduce_moments_e4m3_haswell", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_haswell,
          nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2_haswell", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_haswell,
          nk_reduce_moments_e5m2_serial);
    check("reduce_moments_e2m3_haswell", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_haswell,
          nk_reduce_moments_e2m3_serial);
    check("reduce_moments_e3m2_haswell", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_haswell,
          nk_reduce_moments_e3m2_serial);
    check("reduce_moments_i4_haswell", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4_haswell,
          nk_reduce_moments_i4_serial);
    check("reduce_moments_u4_haswell", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4_haswell,
          nk_reduce_moments_u4_serial);
    check("reduce_moments_u1_haswell", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1_haswell,
          nk_reduce_moments_u1_serial);
    check("reduce_moments_bf16_haswell", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_haswell,
          nk_reduce_moments_bf16_serial);
    check("reduce_moments_f16_haswell", test_reduce_moments<f16_t>, nk_reduce_moments_f16_haswell,
          nk_reduce_moments_f16_serial);
    check("reduce_minmax_f32_haswell", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_haswell,
          nk_reduce_minmax_f32_serial);
    check("reduce_minmax_f64_haswell", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_haswell,
          nk_reduce_minmax_f64_serial);
    check("reduce_minmax_i8_haswell", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_haswell,
          nk_reduce_minmax_i8_serial);
    check("reduce_minmax_u8_haswell", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_haswell,
          nk_reduce_minmax_u8_serial);
    check("reduce_minmax_i16_haswell", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_haswell,
          nk_reduce_minmax_i16_serial);
    check("reduce_minmax_u16_haswell", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_haswell,
          nk_reduce_minmax_u16_serial);
    check("reduce_minmax_i32_haswell", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_haswell,
          nk_reduce_minmax_i32_serial);
    check("reduce_minmax_u32_haswell", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_haswell,
          nk_reduce_minmax_u32_serial);
    check("reduce_minmax_i64_haswell", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_haswell,
          nk_reduce_minmax_i64_serial);
    check("reduce_minmax_u64_haswell", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_haswell,
          nk_reduce_minmax_u64_serial);
    check("reduce_minmax_e4m3_haswell", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_haswell,
          nk_reduce_minmax_e4m3_serial);
    check("reduce_minmax_e5m2_haswell", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_haswell,
          nk_reduce_minmax_e5m2_serial);
    check("reduce_minmax_e2m3_haswell", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_haswell,
          nk_reduce_minmax_e2m3_serial);
    check("reduce_minmax_e3m2_haswell", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_haswell,
          nk_reduce_minmax_e3m2_serial);
    check("reduce_minmax_bf16_haswell", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_haswell,
          nk_reduce_minmax_bf16_serial);
    check("reduce_minmax_f16_haswell", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_haswell,
          nk_reduce_minmax_f16_serial);
#endif

#if NK_TARGET_SKYLAKE
    check("reduce_moments_f32_skylake", test_reduce_moments<f32_t>, nk_reduce_moments_f32_skylake,
          nk_reduce_moments_f32_serial);
    check("reduce_moments_f64_skylake", test_reduce_moments<f64_t>, nk_reduce_moments_f64_skylake,
          nk_reduce_moments_f64_serial);
    check("reduce_moments_i8_skylake", test_reduce_moments<i8_t>, nk_reduce_moments_i8_skylake,
          nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_skylake", test_reduce_moments<u8_t>, nk_reduce_moments_u8_skylake,
          nk_reduce_moments_u8_serial);
    check("reduce_moments_i16_skylake", test_reduce_moments<i16_t>, nk_reduce_moments_i16_skylake,
          nk_reduce_moments_i16_serial);
    check("reduce_moments_u16_skylake", test_reduce_moments<u16_t>, nk_reduce_moments_u16_skylake,
          nk_reduce_moments_u16_serial);
    check("reduce_moments_i32_skylake", test_reduce_moments<i32_t>, nk_reduce_moments_i32_skylake,
          nk_reduce_moments_i32_serial);
    check("reduce_moments_u32_skylake", test_reduce_moments<u32_t>, nk_reduce_moments_u32_skylake,
          nk_reduce_moments_u32_serial);
    check("reduce_moments_i64_skylake", test_reduce_moments<i64_t>, nk_reduce_moments_i64_skylake,
          nk_reduce_moments_i64_serial);
    check("reduce_moments_u64_skylake", test_reduce_moments<u64_t>, nk_reduce_moments_u64_skylake,
          nk_reduce_moments_u64_serial);
    check("reduce_moments_e4m3_skylake", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_skylake,
          nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2_skylake", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_skylake,
          nk_reduce_moments_e5m2_serial);
    check("reduce_moments_e2m3_skylake", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_skylake,
          nk_reduce_moments_e2m3_serial);
    check("reduce_moments_e3m2_skylake", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_skylake,
          nk_reduce_moments_e3m2_serial);
    check("reduce_moments_i4_skylake", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4_skylake,
          nk_reduce_moments_i4_serial);
    check("reduce_moments_u4_skylake", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4_skylake,
          nk_reduce_moments_u4_serial);
    check("reduce_moments_u1_skylake", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1_skylake,
          nk_reduce_moments_u1_serial);
    check("reduce_minmax_f32_skylake", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_skylake,
          nk_reduce_minmax_f32_serial);
    check("reduce_minmax_f64_skylake", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_skylake,
          nk_reduce_minmax_f64_serial);
    check("reduce_minmax_i8_skylake", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_skylake,
          nk_reduce_minmax_i8_serial);
    check("reduce_minmax_u8_skylake", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_skylake,
          nk_reduce_minmax_u8_serial);
    check("reduce_minmax_i16_skylake", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_skylake,
          nk_reduce_minmax_i16_serial);
    check("reduce_minmax_u16_skylake", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_skylake,
          nk_reduce_minmax_u16_serial);
    check("reduce_minmax_i32_skylake", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_skylake,
          nk_reduce_minmax_i32_serial);
    check("reduce_minmax_u32_skylake", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_skylake,
          nk_reduce_minmax_u32_serial);
    check("reduce_minmax_i64_skylake", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_skylake,
          nk_reduce_minmax_i64_serial);
    check("reduce_minmax_u64_skylake", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_skylake,
          nk_reduce_minmax_u64_serial);
    check("reduce_minmax_e4m3_skylake", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_skylake,
          nk_reduce_minmax_e4m3_serial);
    check("reduce_minmax_e5m2_skylake", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_skylake,
          nk_reduce_minmax_e5m2_serial);
    check("reduce_minmax_e2m3_skylake", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_skylake,
          nk_reduce_minmax_e2m3_serial);
    check("reduce_minmax_e3m2_skylake", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_skylake,
          nk_reduce_minmax_e3m2_serial);
    check("reduce_moments_bf16_skylake", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_skylake,
          nk_reduce_moments_bf16_serial);
    check("reduce_minmax_bf16_skylake", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_skylake,
          nk_reduce_minmax_bf16_serial);
    check("reduce_moments_f16_skylake", test_reduce_moments<f16_t>, nk_reduce_moments_f16_skylake,
          nk_reduce_moments_f16_serial);
    check("reduce_minmax_f16_skylake", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_skylake,
          nk_reduce_minmax_f16_serial);
#endif

#if NK_TARGET_ICELAKE
    check("reduce_moments_i8_icelake", test_reduce_moments<i8_t>, nk_reduce_moments_i8_icelake,
          nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_icelake", test_reduce_moments<u8_t>, nk_reduce_moments_u8_icelake,
          nk_reduce_moments_u8_serial);
    check("reduce_moments_i16_icelake", test_reduce_moments<i16_t>, nk_reduce_moments_i16_icelake,
          nk_reduce_moments_i16_serial);
    check("reduce_moments_e2m3_icelake", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_icelake,
          nk_reduce_moments_e2m3_serial);
    check("reduce_moments_e3m2_icelake", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_icelake,
          nk_reduce_moments_e3m2_serial);
#endif

#if NK_TARGET_GENOA
    check("reduce_moments_bf16_genoa", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_genoa,
          nk_reduce_moments_bf16_serial);
    check("reduce_moments_e4m3_genoa", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_genoa,
          nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2_genoa", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_genoa,
          nk_reduce_moments_e5m2_serial);
#endif

#if NK_TARGET_ALDER
    check("reduce_moments_u8_alder", test_reduce_moments<u8_t>, nk_reduce_moments_u8_alder,
          nk_reduce_moments_u8_serial);
    check("reduce_moments_i16_alder", test_reduce_moments<i16_t>, nk_reduce_moments_i16_alder,
          nk_reduce_moments_i16_serial);
    check("reduce_moments_u16_alder", test_reduce_moments<u16_t>, nk_reduce_moments_u16_alder,
          nk_reduce_moments_u16_serial);
    check("reduce_moments_e3m2_alder", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_alder,
          nk_reduce_moments_e3m2_serial);
#endif

#if NK_TARGET_SIERRA
    check("reduce_moments_i8_sierra", test_reduce_moments<i8_t>, nk_reduce_moments_i8_sierra,
          nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_sierra", test_reduce_moments<u8_t>, nk_reduce_moments_u8_sierra,
          nk_reduce_moments_u8_serial);
    check("reduce_moments_e2m3_sierra", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_sierra,
          nk_reduce_moments_e2m3_serial);
#endif

#if NK_TARGET_RVV
    check("reduce_moments_f32_rvv", test_reduce_moments<f32_t>, nk_reduce_moments_f32_rvv,
          nk_reduce_moments_f32_serial);
    check("reduce_moments_f64_rvv", test_reduce_moments<f64_t>, nk_reduce_moments_f64_rvv,
          nk_reduce_moments_f64_serial);
    check("reduce_moments_i8_rvv", test_reduce_moments<i8_t>, nk_reduce_moments_i8_rvv, nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_rvv", test_reduce_moments<u8_t>, nk_reduce_moments_u8_rvv, nk_reduce_moments_u8_serial);
    check("reduce_moments_i16_rvv", test_reduce_moments<i16_t>, nk_reduce_moments_i16_rvv,
          nk_reduce_moments_i16_serial);
    check("reduce_moments_u16_rvv", test_reduce_moments<u16_t>, nk_reduce_moments_u16_rvv,
          nk_reduce_moments_u16_serial);
    check("reduce_moments_i32_rvv", test_reduce_moments<i32_t>, nk_reduce_moments_i32_rvv,
          nk_reduce_moments_i32_serial);
    check("reduce_moments_u32_rvv", test_reduce_moments<u32_t>, nk_reduce_moments_u32_rvv,
          nk_reduce_moments_u32_serial);
    check("reduce_moments_i64_rvv", test_reduce_moments<i64_t>, nk_reduce_moments_i64_rvv,
          nk_reduce_moments_i64_serial);
    check("reduce_moments_u64_rvv", test_reduce_moments<u64_t>, nk_reduce_moments_u64_rvv,
          nk_reduce_moments_u64_serial);
    check("reduce_moments_bf16_rvv", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_rvv,
          nk_reduce_moments_bf16_serial);
    check("reduce_moments_f16_rvv", test_reduce_moments<f16_t>, nk_reduce_moments_f16_rvv,
          nk_reduce_moments_f16_serial);
    check("reduce_moments_e4m3_rvv", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_rvv,
          nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2_rvv", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_rvv,
          nk_reduce_moments_e5m2_serial);
    check("reduce_moments_e2m3_rvv", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_rvv,
          nk_reduce_moments_e2m3_serial);
    check("reduce_moments_e3m2_rvv", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_rvv,
          nk_reduce_moments_e3m2_serial);
    check("reduce_minmax_f32_rvv", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_rvv, nk_reduce_minmax_f32_serial);
    check("reduce_minmax_f64_rvv", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_rvv, nk_reduce_minmax_f64_serial);
    check("reduce_minmax_i8_rvv", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_rvv, nk_reduce_minmax_i8_serial);
    check("reduce_minmax_u8_rvv", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_rvv, nk_reduce_minmax_u8_serial);
    check("reduce_minmax_i16_rvv", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_rvv, nk_reduce_minmax_i16_serial);
    check("reduce_minmax_u16_rvv", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_rvv, nk_reduce_minmax_u16_serial);
    check("reduce_minmax_i32_rvv", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_rvv, nk_reduce_minmax_i32_serial);
    check("reduce_minmax_u32_rvv", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_rvv, nk_reduce_minmax_u32_serial);
    check("reduce_minmax_i64_rvv", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_rvv, nk_reduce_minmax_i64_serial);
    check("reduce_minmax_u64_rvv", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_rvv, nk_reduce_minmax_u64_serial);
    check("reduce_minmax_bf16_rvv", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_rvv,
          nk_reduce_minmax_bf16_serial);
    check("reduce_minmax_f16_rvv", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_rvv, nk_reduce_minmax_f16_serial);
    check("reduce_minmax_e4m3_rvv", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_rvv,
          nk_reduce_minmax_e4m3_serial);
    check("reduce_minmax_e5m2_rvv", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_rvv,
          nk_reduce_minmax_e5m2_serial);
    check("reduce_minmax_e2m3_rvv", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_rvv,
          nk_reduce_minmax_e2m3_serial);
    check("reduce_minmax_e3m2_rvv", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_rvv,
          nk_reduce_minmax_e3m2_serial);
#endif

#if NK_TARGET_V128RELAXED
    check("reduce_moments_f32_v128relaxed", test_reduce_moments<f32_t>, nk_reduce_moments_f32_v128relaxed,
          nk_reduce_moments_f32_serial);
    check("reduce_moments_f64_v128relaxed", test_reduce_moments<f64_t>, nk_reduce_moments_f64_v128relaxed,
          nk_reduce_moments_f64_serial);
    check("reduce_moments_i8_v128relaxed", test_reduce_moments<i8_t>, nk_reduce_moments_i8_v128relaxed,
          nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_v128relaxed", test_reduce_moments<u8_t>, nk_reduce_moments_u8_v128relaxed,
          nk_reduce_moments_u8_serial);
    check("reduce_moments_i16_v128relaxed", test_reduce_moments<i16_t>, nk_reduce_moments_i16_v128relaxed,
          nk_reduce_moments_i16_serial);
    check("reduce_moments_u16_v128relaxed", test_reduce_moments<u16_t>, nk_reduce_moments_u16_v128relaxed,
          nk_reduce_moments_u16_serial);
    check("reduce_moments_i32_v128relaxed", test_reduce_moments<i32_t>, nk_reduce_moments_i32_v128relaxed,
          nk_reduce_moments_i32_serial);
    check("reduce_moments_u32_v128relaxed", test_reduce_moments<u32_t>, nk_reduce_moments_u32_v128relaxed,
          nk_reduce_moments_u32_serial);
    check("reduce_moments_i64_v128relaxed", test_reduce_moments<i64_t>, nk_reduce_moments_i64_v128relaxed,
          nk_reduce_moments_i64_serial);
    check("reduce_moments_u64_v128relaxed", test_reduce_moments<u64_t>, nk_reduce_moments_u64_v128relaxed,
          nk_reduce_moments_u64_serial);
    check("reduce_moments_e4m3_v128relaxed", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_v128relaxed,
          nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2_v128relaxed", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_v128relaxed,
          nk_reduce_moments_e5m2_serial);
    check("reduce_moments_e2m3_v128relaxed", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_v128relaxed,
          nk_reduce_moments_e2m3_serial);
    check("reduce_moments_e3m2_v128relaxed", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_v128relaxed,
          nk_reduce_moments_e3m2_serial);
    check("reduce_moments_bf16_v128relaxed", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_v128relaxed,
          nk_reduce_moments_bf16_serial);
    check("reduce_moments_f16_v128relaxed", test_reduce_moments<f16_t>, nk_reduce_moments_f16_v128relaxed,
          nk_reduce_moments_f16_serial);
    check("reduce_minmax_f32_v128relaxed", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_v128relaxed,
          nk_reduce_minmax_f32_serial);
    check("reduce_minmax_f64_v128relaxed", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_v128relaxed,
          nk_reduce_minmax_f64_serial);
    check("reduce_minmax_i8_v128relaxed", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_v128relaxed,
          nk_reduce_minmax_i8_serial);
    check("reduce_minmax_u8_v128relaxed", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_v128relaxed,
          nk_reduce_minmax_u8_serial);
    check("reduce_minmax_i16_v128relaxed", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_v128relaxed,
          nk_reduce_minmax_i16_serial);
    check("reduce_minmax_u16_v128relaxed", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_v128relaxed,
          nk_reduce_minmax_u16_serial);
    check("reduce_minmax_i32_v128relaxed", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_v128relaxed,
          nk_reduce_minmax_i32_serial);
    check("reduce_minmax_u32_v128relaxed", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_v128relaxed,
          nk_reduce_minmax_u32_serial);
    check("reduce_minmax_i64_v128relaxed", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_v128relaxed,
          nk_reduce_minmax_i64_serial);
    check("reduce_minmax_u64_v128relaxed", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_v128relaxed,
          nk_reduce_minmax_u64_serial);
    check("reduce_minmax_e4m3_v128relaxed", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_v128relaxed,
          nk_reduce_minmax_e4m3_serial);
    check("reduce_minmax_e5m2_v128relaxed", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_v128relaxed,
          nk_reduce_minmax_e5m2_serial);
    check("reduce_minmax_e2m3_v128relaxed", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_v128relaxed,
          nk_reduce_minmax_e2m3_serial);
    check("reduce_minmax_e3m2_v128relaxed", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_v128relaxed,
          nk_reduce_minmax_e3m2_serial);
    check("reduce_minmax_bf16_v128relaxed", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_v128relaxed,
          nk_reduce_minmax_bf16_serial);
    check("reduce_minmax_f16_v128relaxed", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_v128relaxed,
          nk_reduce_minmax_f16_serial);
#endif

    check("reduce_moments_f32_serial", test_reduce_moments<f32_t>, nk_reduce_moments_f32_serial,
          nk_reduce_moments_f32_serial);
    check("reduce_moments_f64_serial", test_reduce_moments<f64_t>, nk_reduce_moments_f64_serial,
          nk_reduce_moments_f64_serial);
    check("reduce_moments_i8_serial", test_reduce_moments<i8_t>, nk_reduce_moments_i8_serial,
          nk_reduce_moments_i8_serial);
    check("reduce_moments_u8_serial", test_reduce_moments<u8_t>, nk_reduce_moments_u8_serial,
          nk_reduce_moments_u8_serial);
    check("reduce_moments_i16_serial", test_reduce_moments<i16_t>, nk_reduce_moments_i16_serial,
          nk_reduce_moments_i16_serial);
    check("reduce_moments_u16_serial", test_reduce_moments<u16_t>, nk_reduce_moments_u16_serial,
          nk_reduce_moments_u16_serial);
    check("reduce_moments_i32_serial", test_reduce_moments<i32_t>, nk_reduce_moments_i32_serial,
          nk_reduce_moments_i32_serial);
    check("reduce_moments_u32_serial", test_reduce_moments<u32_t>, nk_reduce_moments_u32_serial,
          nk_reduce_moments_u32_serial);
    check("reduce_moments_i64_serial", test_reduce_moments<i64_t>, nk_reduce_moments_i64_serial,
          nk_reduce_moments_i64_serial);
    check("reduce_moments_u64_serial", test_reduce_moments<u64_t>, nk_reduce_moments_u64_serial,
          nk_reduce_moments_u64_serial);
    check("reduce_moments_f16_serial", test_reduce_moments<f16_t>, nk_reduce_moments_f16_serial,
          nk_reduce_moments_f16_serial);
    check("reduce_moments_bf16_serial", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_serial,
          nk_reduce_moments_bf16_serial);
    check("reduce_moments_e4m3_serial", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_serial,
          nk_reduce_moments_e4m3_serial);
    check("reduce_moments_e5m2_serial", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_serial,
          nk_reduce_moments_e5m2_serial);
    check("reduce_moments_e2m3_serial", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_serial,
          nk_reduce_moments_e2m3_serial);
    check("reduce_moments_e3m2_serial", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_serial,
          nk_reduce_moments_e3m2_serial);
    check("reduce_moments_i4_serial", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4_serial,
          nk_reduce_moments_i4_serial);
    check("reduce_moments_u4_serial", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4_serial,
          nk_reduce_moments_u4_serial);
    check("reduce_moments_u1_serial", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1_serial,
          nk_reduce_moments_u1_serial);
    check("reduce_minmax_f32_serial", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_serial,
          nk_reduce_minmax_f32_serial);
    check("reduce_minmax_f64_serial", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_serial,
          nk_reduce_minmax_f64_serial);
    check("reduce_minmax_i8_serial", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_serial, nk_reduce_minmax_i8_serial);
    check("reduce_minmax_u8_serial", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_serial, nk_reduce_minmax_u8_serial);
    check("reduce_minmax_i16_serial", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_serial,
          nk_reduce_minmax_i16_serial);
    check("reduce_minmax_u16_serial", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_serial,
          nk_reduce_minmax_u16_serial);
    check("reduce_minmax_i32_serial", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_serial,
          nk_reduce_minmax_i32_serial);
    check("reduce_minmax_u32_serial", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_serial,
          nk_reduce_minmax_u32_serial);
    check("reduce_minmax_i64_serial", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_serial,
          nk_reduce_minmax_i64_serial);
    check("reduce_minmax_u64_serial", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_serial,
          nk_reduce_minmax_u64_serial);
    check("reduce_minmax_f16_serial", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_serial,
          nk_reduce_minmax_f16_serial);
    check("reduce_minmax_bf16_serial", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_serial,
          nk_reduce_minmax_bf16_serial);
    check("reduce_minmax_e4m3_serial", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_serial,
          nk_reduce_minmax_e4m3_serial);
    check("reduce_minmax_e5m2_serial", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_serial,
          nk_reduce_minmax_e5m2_serial);
    check("reduce_minmax_e2m3_serial", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_serial,
          nk_reduce_minmax_e2m3_serial);
    check("reduce_minmax_e3m2_serial", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_serial,
          nk_reduce_minmax_e3m2_serial);
    check("reduce_minmax_i4_serial", test_reduce_minmax<i4x2_t>, nk_reduce_minmax_i4_serial,
          nk_reduce_minmax_i4_serial);
    check("reduce_minmax_u4_serial", test_reduce_minmax<u4x2_t>, nk_reduce_minmax_u4_serial,
          nk_reduce_minmax_u4_serial);
    check("reduce_minmax_u1_serial", test_reduce_minmax<u1x8_t>, nk_reduce_minmax_u1_serial,
          nk_reduce_minmax_u1_serial);
#endif
}
