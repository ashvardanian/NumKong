/**
 *  @file test/reduce.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Reduction tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "reduce.hpp" // `test_reduce_moments`, `test_reduce_minmax`

namespace ashvardanian::numkong::test {

/** Known-value test for the vector-shaped reduction wrappers. */
inline error_stats_t test_vector_reductions() {
    // Known values, not `assert` — Release defines `NDEBUG`, which would delete the checks.
    error_stats_t stats(comparison_family_t::exact_k);
    nk::f32_t data[] = {nk::f32_t(1), nk::f32_t(2), nk::f32_t(3), nk::f32_t(4), nk::f32_t(5)};
    auto view = nk::vector_view<nk::f32_t>(data, std::size_t {5});

    auto [m, moments_status] = nk::moments(view);
    stats.expect_status(moments_status);
    stats.accumulate(m.sum, nk::f64_t(15));
    stats.accumulate(m.sumsq, nk::f64_t(55));

    auto [mm, minmax_status] = nk::minmax(view);
    stats.expect_status(minmax_status);
    stats.accumulate(mm.min_value, nk::f32_t(1));
    stats.accumulate(mm.max_value, nk::f32_t(5));
    stats.accumulate(static_cast<nk_size_t>(mm.min_index), static_cast<nk_size_t>(0));
    stats.accumulate(static_cast<nk_size_t>(mm.max_index), static_cast<nk_size_t>(4));

    // Every scalar reduction also takes the dispatch pair, here the default mask spelled out
    auto [sum, sum_status] = nk::sum(view, nk::default_capabilities(), nullptr);
    auto [minimum, min_status] = (nk::min)(view, nk::default_capabilities(), nullptr);
    auto [maximum, max_status] = (nk::max)(view, nk::default_capabilities(), nullptr);
    auto [minimum_index, argmin_status] = nk::argmin(view, nk::default_capabilities(), nullptr);
    auto [maximum_index, argmax_status] = nk::argmax(view, nk::default_capabilities(), nullptr);
    auto [serial_moments, serial_moments_status] = nk::moments(view, nk::default_capabilities(), nullptr);
    auto [serial_minmax, serial_minmax_status] = nk::minmax(view, nk::default_capabilities(), nullptr);
    for (nk::status_t status : {sum_status, min_status, max_status, argmin_status, argmax_status, serial_moments_status,
                                serial_minmax_status})
        stats.expect_status(status);
    stats.accumulate(sum, nk::f64_t(15));
    stats.accumulate(serial_moments.sumsq, nk::f64_t(55));
    stats.accumulate(minimum, nk::f32_t(1));
    stats.accumulate(maximum, serial_minmax.max_value);
    stats.accumulate(static_cast<nk_size_t>(minimum_index), static_cast<nk_size_t>(0));
    stats.accumulate(static_cast<nk_size_t>(maximum_index), static_cast<nk_size_t>(4));

    nk::u1x8_t bits[] = {nk::u1x8_t(0xFF), nk::u1x8_t(0x01)};
    auto bit_view = nk::vector_view<nk::u1x8_t>(std::span<nk::u1x8_t const>(bits));
    auto [count, popcount_status] = nk::popcount(bit_view, nk::default_capabilities(), nullptr);
    auto [any, any_status] = nk::any_set(bit_view, nk::default_capabilities(), nullptr);
    auto [none, none_status] = nk::none_set(bit_view, nk::default_capabilities(), nullptr);
    auto [all, all_status] = nk::all_set(bit_view, nk::default_capabilities(), nullptr);
    for (nk::status_t status : {popcount_status, any_status, none_status, all_status}) stats.expect_status(status);
    stats.accumulate(static_cast<nk_size_t>(count.raw_), static_cast<nk_size_t>(9));
    stats.expect_true(any && !none && !all, "set predicates of 9 bits out of 16");

    auto [grid, grid_status] = nk::tensor<nk::f32_t>::zeros({2, 3});
    stats.expect_status(grid_status);
    auto [grid_sum, grid_sum_status] = nk::sum<nk::f32_t>(grid.view(), nk::default_capabilities(), nullptr);
    stats.expect_status(grid_sum_status);
    stats.accumulate(grid_sum, nk::f64_t(0));
    stats.expect_true(nk::sum(view, 0, nullptr).status == nk::status_t::missing_kernel_k,
                      "a zero mask names no kernel");
    return stats;
}

void run_reduce_cases() {
    auto const serial = nk_cap_serial_k;

    run_kernel_case("test_vector_reductions", test_vector_reductions, serial);
    run_kernel_case("nk_reduce_moments_f32_serial", test_reduce_moments<f32_t>, serial, nk_reduce_moments_f32_serial);
    run_kernel_case("nk_reduce_moments_f64_serial", test_reduce_moments<f64_t>, serial, nk_reduce_moments_f64_serial);
    run_kernel_case("nk_reduce_moments_i8_serial", test_reduce_moments<i8_t>, serial, nk_reduce_moments_i8_serial);
    run_kernel_case("nk_reduce_moments_u8_serial", test_reduce_moments<u8_t>, serial, nk_reduce_moments_u8_serial);
    run_kernel_case("nk_reduce_moments_i16_serial", test_reduce_moments<i16_t>, serial, nk_reduce_moments_i16_serial);
    run_kernel_case("nk_reduce_moments_u16_serial", test_reduce_moments<u16_t>, serial, nk_reduce_moments_u16_serial);
    run_kernel_case("nk_reduce_moments_i32_serial", test_reduce_moments<i32_t>, serial, nk_reduce_moments_i32_serial);
    run_kernel_case("nk_reduce_moments_u32_serial", test_reduce_moments<u32_t>, serial, nk_reduce_moments_u32_serial);
    run_kernel_case("nk_reduce_moments_i64_serial", test_reduce_moments<i64_t>, serial, nk_reduce_moments_i64_serial);
    run_kernel_case("nk_reduce_moments_u64_serial", test_reduce_moments<u64_t>, serial, nk_reduce_moments_u64_serial);
    run_kernel_case("nk_reduce_moments_f16_serial", test_reduce_moments<f16_t>, serial, nk_reduce_moments_f16_serial);
    run_kernel_case("nk_reduce_moments_bf16_serial", test_reduce_moments<bf16_t>, serial,
                    nk_reduce_moments_bf16_serial);
    run_kernel_case("nk_reduce_moments_e4m3_serial", test_reduce_moments<e4m3_t>, serial,
                    nk_reduce_moments_e4m3_serial);
    run_kernel_case("nk_reduce_moments_e5m2_serial", test_reduce_moments<e5m2_t>, serial,
                    nk_reduce_moments_e5m2_serial);
    run_kernel_case("nk_reduce_moments_e2m3_serial", test_reduce_moments<e2m3_t>, serial,
                    nk_reduce_moments_e2m3_serial);
    run_kernel_case("nk_reduce_moments_e3m2_serial", test_reduce_moments<e3m2_t>, serial,
                    nk_reduce_moments_e3m2_serial);
    run_kernel_case("nk_reduce_moments_e2m1_serial", test_reduce_moments<e2m1x2_t>, serial,
                    nk_reduce_moments_e2m1_serial);
    run_kernel_case("nk_reduce_moments_i4_serial", test_reduce_moments<i4x2_t>, serial, nk_reduce_moments_i4_serial);
    run_kernel_case("nk_reduce_moments_u4_serial", test_reduce_moments<u4x2_t>, serial, nk_reduce_moments_u4_serial);
    run_kernel_case("nk_reduce_moments_u1_serial", test_reduce_moments<u1x8_t>, serial, nk_reduce_moments_u1_serial);
    run_kernel_case("nk_reduce_minmax_f32_serial", test_reduce_minmax<f32_t>, serial, nk_reduce_minmax_f32_serial);
    run_kernel_case("nk_reduce_minmax_f64_serial", test_reduce_minmax<f64_t>, serial, nk_reduce_minmax_f64_serial);
    run_kernel_case("nk_reduce_minmax_i8_serial", test_reduce_minmax<i8_t>, serial, nk_reduce_minmax_i8_serial);
    run_kernel_case("nk_reduce_minmax_u8_serial", test_reduce_minmax<u8_t>, serial, nk_reduce_minmax_u8_serial);
    run_kernel_case("nk_reduce_minmax_i16_serial", test_reduce_minmax<i16_t>, serial, nk_reduce_minmax_i16_serial);
    run_kernel_case("nk_reduce_minmax_u16_serial", test_reduce_minmax<u16_t>, serial, nk_reduce_minmax_u16_serial);
    run_kernel_case("nk_reduce_minmax_i32_serial", test_reduce_minmax<i32_t>, serial, nk_reduce_minmax_i32_serial);
    run_kernel_case("nk_reduce_minmax_u32_serial", test_reduce_minmax<u32_t>, serial, nk_reduce_minmax_u32_serial);
    run_kernel_case("nk_reduce_minmax_i64_serial", test_reduce_minmax<i64_t>, serial, nk_reduce_minmax_i64_serial);
    run_kernel_case("nk_reduce_minmax_u64_serial", test_reduce_minmax<u64_t>, serial, nk_reduce_minmax_u64_serial);
    run_kernel_case("nk_reduce_minmax_f16_serial", test_reduce_minmax<f16_t>, serial, nk_reduce_minmax_f16_serial);
    run_kernel_case("nk_reduce_minmax_bf16_serial", test_reduce_minmax<bf16_t>, serial, nk_reduce_minmax_bf16_serial);
    run_kernel_case("nk_reduce_minmax_e4m3_serial", test_reduce_minmax<e4m3_t>, serial, nk_reduce_minmax_e4m3_serial);
    run_kernel_case("nk_reduce_minmax_e5m2_serial", test_reduce_minmax<e5m2_t>, serial, nk_reduce_minmax_e5m2_serial);
    run_kernel_case("nk_reduce_minmax_e2m3_serial", test_reduce_minmax<e2m3_t>, serial, nk_reduce_minmax_e2m3_serial);
    run_kernel_case("nk_reduce_minmax_e3m2_serial", test_reduce_minmax<e3m2_t>, serial, nk_reduce_minmax_e3m2_serial);
    run_kernel_case("nk_reduce_minmax_i4_serial", test_reduce_minmax<i4x2_t>, serial, nk_reduce_minmax_i4_serial);
    run_kernel_case("nk_reduce_minmax_u4_serial", test_reduce_minmax<u4x2_t>, serial, nk_reduce_minmax_u4_serial);
    run_kernel_case("nk_reduce_minmax_u1_serial", test_reduce_minmax<u1x8_t>, serial, nk_reduce_minmax_u1_serial);

    run_kernel_case("nk_reduce_moments_f32_best", test_reduce_moments<f32_t>, serial,
                    cpu_best<nk_reduce_moments_f32_best>);
    run_kernel_case("nk_reduce_moments_f64_best", test_reduce_moments<f64_t>, serial,
                    cpu_best<nk_reduce_moments_f64_best>);
    run_kernel_case("nk_reduce_moments_i8_best", test_reduce_moments<i8_t>, serial,
                    cpu_best<nk_reduce_moments_i8_best>);
    run_kernel_case("nk_reduce_moments_u8_best", test_reduce_moments<u8_t>, serial,
                    cpu_best<nk_reduce_moments_u8_best>);
    run_kernel_case("nk_reduce_moments_i16_best", test_reduce_moments<i16_t>, serial,
                    cpu_best<nk_reduce_moments_i16_best>);
    run_kernel_case("nk_reduce_moments_u16_best", test_reduce_moments<u16_t>, serial,
                    cpu_best<nk_reduce_moments_u16_best>);
    run_kernel_case("nk_reduce_moments_i32_best", test_reduce_moments<i32_t>, serial,
                    cpu_best<nk_reduce_moments_i32_best>);
    run_kernel_case("nk_reduce_moments_u32_best", test_reduce_moments<u32_t>, serial,
                    cpu_best<nk_reduce_moments_u32_best>);
    run_kernel_case("nk_reduce_moments_i64_best", test_reduce_moments<i64_t>, serial,
                    cpu_best<nk_reduce_moments_i64_best>);
    run_kernel_case("nk_reduce_moments_u64_best", test_reduce_moments<u64_t>, serial,
                    cpu_best<nk_reduce_moments_u64_best>);
    run_kernel_case("nk_reduce_moments_f16_best", test_reduce_moments<f16_t>, serial,
                    cpu_best<nk_reduce_moments_f16_best>);
    run_kernel_case("nk_reduce_moments_bf16_best", test_reduce_moments<bf16_t>, serial,
                    cpu_best<nk_reduce_moments_bf16_best>);
    run_kernel_case("nk_reduce_moments_e4m3_best", test_reduce_moments<e4m3_t>, serial,
                    cpu_best<nk_reduce_moments_e4m3_best>);
    run_kernel_case("nk_reduce_moments_e5m2_best", test_reduce_moments<e5m2_t>, serial,
                    cpu_best<nk_reduce_moments_e5m2_best>);
    run_kernel_case("nk_reduce_moments_e2m3_best", test_reduce_moments<e2m3_t>, serial,
                    cpu_best<nk_reduce_moments_e2m3_best>);
    run_kernel_case("nk_reduce_moments_e3m2_best", test_reduce_moments<e3m2_t>, serial,
                    cpu_best<nk_reduce_moments_e3m2_best>);
    run_kernel_case("nk_reduce_moments_i4_best", test_reduce_moments<i4x2_t>, serial,
                    cpu_best<nk_reduce_moments_i4_best>);
    run_kernel_case("nk_reduce_moments_u4_best", test_reduce_moments<u4x2_t>, serial,
                    cpu_best<nk_reduce_moments_u4_best>);
    run_kernel_case("nk_reduce_moments_u1_best", test_reduce_moments<u1x8_t>, serial,
                    cpu_best<nk_reduce_moments_u1_best>);
    run_kernel_case("nk_reduce_minmax_f32_best", test_reduce_minmax<f32_t>, serial,
                    cpu_best<nk_reduce_minmax_f32_best>);
    run_kernel_case("nk_reduce_minmax_f64_best", test_reduce_minmax<f64_t>, serial,
                    cpu_best<nk_reduce_minmax_f64_best>);
    run_kernel_case("nk_reduce_minmax_i8_best", test_reduce_minmax<i8_t>, serial, cpu_best<nk_reduce_minmax_i8_best>);
    run_kernel_case("nk_reduce_minmax_u8_best", test_reduce_minmax<u8_t>, serial, cpu_best<nk_reduce_minmax_u8_best>);
    run_kernel_case("nk_reduce_minmax_i16_best", test_reduce_minmax<i16_t>, serial,
                    cpu_best<nk_reduce_minmax_i16_best>);
    run_kernel_case("nk_reduce_minmax_u16_best", test_reduce_minmax<u16_t>, serial,
                    cpu_best<nk_reduce_minmax_u16_best>);
    run_kernel_case("nk_reduce_minmax_i32_best", test_reduce_minmax<i32_t>, serial,
                    cpu_best<nk_reduce_minmax_i32_best>);
    run_kernel_case("nk_reduce_minmax_u32_best", test_reduce_minmax<u32_t>, serial,
                    cpu_best<nk_reduce_minmax_u32_best>);
    run_kernel_case("nk_reduce_minmax_i64_best", test_reduce_minmax<i64_t>, serial,
                    cpu_best<nk_reduce_minmax_i64_best>);
    run_kernel_case("nk_reduce_minmax_u64_best", test_reduce_minmax<u64_t>, serial,
                    cpu_best<nk_reduce_minmax_u64_best>);
    run_kernel_case("nk_reduce_minmax_f16_best", test_reduce_minmax<f16_t>, serial,
                    cpu_best<nk_reduce_minmax_f16_best>);
    run_kernel_case("nk_reduce_minmax_bf16_best", test_reduce_minmax<bf16_t>, serial,
                    cpu_best<nk_reduce_minmax_bf16_best>);
    run_kernel_case("nk_reduce_minmax_e4m3_best", test_reduce_minmax<e4m3_t>, serial,
                    cpu_best<nk_reduce_minmax_e4m3_best>);
    run_kernel_case("nk_reduce_minmax_e5m2_best", test_reduce_minmax<e5m2_t>, serial,
                    cpu_best<nk_reduce_minmax_e5m2_best>);
    run_kernel_case("nk_reduce_minmax_e2m3_best", test_reduce_minmax<e2m3_t>, serial,
                    cpu_best<nk_reduce_minmax_e2m3_best>);
    run_kernel_case("nk_reduce_minmax_e3m2_best", test_reduce_minmax<e3m2_t>, serial,
                    cpu_best<nk_reduce_minmax_e3m2_best>);
    run_kernel_case("nk_reduce_minmax_i4_best", test_reduce_minmax<i4x2_t>, serial, cpu_best<nk_reduce_minmax_i4_best>);
    run_kernel_case("nk_reduce_minmax_u4_best", test_reduce_minmax<u4x2_t>, serial, cpu_best<nk_reduce_minmax_u4_best>);
    run_kernel_case("nk_reduce_minmax_u1_best", test_reduce_minmax<u1x8_t>, serial, cpu_best<nk_reduce_minmax_u1_best>);

#if NUMKONG_TARGET_NEON
    auto const neon = nk_cap_neon_k;
    run_kernel_case("nk_reduce_moments_bf16_neon", test_reduce_moments<bf16_t>, neon, nk_reduce_moments_bf16_neon);
    run_kernel_case("nk_reduce_moments_f32_neon", test_reduce_moments<f32_t>, neon, nk_reduce_moments_f32_neon);
    run_kernel_case("nk_reduce_moments_f64_neon", test_reduce_moments<f64_t>, neon, nk_reduce_moments_f64_neon);
    run_kernel_case("nk_reduce_moments_f16_neon", test_reduce_moments<f16_t>, neon, nk_reduce_moments_f16_neon);
    run_kernel_case("nk_reduce_moments_i8_neon", test_reduce_moments<i8_t>, neon, nk_reduce_moments_i8_neon);
    run_kernel_case("nk_reduce_moments_u8_neon", test_reduce_moments<u8_t>, neon, nk_reduce_moments_u8_neon);
    run_kernel_case("nk_reduce_moments_i16_neon", test_reduce_moments<i16_t>, neon, nk_reduce_moments_i16_neon);
    run_kernel_case("nk_reduce_moments_u16_neon", test_reduce_moments<u16_t>, neon, nk_reduce_moments_u16_neon);
    run_kernel_case("nk_reduce_moments_i32_neon", test_reduce_moments<i32_t>, neon, nk_reduce_moments_i32_neon);
    run_kernel_case("nk_reduce_moments_u32_neon", test_reduce_moments<u32_t>, neon, nk_reduce_moments_u32_neon);
    run_kernel_case("nk_reduce_moments_i64_neon", test_reduce_moments<i64_t>, neon, nk_reduce_moments_i64_neon);
    run_kernel_case("nk_reduce_moments_u64_neon", test_reduce_moments<u64_t>, neon, nk_reduce_moments_u64_neon);
    run_kernel_case("nk_reduce_moments_e4m3_neon", test_reduce_moments<e4m3_t>, neon, nk_reduce_moments_e4m3_neon);
    run_kernel_case("nk_reduce_moments_e5m2_neon", test_reduce_moments<e5m2_t>, neon, nk_reduce_moments_e5m2_neon);
    run_kernel_case("nk_reduce_moments_e2m3_neon", test_reduce_moments<e2m3_t>, neon, nk_reduce_moments_e2m3_neon);
    run_kernel_case("nk_reduce_moments_e3m2_neon", test_reduce_moments<e3m2_t>, neon, nk_reduce_moments_e3m2_neon);
    run_kernel_case("nk_reduce_moments_u1_neon", test_reduce_moments<u1x8_t>, neon, nk_reduce_moments_u1_neon);
    run_kernel_case("nk_reduce_minmax_f32_neon", test_reduce_minmax<f32_t>, neon, nk_reduce_minmax_f32_neon);
    run_kernel_case("nk_reduce_minmax_f64_neon", test_reduce_minmax<f64_t>, neon, nk_reduce_minmax_f64_neon);
    run_kernel_case("nk_reduce_minmax_i8_neon", test_reduce_minmax<i8_t>, neon, nk_reduce_minmax_i8_neon);
    run_kernel_case("nk_reduce_minmax_u8_neon", test_reduce_minmax<u8_t>, neon, nk_reduce_minmax_u8_neon);
    run_kernel_case("nk_reduce_minmax_i16_neon", test_reduce_minmax<i16_t>, neon, nk_reduce_minmax_i16_neon);
    run_kernel_case("nk_reduce_minmax_u16_neon", test_reduce_minmax<u16_t>, neon, nk_reduce_minmax_u16_neon);
    run_kernel_case("nk_reduce_minmax_i32_neon", test_reduce_minmax<i32_t>, neon, nk_reduce_minmax_i32_neon);
    run_kernel_case("nk_reduce_minmax_u32_neon", test_reduce_minmax<u32_t>, neon, nk_reduce_minmax_u32_neon);
    run_kernel_case("nk_reduce_minmax_i64_neon", test_reduce_minmax<i64_t>, neon, nk_reduce_minmax_i64_neon);
    run_kernel_case("nk_reduce_minmax_u64_neon", test_reduce_minmax<u64_t>, neon, nk_reduce_minmax_u64_neon);
    run_kernel_case("nk_reduce_minmax_f16_neon", test_reduce_minmax<f16_t>, neon, nk_reduce_minmax_f16_neon);
    run_kernel_case("nk_reduce_minmax_bf16_neon", test_reduce_minmax<bf16_t>, neon, nk_reduce_minmax_bf16_neon);
    run_kernel_case("nk_reduce_minmax_e4m3_neon", test_reduce_minmax<e4m3_t>, neon, nk_reduce_minmax_e4m3_neon);
    run_kernel_case("nk_reduce_minmax_e5m2_neon", test_reduce_minmax<e5m2_t>, neon, nk_reduce_minmax_e5m2_neon);
    run_kernel_case("nk_reduce_minmax_e2m3_neon", test_reduce_minmax<e2m3_t>, neon, nk_reduce_minmax_e2m3_neon);
    run_kernel_case("nk_reduce_minmax_e3m2_neon", test_reduce_minmax<e3m2_t>, neon, nk_reduce_minmax_e3m2_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
    auto const neonbfdot = nk_cap_neonbfdot_k;
    run_kernel_case("nk_reduce_moments_bf16_neonbfdot", test_reduce_moments<bf16_t>, neonbfdot,
                    nk_reduce_moments_bf16_neonbfdot);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONSDOT
    auto const neonsdot = nk_cap_neonsdot_k;
    run_kernel_case("nk_reduce_moments_i8_neonsdot", test_reduce_moments<i8_t>, neonsdot,
                    nk_reduce_moments_i8_neonsdot);
    run_kernel_case("nk_reduce_moments_u8_neonsdot", test_reduce_moments<u8_t>, neonsdot,
                    nk_reduce_moments_u8_neonsdot);
    run_kernel_case("nk_reduce_moments_i4_neonsdot", test_reduce_moments<i4x2_t>, neonsdot,
                    nk_reduce_moments_i4_neonsdot);
    run_kernel_case("nk_reduce_moments_u4_neonsdot", test_reduce_moments<u4x2_t>, neonsdot,
                    nk_reduce_moments_u4_neonsdot);
    run_kernel_case("nk_reduce_moments_e2m3_neonsdot", test_reduce_moments<e2m3_t>, neonsdot,
                    nk_reduce_moments_e2m3_neonsdot);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_NEONFHM
    auto const neonfhm = nk_cap_neonfhm_k;
    run_kernel_case("nk_reduce_moments_e4m3_neonfhm", test_reduce_moments<e4m3_t>, neonfhm,
                    nk_reduce_moments_e4m3_neonfhm);
    run_kernel_case("nk_reduce_moments_e5m2_neonfhm", test_reduce_moments<e5m2_t>, neonfhm,
                    nk_reduce_moments_e5m2_neonfhm);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_HASWELL
    auto const haswell = nk_cap_haswell_k;
    run_kernel_case("nk_reduce_moments_f32_haswell", test_reduce_moments<f32_t>, haswell,
                    nk_reduce_moments_f32_haswell);
    run_kernel_case("nk_reduce_moments_f64_haswell", test_reduce_moments<f64_t>, haswell,
                    nk_reduce_moments_f64_haswell);
    run_kernel_case("nk_reduce_moments_i8_haswell", test_reduce_moments<i8_t>, haswell, nk_reduce_moments_i8_haswell);
    run_kernel_case("nk_reduce_moments_u8_haswell", test_reduce_moments<u8_t>, haswell, nk_reduce_moments_u8_haswell);
    run_kernel_case("nk_reduce_moments_i16_haswell", test_reduce_moments<i16_t>, haswell,
                    nk_reduce_moments_i16_haswell);
    run_kernel_case("nk_reduce_moments_u16_haswell", test_reduce_moments<u16_t>, haswell,
                    nk_reduce_moments_u16_haswell);
    run_kernel_case("nk_reduce_moments_i32_haswell", test_reduce_moments<i32_t>, haswell,
                    nk_reduce_moments_i32_haswell);
    run_kernel_case("nk_reduce_moments_u32_haswell", test_reduce_moments<u32_t>, haswell,
                    nk_reduce_moments_u32_haswell);
    run_kernel_case("nk_reduce_moments_i64_haswell", test_reduce_moments<i64_t>, haswell,
                    nk_reduce_moments_i64_haswell);
    run_kernel_case("nk_reduce_moments_u64_haswell", test_reduce_moments<u64_t>, haswell,
                    nk_reduce_moments_u64_haswell);
    run_kernel_case("nk_reduce_moments_e4m3_haswell", test_reduce_moments<e4m3_t>, haswell,
                    nk_reduce_moments_e4m3_haswell);
    run_kernel_case("nk_reduce_moments_e5m2_haswell", test_reduce_moments<e5m2_t>, haswell,
                    nk_reduce_moments_e5m2_haswell);
    run_kernel_case("nk_reduce_moments_e2m3_haswell", test_reduce_moments<e2m3_t>, haswell,
                    nk_reduce_moments_e2m3_haswell);
    run_kernel_case("nk_reduce_moments_e3m2_haswell", test_reduce_moments<e3m2_t>, haswell,
                    nk_reduce_moments_e3m2_haswell);
    run_kernel_case("nk_reduce_moments_i4_haswell", test_reduce_moments<i4x2_t>, haswell, nk_reduce_moments_i4_haswell);
    run_kernel_case("nk_reduce_moments_u4_haswell", test_reduce_moments<u4x2_t>, haswell, nk_reduce_moments_u4_haswell);
    run_kernel_case("nk_reduce_moments_u1_haswell", test_reduce_moments<u1x8_t>, haswell, nk_reduce_moments_u1_haswell);
    run_kernel_case("nk_reduce_moments_bf16_haswell", test_reduce_moments<bf16_t>, haswell,
                    nk_reduce_moments_bf16_haswell);
    run_kernel_case("nk_reduce_moments_f16_haswell", test_reduce_moments<f16_t>, haswell,
                    nk_reduce_moments_f16_haswell);
    run_kernel_case("nk_reduce_minmax_f32_haswell", test_reduce_minmax<f32_t>, haswell, nk_reduce_minmax_f32_haswell);
    run_kernel_case("nk_reduce_minmax_f64_haswell", test_reduce_minmax<f64_t>, haswell, nk_reduce_minmax_f64_haswell);
    run_kernel_case("nk_reduce_minmax_i8_haswell", test_reduce_minmax<i8_t>, haswell, nk_reduce_minmax_i8_haswell);
    run_kernel_case("nk_reduce_minmax_u8_haswell", test_reduce_minmax<u8_t>, haswell, nk_reduce_minmax_u8_haswell);
    run_kernel_case("nk_reduce_minmax_i16_haswell", test_reduce_minmax<i16_t>, haswell, nk_reduce_minmax_i16_haswell);
    run_kernel_case("nk_reduce_minmax_u16_haswell", test_reduce_minmax<u16_t>, haswell, nk_reduce_minmax_u16_haswell);
    run_kernel_case("nk_reduce_minmax_i32_haswell", test_reduce_minmax<i32_t>, haswell, nk_reduce_minmax_i32_haswell);
    run_kernel_case("nk_reduce_minmax_u32_haswell", test_reduce_minmax<u32_t>, haswell, nk_reduce_minmax_u32_haswell);
    run_kernel_case("nk_reduce_minmax_i64_haswell", test_reduce_minmax<i64_t>, haswell, nk_reduce_minmax_i64_haswell);
    run_kernel_case("nk_reduce_minmax_u64_haswell", test_reduce_minmax<u64_t>, haswell, nk_reduce_minmax_u64_haswell);
    run_kernel_case("nk_reduce_minmax_e4m3_haswell", test_reduce_minmax<e4m3_t>, haswell,
                    nk_reduce_minmax_e4m3_haswell);
    run_kernel_case("nk_reduce_minmax_e5m2_haswell", test_reduce_minmax<e5m2_t>, haswell,
                    nk_reduce_minmax_e5m2_haswell);
    run_kernel_case("nk_reduce_minmax_e2m3_haswell", test_reduce_minmax<e2m3_t>, haswell,
                    nk_reduce_minmax_e2m3_haswell);
    run_kernel_case("nk_reduce_minmax_e3m2_haswell", test_reduce_minmax<e3m2_t>, haswell,
                    nk_reduce_minmax_e3m2_haswell);
    run_kernel_case("nk_reduce_minmax_bf16_haswell", test_reduce_minmax<bf16_t>, haswell,
                    nk_reduce_minmax_bf16_haswell);
    run_kernel_case("nk_reduce_minmax_f16_haswell", test_reduce_minmax<f16_t>, haswell, nk_reduce_minmax_f16_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    auto const skylake = nk_cap_skylake_k;
    run_kernel_case("nk_reduce_moments_f32_skylake", test_reduce_moments<f32_t>, skylake,
                    nk_reduce_moments_f32_skylake);
    run_kernel_case("nk_reduce_moments_f64_skylake", test_reduce_moments<f64_t>, skylake,
                    nk_reduce_moments_f64_skylake);
    run_kernel_case("nk_reduce_moments_i8_skylake", test_reduce_moments<i8_t>, skylake, nk_reduce_moments_i8_skylake);
    run_kernel_case("nk_reduce_moments_u8_skylake", test_reduce_moments<u8_t>, skylake, nk_reduce_moments_u8_skylake);
    run_kernel_case("nk_reduce_moments_i16_skylake", test_reduce_moments<i16_t>, skylake,
                    nk_reduce_moments_i16_skylake);
    run_kernel_case("nk_reduce_moments_u16_skylake", test_reduce_moments<u16_t>, skylake,
                    nk_reduce_moments_u16_skylake);
    run_kernel_case("nk_reduce_moments_i32_skylake", test_reduce_moments<i32_t>, skylake,
                    nk_reduce_moments_i32_skylake);
    run_kernel_case("nk_reduce_moments_u32_skylake", test_reduce_moments<u32_t>, skylake,
                    nk_reduce_moments_u32_skylake);
    run_kernel_case("nk_reduce_moments_i64_skylake", test_reduce_moments<i64_t>, skylake,
                    nk_reduce_moments_i64_skylake);
    run_kernel_case("nk_reduce_moments_u64_skylake", test_reduce_moments<u64_t>, skylake,
                    nk_reduce_moments_u64_skylake);
    run_kernel_case("nk_reduce_moments_e4m3_skylake", test_reduce_moments<e4m3_t>, skylake,
                    nk_reduce_moments_e4m3_skylake);
    run_kernel_case("nk_reduce_moments_e5m2_skylake", test_reduce_moments<e5m2_t>, skylake,
                    nk_reduce_moments_e5m2_skylake);
    run_kernel_case("nk_reduce_moments_e2m3_skylake", test_reduce_moments<e2m3_t>, skylake,
                    nk_reduce_moments_e2m3_skylake);
    run_kernel_case("nk_reduce_moments_e3m2_skylake", test_reduce_moments<e3m2_t>, skylake,
                    nk_reduce_moments_e3m2_skylake);
    run_kernel_case("nk_reduce_moments_i4_skylake", test_reduce_moments<i4x2_t>, skylake, nk_reduce_moments_i4_skylake);
    run_kernel_case("nk_reduce_moments_u4_skylake", test_reduce_moments<u4x2_t>, skylake, nk_reduce_moments_u4_skylake);
    run_kernel_case("nk_reduce_moments_u1_skylake", test_reduce_moments<u1x8_t>, skylake, nk_reduce_moments_u1_skylake);
    run_kernel_case("nk_reduce_minmax_f32_skylake", test_reduce_minmax<f32_t>, skylake, nk_reduce_minmax_f32_skylake);
    run_kernel_case("nk_reduce_minmax_f64_skylake", test_reduce_minmax<f64_t>, skylake, nk_reduce_minmax_f64_skylake);
    run_kernel_case("nk_reduce_minmax_i8_skylake", test_reduce_minmax<i8_t>, skylake, nk_reduce_minmax_i8_skylake);
    run_kernel_case("nk_reduce_minmax_u8_skylake", test_reduce_minmax<u8_t>, skylake, nk_reduce_minmax_u8_skylake);
    run_kernel_case("nk_reduce_minmax_i16_skylake", test_reduce_minmax<i16_t>, skylake, nk_reduce_minmax_i16_skylake);
    run_kernel_case("nk_reduce_minmax_u16_skylake", test_reduce_minmax<u16_t>, skylake, nk_reduce_minmax_u16_skylake);
    run_kernel_case("nk_reduce_minmax_i32_skylake", test_reduce_minmax<i32_t>, skylake, nk_reduce_minmax_i32_skylake);
    run_kernel_case("nk_reduce_minmax_u32_skylake", test_reduce_minmax<u32_t>, skylake, nk_reduce_minmax_u32_skylake);
    run_kernel_case("nk_reduce_minmax_i64_skylake", test_reduce_minmax<i64_t>, skylake, nk_reduce_minmax_i64_skylake);
    run_kernel_case("nk_reduce_minmax_u64_skylake", test_reduce_minmax<u64_t>, skylake, nk_reduce_minmax_u64_skylake);
    run_kernel_case("nk_reduce_minmax_e4m3_skylake", test_reduce_minmax<e4m3_t>, skylake,
                    nk_reduce_minmax_e4m3_skylake);
    run_kernel_case("nk_reduce_minmax_e5m2_skylake", test_reduce_minmax<e5m2_t>, skylake,
                    nk_reduce_minmax_e5m2_skylake);
    run_kernel_case("nk_reduce_minmax_e2m3_skylake", test_reduce_minmax<e2m3_t>, skylake,
                    nk_reduce_minmax_e2m3_skylake);
    run_kernel_case("nk_reduce_minmax_e3m2_skylake", test_reduce_minmax<e3m2_t>, skylake,
                    nk_reduce_minmax_e3m2_skylake);
    run_kernel_case("nk_reduce_moments_bf16_skylake", test_reduce_moments<bf16_t>, skylake,
                    nk_reduce_moments_bf16_skylake);
    run_kernel_case("nk_reduce_minmax_bf16_skylake", test_reduce_minmax<bf16_t>, skylake,
                    nk_reduce_minmax_bf16_skylake);
    run_kernel_case("nk_reduce_moments_f16_skylake", test_reduce_moments<f16_t>, skylake,
                    nk_reduce_moments_f16_skylake);
    run_kernel_case("nk_reduce_minmax_f16_skylake", test_reduce_minmax<f16_t>, skylake, nk_reduce_minmax_f16_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
    auto const icelake = nk_cap_icelake_k;
    run_kernel_case("nk_reduce_moments_i8_icelake", test_reduce_moments<i8_t>, icelake, nk_reduce_moments_i8_icelake);
    run_kernel_case("nk_reduce_moments_u8_icelake", test_reduce_moments<u8_t>, icelake, nk_reduce_moments_u8_icelake);
    run_kernel_case("nk_reduce_moments_i16_icelake", test_reduce_moments<i16_t>, icelake,
                    nk_reduce_moments_i16_icelake);
    run_kernel_case("nk_reduce_moments_e2m3_icelake", test_reduce_moments<e2m3_t>, icelake,
                    nk_reduce_moments_e2m3_icelake);
    run_kernel_case("nk_reduce_moments_e3m2_icelake", test_reduce_moments<e3m2_t>, icelake,
                    nk_reduce_moments_e3m2_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
    auto const genoa = nk_cap_genoa_k;
    run_kernel_case("nk_reduce_moments_bf16_genoa", test_reduce_moments<bf16_t>, genoa, nk_reduce_moments_bf16_genoa);
    run_kernel_case("nk_reduce_moments_e4m3_genoa", test_reduce_moments<e4m3_t>, genoa, nk_reduce_moments_e4m3_genoa);
    run_kernel_case("nk_reduce_moments_e5m2_genoa", test_reduce_moments<e5m2_t>, genoa, nk_reduce_moments_e5m2_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_ALDER
    auto const alder = nk_cap_alder_k;
    run_kernel_case("nk_reduce_moments_u8_alder", test_reduce_moments<u8_t>, alder, nk_reduce_moments_u8_alder);
    run_kernel_case("nk_reduce_moments_i16_alder", test_reduce_moments<i16_t>, alder, nk_reduce_moments_i16_alder);
    run_kernel_case("nk_reduce_moments_u16_alder", test_reduce_moments<u16_t>, alder, nk_reduce_moments_u16_alder);
    run_kernel_case("nk_reduce_moments_e3m2_alder", test_reduce_moments<e3m2_t>, alder, nk_reduce_moments_e3m2_alder);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_SIERRA
    auto const sierra = nk_cap_sierra_k;
    run_kernel_case("nk_reduce_moments_i8_sierra", test_reduce_moments<i8_t>, sierra, nk_reduce_moments_i8_sierra);
    run_kernel_case("nk_reduce_moments_u8_sierra", test_reduce_moments<u8_t>, sierra, nk_reduce_moments_u8_sierra);
    run_kernel_case("nk_reduce_moments_e2m3_sierra", test_reduce_moments<e2m3_t>, sierra,
                    nk_reduce_moments_e2m3_sierra);
#endif // NUMKONG_TARGET_SIERRA

#if NUMKONG_TARGET_RVV
    auto const rvv = nk_cap_rvv_k;
    run_kernel_case("nk_reduce_moments_f32_rvv", test_reduce_moments<f32_t>, rvv, nk_reduce_moments_f32_rvv);
    run_kernel_case("nk_reduce_moments_f64_rvv", test_reduce_moments<f64_t>, rvv, nk_reduce_moments_f64_rvv);
    run_kernel_case("nk_reduce_moments_i8_rvv", test_reduce_moments<i8_t>, rvv, nk_reduce_moments_i8_rvv);
    run_kernel_case("nk_reduce_moments_u8_rvv", test_reduce_moments<u8_t>, rvv, nk_reduce_moments_u8_rvv);
    run_kernel_case("nk_reduce_moments_i16_rvv", test_reduce_moments<i16_t>, rvv, nk_reduce_moments_i16_rvv);
    run_kernel_case("nk_reduce_moments_u16_rvv", test_reduce_moments<u16_t>, rvv, nk_reduce_moments_u16_rvv);
    run_kernel_case("nk_reduce_moments_i32_rvv", test_reduce_moments<i32_t>, rvv, nk_reduce_moments_i32_rvv);
    run_kernel_case("nk_reduce_moments_u32_rvv", test_reduce_moments<u32_t>, rvv, nk_reduce_moments_u32_rvv);
    run_kernel_case("nk_reduce_moments_i64_rvv", test_reduce_moments<i64_t>, rvv, nk_reduce_moments_i64_rvv);
    run_kernel_case("nk_reduce_moments_u64_rvv", test_reduce_moments<u64_t>, rvv, nk_reduce_moments_u64_rvv);
    run_kernel_case("nk_reduce_moments_bf16_rvv", test_reduce_moments<bf16_t>, rvv, nk_reduce_moments_bf16_rvv);
    run_kernel_case("nk_reduce_moments_f16_rvv", test_reduce_moments<f16_t>, rvv, nk_reduce_moments_f16_rvv);
    run_kernel_case("nk_reduce_moments_e4m3_rvv", test_reduce_moments<e4m3_t>, rvv, nk_reduce_moments_e4m3_rvv);
    run_kernel_case("nk_reduce_moments_e5m2_rvv", test_reduce_moments<e5m2_t>, rvv, nk_reduce_moments_e5m2_rvv);
    run_kernel_case("nk_reduce_moments_e2m3_rvv", test_reduce_moments<e2m3_t>, rvv, nk_reduce_moments_e2m3_rvv);
    run_kernel_case("nk_reduce_moments_e3m2_rvv", test_reduce_moments<e3m2_t>, rvv, nk_reduce_moments_e3m2_rvv);
    run_kernel_case("nk_reduce_minmax_f32_rvv", test_reduce_minmax<f32_t>, rvv, nk_reduce_minmax_f32_rvv);
    run_kernel_case("nk_reduce_minmax_f64_rvv", test_reduce_minmax<f64_t>, rvv, nk_reduce_minmax_f64_rvv);
    run_kernel_case("nk_reduce_minmax_i8_rvv", test_reduce_minmax<i8_t>, rvv, nk_reduce_minmax_i8_rvv);
    run_kernel_case("nk_reduce_minmax_u8_rvv", test_reduce_minmax<u8_t>, rvv, nk_reduce_minmax_u8_rvv);
    run_kernel_case("nk_reduce_minmax_i16_rvv", test_reduce_minmax<i16_t>, rvv, nk_reduce_minmax_i16_rvv);
    run_kernel_case("nk_reduce_minmax_u16_rvv", test_reduce_minmax<u16_t>, rvv, nk_reduce_minmax_u16_rvv);
    run_kernel_case("nk_reduce_minmax_i32_rvv", test_reduce_minmax<i32_t>, rvv, nk_reduce_minmax_i32_rvv);
    run_kernel_case("nk_reduce_minmax_u32_rvv", test_reduce_minmax<u32_t>, rvv, nk_reduce_minmax_u32_rvv);
    run_kernel_case("nk_reduce_minmax_i64_rvv", test_reduce_minmax<i64_t>, rvv, nk_reduce_minmax_i64_rvv);
    run_kernel_case("nk_reduce_minmax_u64_rvv", test_reduce_minmax<u64_t>, rvv, nk_reduce_minmax_u64_rvv);
    run_kernel_case("nk_reduce_minmax_bf16_rvv", test_reduce_minmax<bf16_t>, rvv, nk_reduce_minmax_bf16_rvv);
    run_kernel_case("nk_reduce_minmax_f16_rvv", test_reduce_minmax<f16_t>, rvv, nk_reduce_minmax_f16_rvv);
    run_kernel_case("nk_reduce_minmax_e4m3_rvv", test_reduce_minmax<e4m3_t>, rvv, nk_reduce_minmax_e4m3_rvv);
    run_kernel_case("nk_reduce_minmax_e5m2_rvv", test_reduce_minmax<e5m2_t>, rvv, nk_reduce_minmax_e5m2_rvv);
    run_kernel_case("nk_reduce_minmax_e2m3_rvv", test_reduce_minmax<e2m3_t>, rvv, nk_reduce_minmax_e2m3_rvv);
    run_kernel_case("nk_reduce_minmax_e3m2_rvv", test_reduce_minmax<e3m2_t>, rvv, nk_reduce_minmax_e3m2_rvv);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
    auto const v128relaxed = nk_cap_v128relaxed_k;
    run_kernel_case("nk_reduce_moments_f32_v128relaxed", test_reduce_moments<f32_t>, v128relaxed,
                    nk_reduce_moments_f32_v128relaxed);
    run_kernel_case("nk_reduce_moments_i64_v128relaxed", test_reduce_moments<i64_t>, v128relaxed,
                    nk_reduce_moments_i64_v128relaxed);
    run_kernel_case("nk_reduce_moments_u64_v128relaxed", test_reduce_moments<u64_t>, v128relaxed,
                    nk_reduce_moments_u64_v128relaxed);
    run_kernel_case("nk_reduce_moments_e4m3_v128relaxed", test_reduce_moments<e4m3_t>, v128relaxed,
                    nk_reduce_moments_e4m3_v128relaxed);
    run_kernel_case("nk_reduce_moments_e5m2_v128relaxed", test_reduce_moments<e5m2_t>, v128relaxed,
                    nk_reduce_moments_e5m2_v128relaxed);
    run_kernel_case("nk_reduce_moments_e2m3_v128relaxed", test_reduce_moments<e2m3_t>, v128relaxed,
                    nk_reduce_moments_e2m3_v128relaxed);
    run_kernel_case("nk_reduce_moments_e3m2_v128relaxed", test_reduce_moments<e3m2_t>, v128relaxed,
                    nk_reduce_moments_e3m2_v128relaxed);
    run_kernel_case("nk_reduce_moments_f16_v128relaxed", test_reduce_moments<f16_t>, v128relaxed,
                    nk_reduce_moments_f16_v128relaxed);
    run_kernel_case("nk_reduce_minmax_f32_v128relaxed", test_reduce_minmax<f32_t>, v128relaxed,
                    nk_reduce_minmax_f32_v128relaxed);
    run_kernel_case("nk_reduce_minmax_f64_v128relaxed", test_reduce_minmax<f64_t>, v128relaxed,
                    nk_reduce_minmax_f64_v128relaxed);
    run_kernel_case("nk_reduce_minmax_i8_v128relaxed", test_reduce_minmax<i8_t>, v128relaxed,
                    nk_reduce_minmax_i8_v128relaxed);
    run_kernel_case("nk_reduce_minmax_u8_v128relaxed", test_reduce_minmax<u8_t>, v128relaxed,
                    nk_reduce_minmax_u8_v128relaxed);
    run_kernel_case("nk_reduce_minmax_i16_v128relaxed", test_reduce_minmax<i16_t>, v128relaxed,
                    nk_reduce_minmax_i16_v128relaxed);
    run_kernel_case("nk_reduce_minmax_u16_v128relaxed", test_reduce_minmax<u16_t>, v128relaxed,
                    nk_reduce_minmax_u16_v128relaxed);
    run_kernel_case("nk_reduce_minmax_i32_v128relaxed", test_reduce_minmax<i32_t>, v128relaxed,
                    nk_reduce_minmax_i32_v128relaxed);
    run_kernel_case("nk_reduce_minmax_u32_v128relaxed", test_reduce_minmax<u32_t>, v128relaxed,
                    nk_reduce_minmax_u32_v128relaxed);
    run_kernel_case("nk_reduce_minmax_i64_v128relaxed", test_reduce_minmax<i64_t>, v128relaxed,
                    nk_reduce_minmax_i64_v128relaxed);
    run_kernel_case("nk_reduce_minmax_u64_v128relaxed", test_reduce_minmax<u64_t>, v128relaxed,
                    nk_reduce_minmax_u64_v128relaxed);
    run_kernel_case("nk_reduce_minmax_e4m3_v128relaxed", test_reduce_minmax<e4m3_t>, v128relaxed,
                    nk_reduce_minmax_e4m3_v128relaxed);
    run_kernel_case("nk_reduce_minmax_e5m2_v128relaxed", test_reduce_minmax<e5m2_t>, v128relaxed,
                    nk_reduce_minmax_e5m2_v128relaxed);
    run_kernel_case("nk_reduce_minmax_e2m3_v128relaxed", test_reduce_minmax<e2m3_t>, v128relaxed,
                    nk_reduce_minmax_e2m3_v128relaxed);
    run_kernel_case("nk_reduce_minmax_e3m2_v128relaxed", test_reduce_minmax<e3m2_t>, v128relaxed,
                    nk_reduce_minmax_e3m2_v128relaxed);
    run_kernel_case("nk_reduce_minmax_bf16_v128relaxed", test_reduce_minmax<bf16_t>, v128relaxed,
                    nk_reduce_minmax_bf16_v128relaxed);
    run_kernel_case("nk_reduce_minmax_f16_v128relaxed", test_reduce_minmax<f16_t>, v128relaxed,
                    nk_reduce_minmax_f16_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_V128
    auto const v128 = nk_cap_v128_k;
    run_kernel_case("nk_reduce_moments_f64_v128", test_reduce_moments<f64_t>, v128, nk_reduce_moments_f64_v128);
    run_kernel_case("nk_reduce_moments_bf16_v128", test_reduce_moments<bf16_t>, v128, nk_reduce_moments_bf16_v128);
    run_kernel_case("nk_reduce_moments_i8_v128", test_reduce_moments<i8_t>, v128, nk_reduce_moments_i8_v128);
    run_kernel_case("nk_reduce_moments_u8_v128", test_reduce_moments<u8_t>, v128, nk_reduce_moments_u8_v128);
    run_kernel_case("nk_reduce_moments_i16_v128", test_reduce_moments<i16_t>, v128, nk_reduce_moments_i16_v128);
    run_kernel_case("nk_reduce_moments_u16_v128", test_reduce_moments<u16_t>, v128, nk_reduce_moments_u16_v128);
    run_kernel_case("nk_reduce_moments_i32_v128", test_reduce_moments<i32_t>, v128, nk_reduce_moments_i32_v128);
    run_kernel_case("nk_reduce_moments_u32_v128", test_reduce_moments<u32_t>, v128, nk_reduce_moments_u32_v128);
#endif // NUMKONG_TARGET_V128
}

} // namespace ashvardanian::numkong::test
