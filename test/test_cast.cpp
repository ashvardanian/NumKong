/**
 *  @brief Type cast tests.
 *  @file test/test_cast.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/cast.h"

using cast_t = void (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t);

/**
 *  @brief Test cast kernel against serial kernel.
 *  SIMD kernels must match serial output exactly (raw byte comparison).
 */
template <typename from_type_, typename to_type_>
error_stats_t test_cast(cast_t kernel) {
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    auto src = make_vector<from_type_>(global_config.dense_dimensions);
    auto dst_simd = make_vector<to_type_>(global_config.dense_dimensions);
    auto dst_serial = make_vector<to_type_>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, src);

        nk_cast_serial(src.raw_values_data(), from_type_::dtype(), global_config.dense_dimensions,
                       dst_serial.raw_values_data(), to_type_::dtype());
        kernel(src.raw_values_data(), from_type_::dtype(), global_config.dense_dimensions, dst_simd.raw_values_data(),
               to_type_::dtype());

        for (std::size_t i = 0; i < global_config.dense_dimensions; ++i)
            stats.accumulate(dst_simd[i].raw_, dst_serial[i].raw_);
    }
    return stats;
}

void test_casts() {
    error_stats_section_t check;

    check.section("Type Casts Serial", nk_cap_serial_k);
    check("cast_bf16_to_f32_serial", test_cast<bf16_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_bf16_serial", test_cast<f32_t, bf16_t>, nk_cast_serial);
    check("cast_e4m3_to_f32_serial", test_cast<e4m3_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_e4m3_serial", test_cast<f32_t, e4m3_t>, nk_cast_serial);
    check("cast_e5m2_to_f32_serial", test_cast<e5m2_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_e5m2_serial", test_cast<f32_t, e5m2_t>, nk_cast_serial);
    check("cast_f16_to_f32_serial", test_cast<f16_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_f16_serial", test_cast<f32_t, f16_t>, nk_cast_serial);
    check("cast_f32_to_f64_serial", test_cast<f32_t, f64_t>, nk_cast_serial);
    check("cast_f64_to_f32_serial", test_cast<f64_t, f32_t>, nk_cast_serial);
    check("cast_f64_to_i32_serial", test_cast<f64_t, i32_t>, nk_cast_serial);
    check("cast_i16_to_i64_serial", test_cast<i16_t, i64_t>, nk_cast_serial);
    check("cast_i32_to_f64_serial", test_cast<i32_t, f64_t>, nk_cast_serial);
    check("cast_i32_to_i8_serial", test_cast<i32_t, i8_t>, nk_cast_serial);
    check("cast_i8_to_f64_serial", test_cast<i8_t, f64_t>, nk_cast_serial);
    check("cast_i8_to_i32_serial", test_cast<i8_t, i32_t>, nk_cast_serial);
    check("cast_u8_to_f32_serial", test_cast<u8_t, f32_t>, nk_cast_serial);

#if NK_DYNAMIC_DISPATCH
    check.section("Type Casts Dynamic", nk_cap_serial_k);
    check("cast_f32_to_f16", test_cast<f32_t, f16_t>, nk_cast);
    check("cast_f16_to_f32", test_cast<f16_t, f32_t>, nk_cast);
    check("cast_f32_to_bf16", test_cast<f32_t, bf16_t>, nk_cast);
    check("cast_bf16_to_f32", test_cast<bf16_t, f32_t>, nk_cast);
    check("cast_f32_to_e4m3", test_cast<f32_t, e4m3_t>, nk_cast);
    check("cast_e4m3_to_f32", test_cast<e4m3_t, f32_t>, nk_cast);
    check("cast_f32_to_e5m2", test_cast<f32_t, e5m2_t>, nk_cast);
    check("cast_e5m2_to_f32", test_cast<e5m2_t, f32_t>, nk_cast);
    check("cast_f32_to_e2m3", test_cast<f32_t, e2m3_t>, nk_cast);
    check("cast_e2m3_to_f32", test_cast<e2m3_t, f32_t>, nk_cast);
    check("cast_f32_to_e3m2", test_cast<f32_t, e3m2_t>, nk_cast);
    check("cast_e3m2_to_f32", test_cast<e3m2_t, f32_t>, nk_cast);
    check("cast_f64_to_f32", test_cast<f64_t, f32_t>, nk_cast);
    check("cast_f32_to_f64", test_cast<f32_t, f64_t>, nk_cast);
    // Integer ↔ integer
    check("cast_i8_to_i32", test_cast<i8_t, i32_t>, nk_cast);
    check("cast_i32_to_i8", test_cast<i32_t, i8_t>, nk_cast);
    check("cast_u8_to_u32", test_cast<u8_t, u32_t>, nk_cast);
    check("cast_u32_to_u8", test_cast<u32_t, u8_t>, nk_cast);
    check("cast_i16_to_i64", test_cast<i16_t, i64_t>, nk_cast);
    check("cast_i64_to_i16", test_cast<i64_t, i16_t>, nk_cast);
    check("cast_i32_to_u32", test_cast<i32_t, u32_t>, nk_cast);
    // Integer ↔ float
    check("cast_i32_to_f64", test_cast<i32_t, f64_t>, nk_cast);
    check("cast_f64_to_i32", test_cast<f64_t, i32_t>, nk_cast);
    check("cast_i16_to_f32", test_cast<i16_t, f32_t>, nk_cast);
    check("cast_u8_to_f32", test_cast<u8_t, f32_t>, nk_cast);
    check("cast_f32_to_i8", test_cast<f32_t, i8_t>, nk_cast);
    check("cast_i8_to_f64", test_cast<i8_t, f64_t>, nk_cast);
    check("cast_f64_to_u8", test_cast<f64_t, u8_t>, nk_cast);
    // Verify serial fallbacks for rare paths
    check("cast_f64_to_f16", test_cast<f64_t, f16_t>, nk_cast);
    check("cast_f16_to_f64", test_cast<f16_t, f64_t>, nk_cast);
    check("cast_f64_to_bf16", test_cast<f64_t, bf16_t>, nk_cast);
    check("cast_bf16_to_f64", test_cast<bf16_t, f64_t>, nk_cast);
#endif

#if NK_TARGET_HASWELL
    check.section("Type Casts Haswell", nk_cap_haswell_k);
    check("cast_f32_to_f16_haswell", test_cast<f32_t, f16_t>, nk_cast_haswell);
    check("cast_f16_to_f32_haswell", test_cast<f16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_bf16_haswell", test_cast<f32_t, bf16_t>, nk_cast_haswell);
    check("cast_bf16_to_f32_haswell", test_cast<bf16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e4m3_haswell", test_cast<f32_t, e4m3_t>, nk_cast_haswell);
    check("cast_e4m3_to_f32_haswell", test_cast<e4m3_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e5m2_haswell", test_cast<f32_t, e5m2_t>, nk_cast_haswell);
    check("cast_e5m2_to_f32_haswell", test_cast<e5m2_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e2m3_haswell", test_cast<f32_t, e2m3_t>, nk_cast_haswell);
    check("cast_e2m3_to_f32_haswell", test_cast<e2m3_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e3m2_haswell", test_cast<f32_t, e3m2_t>, nk_cast_haswell);
    check("cast_e3m2_to_f32_haswell", test_cast<e3m2_t, f32_t>, nk_cast_haswell);
    check("cast_i8_to_f32_haswell", test_cast<i8_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_i8_haswell", test_cast<f32_t, i8_t>, nk_cast_haswell);
    check("cast_i16_to_f32_haswell", test_cast<i16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_i16_haswell", test_cast<f32_t, i16_t>, nk_cast_haswell);
    check("cast_u16_to_f32_haswell", test_cast<u16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_u16_haswell", test_cast<f32_t, u16_t>, nk_cast_haswell);
    check("cast_u8_to_f32_haswell", test_cast<u8_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_u8_haswell", test_cast<f32_t, u8_t>, nk_cast_haswell);
    // Verify serial fallbacks for rare paths
    check("cast_i32_to_f64_haswell", test_cast<i32_t, f64_t>, nk_cast_haswell);
    check("cast_f64_to_f32_haswell", test_cast<f64_t, f32_t>, nk_cast_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check.section("Type Casts Skylake", nk_cap_skylake_k);
    check("cast_f32_to_f16_skylake", test_cast<f32_t, f16_t>, nk_cast_skylake);
    check("cast_f16_to_f32_skylake", test_cast<f16_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_bf16_skylake", test_cast<f32_t, bf16_t>, nk_cast_skylake);
    check("cast_bf16_to_f32_skylake", test_cast<bf16_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_e4m3_skylake", test_cast<f32_t, e4m3_t>, nk_cast_skylake);
    check("cast_e4m3_to_f32_skylake", test_cast<e4m3_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_e5m2_skylake", test_cast<f32_t, e5m2_t>, nk_cast_skylake);
    check("cast_e5m2_to_f32_skylake", test_cast<e5m2_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_e2m3_skylake", test_cast<f32_t, e2m3_t>, nk_cast_skylake);
    check("cast_e2m3_to_f32_skylake", test_cast<e2m3_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_e3m2_skylake", test_cast<f32_t, e3m2_t>, nk_cast_skylake);
    check("cast_e3m2_to_f32_skylake", test_cast<e3m2_t, f32_t>, nk_cast_skylake);
    check("cast_f16_to_bf16_skylake", test_cast<f16_t, bf16_t>, nk_cast_skylake);
    check("cast_bf16_to_f16_skylake", test_cast<bf16_t, f16_t>, nk_cast_skylake);
    check("cast_e4m3_to_f16_skylake", test_cast<e4m3_t, f16_t>, nk_cast_skylake);
    check("cast_f16_to_e4m3_skylake", test_cast<f16_t, e4m3_t>, nk_cast_skylake);
    check("cast_e5m2_to_f16_skylake", test_cast<e5m2_t, f16_t>, nk_cast_skylake);
    check("cast_f16_to_e5m2_skylake", test_cast<f16_t, e5m2_t>, nk_cast_skylake);
    check("cast_e4m3_to_bf16_skylake", test_cast<e4m3_t, bf16_t>, nk_cast_skylake);
    check("cast_bf16_to_e4m3_skylake", test_cast<bf16_t, e4m3_t>, nk_cast_skylake);
    check("cast_e5m2_to_bf16_skylake", test_cast<e5m2_t, bf16_t>, nk_cast_skylake);
    check("cast_bf16_to_e5m2_skylake", test_cast<bf16_t, e5m2_t>, nk_cast_skylake);
    check("cast_f64_to_f32_skylake", test_cast<f64_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_f64_skylake", test_cast<f32_t, f64_t>, nk_cast_skylake);
    check("cast_i32_to_f64_skylake", test_cast<i32_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_i32_skylake", test_cast<f64_t, i32_t>, nk_cast_skylake);
    check("cast_i8_to_i32_skylake", test_cast<i8_t, i32_t>, nk_cast_skylake);
    check("cast_i32_to_i8_skylake", test_cast<i32_t, i8_t>, nk_cast_skylake);
    check("cast_i16_to_f32_skylake", test_cast<i16_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_i16_skylake", test_cast<f32_t, i16_t>, nk_cast_skylake);
    check("cast_u16_to_f32_skylake", test_cast<u16_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_u16_skylake", test_cast<f32_t, u16_t>, nk_cast_skylake);
    check("cast_u8_to_f32_skylake", test_cast<u8_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_u8_skylake", test_cast<f32_t, u8_t>, nk_cast_skylake);
    check("cast_i64_to_f64_skylake", test_cast<i64_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_i64_skylake", test_cast<f64_t, i64_t>, nk_cast_skylake);
    check("cast_u64_to_f64_skylake", test_cast<u64_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_u64_skylake", test_cast<f64_t, u64_t>, nk_cast_skylake);
    check("cast_u32_to_f64_skylake", test_cast<u32_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_u32_skylake", test_cast<f64_t, u32_t>, nk_cast_skylake);
    // Verify serial fallbacks for rare paths
    check("cast_i8_to_f64_skylake", test_cast<i8_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_bf16_skylake", test_cast<f64_t, bf16_t>, nk_cast_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
    check.section("Type Casts Ice Lake", nk_cap_icelake_k);
    check("cast_e4m3_to_bf16_icelake", test_cast<e4m3_t, bf16_t>, nk_cast_icelake);
    check("cast_bf16_to_e4m3_icelake", test_cast<bf16_t, e4m3_t>, nk_cast_icelake);
    check("cast_e5m2_to_bf16_icelake", test_cast<e5m2_t, bf16_t>, nk_cast_icelake);
    check("cast_bf16_to_e5m2_icelake", test_cast<bf16_t, e5m2_t>, nk_cast_icelake);
    check("cast_e4m3_to_f16_icelake", test_cast<e4m3_t, f16_t>, nk_cast_icelake);
    check("cast_e5m2_to_f16_icelake", test_cast<e5m2_t, f16_t>, nk_cast_icelake);
    check("cast_e4m3_to_f32_icelake", test_cast<e4m3_t, f32_t>, nk_cast_icelake);
    check("cast_f32_to_e4m3_icelake", test_cast<f32_t, e4m3_t>, nk_cast_icelake);
    check("cast_f16_to_f32_icelake", test_cast<f16_t, f32_t>, nk_cast_icelake);
    check("cast_f32_to_f16_icelake", test_cast<f32_t, f16_t>, nk_cast_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_SAPPHIRE
    check.section("Type Casts Sapphire", nk_cap_sapphire_k);
    check("cast_e4m3_to_f16_sapphire", test_cast<e4m3_t, f16_t>, nk_cast_sapphire);
    check("cast_f16_to_e4m3_sapphire", test_cast<f16_t, e4m3_t>, nk_cast_sapphire);
    check("cast_e5m2_to_f16_sapphire", test_cast<e5m2_t, f16_t>, nk_cast_sapphire);
    check("cast_f16_to_e5m2_sapphire", test_cast<f16_t, e5m2_t>, nk_cast_sapphire);
    check("cast_f16_to_f32_sapphire", test_cast<f16_t, f32_t>, nk_cast_sapphire);
    check("cast_f32_to_f16_sapphire", test_cast<f32_t, f16_t>, nk_cast_sapphire);
#endif // NK_TARGET_SAPPHIRE

#if NK_TARGET_NEON
    check.section("Type Casts NEON", nk_cap_neon_k);
    check("cast_e4m3_to_f32_neon", test_cast<e4m3_t, f32_t>, nk_cast_neon);
    check("cast_f32_to_e4m3_neon", test_cast<f32_t, e4m3_t>, nk_cast_neon);
    check("cast_e5m2_to_f32_neon", test_cast<e5m2_t, f32_t>, nk_cast_neon);
    check("cast_f32_to_e5m2_neon", test_cast<f32_t, e5m2_t>, nk_cast_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_V128RELAXED
    check.section("Type Casts V128 Relaxed", nk_cap_v128relaxed_k);
    check("cast_f32_to_f16_v128relaxed", test_cast<f32_t, f16_t>, nk_cast_v128relaxed);
    check("cast_f16_to_f32_v128relaxed", test_cast<f16_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_bf16_v128relaxed", test_cast<f32_t, bf16_t>, nk_cast_v128relaxed);
    check("cast_bf16_to_f32_v128relaxed", test_cast<bf16_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e4m3_v128relaxed", test_cast<f32_t, e4m3_t>, nk_cast_v128relaxed);
    check("cast_e4m3_to_f32_v128relaxed", test_cast<e4m3_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e5m2_v128relaxed", test_cast<f32_t, e5m2_t>, nk_cast_v128relaxed);
    check("cast_e5m2_to_f32_v128relaxed", test_cast<e5m2_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e2m3_v128relaxed", test_cast<f32_t, e2m3_t>, nk_cast_v128relaxed);
    check("cast_e2m3_to_f32_v128relaxed", test_cast<e2m3_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e3m2_v128relaxed", test_cast<f32_t, e3m2_t>, nk_cast_v128relaxed);
    check("cast_e3m2_to_f32_v128relaxed", test_cast<e3m2_t, f32_t>, nk_cast_v128relaxed);
    check("cast_i8_to_f32_v128relaxed", test_cast<i8_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_i8_v128relaxed", test_cast<f32_t, i8_t>, nk_cast_v128relaxed);
    check("cast_u8_to_f32_v128relaxed", test_cast<u8_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_u8_v128relaxed", test_cast<f32_t, u8_t>, nk_cast_v128relaxed);
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_RVV
    check.section("Type Casts RVV", nk_cap_rvv_k);
    check("cast_bf16_to_f32_rvv", test_cast<bf16_t, f32_t>, nk_cast_rvv);
    check("cast_f32_to_bf16_rvv", test_cast<f32_t, bf16_t>, nk_cast_rvv);
    check("cast_e4m3_to_f32_rvv", test_cast<e4m3_t, f32_t>, nk_cast_rvv);
    check("cast_e5m2_to_f32_rvv", test_cast<e5m2_t, f32_t>, nk_cast_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_POWERVSX
    check.section("Type Casts Power VSX", nk_cap_powervsx_k);
    check("cast_f32_to_f16_powervsx", test_cast<f32_t, f16_t>, nk_cast_powervsx);
    check("cast_f16_to_f32_powervsx", test_cast<f16_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_bf16_powervsx", test_cast<f32_t, bf16_t>, nk_cast_powervsx);
    check("cast_bf16_to_f32_powervsx", test_cast<bf16_t, f32_t>, nk_cast_powervsx);
    check("cast_i8_to_f32_powervsx", test_cast<i8_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_i8_powervsx", test_cast<f32_t, i8_t>, nk_cast_powervsx);
    check("cast_u8_to_f32_powervsx", test_cast<u8_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_u8_powervsx", test_cast<f32_t, u8_t>, nk_cast_powervsx);
    check("cast_i16_to_f32_powervsx", test_cast<i16_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_i16_powervsx", test_cast<f32_t, i16_t>, nk_cast_powervsx);
    check("cast_u16_to_f32_powervsx", test_cast<u16_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_u16_powervsx", test_cast<f32_t, u16_t>, nk_cast_powervsx);
#endif // NK_TARGET_POWERVSX
}
