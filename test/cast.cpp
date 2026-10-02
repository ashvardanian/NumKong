/**
 *  @file test/cast.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Type cast tests.
 */

#include "harness.hpp"
#include "cast.hpp" // `test_cast`, `check_block_scaled_casts`

namespace ashvardanian::numkong::test {

void test_casts(error_stats_section_t &check) {

    check.section("Type Casts Serial", nk_cap_serial_k);
    check("cast_bf16_to_f32_serial", test_cast<bf16_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_bf16_serial", test_cast<f32_t, bf16_t>, nk_cast_serial);
    check("cast_e4m3_to_f32_serial", test_cast<e4m3_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_e4m3_serial", test_cast<f32_t, e4m3_t>, nk_cast_serial);
    check("cast_e5m2_to_f32_serial", test_cast<e5m2_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_e5m2_serial", test_cast<f32_t, e5m2_t>, nk_cast_serial);
    check("cast_e2m1_to_f32_serial", test_cast<e2m1x2_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_e2m1_serial", test_cast<f32_t, e2m1x2_t>, nk_cast_serial);
    check("cast_ue8m0_to_f32_serial", test_cast<ue8m0_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_ue8m0_serial", test_cast<f32_t, ue8m0_t>, nk_cast_serial);
    check("cast_ue4m3_to_f32_serial", test_cast<ue4m3_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_ue4m3_serial", test_cast<f32_t, ue4m3_t>, nk_cast_serial);
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

    check_block_scaled_casts(check, "serial", nk_cast_serial);

#if !NUMKONG_HEADER_ONLY
    check.section("Type Casts Runtime Dispatch", nk_cap_serial_k);
    check("cast_f32_to_f16", test_cast<f32_t, f16_t>, cpu_best<nk_cast_best>);
    check("cast_f16_to_f32", test_cast<f16_t, f32_t>, cpu_best<nk_cast_best>);
    check("cast_f32_to_bf16", test_cast<f32_t, bf16_t>, cpu_best<nk_cast_best>);
    check("cast_bf16_to_f32", test_cast<bf16_t, f32_t>, cpu_best<nk_cast_best>);
    check("cast_f32_to_e4m3", test_cast<f32_t, e4m3_t>, cpu_best<nk_cast_best>);
    check("cast_e4m3_to_f32", test_cast<e4m3_t, f32_t>, cpu_best<nk_cast_best>);
    check("cast_f32_to_e5m2", test_cast<f32_t, e5m2_t>, cpu_best<nk_cast_best>);
    check("cast_e5m2_to_f32", test_cast<e5m2_t, f32_t>, cpu_best<nk_cast_best>);
    check("cast_f32_to_e2m3", test_cast<f32_t, e2m3_t>, cpu_best<nk_cast_best>);
    check("cast_e2m3_to_f32", test_cast<e2m3_t, f32_t>, cpu_best<nk_cast_best>);
    check("cast_f32_to_e3m2", test_cast<f32_t, e3m2_t>, cpu_best<nk_cast_best>);
    check("cast_e3m2_to_f32", test_cast<e3m2_t, f32_t>, cpu_best<nk_cast_best>);
    check("cast_f64_to_f32", test_cast<f64_t, f32_t>, cpu_best<nk_cast_best>);
    check("cast_f32_to_f64", test_cast<f32_t, f64_t>, cpu_best<nk_cast_best>);
    // Integer ↔ integer
    check("cast_i8_to_i32", test_cast<i8_t, i32_t>, cpu_best<nk_cast_best>);
    check("cast_i32_to_i8", test_cast<i32_t, i8_t>, cpu_best<nk_cast_best>);
    check("cast_u8_to_u32", test_cast<u8_t, u32_t>, cpu_best<nk_cast_best>);
    check("cast_u32_to_u8", test_cast<u32_t, u8_t>, cpu_best<nk_cast_best>);
    check("cast_i16_to_i64", test_cast<i16_t, i64_t>, cpu_best<nk_cast_best>);
    check("cast_i64_to_i16", test_cast<i64_t, i16_t>, cpu_best<nk_cast_best>);
    check("cast_i32_to_u32", test_cast<i32_t, u32_t>, cpu_best<nk_cast_best>);
    // Integer ↔ float
    check("cast_i32_to_f64", test_cast<i32_t, f64_t>, cpu_best<nk_cast_best>);
    check("cast_f64_to_i32", test_cast<f64_t, i32_t>, cpu_best<nk_cast_best>);
    check("cast_i16_to_f32", test_cast<i16_t, f32_t>, cpu_best<nk_cast_best>);
    check("cast_u8_to_f32", test_cast<u8_t, f32_t>, cpu_best<nk_cast_best>);
    check("cast_f32_to_i8", test_cast<f32_t, i8_t>, cpu_best<nk_cast_best>);
    check("cast_i8_to_f64", test_cast<i8_t, f64_t>, cpu_best<nk_cast_best>);
    check("cast_f64_to_u8", test_cast<f64_t, u8_t>, cpu_best<nk_cast_best>);
    // Verify serial fallbacks for rare paths
    check("cast_f64_to_f16", test_cast<f64_t, f16_t>, cpu_best<nk_cast_best>);
    check("cast_f16_to_f64", test_cast<f16_t, f64_t>, cpu_best<nk_cast_best>);
    check("cast_f64_to_bf16", test_cast<f64_t, bf16_t>, cpu_best<nk_cast_best>);
    check("cast_bf16_to_f64", test_cast<bf16_t, f64_t>, cpu_best<nk_cast_best>);
#endif

#if NUMKONG_TARGET_HASWELL
    check.section("Type Casts Haswell", nk_cap_haswell_k);
    check("cast_f32_to_f16_haswell", test_cast<f32_t, f16_t>, nk_cast_haswell);
    check("cast_f16_to_f32_haswell", test_cast<f16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_bf16_haswell", test_cast<f32_t, bf16_t>, nk_cast_haswell);
    check("cast_bf16_to_f32_haswell", test_cast<bf16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e4m3_haswell", test_cast<f32_t, e4m3_t>, nk_cast_haswell);
    check("cast_e4m3_to_f32_haswell", test_cast<e4m3_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e5m2_haswell", test_cast<f32_t, e5m2_t>, nk_cast_haswell);
    check("cast_e5m2_to_f32_haswell", test_cast<e5m2_t, f32_t>, nk_cast_haswell);
    check("cast_e5m2_to_f16_haswell", test_cast<e5m2_t, f16_t>, nk_cast_haswell);
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
    check("cast_f32_to_i32_haswell", test_cast<f32_t, i32_t>, nk_cast_haswell);
    check("cast_f32_to_u32_haswell", test_cast<f32_t, u32_t>, nk_cast_haswell);
    // Verify serial fallbacks for rare paths
    check("cast_i32_to_f64_haswell", test_cast<i32_t, f64_t>, nk_cast_haswell);
    check("cast_f64_to_f32_haswell", test_cast<f64_t, f32_t>, nk_cast_haswell);
    check_block_scaled_casts(check, "haswell", nk_cast_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
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
    check("cast_u64_to_i64_skylake", test_cast<u64_t, i64_t>, nk_cast_skylake);
    check("cast_i64_to_u64_skylake", test_cast<i64_t, u64_t>, nk_cast_skylake);
    check("cast_u64_to_i32_skylake", test_cast<u64_t, i32_t>, nk_cast_skylake);
    // Verify serial fallbacks for rare paths
    check("cast_i8_to_f64_skylake", test_cast<i8_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_bf16_skylake", test_cast<f64_t, bf16_t>, nk_cast_skylake);
    check_block_scaled_casts(check, "skylake", nk_cast_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
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
    check("cast_e2m3_to_f32_icelake", test_cast<e2m3_t, f32_t>, nk_cast_icelake);
    check("cast_f32_to_e2m3_icelake", test_cast<f32_t, e2m3_t>, nk_cast_icelake);
    check("cast_e3m2_to_f32_icelake", test_cast<e3m2_t, f32_t>, nk_cast_icelake);
    check("cast_f32_to_e3m2_icelake", test_cast<f32_t, e3m2_t>, nk_cast_icelake);
    check_block_scaled_casts(check, "icelake", nk_cast_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_SAPPHIRE
    check.section("Type Casts Sapphire", nk_cap_sapphire_k);
    check("cast_e4m3_to_f16_sapphire", test_cast<e4m3_t, f16_t>, nk_cast_sapphire);
    check("cast_f16_to_e4m3_sapphire", test_cast<f16_t, e4m3_t>, nk_cast_sapphire);
    check("cast_e5m2_to_f16_sapphire", test_cast<e5m2_t, f16_t>, nk_cast_sapphire);
    check("cast_f16_to_e5m2_sapphire", test_cast<f16_t, e5m2_t>, nk_cast_sapphire);
    check("cast_f16_to_f32_sapphire", test_cast<f16_t, f32_t>, nk_cast_sapphire);
    check("cast_f32_to_f16_sapphire", test_cast<f32_t, f16_t>, nk_cast_sapphire);
#endif // NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_NEON
    check.section("Type Casts NEON", nk_cap_neon_k);
    check("cast_bf16_to_f32_neon", test_cast<bf16_t, f32_t>, nk_cast_neon);
    check("cast_f32_to_bf16_neon", test_cast<f32_t, bf16_t>, nk_cast_neon);
    check("cast_e4m3_to_f32_neon", test_cast<e4m3_t, f32_t>, nk_cast_neon);
    check("cast_f32_to_e4m3_neon", test_cast<f32_t, e4m3_t>, nk_cast_neon);
    check("cast_e5m2_to_f32_neon", test_cast<e5m2_t, f32_t>, nk_cast_neon);
    check("cast_e5m2_to_f16_neon", test_cast<e5m2_t, f16_t>, nk_cast_neon);
    check("cast_f32_to_e5m2_neon", test_cast<f32_t, e5m2_t>, nk_cast_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_V128RELAXED
    check.section("Type Casts V128 Relaxed", nk_cap_v128relaxed_k);
    check("cast_f32_to_f16_v128relaxed", test_cast<f32_t, f16_t>, nk_cast_v128relaxed);
    check("cast_f16_to_f32_v128relaxed", test_cast<f16_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_bf16_v128relaxed", test_cast<f32_t, bf16_t>, nk_cast_v128relaxed);
    check("cast_bf16_to_f32_v128relaxed", test_cast<bf16_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e4m3_v128relaxed", test_cast<f32_t, e4m3_t>, nk_cast_v128relaxed);
    check("cast_e4m3_to_f32_v128relaxed", test_cast<e4m3_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e5m2_v128relaxed", test_cast<f32_t, e5m2_t>, nk_cast_v128relaxed);
    check("cast_e5m2_to_f32_v128relaxed", test_cast<e5m2_t, f32_t>, nk_cast_v128relaxed);
    check("cast_e5m2_to_bf16_v128relaxed", test_cast<e5m2_t, bf16_t>, nk_cast_v128relaxed);
    check("cast_e5m2_to_f16_v128relaxed", test_cast<e5m2_t, f16_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e2m3_v128relaxed", test_cast<f32_t, e2m3_t>, nk_cast_v128relaxed);
    check("cast_e2m3_to_f32_v128relaxed", test_cast<e2m3_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e3m2_v128relaxed", test_cast<f32_t, e3m2_t>, nk_cast_v128relaxed);
    check("cast_e3m2_to_f32_v128relaxed", test_cast<e3m2_t, f32_t>, nk_cast_v128relaxed);
    check("cast_i8_to_f32_v128relaxed", test_cast<i8_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_i8_v128relaxed", test_cast<f32_t, i8_t>, nk_cast_v128relaxed);
    check("cast_u8_to_f32_v128relaxed", test_cast<u8_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_u8_v128relaxed", test_cast<f32_t, u8_t>, nk_cast_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_RVV
    check.section("Type Casts RVV", nk_cap_rvv_k);
    check("cast_bf16_to_f32_rvv", test_cast<bf16_t, f32_t>, nk_cast_rvv);
    check("cast_f32_to_bf16_rvv", test_cast<f32_t, bf16_t>, nk_cast_rvv);
    check("cast_e4m3_to_f32_rvv", test_cast<e4m3_t, f32_t>, nk_cast_rvv);
    check("cast_e5m2_to_f32_rvv", test_cast<e5m2_t, f32_t>, nk_cast_rvv);
    check("cast_e5m2_to_bf16_rvv", test_cast<e5m2_t, bf16_t>, nk_cast_rvv);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_POWERVSX
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
#endif // NUMKONG_TARGET_POWERVSX
}

} // namespace ashvardanian::numkong::test
