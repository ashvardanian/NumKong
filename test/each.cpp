/**
 *  @file test/each.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief Elementwise operations tests.
 */

#include "harness.hpp"
#include "each.hpp"                 // `test_scale`, `test_sum`, `test_blend`, `test_fma`, `test_rmsnorm`
#include "numkong/each.hpp"         // `nk::add`, `nk::scale`, `nk::blend`, `nk::fma`
#include "numkong/trigonometry.hpp" // `nk::sin`, `nk::cos`, `nk::atan` wrappers

namespace ashvardanian::numkong::test {

/** Smoke-test for the tensor-shaped trig wrappers @c nk::sin, @c cos and @c atan, running
 *  allocating and into-span variants on a small zero tensor, just exercising the dispatch paths,
 *  not the numerical accuracy, which the scenarios of `each.hpp` cover. */
void test_each(error_stats_section_t &check) {

    check.section("Elementwise Operations Serial", nk_cap_serial_k);
    check("each_scale_f32_serial", test_scale<f32_t>, nk_each_scale_f32_serial);
    check("each_sum_f32_serial", test_sum<f32_t>, nk_each_sum_f32_serial);
    check("each_blend_f32_serial", test_blend<f32_t>, nk_each_blend_f32_serial);
    check("each_fma_f32_serial", test_fma<f32_t>, nk_each_fma_f32_serial);
    check("each_swiglu_f32_serial", test_swiglu<f32_t>, nk_each_swiglu_f32_serial);
    check("each_rmsnorm_f32_serial", test_rmsnorm<f32_t>, nk_each_rmsnorm_f32_serial);
    check("each_scale_e4m3_serial", test_scale<e4m3_t>, nk_each_scale_e4m3_serial);
    check("each_scale_e5m2_serial", test_scale<e5m2_t>, nk_each_scale_e5m2_serial);
    check("each_sum_e4m3_serial", test_sum<e4m3_t>, nk_each_sum_e4m3_serial);
    check("each_sum_e5m2_serial", test_sum<e5m2_t>, nk_each_sum_e5m2_serial);
    check("each_blend_e4m3_serial", test_blend<e4m3_t>, nk_each_blend_e4m3_serial);
    check("each_blend_e5m2_serial", test_blend<e5m2_t>, nk_each_blend_e5m2_serial);
    check("each_fma_e4m3_serial", test_fma<e4m3_t>, nk_each_fma_e4m3_serial);
    check("each_swiglu_e4m3_serial", test_swiglu<e4m3_t>, nk_each_swiglu_e4m3_serial);
    check("each_rmsnorm_e4m3_serial", test_rmsnorm<e4m3_t>, nk_each_rmsnorm_e4m3_serial);
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
    check("each_swiglu_f16_serial", test_swiglu<f16_t>, nk_each_swiglu_f16_serial);
    check("each_swiglu_bf16_serial", test_swiglu<bf16_t>, nk_each_swiglu_bf16_serial);
    check("each_rmsnorm_f16_serial", test_rmsnorm<f16_t>, nk_each_rmsnorm_f16_serial);
    check("each_rmsnorm_bf16_serial", test_rmsnorm<bf16_t>, nk_each_rmsnorm_bf16_serial);
    check("each_sum_f16_serial", test_sum<f16_t>, nk_each_sum_f16_serial);
    check("each_scale_f16_serial", test_scale<f16_t>, nk_each_scale_f16_serial);

#if !NUMKONG_HEADER_ONLY
    check.section("Elementwise Operations Runtime Dispatch", nk_cap_serial_k);
    check("each_scale_f32", test_scale<f32_t>, cpu_best<nk_each_scale_f32_best>);
    check("each_sum_f32", test_sum<f32_t>, cpu_best<nk_each_sum_f32_best>);
    check("each_blend_f32", test_blend<f32_t>, cpu_best<nk_each_blend_f32_best>);
    check("each_fma_f32", test_fma<f32_t>, cpu_best<nk_each_fma_f32_best>);
    check("each_swiglu_f32", test_swiglu<f32_t>, cpu_best<nk_each_swiglu_f32_best>);
    check("each_scale_e4m3", test_scale<e4m3_t>, cpu_best<nk_each_scale_e4m3_best>);
    check("each_scale_e5m2", test_scale<e5m2_t>, cpu_best<nk_each_scale_e5m2_best>);
    check("each_sum_e4m3", test_sum<e4m3_t>, cpu_best<nk_each_sum_e4m3_best>);
    check("each_sum_e5m2", test_sum<e5m2_t>, cpu_best<nk_each_sum_e5m2_best>);
    check("each_blend_e4m3", test_blend<e4m3_t>, cpu_best<nk_each_blend_e4m3_best>);
    check("each_blend_e5m2", test_blend<e5m2_t>, cpu_best<nk_each_blend_e5m2_best>);
    check("each_fma_e4m3", test_fma<e4m3_t>, cpu_best<nk_each_fma_e4m3_best>);
    check("each_swiglu_e4m3", test_swiglu<e4m3_t>, cpu_best<nk_each_swiglu_e4m3_best>);
    check("each_fma_e5m2", test_fma<e5m2_t>, cpu_best<nk_each_fma_e5m2_best>);
    check("each_sum_f32c", test_sum<f32c_t>, cpu_best<nk_each_sum_f32c_best>);
    check("each_sum_f64c", test_sum<f64c_t>, cpu_best<nk_each_sum_f64c_best>);
    check("each_scale_f32c", test_scale<f32c_t>, cpu_best<nk_each_scale_f32c_best>);
    check("each_scale_f64c", test_scale<f64c_t>, cpu_best<nk_each_scale_f64c_best>);
    check("each_blend_f32c", test_blend<f32c_t>, cpu_best<nk_each_blend_f32c_best>);
    check("each_blend_f64c", test_blend<f64c_t>, cpu_best<nk_each_blend_f64c_best>);
    check("each_fma_f32c", test_fma<f32c_t>, cpu_best<nk_each_fma_f32c_best>);
    check("each_fma_f64c", test_fma<f64c_t>, cpu_best<nk_each_fma_f64c_best>);
    check("each_swiglu_f16", test_swiglu<f16_t>, cpu_best<nk_each_swiglu_f16_best>);
    check("each_rmsnorm_f16", test_rmsnorm<f16_t>, cpu_best<nk_each_rmsnorm_f16_best>);
    check("each_swiglu_bf16", test_swiglu<bf16_t>, cpu_best<nk_each_swiglu_bf16_best>);
#endif

#if NUMKONG_TARGET_NEON
    check.section("Elementwise Operations NEON", nk_cap_neon_k);
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
    check("each_swiglu_f32_neon", test_swiglu<f32_t>, nk_each_swiglu_f32_neon);
    check("each_rmsnorm_f32_neon", test_rmsnorm<f32_t>, nk_each_rmsnorm_f32_neon);
    // f16, bf16
    check("each_scale_f16_neon", test_scale<f16_t>, nk_each_scale_f16_neon);
    check("each_blend_f16_neon", test_blend<f16_t>, nk_each_blend_f16_neon);
    check("each_fma_f16_neon", test_fma<f16_t>, nk_each_fma_f16_neon);
    check("each_swiglu_f16_neon", test_swiglu<f16_t>, nk_each_swiglu_f16_neon);
    check("each_swiglu_bf16_neon", test_swiglu<bf16_t>, nk_each_swiglu_bf16_neon);
    check("each_rmsnorm_f16_neon", test_rmsnorm<f16_t>, nk_each_rmsnorm_f16_neon);
    check("each_rmsnorm_bf16_neon", test_rmsnorm<bf16_t>, nk_each_rmsnorm_bf16_neon);
    // e4m3, e5m2
    check("each_sum_e4m3_neon", test_sum<e4m3_t>, nk_each_sum_e4m3_neon);
    check("each_scale_e4m3_neon", test_scale<e4m3_t>, nk_each_scale_e4m3_neon);
    check("each_blend_e4m3_neon", test_blend<e4m3_t>, nk_each_blend_e4m3_neon);
    check("each_fma_e4m3_neon", test_fma<e4m3_t>, nk_each_fma_e4m3_neon);
    check("each_swiglu_e4m3_neon", test_swiglu<e4m3_t>, nk_each_swiglu_e4m3_neon);
    check("each_rmsnorm_e4m3_neon", test_rmsnorm<e4m3_t>, nk_each_rmsnorm_e4m3_neon);
    check("each_sum_e5m2_neon", test_sum<e5m2_t>, nk_each_sum_e5m2_neon);
    check("each_scale_e5m2_neon", test_scale<e5m2_t>, nk_each_scale_e5m2_neon);
    check("each_blend_e5m2_neon", test_blend<e5m2_t>, nk_each_blend_e5m2_neon);
    check("each_fma_e5m2_neon", test_fma<e5m2_t>, nk_each_fma_e5m2_neon);
    // u8, i8
    check("each_sum_u8_neon", test_sum<u8_t>, nk_each_sum_u8_neon);
    check("each_scale_u8_neon", test_scale<u8_t>, nk_each_scale_u8_neon);
    check("each_blend_u8_neon", test_blend<u8_t>, nk_each_blend_u8_neon);
    check("each_sum_i8_neon", test_sum<i8_t>, nk_each_sum_i8_neon);
    check("each_scale_i8_neon", test_scale<i8_t>, nk_each_scale_i8_neon);
    check("each_blend_i8_neon", test_blend<i8_t>, nk_each_blend_i8_neon);
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
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONHALF
    check.section("Elementwise Operations NEON HALF", nk_cap_neonhalf_k);
    check("each_sum_f16_neonhalf", test_sum<f16_t>, nk_each_sum_f16_neonhalf);
#endif // NUMKONG_TARGET_NEONHALF

#if NUMKONG_TARGET_NEONBFDOT
    check.section("Elementwise Operations NEON BF16", nk_cap_neonbfdot_k);
    check("each_scale_bf16_neonbfdot", test_scale<bf16_t>, nk_each_scale_bf16_neonbfdot);
    check("each_sum_bf16_neonbfdot", test_sum<bf16_t>, nk_each_sum_bf16_neonbfdot);
    check("each_blend_bf16_neonbfdot", test_blend<bf16_t>, nk_each_blend_bf16_neonbfdot);
    check("each_fma_bf16_neonbfdot", test_fma<bf16_t>, nk_each_fma_bf16_neonbfdot);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_HASWELL
    check.section("Elementwise Operations Haswell", nk_cap_haswell_k);
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
    check("each_sum_i8_haswell", test_sum<i8_t>, nk_each_sum_i8_haswell);
    check("each_sum_u8_haswell", test_sum<u8_t>, nk_each_sum_u8_haswell);
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
    check("each_rmsnorm_f32_haswell", test_rmsnorm<f32_t>, nk_each_rmsnorm_f32_haswell);
    check("each_rmsnorm_bf16_haswell", test_rmsnorm<bf16_t>, nk_each_rmsnorm_bf16_haswell);
    check("each_rmsnorm_e4m3_haswell", test_rmsnorm<e4m3_t>, nk_each_rmsnorm_e4m3_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    check.section("Elementwise Operations Skylake", nk_cap_skylake_k);
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
    check("each_scale_i8_skylake", test_scale<i8_t>, nk_each_scale_i8_skylake);
    check("each_blend_i8_skylake", test_blend<i8_t>, nk_each_blend_i8_skylake);
    check("each_fma_i8_skylake", test_fma<i8_t>, nk_each_fma_i8_skylake);
    check("each_scale_u8_skylake", test_scale<u8_t>, nk_each_scale_u8_skylake);
    check("each_blend_u8_skylake", test_blend<u8_t>, nk_each_blend_u8_skylake);
    check("each_fma_u8_skylake", test_fma<u8_t>, nk_each_fma_u8_skylake);
    check("each_rmsnorm_f32_skylake", test_rmsnorm<f32_t>, nk_each_rmsnorm_f32_skylake);
    check("each_rmsnorm_bf16_skylake", test_rmsnorm<bf16_t>, nk_each_rmsnorm_bf16_skylake);
    check("each_rmsnorm_e4m3_skylake", test_rmsnorm<e4m3_t>, nk_each_rmsnorm_e4m3_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
    check.section("Elementwise Operations Ice Lake", nk_cap_icelake_k);
    check("each_sum_i8_icelake", test_sum<i8_t>, nk_each_sum_i8_icelake);
    check("each_sum_u8_icelake", test_sum<u8_t>, nk_each_sum_u8_icelake);
    check("each_sum_i16_icelake", test_sum<i16_t>, nk_each_sum_i16_icelake);
    check("each_sum_u16_icelake", test_sum<u16_t>, nk_each_sum_u16_icelake);
    check("each_sum_i32_icelake", test_sum<i32_t>, nk_each_sum_i32_icelake);
    check("each_sum_u32_icelake", test_sum<u32_t>, nk_each_sum_u32_icelake);
    check("each_sum_i64_icelake", test_sum<i64_t>, nk_each_sum_i64_icelake);
    check("each_sum_u64_icelake", test_sum<u64_t>, nk_each_sum_u64_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
    check.section("Elementwise Operations Genoa", nk_cap_genoa_k);
    check("each_rmsnorm_bf16_genoa", test_rmsnorm<bf16_t>, nk_each_rmsnorm_bf16_genoa);
    check("each_rmsnorm_e4m3_genoa", test_rmsnorm<e4m3_t>, nk_each_rmsnorm_e4m3_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SAPPHIRE
    check.section("Elementwise Operations Sapphire", nk_cap_sapphire_k);
    check("each_sum_f16_sapphire", test_sum<f16_t>, nk_each_sum_f16_sapphire);
    check("each_sum_e4m3_sapphire", test_sum<e4m3_t>, nk_each_sum_e4m3_sapphire);
#endif // NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_RVV
    check.section("Elementwise Operations RVV", nk_cap_rvv_k);
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
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
    check.section("Elementwise Operations V128 Relaxed", nk_cap_v128relaxed_k);
    check("each_scale_f32_v128relaxed", test_scale<f32_t>, nk_each_scale_f32_v128relaxed);
    check("each_blend_f32_v128relaxed", test_blend<f32_t>, nk_each_blend_f32_v128relaxed);
    check("each_fma_f32_v128relaxed", test_fma<f32_t>, nk_each_fma_f32_v128relaxed);
    check("each_sum_f16_v128relaxed", test_sum<f16_t>, nk_each_sum_f16_v128relaxed);
    check("each_scale_f16_v128relaxed", test_scale<f16_t>, nk_each_scale_f16_v128relaxed);
    check("each_blend_f16_v128relaxed", test_blend<f16_t>, nk_each_blend_f16_v128relaxed);
    check("each_fma_f16_v128relaxed", test_fma<f16_t>, nk_each_fma_f16_v128relaxed);
    check("each_scale_bf16_v128relaxed", test_scale<bf16_t>, nk_each_scale_bf16_v128relaxed);
    check("each_blend_bf16_v128relaxed", test_blend<bf16_t>, nk_each_blend_bf16_v128relaxed);
    check("each_fma_bf16_v128relaxed", test_fma<bf16_t>, nk_each_fma_bf16_v128relaxed);
    check("each_scale_i8_v128relaxed", test_scale<i8_t>, nk_each_scale_i8_v128relaxed);
    check("each_blend_i8_v128relaxed", test_blend<i8_t>, nk_each_blend_i8_v128relaxed);
    check("each_fma_i8_v128relaxed", test_fma<i8_t>, nk_each_fma_i8_v128relaxed);
    check("each_scale_u8_v128relaxed", test_scale<u8_t>, nk_each_scale_u8_v128relaxed);
    check("each_blend_u8_v128relaxed", test_blend<u8_t>, nk_each_blend_u8_v128relaxed);
    check("each_fma_u8_v128relaxed", test_fma<u8_t>, nk_each_fma_u8_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_V128
    check.section("Elementwise Operations V128", nk_cap_v128_k);
    check("each_sum_f32_v128", test_sum<f32_t>, nk_each_sum_f32_v128);
    check("each_sum_bf16_v128", test_sum<bf16_t>, nk_each_sum_bf16_v128);
    check("each_sum_i8_v128", test_sum<i8_t>, nk_each_sum_i8_v128);
    check("each_sum_u8_v128", test_sum<u8_t>, nk_each_sum_u8_v128);
#endif // NUMKONG_TARGET_V128
}

} // namespace ashvardanian::numkong::test
