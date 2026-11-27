/**
 *  @file test/each.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief Elementwise operations tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "each.hpp"                 // `test_scale`, `test_sum`, `test_blend`, `test_fma`, `test_rmsnorm`
#include "numkong/each.hpp"         // `nk::add`, `nk::scale`, `nk::blend`, `nk::fma`
#include "numkong/trigonometry.hpp" // `nk::sin`, `nk::cos`, `nk::atan` wrappers

namespace ashvardanian::numkong::test {

/** Smoke-test for the tensor-shaped trig wrappers @c nk::sin, @c cos and @c atan, running
 *  allocating and into-span variants on a small zero tensor, just exercising the dispatch paths,
 *  not the numerical accuracy, which the scenarios of `each.hpp` cover. */
void run_each_cases() {
    auto const serial = nk_cap_serial_k;

    run_kernel_case("nk_each_scale_f32_serial", test_scale<f32_t>, serial, nk_each_scale_f32_serial);
    run_kernel_case("nk_each_sum_f32_serial", test_sum<f32_t>, serial, nk_each_sum_f32_serial);
    run_kernel_case("nk_each_blend_f32_serial", test_blend<f32_t>, serial, nk_each_blend_f32_serial);
    run_kernel_case("nk_each_fma_f32_serial", test_fma<f32_t>, serial, nk_each_fma_f32_serial);
    run_kernel_case("nk_each_swiglu_f32_serial", test_swiglu<f32_t>, serial, nk_each_swiglu_f32_serial);
    run_kernel_case("nk_each_rmsnorm_f32_serial", test_rmsnorm<f32_t>, serial, nk_each_rmsnorm_f32_serial);
    run_kernel_case("nk_each_scale_e4m3_serial", test_scale<e4m3_t>, serial, nk_each_scale_e4m3_serial);
    run_kernel_case("nk_each_scale_e5m2_serial", test_scale<e5m2_t>, serial, nk_each_scale_e5m2_serial);
    run_kernel_case("nk_each_sum_e4m3_serial", test_sum<e4m3_t>, serial, nk_each_sum_e4m3_serial);
    run_kernel_case("nk_each_sum_e5m2_serial", test_sum<e5m2_t>, serial, nk_each_sum_e5m2_serial);
    run_kernel_case("nk_each_blend_e4m3_serial", test_blend<e4m3_t>, serial, nk_each_blend_e4m3_serial);
    run_kernel_case("nk_each_blend_e5m2_serial", test_blend<e5m2_t>, serial, nk_each_blend_e5m2_serial);
    run_kernel_case("nk_each_fma_e4m3_serial", test_fma<e4m3_t>, serial, nk_each_fma_e4m3_serial);
    run_kernel_case("nk_each_swiglu_e4m3_serial", test_swiglu<e4m3_t>, serial, nk_each_swiglu_e4m3_serial);
    run_kernel_case("nk_each_rmsnorm_e4m3_serial", test_rmsnorm<e4m3_t>, serial, nk_each_rmsnorm_e4m3_serial);
    run_kernel_case("nk_each_fma_e5m2_serial", test_fma<e5m2_t>, serial, nk_each_fma_e5m2_serial);
    run_kernel_case("nk_each_sum_f32c_serial", test_sum<f32c_t>, serial, nk_each_sum_f32c_serial);
    run_kernel_case("nk_each_sum_f64c_serial", test_sum<f64c_t>, serial, nk_each_sum_f64c_serial);
    run_kernel_case("nk_each_scale_f32c_serial", test_scale<f32c_t>, serial, nk_each_scale_f32c_serial);
    run_kernel_case("nk_each_scale_f64c_serial", test_scale<f64c_t>, serial, nk_each_scale_f64c_serial);
    run_kernel_case("nk_each_blend_f32c_serial", test_blend<f32c_t>, serial, nk_each_blend_f32c_serial);
    run_kernel_case("nk_each_blend_f64c_serial", test_blend<f64c_t>, serial, nk_each_blend_f64c_serial);
    run_kernel_case("nk_each_fma_f32c_serial", test_fma<f32c_t>, serial, nk_each_fma_f32c_serial);
    run_kernel_case("nk_each_fma_f64c_serial", test_fma<f64c_t>, serial, nk_each_fma_f64c_serial);
    run_kernel_case("nk_each_blend_f16_serial", test_blend<f16_t>, serial, nk_each_blend_f16_serial);
    run_kernel_case("nk_each_blend_i8_serial", test_blend<i8_t>, serial, nk_each_blend_i8_serial);
    run_kernel_case("nk_each_blend_u8_serial", test_blend<u8_t>, serial, nk_each_blend_u8_serial);
    run_kernel_case("nk_each_fma_f16_serial", test_fma<f16_t>, serial, nk_each_fma_f16_serial);
    run_kernel_case("nk_each_fma_i8_serial", test_fma<i8_t>, serial, nk_each_fma_i8_serial);
    run_kernel_case("nk_each_fma_u8_serial", test_fma<u8_t>, serial, nk_each_fma_u8_serial);
    run_kernel_case("nk_each_sum_f64_serial", test_sum<f64_t>, serial, nk_each_sum_f64_serial);
    run_kernel_case("nk_each_scale_f64_serial", test_scale<f64_t>, serial, nk_each_scale_f64_serial);
    run_kernel_case("nk_each_blend_f64_serial", test_blend<f64_t>, serial, nk_each_blend_f64_serial);
    run_kernel_case("nk_each_fma_f64_serial", test_fma<f64_t>, serial, nk_each_fma_f64_serial);
    run_kernel_case("nk_each_sum_bf16_serial", test_sum<bf16_t>, serial, nk_each_sum_bf16_serial);
    run_kernel_case("nk_each_scale_bf16_serial", test_scale<bf16_t>, serial, nk_each_scale_bf16_serial);
    run_kernel_case("nk_each_blend_bf16_serial", test_blend<bf16_t>, serial, nk_each_blend_bf16_serial);
    run_kernel_case("nk_each_fma_bf16_serial", test_fma<bf16_t>, serial, nk_each_fma_bf16_serial);
    run_kernel_case("nk_each_swiglu_f16_serial", test_swiglu<f16_t>, serial, nk_each_swiglu_f16_serial);
    run_kernel_case("nk_each_swiglu_bf16_serial", test_swiglu<bf16_t>, serial, nk_each_swiglu_bf16_serial);
    run_kernel_case("nk_each_rmsnorm_f16_serial", test_rmsnorm<f16_t>, serial, nk_each_rmsnorm_f16_serial);
    run_kernel_case("nk_each_rmsnorm_bf16_serial", test_rmsnorm<bf16_t>, serial, nk_each_rmsnorm_bf16_serial);
    run_kernel_case("nk_each_rmscast_bf16_serial", test_rmsnorm<bf16_t, host_backend_t, f32_t>, serial,
                    nk_each_rmscast_bf16_serial);
    run_kernel_case("nk_each_rmscast_f16_serial", test_rmsnorm<f16_t, host_backend_t, f32_t>, serial,
                    nk_each_rmscast_f16_serial);
    run_kernel_case("nk_each_rmscast_e4m3_serial", test_rmsnorm<e4m3_t, host_backend_t, f32_t>, serial,
                    nk_each_rmscast_e4m3_serial);
    run_kernel_case("nk_each_rmscast_e5m2_serial", test_rmsnorm<e5m2_t, host_backend_t, f32_t>, serial,
                    nk_each_rmscast_e5m2_serial);
    run_kernel_case("nk_each_rmscast_e2m3_serial", test_rmsnorm<e2m3_t, host_backend_t, f32_t>, serial,
                    nk_each_rmscast_e2m3_serial);
    run_kernel_case("nk_each_rmscast_e3m2_serial", test_rmsnorm<e3m2_t, host_backend_t, f32_t>, serial,
                    nk_each_rmscast_e3m2_serial);
    run_kernel_case("nk_each_rmscast_f32_serial", test_rmsnorm<f32_t, host_backend_t, f64_t>, serial,
                    nk_each_rmscast_f32_serial);
    run_kernel_case("nk_each_rmscast_i8_serial", test_rmsnorm<i8_t, host_backend_t, i32_t>, serial,
                    nk_each_rmscast_i8_serial);
    run_kernel_case("nk_each_rmscast_u8_serial", test_rmsnorm<u8_t, host_backend_t, u32_t>, serial,
                    nk_each_rmscast_u8_serial);
    run_kernel_case("nk_each_sum_f16_serial", test_sum<f16_t>, serial, nk_each_sum_f16_serial);
    run_kernel_case("nk_each_scale_f16_serial", test_scale<f16_t>, serial, nk_each_scale_f16_serial);

    run_kernel_case("nk_each_scale_f32_best", test_scale<f32_t>, serial, cpu_best<nk_each_scale_f32_best>);
    run_kernel_case("nk_each_sum_f32_best", test_sum<f32_t>, serial, cpu_best<nk_each_sum_f32_best>);
    run_kernel_case("nk_each_blend_f32_best", test_blend<f32_t>, serial, cpu_best<nk_each_blend_f32_best>);
    run_kernel_case("nk_each_fma_f32_best", test_fma<f32_t>, serial, cpu_best<nk_each_fma_f32_best>);
    run_kernel_case("nk_each_swiglu_f32_best", test_swiglu<f32_t>, serial, cpu_best<nk_each_swiglu_f32_best>);
    run_kernel_case("nk_each_scale_e4m3_best", test_scale<e4m3_t>, serial, cpu_best<nk_each_scale_e4m3_best>);
    run_kernel_case("nk_each_scale_e5m2_best", test_scale<e5m2_t>, serial, cpu_best<nk_each_scale_e5m2_best>);
    run_kernel_case("nk_each_sum_e4m3_best", test_sum<e4m3_t>, serial, cpu_best<nk_each_sum_e4m3_best>);
    run_kernel_case("nk_each_sum_e5m2_best", test_sum<e5m2_t>, serial, cpu_best<nk_each_sum_e5m2_best>);
    run_kernel_case("nk_each_blend_e4m3_best", test_blend<e4m3_t>, serial, cpu_best<nk_each_blend_e4m3_best>);
    run_kernel_case("nk_each_blend_e5m2_best", test_blend<e5m2_t>, serial, cpu_best<nk_each_blend_e5m2_best>);
    run_kernel_case("nk_each_fma_e4m3_best", test_fma<e4m3_t>, serial, cpu_best<nk_each_fma_e4m3_best>);
    run_kernel_case("nk_each_swiglu_e4m3_best", test_swiglu<e4m3_t>, serial, cpu_best<nk_each_swiglu_e4m3_best>);
    run_kernel_case("nk_each_fma_e5m2_best", test_fma<e5m2_t>, serial, cpu_best<nk_each_fma_e5m2_best>);
    run_kernel_case("nk_each_sum_f32c_best", test_sum<f32c_t>, serial, cpu_best<nk_each_sum_f32c_best>);
    run_kernel_case("nk_each_sum_f64c_best", test_sum<f64c_t>, serial, cpu_best<nk_each_sum_f64c_best>);
    run_kernel_case("nk_each_scale_f32c_best", test_scale<f32c_t>, serial, cpu_best<nk_each_scale_f32c_best>);
    run_kernel_case("nk_each_scale_f64c_best", test_scale<f64c_t>, serial, cpu_best<nk_each_scale_f64c_best>);
    run_kernel_case("nk_each_blend_f32c_best", test_blend<f32c_t>, serial, cpu_best<nk_each_blend_f32c_best>);
    run_kernel_case("nk_each_blend_f64c_best", test_blend<f64c_t>, serial, cpu_best<nk_each_blend_f64c_best>);
    run_kernel_case("nk_each_fma_f32c_best", test_fma<f32c_t>, serial, cpu_best<nk_each_fma_f32c_best>);
    run_kernel_case("nk_each_fma_f64c_best", test_fma<f64c_t>, serial, cpu_best<nk_each_fma_f64c_best>);
    run_kernel_case("nk_each_swiglu_f16_best", test_swiglu<f16_t>, serial, cpu_best<nk_each_swiglu_f16_best>);
    run_kernel_case("nk_each_rmsnorm_f16_best", test_rmsnorm<f16_t>, serial, cpu_best<nk_each_rmsnorm_f16_best>);
    run_kernel_case("nk_each_swiglu_bf16_best", test_swiglu<bf16_t>, serial, cpu_best<nk_each_swiglu_bf16_best>);

#if NUMKONG_TARGET_NEON
    auto const neon = nk_cap_neon_k;
    // f64
    run_kernel_case("nk_each_sum_f64_neon", test_sum<f64_t>, neon, nk_each_sum_f64_neon);
    run_kernel_case("nk_each_scale_f64_neon", test_scale<f64_t>, neon, nk_each_scale_f64_neon);
    run_kernel_case("nk_each_blend_f64_neon", test_blend<f64_t>, neon, nk_each_blend_f64_neon);
    run_kernel_case("nk_each_fma_f64_neon", test_fma<f64_t>, neon, nk_each_fma_f64_neon);
    // f32
    run_kernel_case("nk_each_sum_f32_neon", test_sum<f32_t>, neon, nk_each_sum_f32_neon);
    run_kernel_case("nk_each_scale_f32_neon", test_scale<f32_t>, neon, nk_each_scale_f32_neon);
    run_kernel_case("nk_each_blend_f32_neon", test_blend<f32_t>, neon, nk_each_blend_f32_neon);
    run_kernel_case("nk_each_fma_f32_neon", test_fma<f32_t>, neon, nk_each_fma_f32_neon);
    run_kernel_case("nk_each_swiglu_f32_neon", test_swiglu<f32_t>, neon, nk_each_swiglu_f32_neon);
    run_kernel_case("nk_each_rmsnorm_f32_neon", test_rmsnorm<f32_t>, neon, nk_each_rmsnorm_f32_neon);
    // f16, bf16
    run_kernel_case("nk_each_scale_f16_neon", test_scale<f16_t>, neon, nk_each_scale_f16_neon);
    run_kernel_case("nk_each_blend_f16_neon", test_blend<f16_t>, neon, nk_each_blend_f16_neon);
    run_kernel_case("nk_each_fma_f16_neon", test_fma<f16_t>, neon, nk_each_fma_f16_neon);
    run_kernel_case("nk_each_swiglu_f16_neon", test_swiglu<f16_t>, neon, nk_each_swiglu_f16_neon);
    run_kernel_case("nk_each_swiglu_bf16_neon", test_swiglu<bf16_t>, neon, nk_each_swiglu_bf16_neon);
    run_kernel_case("nk_each_rmsnorm_f16_neon", test_rmsnorm<f16_t>, neon, nk_each_rmsnorm_f16_neon);
    run_kernel_case("nk_each_rmsnorm_bf16_neon", test_rmsnorm<bf16_t>, neon, nk_each_rmsnorm_bf16_neon);
    // e4m3, e5m2
    run_kernel_case("nk_each_sum_e4m3_neon", test_sum<e4m3_t>, neon, nk_each_sum_e4m3_neon);
    run_kernel_case("nk_each_scale_e4m3_neon", test_scale<e4m3_t>, neon, nk_each_scale_e4m3_neon);
    run_kernel_case("nk_each_blend_e4m3_neon", test_blend<e4m3_t>, neon, nk_each_blend_e4m3_neon);
    run_kernel_case("nk_each_fma_e4m3_neon", test_fma<e4m3_t>, neon, nk_each_fma_e4m3_neon);
    run_kernel_case("nk_each_swiglu_e4m3_neon", test_swiglu<e4m3_t>, neon, nk_each_swiglu_e4m3_neon);
    run_kernel_case("nk_each_rmsnorm_e4m3_neon", test_rmsnorm<e4m3_t>, neon, nk_each_rmsnorm_e4m3_neon);
    run_kernel_case("nk_each_rmscast_bf16_neon", test_rmsnorm<bf16_t, host_backend_t, f32_t>, neon,
                    nk_each_rmscast_bf16_neon);
    run_kernel_case("nk_each_rmscast_f16_neon", test_rmsnorm<f16_t, host_backend_t, f32_t>, neon,
                    nk_each_rmscast_f16_neon);
    run_kernel_case("nk_each_rmscast_e4m3_neon", test_rmsnorm<e4m3_t, host_backend_t, f32_t>, neon,
                    nk_each_rmscast_e4m3_neon);
    run_kernel_case("nk_each_rmscast_f32_neon", test_rmsnorm<f32_t, host_backend_t, f64_t>, neon,
                    nk_each_rmscast_f32_neon);
    run_kernel_case("nk_each_sum_e5m2_neon", test_sum<e5m2_t>, neon, nk_each_sum_e5m2_neon);
    run_kernel_case("nk_each_scale_e5m2_neon", test_scale<e5m2_t>, neon, nk_each_scale_e5m2_neon);
    run_kernel_case("nk_each_blend_e5m2_neon", test_blend<e5m2_t>, neon, nk_each_blend_e5m2_neon);
    run_kernel_case("nk_each_fma_e5m2_neon", test_fma<e5m2_t>, neon, nk_each_fma_e5m2_neon);
    // e2m3, e3m2
    run_kernel_case("nk_each_sum_e2m3_neon", test_sum<e2m3_t>, neon, nk_each_sum_e2m3_neon);
    run_kernel_case("nk_each_scale_e2m3_neon", test_scale<e2m3_t>, neon, nk_each_scale_e2m3_neon);
    run_kernel_case("nk_each_blend_e2m3_neon", test_blend<e2m3_t>, neon, nk_each_blend_e2m3_neon);
    run_kernel_case("nk_each_fma_e2m3_neon", test_fma<e2m3_t>, neon, nk_each_fma_e2m3_neon);
    run_kernel_case("nk_each_sum_e3m2_neon", test_sum<e3m2_t>, neon, nk_each_sum_e3m2_neon);
    run_kernel_case("nk_each_scale_e3m2_neon", test_scale<e3m2_t>, neon, nk_each_scale_e3m2_neon);
    run_kernel_case("nk_each_blend_e3m2_neon", test_blend<e3m2_t>, neon, nk_each_blend_e3m2_neon);
    run_kernel_case("nk_each_fma_e3m2_neon", test_fma<e3m2_t>, neon, nk_each_fma_e3m2_neon);
    // u8, i8
    run_kernel_case("nk_each_sum_u8_neon", test_sum<u8_t>, neon, nk_each_sum_u8_neon);
    run_kernel_case("nk_each_scale_u8_neon", test_scale<u8_t>, neon, nk_each_scale_u8_neon);
    run_kernel_case("nk_each_blend_u8_neon", test_blend<u8_t>, neon, nk_each_blend_u8_neon);
    run_kernel_case("nk_each_fma_u8_neon", test_fma<u8_t>, neon, nk_each_fma_u8_neon);
    run_kernel_case("nk_each_sum_i8_neon", test_sum<i8_t>, neon, nk_each_sum_i8_neon);
    run_kernel_case("nk_each_scale_i8_neon", test_scale<i8_t>, neon, nk_each_scale_i8_neon);
    run_kernel_case("nk_each_blend_i8_neon", test_blend<i8_t>, neon, nk_each_blend_i8_neon);
    run_kernel_case("nk_each_fma_i8_neon", test_fma<i8_t>, neon, nk_each_fma_i8_neon);
    // i16, u16
    run_kernel_case("nk_each_sum_i16_neon", test_sum<i16_t>, neon, nk_each_sum_i16_neon);
    run_kernel_case("nk_each_scale_i16_neon", test_scale<i16_t>, neon, nk_each_scale_i16_neon);
    run_kernel_case("nk_each_fma_i16_neon", test_fma<i16_t>, neon, nk_each_fma_i16_neon);
    run_kernel_case("nk_each_sum_u16_neon", test_sum<u16_t>, neon, nk_each_sum_u16_neon);
    run_kernel_case("nk_each_scale_u16_neon", test_scale<u16_t>, neon, nk_each_scale_u16_neon);
    run_kernel_case("nk_each_fma_u16_neon", test_fma<u16_t>, neon, nk_each_fma_u16_neon);
    // i32, u32
    run_kernel_case("nk_each_sum_i32_neon", test_sum<i32_t>, neon, nk_each_sum_i32_neon);
    run_kernel_case("nk_each_scale_i32_neon", test_scale<i32_t>, neon, nk_each_scale_i32_neon);
    run_kernel_case("nk_each_fma_i32_neon", test_fma<i32_t>, neon, nk_each_fma_i32_neon);
    run_kernel_case("nk_each_sum_u32_neon", test_sum<u32_t>, neon, nk_each_sum_u32_neon);
    run_kernel_case("nk_each_scale_u32_neon", test_scale<u32_t>, neon, nk_each_scale_u32_neon);
    run_kernel_case("nk_each_fma_u32_neon", test_fma<u32_t>, neon, nk_each_fma_u32_neon);
    // i64, u64
    run_kernel_case("nk_each_sum_i64_neon", test_sum<i64_t>, neon, nk_each_sum_i64_neon);
    run_kernel_case("nk_each_scale_i64_neon", test_scale<i64_t>, neon, nk_each_scale_i64_neon);
    run_kernel_case("nk_each_fma_i64_neon", test_fma<i64_t>, neon, nk_each_fma_i64_neon);
    run_kernel_case("nk_each_sum_u64_neon", test_sum<u64_t>, neon, nk_each_sum_u64_neon);
    run_kernel_case("nk_each_scale_u64_neon", test_scale<u64_t>, neon, nk_each_scale_u64_neon);
    run_kernel_case("nk_each_fma_u64_neon", test_fma<u64_t>, neon, nk_each_fma_u64_neon);
    // complex
    run_kernel_case("nk_each_scale_f32c_neon", test_scale<f32c_t>, neon, nk_each_scale_f32c_neon);
    run_kernel_case("nk_each_blend_f32c_neon", test_blend<f32c_t>, neon, nk_each_blend_f32c_neon);
    run_kernel_case("nk_each_fma_f32c_neon", test_fma<f32c_t>, neon, nk_each_fma_f32c_neon);
    run_kernel_case("nk_each_scale_f64c_neon", test_scale<f64c_t>, neon, nk_each_scale_f64c_neon);
    run_kernel_case("nk_each_blend_f64c_neon", test_blend<f64c_t>, neon, nk_each_blend_f64c_neon);
    run_kernel_case("nk_each_fma_f64c_neon", test_fma<f64c_t>, neon, nk_each_fma_f64c_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONHALF
    run_kernel_case("nk_each_sum_f16_neonhalf", test_sum<f16_t>, nk_cap_neonhalf_k, nk_each_sum_f16_neonhalf);
#endif // NUMKONG_TARGET_NEONHALF

#if NUMKONG_TARGET_NEONFHM
    run_kernel_case("nk_each_fma_f16_neonfhm", test_fma<f16_t>, nk_cap_neonfhm_k, nk_each_fma_f16_neonfhm);
    run_kernel_case("nk_each_sum_e4m3_neonfhm", test_sum<e4m3_t>, nk_cap_neonfhm_k, nk_each_sum_e4m3_neonfhm);
    run_kernel_case("nk_each_fma_e4m3_neonfhm", test_fma<e4m3_t>, nk_cap_neonfhm_k, nk_each_fma_e4m3_neonfhm);
    run_kernel_case("nk_each_sum_e5m2_neonfhm", test_sum<e5m2_t>, nk_cap_neonfhm_k, nk_each_sum_e5m2_neonfhm);
    run_kernel_case("nk_each_fma_e5m2_neonfhm", test_fma<e5m2_t>, nk_cap_neonfhm_k, nk_each_fma_e5m2_neonfhm);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONBFDOT
    auto const neonbfdot = nk_cap_neonbfdot_k;
    run_kernel_case("nk_each_scale_bf16_neonbfdot", test_scale<bf16_t>, neonbfdot, nk_each_scale_bf16_neonbfdot);
    run_kernel_case("nk_each_sum_bf16_neonbfdot", test_sum<bf16_t>, neonbfdot, nk_each_sum_bf16_neonbfdot);
    run_kernel_case("nk_each_blend_bf16_neonbfdot", test_blend<bf16_t>, neonbfdot, nk_each_blend_bf16_neonbfdot);
    run_kernel_case("nk_each_fma_bf16_neonbfdot", test_fma<bf16_t>, neonbfdot, nk_each_fma_bf16_neonbfdot);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_HASWELL
    auto const haswell = nk_cap_haswell_k;
    run_kernel_case("nk_each_scale_f32_haswell", test_scale<f32_t>, haswell, nk_each_scale_f32_haswell);
    run_kernel_case("nk_each_sum_f32_haswell", test_sum<f32_t>, haswell, nk_each_sum_f32_haswell);
    run_kernel_case("nk_each_blend_f32_haswell", test_blend<f32_t>, haswell, nk_each_blend_f32_haswell);
    run_kernel_case("nk_each_fma_f32_haswell", test_fma<f32_t>, haswell, nk_each_fma_f32_haswell);
    run_kernel_case("nk_each_scale_e4m3_haswell", test_scale<e4m3_t>, haswell, nk_each_scale_e4m3_haswell);
    run_kernel_case("nk_each_scale_e5m2_haswell", test_scale<e5m2_t>, haswell, nk_each_scale_e5m2_haswell);
    run_kernel_case("nk_each_sum_e4m3_haswell", test_sum<e4m3_t>, haswell, nk_each_sum_e4m3_haswell);
    run_kernel_case("nk_each_sum_e5m2_haswell", test_sum<e5m2_t>, haswell, nk_each_sum_e5m2_haswell);
    run_kernel_case("nk_each_blend_e4m3_haswell", test_blend<e4m3_t>, haswell, nk_each_blend_e4m3_haswell);
    run_kernel_case("nk_each_blend_e5m2_haswell", test_blend<e5m2_t>, haswell, nk_each_blend_e5m2_haswell);
    run_kernel_case("nk_each_fma_e4m3_haswell", test_fma<e4m3_t>, haswell, nk_each_fma_e4m3_haswell);
    run_kernel_case("nk_each_fma_e5m2_haswell", test_fma<e5m2_t>, haswell, nk_each_fma_e5m2_haswell);
    run_kernel_case("nk_each_scale_f32c_haswell", test_scale<f32c_t>, haswell, nk_each_scale_f32c_haswell);
    run_kernel_case("nk_each_scale_f64c_haswell", test_scale<f64c_t>, haswell, nk_each_scale_f64c_haswell);
    run_kernel_case("nk_each_blend_f32c_haswell", test_blend<f32c_t>, haswell, nk_each_blend_f32c_haswell);
    run_kernel_case("nk_each_blend_f64c_haswell", test_blend<f64c_t>, haswell, nk_each_blend_f64c_haswell);
    run_kernel_case("nk_each_fma_f32c_haswell", test_fma<f32c_t>, haswell, nk_each_fma_f32c_haswell);
    run_kernel_case("nk_each_fma_f64c_haswell", test_fma<f64c_t>, haswell, nk_each_fma_f64c_haswell);
    run_kernel_case("nk_each_blend_bf16_haswell", test_blend<bf16_t>, haswell, nk_each_blend_bf16_haswell);
    run_kernel_case("nk_each_blend_f64_haswell", test_blend<f64_t>, haswell, nk_each_blend_f64_haswell);
    run_kernel_case("nk_each_sum_i8_haswell", test_sum<i8_t>, haswell, nk_each_sum_i8_haswell);
    run_kernel_case("nk_each_sum_u8_haswell", test_sum<u8_t>, haswell, nk_each_sum_u8_haswell);
    run_kernel_case("nk_each_blend_i8_haswell", test_blend<i8_t>, haswell, nk_each_blend_i8_haswell);
    run_kernel_case("nk_each_blend_u8_haswell", test_blend<u8_t>, haswell, nk_each_blend_u8_haswell);
    run_kernel_case("nk_each_blend_f16_haswell", test_blend<f16_t>, haswell, nk_each_blend_f16_haswell);
    run_kernel_case("nk_each_fma_bf16_haswell", test_fma<bf16_t>, haswell, nk_each_fma_bf16_haswell);
    run_kernel_case("nk_each_fma_f64_haswell", test_fma<f64_t>, haswell, nk_each_fma_f64_haswell);
    run_kernel_case("nk_each_fma_i16_haswell", test_fma<i16_t>, haswell, nk_each_fma_i16_haswell);
    run_kernel_case("nk_each_fma_i8_haswell", test_fma<i8_t>, haswell, nk_each_fma_i8_haswell);
    run_kernel_case("nk_each_fma_u16_haswell", test_fma<u16_t>, haswell, nk_each_fma_u16_haswell);
    run_kernel_case("nk_each_fma_u8_haswell", test_fma<u8_t>, haswell, nk_each_fma_u8_haswell);
    run_kernel_case("nk_each_fma_f16_haswell", test_fma<f16_t>, haswell, nk_each_fma_f16_haswell);
    run_kernel_case("nk_each_scale_bf16_haswell", test_scale<bf16_t>, haswell, nk_each_scale_bf16_haswell);
    run_kernel_case("nk_each_scale_f16_haswell", test_scale<f16_t>, haswell, nk_each_scale_f16_haswell);
    run_kernel_case("nk_each_scale_f64_haswell", test_scale<f64_t>, haswell, nk_each_scale_f64_haswell);
    run_kernel_case("nk_each_scale_i16_haswell", test_scale<i16_t>, haswell, nk_each_scale_i16_haswell);
    run_kernel_case("nk_each_scale_i8_haswell", test_scale<i8_t>, haswell, nk_each_scale_i8_haswell);
    run_kernel_case("nk_each_scale_u16_haswell", test_scale<u16_t>, haswell, nk_each_scale_u16_haswell);
    run_kernel_case("nk_each_scale_u8_haswell", test_scale<u8_t>, haswell, nk_each_scale_u8_haswell);
    run_kernel_case("nk_each_swiglu_f32_haswell", test_swiglu<f32_t>, haswell, nk_each_swiglu_f32_haswell);
    run_kernel_case("nk_each_swiglu_f16_haswell", test_swiglu<f16_t>, haswell, nk_each_swiglu_f16_haswell);
    run_kernel_case("nk_each_swiglu_bf16_haswell", test_swiglu<bf16_t>, haswell, nk_each_swiglu_bf16_haswell);
    run_kernel_case("nk_each_swiglu_e4m3_haswell", test_swiglu<e4m3_t>, haswell, nk_each_swiglu_e4m3_haswell);
    run_kernel_case("nk_each_rmsnorm_f32_haswell", test_rmsnorm<f32_t>, haswell, nk_each_rmsnorm_f32_haswell);
    run_kernel_case("nk_each_rmsnorm_f16_haswell", test_rmsnorm<f16_t>, haswell, nk_each_rmsnorm_f16_haswell);
    run_kernel_case("nk_each_rmsnorm_bf16_haswell", test_rmsnorm<bf16_t>, haswell, nk_each_rmsnorm_bf16_haswell);
    run_kernel_case("nk_each_rmsnorm_e4m3_haswell", test_rmsnorm<e4m3_t>, haswell, nk_each_rmsnorm_e4m3_haswell);
    run_kernel_case("nk_each_rmscast_bf16_haswell", test_rmsnorm<bf16_t, host_backend_t, f32_t>, haswell,
                    nk_each_rmscast_bf16_haswell);
    run_kernel_case("nk_each_rmscast_e4m3_haswell", test_rmsnorm<e4m3_t, host_backend_t, f32_t>, haswell,
                    nk_each_rmscast_e4m3_haswell);
    run_kernel_case("nk_each_rmscast_f32_haswell", test_rmsnorm<f32_t, host_backend_t, f64_t>, haswell,
                    nk_each_rmscast_f32_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    auto const skylake = nk_cap_skylake_k;
    run_kernel_case("nk_each_scale_f32_skylake", test_scale<f32_t>, skylake, nk_each_scale_f32_skylake);
    run_kernel_case("nk_each_sum_f32_skylake", test_sum<f32_t>, skylake, nk_each_sum_f32_skylake);
    run_kernel_case("nk_each_blend_f32_skylake", test_blend<f32_t>, skylake, nk_each_blend_f32_skylake);
    run_kernel_case("nk_each_fma_f32_skylake", test_fma<f32_t>, skylake, nk_each_fma_f32_skylake);
    run_kernel_case("nk_each_scale_e4m3_skylake", test_scale<e4m3_t>, skylake, nk_each_scale_e4m3_skylake);
    run_kernel_case("nk_each_scale_e5m2_skylake", test_scale<e5m2_t>, skylake, nk_each_scale_e5m2_skylake);
    run_kernel_case("nk_each_sum_e4m3_skylake", test_sum<e4m3_t>, skylake, nk_each_sum_e4m3_skylake);
    run_kernel_case("nk_each_sum_e5m2_skylake", test_sum<e5m2_t>, skylake, nk_each_sum_e5m2_skylake);
    run_kernel_case("nk_each_blend_e4m3_skylake", test_blend<e4m3_t>, skylake, nk_each_blend_e4m3_skylake);
    run_kernel_case("nk_each_blend_e5m2_skylake", test_blend<e5m2_t>, skylake, nk_each_blend_e5m2_skylake);
    run_kernel_case("nk_each_fma_e4m3_skylake", test_fma<e4m3_t>, skylake, nk_each_fma_e4m3_skylake);
    run_kernel_case("nk_each_fma_e5m2_skylake", test_fma<e5m2_t>, skylake, nk_each_fma_e5m2_skylake);
    run_kernel_case("nk_each_scale_f32c_skylake", test_scale<f32c_t>, skylake, nk_each_scale_f32c_skylake);
    run_kernel_case("nk_each_scale_f64c_skylake", test_scale<f64c_t>, skylake, nk_each_scale_f64c_skylake);
    run_kernel_case("nk_each_blend_f32c_skylake", test_blend<f32c_t>, skylake, nk_each_blend_f32c_skylake);
    run_kernel_case("nk_each_blend_f64c_skylake", test_blend<f64c_t>, skylake, nk_each_blend_f64c_skylake);
    run_kernel_case("nk_each_fma_f32c_skylake", test_fma<f32c_t>, skylake, nk_each_fma_f32c_skylake);
    run_kernel_case("nk_each_fma_f64c_skylake", test_fma<f64c_t>, skylake, nk_each_fma_f64c_skylake);
    run_kernel_case("nk_each_scale_f16_skylake", test_scale<f16_t>, skylake, nk_each_scale_f16_skylake);
    run_kernel_case("nk_each_blend_f16_skylake", test_blend<f16_t>, skylake, nk_each_blend_f16_skylake);
    run_kernel_case("nk_each_fma_f16_skylake", test_fma<f16_t>, skylake, nk_each_fma_f16_skylake);
    run_kernel_case("nk_each_blend_bf16_skylake", test_blend<bf16_t>, skylake, nk_each_blend_bf16_skylake);
    run_kernel_case("nk_each_blend_f64_skylake", test_blend<f64_t>, skylake, nk_each_blend_f64_skylake);
    run_kernel_case("nk_each_fma_bf16_skylake", test_fma<bf16_t>, skylake, nk_each_fma_bf16_skylake);
    run_kernel_case("nk_each_fma_f64_skylake", test_fma<f64_t>, skylake, nk_each_fma_f64_skylake);
    run_kernel_case("nk_each_scale_i8_skylake", test_scale<i8_t>, skylake, nk_each_scale_i8_skylake);
    run_kernel_case("nk_each_blend_i8_skylake", test_blend<i8_t>, skylake, nk_each_blend_i8_skylake);
    run_kernel_case("nk_each_fma_i8_skylake", test_fma<i8_t>, skylake, nk_each_fma_i8_skylake);
    run_kernel_case("nk_each_scale_u8_skylake", test_scale<u8_t>, skylake, nk_each_scale_u8_skylake);
    run_kernel_case("nk_each_blend_u8_skylake", test_blend<u8_t>, skylake, nk_each_blend_u8_skylake);
    run_kernel_case("nk_each_fma_u8_skylake", test_fma<u8_t>, skylake, nk_each_fma_u8_skylake);
    run_kernel_case("nk_each_swiglu_f32_skylake", test_swiglu<f32_t>, skylake, nk_each_swiglu_f32_skylake);
    run_kernel_case("nk_each_swiglu_f16_skylake", test_swiglu<f16_t>, skylake, nk_each_swiglu_f16_skylake);
    run_kernel_case("nk_each_swiglu_bf16_skylake", test_swiglu<bf16_t>, skylake, nk_each_swiglu_bf16_skylake);
    run_kernel_case("nk_each_swiglu_e4m3_skylake", test_swiglu<e4m3_t>, skylake, nk_each_swiglu_e4m3_skylake);
    run_kernel_case("nk_each_rmsnorm_f32_skylake", test_rmsnorm<f32_t>, skylake, nk_each_rmsnorm_f32_skylake);
    run_kernel_case("nk_each_rmsnorm_f16_skylake", test_rmsnorm<f16_t>, skylake, nk_each_rmsnorm_f16_skylake);
    run_kernel_case("nk_each_rmsnorm_bf16_skylake", test_rmsnorm<bf16_t>, skylake, nk_each_rmsnorm_bf16_skylake);
    run_kernel_case("nk_each_rmsnorm_e4m3_skylake", test_rmsnorm<e4m3_t>, skylake, nk_each_rmsnorm_e4m3_skylake);
    run_kernel_case("nk_each_rmscast_bf16_skylake", test_rmsnorm<bf16_t, host_backend_t, f32_t>, skylake,
                    nk_each_rmscast_bf16_skylake);
    run_kernel_case("nk_each_rmscast_e4m3_skylake", test_rmsnorm<e4m3_t, host_backend_t, f32_t>, skylake,
                    nk_each_rmscast_e4m3_skylake);
    run_kernel_case("nk_each_rmscast_f32_skylake", test_rmsnorm<f32_t, host_backend_t, f64_t>, skylake,
                    nk_each_rmscast_f32_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
    run_kernel_case("nk_each_sum_i8_icelake", test_sum<i8_t>, nk_cap_icelake_k, nk_each_sum_i8_icelake);
    run_kernel_case("nk_each_sum_u8_icelake", test_sum<u8_t>, nk_cap_icelake_k, nk_each_sum_u8_icelake);
    run_kernel_case("nk_each_sum_i16_icelake", test_sum<i16_t>, nk_cap_icelake_k, nk_each_sum_i16_icelake);
    run_kernel_case("nk_each_sum_u16_icelake", test_sum<u16_t>, nk_cap_icelake_k, nk_each_sum_u16_icelake);
    run_kernel_case("nk_each_sum_i32_icelake", test_sum<i32_t>, nk_cap_icelake_k, nk_each_sum_i32_icelake);
    run_kernel_case("nk_each_sum_u32_icelake", test_sum<u32_t>, nk_cap_icelake_k, nk_each_sum_u32_icelake);
    run_kernel_case("nk_each_sum_i64_icelake", test_sum<i64_t>, nk_cap_icelake_k, nk_each_sum_i64_icelake);
    run_kernel_case("nk_each_sum_u64_icelake", test_sum<u64_t>, nk_cap_icelake_k, nk_each_sum_u64_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
    run_kernel_case("nk_each_rmsnorm_bf16_genoa", test_rmsnorm<bf16_t>, nk_cap_genoa_k, nk_each_rmsnorm_bf16_genoa);
    run_kernel_case("nk_each_rmsnorm_e4m3_genoa", test_rmsnorm<e4m3_t>, nk_cap_genoa_k, nk_each_rmsnorm_e4m3_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SAPPHIRE
    run_kernel_case("nk_each_sum_f16_sapphire", test_sum<f16_t>, nk_cap_sapphire_k, nk_each_sum_f16_sapphire);
    run_kernel_case("nk_each_sum_e4m3_sapphire", test_sum<e4m3_t>, nk_cap_sapphire_k, nk_each_sum_e4m3_sapphire);
#endif // NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_RVV
    run_kernel_case("nk_each_sum_f64_rvv", test_sum<f64_t>, nk_cap_rvv_k, nk_each_sum_f64_rvv);
    run_kernel_case("nk_each_scale_f64_rvv", test_scale<f64_t>, nk_cap_rvv_k, nk_each_scale_f64_rvv);
    run_kernel_case("nk_each_blend_f64_rvv", test_blend<f64_t>, nk_cap_rvv_k, nk_each_blend_f64_rvv);
    run_kernel_case("nk_each_fma_f64_rvv", test_fma<f64_t>, nk_cap_rvv_k, nk_each_fma_f64_rvv);
    run_kernel_case("nk_each_sum_f32_rvv", test_sum<f32_t>, nk_cap_rvv_k, nk_each_sum_f32_rvv);
    run_kernel_case("nk_each_scale_f32_rvv", test_scale<f32_t>, nk_cap_rvv_k, nk_each_scale_f32_rvv);
    run_kernel_case("nk_each_blend_f32_rvv", test_blend<f32_t>, nk_cap_rvv_k, nk_each_blend_f32_rvv);
    run_kernel_case("nk_each_fma_f32_rvv", test_fma<f32_t>, nk_cap_rvv_k, nk_each_fma_f32_rvv);
    run_kernel_case("nk_each_sum_f16_rvv", test_sum<f16_t>, nk_cap_rvv_k, nk_each_sum_f16_rvv);
    run_kernel_case("nk_each_scale_f16_rvv", test_scale<f16_t>, nk_cap_rvv_k, nk_each_scale_f16_rvv);
    run_kernel_case("nk_each_blend_f16_rvv", test_blend<f16_t>, nk_cap_rvv_k, nk_each_blend_f16_rvv);
    run_kernel_case("nk_each_fma_f16_rvv", test_fma<f16_t>, nk_cap_rvv_k, nk_each_fma_f16_rvv);
    run_kernel_case("nk_each_sum_bf16_rvv", test_sum<bf16_t>, nk_cap_rvv_k, nk_each_sum_bf16_rvv);
    run_kernel_case("nk_each_scale_bf16_rvv", test_scale<bf16_t>, nk_cap_rvv_k, nk_each_scale_bf16_rvv);
    run_kernel_case("nk_each_blend_bf16_rvv", test_blend<bf16_t>, nk_cap_rvv_k, nk_each_blend_bf16_rvv);
    run_kernel_case("nk_each_fma_bf16_rvv", test_fma<bf16_t>, nk_cap_rvv_k, nk_each_fma_bf16_rvv);
    run_kernel_case("nk_each_sum_e4m3_rvv", test_sum<e4m3_t>, nk_cap_rvv_k, nk_each_sum_e4m3_rvv);
    run_kernel_case("nk_each_scale_e4m3_rvv", test_scale<e4m3_t>, nk_cap_rvv_k, nk_each_scale_e4m3_rvv);
    run_kernel_case("nk_each_blend_e4m3_rvv", test_blend<e4m3_t>, nk_cap_rvv_k, nk_each_blend_e4m3_rvv);
    run_kernel_case("nk_each_fma_e4m3_rvv", test_fma<e4m3_t>, nk_cap_rvv_k, nk_each_fma_e4m3_rvv);
    run_kernel_case("nk_each_sum_e5m2_rvv", test_sum<e5m2_t>, nk_cap_rvv_k, nk_each_sum_e5m2_rvv);
    run_kernel_case("nk_each_scale_e5m2_rvv", test_scale<e5m2_t>, nk_cap_rvv_k, nk_each_scale_e5m2_rvv);
    run_kernel_case("nk_each_blend_e5m2_rvv", test_blend<e5m2_t>, nk_cap_rvv_k, nk_each_blend_e5m2_rvv);
    run_kernel_case("nk_each_fma_e5m2_rvv", test_fma<e5m2_t>, nk_cap_rvv_k, nk_each_fma_e5m2_rvv);
    run_kernel_case("nk_each_sum_i8_rvv", test_sum<i8_t>, nk_cap_rvv_k, nk_each_sum_i8_rvv);
    run_kernel_case("nk_each_scale_i8_rvv", test_scale<i8_t>, nk_cap_rvv_k, nk_each_scale_i8_rvv);
    run_kernel_case("nk_each_blend_i8_rvv", test_blend<i8_t>, nk_cap_rvv_k, nk_each_blend_i8_rvv);
    run_kernel_case("nk_each_fma_i8_rvv", test_fma<i8_t>, nk_cap_rvv_k, nk_each_fma_i8_rvv);
    run_kernel_case("nk_each_sum_u8_rvv", test_sum<u8_t>, nk_cap_rvv_k, nk_each_sum_u8_rvv);
    run_kernel_case("nk_each_scale_u8_rvv", test_scale<u8_t>, nk_cap_rvv_k, nk_each_scale_u8_rvv);
    run_kernel_case("nk_each_blend_u8_rvv", test_blend<u8_t>, nk_cap_rvv_k, nk_each_blend_u8_rvv);
    run_kernel_case("nk_each_fma_u8_rvv", test_fma<u8_t>, nk_cap_rvv_k, nk_each_fma_u8_rvv);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
    auto const v128relaxed = nk_cap_v128relaxed_k;
    run_kernel_case("nk_each_scale_f32_v128relaxed", test_scale<f32_t>, v128relaxed, nk_each_scale_f32_v128relaxed);
    run_kernel_case("nk_each_blend_f32_v128relaxed", test_blend<f32_t>, v128relaxed, nk_each_blend_f32_v128relaxed);
    run_kernel_case("nk_each_fma_f32_v128relaxed", test_fma<f32_t>, v128relaxed, nk_each_fma_f32_v128relaxed);
    run_kernel_case("nk_each_sum_f16_v128relaxed", test_sum<f16_t>, v128relaxed, nk_each_sum_f16_v128relaxed);
    run_kernel_case("nk_each_scale_f16_v128relaxed", test_scale<f16_t>, v128relaxed, nk_each_scale_f16_v128relaxed);
    run_kernel_case("nk_each_blend_f16_v128relaxed", test_blend<f16_t>, v128relaxed, nk_each_blend_f16_v128relaxed);
    run_kernel_case("nk_each_fma_f16_v128relaxed", test_fma<f16_t>, v128relaxed, nk_each_fma_f16_v128relaxed);
    run_kernel_case("nk_each_scale_bf16_v128relaxed", test_scale<bf16_t>, v128relaxed, nk_each_scale_bf16_v128relaxed);
    run_kernel_case("nk_each_blend_bf16_v128relaxed", test_blend<bf16_t>, v128relaxed, nk_each_blend_bf16_v128relaxed);
    run_kernel_case("nk_each_fma_bf16_v128relaxed", test_fma<bf16_t>, v128relaxed, nk_each_fma_bf16_v128relaxed);
    run_kernel_case("nk_each_scale_i8_v128relaxed", test_scale<i8_t>, v128relaxed, nk_each_scale_i8_v128relaxed);
    run_kernel_case("nk_each_blend_i8_v128relaxed", test_blend<i8_t>, v128relaxed, nk_each_blend_i8_v128relaxed);
    run_kernel_case("nk_each_fma_i8_v128relaxed", test_fma<i8_t>, v128relaxed, nk_each_fma_i8_v128relaxed);
    run_kernel_case("nk_each_scale_u8_v128relaxed", test_scale<u8_t>, v128relaxed, nk_each_scale_u8_v128relaxed);
    run_kernel_case("nk_each_blend_u8_v128relaxed", test_blend<u8_t>, v128relaxed, nk_each_blend_u8_v128relaxed);
    run_kernel_case("nk_each_fma_u8_v128relaxed", test_fma<u8_t>, v128relaxed, nk_each_fma_u8_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_V128
    run_kernel_case("nk_each_sum_f32_v128", test_sum<f32_t>, nk_cap_v128_k, nk_each_sum_f32_v128);
    run_kernel_case("nk_each_sum_bf16_v128", test_sum<bf16_t>, nk_cap_v128_k, nk_each_sum_bf16_v128);
    run_kernel_case("nk_each_sum_i8_v128", test_sum<i8_t>, nk_cap_v128_k, nk_each_sum_i8_v128);
    run_kernel_case("nk_each_sum_u8_v128", test_sum<u8_t>, nk_cap_v128_k, nk_each_sum_u8_v128);
#endif // NUMKONG_TARGET_V128
}

} // namespace ashvardanian::numkong::test
