/**
 *  @file bench/probability.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief KL-divergence and Jensen-Shannon distance benchmarks.
 */

#include "numkong/probability.h"

#include "harness.hpp"

using namespace ashvardanian::numkong::bench;

void bench_probability() {
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;

#if NUMKONG_TARGET_NEON
    run_dense<f32_k, f64_k>("kld_f32_neon", nk_kld_f32_neon);
    run_dense<f32_k, f64_k>("jsd_f32_neon", nk_jsd_f32_neon);
    run_dense<f16_k, f32_k>("kld_f16_neon", nk_kld_f16_neon);
    run_dense<f16_k, f32_k>("jsd_f16_neon", nk_jsd_f16_neon);
#endif

#if NUMKONG_TARGET_HASWELL
    run_dense<f16_k, f32_k>("kld_f16_haswell", nk_kld_f16_haswell);
    run_dense<f16_k, f32_k>("jsd_f16_haswell", nk_jsd_f16_haswell);
    run_dense<f64_k, f64_k>("kld_f64_haswell", nk_kld_f64_haswell);
    run_dense<f64_k, f64_k>("jsd_f64_haswell", nk_jsd_f64_haswell);
#endif

#if NUMKONG_TARGET_SKYLAKE
    run_dense<f32_k, f64_k>("kld_f32_skylake", nk_kld_f32_skylake);
    run_dense<f32_k, f64_k>("jsd_f32_skylake", nk_jsd_f32_skylake);
    run_dense<f64_k, f64_k>("kld_f64_skylake", nk_kld_f64_skylake);
    run_dense<f64_k, f64_k>("jsd_f64_skylake", nk_jsd_f64_skylake);
    run_dense<f16_k, f32_k>("kld_f16_skylake", nk_kld_f16_skylake);
    run_dense<f16_k, f32_k>("jsd_f16_skylake", nk_jsd_f16_skylake);
#endif

#if NUMKONG_TARGET_RVV
    run_dense<f32_k, f64_k>("kld_f32_rvv", nk_kld_f32_rvv);
    run_dense<f32_k, f64_k>("jsd_f32_rvv", nk_jsd_f32_rvv);
    run_dense<f64_k, f64_k>("kld_f64_rvv", nk_kld_f64_rvv);
    run_dense<f64_k, f64_k>("jsd_f64_rvv", nk_jsd_f64_rvv);
    run_dense<f16_k, f32_k>("kld_f16_rvv", nk_kld_f16_rvv);
    run_dense<f16_k, f32_k>("jsd_f16_rvv", nk_jsd_f16_rvv);
    run_dense<bf16_k, f32_k>("kld_bf16_rvv", nk_kld_bf16_rvv);
    run_dense<bf16_k, f32_k>("jsd_bf16_rvv", nk_jsd_bf16_rvv);
#endif
    // Serial fallbacks
    run_dense<bf16_k, f32_k>("kld_bf16_serial", nk_kld_bf16_serial);
    run_dense<bf16_k, f32_k>("jsd_bf16_serial", nk_jsd_bf16_serial);
    run_dense<f16_k, f32_k>("kld_f16_serial", nk_kld_f16_serial);
    run_dense<f16_k, f32_k>("jsd_f16_serial", nk_jsd_f16_serial);
    run_dense<f32_k, f64_k>("kld_f32_serial", nk_kld_f32_serial);
    run_dense<f32_k, f64_k>("jsd_f32_serial", nk_jsd_f32_serial);
    run_dense<f64_k, f64_k>("kld_f64_serial", nk_kld_f64_serial);
    run_dense<f64_k, f64_k>("jsd_f64_serial", nk_jsd_f64_serial);
}
