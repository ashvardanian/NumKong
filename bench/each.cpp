/**
 *  @file bench/each.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Elementwise operations benchmarks, fma, blend, sum, scale.
 */

#include <cstring> // std::memset

#include "numkong/capabilities.h" // nk_kernel_kind_t
#include "numkong/each.h"         // nk_each_swiglu_*, nk_each_rmsnorm_*

#include "cross.hpp"

namespace ashvardanian::numkong::bench {

#if NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE

nk_status_t sum_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result, void *) {
    int const ni = static_cast<int>(n);
    cblas_scopy(ni, a, 1, result, 1);
    cblas_saxpy(ni, 1.0f, b, 1, result, 1);
    return nk_success_k;
}

nk_status_t sum_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result, void *) {
    int const ni = static_cast<int>(n);
    cblas_dcopy(ni, a, 1, result, 1);
    cblas_daxpy(ni, 1.0, b, 1, result, 1);
    return nk_success_k;
}

nk_status_t blend_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                nk_f32_t const *beta, nk_f32_t *result, void *) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(nk_f32_t));
    if (*alpha != 0) cblas_saxpy(ni, *alpha, a, 1, result, 1);
    if (*beta != 0) cblas_saxpy(ni, *beta, b, 1, result, 1);
    return nk_success_k;
}

nk_status_t blend_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                nk_f64_t const *beta, nk_f64_t *result, void *) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(nk_f64_t));
    if (*alpha != 0) cblas_daxpy(ni, *alpha, a, 1, result, 1);
    if (*beta != 0) cblas_daxpy(ni, *beta, b, 1, result, 1);
    return nk_success_k;
}

#endif // NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE

void bench_each(environment_t const &env) {
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t i16_k = nk_i16_k;
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t i64_k = nk_i64_k;
    constexpr nk_dtype_t u64_k = nk_u64_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t f32c_k = nk_f32c_k;
    constexpr nk_dtype_t f64c_k = nk_f64c_k;

    constexpr nk_kernel_kind_t fma_k = nk_kernel_each_fma_k;
    constexpr nk_kernel_kind_t blend_k = nk_kernel_each_blend_k;
    constexpr nk_kernel_kind_t sum_k = nk_kernel_each_sum_k;
    constexpr nk_kernel_kind_t scale_k = nk_kernel_each_scale_k;

#if NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE
    section(env, "Elementwise Operations External Baselines", nk_cap_serial_k);
    run_each<f32_k, sum_k, f32_k>(env, "sum_f32_with_blas", sum_f32_with_blas);
    run_each<f32_k, blend_k, f32_k>(env, "each_blend_f32_with_blas", blend_f32_with_blas);
    run_each<f64_k, sum_k, f64_k>(env, "sum_f64_with_blas", sum_f64_with_blas);
    run_each<f64_k, blend_k, f64_k>(env, "each_blend_f64_with_blas", blend_f64_with_blas);
#endif

#if NUMKONG_TARGET_NEON
    if (section(env, "Elementwise Operations NEON", nk_cap_neon_k)) {
        // f64
        run_each<f64_k, sum_k, f64_k>(env, "each_sum_f64_neon", nk_each_sum_f64_neon);
        run_each<f64_k, scale_k, f64_k>(env, "each_scale_f64_neon", nk_each_scale_f64_neon);
        run_each<f64_k, blend_k, f64_k>(env, "each_blend_f64_neon", nk_each_blend_f64_neon);
        run_each<f64_k, fma_k, f64_k>(env, "each_fma_f64_neon", nk_each_fma_f64_neon);
        // f32
        run_each<f32_k, sum_k, f32_k>(env, "each_sum_f32_neon", nk_each_sum_f32_neon);
        run_each<f32_k, scale_k, f32_k>(env, "each_scale_f32_neon", nk_each_scale_f32_neon);
        run_each<f32_k, blend_k, f32_k>(env, "each_blend_f32_neon", nk_each_blend_f32_neon);
        run_each<f32_k, fma_k, f32_k>(env, "each_fma_f32_neon", nk_each_fma_f32_neon);
        // f16
        run_each<f16_k, scale_k, f32_k>(env, "each_scale_f16_neon", nk_each_scale_f16_neon);
        run_each<f16_k, blend_k, f32_k>(env, "each_blend_f16_neon", nk_each_blend_f16_neon);
        run_each<f16_k, fma_k, f32_k>(env, "each_fma_f16_neon", nk_each_fma_f16_neon);
        // e4m3
        run_each<e4m3_k, sum_k, f32_k>(env, "each_sum_e4m3_neon", nk_each_sum_e4m3_neon);
        run_each<e4m3_k, scale_k, f32_k>(env, "each_scale_e4m3_neon", nk_each_scale_e4m3_neon);
        run_each<e4m3_k, blend_k, f32_k>(env, "each_blend_e4m3_neon", nk_each_blend_e4m3_neon);
        run_each<e4m3_k, fma_k, f32_k>(env, "each_fma_e4m3_neon", nk_each_fma_e4m3_neon);
        // e5m2
        run_each<e5m2_k, sum_k, f32_k>(env, "each_sum_e5m2_neon", nk_each_sum_e5m2_neon);
        run_each<e5m2_k, scale_k, f32_k>(env, "each_scale_e5m2_neon", nk_each_scale_e5m2_neon);
        run_each<e5m2_k, blend_k, f32_k>(env, "each_blend_e5m2_neon", nk_each_blend_e5m2_neon);
        run_each<e5m2_k, fma_k, f32_k>(env, "each_fma_e5m2_neon", nk_each_fma_e5m2_neon);
        // u8, i8
        run_each<u8_k, sum_k, f32_k>(env, "each_sum_u8_neon", nk_each_sum_u8_neon);
        run_each<u8_k, scale_k, f32_k>(env, "each_scale_u8_neon", nk_each_scale_u8_neon);
        run_each<u8_k, blend_k, f32_k>(env, "each_blend_u8_neon", nk_each_blend_u8_neon);
        run_each<i8_k, sum_k, f32_k>(env, "each_sum_i8_neon", nk_each_sum_i8_neon);
        run_each<i8_k, scale_k, f32_k>(env, "each_scale_i8_neon", nk_each_scale_i8_neon);
        run_each<i8_k, blend_k, f32_k>(env, "each_blend_i8_neon", nk_each_blend_i8_neon);
        // i16, u16
        run_each<i16_k, sum_k, f32_k>(env, "each_sum_i16_neon", nk_each_sum_i16_neon);
        run_each<i16_k, scale_k, f32_k>(env, "each_scale_i16_neon", nk_each_scale_i16_neon);
        run_each<i16_k, fma_k, f32_k>(env, "each_fma_i16_neon", nk_each_fma_i16_neon);
        run_each<u16_k, sum_k, f32_k>(env, "each_sum_u16_neon", nk_each_sum_u16_neon);
        run_each<u16_k, scale_k, f32_k>(env, "each_scale_u16_neon", nk_each_scale_u16_neon);
        run_each<u16_k, fma_k, f32_k>(env, "each_fma_u16_neon", nk_each_fma_u16_neon);
        // i32, u32
        run_each<i32_k, sum_k, f64_k>(env, "each_sum_i32_neon", nk_each_sum_i32_neon);
        run_each<i32_k, scale_k, f64_k>(env, "each_scale_i32_neon", nk_each_scale_i32_neon);
        run_each<i32_k, fma_k, f64_k>(env, "each_fma_i32_neon", nk_each_fma_i32_neon);
        run_each<u32_k, sum_k, f64_k>(env, "each_sum_u32_neon", nk_each_sum_u32_neon);
        run_each<u32_k, scale_k, f64_k>(env, "each_scale_u32_neon", nk_each_scale_u32_neon);
        run_each<u32_k, fma_k, f64_k>(env, "each_fma_u32_neon", nk_each_fma_u32_neon);
        // i64, u64
        run_each<i64_k, sum_k, f64_k>(env, "each_sum_i64_neon", nk_each_sum_i64_neon);
        run_each<i64_k, scale_k, f64_k>(env, "each_scale_i64_neon", nk_each_scale_i64_neon);
        run_each<i64_k, fma_k, f64_k>(env, "each_fma_i64_neon", nk_each_fma_i64_neon);
        run_each<u64_k, sum_k, f64_k>(env, "each_sum_u64_neon", nk_each_sum_u64_neon);
        run_each<u64_k, scale_k, f64_k>(env, "each_scale_u64_neon", nk_each_scale_u64_neon);
        run_each<u64_k, fma_k, f64_k>(env, "each_fma_u64_neon", nk_each_fma_u64_neon);
        // complex
        run_each<f32c_k, scale_k, f32c_k>(env, "each_scale_f32c_neon", nk_each_scale_f32c_neon);
        run_each<f32c_k, blend_k, f32c_k>(env, "each_blend_f32c_neon", nk_each_blend_f32c_neon);
        run_each<f32c_k, fma_k, f32c_k>(env, "each_fma_f32c_neon", nk_each_fma_f32c_neon);
        run_each<f64c_k, scale_k, f64c_k>(env, "each_scale_f64c_neon", nk_each_scale_f64c_neon);
        run_each<f64c_k, blend_k, f64c_k>(env, "each_blend_f64c_neon", nk_each_blend_f64c_neon);
        run_each<f64c_k, fma_k, f64c_k>(env, "each_fma_f64c_neon", nk_each_fma_f64c_neon);
    }
#endif

#if NUMKONG_TARGET_NEONHALF
    if (section(env, "Elementwise Operations NEON HALF", nk_cap_neonhalf_k)) {
        run_each<f16_k, sum_k, f32_k>(env, "each_sum_f16_neonhalf", nk_each_sum_f16_neonhalf);
    }
#endif

#if NUMKONG_TARGET_NEONBFDOT
    if (section(env, "Elementwise Operations NEON BF16", nk_cap_neonbfdot_k)) {
        run_each<bf16_k, sum_k, f32_k>(env, "each_sum_bf16_neonbfdot", nk_each_sum_bf16_neonbfdot);
        run_each<bf16_k, scale_k, f32_k>(env, "each_scale_bf16_neonbfdot", nk_each_scale_bf16_neonbfdot);
        run_each<bf16_k, blend_k, f32_k>(env, "each_blend_bf16_neonbfdot", nk_each_blend_bf16_neonbfdot);
        run_each<bf16_k, fma_k, f32_k>(env, "each_fma_bf16_neonbfdot", nk_each_fma_bf16_neonbfdot);
    }
#endif

#if NUMKONG_TARGET_HASWELL
    if (section(env, "Elementwise Operations Haswell", nk_cap_haswell_k)) {
        // f64
        run_each<f64_k, sum_k, f64_k>(env, "each_sum_f64_haswell", nk_each_sum_f64_haswell);
        run_each<f64_k, scale_k, f64_k>(env, "each_scale_f64_haswell", nk_each_scale_f64_haswell);
        run_each<f64_k, blend_k, f64_k>(env, "each_blend_f64_haswell", nk_each_blend_f64_haswell);
        run_each<f64_k, fma_k, f64_k>(env, "each_fma_f64_haswell", nk_each_fma_f64_haswell);
        // f32
        run_each<f32_k, sum_k, f32_k>(env, "each_sum_f32_haswell", nk_each_sum_f32_haswell);
        run_each<f32_k, scale_k, f32_k>(env, "each_scale_f32_haswell", nk_each_scale_f32_haswell);
        run_each<f32_k, blend_k, f32_k>(env, "each_blend_f32_haswell", nk_each_blend_f32_haswell);
        run_each<f32_k, fma_k, f32_k>(env, "each_fma_f32_haswell", nk_each_fma_f32_haswell);
        // f16
        run_each<f16_k, sum_k, f32_k>(env, "each_sum_f16_haswell", nk_each_sum_f16_haswell);
        run_each<f16_k, scale_k, f32_k>(env, "each_scale_f16_haswell", nk_each_scale_f16_haswell);
        run_each<f16_k, blend_k, f32_k>(env, "each_blend_f16_haswell", nk_each_blend_f16_haswell);
        run_each<f16_k, fma_k, f32_k>(env, "each_fma_f16_haswell", nk_each_fma_f16_haswell);
        // bf16
        run_each<bf16_k, sum_k, f32_k>(env, "each_sum_bf16_haswell", nk_each_sum_bf16_haswell);
        run_each<bf16_k, scale_k, f32_k>(env, "each_scale_bf16_haswell", nk_each_scale_bf16_haswell);
        run_each<bf16_k, blend_k, f32_k>(env, "each_blend_bf16_haswell", nk_each_blend_bf16_haswell);
        run_each<bf16_k, fma_k, f32_k>(env, "each_fma_bf16_haswell", nk_each_fma_bf16_haswell);
        run_swiglu<f32_k>(env, "each_swiglu_f32_haswell", nk_each_swiglu_f32_haswell);
        run_swiglu<bf16_k>(env, "each_swiglu_bf16_haswell", nk_each_swiglu_bf16_haswell);
        run_swiglu<e4m3_k>(env, "each_swiglu_e4m3_haswell", nk_each_swiglu_e4m3_haswell);
        run_rmsnorm<f32_k>(env, "each_rmsnorm_f32_haswell", nk_each_rmsnorm_f32_haswell);
        run_rmsnorm<bf16_k>(env, "each_rmsnorm_bf16_haswell", nk_each_rmsnorm_bf16_haswell);
        run_rmsnorm<e4m3_k>(env, "each_rmsnorm_e4m3_haswell", nk_each_rmsnorm_e4m3_haswell);
        // e4m3
        run_each<e4m3_k, sum_k, f32_k>(env, "each_sum_e4m3_haswell", nk_each_sum_e4m3_haswell);
        run_each<e4m3_k, scale_k, f32_k>(env, "each_scale_e4m3_haswell", nk_each_scale_e4m3_haswell);
        run_each<e4m3_k, blend_k, f32_k>(env, "each_blend_e4m3_haswell", nk_each_blend_e4m3_haswell);
        run_each<e4m3_k, fma_k, f32_k>(env, "each_fma_e4m3_haswell", nk_each_fma_e4m3_haswell);
        // e5m2
        run_each<e5m2_k, sum_k, f32_k>(env, "each_sum_e5m2_haswell", nk_each_sum_e5m2_haswell);
        run_each<e5m2_k, scale_k, f32_k>(env, "each_scale_e5m2_haswell", nk_each_scale_e5m2_haswell);
        run_each<e5m2_k, blend_k, f32_k>(env, "each_blend_e5m2_haswell", nk_each_blend_e5m2_haswell);
        run_each<e5m2_k, fma_k, f32_k>(env, "each_fma_e5m2_haswell", nk_each_fma_e5m2_haswell);
        // i8, u8
        run_each<i8_k, sum_k, f32_k>(env, "each_sum_i8_haswell", nk_each_sum_i8_haswell);
        run_each<i8_k, scale_k, f32_k>(env, "each_scale_i8_haswell", nk_each_scale_i8_haswell);
        run_each<i8_k, blend_k, f32_k>(env, "each_blend_i8_haswell", nk_each_blend_i8_haswell);
        run_each<i8_k, fma_k, f32_k>(env, "each_fma_i8_haswell", nk_each_fma_i8_haswell);
        run_each<u8_k, sum_k, f32_k>(env, "each_sum_u8_haswell", nk_each_sum_u8_haswell);
        run_each<u8_k, scale_k, f32_k>(env, "each_scale_u8_haswell", nk_each_scale_u8_haswell);
        run_each<u8_k, blend_k, f32_k>(env, "each_blend_u8_haswell", nk_each_blend_u8_haswell);
        run_each<u8_k, fma_k, f32_k>(env, "each_fma_u8_haswell", nk_each_fma_u8_haswell);
        // i16, u16
        run_each<i16_k, sum_k, f32_k>(env, "each_sum_i16_haswell", nk_each_sum_i16_haswell);
        run_each<i16_k, scale_k, f32_k>(env, "each_scale_i16_haswell", nk_each_scale_i16_haswell);
        run_each<i16_k, fma_k, f32_k>(env, "each_fma_i16_haswell", nk_each_fma_i16_haswell);
        run_each<u16_k, sum_k, f32_k>(env, "each_sum_u16_haswell", nk_each_sum_u16_haswell);
        run_each<u16_k, scale_k, f32_k>(env, "each_scale_u16_haswell", nk_each_scale_u16_haswell);
        run_each<u16_k, fma_k, f32_k>(env, "each_fma_u16_haswell", nk_each_fma_u16_haswell);
        // i32, u32
        run_each<i32_k, sum_k, f64_k>(env, "each_sum_i32_haswell", nk_each_sum_i32_haswell);
        run_each<i32_k, scale_k, f64_k>(env, "each_scale_i32_haswell", nk_each_scale_i32_haswell);
        run_each<i32_k, fma_k, f64_k>(env, "each_fma_i32_haswell", nk_each_fma_i32_haswell);
        run_each<u32_k, sum_k, f64_k>(env, "each_sum_u32_haswell", nk_each_sum_u32_haswell);
        run_each<u32_k, scale_k, f64_k>(env, "each_scale_u32_haswell", nk_each_scale_u32_haswell);
        run_each<u32_k, fma_k, f64_k>(env, "each_fma_u32_haswell", nk_each_fma_u32_haswell);
        // complex
        run_each<f32c_k, scale_k, f32c_k>(env, "each_scale_f32c_haswell", nk_each_scale_f32c_haswell);
        run_each<f32c_k, blend_k, f32c_k>(env, "each_blend_f32c_haswell", nk_each_blend_f32c_haswell);
        run_each<f32c_k, fma_k, f32c_k>(env, "each_fma_f32c_haswell", nk_each_fma_f32c_haswell);
        run_each<f64c_k, scale_k, f64c_k>(env, "each_scale_f64c_haswell", nk_each_scale_f64c_haswell);
        run_each<f64c_k, blend_k, f64c_k>(env, "each_blend_f64c_haswell", nk_each_blend_f64c_haswell);
        run_each<f64c_k, fma_k, f64c_k>(env, "each_fma_f64c_haswell", nk_each_fma_f64c_haswell);
    }
#endif

#if NUMKONG_TARGET_SKYLAKE
    if (section(env, "Elementwise Operations Skylake", nk_cap_skylake_k)) {
        // f64
        run_each<f64_k, sum_k, f64_k>(env, "each_sum_f64_skylake", nk_each_sum_f64_skylake);
        run_each<f64_k, scale_k, f64_k>(env, "each_scale_f64_skylake", nk_each_scale_f64_skylake);
        run_each<f64_k, blend_k, f64_k>(env, "each_blend_f64_skylake", nk_each_blend_f64_skylake);
        run_each<f64_k, fma_k, f64_k>(env, "each_fma_f64_skylake", nk_each_fma_f64_skylake);
        // f32
        run_each<f32_k, sum_k, f32_k>(env, "each_sum_f32_skylake", nk_each_sum_f32_skylake);
        run_each<f32_k, scale_k, f32_k>(env, "each_scale_f32_skylake", nk_each_scale_f32_skylake);
        run_each<f32_k, blend_k, f32_k>(env, "each_blend_f32_skylake", nk_each_blend_f32_skylake);
        run_each<f32_k, fma_k, f32_k>(env, "each_fma_f32_skylake", nk_each_fma_f32_skylake);
        // f16
        run_each<f16_k, scale_k, f32_k>(env, "each_scale_f16_skylake", nk_each_scale_f16_skylake);
        run_each<f16_k, blend_k, f32_k>(env, "each_blend_f16_skylake", nk_each_blend_f16_skylake);
        run_each<f16_k, fma_k, f32_k>(env, "each_fma_f16_skylake", nk_each_fma_f16_skylake);
        // bf16
        run_each<bf16_k, sum_k, f32_k>(env, "each_sum_bf16_skylake", nk_each_sum_bf16_skylake);
        run_each<bf16_k, scale_k, f32_k>(env, "each_scale_bf16_skylake", nk_each_scale_bf16_skylake);
        run_each<bf16_k, blend_k, f32_k>(env, "each_blend_bf16_skylake", nk_each_blend_bf16_skylake);
        run_each<bf16_k, fma_k, f32_k>(env, "each_fma_bf16_skylake", nk_each_fma_bf16_skylake);
        run_swiglu<f32_k>(env, "each_swiglu_f32_skylake", nk_each_swiglu_f32_skylake);
        run_swiglu<bf16_k>(env, "each_swiglu_bf16_skylake", nk_each_swiglu_bf16_skylake);
        run_swiglu<e4m3_k>(env, "each_swiglu_e4m3_skylake", nk_each_swiglu_e4m3_skylake);
        run_rmsnorm<f32_k>(env, "each_rmsnorm_f32_skylake", nk_each_rmsnorm_f32_skylake);
        run_rmsnorm<bf16_k>(env, "each_rmsnorm_bf16_skylake", nk_each_rmsnorm_bf16_skylake);
        run_rmsnorm<e4m3_k>(env, "each_rmsnorm_e4m3_skylake", nk_each_rmsnorm_e4m3_skylake);
        // e4m3
        run_each<e4m3_k, sum_k, f32_k>(env, "each_sum_e4m3_skylake", nk_each_sum_e4m3_skylake);
        run_each<e4m3_k, scale_k, f32_k>(env, "each_scale_e4m3_skylake", nk_each_scale_e4m3_skylake);
        run_each<e4m3_k, blend_k, f32_k>(env, "each_blend_e4m3_skylake", nk_each_blend_e4m3_skylake);
        run_each<e4m3_k, fma_k, f32_k>(env, "each_fma_e4m3_skylake", nk_each_fma_e4m3_skylake);
        // e5m2
        run_each<e5m2_k, sum_k, f32_k>(env, "each_sum_e5m2_skylake", nk_each_sum_e5m2_skylake);
        run_each<e5m2_k, scale_k, f32_k>(env, "each_scale_e5m2_skylake", nk_each_scale_e5m2_skylake);
        run_each<e5m2_k, blend_k, f32_k>(env, "each_blend_e5m2_skylake", nk_each_blend_e5m2_skylake);
        run_each<e5m2_k, fma_k, f32_k>(env, "each_fma_e5m2_skylake", nk_each_fma_e5m2_skylake);
        // i8, u8
        run_each<i8_k, scale_k, f32_k>(env, "each_scale_i8_skylake", nk_each_scale_i8_skylake);
        run_each<i8_k, blend_k, f32_k>(env, "each_blend_i8_skylake", nk_each_blend_i8_skylake);
        run_each<i8_k, fma_k, f32_k>(env, "each_fma_i8_skylake", nk_each_fma_i8_skylake);
        run_each<u8_k, scale_k, f32_k>(env, "each_scale_u8_skylake", nk_each_scale_u8_skylake);
        run_each<u8_k, blend_k, f32_k>(env, "each_blend_u8_skylake", nk_each_blend_u8_skylake);
        run_each<u8_k, fma_k, f32_k>(env, "each_fma_u8_skylake", nk_each_fma_u8_skylake);
        // i16, u16
        run_each<i16_k, scale_k, f32_k>(env, "each_scale_i16_skylake", nk_each_scale_i16_skylake);
        run_each<i16_k, fma_k, f32_k>(env, "each_fma_i16_skylake", nk_each_fma_i16_skylake);
        run_each<u16_k, scale_k, f32_k>(env, "each_scale_u16_skylake", nk_each_scale_u16_skylake);
        run_each<u16_k, fma_k, f32_k>(env, "each_fma_u16_skylake", nk_each_fma_u16_skylake);
        // i32, u32
        run_each<i32_k, scale_k, f64_k>(env, "each_scale_i32_skylake", nk_each_scale_i32_skylake);
        run_each<i32_k, fma_k, f64_k>(env, "each_fma_i32_skylake", nk_each_fma_i32_skylake);
        run_each<u32_k, scale_k, f64_k>(env, "each_scale_u32_skylake", nk_each_scale_u32_skylake);
        run_each<u32_k, fma_k, f64_k>(env, "each_fma_u32_skylake", nk_each_fma_u32_skylake);
        // i64, u64
        run_each<i64_k, scale_k, f64_k>(env, "each_scale_i64_skylake", nk_each_scale_i64_skylake);
        run_each<i64_k, fma_k, f64_k>(env, "each_fma_i64_skylake", nk_each_fma_i64_skylake);
        run_each<u64_k, scale_k, f64_k>(env, "each_scale_u64_skylake", nk_each_scale_u64_skylake);
        run_each<u64_k, fma_k, f64_k>(env, "each_fma_u64_skylake", nk_each_fma_u64_skylake);
        // complex
        run_each<f32c_k, scale_k, f32c_k>(env, "each_scale_f32c_skylake", nk_each_scale_f32c_skylake);
        run_each<f32c_k, blend_k, f32c_k>(env, "each_blend_f32c_skylake", nk_each_blend_f32c_skylake);
        run_each<f32c_k, fma_k, f32c_k>(env, "each_fma_f32c_skylake", nk_each_fma_f32c_skylake);
        run_each<f64c_k, scale_k, f64c_k>(env, "each_scale_f64c_skylake", nk_each_scale_f64c_skylake);
        run_each<f64c_k, blend_k, f64c_k>(env, "each_blend_f64c_skylake", nk_each_blend_f64c_skylake);
        run_each<f64c_k, fma_k, f64c_k>(env, "each_fma_f64c_skylake", nk_each_fma_f64c_skylake);
    }
#endif

#if NUMKONG_TARGET_ICELAKE
    if (section(env, "Elementwise Operations Ice Lake", nk_cap_icelake_k)) {
        run_each<i8_k, sum_k, f32_k>(env, "each_sum_i8_icelake", nk_each_sum_i8_icelake);
        run_each<u8_k, sum_k, f32_k>(env, "each_sum_u8_icelake", nk_each_sum_u8_icelake);
        run_each<i16_k, sum_k, f32_k>(env, "each_sum_i16_icelake", nk_each_sum_i16_icelake);
        run_each<u16_k, sum_k, f32_k>(env, "each_sum_u16_icelake", nk_each_sum_u16_icelake);
        run_each<i32_k, sum_k, f32_k>(env, "each_sum_i32_icelake", nk_each_sum_i32_icelake);
        run_each<u32_k, sum_k, f32_k>(env, "each_sum_u32_icelake", nk_each_sum_u32_icelake);
        run_each<i64_k, sum_k, f64_k>(env, "each_sum_i64_icelake", nk_each_sum_i64_icelake);
        run_each<u64_k, sum_k, f64_k>(env, "each_sum_u64_icelake", nk_each_sum_u64_icelake);
    }
#endif

#if NUMKONG_TARGET_GENOA
    if (section(env, "Elementwise Operations Genoa", nk_cap_genoa_k)) {
        run_rmsnorm<bf16_k>(env, "each_rmsnorm_bf16_genoa", nk_each_rmsnorm_bf16_genoa);
        run_rmsnorm<e4m3_k>(env, "each_rmsnorm_e4m3_genoa", nk_each_rmsnorm_e4m3_genoa);
    }
#endif

#if NUMKONG_TARGET_SAPPHIRE
    if (section(env, "Elementwise Operations Sapphire", nk_cap_sapphire_k)) {
        // f16, e4m3
        run_each<f16_k, sum_k, f32_k>(env, "each_sum_f16_sapphire", nk_each_sum_f16_sapphire);
        run_each<e4m3_k, sum_k, f32_k>(env, "each_sum_e4m3_sapphire", nk_each_sum_e4m3_sapphire);
    }
#endif

#if NUMKONG_TARGET_RVV
    if (section(env, "Elementwise Operations RVV", nk_cap_rvv_k)) {
        run_each<f64_k, scale_k, f64_k>(env, "each_scale_f64_rvv", nk_each_scale_f64_rvv);
        run_each<f64_k, fma_k, f64_k>(env, "each_fma_f64_rvv", nk_each_fma_f64_rvv);
        run_each<f64_k, blend_k, f64_k>(env, "each_blend_f64_rvv", nk_each_blend_f64_rvv);
        run_each<f32_k, scale_k, f32_k>(env, "each_scale_f32_rvv", nk_each_scale_f32_rvv);
        run_each<f32_k, fma_k, f32_k>(env, "each_fma_f32_rvv", nk_each_fma_f32_rvv);
        run_each<f32_k, blend_k, f32_k>(env, "each_blend_f32_rvv", nk_each_blend_f32_rvv);
        run_each<f16_k, scale_k, f32_k>(env, "each_scale_f16_rvv", nk_each_scale_f16_rvv);
        run_each<f16_k, fma_k, f32_k>(env, "each_fma_f16_rvv", nk_each_fma_f16_rvv);
        run_each<f16_k, blend_k, f32_k>(env, "each_blend_f16_rvv", nk_each_blend_f16_rvv);
        run_each<bf16_k, scale_k, f32_k>(env, "each_scale_bf16_rvv", nk_each_scale_bf16_rvv);
        run_each<bf16_k, fma_k, f32_k>(env, "each_fma_bf16_rvv", nk_each_fma_bf16_rvv);
        run_each<bf16_k, blend_k, f32_k>(env, "each_blend_bf16_rvv", nk_each_blend_bf16_rvv);
        run_each<i8_k, scale_k, f32_k>(env, "each_scale_i8_rvv", nk_each_scale_i8_rvv);
        run_each<i8_k, fma_k, f32_k>(env, "each_fma_i8_rvv", nk_each_fma_i8_rvv);
        run_each<i8_k, blend_k, f32_k>(env, "each_blend_i8_rvv", nk_each_blend_i8_rvv);
        run_each<u8_k, scale_k, f32_k>(env, "each_scale_u8_rvv", nk_each_scale_u8_rvv);
        run_each<u8_k, fma_k, f32_k>(env, "each_fma_u8_rvv", nk_each_fma_u8_rvv);
        run_each<u8_k, blend_k, f32_k>(env, "each_blend_u8_rvv", nk_each_blend_u8_rvv);
    }
#endif

#if NUMKONG_TARGET_V128RELAXED
    if (section(env, "Elementwise Operations V128 Relaxed", nk_cap_v128relaxed_k)) {
        run_each<f32_k, scale_k, f32_k>(env, "each_scale_f32_v128relaxed", nk_each_scale_f32_v128relaxed);
        run_each<f32_k, blend_k, f32_k>(env, "each_blend_f32_v128relaxed", nk_each_blend_f32_v128relaxed);
        run_each<f32_k, fma_k, f32_k>(env, "each_fma_f32_v128relaxed", nk_each_fma_f32_v128relaxed);
        run_each<f16_k, sum_k, f32_k>(env, "each_sum_f16_v128relaxed", nk_each_sum_f16_v128relaxed);
        run_each<f16_k, scale_k, f32_k>(env, "each_scale_f16_v128relaxed", nk_each_scale_f16_v128relaxed);
        run_each<f16_k, blend_k, f32_k>(env, "each_blend_f16_v128relaxed", nk_each_blend_f16_v128relaxed);
        run_each<f16_k, fma_k, f32_k>(env, "each_fma_f16_v128relaxed", nk_each_fma_f16_v128relaxed);
        run_each<bf16_k, scale_k, f32_k>(env, "each_scale_bf16_v128relaxed", nk_each_scale_bf16_v128relaxed);
        run_each<bf16_k, blend_k, f32_k>(env, "each_blend_bf16_v128relaxed", nk_each_blend_bf16_v128relaxed);
        run_each<bf16_k, fma_k, f32_k>(env, "each_fma_bf16_v128relaxed", nk_each_fma_bf16_v128relaxed);
        run_each<i8_k, scale_k, f32_k>(env, "each_scale_i8_v128relaxed", nk_each_scale_i8_v128relaxed);
        run_each<i8_k, blend_k, f32_k>(env, "each_blend_i8_v128relaxed", nk_each_blend_i8_v128relaxed);
        run_each<i8_k, fma_k, f32_k>(env, "each_fma_i8_v128relaxed", nk_each_fma_i8_v128relaxed);
        run_each<u8_k, scale_k, f32_k>(env, "each_scale_u8_v128relaxed", nk_each_scale_u8_v128relaxed);
        run_each<u8_k, blend_k, f32_k>(env, "each_blend_u8_v128relaxed", nk_each_blend_u8_v128relaxed);
        run_each<u8_k, fma_k, f32_k>(env, "each_fma_u8_v128relaxed", nk_each_fma_u8_v128relaxed);
    }
#endif

#if NUMKONG_TARGET_V128
    if (section(env, "Elementwise Operations V128", nk_cap_v128_k)) {
        run_each<f32_k, sum_k, f32_k>(env, "each_sum_f32_v128", nk_each_sum_f32_v128);
        run_each<bf16_k, sum_k, f32_k>(env, "each_sum_bf16_v128", nk_each_sum_bf16_v128);
        run_each<i8_k, sum_k, f32_k>(env, "each_sum_i8_v128", nk_each_sum_i8_v128);
        run_each<u8_k, sum_k, f32_k>(env, "each_sum_u8_v128", nk_each_sum_u8_v128);
    }
#endif

    // Serial fallbacks — f64
    section(env, "Elementwise Operations Serial", nk_cap_serial_k);
    run_each<f64_k, sum_k, f64_k>(env, "each_sum_f64_serial", nk_each_sum_f64_serial);
    run_each<f64_k, scale_k, f64_k>(env, "each_scale_f64_serial", nk_each_scale_f64_serial);
    run_each<f64_k, blend_k, f64_k>(env, "each_blend_f64_serial", nk_each_blend_f64_serial);
    run_each<f64_k, fma_k, f64_k>(env, "each_fma_f64_serial", nk_each_fma_f64_serial);
    // Serial fallbacks — f32
    run_each<f32_k, sum_k, f32_k>(env, "each_sum_f32_serial", nk_each_sum_f32_serial);
    run_each<f32_k, scale_k, f32_k>(env, "each_scale_f32_serial", nk_each_scale_f32_serial);
    run_each<f32_k, blend_k, f32_k>(env, "each_blend_f32_serial", nk_each_blend_f32_serial);
    run_each<f32_k, fma_k, f32_k>(env, "each_fma_f32_serial", nk_each_fma_f32_serial);
    // Serial fallbacks — f16, bf16
    run_each<f16_k, sum_k, f32_k>(env, "each_sum_f16_serial", nk_each_sum_f16_serial);
    run_each<f16_k, scale_k, f32_k>(env, "each_scale_f16_serial", nk_each_scale_f16_serial);
    run_each<f16_k, blend_k, f32_k>(env, "each_blend_f16_serial", nk_each_blend_f16_serial);
    run_each<f16_k, fma_k, f32_k>(env, "each_fma_f16_serial", nk_each_fma_f16_serial);
    run_each<bf16_k, sum_k, f32_k>(env, "each_sum_bf16_serial", nk_each_sum_bf16_serial);
    run_each<bf16_k, scale_k, f32_k>(env, "each_scale_bf16_serial", nk_each_scale_bf16_serial);
    run_each<bf16_k, blend_k, f32_k>(env, "each_blend_bf16_serial", nk_each_blend_bf16_serial);
    run_each<bf16_k, fma_k, f32_k>(env, "each_fma_bf16_serial", nk_each_fma_bf16_serial);
    run_swiglu<f32_k>(env, "each_swiglu_f32_serial", nk_each_swiglu_f32_serial);
    run_swiglu<bf16_k>(env, "each_swiglu_bf16_serial", nk_each_swiglu_bf16_serial);
    run_swiglu<e4m3_k>(env, "each_swiglu_e4m3_serial", nk_each_swiglu_e4m3_serial);
    run_rmsnorm<f32_k>(env, "each_rmsnorm_f32_serial", nk_each_rmsnorm_f32_serial);
    run_rmsnorm<bf16_k>(env, "each_rmsnorm_bf16_serial", nk_each_rmsnorm_bf16_serial);
    run_rmsnorm<e4m3_k>(env, "each_rmsnorm_e4m3_serial", nk_each_rmsnorm_e4m3_serial);
    // Serial fallbacks — e4m3, e5m2, e2m3, e3m2
    run_each<e4m3_k, sum_k, f32_k>(env, "each_sum_e4m3_serial", nk_each_sum_e4m3_serial);
    run_each<e4m3_k, scale_k, f32_k>(env, "each_scale_e4m3_serial", nk_each_scale_e4m3_serial);
    run_each<e4m3_k, blend_k, f32_k>(env, "each_blend_e4m3_serial", nk_each_blend_e4m3_serial);
    run_each<e4m3_k, fma_k, f32_k>(env, "each_fma_e4m3_serial", nk_each_fma_e4m3_serial);
    run_each<e5m2_k, sum_k, f32_k>(env, "each_sum_e5m2_serial", nk_each_sum_e5m2_serial);
    run_each<e5m2_k, scale_k, f32_k>(env, "each_scale_e5m2_serial", nk_each_scale_e5m2_serial);
    run_each<e5m2_k, blend_k, f32_k>(env, "each_blend_e5m2_serial", nk_each_blend_e5m2_serial);
    run_each<e5m2_k, fma_k, f32_k>(env, "each_fma_e5m2_serial", nk_each_fma_e5m2_serial);
    run_each<e2m3_k, sum_k, f32_k>(env, "each_sum_e2m3_serial", nk_each_sum_e2m3_serial);
    run_each<e2m3_k, scale_k, f32_k>(env, "each_scale_e2m3_serial", nk_each_scale_e2m3_serial);
    run_each<e2m3_k, blend_k, f32_k>(env, "each_blend_e2m3_serial", nk_each_blend_e2m3_serial);
    run_each<e2m3_k, fma_k, f32_k>(env, "each_fma_e2m3_serial", nk_each_fma_e2m3_serial);
    run_each<e3m2_k, sum_k, f32_k>(env, "each_sum_e3m2_serial", nk_each_sum_e3m2_serial);
    run_each<e3m2_k, scale_k, f32_k>(env, "each_scale_e3m2_serial", nk_each_scale_e3m2_serial);
    run_each<e3m2_k, blend_k, f32_k>(env, "each_blend_e3m2_serial", nk_each_blend_e3m2_serial);
    run_each<e3m2_k, fma_k, f32_k>(env, "each_fma_e3m2_serial", nk_each_fma_e3m2_serial);
    // Serial fallbacks — i8, u8
    run_each<i8_k, sum_k, f32_k>(env, "each_sum_i8_serial", nk_each_sum_i8_serial);
    run_each<i8_k, scale_k, f32_k>(env, "each_scale_i8_serial", nk_each_scale_i8_serial);
    run_each<i8_k, blend_k, f32_k>(env, "each_blend_i8_serial", nk_each_blend_i8_serial);
    run_each<i8_k, fma_k, f32_k>(env, "each_fma_i8_serial", nk_each_fma_i8_serial);
    run_each<u8_k, sum_k, f32_k>(env, "each_sum_u8_serial", nk_each_sum_u8_serial);
    run_each<u8_k, scale_k, f32_k>(env, "each_scale_u8_serial", nk_each_scale_u8_serial);
    run_each<u8_k, blend_k, f32_k>(env, "each_blend_u8_serial", nk_each_blend_u8_serial);
    run_each<u8_k, fma_k, f32_k>(env, "each_fma_u8_serial", nk_each_fma_u8_serial);
    // Serial fallbacks — i16, u16
    run_each<i16_k, sum_k, f32_k>(env, "each_sum_i16_serial", nk_each_sum_i16_serial);
    run_each<i16_k, scale_k, f32_k>(env, "each_scale_i16_serial", nk_each_scale_i16_serial);
    run_each<i16_k, blend_k, f32_k>(env, "each_blend_i16_serial", nk_each_blend_i16_serial);
    run_each<i16_k, fma_k, f32_k>(env, "each_fma_i16_serial", nk_each_fma_i16_serial);
    run_each<u16_k, sum_k, f32_k>(env, "each_sum_u16_serial", nk_each_sum_u16_serial);
    run_each<u16_k, scale_k, f32_k>(env, "each_scale_u16_serial", nk_each_scale_u16_serial);
    run_each<u16_k, blend_k, f32_k>(env, "each_blend_u16_serial", nk_each_blend_u16_serial);
    run_each<u16_k, fma_k, f32_k>(env, "each_fma_u16_serial", nk_each_fma_u16_serial);
    // Serial fallbacks — i32, u32
    run_each<i32_k, sum_k, f64_k>(env, "each_sum_i32_serial", nk_each_sum_i32_serial);
    run_each<i32_k, scale_k, f64_k>(env, "each_scale_i32_serial", nk_each_scale_i32_serial);
    run_each<i32_k, blend_k, f64_k>(env, "each_blend_i32_serial", nk_each_blend_i32_serial);
    run_each<i32_k, fma_k, f64_k>(env, "each_fma_i32_serial", nk_each_fma_i32_serial);
    run_each<u32_k, sum_k, f64_k>(env, "each_sum_u32_serial", nk_each_sum_u32_serial);
    run_each<u32_k, scale_k, f64_k>(env, "each_scale_u32_serial", nk_each_scale_u32_serial);
    run_each<u32_k, blend_k, f64_k>(env, "each_blend_u32_serial", nk_each_blend_u32_serial);
    run_each<u32_k, fma_k, f64_k>(env, "each_fma_u32_serial", nk_each_fma_u32_serial);
    // Serial fallbacks — i64, u64
    run_each<i64_k, sum_k, f64_k>(env, "each_sum_i64_serial", nk_each_sum_i64_serial);
    run_each<i64_k, scale_k, f64_k>(env, "each_scale_i64_serial", nk_each_scale_i64_serial);
    run_each<i64_k, blend_k, f64_k>(env, "each_blend_i64_serial", nk_each_blend_i64_serial);
    run_each<i64_k, fma_k, f64_k>(env, "each_fma_i64_serial", nk_each_fma_i64_serial);
    run_each<u64_k, sum_k, f64_k>(env, "each_sum_u64_serial", nk_each_sum_u64_serial);
    run_each<u64_k, scale_k, f64_k>(env, "each_scale_u64_serial", nk_each_scale_u64_serial);
    run_each<u64_k, blend_k, f64_k>(env, "each_blend_u64_serial", nk_each_blend_u64_serial);
    run_each<u64_k, fma_k, f64_k>(env, "each_fma_u64_serial", nk_each_fma_u64_serial);
    // Serial fallbacks — complex
    run_each<f32c_k, sum_k, f32c_k>(env, "each_sum_f32c_serial", nk_each_sum_f32c_serial);
    run_each<f32c_k, scale_k, f32c_k>(env, "each_scale_f32c_serial", nk_each_scale_f32c_serial);
    run_each<f32c_k, blend_k, f32c_k>(env, "each_blend_f32c_serial", nk_each_blend_f32c_serial);
    run_each<f32c_k, fma_k, f32c_k>(env, "each_fma_f32c_serial", nk_each_fma_f32c_serial);
    run_each<f64c_k, sum_k, f64c_k>(env, "each_sum_f64c_serial", nk_each_sum_f64c_serial);
    run_each<f64c_k, scale_k, f64c_k>(env, "each_scale_f64c_serial", nk_each_scale_f64c_serial);
    run_each<f64c_k, blend_k, f64c_k>(env, "each_blend_f64c_serial", nk_each_blend_f64c_serial);
    run_each<f64c_k, fma_k, f64c_k>(env, "each_fma_f64c_serial", nk_each_fma_f64c_serial);
}

} // namespace ashvardanian::numkong::bench
