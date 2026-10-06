/**
 *  @file bench/each_cuda.cu
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Elementwise operations benchmarks on CUDA, over the backend's token rows.
 */

#include "numkong/each.h" // `nk_each_*_cuda`, `nk_each_*_ampere`, `nk_each_*_ada`

#include "cross.hpp"
#include "harness_cuda.hpp"

namespace ashvardanian::numkong::bench {

#if NUMKONG_ARCH_CUDA_

void bench_each_cuda([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend,
                     [[maybe_unused]] nk_capability_t enabled) {
    [[maybe_unused]] constexpr nk_kernel_kind_t sum_k = nk_kernel_each_sum_k, scale_k = nk_kernel_each_scale_k;
    [[maybe_unused]] constexpr nk_kernel_kind_t blend_k = nk_kernel_each_blend_k, fma_k = nk_kernel_each_fma_k;
#if NUMKONG_TARGET_CUDA
    if (enabled & nk_cap_cuda_k) {
        run_each<nk_f64_k, sum_k, nk_f64_k>(env, "each_sum_f64_cuda", nk_each_sum_f64_cuda, backend);
        run_each<nk_f64_k, scale_k, nk_f64_k>(env, "each_scale_f64_cuda", nk_each_scale_f64_cuda, backend);
        run_each<nk_f64_k, blend_k, nk_f64_k>(env, "each_blend_f64_cuda", nk_each_blend_f64_cuda, backend);
        run_each<nk_f64_k, fma_k, nk_f64_k>(env, "each_fma_f64_cuda", nk_each_fma_f64_cuda, backend);
        run_each<nk_f32_k, sum_k, nk_f32_k>(env, "each_sum_f32_cuda", nk_each_sum_f32_cuda, backend);
        run_each<nk_f32_k, scale_k, nk_f32_k>(env, "each_scale_f32_cuda", nk_each_scale_f32_cuda, backend);
        run_each<nk_f32_k, blend_k, nk_f32_k>(env, "each_blend_f32_cuda", nk_each_blend_f32_cuda, backend);
        run_each<nk_f32_k, fma_k, nk_f32_k>(env, "each_fma_f32_cuda", nk_each_fma_f32_cuda, backend);
        run_each<nk_bf16_k, sum_k, nk_f32_k>(env, "each_sum_bf16_cuda", nk_each_sum_bf16_cuda, backend);
        run_each<nk_bf16_k, scale_k, nk_f32_k>(env, "each_scale_bf16_cuda", nk_each_scale_bf16_cuda, backend);
        run_each<nk_bf16_k, blend_k, nk_f32_k>(env, "each_blend_bf16_cuda", nk_each_blend_bf16_cuda, backend);
        run_each<nk_bf16_k, fma_k, nk_f32_k>(env, "each_fma_bf16_cuda", nk_each_fma_bf16_cuda, backend);
        run_each<nk_e4m3_k, sum_k, nk_f32_k>(env, "each_sum_e4m3_cuda", nk_each_sum_e4m3_cuda, backend);
        run_each<nk_e4m3_k, scale_k, nk_f32_k>(env, "each_scale_e4m3_cuda", nk_each_scale_e4m3_cuda, backend);
        run_each<nk_e4m3_k, blend_k, nk_f32_k>(env, "each_blend_e4m3_cuda", nk_each_blend_e4m3_cuda, backend);
        run_each<nk_e4m3_k, fma_k, nk_f32_k>(env, "each_fma_e4m3_cuda", nk_each_fma_e4m3_cuda, backend);
        run_each<nk_i8_k, sum_k, nk_f32_k>(env, "each_sum_i8_cuda", nk_each_sum_i8_cuda, backend);
        run_each<nk_i8_k, scale_k, nk_f32_k>(env, "each_scale_i8_cuda", nk_each_scale_i8_cuda, backend);
        run_each<nk_i8_k, blend_k, nk_f32_k>(env, "each_blend_i8_cuda", nk_each_blend_i8_cuda, backend);
        run_each<nk_i8_k, fma_k, nk_f32_k>(env, "each_fma_i8_cuda", nk_each_fma_i8_cuda, backend);
        run_each<nk_u8_k, sum_k, nk_f32_k>(env, "each_sum_u8_cuda", nk_each_sum_u8_cuda, backend);
        run_each<nk_u8_k, scale_k, nk_f32_k>(env, "each_scale_u8_cuda", nk_each_scale_u8_cuda, backend);
        run_each<nk_u8_k, blend_k, nk_f32_k>(env, "each_blend_u8_cuda", nk_each_blend_u8_cuda, backend);
        run_each<nk_u8_k, fma_k, nk_f32_k>(env, "each_fma_u8_cuda", nk_each_fma_u8_cuda, backend);
        run_each<nk_i32_k, sum_k, nk_f64_k>(env, "each_sum_i32_cuda", nk_each_sum_i32_cuda, backend);
        run_each<nk_i32_k, scale_k, nk_f64_k>(env, "each_scale_i32_cuda", nk_each_scale_i32_cuda, backend);
        run_each<nk_i32_k, blend_k, nk_f64_k>(env, "each_blend_i32_cuda", nk_each_blend_i32_cuda, backend);
        run_each<nk_i32_k, fma_k, nk_f64_k>(env, "each_fma_i32_cuda", nk_each_fma_i32_cuda, backend);
        run_rmsnorm<nk_f32_k>(env, "each_rmsnorm_f32_cuda", nk_each_rmsnorm_f32_cuda, backend);
        run_rmsnorm<nk_bf16_k>(env, "each_rmsnorm_bf16_cuda", nk_each_rmsnorm_bf16_cuda, backend);
        run_rmsnorm<nk_e4m3_k>(env, "each_rmsnorm_e4m3_cuda", nk_each_rmsnorm_e4m3_cuda, backend);
        run_rmsnorm<nk_f32_k, nk_bf16_k>(env, "each_rmscast_bf16_cuda", nk_each_rmscast_bf16_cuda, backend);
        run_rmsnorm<nk_f32_k, nk_e4m3_k>(env, "each_rmscast_e4m3_cuda", nk_each_rmscast_e4m3_cuda, backend);
        run_rmsnorm<nk_f32_k, nk_e5m2_k>(env, "each_rmscast_e5m2_cuda", nk_each_rmscast_e5m2_cuda, backend);
        run_rmsnorm<nk_f32_k, nk_e2m3_k>(env, "each_rmscast_e2m3_cuda", nk_each_rmscast_e2m3_cuda, backend);
        run_rmsnorm<nk_f32_k, nk_e3m2_k>(env, "each_rmscast_e3m2_cuda", nk_each_rmscast_e3m2_cuda, backend);
        run_rmsnorm<nk_f64_k, nk_f32_k>(env, "each_rmscast_f32_cuda", nk_each_rmscast_f32_cuda, backend);
        run_rmsnorm<nk_i32_k, nk_i8_k>(env, "each_rmscast_i8_cuda", nk_each_rmscast_i8_cuda, backend);
        run_rmsnorm<nk_u32_k, nk_u8_k>(env, "each_rmscast_u8_cuda", nk_each_rmscast_u8_cuda, backend);
        run_swiglu<nk_f32_k>(env, "each_swiglu_f32_cuda", nk_each_swiglu_f32_cuda, backend);
        run_swiglu<nk_bf16_k>(env, "each_swiglu_bf16_cuda", nk_each_swiglu_bf16_cuda, backend);
        run_swiglu<nk_e4m3_k>(env, "each_swiglu_e4m3_cuda", nk_each_swiglu_e4m3_cuda, backend);
    }
#endif // NUMKONG_TARGET_CUDA
#if NUMKONG_TARGET_AMPERE
    if (enabled & nk_cap_ampere_k) {
        run_each<nk_bf16_k, sum_k, nk_f32_k>(env, "each_sum_bf16_ampere", nk_each_sum_bf16_ampere, backend);
        run_each<nk_bf16_k, scale_k, nk_f32_k>(env, "each_scale_bf16_ampere", nk_each_scale_bf16_ampere, backend);
        run_each<nk_bf16_k, blend_k, nk_f32_k>(env, "each_blend_bf16_ampere", nk_each_blend_bf16_ampere, backend);
        run_each<nk_bf16_k, fma_k, nk_f32_k>(env, "each_fma_bf16_ampere", nk_each_fma_bf16_ampere, backend);
        run_rmsnorm<nk_bf16_k>(env, "each_rmsnorm_bf16_ampere", nk_each_rmsnorm_bf16_ampere, backend);
        run_rmsnorm<nk_f32_k, nk_bf16_k>(env, "each_rmscast_bf16_ampere", nk_each_rmscast_bf16_ampere, backend);
        run_swiglu<nk_bf16_k>(env, "each_swiglu_bf16_ampere", nk_each_swiglu_bf16_ampere, backend);
    }
#endif // NUMKONG_TARGET_AMPERE
#if NUMKONG_TARGET_ADA
    if (enabled & nk_cap_ada_k) {
        run_each<nk_e4m3_k, sum_k, nk_f32_k>(env, "each_sum_e4m3_ada", nk_each_sum_e4m3_ada, backend);
        run_each<nk_e4m3_k, scale_k, nk_f32_k>(env, "each_scale_e4m3_ada", nk_each_scale_e4m3_ada, backend);
        run_each<nk_e4m3_k, blend_k, nk_f32_k>(env, "each_blend_e4m3_ada", nk_each_blend_e4m3_ada, backend);
        run_each<nk_e4m3_k, fma_k, nk_f32_k>(env, "each_fma_e4m3_ada", nk_each_fma_e4m3_ada, backend);
        run_rmsnorm<nk_e4m3_k>(env, "each_rmsnorm_e4m3_ada", nk_each_rmsnorm_e4m3_ada, backend);
        run_rmsnorm<nk_f32_k, nk_e4m3_k>(env, "each_rmscast_e4m3_ada", nk_each_rmscast_e4m3_ada, backend);
        run_rmsnorm<nk_f32_k, nk_e5m2_k>(env, "each_rmscast_e5m2_ada", nk_each_rmscast_e5m2_ada, backend);
        run_swiglu<nk_e4m3_k>(env, "each_swiglu_e4m3_ada", nk_each_swiglu_e4m3_ada, backend);
    }
#endif // NUMKONG_TARGET_ADA
}

#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::bench
