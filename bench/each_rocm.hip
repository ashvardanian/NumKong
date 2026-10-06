/**
 *  @file bench/each_rocm.hip
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Elementwise operations benchmarks on ROCm, over the backend's token rows.
 */

#include "numkong/each.h" // `nk_each_*_rocm`

#include "cross.hpp"
#include "harness_rocm.hpp"

namespace ashvardanian::numkong::bench {

#if NUMKONG_ARCH_ROCM_

void bench_each_rocm([[maybe_unused]] environment_t const &env, [[maybe_unused]] rocm_backend_t const &backend,
                     [[maybe_unused]] nk_capability_t enabled) {
    [[maybe_unused]] constexpr nk_kernel_kind_t sum_k = nk_kernel_each_sum_k;
#if NUMKONG_TARGET_ROCM
    if (enabled & nk_cap_rocm_k) {
        run_each<nk_f32_k, sum_k, nk_f32_k>(env, "each_sum_f32_rocm", nk_each_sum_f32_rocm, backend);
        run_each<nk_bf16_k, sum_k, nk_f32_k>(env, "each_sum_bf16_rocm", nk_each_sum_bf16_rocm, backend);
        run_rmsnorm<nk_f32_k>(env, "each_rmsnorm_f32_rocm", nk_each_rmsnorm_f32_rocm, backend);
        run_rmsnorm<nk_bf16_k>(env, "each_rmsnorm_bf16_rocm", nk_each_rmsnorm_bf16_rocm, backend);
        run_rmsnorm<nk_e4m3_k>(env, "each_rmsnorm_e4m3_rocm", nk_each_rmsnorm_e4m3_rocm, backend);
        run_rmsnorm<nk_f32_k, nk_bf16_k>(env, "each_rmscast_bf16_rocm", nk_each_rmscast_bf16_rocm, backend);
        run_rmsnorm<nk_f32_k, nk_e4m3_k>(env, "each_rmscast_e4m3_rocm", nk_each_rmscast_e4m3_rocm, backend);
        run_rmsnorm<nk_f64_k, nk_f32_k>(env, "each_rmscast_f32_rocm", nk_each_rmscast_f32_rocm, backend);
        run_rmsnorm<nk_i32_k, nk_i8_k>(env, "each_rmscast_i8_rocm", nk_each_rmscast_i8_rocm, backend);
        run_swiglu<nk_f32_k>(env, "each_swiglu_f32_rocm", nk_each_swiglu_f32_rocm, backend);
        run_swiglu<nk_bf16_k>(env, "each_swiglu_bf16_rocm", nk_each_swiglu_bf16_rocm, backend);
        run_swiglu<nk_e4m3_k>(env, "each_swiglu_e4m3_rocm", nk_each_swiglu_e4m3_rocm, backend);
    }
#endif // NUMKONG_TARGET_ROCM
}

#endif // NUMKONG_ARCH_ROCM_

} // namespace ashvardanian::numkong::bench
