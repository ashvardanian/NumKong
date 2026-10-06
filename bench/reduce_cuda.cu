/**
 *  @file bench/reduce_cuda.cu
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Moments and min/max benchmarks on CUDA, over 4096 tokens of a 4096-wide hidden state.
 */

#include "numkong/reduce.h" // `nk_reduce_moments_*_cuda`, `nk_reduce_minmax_*_cuda`

#include "cross.hpp"
#include "harness_cuda.hpp"

namespace ashvardanian::numkong::bench {

#if NUMKONG_ARCH_CUDA_

void bench_reduce_cuda([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend,
                       [[maybe_unused]] nk_capability_t enabled) {
    [[maybe_unused]] std::size_t const tokens = 4096, hidden = 4096;
#if NUMKONG_TARGET_CUDA
    if (enabled & nk_cap_cuda_k) {
        run_moments_rows<nk_f64_k>(env, "reduce_moments_f64_cuda", nk_reduce_moments_f64_cuda, tokens, hidden, backend);
        run_moments_rows<nk_f32_k>(env, "reduce_moments_f32_cuda", nk_reduce_moments_f32_cuda, tokens, hidden, backend);
        run_moments_rows<nk_bf16_k>(env, "reduce_moments_bf16_cuda", nk_reduce_moments_bf16_cuda, tokens, hidden,
                                    backend);
        run_moments_rows<nk_f16_k>(env, "reduce_moments_f16_cuda", nk_reduce_moments_f16_cuda, tokens, hidden, backend);
        run_moments_rows<nk_e4m3_k>(env, "reduce_moments_e4m3_cuda", nk_reduce_moments_e4m3_cuda, tokens, hidden,
                                    backend);
        run_moments_rows<nk_i8_k>(env, "reduce_moments_i8_cuda", nk_reduce_moments_i8_cuda, tokens, hidden, backend);
        run_minmax_rows<nk_f32_k>(env, "reduce_minmax_f32_cuda", nk_reduce_minmax_f32_cuda, tokens, hidden, backend);
        run_minmax_rows<nk_bf16_k>(env, "reduce_minmax_bf16_cuda", nk_reduce_minmax_bf16_cuda, tokens, hidden, backend);
        run_minmax_rows<nk_i8_k>(env, "reduce_minmax_i8_cuda", nk_reduce_minmax_i8_cuda, tokens, hidden, backend);
    }
#endif // NUMKONG_TARGET_CUDA
}

#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::bench
