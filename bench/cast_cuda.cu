/**
 *  @file bench/cast_cuda.cu
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Bulk and block-scaled conversion benchmarks on CUDA, over 4096 tokens of a
 *      4096-wide hidden state.
 */

#include "numkong/cast.h" // `nk_cast_cuda`, `nk_cast_ampere`, `nk_cast_ada`

#include "cross.hpp"
#include "harness_cuda.hpp"

namespace ashvardanian::numkong::bench {

#if NUMKONG_ARCH_CUDA_

void bench_cast_cuda([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend,
                     [[maybe_unused]] nk_capability_t enabled) {
    [[maybe_unused]] std::size_t const tokens = 4096, hidden = 4096;
#if NUMKONG_TARGET_CUDA
    if (enabled & nk_cap_cuda_k) {
        run_cast_rows<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_cast_rows<nk_bf16_k, nk_f32_k>(env, "cast_bf16_to_f32_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_cast_rows<nk_f32_k, nk_f16_k>(env, "cast_f32_to_f16_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_cast_rows<nk_f32_k, nk_e4m3_k>(env, "cast_f32_to_e4m3_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_cast_rows<nk_bf16_k, nk_e4m3_k>(env, "cast_bf16_to_e4m3_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_cast_rows<nk_e4m3_k, nk_bf16_k>(env, "cast_e4m3_to_bf16_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_cast_rows<nk_e4m3_k, nk_f32_k>(env, "cast_e4m3_to_f32_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_cast_rows<nk_f32_k, nk_i8_k>(env, "cast_f32_to_i8_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_cast_rows<nk_f64_k, nk_f32_k>(env, "cast_f64_to_f32_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_block_scaled_rows<block_scaled_direction_t::encode_k, nk_nvfp4_k>(
            env, "cast_block_scaled_nvfp4_encode_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_block_scaled_rows<block_scaled_direction_t::decode_k, nk_nvfp4_k>(
            env, "cast_block_scaled_nvfp4_decode_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_block_scaled_rows<block_scaled_direction_t::encode_k, nk_mxfp4_k>(
            env, "cast_block_scaled_mxfp4_encode_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_block_scaled_rows<block_scaled_direction_t::encode_k, nk_mxfp8e4m3_k>(
            env, "cast_block_scaled_mxfp8e4m3_encode_cuda", nk_cast_cuda, tokens, hidden, backend);
        run_block_scaled_rows<block_scaled_direction_t::decode_k, nk_mxfp8e4m3_k>(
            env, "cast_block_scaled_mxfp8e4m3_decode_cuda", nk_cast_cuda, tokens, hidden, backend);
    }
#endif // NUMKONG_TARGET_CUDA
#if NUMKONG_TARGET_AMPERE
    if (enabled & nk_cap_ampere_k) {
        run_cast_rows<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_ampere", nk_cast_ampere, tokens, hidden, backend);
    }
#endif // NUMKONG_TARGET_AMPERE
#if NUMKONG_TARGET_ADA
    if (enabled & nk_cap_ada_k) {
        run_cast_rows<nk_f32_k, nk_bf16_k>(env, "cast_f32_to_bf16_ada", nk_cast_ada, tokens, hidden, backend);
        run_cast_rows<nk_f32_k, nk_e4m3_k>(env, "cast_f32_to_e4m3_ada", nk_cast_ada, tokens, hidden, backend);
        run_cast_rows<nk_e4m3_k, nk_f32_k>(env, "cast_e4m3_to_f32_ada", nk_cast_ada, tokens, hidden, backend);
    }
#endif // NUMKONG_TARGET_ADA
}

#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::bench
