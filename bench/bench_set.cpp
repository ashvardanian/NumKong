/**
 *  @brief Binary set metric benchmarks (hamming, jaccard).
 *  @file bench/bench_set.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/set.h"

#include "bench.hpp"

using namespace ashvardanian::numkong::bench;

void bench_set() {
    constexpr nk_dtype_t u1_k = nk_u1_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;

#if NK_TARGET_NEON
    run_dense<u1_k, u32_k>("hamming_u1_neon", nk_hamming_u1_neon);
    run_dense<u1_k, f32_k>("jaccard_u1_neon", nk_jaccard_u1_neon);
    run_dense<u8_k, u32_k>("hamming_u8_neon", nk_hamming_u8_neon);
    run_dense<u16_k, f32_k>("jaccard_u16_neon", nk_jaccard_u16_neon);
    run_dense<u32_k, f32_k>("jaccard_u32_neon", nk_jaccard_u32_neon);
#endif

#if NK_TARGET_SVE
    run_dense<u1_k, u32_k>("hamming_u1_sve", nk_hamming_u1_sve);
    run_dense<u1_k, f32_k>("jaccard_u1_sve", nk_jaccard_u1_sve);
    run_dense<u8_k, u32_k>("hamming_u8_sve", nk_hamming_u8_sve);
    run_dense<u16_k, f32_k>("jaccard_u16_sve", nk_jaccard_u16_sve);
    run_dense<u32_k, f32_k>("jaccard_u32_sve", nk_jaccard_u32_sve);
#endif

#if NK_TARGET_HASWELL
    run_dense<u1_k, u32_k>("hamming_u1_haswell", nk_hamming_u1_haswell);
    run_dense<u1_k, f32_k>("jaccard_u1_haswell", nk_jaccard_u1_haswell);
    run_dense<u8_k, u32_k>("hamming_u8_haswell", nk_hamming_u8_haswell);
    run_dense<u16_k, f32_k>("jaccard_u16_haswell", nk_jaccard_u16_haswell);
    run_dense<u32_k, f32_k>("jaccard_u32_haswell", nk_jaccard_u32_haswell);
#endif

#if NK_TARGET_ICELAKE
    run_dense<u1_k, u32_k>("hamming_u1_icelake", nk_hamming_u1_icelake);
    run_dense<u1_k, f32_k>("jaccard_u1_icelake", nk_jaccard_u1_icelake);
    run_dense<u8_k, u32_k>("hamming_u8_icelake", nk_hamming_u8_icelake);
    run_dense<u16_k, f32_k>("jaccard_u16_icelake", nk_jaccard_u16_icelake);
    run_dense<u32_k, f32_k>("jaccard_u32_icelake", nk_jaccard_u32_icelake);
#endif

#if NK_TARGET_RVV
    run_dense<u1_k, u32_k>("hamming_u1_rvv", nk_hamming_u1_rvv);
    run_dense<u1_k, f32_k>("jaccard_u1_rvv", nk_jaccard_u1_rvv);
    run_dense<u8_k, u32_k>("hamming_u8_rvv", nk_hamming_u8_rvv);
    run_dense<u16_k, f32_k>("jaccard_u16_rvv", nk_jaccard_u16_rvv);
    run_dense<u32_k, f32_k>("jaccard_u32_rvv", nk_jaccard_u32_rvv);
#endif

#if NK_TARGET_V128
    run_dense<u1_k, u32_k>("hamming_u1_v128", nk_hamming_u1_v128);
    run_dense<u1_k, f32_k>("jaccard_u1_v128", nk_jaccard_u1_v128);
    run_dense<u8_k, u32_k>("hamming_u8_v128", nk_hamming_u8_v128);
    run_dense<u16_k, f32_k>("jaccard_u16_v128", nk_jaccard_u16_v128);
    run_dense<u32_k, f32_k>("jaccard_u32_v128", nk_jaccard_u32_v128);
#endif

    // Serial fallbacks
    run_dense<u1_k, u32_k>("hamming_u1_serial", nk_hamming_u1_serial);
    run_dense<u1_k, f32_k>("jaccard_u1_serial", nk_jaccard_u1_serial);
    run_dense<u8_k, u32_k>("hamming_u8_serial", nk_hamming_u8_serial);
    run_dense<u16_k, f32_k>("jaccard_u16_serial", nk_jaccard_u16_serial);
    run_dense<u32_k, f32_k>("jaccard_u32_serial", nk_jaccard_u32_serial);
}
