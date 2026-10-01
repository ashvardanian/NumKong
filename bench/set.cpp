/**
 *  @file bench/set.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Binary set metric benchmarks, hamming and jaccard.
 */

#include "numkong/set.h"

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

void bench_set(environment_t const &env) {
    constexpr nk_dtype_t u1_k = nk_u1_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;

#if NUMKONG_TARGET_NEON
    if (section(env, "Binary Distances NEON", nk_cap_neon_k)) {
        run_dense<u1_k, u32_k>(env, "hamming_u1_neon", nk_hamming_u1_neon);
        run_dense<u1_k, f32_k>(env, "jaccard_u1_neon", nk_jaccard_u1_neon);
        run_dense<u8_k, u32_k>(env, "hamming_u8_neon", nk_hamming_u8_neon);
        run_dense<u16_k, f32_k>(env, "jaccard_u16_neon", nk_jaccard_u16_neon);
        run_dense<u32_k, f32_k>(env, "jaccard_u32_neon", nk_jaccard_u32_neon);
    }
#endif

#if NUMKONG_TARGET_SVE
    if (section(env, "Binary Distances SVE", nk_cap_sve_k)) {
        run_dense<u1_k, u32_k>(env, "hamming_u1_sve", nk_hamming_u1_sve);
        run_dense<u1_k, f32_k>(env, "jaccard_u1_sve", nk_jaccard_u1_sve);
        run_dense<u8_k, u32_k>(env, "hamming_u8_sve", nk_hamming_u8_sve);
        run_dense<u16_k, f32_k>(env, "jaccard_u16_sve", nk_jaccard_u16_sve);
        run_dense<u32_k, f32_k>(env, "jaccard_u32_sve", nk_jaccard_u32_sve);
    }
#endif

#if NUMKONG_TARGET_HASWELL
    if (section(env, "Binary Distances Haswell", nk_cap_haswell_k)) {
        run_dense<u1_k, u32_k>(env, "hamming_u1_haswell", nk_hamming_u1_haswell);
        run_dense<u1_k, f32_k>(env, "jaccard_u1_haswell", nk_jaccard_u1_haswell);
        run_dense<u8_k, u32_k>(env, "hamming_u8_haswell", nk_hamming_u8_haswell);
        run_dense<u16_k, f32_k>(env, "jaccard_u16_haswell", nk_jaccard_u16_haswell);
        run_dense<u32_k, f32_k>(env, "jaccard_u32_haswell", nk_jaccard_u32_haswell);
    }
#endif

#if NUMKONG_TARGET_ICELAKE
    if (section(env, "Binary Distances Ice Lake", nk_cap_icelake_k)) {
        run_dense<u1_k, u32_k>(env, "hamming_u1_icelake", nk_hamming_u1_icelake);
        run_dense<u1_k, f32_k>(env, "jaccard_u1_icelake", nk_jaccard_u1_icelake);
        run_dense<u8_k, u32_k>(env, "hamming_u8_icelake", nk_hamming_u8_icelake);
        run_dense<u16_k, f32_k>(env, "jaccard_u16_icelake", nk_jaccard_u16_icelake);
        run_dense<u32_k, f32_k>(env, "jaccard_u32_icelake", nk_jaccard_u32_icelake);
    }
#endif

#if NUMKONG_TARGET_RVV
    if (section(env, "Binary Distances RVV", nk_cap_rvv_k)) {
        run_dense<u1_k, u32_k>(env, "hamming_u1_rvv", nk_hamming_u1_rvv);
        run_dense<u1_k, f32_k>(env, "jaccard_u1_rvv", nk_jaccard_u1_rvv);
        run_dense<u8_k, u32_k>(env, "hamming_u8_rvv", nk_hamming_u8_rvv);
        run_dense<u16_k, f32_k>(env, "jaccard_u16_rvv", nk_jaccard_u16_rvv);
        run_dense<u32_k, f32_k>(env, "jaccard_u32_rvv", nk_jaccard_u32_rvv);
    }
#endif

#if NUMKONG_TARGET_V128
    if (section(env, "Binary Distances V128", nk_cap_v128_k)) {
        run_dense<u1_k, u32_k>(env, "hamming_u1_v128", nk_hamming_u1_v128);
        run_dense<u1_k, f32_k>(env, "jaccard_u1_v128", nk_jaccard_u1_v128);
        run_dense<u8_k, u32_k>(env, "hamming_u8_v128", nk_hamming_u8_v128);
        run_dense<u16_k, f32_k>(env, "jaccard_u16_v128", nk_jaccard_u16_v128);
        run_dense<u32_k, f32_k>(env, "jaccard_u32_v128", nk_jaccard_u32_v128);
    }
#endif

    // Serial fallbacks
    section(env, "Binary Distances Serial", nk_cap_serial_k);
    run_dense<u1_k, u32_k>(env, "hamming_u1_serial", nk_hamming_u1_serial);
    run_dense<u1_k, f32_k>(env, "jaccard_u1_serial", nk_jaccard_u1_serial);
    run_dense<u8_k, u32_k>(env, "hamming_u8_serial", nk_hamming_u8_serial);
    run_dense<u16_k, f32_k>(env, "jaccard_u16_serial", nk_jaccard_u16_serial);
    run_dense<u32_k, f32_k>(env, "jaccard_u32_serial", nk_jaccard_u32_serial);
}

} // namespace ashvardanian::numkong::bench
