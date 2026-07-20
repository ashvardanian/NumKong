/**
 *  @brief Batch operation benchmarks - RVV ISA family (RISC-V Vector).
 *  @file bench/bench_cross_rvv.cpp
 *  @author Ash Vardanian
 *  @date February 15, 2026
 */

#include "numkong/attention.h"
#include "numkong/dot.h"
#include "numkong/dots.h"
#include "numkong/spatials.h"

#include "bench.hpp"

void bench_cross_rvv() {
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;

#if NK_TARGET_RVV
    run_dots_packed<f64_k>("dots_packed_f64_rvv", nk_dots_pack_size_f64_rvv, nk_dots_pack_f64_rvv,
                           nk_dots_packed_f64_rvv);
    run_dots_packed<f32_k>("dots_packed_f32_rvv", nk_dots_pack_size_f32_rvv, nk_dots_pack_f32_rvv,
                           nk_dots_packed_f32_rvv);
    run_dots_packed<f16_k>("dots_packed_f16_rvv", nk_dots_pack_size_f16_rvv, nk_dots_pack_f16_rvv,
                           nk_dots_packed_f16_rvv);
    run_dots_packed<bf16_k>("dots_packed_bf16_rvv", nk_dots_pack_size_bf16_rvv, nk_dots_pack_bf16_rvv,
                            nk_dots_packed_bf16_rvv);
    run_dots_packed<e4m3_k>("dots_packed_e4m3_rvv", nk_dots_pack_size_e4m3_rvv, nk_dots_pack_e4m3_rvv,
                            nk_dots_packed_e4m3_rvv);
    run_dots_packed<e5m2_k>("dots_packed_e5m2_rvv", nk_dots_pack_size_e5m2_rvv, nk_dots_pack_e5m2_rvv,
                            nk_dots_packed_e5m2_rvv);
    run_dots_packed<e2m3_k>("dots_packed_e2m3_rvv", nk_dots_pack_size_e2m3_rvv, nk_dots_pack_e2m3_rvv,
                            nk_dots_packed_e2m3_rvv);
    run_dots_packed<e3m2_k>("dots_packed_e3m2_rvv", nk_dots_pack_size_e3m2_rvv, nk_dots_pack_e3m2_rvv,
                            nk_dots_packed_e3m2_rvv);
    run_dots_packed<i8_k>("dots_packed_i8_rvv", nk_dots_pack_size_i8_rvv, nk_dots_pack_i8_rvv, nk_dots_packed_i8_rvv);
    run_dots_packed<u8_k>("dots_packed_u8_rvv", nk_dots_pack_size_u8_rvv, nk_dots_pack_u8_rvv, nk_dots_packed_u8_rvv);

    run_dots_symmetric<f64_k>("dots_symmetric_f64_rvv", nk_dots_symmetric_f64_rvv);
    run_dots_symmetric<f32_k>("dots_symmetric_f32_rvv", nk_dots_symmetric_f32_rvv);
    run_dots_symmetric<f16_k>("dots_symmetric_f16_rvv", nk_dots_symmetric_f16_rvv);
    run_dots_symmetric<bf16_k>("dots_symmetric_bf16_rvv", nk_dots_symmetric_bf16_rvv);
    run_dots_symmetric<e4m3_k>("dots_symmetric_e4m3_rvv", nk_dots_symmetric_e4m3_rvv);
    run_dots_symmetric<e5m2_k>("dots_symmetric_e5m2_rvv", nk_dots_symmetric_e5m2_rvv);
    run_dots_symmetric<e2m3_k>("dots_symmetric_e2m3_rvv", nk_dots_symmetric_e2m3_rvv);
    run_dots_symmetric<e3m2_k>("dots_symmetric_e3m2_rvv", nk_dots_symmetric_e3m2_rvv);
    run_dots_symmetric<i8_k>("dots_symmetric_i8_rvv", nk_dots_symmetric_i8_rvv);
    run_dots_symmetric<u8_k>("dots_symmetric_u8_rvv", nk_dots_symmetric_u8_rvv);

    run_angulars_packed<f64_k>("angulars_packed_f64_rvv", nk_dots_pack_size_f64_rvv, nk_dots_pack_f64_rvv,
                               nk_angulars_packed_f64_rvv);
    run_angulars_packed<f32_k>("angulars_packed_f32_rvv", nk_dots_pack_size_f32_rvv, nk_dots_pack_f32_rvv,
                               nk_angulars_packed_f32_rvv);
    run_angulars_packed<f16_k>("angulars_packed_f16_rvv", nk_dots_pack_size_f16_rvv, nk_dots_pack_f16_rvv,
                               nk_angulars_packed_f16_rvv);
    run_angulars_packed<bf16_k>("angulars_packed_bf16_rvv", nk_dots_pack_size_bf16_rvv, nk_dots_pack_bf16_rvv,
                                nk_angulars_packed_bf16_rvv);
    run_angulars_packed<e4m3_k>("angulars_packed_e4m3_rvv", nk_dots_pack_size_e4m3_rvv, nk_dots_pack_e4m3_rvv,
                                nk_angulars_packed_e4m3_rvv);
    run_angulars_packed<e5m2_k>("angulars_packed_e5m2_rvv", nk_dots_pack_size_e5m2_rvv, nk_dots_pack_e5m2_rvv,
                                nk_angulars_packed_e5m2_rvv);
    run_angulars_packed<e2m3_k>("angulars_packed_e2m3_rvv", nk_dots_pack_size_e2m3_rvv, nk_dots_pack_e2m3_rvv,
                                nk_angulars_packed_e2m3_rvv);
    run_angulars_packed<e3m2_k>("angulars_packed_e3m2_rvv", nk_dots_pack_size_e3m2_rvv, nk_dots_pack_e3m2_rvv,
                                nk_angulars_packed_e3m2_rvv);
    run_angulars_packed<i8_k>("angulars_packed_i8_rvv", nk_dots_pack_size_i8_rvv, nk_dots_pack_i8_rvv,
                              nk_angulars_packed_i8_rvv);
    run_angulars_packed<u8_k>("angulars_packed_u8_rvv", nk_dots_pack_size_u8_rvv, nk_dots_pack_u8_rvv,
                              nk_angulars_packed_u8_rvv);

    run_angulars_symmetric<f64_k>("angulars_symmetric_f64_rvv", nk_angulars_symmetric_f64_rvv);
    run_angulars_symmetric<f32_k>("angulars_symmetric_f32_rvv", nk_angulars_symmetric_f32_rvv);
    run_angulars_symmetric<f16_k>("angulars_symmetric_f16_rvv", nk_angulars_symmetric_f16_rvv);
    run_angulars_symmetric<bf16_k>("angulars_symmetric_bf16_rvv", nk_angulars_symmetric_bf16_rvv);
    run_angulars_symmetric<e4m3_k>("angulars_symmetric_e4m3_rvv", nk_angulars_symmetric_e4m3_rvv);
    run_angulars_symmetric<e5m2_k>("angulars_symmetric_e5m2_rvv", nk_angulars_symmetric_e5m2_rvv);
    run_angulars_symmetric<e2m3_k>("angulars_symmetric_e2m3_rvv", nk_angulars_symmetric_e2m3_rvv);
    run_angulars_symmetric<e3m2_k>("angulars_symmetric_e3m2_rvv", nk_angulars_symmetric_e3m2_rvv);
    run_angulars_symmetric<i8_k>("angulars_symmetric_i8_rvv", nk_angulars_symmetric_i8_rvv);
    run_angulars_symmetric<u8_k>("angulars_symmetric_u8_rvv", nk_angulars_symmetric_u8_rvv);

    run_euclideans_packed<f64_k>("euclideans_packed_f64_rvv", nk_dots_pack_size_f64_rvv, nk_dots_pack_f64_rvv,
                                 nk_euclideans_packed_f64_rvv);
    run_euclideans_packed<f32_k>("euclideans_packed_f32_rvv", nk_dots_pack_size_f32_rvv, nk_dots_pack_f32_rvv,
                                 nk_euclideans_packed_f32_rvv);
    run_euclideans_packed<f16_k>("euclideans_packed_f16_rvv", nk_dots_pack_size_f16_rvv, nk_dots_pack_f16_rvv,
                                 nk_euclideans_packed_f16_rvv);
    run_euclideans_packed<bf16_k>("euclideans_packed_bf16_rvv", nk_dots_pack_size_bf16_rvv, nk_dots_pack_bf16_rvv,
                                  nk_euclideans_packed_bf16_rvv);
    run_euclideans_packed<e4m3_k>("euclideans_packed_e4m3_rvv", nk_dots_pack_size_e4m3_rvv, nk_dots_pack_e4m3_rvv,
                                  nk_euclideans_packed_e4m3_rvv);
    run_euclideans_packed<e5m2_k>("euclideans_packed_e5m2_rvv", nk_dots_pack_size_e5m2_rvv, nk_dots_pack_e5m2_rvv,
                                  nk_euclideans_packed_e5m2_rvv);
    run_euclideans_packed<e2m3_k>("euclideans_packed_e2m3_rvv", nk_dots_pack_size_e2m3_rvv, nk_dots_pack_e2m3_rvv,
                                  nk_euclideans_packed_e2m3_rvv);
    run_euclideans_packed<e3m2_k>("euclideans_packed_e3m2_rvv", nk_dots_pack_size_e3m2_rvv, nk_dots_pack_e3m2_rvv,
                                  nk_euclideans_packed_e3m2_rvv);
    run_euclideans_packed<i8_k>("euclideans_packed_i8_rvv", nk_dots_pack_size_i8_rvv, nk_dots_pack_i8_rvv,
                                nk_euclideans_packed_i8_rvv);
    run_euclideans_packed<u8_k>("euclideans_packed_u8_rvv", nk_dots_pack_size_u8_rvv, nk_dots_pack_u8_rvv,
                                nk_euclideans_packed_u8_rvv);

    run_euclideans_symmetric<f64_k>("euclideans_symmetric_f64_rvv", nk_euclideans_symmetric_f64_rvv);
    run_euclideans_symmetric<f32_k>("euclideans_symmetric_f32_rvv", nk_euclideans_symmetric_f32_rvv);
    run_euclideans_symmetric<f16_k>("euclideans_symmetric_f16_rvv", nk_euclideans_symmetric_f16_rvv);
    run_euclideans_symmetric<bf16_k>("euclideans_symmetric_bf16_rvv", nk_euclideans_symmetric_bf16_rvv);
    run_euclideans_symmetric<e4m3_k>("euclideans_symmetric_e4m3_rvv", nk_euclideans_symmetric_e4m3_rvv);
    run_euclideans_symmetric<e5m2_k>("euclideans_symmetric_e5m2_rvv", nk_euclideans_symmetric_e5m2_rvv);
    run_euclideans_symmetric<e2m3_k>("euclideans_symmetric_e2m3_rvv", nk_euclideans_symmetric_e2m3_rvv);
    run_euclideans_symmetric<e3m2_k>("euclideans_symmetric_e3m2_rvv", nk_euclideans_symmetric_e3m2_rvv);
    run_euclideans_symmetric<i8_k>("euclideans_symmetric_i8_rvv", nk_euclideans_symmetric_i8_rvv);
    run_euclideans_symmetric<u8_k>("euclideans_symmetric_u8_rvv", nk_euclideans_symmetric_u8_rvv);

    run_attention<bf16_k>("attention_packed_bf16_rvv", nk_attention_pack_size_bf16_rvv, nk_attention_pack_bf16_rvv,
                          nk_attention_packed_bf16_rvv);
    run_attention<e4m3_k>("attention_packed_e4m3_rvv", nk_attention_pack_size_e4m3_rvv, nk_attention_pack_e4m3_rvv,
                          nk_attention_packed_e4m3_rvv);
    run_attention<i8_k>("attention_packed_i8_rvv", nk_attention_pack_size_i8_rvv, nk_attention_pack_i8_rvv,
                        nk_attention_packed_i8_rvv);
#endif
}
