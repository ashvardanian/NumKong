/**
 *  @brief Batch operation benchmarks - AMX ISA family (Sapphire Rapids AMX).
 *  @file bench/bench_cross_amx.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "numkong/dot.h"
#include "numkong/dots.h"
#include "numkong/spatials.h"

#include "bench.hpp"

void bench_cross_amx() {
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;

#if NK_TARGET_SAPPHIREAMX
    run_dots_packed<bf16_k>("dots_packed_bf16_sapphireamx", nk_dots_packed_size_bf16_sapphireamx,
                            nk_dots_pack_bf16_sapphireamx, nk_dots_packed_bf16_sapphireamx);
    run_dots_packed<e5m2_k>("dots_packed_e5m2_sapphireamx", nk_dots_packed_size_e5m2_sapphireamx,
                            nk_dots_pack_e5m2_sapphireamx, nk_dots_packed_e5m2_sapphireamx);
    run_dots_packed<e4m3_k>("dots_packed_e4m3_sapphireamx", nk_dots_packed_size_e4m3_sapphireamx,
                            nk_dots_pack_e4m3_sapphireamx, nk_dots_packed_e4m3_sapphireamx);
    run_dots_packed<e3m2_k>("dots_packed_e3m2_sapphireamx", nk_dots_packed_size_e3m2_sapphireamx,
                            nk_dots_pack_e3m2_sapphireamx, nk_dots_packed_e3m2_sapphireamx);
    run_dots_packed<e2m3_k>("dots_packed_e2m3_sapphireamx", nk_dots_packed_size_e2m3_sapphireamx,
                            nk_dots_pack_e2m3_sapphireamx, nk_dots_packed_e2m3_sapphireamx);
    run_dots_packed<i8_k>("dots_packed_i8_sapphireamx", nk_dots_packed_size_i8_sapphireamx, nk_dots_pack_i8_sapphireamx,
                          nk_dots_packed_i8_sapphireamx);
    run_dots_packed<u8_k>("dots_packed_u8_sapphireamx", nk_dots_packed_size_u8_sapphireamx, nk_dots_pack_u8_sapphireamx,
                          nk_dots_packed_u8_sapphireamx);

    run_dots_symmetric<bf16_k>("dots_symmetric_bf16_sapphireamx", nk_dots_symmetric_bf16_sapphireamx);
    run_dots_symmetric<e5m2_k>("dots_symmetric_e5m2_sapphireamx", nk_dots_symmetric_e5m2_sapphireamx);
    run_dots_symmetric<e4m3_k>("dots_symmetric_e4m3_sapphireamx", nk_dots_symmetric_e4m3_sapphireamx);
    run_dots_symmetric<e3m2_k>("dots_symmetric_e3m2_sapphireamx", nk_dots_symmetric_e3m2_sapphireamx);
    run_dots_symmetric<e2m3_k>("dots_symmetric_e2m3_sapphireamx", nk_dots_symmetric_e2m3_sapphireamx);
    run_dots_symmetric<i8_k>("dots_symmetric_i8_sapphireamx", nk_dots_symmetric_i8_sapphireamx);
    run_dots_symmetric<u8_k>("dots_symmetric_u8_sapphireamx", nk_dots_symmetric_u8_sapphireamx);
#endif

#if NK_TARGET_GRANITEAMX
    run_dots_packed<f16_k>("dots_packed_f16_graniteamx", nk_dots_packed_size_f16_graniteamx,
                           nk_dots_pack_f16_graniteamx, nk_dots_packed_f16_graniteamx);
    run_dots_symmetric<f16_k>("dots_symmetric_f16_graniteamx", nk_dots_symmetric_f16_graniteamx);

    run_angulars_packed<f16_k>("angulars_packed_f16_graniteamx", nk_dots_packed_size_f16_graniteamx,
                               nk_dots_pack_f16_graniteamx, nk_angulars_packed_f16_graniteamx);
    run_angulars_symmetric<f16_k>("angulars_symmetric_f16_graniteamx", nk_angulars_symmetric_f16_graniteamx);

    run_euclideans_packed<f16_k>("euclideans_packed_f16_graniteamx", nk_dots_packed_size_f16_graniteamx,
                                 nk_dots_pack_f16_graniteamx, nk_euclideans_packed_f16_graniteamx);
    run_euclideans_symmetric<f16_k>("euclideans_symmetric_f16_graniteamx", nk_euclideans_symmetric_f16_graniteamx);

    run_dots_packed<e5m2_k>("dots_packed_e5m2_graniteamx", nk_dots_packed_size_e5m2_graniteamx,
                            nk_dots_pack_e5m2_graniteamx, nk_dots_packed_e5m2_graniteamx);
    run_dots_symmetric<e5m2_k>("dots_symmetric_e5m2_graniteamx", nk_dots_symmetric_e5m2_graniteamx);

    run_angulars_packed<e5m2_k>("angulars_packed_e5m2_graniteamx", nk_dots_packed_size_e5m2_graniteamx,
                                nk_dots_pack_e5m2_graniteamx, nk_angulars_packed_e5m2_graniteamx);
    run_angulars_symmetric<e5m2_k>("angulars_symmetric_e5m2_graniteamx", nk_angulars_symmetric_e5m2_graniteamx);

    run_euclideans_packed<e5m2_k>("euclideans_packed_e5m2_graniteamx", nk_dots_packed_size_e5m2_graniteamx,
                                  nk_dots_pack_e5m2_graniteamx, nk_euclideans_packed_e5m2_graniteamx);
    run_euclideans_symmetric<e5m2_k>("euclideans_symmetric_e5m2_graniteamx", nk_euclideans_symmetric_e5m2_graniteamx);
#endif
}
