/**
 *  @brief Batch operation benchmarks - LoongArch ISA family (LASX 256-bit).
 *  @file bench/bench_cross_loongarch.cpp
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  Covers LoongSON Advanced SIMD Extension (LASX).
 */

#include "numkong/dot.h"
#include "numkong/dots.h"
#include "numkong/sets.h"
#include "numkong/spatials.h"

#include "bench.hpp"

void bench_cross_loongarch() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t u1_k = nk_u1_k;

#if NK_TARGET_LOONGSONASX

    // Dots: packed
    run_dots_packed<f64_k>("dots_packed_f64_loongsonasx", nk_dots_pack_size_f64_loongsonasx,
                           nk_dots_pack_f64_loongsonasx, nk_dots_packed_f64_loongsonasx);
    run_dots_packed<f32_k>("dots_packed_f32_loongsonasx", nk_dots_pack_size_f32_loongsonasx,
                           nk_dots_pack_f32_loongsonasx, nk_dots_packed_f32_loongsonasx);
    run_dots_packed<bf16_k>("dots_packed_bf16_loongsonasx", nk_dots_pack_size_bf16_loongsonasx,
                            nk_dots_pack_bf16_loongsonasx, nk_dots_packed_bf16_loongsonasx);
    run_dots_packed<f16_k>("dots_packed_f16_loongsonasx", nk_dots_pack_size_f16_loongsonasx,
                           nk_dots_pack_f16_loongsonasx, nk_dots_packed_f16_loongsonasx);
    run_dots_packed<i8_k>("dots_packed_i8_loongsonasx", nk_dots_pack_size_i8_loongsonasx, nk_dots_pack_i8_loongsonasx,
                          nk_dots_packed_i8_loongsonasx);
    run_dots_packed<u8_k>("dots_packed_u8_loongsonasx", nk_dots_pack_size_u8_loongsonasx, nk_dots_pack_u8_loongsonasx,
                          nk_dots_packed_u8_loongsonasx);
    run_dots_packed<u1_k>("dots_packed_u1_loongsonasx", nk_dots_pack_size_u1_loongsonasx, nk_dots_pack_u1_loongsonasx,
                          nk_dots_packed_u1_loongsonasx);

    // Dots: symmetric
    run_dots_symmetric<f64_k>("dots_symmetric_f64_loongsonasx", nk_dots_symmetric_f64_loongsonasx);
    run_dots_symmetric<f32_k>("dots_symmetric_f32_loongsonasx", nk_dots_symmetric_f32_loongsonasx);
    run_dots_symmetric<bf16_k>("dots_symmetric_bf16_loongsonasx", nk_dots_symmetric_bf16_loongsonasx);
    run_dots_symmetric<f16_k>("dots_symmetric_f16_loongsonasx", nk_dots_symmetric_f16_loongsonasx);
    run_dots_symmetric<i8_k>("dots_symmetric_i8_loongsonasx", nk_dots_symmetric_i8_loongsonasx);
    run_dots_symmetric<u8_k>("dots_symmetric_u8_loongsonasx", nk_dots_symmetric_u8_loongsonasx);
    run_dots_symmetric<u1_k>("dots_symmetric_u1_loongsonasx", nk_dots_symmetric_u1_loongsonasx);

    // Angulars: packed
    run_angulars_packed<f64_k>("angulars_packed_f64_loongsonasx", nk_dots_pack_size_f64_loongsonasx,
                               nk_dots_pack_f64_loongsonasx, nk_angulars_packed_f64_loongsonasx);
    run_angulars_packed<f32_k>("angulars_packed_f32_loongsonasx", nk_dots_pack_size_f32_loongsonasx,
                               nk_dots_pack_f32_loongsonasx, nk_angulars_packed_f32_loongsonasx);
    run_angulars_packed<bf16_k>("angulars_packed_bf16_loongsonasx", nk_dots_pack_size_bf16_loongsonasx,
                                nk_dots_pack_bf16_loongsonasx, nk_angulars_packed_bf16_loongsonasx);
    run_angulars_packed<f16_k>("angulars_packed_f16_loongsonasx", nk_dots_pack_size_f16_loongsonasx,
                               nk_dots_pack_f16_loongsonasx, nk_angulars_packed_f16_loongsonasx);
    run_angulars_packed<i8_k>("angulars_packed_i8_loongsonasx", nk_dots_pack_size_i8_loongsonasx,
                              nk_dots_pack_i8_loongsonasx, nk_angulars_packed_i8_loongsonasx);
    run_angulars_packed<u8_k>("angulars_packed_u8_loongsonasx", nk_dots_pack_size_u8_loongsonasx,
                              nk_dots_pack_u8_loongsonasx, nk_angulars_packed_u8_loongsonasx);

    // Angulars: symmetric
    run_angulars_symmetric<f64_k>("angulars_symmetric_f64_loongsonasx", nk_angulars_symmetric_f64_loongsonasx);
    run_angulars_symmetric<f32_k>("angulars_symmetric_f32_loongsonasx", nk_angulars_symmetric_f32_loongsonasx);
    run_angulars_symmetric<bf16_k>("angulars_symmetric_bf16_loongsonasx", nk_angulars_symmetric_bf16_loongsonasx);
    run_angulars_symmetric<f16_k>("angulars_symmetric_f16_loongsonasx", nk_angulars_symmetric_f16_loongsonasx);
    run_angulars_symmetric<i8_k>("angulars_symmetric_i8_loongsonasx", nk_angulars_symmetric_i8_loongsonasx);
    run_angulars_symmetric<u8_k>("angulars_symmetric_u8_loongsonasx", nk_angulars_symmetric_u8_loongsonasx);

    // Euclideans: packed
    run_euclideans_packed<f64_k>("euclideans_packed_f64_loongsonasx", nk_dots_pack_size_f64_loongsonasx,
                                 nk_dots_pack_f64_loongsonasx, nk_euclideans_packed_f64_loongsonasx);
    run_euclideans_packed<f32_k>("euclideans_packed_f32_loongsonasx", nk_dots_pack_size_f32_loongsonasx,
                                 nk_dots_pack_f32_loongsonasx, nk_euclideans_packed_f32_loongsonasx);
    run_euclideans_packed<bf16_k>("euclideans_packed_bf16_loongsonasx", nk_dots_pack_size_bf16_loongsonasx,
                                  nk_dots_pack_bf16_loongsonasx, nk_euclideans_packed_bf16_loongsonasx);
    run_euclideans_packed<f16_k>("euclideans_packed_f16_loongsonasx", nk_dots_pack_size_f16_loongsonasx,
                                 nk_dots_pack_f16_loongsonasx, nk_euclideans_packed_f16_loongsonasx);
    run_euclideans_packed<i8_k>("euclideans_packed_i8_loongsonasx", nk_dots_pack_size_i8_loongsonasx,
                                nk_dots_pack_i8_loongsonasx, nk_euclideans_packed_i8_loongsonasx);
    run_euclideans_packed<u8_k>("euclideans_packed_u8_loongsonasx", nk_dots_pack_size_u8_loongsonasx,
                                nk_dots_pack_u8_loongsonasx, nk_euclideans_packed_u8_loongsonasx);

    // Euclideans: symmetric
    run_euclideans_symmetric<f64_k>("euclideans_symmetric_f64_loongsonasx", nk_euclideans_symmetric_f64_loongsonasx);
    run_euclideans_symmetric<f32_k>("euclideans_symmetric_f32_loongsonasx", nk_euclideans_symmetric_f32_loongsonasx);
    run_euclideans_symmetric<bf16_k>("euclideans_symmetric_bf16_loongsonasx", nk_euclideans_symmetric_bf16_loongsonasx);
    run_euclideans_symmetric<f16_k>("euclideans_symmetric_f16_loongsonasx", nk_euclideans_symmetric_f16_loongsonasx);
    run_euclideans_symmetric<i8_k>("euclideans_symmetric_i8_loongsonasx", nk_euclideans_symmetric_i8_loongsonasx);
    run_euclideans_symmetric<u8_k>("euclideans_symmetric_u8_loongsonasx", nk_euclideans_symmetric_u8_loongsonasx);

    // Sets
    run_hammings_packed<u1_k>("hammings_packed_u1_loongsonasx", nk_dots_pack_size_u1_serial, nk_dots_pack_u1_serial,
                              nk_hammings_packed_u1_loongsonasx);
    run_hammings_symmetric<u1_k>("hammings_symmetric_u1_loongsonasx", nk_hammings_symmetric_u1_loongsonasx);
    run_jaccards_packed<u1_k>("jaccards_packed_u1_loongsonasx", nk_dots_pack_size_u1_serial, nk_dots_pack_u1_serial,
                              nk_jaccards_packed_u1_loongsonasx);
    run_jaccards_symmetric<u1_k>("jaccards_symmetric_u1_loongsonasx", nk_jaccards_symmetric_u1_loongsonasx);

#endif // NK_TARGET_LOONGSONASX
}
