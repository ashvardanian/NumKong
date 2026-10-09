/**
 *  @file bench/cross_rocm.hip
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief ROCm batch benchmarks using the shared cross-operation drivers.
 */
#include <algorithm> // `std::max`

#include "numkong/numkong.h"
#include "cross.hpp"
#include "harness_rocm.hpp"

#if NUMKONG_ARCH_ROCM_

namespace ashvardanian::numkong::bench {

void bench_cross_rocm(environment_t const &env, rocm_backend_t const &backend, nk_capability_t enabled) {
    if (!(enabled & nk_cap_rocm_k)) return;
    run_dots_packed<nk_f64_k>(env, "dots_packed_f64_rocm", nk_dots_pack_size_f64_rocm, nk_dots_pack_f64_rocm,
                              nk_dots_packed_f64_rocm, backend);
    run_dots_packed<nk_f32_k>(env, "dots_packed_f32_rocm", nk_dots_pack_size_f32_rocm, nk_dots_pack_f32_rocm,
                              nk_dots_packed_f32_rocm, backend);
    run_dots_symmetric<nk_f64_k>(env, "dots_symmetric_f64_rocm", nk_dots_symmetric_f64_rocm, backend);
    run_dots_symmetric<nk_f32_k>(env, "dots_symmetric_f32_rocm", nk_dots_symmetric_f32_rocm, backend);
    run_angulars_packed<nk_f64_k>(env, "angulars_packed_f64_rocm", nk_dots_pack_size_f64_rocm, nk_dots_pack_f64_rocm,
                                  nk_angulars_packed_f64_rocm, backend);
    run_angulars_packed<nk_f32_k>(env, "angulars_packed_f32_rocm", nk_dots_pack_size_f32_rocm, nk_dots_pack_f32_rocm,
                                  nk_angulars_packed_f32_rocm, backend);
    run_angulars_symmetric<nk_f64_k>(env, "angulars_symmetric_f64_rocm", nk_angulars_symmetric_f64_rocm, backend);
    run_angulars_symmetric<nk_f32_k>(env, "angulars_symmetric_f32_rocm", nk_angulars_symmetric_f32_rocm, backend);
    run_euclideans_packed<nk_f64_k>(env, "euclideans_packed_f64_rocm", nk_dots_pack_size_f64_rocm,
                                    nk_dots_pack_f64_rocm, nk_euclideans_packed_f64_rocm, backend);
    run_euclideans_packed<nk_f32_k>(env, "euclideans_packed_f32_rocm", nk_dots_pack_size_f32_rocm,
                                    nk_dots_pack_f32_rocm, nk_euclideans_packed_f32_rocm, backend);
    run_euclideans_symmetric<nk_f64_k>(env, "euclideans_symmetric_f64_rocm", nk_euclideans_symmetric_f64_rocm, backend);
    run_euclideans_symmetric<nk_f32_k>(env, "euclideans_symmetric_f32_rocm", nk_euclideans_symmetric_f32_rocm, backend);
    std::size_t const tokens = 4096, hidden = 4096;
    run_moments_rows<nk_f64_k>(env, "reduce_moments_f64_rocm", nk_reduce_moments_f64_rocm, tokens, hidden, backend);
    run_moments_rows<nk_f32_k>(env, "reduce_moments_f32_rocm", nk_reduce_moments_f32_rocm, tokens, hidden, backend);
    run_moments_rows<nk_bf16_k>(env, "reduce_moments_bf16_rocm", nk_reduce_moments_bf16_rocm, tokens, hidden, backend);
    run_moments_rows<nk_f16_k>(env, "reduce_moments_f16_rocm", nk_reduce_moments_f16_rocm, tokens, hidden, backend);
    run_moments_rows<nk_e4m3_k>(env, "reduce_moments_e4m3_rocm", nk_reduce_moments_e4m3_rocm, tokens, hidden, backend);
    run_moments_rows<nk_i8_k>(env, "reduce_moments_i8_rocm", nk_reduce_moments_i8_rocm, tokens, hidden, backend);
    run_minmax_rows<nk_f32_k>(env, "reduce_minmax_f32_rocm", nk_reduce_minmax_f32_rocm, tokens, hidden, backend);
    run_minmax_rows<nk_bf16_k>(env, "reduce_minmax_bf16_rocm", nk_reduce_minmax_bf16_rocm, tokens, hidden, backend);
    run_minmax_rows<nk_i8_k>(env, "reduce_minmax_i8_rocm", nk_reduce_minmax_i8_rocm, tokens, hidden, backend);
}

void bench_cross_cdna3([[maybe_unused]] environment_t const &env, [[maybe_unused]] rocm_backend_t const &backend,
                       [[maybe_unused]] nk_capability_t enabled) {
#if NUMKONG_TARGET_CDNA3
    if (!(enabled & nk_cap_cdna3_k)) return;
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_cdna3", nk_dots_pack_size_bf16_cdna3, nk_dots_pack_bf16_cdna3,
                               nk_dots_packed_bf16_cdna3, backend);
    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_cdna3", nk_dots_symmetric_bf16_cdna3, backend);
    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_cdna3", nk_dots_pack_size_bf16_cdna3,
                                   nk_dots_pack_bf16_cdna3, nk_angulars_packed_bf16_cdna3, backend);
    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_cdna3", nk_angulars_symmetric_bf16_cdna3, backend);
    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_cdna3", nk_dots_pack_size_bf16_cdna3,
                                     nk_dots_pack_bf16_cdna3, nk_euclideans_packed_bf16_cdna3, backend);
    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_cdna3", nk_euclideans_symmetric_bf16_cdna3,
                                        backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_cdna3", nk_dots_pack_size_f16_cdna3, nk_dots_pack_f16_cdna3,
                              nk_dots_packed_f16_cdna3, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_cdna3", nk_dots_symmetric_f16_cdna3, backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_cdna3", nk_dots_pack_size_f16_cdna3, nk_dots_pack_f16_cdna3,
                                  nk_angulars_packed_f16_cdna3, backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_cdna3", nk_angulars_symmetric_f16_cdna3, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_cdna3", nk_dots_pack_size_f16_cdna3,
                                    nk_dots_pack_f16_cdna3, nk_euclideans_packed_f16_cdna3, backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_cdna3", nk_euclideans_symmetric_f16_cdna3,
                                       backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_cdna3", nk_dots_pack_size_e4m3_cdna3, nk_dots_pack_e4m3_cdna3,
                               nk_dots_packed_e4m3_cdna3, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_cdna3", nk_dots_symmetric_e4m3_cdna3, backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_cdna3", nk_dots_pack_size_e4m3_cdna3,
                                   nk_dots_pack_e4m3_cdna3, nk_angulars_packed_e4m3_cdna3, backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_cdna3", nk_angulars_symmetric_e4m3_cdna3, backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_cdna3", nk_dots_pack_size_e4m3_cdna3,
                                     nk_dots_pack_e4m3_cdna3, nk_euclideans_packed_e4m3_cdna3, backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_cdna3", nk_euclideans_symmetric_e4m3_cdna3,
                                        backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_cdna3", nk_dots_pack_size_e5m2_cdna3, nk_dots_pack_e5m2_cdna3,
                               nk_dots_packed_e5m2_cdna3, backend);
    run_dots_symmetric<nk_e5m2_k>(env, "dots_symmetric_e5m2_cdna3", nk_dots_symmetric_e5m2_cdna3, backend);
    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_cdna3", nk_dots_pack_size_e5m2_cdna3,
                                   nk_dots_pack_e5m2_cdna3, nk_angulars_packed_e5m2_cdna3, backend);
    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_cdna3", nk_angulars_symmetric_e5m2_cdna3, backend);
    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_cdna3", nk_dots_pack_size_e5m2_cdna3,
                                     nk_dots_pack_e5m2_cdna3, nk_euclideans_packed_e5m2_cdna3, backend);
    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_cdna3", nk_euclideans_symmetric_e5m2_cdna3,
                                        backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_cdna3", nk_dots_pack_size_e3m2_cdna3, nk_dots_pack_e3m2_cdna3,
                               nk_dots_packed_e3m2_cdna3, backend);
    run_dots_symmetric<nk_e3m2_k>(env, "dots_symmetric_e3m2_cdna3", nk_dots_symmetric_e3m2_cdna3, backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_cdna3", nk_dots_pack_size_e3m2_cdna3,
                                   nk_dots_pack_e3m2_cdna3, nk_angulars_packed_e3m2_cdna3, backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_cdna3", nk_angulars_symmetric_e3m2_cdna3, backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_cdna3", nk_dots_pack_size_e3m2_cdna3,
                                     nk_dots_pack_e3m2_cdna3, nk_euclideans_packed_e3m2_cdna3, backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_cdna3", nk_euclideans_symmetric_e3m2_cdna3,
                                        backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_cdna3", nk_dots_pack_size_e2m3_cdna3, nk_dots_pack_e2m3_cdna3,
                               nk_dots_packed_e2m3_cdna3, backend);
    run_dots_symmetric<nk_e2m3_k>(env, "dots_symmetric_e2m3_cdna3", nk_dots_symmetric_e2m3_cdna3, backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_cdna3", nk_dots_pack_size_e2m3_cdna3,
                                   nk_dots_pack_e2m3_cdna3, nk_angulars_packed_e2m3_cdna3, backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_cdna3", nk_angulars_symmetric_e2m3_cdna3, backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_cdna3", nk_dots_pack_size_e2m3_cdna3,
                                     nk_dots_pack_e2m3_cdna3, nk_euclideans_packed_e2m3_cdna3, backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_cdna3", nk_euclideans_symmetric_e2m3_cdna3,
                                        backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_cdna3", nk_dots_pack_size_e2m1_cdna3, nk_dots_pack_e2m1_cdna3,
                               nk_dots_packed_e2m1_cdna3, backend);
    run_dots_symmetric<nk_e2m1_k>(env, "dots_symmetric_e2m1_cdna3", nk_dots_symmetric_e2m1_cdna3, backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_cdna3", nk_dots_pack_size_e2m1_cdna3,
                                   nk_dots_pack_e2m1_cdna3, nk_angulars_packed_e2m1_cdna3, backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_cdna3", nk_angulars_symmetric_e2m1_cdna3, backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_cdna3", nk_dots_pack_size_e2m1_cdna3,
                                     nk_dots_pack_e2m1_cdna3, nk_euclideans_packed_e2m1_cdna3, backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_cdna3", nk_euclideans_symmetric_e2m1_cdna3,
                                        backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_cdna3", nk_dots_pack_size_i8_cdna3, nk_dots_pack_i8_cdna3,
                             nk_dots_packed_i8_cdna3, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_cdna3", nk_dots_symmetric_i8_cdna3, backend);
    run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_cdna3", nk_dots_pack_size_i8_cdna3, nk_dots_pack_i8_cdna3,
                                 nk_angulars_packed_i8_cdna3, backend);
    run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_cdna3", nk_angulars_symmetric_i8_cdna3, backend);
    run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_cdna3", nk_dots_pack_size_i8_cdna3, nk_dots_pack_i8_cdna3,
                                   nk_euclideans_packed_i8_cdna3, backend);
    run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_cdna3", nk_euclideans_symmetric_i8_cdna3, backend);
    run_dots_packed<nk_i4_k>(env, "dots_packed_i4_cdna3", nk_dots_pack_size_i4_cdna3, nk_dots_pack_i4_cdna3,
                             nk_dots_packed_i4_cdna3, backend);
    run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_cdna3", nk_dots_symmetric_i4_cdna3, backend);
    run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_cdna3", nk_dots_pack_size_i4_cdna3, nk_dots_pack_i4_cdna3,
                                 nk_angulars_packed_i4_cdna3, backend);
    run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_cdna3", nk_angulars_symmetric_i4_cdna3, backend);
    run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_cdna3", nk_dots_pack_size_i4_cdna3, nk_dots_pack_i4_cdna3,
                                   nk_euclideans_packed_i4_cdna3, backend);
    run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_cdna3", nk_euclideans_symmetric_i4_cdna3, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_cdna3", nk_dots_pack_size_u8_cdna3, nk_dots_pack_u8_cdna3,
                             nk_dots_packed_u8_cdna3, backend);
    run_dots_symmetric<nk_u8_k>(env, "dots_symmetric_u8_cdna3", nk_dots_symmetric_u8_cdna3, backend);
    run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_cdna3", nk_dots_pack_size_u8_cdna3, nk_dots_pack_u8_cdna3,
                                 nk_angulars_packed_u8_cdna3, backend);
    run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_cdna3", nk_angulars_symmetric_u8_cdna3, backend);
    run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_cdna3", nk_dots_pack_size_u8_cdna3, nk_dots_pack_u8_cdna3,
                                   nk_euclideans_packed_u8_cdna3, backend);
    run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_cdna3", nk_euclideans_symmetric_u8_cdna3, backend);
    run_dots_packed<nk_u4_k>(env, "dots_packed_u4_cdna3", nk_dots_pack_size_u4_cdna3, nk_dots_pack_u4_cdna3,
                             nk_dots_packed_u4_cdna3, backend);
    run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_cdna3", nk_dots_symmetric_u4_cdna3, backend);
    run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_cdna3", nk_dots_pack_size_u4_cdna3, nk_dots_pack_u4_cdna3,
                                 nk_angulars_packed_u4_cdna3, backend);
    run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_cdna3", nk_angulars_symmetric_u4_cdna3, backend);
    run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_cdna3", nk_dots_pack_size_u4_cdna3, nk_dots_pack_u4_cdna3,
                                   nk_euclideans_packed_u4_cdna3, backend);
    run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_cdna3", nk_euclideans_symmetric_u4_cdna3, backend);
    run_attention<nk_bf16_k>(env, "attention_packed_bf16_cdna3", nk_attention_pack_size_bf16_cdna3,
                             nk_attention_pack_bf16_cdna3, nk_attention_packed_bf16_cdna3, backend);
    run_attention<nk_i8_k>(env, "attention_packed_i8_cdna3", nk_attention_pack_size_i8_cdna3,
                           nk_attention_pack_i8_cdna3, nk_attention_packed_i8_cdna3, backend);
    run_attention_gradients<nk_bf16_k>(env, "attention_packed_gradients_bf16_cdna3", nk_attention_pack_size_bf16_cdna3,
                                       nk_attention_pack_bf16_cdna3, nk_attention_packed_bf16_cdna3,
                                       nk_attention_packed_gradients_bf16_cdna3, backend);
#endif // NUMKONG_TARGET_CDNA3
}

void bench_cross_cdna4([[maybe_unused]] environment_t const &env, [[maybe_unused]] rocm_backend_t const &backend,
                       [[maybe_unused]] nk_capability_t enabled) {
#if NUMKONG_TARGET_CDNA4
    if (!(enabled & nk_cap_cdna4_k)) return;
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_cdna4", nk_dots_pack_size_bf16_cdna4, nk_dots_pack_bf16_cdna4,
                               nk_dots_packed_bf16_cdna4, backend);
    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_cdna4", nk_dots_symmetric_bf16_cdna4, backend);
    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_cdna4", nk_dots_pack_size_bf16_cdna4,
                                   nk_dots_pack_bf16_cdna4, nk_angulars_packed_bf16_cdna4, backend);
    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_cdna4", nk_angulars_symmetric_bf16_cdna4, backend);
    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_cdna4", nk_dots_pack_size_bf16_cdna4,
                                     nk_dots_pack_bf16_cdna4, nk_euclideans_packed_bf16_cdna4, backend);
    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_cdna4", nk_euclideans_symmetric_bf16_cdna4,
                                        backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_cdna4", nk_dots_pack_size_f16_cdna4, nk_dots_pack_f16_cdna4,
                              nk_dots_packed_f16_cdna4, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_cdna4", nk_dots_symmetric_f16_cdna4, backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_cdna4", nk_dots_pack_size_f16_cdna4, nk_dots_pack_f16_cdna4,
                                  nk_angulars_packed_f16_cdna4, backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_cdna4", nk_angulars_symmetric_f16_cdna4, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_cdna4", nk_dots_pack_size_f16_cdna4,
                                    nk_dots_pack_f16_cdna4, nk_euclideans_packed_f16_cdna4, backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_cdna4", nk_euclideans_symmetric_f16_cdna4,
                                       backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_cdna4", nk_dots_pack_size_e4m3_cdna4, nk_dots_pack_e4m3_cdna4,
                               nk_dots_packed_e4m3_cdna4, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_cdna4", nk_dots_symmetric_e4m3_cdna4, backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_cdna4", nk_dots_pack_size_e4m3_cdna4,
                                   nk_dots_pack_e4m3_cdna4, nk_angulars_packed_e4m3_cdna4, backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_cdna4", nk_angulars_symmetric_e4m3_cdna4, backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_cdna4", nk_dots_pack_size_e4m3_cdna4,
                                     nk_dots_pack_e4m3_cdna4, nk_euclideans_packed_e4m3_cdna4, backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_cdna4", nk_euclideans_symmetric_e4m3_cdna4,
                                        backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_cdna4", nk_dots_pack_size_e5m2_cdna4, nk_dots_pack_e5m2_cdna4,
                               nk_dots_packed_e5m2_cdna4, backend);
    run_dots_symmetric<nk_e5m2_k>(env, "dots_symmetric_e5m2_cdna4", nk_dots_symmetric_e5m2_cdna4, backend);
    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_cdna4", nk_dots_pack_size_e5m2_cdna4,
                                   nk_dots_pack_e5m2_cdna4, nk_angulars_packed_e5m2_cdna4, backend);
    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_cdna4", nk_angulars_symmetric_e5m2_cdna4, backend);
    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_cdna4", nk_dots_pack_size_e5m2_cdna4,
                                     nk_dots_pack_e5m2_cdna4, nk_euclideans_packed_e5m2_cdna4, backend);
    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_cdna4", nk_euclideans_symmetric_e5m2_cdna4,
                                        backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_cdna4", nk_dots_pack_size_e3m2_cdna4, nk_dots_pack_e3m2_cdna4,
                               nk_dots_packed_e3m2_cdna4, backend);
    run_dots_symmetric<nk_e3m2_k>(env, "dots_symmetric_e3m2_cdna4", nk_dots_symmetric_e3m2_cdna4, backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_cdna4", nk_dots_pack_size_e3m2_cdna4,
                                   nk_dots_pack_e3m2_cdna4, nk_angulars_packed_e3m2_cdna4, backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_cdna4", nk_angulars_symmetric_e3m2_cdna4, backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_cdna4", nk_dots_pack_size_e3m2_cdna4,
                                     nk_dots_pack_e3m2_cdna4, nk_euclideans_packed_e3m2_cdna4, backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_cdna4", nk_euclideans_symmetric_e3m2_cdna4,
                                        backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_cdna4", nk_dots_pack_size_e2m3_cdna4, nk_dots_pack_e2m3_cdna4,
                               nk_dots_packed_e2m3_cdna4, backend);
    run_dots_symmetric<nk_e2m3_k>(env, "dots_symmetric_e2m3_cdna4", nk_dots_symmetric_e2m3_cdna4, backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_cdna4", nk_dots_pack_size_e2m3_cdna4,
                                   nk_dots_pack_e2m3_cdna4, nk_angulars_packed_e2m3_cdna4, backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_cdna4", nk_angulars_symmetric_e2m3_cdna4, backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_cdna4", nk_dots_pack_size_e2m3_cdna4,
                                     nk_dots_pack_e2m3_cdna4, nk_euclideans_packed_e2m3_cdna4, backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_cdna4", nk_euclideans_symmetric_e2m3_cdna4,
                                        backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_cdna4", nk_dots_pack_size_e2m1_cdna4, nk_dots_pack_e2m1_cdna4,
                               nk_dots_packed_e2m1_cdna4, backend);
    run_dots_symmetric<nk_e2m1_k>(env, "dots_symmetric_e2m1_cdna4", nk_dots_symmetric_e2m1_cdna4, backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_cdna4", nk_dots_pack_size_e2m1_cdna4,
                                   nk_dots_pack_e2m1_cdna4, nk_angulars_packed_e2m1_cdna4, backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_cdna4", nk_angulars_symmetric_e2m1_cdna4, backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_cdna4", nk_dots_pack_size_e2m1_cdna4,
                                     nk_dots_pack_e2m1_cdna4, nk_euclideans_packed_e2m1_cdna4, backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_cdna4", nk_euclideans_symmetric_e2m1_cdna4,
                                        backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_cdna4", nk_dots_pack_size_i8_cdna4, nk_dots_pack_i8_cdna4,
                             nk_dots_packed_i8_cdna4, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_cdna4", nk_dots_symmetric_i8_cdna4, backend);
    run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_cdna4", nk_dots_pack_size_i8_cdna4, nk_dots_pack_i8_cdna4,
                                 nk_angulars_packed_i8_cdna4, backend);
    run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_cdna4", nk_angulars_symmetric_i8_cdna4, backend);
    run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_cdna4", nk_dots_pack_size_i8_cdna4, nk_dots_pack_i8_cdna4,
                                   nk_euclideans_packed_i8_cdna4, backend);
    run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_cdna4", nk_euclideans_symmetric_i8_cdna4, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_cdna4", nk_dots_pack_size_u8_cdna4, nk_dots_pack_u8_cdna4,
                             nk_dots_packed_u8_cdna4, backend);
    run_dots_symmetric<nk_u8_k>(env, "dots_symmetric_u8_cdna4", nk_dots_symmetric_u8_cdna4, backend);
    run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_cdna4", nk_dots_pack_size_u8_cdna4, nk_dots_pack_u8_cdna4,
                                 nk_angulars_packed_u8_cdna4, backend);
    run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_cdna4", nk_angulars_symmetric_u8_cdna4, backend);
    run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_cdna4", nk_dots_pack_size_u8_cdna4, nk_dots_pack_u8_cdna4,
                                   nk_euclideans_packed_u8_cdna4, backend);
    run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_cdna4", nk_euclideans_symmetric_u8_cdna4, backend);
    run_dots_packed<nk_i4_k>(env, "dots_packed_i4_cdna4", nk_dots_pack_size_i4_cdna4, nk_dots_pack_i4_cdna4,
                             nk_dots_packed_i4_cdna4, backend);
    run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_cdna4", nk_dots_symmetric_i4_cdna4, backend);
    run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_cdna4", nk_dots_pack_size_i4_cdna4, nk_dots_pack_i4_cdna4,
                                 nk_angulars_packed_i4_cdna4, backend);
    run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_cdna4", nk_angulars_symmetric_i4_cdna4, backend);
    run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_cdna4", nk_dots_pack_size_i4_cdna4, nk_dots_pack_i4_cdna4,
                                   nk_euclideans_packed_i4_cdna4, backend);
    run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_cdna4", nk_euclideans_symmetric_i4_cdna4, backend);
    run_dots_packed<nk_u4_k>(env, "dots_packed_u4_cdna4", nk_dots_pack_size_u4_cdna4, nk_dots_pack_u4_cdna4,
                             nk_dots_packed_u4_cdna4, backend);
    run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_cdna4", nk_dots_symmetric_u4_cdna4, backend);
    run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_cdna4", nk_dots_pack_size_u4_cdna4, nk_dots_pack_u4_cdna4,
                                 nk_angulars_packed_u4_cdna4, backend);
    run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_cdna4", nk_angulars_symmetric_u4_cdna4, backend);
    run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_cdna4", nk_dots_pack_size_u4_cdna4, nk_dots_pack_u4_cdna4,
                                   nk_euclideans_packed_u4_cdna4, backend);
    run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_cdna4", nk_euclideans_symmetric_u4_cdna4, backend);
    run_attention<nk_bf16_k>(env, "attention_packed_bf16_cdna4", nk_attention_pack_size_bf16_cdna4,
                             nk_attention_pack_bf16_cdna4, nk_attention_packed_bf16_cdna4, backend);
    run_attention<nk_e4m3_k>(env, "attention_packed_e4m3_cdna4", nk_attention_pack_size_e4m3_cdna4,
                             nk_attention_pack_e4m3_cdna4, nk_attention_packed_e4m3_cdna4, backend);
    run_attention<nk_i8_k>(env, "attention_packed_i8_cdna4", nk_attention_pack_size_i8_cdna4,
                           nk_attention_pack_i8_cdna4, nk_attention_packed_i8_cdna4, backend);
    run_attention_gradients<nk_bf16_k>(env, "attention_packed_gradients_bf16_cdna4", nk_attention_pack_size_bf16_cdna4,
                                       nk_attention_pack_bf16_cdna4, nk_attention_packed_bf16_cdna4,
                                       nk_attention_packed_gradients_bf16_cdna4, backend);
#endif // NUMKONG_TARGET_CDNA4
}
void bench_cross_cdna5([[maybe_unused]] environment_t const &env, [[maybe_unused]] rocm_backend_t const &backend,
                       [[maybe_unused]] nk_capability_t enabled) {
#if NUMKONG_TARGET_CDNA5
    if (!(enabled & nk_cap_cdna5_k)) return;
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_cdna5", nk_dots_pack_size_bf16_cdna5, nk_dots_pack_bf16_cdna5,
                               nk_dots_packed_bf16_cdna5, backend);
    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_cdna5", nk_dots_symmetric_bf16_cdna5, backend);
    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_cdna5", nk_dots_pack_size_bf16_cdna5,
                                   nk_dots_pack_bf16_cdna5, nk_angulars_packed_bf16_cdna5, backend);
    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_cdna5", nk_angulars_symmetric_bf16_cdna5, backend);
    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_cdna5", nk_dots_pack_size_bf16_cdna5,
                                     nk_dots_pack_bf16_cdna5, nk_euclideans_packed_bf16_cdna5, backend);
    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_cdna5", nk_euclideans_symmetric_bf16_cdna5,
                                        backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_cdna5", nk_dots_pack_size_f16_cdna5, nk_dots_pack_f16_cdna5,
                              nk_dots_packed_f16_cdna5, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_cdna5", nk_dots_symmetric_f16_cdna5, backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_cdna5", nk_dots_pack_size_f16_cdna5, nk_dots_pack_f16_cdna5,
                                  nk_angulars_packed_f16_cdna5, backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_cdna5", nk_angulars_symmetric_f16_cdna5, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_cdna5", nk_dots_pack_size_f16_cdna5,
                                    nk_dots_pack_f16_cdna5, nk_euclideans_packed_f16_cdna5, backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_cdna5", nk_euclideans_symmetric_f16_cdna5,
                                       backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_cdna5", nk_dots_pack_size_e4m3_cdna5, nk_dots_pack_e4m3_cdna5,
                               nk_dots_packed_e4m3_cdna5, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_cdna5", nk_dots_symmetric_e4m3_cdna5, backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_cdna5", nk_dots_pack_size_e4m3_cdna5,
                                   nk_dots_pack_e4m3_cdna5, nk_angulars_packed_e4m3_cdna5, backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_cdna5", nk_angulars_symmetric_e4m3_cdna5, backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_cdna5", nk_dots_pack_size_e4m3_cdna5,
                                     nk_dots_pack_e4m3_cdna5, nk_euclideans_packed_e4m3_cdna5, backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_cdna5", nk_euclideans_symmetric_e4m3_cdna5,
                                        backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_cdna5", nk_dots_pack_size_e5m2_cdna5, nk_dots_pack_e5m2_cdna5,
                               nk_dots_packed_e5m2_cdna5, backend);
    run_dots_symmetric<nk_e5m2_k>(env, "dots_symmetric_e5m2_cdna5", nk_dots_symmetric_e5m2_cdna5, backend);
    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_cdna5", nk_dots_pack_size_e5m2_cdna5,
                                   nk_dots_pack_e5m2_cdna5, nk_angulars_packed_e5m2_cdna5, backend);
    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_cdna5", nk_angulars_symmetric_e5m2_cdna5, backend);
    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_cdna5", nk_dots_pack_size_e5m2_cdna5,
                                     nk_dots_pack_e5m2_cdna5, nk_euclideans_packed_e5m2_cdna5, backend);
    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_cdna5", nk_euclideans_symmetric_e5m2_cdna5,
                                        backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_cdna5", nk_dots_pack_size_e3m2_cdna5, nk_dots_pack_e3m2_cdna5,
                               nk_dots_packed_e3m2_cdna5, backend);
    run_dots_symmetric<nk_e3m2_k>(env, "dots_symmetric_e3m2_cdna5", nk_dots_symmetric_e3m2_cdna5, backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_cdna5", nk_dots_pack_size_e3m2_cdna5,
                                   nk_dots_pack_e3m2_cdna5, nk_angulars_packed_e3m2_cdna5, backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_cdna5", nk_angulars_symmetric_e3m2_cdna5, backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_cdna5", nk_dots_pack_size_e3m2_cdna5,
                                     nk_dots_pack_e3m2_cdna5, nk_euclideans_packed_e3m2_cdna5, backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_cdna5", nk_euclideans_symmetric_e3m2_cdna5,
                                        backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_cdna5", nk_dots_pack_size_e2m3_cdna5, nk_dots_pack_e2m3_cdna5,
                               nk_dots_packed_e2m3_cdna5, backend);
    run_dots_symmetric<nk_e2m3_k>(env, "dots_symmetric_e2m3_cdna5", nk_dots_symmetric_e2m3_cdna5, backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_cdna5", nk_dots_pack_size_e2m3_cdna5,
                                   nk_dots_pack_e2m3_cdna5, nk_angulars_packed_e2m3_cdna5, backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_cdna5", nk_angulars_symmetric_e2m3_cdna5, backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_cdna5", nk_dots_pack_size_e2m3_cdna5,
                                     nk_dots_pack_e2m3_cdna5, nk_euclideans_packed_e2m3_cdna5, backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_cdna5", nk_euclideans_symmetric_e2m3_cdna5,
                                        backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_cdna5", nk_dots_pack_size_e2m1_cdna5, nk_dots_pack_e2m1_cdna5,
                               nk_dots_packed_e2m1_cdna5, backend);
    run_dots_symmetric<nk_e2m1_k>(env, "dots_symmetric_e2m1_cdna5", nk_dots_symmetric_e2m1_cdna5, backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_cdna5", nk_dots_pack_size_e2m1_cdna5,
                                   nk_dots_pack_e2m1_cdna5, nk_angulars_packed_e2m1_cdna5, backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_cdna5", nk_angulars_symmetric_e2m1_cdna5, backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_cdna5", nk_dots_pack_size_e2m1_cdna5,
                                     nk_dots_pack_e2m1_cdna5, nk_euclideans_packed_e2m1_cdna5, backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_cdna5", nk_euclideans_symmetric_e2m1_cdna5,
                                        backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_cdna5", nk_dots_pack_size_i8_cdna5, nk_dots_pack_i8_cdna5,
                             nk_dots_packed_i8_cdna5, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_cdna5", nk_dots_symmetric_i8_cdna5, backend);
    run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_cdna5", nk_dots_pack_size_i8_cdna5, nk_dots_pack_i8_cdna5,
                                 nk_angulars_packed_i8_cdna5, backend);
    run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_cdna5", nk_angulars_symmetric_i8_cdna5, backend);
    run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_cdna5", nk_dots_pack_size_i8_cdna5, nk_dots_pack_i8_cdna5,
                                   nk_euclideans_packed_i8_cdna5, backend);
    run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_cdna5", nk_euclideans_symmetric_i8_cdna5, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_cdna5", nk_dots_pack_size_u8_cdna5, nk_dots_pack_u8_cdna5,
                             nk_dots_packed_u8_cdna5, backend);
    run_dots_symmetric<nk_u8_k>(env, "dots_symmetric_u8_cdna5", nk_dots_symmetric_u8_cdna5, backend);
    run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_cdna5", nk_dots_pack_size_u8_cdna5, nk_dots_pack_u8_cdna5,
                                 nk_angulars_packed_u8_cdna5, backend);
    run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_cdna5", nk_angulars_symmetric_u8_cdna5, backend);
    run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_cdna5", nk_dots_pack_size_u8_cdna5, nk_dots_pack_u8_cdna5,
                                   nk_euclideans_packed_u8_cdna5, backend);
    run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_cdna5", nk_euclideans_symmetric_u8_cdna5, backend);
    run_dots_packed<nk_i4_k>(env, "dots_packed_i4_cdna5", nk_dots_pack_size_i4_cdna5, nk_dots_pack_i4_cdna5,
                             nk_dots_packed_i4_cdna5, backend);
    run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_cdna5", nk_dots_symmetric_i4_cdna5, backend);
    run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_cdna5", nk_dots_pack_size_i4_cdna5, nk_dots_pack_i4_cdna5,
                                 nk_angulars_packed_i4_cdna5, backend);
    run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_cdna5", nk_angulars_symmetric_i4_cdna5, backend);
    run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_cdna5", nk_dots_pack_size_i4_cdna5, nk_dots_pack_i4_cdna5,
                                   nk_euclideans_packed_i4_cdna5, backend);
    run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_cdna5", nk_euclideans_symmetric_i4_cdna5, backend);
    run_dots_packed<nk_u4_k>(env, "dots_packed_u4_cdna5", nk_dots_pack_size_u4_cdna5, nk_dots_pack_u4_cdna5,
                             nk_dots_packed_u4_cdna5, backend);
    run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_cdna5", nk_dots_symmetric_u4_cdna5, backend);
    run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_cdna5", nk_dots_pack_size_u4_cdna5, nk_dots_pack_u4_cdna5,
                                 nk_angulars_packed_u4_cdna5, backend);
    run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_cdna5", nk_angulars_symmetric_u4_cdna5, backend);
    run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_cdna5", nk_dots_pack_size_u4_cdna5, nk_dots_pack_u4_cdna5,
                                   nk_euclideans_packed_u4_cdna5, backend);
    run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_cdna5", nk_euclideans_symmetric_u4_cdna5, backend);
    run_attention<nk_bf16_k>(env, "attention_packed_bf16_cdna5", nk_attention_pack_size_bf16_cdna5,
                             nk_attention_pack_bf16_cdna5, nk_attention_packed_bf16_cdna5, backend);
    run_attention<nk_e4m3_k>(env, "attention_packed_e4m3_cdna5", nk_attention_pack_size_e4m3_cdna5,
                             nk_attention_pack_e4m3_cdna5, nk_attention_packed_e4m3_cdna5, backend);
    run_attention<nk_i8_k>(env, "attention_packed_i8_cdna5", nk_attention_pack_size_i8_cdna5,
                           nk_attention_pack_i8_cdna5, nk_attention_packed_i8_cdna5, backend);
    run_attention_gradients<nk_bf16_k>(env, "attention_packed_gradients_bf16_cdna5", nk_attention_pack_size_bf16_cdna5,
                                       nk_attention_pack_bf16_cdna5, nk_attention_packed_bf16_cdna5,
                                       nk_attention_packed_gradients_bf16_cdna5, backend);
#endif // NUMKONG_TARGET_CDNA5
}
} // namespace ashvardanian::numkong::bench
#endif // NUMKONG_ARCH_ROCM_

namespace ashvardanian::numkong::bench {
nk::status_t bench_cross_rocm([[maybe_unused]] environment_t const &env,
                              [[maybe_unused]] device_backend_t const &runtime) {
#if NUMKONG_ARCH_ROCM_
    device_scope<hipGetDevice, hipSetDevice> const selected(static_cast<int>(runtime.device.ordinal()));
    if (selected.status != hipSuccess) return nk::status_t::device_code_mismatch_k;
    int l2_bytes = 0;
    if (hipDeviceGetAttribute(&l2_bytes, hipDeviceAttributeL2CacheSize, static_cast<int>(runtime.device.ordinal())) !=
        hipSuccess)
        return nk::status_t::device_code_mismatch_k;
    nk_capability_t const capabilities = runtime.capabilities;
    rocm_backend_t const backend {runtime, std::size_t(std::max(l2_bytes, 0))};
    bench_cross_rocm(env, backend, capabilities);
    bench_each_rocm(env, backend, capabilities);
    bench_cross_cdna3(env, backend, capabilities);
    bench_cross_cdna4(env, backend, capabilities);
    bench_cross_cdna5(env, backend, capabilities);
    return nk::status_t::success_k;
#else  // !NUMKONG_ARCH_ROCM_
    return nk::status_t::missing_gpu_k;
#endif // NUMKONG_ARCH_ROCM_
}
} // namespace ashvardanian::numkong::bench
