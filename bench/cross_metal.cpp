/**
 *  @file bench/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batch operation benchmarks for the Metal kernels, the twin of `cross_cuda.cu`.
 *
 *  Runs the drivers of `cross.hpp` through a @c metal_backend_t, on an owned queue on the selected
 *  device, over its unified memory. Metal has no C-level events, so every window of launches is
 *  timed by wall clock through the synchronization after it. Input sets rotate until their
 *  footprint is at least twice the system-level cache.
 */
#include "numkong/numkong.h" // `nk_dots_packed_i8_metal`

#include "cross.hpp"
#include "harness_metal.hpp"

namespace ashvardanian::numkong::bench {

/** Every Metal baseline entry point beside every Apple9 and Apple10 one, so the matrix units'
 *  speedup shows, on devices whose families include each. */
nk::status_t bench_cross_metal([[maybe_unused]] environment_t const &env,
                               [[maybe_unused]] device_backend_t const &runtime) {
#if NUMKONG_ARCH_METAL_
    nk_capability_t const enabled = runtime.capabilities;
    metal_backend_t const backend {runtime};
    if (enabled & nk_cap_metal_k) {
        run_dots_packed<nk_nvfp4_k>(env, "dots_packed_nvfp4_metal", nk_dots_pack_size_nvfp4_metal,
                                    nk_dots_pack_nvfp4_metal, nk_dots_packed_nvfp4_metal, backend);
        run_dots_symmetric<nk_nvfp4_k>(env, "dots_symmetric_nvfp4_metal", nk_dots_symmetric_nvfp4_metal, backend);
        run_angulars_packed<nk_nvfp4_k>(env, "angulars_packed_nvfp4_metal", nk_dots_pack_size_nvfp4_metal,
                                        nk_dots_pack_nvfp4_metal, nk_angulars_packed_nvfp4_metal, backend);
        run_angulars_symmetric<nk_nvfp4_k>(env, "angulars_symmetric_nvfp4_metal", nk_angulars_symmetric_nvfp4_metal,
                                           backend);
        run_euclideans_packed<nk_nvfp4_k>(env, "euclideans_packed_nvfp4_metal", nk_dots_pack_size_nvfp4_metal,
                                          nk_dots_pack_nvfp4_metal, nk_euclideans_packed_nvfp4_metal, backend);
        run_euclideans_symmetric<nk_nvfp4_k>(env, "euclideans_symmetric_nvfp4_metal",
                                             nk_euclideans_symmetric_nvfp4_metal, backend);
        run_dots_packed<nk_mxfp4_k>(env, "dots_packed_mxfp4_metal", nk_dots_pack_size_mxfp4_metal,
                                    nk_dots_pack_mxfp4_metal, nk_dots_packed_mxfp4_metal, backend);
        run_dots_symmetric<nk_mxfp4_k>(env, "dots_symmetric_mxfp4_metal", nk_dots_symmetric_mxfp4_metal, backend);
        run_angulars_packed<nk_mxfp4_k>(env, "angulars_packed_mxfp4_metal", nk_dots_pack_size_mxfp4_metal,
                                        nk_dots_pack_mxfp4_metal, nk_angulars_packed_mxfp4_metal, backend);
        run_angulars_symmetric<nk_mxfp4_k>(env, "angulars_symmetric_mxfp4_metal", nk_angulars_symmetric_mxfp4_metal,
                                           backend);
        run_euclideans_packed<nk_mxfp4_k>(env, "euclideans_packed_mxfp4_metal", nk_dots_pack_size_mxfp4_metal,
                                          nk_dots_pack_mxfp4_metal, nk_euclideans_packed_mxfp4_metal, backend);
        run_euclideans_symmetric<nk_mxfp4_k>(env, "euclideans_symmetric_mxfp4_metal",
                                             nk_euclideans_symmetric_mxfp4_metal, backend);
        run_dots_packed<nk_mxfp6e2m3_k>(env, "dots_packed_mxfp6e2m3_metal", nk_dots_pack_size_mxfp6e2m3_metal,
                                        nk_dots_pack_mxfp6e2m3_metal, nk_dots_packed_mxfp6e2m3_metal, backend);
        run_dots_symmetric<nk_mxfp6e2m3_k>(env, "dots_symmetric_mxfp6e2m3_metal", nk_dots_symmetric_mxfp6e2m3_metal,
                                           backend);
        run_angulars_packed<nk_mxfp6e2m3_k>(env, "angulars_packed_mxfp6e2m3_metal", nk_dots_pack_size_mxfp6e2m3_metal,
                                            nk_dots_pack_mxfp6e2m3_metal, nk_angulars_packed_mxfp6e2m3_metal, backend);
        run_angulars_symmetric<nk_mxfp6e2m3_k>(env, "angulars_symmetric_mxfp6e2m3_metal",
                                               nk_angulars_symmetric_mxfp6e2m3_metal, backend);
        run_euclideans_packed<nk_mxfp6e2m3_k>(env, "euclideans_packed_mxfp6e2m3_metal",
                                              nk_dots_pack_size_mxfp6e2m3_metal, nk_dots_pack_mxfp6e2m3_metal,
                                              nk_euclideans_packed_mxfp6e2m3_metal, backend);
        run_euclideans_symmetric<nk_mxfp6e2m3_k>(env, "euclideans_symmetric_mxfp6e2m3_metal",
                                                 nk_euclideans_symmetric_mxfp6e2m3_metal, backend);
        run_dots_packed<nk_mxfp6e3m2_k>(env, "dots_packed_mxfp6e3m2_metal", nk_dots_pack_size_mxfp6e3m2_metal,
                                        nk_dots_pack_mxfp6e3m2_metal, nk_dots_packed_mxfp6e3m2_metal, backend);
        run_dots_symmetric<nk_mxfp6e3m2_k>(env, "dots_symmetric_mxfp6e3m2_metal", nk_dots_symmetric_mxfp6e3m2_metal,
                                           backend);
        run_angulars_packed<nk_mxfp6e3m2_k>(env, "angulars_packed_mxfp6e3m2_metal", nk_dots_pack_size_mxfp6e3m2_metal,
                                            nk_dots_pack_mxfp6e3m2_metal, nk_angulars_packed_mxfp6e3m2_metal, backend);
        run_angulars_symmetric<nk_mxfp6e3m2_k>(env, "angulars_symmetric_mxfp6e3m2_metal",
                                               nk_angulars_symmetric_mxfp6e3m2_metal, backend);
        run_euclideans_packed<nk_mxfp6e3m2_k>(env, "euclideans_packed_mxfp6e3m2_metal",
                                              nk_dots_pack_size_mxfp6e3m2_metal, nk_dots_pack_mxfp6e3m2_metal,
                                              nk_euclideans_packed_mxfp6e3m2_metal, backend);
        run_euclideans_symmetric<nk_mxfp6e3m2_k>(env, "euclideans_symmetric_mxfp6e3m2_metal",
                                                 nk_euclideans_symmetric_mxfp6e3m2_metal, backend);
        run_dots_packed<nk_mxfp8e4m3_k>(env, "dots_packed_mxfp8e4m3_metal", nk_dots_pack_size_mxfp8e4m3_metal,
                                        nk_dots_pack_mxfp8e4m3_metal, nk_dots_packed_mxfp8e4m3_metal, backend);
        run_dots_symmetric<nk_mxfp8e4m3_k>(env, "dots_symmetric_mxfp8e4m3_metal", nk_dots_symmetric_mxfp8e4m3_metal,
                                           backend);
        run_angulars_packed<nk_mxfp8e4m3_k>(env, "angulars_packed_mxfp8e4m3_metal", nk_dots_pack_size_mxfp8e4m3_metal,
                                            nk_dots_pack_mxfp8e4m3_metal, nk_angulars_packed_mxfp8e4m3_metal, backend);
        run_angulars_symmetric<nk_mxfp8e4m3_k>(env, "angulars_symmetric_mxfp8e4m3_metal",
                                               nk_angulars_symmetric_mxfp8e4m3_metal, backend);
        run_euclideans_packed<nk_mxfp8e4m3_k>(env, "euclideans_packed_mxfp8e4m3_metal",
                                              nk_dots_pack_size_mxfp8e4m3_metal, nk_dots_pack_mxfp8e4m3_metal,
                                              nk_euclideans_packed_mxfp8e4m3_metal, backend);
        run_euclideans_symmetric<nk_mxfp8e4m3_k>(env, "euclideans_symmetric_mxfp8e4m3_metal",
                                                 nk_euclideans_symmetric_mxfp8e4m3_metal, backend);
        run_dots_packed<nk_mxfp8e5m2_k>(env, "dots_packed_mxfp8e5m2_metal", nk_dots_pack_size_mxfp8e5m2_metal,
                                        nk_dots_pack_mxfp8e5m2_metal, nk_dots_packed_mxfp8e5m2_metal, backend);
        run_dots_symmetric<nk_mxfp8e5m2_k>(env, "dots_symmetric_mxfp8e5m2_metal", nk_dots_symmetric_mxfp8e5m2_metal,
                                           backend);
        run_angulars_packed<nk_mxfp8e5m2_k>(env, "angulars_packed_mxfp8e5m2_metal", nk_dots_pack_size_mxfp8e5m2_metal,
                                            nk_dots_pack_mxfp8e5m2_metal, nk_angulars_packed_mxfp8e5m2_metal, backend);
        run_angulars_symmetric<nk_mxfp8e5m2_k>(env, "angulars_symmetric_mxfp8e5m2_metal",
                                               nk_angulars_symmetric_mxfp8e5m2_metal, backend);
        run_euclideans_packed<nk_mxfp8e5m2_k>(env, "euclideans_packed_mxfp8e5m2_metal",
                                              nk_dots_pack_size_mxfp8e5m2_metal, nk_dots_pack_mxfp8e5m2_metal,
                                              nk_euclideans_packed_mxfp8e5m2_metal, backend);
        run_euclideans_symmetric<nk_mxfp8e5m2_k>(env, "euclideans_symmetric_mxfp8e5m2_metal",
                                                 nk_euclideans_symmetric_mxfp8e5m2_metal, backend);
        run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_metal", nk_dots_pack_size_bf16_metal, nk_dots_pack_bf16_metal,
                                   nk_dots_packed_bf16_metal, backend);
        run_dots_packed<nk_f16_k>(env, "dots_packed_f16_metal", nk_dots_pack_size_f16_metal, nk_dots_pack_f16_metal,
                                  nk_dots_packed_f16_metal, backend);
        run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_metal", nk_dots_pack_size_e5m2_metal, nk_dots_pack_e5m2_metal,
                                   nk_dots_packed_e5m2_metal, backend);
        run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_metal", nk_dots_pack_size_e4m3_metal, nk_dots_pack_e4m3_metal,
                                   nk_dots_packed_e4m3_metal, backend);
        run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_metal", nk_dots_pack_size_e3m2_metal, nk_dots_pack_e3m2_metal,
                                   nk_dots_packed_e3m2_metal, backend);
        run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_metal", nk_dots_pack_size_e2m3_metal, nk_dots_pack_e2m3_metal,
                                   nk_dots_packed_e2m3_metal, backend);
        run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_metal", nk_dots_pack_size_e2m1_metal, nk_dots_pack_e2m1_metal,
                                   nk_dots_packed_e2m1_metal, backend);
        run_dots_packed<nk_i8_k>(env, "dots_packed_i8_metal", nk_dots_pack_size_i8_metal, nk_dots_pack_i8_metal,
                                 nk_dots_packed_i8_metal, backend);
        run_dots_packed<nk_u8_k>(env, "dots_packed_u8_metal", nk_dots_pack_size_u8_metal, nk_dots_pack_u8_metal,
                                 nk_dots_packed_u8_metal, backend);

        run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_metal", nk_dots_symmetric_bf16_metal, backend);
        run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_metal", nk_dots_symmetric_f16_metal, backend);
        run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_metal", nk_dots_symmetric_e4m3_metal, backend);
        run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_metal", nk_dots_symmetric_i8_metal, backend);
        run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_metal", nk_dots_pack_size_i8_metal, nk_dots_pack_i8_metal,
                                     nk_angulars_packed_i8_metal, backend);
        run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_metal", nk_angulars_symmetric_i8_metal, backend);
        run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_metal", nk_dots_pack_size_i8_metal,
                                       nk_dots_pack_i8_metal, nk_euclideans_packed_i8_metal, backend);
        run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_metal", nk_euclideans_symmetric_i8_metal,
                                          backend);
        run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_metal", nk_dots_pack_size_u8_metal, nk_dots_pack_u8_metal,
                                     nk_angulars_packed_u8_metal, backend);
        run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_metal", nk_angulars_symmetric_u8_metal, backend);
        run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_metal", nk_dots_pack_size_u8_metal,
                                       nk_dots_pack_u8_metal, nk_euclideans_packed_u8_metal, backend);
        run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_metal", nk_euclideans_symmetric_u8_metal,
                                          backend);
        run_dots_packed<nk_i4_k>(env, "dots_packed_i4_metal", nk_dots_pack_size_i4_metal, nk_dots_pack_i4_metal,
                                 nk_dots_packed_i4_metal, backend);
        run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_metal", nk_dots_symmetric_i4_metal, backend);
        run_dots_packed<nk_u4_k>(env, "dots_packed_u4_metal", nk_dots_pack_size_u4_metal, nk_dots_pack_u4_metal,
                                 nk_dots_packed_u4_metal, backend);
        run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_metal", nk_dots_symmetric_u4_metal, backend);
        run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_metal", nk_dots_pack_size_i4_metal, nk_dots_pack_i4_metal,
                                     nk_angulars_packed_i4_metal, backend);
        run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_metal", nk_angulars_symmetric_i4_metal, backend);
        run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_metal", nk_dots_pack_size_i4_metal,
                                       nk_dots_pack_i4_metal, nk_euclideans_packed_i4_metal, backend);
        run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_metal", nk_euclideans_symmetric_i4_metal,
                                          backend);
        run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_metal", nk_dots_pack_size_u4_metal, nk_dots_pack_u4_metal,
                                     nk_angulars_packed_u4_metal, backend);
        run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_metal", nk_angulars_symmetric_u4_metal, backend);
        run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_metal", nk_dots_pack_size_u4_metal,
                                       nk_dots_pack_u4_metal, nk_euclideans_packed_u4_metal, backend);
        run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_metal", nk_euclideans_symmetric_u4_metal,
                                          backend);
        run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_metal", nk_dots_pack_size_f16_metal,
                                      nk_dots_pack_f16_metal, nk_angulars_packed_f16_metal, backend);
        run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_metal", nk_angulars_symmetric_f16_metal, backend);
        run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_metal", nk_dots_pack_size_f16_metal,
                                        nk_dots_pack_f16_metal, nk_euclideans_packed_f16_metal, backend);
        run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_metal", nk_euclideans_symmetric_f16_metal,
                                           backend);
        run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_metal", nk_dots_pack_size_bf16_metal,
                                       nk_dots_pack_bf16_metal, nk_angulars_packed_bf16_metal, backend);
        run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_metal", nk_angulars_symmetric_bf16_metal,
                                          backend);
        run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_metal", nk_dots_pack_size_bf16_metal,
                                         nk_dots_pack_bf16_metal, nk_euclideans_packed_bf16_metal, backend);
        run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_metal", nk_euclideans_symmetric_bf16_metal,
                                            backend);
        run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_metal", nk_dots_pack_size_e4m3_metal,
                                       nk_dots_pack_e4m3_metal, nk_angulars_packed_e4m3_metal, backend);
        run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_metal", nk_angulars_symmetric_e4m3_metal,
                                          backend);
        run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_metal", nk_dots_pack_size_e4m3_metal,
                                         nk_dots_pack_e4m3_metal, nk_euclideans_packed_e4m3_metal, backend);
        run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_metal", nk_euclideans_symmetric_e4m3_metal,
                                            backend);
        run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_metal", nk_dots_pack_size_e5m2_metal,
                                       nk_dots_pack_e5m2_metal, nk_angulars_packed_e5m2_metal, backend);
        run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_metal", nk_angulars_symmetric_e5m2_metal,
                                          backend);
        run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_metal", nk_dots_pack_size_e5m2_metal,
                                         nk_dots_pack_e5m2_metal, nk_euclideans_packed_e5m2_metal, backend);
        run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_metal", nk_euclideans_symmetric_e5m2_metal,
                                            backend);
        run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_metal", nk_dots_pack_size_e3m2_metal,
                                       nk_dots_pack_e3m2_metal, nk_angulars_packed_e3m2_metal, backend);
        run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_metal", nk_angulars_symmetric_e3m2_metal,
                                          backend);
        run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_metal", nk_dots_pack_size_e3m2_metal,
                                         nk_dots_pack_e3m2_metal, nk_euclideans_packed_e3m2_metal, backend);
        run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_metal", nk_euclideans_symmetric_e3m2_metal,
                                            backend);
        run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_metal", nk_dots_pack_size_e2m3_metal,
                                       nk_dots_pack_e2m3_metal, nk_angulars_packed_e2m3_metal, backend);
        run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_metal", nk_angulars_symmetric_e2m3_metal,
                                          backend);
        run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_metal", nk_dots_pack_size_e2m3_metal,
                                         nk_dots_pack_e2m3_metal, nk_euclideans_packed_e2m3_metal, backend);
        run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_metal", nk_euclideans_symmetric_e2m3_metal,
                                            backend);
        run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_metal", nk_dots_pack_size_e2m1_metal,
                                       nk_dots_pack_e2m1_metal, nk_angulars_packed_e2m1_metal, backend);
        run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_metal", nk_angulars_symmetric_e2m1_metal,
                                          backend);
        run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_metal", nk_dots_pack_size_e2m1_metal,
                                         nk_dots_pack_e2m1_metal, nk_euclideans_packed_e2m1_metal, backend);
        run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_metal", nk_euclideans_symmetric_e2m1_metal,
                                            backend);
    }
#if NUMKONG_TARGET_APPLE9
    if (enabled & nk_cap_apple9_k) {
        run_dots_packed<nk_nvfp4_k>(env, "dots_packed_nvfp4_apple9", nk_dots_pack_size_nvfp4_apple9,
                                    nk_dots_pack_nvfp4_apple9, nk_dots_packed_nvfp4_apple9, backend);
        run_dots_symmetric<nk_nvfp4_k>(env, "dots_symmetric_nvfp4_apple9", nk_dots_symmetric_nvfp4_apple9, backend);
        run_angulars_packed<nk_nvfp4_k>(env, "angulars_packed_nvfp4_apple9", nk_dots_pack_size_nvfp4_apple9,
                                        nk_dots_pack_nvfp4_apple9, nk_angulars_packed_nvfp4_apple9, backend);
        run_angulars_symmetric<nk_nvfp4_k>(env, "angulars_symmetric_nvfp4_apple9", nk_angulars_symmetric_nvfp4_apple9,
                                           backend);
        run_euclideans_packed<nk_nvfp4_k>(env, "euclideans_packed_nvfp4_apple9", nk_dots_pack_size_nvfp4_apple9,
                                          nk_dots_pack_nvfp4_apple9, nk_euclideans_packed_nvfp4_apple9, backend);
        run_euclideans_symmetric<nk_nvfp4_k>(env, "euclideans_symmetric_nvfp4_apple9",
                                             nk_euclideans_symmetric_nvfp4_apple9, backend);
        run_dots_packed<nk_mxfp4_k>(env, "dots_packed_mxfp4_apple9", nk_dots_pack_size_mxfp4_apple9,
                                    nk_dots_pack_mxfp4_apple9, nk_dots_packed_mxfp4_apple9, backend);
        run_dots_symmetric<nk_mxfp4_k>(env, "dots_symmetric_mxfp4_apple9", nk_dots_symmetric_mxfp4_apple9, backend);
        run_angulars_packed<nk_mxfp4_k>(env, "angulars_packed_mxfp4_apple9", nk_dots_pack_size_mxfp4_apple9,
                                        nk_dots_pack_mxfp4_apple9, nk_angulars_packed_mxfp4_apple9, backend);
        run_angulars_symmetric<nk_mxfp4_k>(env, "angulars_symmetric_mxfp4_apple9", nk_angulars_symmetric_mxfp4_apple9,
                                           backend);
        run_euclideans_packed<nk_mxfp4_k>(env, "euclideans_packed_mxfp4_apple9", nk_dots_pack_size_mxfp4_apple9,
                                          nk_dots_pack_mxfp4_apple9, nk_euclideans_packed_mxfp4_apple9, backend);
        run_euclideans_symmetric<nk_mxfp4_k>(env, "euclideans_symmetric_mxfp4_apple9",
                                             nk_euclideans_symmetric_mxfp4_apple9, backend);
        run_dots_packed<nk_mxfp6e2m3_k>(env, "dots_packed_mxfp6e2m3_apple9", nk_dots_pack_size_mxfp6e2m3_apple9,
                                        nk_dots_pack_mxfp6e2m3_apple9, nk_dots_packed_mxfp6e2m3_apple9, backend);
        run_dots_symmetric<nk_mxfp6e2m3_k>(env, "dots_symmetric_mxfp6e2m3_apple9", nk_dots_symmetric_mxfp6e2m3_apple9,
                                           backend);
        run_angulars_packed<nk_mxfp6e2m3_k>(env, "angulars_packed_mxfp6e2m3_apple9", nk_dots_pack_size_mxfp6e2m3_apple9,
                                            nk_dots_pack_mxfp6e2m3_apple9, nk_angulars_packed_mxfp6e2m3_apple9,
                                            backend);
        run_angulars_symmetric<nk_mxfp6e2m3_k>(env, "angulars_symmetric_mxfp6e2m3_apple9",
                                               nk_angulars_symmetric_mxfp6e2m3_apple9, backend);
        run_euclideans_packed<nk_mxfp6e2m3_k>(env, "euclideans_packed_mxfp6e2m3_apple9",
                                              nk_dots_pack_size_mxfp6e2m3_apple9, nk_dots_pack_mxfp6e2m3_apple9,
                                              nk_euclideans_packed_mxfp6e2m3_apple9, backend);
        run_euclideans_symmetric<nk_mxfp6e2m3_k>(env, "euclideans_symmetric_mxfp6e2m3_apple9",
                                                 nk_euclideans_symmetric_mxfp6e2m3_apple9, backend);
        run_dots_packed<nk_mxfp6e3m2_k>(env, "dots_packed_mxfp6e3m2_apple9", nk_dots_pack_size_mxfp6e3m2_apple9,
                                        nk_dots_pack_mxfp6e3m2_apple9, nk_dots_packed_mxfp6e3m2_apple9, backend);
        run_dots_symmetric<nk_mxfp6e3m2_k>(env, "dots_symmetric_mxfp6e3m2_apple9", nk_dots_symmetric_mxfp6e3m2_apple9,
                                           backend);
        run_angulars_packed<nk_mxfp6e3m2_k>(env, "angulars_packed_mxfp6e3m2_apple9", nk_dots_pack_size_mxfp6e3m2_apple9,
                                            nk_dots_pack_mxfp6e3m2_apple9, nk_angulars_packed_mxfp6e3m2_apple9,
                                            backend);
        run_angulars_symmetric<nk_mxfp6e3m2_k>(env, "angulars_symmetric_mxfp6e3m2_apple9",
                                               nk_angulars_symmetric_mxfp6e3m2_apple9, backend);
        run_euclideans_packed<nk_mxfp6e3m2_k>(env, "euclideans_packed_mxfp6e3m2_apple9",
                                              nk_dots_pack_size_mxfp6e3m2_apple9, nk_dots_pack_mxfp6e3m2_apple9,
                                              nk_euclideans_packed_mxfp6e3m2_apple9, backend);
        run_euclideans_symmetric<nk_mxfp6e3m2_k>(env, "euclideans_symmetric_mxfp6e3m2_apple9",
                                                 nk_euclideans_symmetric_mxfp6e3m2_apple9, backend);
        run_dots_packed<nk_mxfp8e4m3_k>(env, "dots_packed_mxfp8e4m3_apple9", nk_dots_pack_size_mxfp8e4m3_apple9,
                                        nk_dots_pack_mxfp8e4m3_apple9, nk_dots_packed_mxfp8e4m3_apple9, backend);
        run_dots_symmetric<nk_mxfp8e4m3_k>(env, "dots_symmetric_mxfp8e4m3_apple9", nk_dots_symmetric_mxfp8e4m3_apple9,
                                           backend);
        run_angulars_packed<nk_mxfp8e4m3_k>(env, "angulars_packed_mxfp8e4m3_apple9", nk_dots_pack_size_mxfp8e4m3_apple9,
                                            nk_dots_pack_mxfp8e4m3_apple9, nk_angulars_packed_mxfp8e4m3_apple9,
                                            backend);
        run_angulars_symmetric<nk_mxfp8e4m3_k>(env, "angulars_symmetric_mxfp8e4m3_apple9",
                                               nk_angulars_symmetric_mxfp8e4m3_apple9, backend);
        run_euclideans_packed<nk_mxfp8e4m3_k>(env, "euclideans_packed_mxfp8e4m3_apple9",
                                              nk_dots_pack_size_mxfp8e4m3_apple9, nk_dots_pack_mxfp8e4m3_apple9,
                                              nk_euclideans_packed_mxfp8e4m3_apple9, backend);
        run_euclideans_symmetric<nk_mxfp8e4m3_k>(env, "euclideans_symmetric_mxfp8e4m3_apple9",
                                                 nk_euclideans_symmetric_mxfp8e4m3_apple9, backend);
        run_dots_packed<nk_mxfp8e5m2_k>(env, "dots_packed_mxfp8e5m2_apple9", nk_dots_pack_size_mxfp8e5m2_apple9,
                                        nk_dots_pack_mxfp8e5m2_apple9, nk_dots_packed_mxfp8e5m2_apple9, backend);
        run_dots_symmetric<nk_mxfp8e5m2_k>(env, "dots_symmetric_mxfp8e5m2_apple9", nk_dots_symmetric_mxfp8e5m2_apple9,
                                           backend);
        run_angulars_packed<nk_mxfp8e5m2_k>(env, "angulars_packed_mxfp8e5m2_apple9", nk_dots_pack_size_mxfp8e5m2_apple9,
                                            nk_dots_pack_mxfp8e5m2_apple9, nk_angulars_packed_mxfp8e5m2_apple9,
                                            backend);
        run_angulars_symmetric<nk_mxfp8e5m2_k>(env, "angulars_symmetric_mxfp8e5m2_apple9",
                                               nk_angulars_symmetric_mxfp8e5m2_apple9, backend);
        run_euclideans_packed<nk_mxfp8e5m2_k>(env, "euclideans_packed_mxfp8e5m2_apple9",
                                              nk_dots_pack_size_mxfp8e5m2_apple9, nk_dots_pack_mxfp8e5m2_apple9,
                                              nk_euclideans_packed_mxfp8e5m2_apple9, backend);
        run_euclideans_symmetric<nk_mxfp8e5m2_k>(env, "euclideans_symmetric_mxfp8e5m2_apple9",
                                                 nk_euclideans_symmetric_mxfp8e5m2_apple9, backend);
        run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_apple9", nk_dots_pack_size_bf16_apple9,
                                   nk_dots_pack_bf16_apple9, nk_dots_packed_bf16_apple9, backend);
        run_dots_packed<nk_f16_k>(env, "dots_packed_f16_apple9", nk_dots_pack_size_f16_apple9, nk_dots_pack_f16_apple9,
                                  nk_dots_packed_f16_apple9, backend);
        run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_apple9", nk_dots_pack_size_e5m2_apple9,
                                   nk_dots_pack_e5m2_apple9, nk_dots_packed_e5m2_apple9, backend);
        run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_apple9", nk_dots_pack_size_e4m3_apple9,
                                   nk_dots_pack_e4m3_apple9, nk_dots_packed_e4m3_apple9, backend);
        run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_apple9", nk_dots_pack_size_e3m2_apple9,
                                   nk_dots_pack_e3m2_apple9, nk_dots_packed_e3m2_apple9, backend);
        run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_apple9", nk_dots_pack_size_e2m3_apple9,
                                   nk_dots_pack_e2m3_apple9, nk_dots_packed_e2m3_apple9, backend);
        run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_apple9", nk_dots_pack_size_e2m1_apple9,
                                   nk_dots_pack_e2m1_apple9, nk_dots_packed_e2m1_apple9, backend);

        run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_apple9", nk_dots_symmetric_bf16_apple9, backend);
        run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_apple9", nk_dots_symmetric_f16_apple9, backend);
        run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_apple9", nk_dots_symmetric_e4m3_apple9, backend);
        run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_apple9", nk_dots_pack_size_f16_apple9,
                                      nk_dots_pack_f16_apple9, nk_angulars_packed_f16_apple9, backend);
        run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_apple9", nk_angulars_symmetric_f16_apple9,
                                         backend);
        run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_apple9", nk_dots_pack_size_f16_apple9,
                                        nk_dots_pack_f16_apple9, nk_euclideans_packed_f16_apple9, backend);
        run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_apple9", nk_euclideans_symmetric_f16_apple9,
                                           backend);
        run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_apple9", nk_dots_pack_size_bf16_apple9,
                                       nk_dots_pack_bf16_apple9, nk_angulars_packed_bf16_apple9, backend);
        run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_apple9", nk_angulars_symmetric_bf16_apple9,
                                          backend);
        run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_apple9", nk_dots_pack_size_bf16_apple9,
                                         nk_dots_pack_bf16_apple9, nk_euclideans_packed_bf16_apple9, backend);
        run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_apple9",
                                            nk_euclideans_symmetric_bf16_apple9, backend);
        run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_apple9", nk_dots_pack_size_e4m3_apple9,
                                       nk_dots_pack_e4m3_apple9, nk_angulars_packed_e4m3_apple9, backend);
        run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_apple9", nk_angulars_symmetric_e4m3_apple9,
                                          backend);
        run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_apple9", nk_dots_pack_size_e4m3_apple9,
                                         nk_dots_pack_e4m3_apple9, nk_euclideans_packed_e4m3_apple9, backend);
        run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_apple9",
                                            nk_euclideans_symmetric_e4m3_apple9, backend);
        run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_apple9", nk_dots_pack_size_e5m2_apple9,
                                       nk_dots_pack_e5m2_apple9, nk_angulars_packed_e5m2_apple9, backend);
        run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_apple9", nk_angulars_symmetric_e5m2_apple9,
                                          backend);
        run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_apple9", nk_dots_pack_size_e5m2_apple9,
                                         nk_dots_pack_e5m2_apple9, nk_euclideans_packed_e5m2_apple9, backend);
        run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_apple9",
                                            nk_euclideans_symmetric_e5m2_apple9, backend);
        run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_apple9", nk_dots_pack_size_e3m2_apple9,
                                       nk_dots_pack_e3m2_apple9, nk_angulars_packed_e3m2_apple9, backend);
        run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_apple9", nk_angulars_symmetric_e3m2_apple9,
                                          backend);
        run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_apple9", nk_dots_pack_size_e3m2_apple9,
                                         nk_dots_pack_e3m2_apple9, nk_euclideans_packed_e3m2_apple9, backend);
        run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_apple9",
                                            nk_euclideans_symmetric_e3m2_apple9, backend);
        run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_apple9", nk_dots_pack_size_e2m3_apple9,
                                       nk_dots_pack_e2m3_apple9, nk_angulars_packed_e2m3_apple9, backend);
        run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_apple9", nk_angulars_symmetric_e2m3_apple9,
                                          backend);
        run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_apple9", nk_dots_pack_size_e2m3_apple9,
                                         nk_dots_pack_e2m3_apple9, nk_euclideans_packed_e2m3_apple9, backend);
        run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_apple9",
                                            nk_euclideans_symmetric_e2m3_apple9, backend);
        run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_apple9", nk_dots_pack_size_e2m1_apple9,
                                       nk_dots_pack_e2m1_apple9, nk_angulars_packed_e2m1_apple9, backend);
        run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_apple9", nk_angulars_symmetric_e2m1_apple9,
                                          backend);
        run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_apple9", nk_dots_pack_size_e2m1_apple9,
                                         nk_dots_pack_e2m1_apple9, nk_euclideans_packed_e2m1_apple9, backend);
        run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_apple9",
                                            nk_euclideans_symmetric_e2m1_apple9, backend);
    }
#endif // NUMKONG_TARGET_APPLE9
#if NUMKONG_TARGET_APPLE10
    if (!(enabled & nk_cap_apple10_k)) return nk::status_t::success_k;
    run_dots_packed<nk_nvfp4_k>(env, "dots_packed_nvfp4_apple10", nk_dots_pack_size_nvfp4_apple10,
                                nk_dots_pack_nvfp4_apple10, nk_dots_packed_nvfp4_apple10, backend);
    run_dots_symmetric<nk_nvfp4_k>(env, "dots_symmetric_nvfp4_apple10", nk_dots_symmetric_nvfp4_apple10, backend);
    run_angulars_packed<nk_nvfp4_k>(env, "angulars_packed_nvfp4_apple10", nk_dots_pack_size_nvfp4_apple10,
                                    nk_dots_pack_nvfp4_apple10, nk_angulars_packed_nvfp4_apple10, backend);
    run_angulars_symmetric<nk_nvfp4_k>(env, "angulars_symmetric_nvfp4_apple10", nk_angulars_symmetric_nvfp4_apple10,
                                       backend);
    run_euclideans_packed<nk_nvfp4_k>(env, "euclideans_packed_nvfp4_apple10", nk_dots_pack_size_nvfp4_apple10,
                                      nk_dots_pack_nvfp4_apple10, nk_euclideans_packed_nvfp4_apple10, backend);
    run_euclideans_symmetric<nk_nvfp4_k>(env, "euclideans_symmetric_nvfp4_apple10",
                                         nk_euclideans_symmetric_nvfp4_apple10, backend);
    run_dots_packed<nk_mxfp4_k>(env, "dots_packed_mxfp4_apple10", nk_dots_pack_size_mxfp4_apple10,
                                nk_dots_pack_mxfp4_apple10, nk_dots_packed_mxfp4_apple10, backend);
    run_dots_symmetric<nk_mxfp4_k>(env, "dots_symmetric_mxfp4_apple10", nk_dots_symmetric_mxfp4_apple10, backend);
    run_angulars_packed<nk_mxfp4_k>(env, "angulars_packed_mxfp4_apple10", nk_dots_pack_size_mxfp4_apple10,
                                    nk_dots_pack_mxfp4_apple10, nk_angulars_packed_mxfp4_apple10, backend);
    run_angulars_symmetric<nk_mxfp4_k>(env, "angulars_symmetric_mxfp4_apple10", nk_angulars_symmetric_mxfp4_apple10,
                                       backend);
    run_euclideans_packed<nk_mxfp4_k>(env, "euclideans_packed_mxfp4_apple10", nk_dots_pack_size_mxfp4_apple10,
                                      nk_dots_pack_mxfp4_apple10, nk_euclideans_packed_mxfp4_apple10, backend);
    run_euclideans_symmetric<nk_mxfp4_k>(env, "euclideans_symmetric_mxfp4_apple10",
                                         nk_euclideans_symmetric_mxfp4_apple10, backend);
    run_dots_packed<nk_mxfp6e2m3_k>(env, "dots_packed_mxfp6e2m3_apple10", nk_dots_pack_size_mxfp6e2m3_apple10,
                                    nk_dots_pack_mxfp6e2m3_apple10, nk_dots_packed_mxfp6e2m3_apple10, backend);
    run_dots_symmetric<nk_mxfp6e2m3_k>(env, "dots_symmetric_mxfp6e2m3_apple10", nk_dots_symmetric_mxfp6e2m3_apple10,
                                       backend);
    run_angulars_packed<nk_mxfp6e2m3_k>(env, "angulars_packed_mxfp6e2m3_apple10", nk_dots_pack_size_mxfp6e2m3_apple10,
                                        nk_dots_pack_mxfp6e2m3_apple10, nk_angulars_packed_mxfp6e2m3_apple10, backend);
    run_angulars_symmetric<nk_mxfp6e2m3_k>(env, "angulars_symmetric_mxfp6e2m3_apple10",
                                           nk_angulars_symmetric_mxfp6e2m3_apple10, backend);
    run_euclideans_packed<nk_mxfp6e2m3_k>(env, "euclideans_packed_mxfp6e2m3_apple10",
                                          nk_dots_pack_size_mxfp6e2m3_apple10, nk_dots_pack_mxfp6e2m3_apple10,
                                          nk_euclideans_packed_mxfp6e2m3_apple10, backend);
    run_euclideans_symmetric<nk_mxfp6e2m3_k>(env, "euclideans_symmetric_mxfp6e2m3_apple10",
                                             nk_euclideans_symmetric_mxfp6e2m3_apple10, backend);
    run_dots_packed<nk_mxfp6e3m2_k>(env, "dots_packed_mxfp6e3m2_apple10", nk_dots_pack_size_mxfp6e3m2_apple10,
                                    nk_dots_pack_mxfp6e3m2_apple10, nk_dots_packed_mxfp6e3m2_apple10, backend);
    run_dots_symmetric<nk_mxfp6e3m2_k>(env, "dots_symmetric_mxfp6e3m2_apple10", nk_dots_symmetric_mxfp6e3m2_apple10,
                                       backend);
    run_angulars_packed<nk_mxfp6e3m2_k>(env, "angulars_packed_mxfp6e3m2_apple10", nk_dots_pack_size_mxfp6e3m2_apple10,
                                        nk_dots_pack_mxfp6e3m2_apple10, nk_angulars_packed_mxfp6e3m2_apple10, backend);
    run_angulars_symmetric<nk_mxfp6e3m2_k>(env, "angulars_symmetric_mxfp6e3m2_apple10",
                                           nk_angulars_symmetric_mxfp6e3m2_apple10, backend);
    run_euclideans_packed<nk_mxfp6e3m2_k>(env, "euclideans_packed_mxfp6e3m2_apple10",
                                          nk_dots_pack_size_mxfp6e3m2_apple10, nk_dots_pack_mxfp6e3m2_apple10,
                                          nk_euclideans_packed_mxfp6e3m2_apple10, backend);
    run_euclideans_symmetric<nk_mxfp6e3m2_k>(env, "euclideans_symmetric_mxfp6e3m2_apple10",
                                             nk_euclideans_symmetric_mxfp6e3m2_apple10, backend);
    run_dots_packed<nk_mxfp8e4m3_k>(env, "dots_packed_mxfp8e4m3_apple10", nk_dots_pack_size_mxfp8e4m3_apple10,
                                    nk_dots_pack_mxfp8e4m3_apple10, nk_dots_packed_mxfp8e4m3_apple10, backend);
    run_dots_symmetric<nk_mxfp8e4m3_k>(env, "dots_symmetric_mxfp8e4m3_apple10", nk_dots_symmetric_mxfp8e4m3_apple10,
                                       backend);
    run_angulars_packed<nk_mxfp8e4m3_k>(env, "angulars_packed_mxfp8e4m3_apple10", nk_dots_pack_size_mxfp8e4m3_apple10,
                                        nk_dots_pack_mxfp8e4m3_apple10, nk_angulars_packed_mxfp8e4m3_apple10, backend);
    run_angulars_symmetric<nk_mxfp8e4m3_k>(env, "angulars_symmetric_mxfp8e4m3_apple10",
                                           nk_angulars_symmetric_mxfp8e4m3_apple10, backend);
    run_euclideans_packed<nk_mxfp8e4m3_k>(env, "euclideans_packed_mxfp8e4m3_apple10",
                                          nk_dots_pack_size_mxfp8e4m3_apple10, nk_dots_pack_mxfp8e4m3_apple10,
                                          nk_euclideans_packed_mxfp8e4m3_apple10, backend);
    run_euclideans_symmetric<nk_mxfp8e4m3_k>(env, "euclideans_symmetric_mxfp8e4m3_apple10",
                                             nk_euclideans_symmetric_mxfp8e4m3_apple10, backend);
    run_dots_packed<nk_mxfp8e5m2_k>(env, "dots_packed_mxfp8e5m2_apple10", nk_dots_pack_size_mxfp8e5m2_apple10,
                                    nk_dots_pack_mxfp8e5m2_apple10, nk_dots_packed_mxfp8e5m2_apple10, backend);
    run_dots_symmetric<nk_mxfp8e5m2_k>(env, "dots_symmetric_mxfp8e5m2_apple10", nk_dots_symmetric_mxfp8e5m2_apple10,
                                       backend);
    run_angulars_packed<nk_mxfp8e5m2_k>(env, "angulars_packed_mxfp8e5m2_apple10", nk_dots_pack_size_mxfp8e5m2_apple10,
                                        nk_dots_pack_mxfp8e5m2_apple10, nk_angulars_packed_mxfp8e5m2_apple10, backend);
    run_angulars_symmetric<nk_mxfp8e5m2_k>(env, "angulars_symmetric_mxfp8e5m2_apple10",
                                           nk_angulars_symmetric_mxfp8e5m2_apple10, backend);
    run_euclideans_packed<nk_mxfp8e5m2_k>(env, "euclideans_packed_mxfp8e5m2_apple10",
                                          nk_dots_pack_size_mxfp8e5m2_apple10, nk_dots_pack_mxfp8e5m2_apple10,
                                          nk_euclideans_packed_mxfp8e5m2_apple10, backend);
    run_euclideans_symmetric<nk_mxfp8e5m2_k>(env, "euclideans_symmetric_mxfp8e5m2_apple10",
                                             nk_euclideans_symmetric_mxfp8e5m2_apple10, backend);
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_apple10", nk_dots_pack_size_bf16_apple10,
                               nk_dots_pack_bf16_apple10, nk_dots_packed_bf16_apple10, backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_apple10", nk_dots_pack_size_f16_apple10, nk_dots_pack_f16_apple10,
                              nk_dots_packed_f16_apple10, backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_apple10", nk_dots_pack_size_e5m2_apple10,
                               nk_dots_pack_e5m2_apple10, nk_dots_packed_e5m2_apple10, backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_apple10", nk_dots_pack_size_e4m3_apple10,
                               nk_dots_pack_e4m3_apple10, nk_dots_packed_e4m3_apple10, backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_apple10", nk_dots_pack_size_e3m2_apple10,
                               nk_dots_pack_e3m2_apple10, nk_dots_packed_e3m2_apple10, backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_apple10", nk_dots_pack_size_e2m3_apple10,
                               nk_dots_pack_e2m3_apple10, nk_dots_packed_e2m3_apple10, backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_apple10", nk_dots_pack_size_e2m1_apple10,
                               nk_dots_pack_e2m1_apple10, nk_dots_packed_e2m1_apple10, backend);
    run_dots_packed<nk_i4_k>(env, "dots_packed_i4_apple10", nk_dots_pack_size_i4_apple10, nk_dots_pack_i4_apple10,
                             nk_dots_packed_i4_apple10, backend);
    run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_apple10", nk_dots_symmetric_i4_apple10, backend);
    run_dots_packed<nk_u4_k>(env, "dots_packed_u4_apple10", nk_dots_pack_size_u4_apple10, nk_dots_pack_u4_apple10,
                             nk_dots_packed_u4_apple10, backend);
    run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_apple10", nk_dots_symmetric_u4_apple10, backend);
    run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_apple10", nk_dots_pack_size_i4_apple10,
                                 nk_dots_pack_i4_apple10, nk_angulars_packed_i4_apple10, backend);
    run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_apple10", nk_angulars_symmetric_i4_apple10, backend);
    run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_apple10", nk_dots_pack_size_i4_apple10,
                                   nk_dots_pack_i4_apple10, nk_euclideans_packed_i4_apple10, backend);
    run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_apple10", nk_euclideans_symmetric_i4_apple10,
                                      backend);
    run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_apple10", nk_dots_pack_size_u4_apple10,
                                 nk_dots_pack_u4_apple10, nk_angulars_packed_u4_apple10, backend);
    run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_apple10", nk_angulars_symmetric_u4_apple10, backend);
    run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_apple10", nk_dots_pack_size_u4_apple10,
                                   nk_dots_pack_u4_apple10, nk_euclideans_packed_u4_apple10, backend);
    run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_apple10", nk_euclideans_symmetric_u4_apple10,
                                      backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_apple10", nk_dots_pack_size_i8_apple10, nk_dots_pack_i8_apple10,
                             nk_dots_packed_i8_apple10, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_apple10", nk_dots_pack_size_u8_apple10, nk_dots_pack_u8_apple10,
                             nk_dots_packed_u8_apple10, backend);

    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_apple10", nk_dots_symmetric_bf16_apple10, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_apple10", nk_dots_symmetric_f16_apple10, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_apple10", nk_dots_symmetric_e4m3_apple10, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_apple10", nk_dots_symmetric_i8_apple10, backend);
    run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_apple10", nk_dots_pack_size_i8_apple10,
                                 nk_dots_pack_i8_apple10, nk_angulars_packed_i8_apple10, backend);
    run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_apple10", nk_angulars_symmetric_i8_apple10, backend);
    run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_apple10", nk_dots_pack_size_i8_apple10,
                                   nk_dots_pack_i8_apple10, nk_euclideans_packed_i8_apple10, backend);
    run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_apple10", nk_euclideans_symmetric_i8_apple10,
                                      backend);
    run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_apple10", nk_dots_pack_size_u8_apple10,
                                 nk_dots_pack_u8_apple10, nk_angulars_packed_u8_apple10, backend);
    run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_apple10", nk_angulars_symmetric_u8_apple10, backend);
    run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_apple10", nk_dots_pack_size_u8_apple10,
                                   nk_dots_pack_u8_apple10, nk_euclideans_packed_u8_apple10, backend);
    run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_apple10", nk_euclideans_symmetric_u8_apple10,
                                      backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_apple10", nk_dots_pack_size_f16_apple10,
                                  nk_dots_pack_f16_apple10, nk_angulars_packed_f16_apple10, backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_apple10", nk_angulars_symmetric_f16_apple10, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_apple10", nk_dots_pack_size_f16_apple10,
                                    nk_dots_pack_f16_apple10, nk_euclideans_packed_f16_apple10, backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_apple10", nk_euclideans_symmetric_f16_apple10,
                                       backend);
    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_apple10", nk_dots_pack_size_bf16_apple10,
                                   nk_dots_pack_bf16_apple10, nk_angulars_packed_bf16_apple10, backend);
    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_apple10", nk_angulars_symmetric_bf16_apple10,
                                      backend);
    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_apple10", nk_dots_pack_size_bf16_apple10,
                                     nk_dots_pack_bf16_apple10, nk_euclideans_packed_bf16_apple10, backend);
    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_apple10", nk_euclideans_symmetric_bf16_apple10,
                                        backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_apple10", nk_dots_pack_size_e4m3_apple10,
                                   nk_dots_pack_e4m3_apple10, nk_angulars_packed_e4m3_apple10, backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_apple10", nk_angulars_symmetric_e4m3_apple10,
                                      backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_apple10", nk_dots_pack_size_e4m3_apple10,
                                     nk_dots_pack_e4m3_apple10, nk_euclideans_packed_e4m3_apple10, backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_apple10", nk_euclideans_symmetric_e4m3_apple10,
                                        backend);
    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_apple10", nk_dots_pack_size_e5m2_apple10,
                                   nk_dots_pack_e5m2_apple10, nk_angulars_packed_e5m2_apple10, backend);
    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_apple10", nk_angulars_symmetric_e5m2_apple10,
                                      backend);
    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_apple10", nk_dots_pack_size_e5m2_apple10,
                                     nk_dots_pack_e5m2_apple10, nk_euclideans_packed_e5m2_apple10, backend);
    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_apple10", nk_euclideans_symmetric_e5m2_apple10,
                                        backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_apple10", nk_dots_pack_size_e3m2_apple10,
                                   nk_dots_pack_e3m2_apple10, nk_angulars_packed_e3m2_apple10, backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_apple10", nk_angulars_symmetric_e3m2_apple10,
                                      backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_apple10", nk_dots_pack_size_e3m2_apple10,
                                     nk_dots_pack_e3m2_apple10, nk_euclideans_packed_e3m2_apple10, backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_apple10", nk_euclideans_symmetric_e3m2_apple10,
                                        backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_apple10", nk_dots_pack_size_e2m3_apple10,
                                   nk_dots_pack_e2m3_apple10, nk_angulars_packed_e2m3_apple10, backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_apple10", nk_angulars_symmetric_e2m3_apple10,
                                      backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_apple10", nk_dots_pack_size_e2m3_apple10,
                                     nk_dots_pack_e2m3_apple10, nk_euclideans_packed_e2m3_apple10, backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_apple10", nk_euclideans_symmetric_e2m3_apple10,
                                        backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_apple10", nk_dots_pack_size_e2m1_apple10,
                                   nk_dots_pack_e2m1_apple10, nk_angulars_packed_e2m1_apple10, backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_apple10", nk_angulars_symmetric_e2m1_apple10,
                                      backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_apple10", nk_dots_pack_size_e2m1_apple10,
                                     nk_dots_pack_e2m1_apple10, nk_euclideans_packed_e2m1_apple10, backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_apple10", nk_euclideans_symmetric_e2m1_apple10,
                                        backend);
#endif // NUMKONG_TARGET_APPLE10
#else  // !NUMKONG_ARCH_METAL_
    return nk::status_t::missing_gpu_k;
#endif // NUMKONG_ARCH_METAL_
    return nk::status_t::success_k;
}

} // namespace ashvardanian::numkong::bench
