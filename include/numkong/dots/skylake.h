/**
 *  @brief SIMD-accelerated Batched Dot Products for Skylake.
 *  @file include/numkong/dots/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h
 *
 *  @section skylake_dots_instructions Relevant Instructions
 *
 *      Intrinsic        Instruction                  SKL        ICL        Genoa
 *      _mm512_fmadd_ps  VFMADD132PS (ZMM, ZMM, ZMM)  4cy @ p05  4cy @ p05  4cy @ p01
 *      _mm512_fmadd_pd  VFMADD132PD (ZMM, ZMM, ZMM)  4cy @ p05  4cy @ p05  4cy @ p01
 *      _mm512_cvtph_ps  VCVTPH2PS (ZMM, YMM)         5cy @ p05  5cy @ p05  5cy @ p01
 *      _mm512_loadu_ps  VMOVUPS (ZMM, M512)          7cy @ p23  7cy @ p23  7cy @ p23
 *
 *  GEMM micro-kernels tile the K dimension to maximize FMA throughput. Skylake-X server chips with
 *  dual FMA units achieve 0.5cy throughput, enabling 32 FLOPs/cycle for f32 or 16 FLOPs/cycle for f64.
 *  FP8 types (E4M3, E5M2) convert to f32 for accumulation, adding ~5cy latency per conversion.
 */
#ifndef NK_DOTS_SKYLAKE_H
#define NK_DOTS_SKYLAKE_H

#if NK_TARGET_X8664_
#if NK_TARGET_SKYLAKE

#include "numkong/dot/skylake.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

/* F64 GEMM: depth_simd_dimensions=8 (8 f64s = 64 bytes = 1 cache line) */
nk_define_cross_pack_size_(dots, f64, skylake, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f64, skylake, f64, f64, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b64x8_skylake_, nk_store_b512_skylake_, nk_partial_store_b64x8_skylake_,
                      /*simd_width=*/8, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f64_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f64, skylake, f64, f64, nk_b512_vec_t, nk_dot_f64x8_state_skylake_t, nk_b256_vec_t,
                           nk_dot_f64x8_init_skylake, nk_load_b512_skylake_, nk_partial_load_b64x8_skylake_,
                           nk_dot_f64x8_update_skylake, nk_dot_f64x8_finalize_skylake, nk_store_b256_haswell_,
                           nk_partial_store_b64x4_skylake_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, skylake, f64, f64, f64, nk_b512_vec_t, nk_dot_f64x8_state_skylake_t, nk_b256_vec_t,
                        nk_dot_f64x8_init_skylake, nk_load_b512_skylake_, nk_partial_load_b64x8_skylake_,
                        nk_load_b512_skylake_, nk_partial_load_b64x8_skylake_, nk_dot_f64x8_update_skylake,
                        nk_dot_f64x8_finalize_skylake, nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* F32 GEMM: depth_simd_dimensions=8 (8 f32s = 32 bytes = half cache line) */
nk_define_cross_pack_size_(dots, f32, skylake, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f32, skylake, f32, f32, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b32x16_skylake_, nk_store_b512_skylake_, nk_partial_store_b32x16_skylake_,
                      /*simd_width=*/16, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f32_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f32, skylake, f32, f64, nk_b256_vec_t, nk_dot_f32x8_state_skylake_t, nk_b256_vec_t,
                           nk_dot_f32x8_init_skylake, nk_load_b256_haswell_, nk_partial_load_b32x8_skylake_,
                           nk_dot_f32x8_update_skylake, nk_dot_f32x8_finalize_skylake, nk_store_b256_haswell_,
                           nk_partial_store_b64x4_skylake_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, skylake, f32, f32, f64, nk_b256_vec_t, nk_dot_f32x8_state_skylake_t, nk_b256_vec_t,
                        nk_dot_f32x8_init_skylake, nk_load_b256_haswell_, nk_partial_load_b32x8_skylake_,
                        nk_load_b256_haswell_, nk_partial_load_b32x8_skylake_, nk_dot_f32x8_update_skylake,
                        nk_dot_f32x8_finalize_skylake, nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* BF16 GEMM: depth_simd_dimensions=16 (16 bf16s = 32 bytes = half cache line), F32 accumulator */
/* BF16 GEMM: depth_simd_dimensions=32, raw bf16 storage, unpack(zero, bf16) → f32 inline */
nk_define_cross_pack_size_(dots, bf16, skylake, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, bf16, skylake, bf16, bf16, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b16x32_skylake_, nk_store_b512_skylake_, nk_partial_store_b16x32_skylake_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_bf16_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, skylake, bf16, f32, nk_b512_vec_t, nk_dot_bf16x32_state_skylake_t, nk_b128_vec_t,
                           nk_dot_bf16x32_init_skylake, nk_load_b512_skylake_, nk_partial_load_b16x32_skylake_,
                           nk_dot_bf16x32_update_skylake, nk_dot_bf16x32_finalize_skylake, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, skylake, bf16, bf16, f32, nk_b512_vec_t, nk_dot_bf16x32_state_skylake_t,
                        nk_b128_vec_t, nk_dot_bf16x32_init_skylake, nk_load_b512_skylake_,
                        nk_partial_load_b16x32_skylake_, nk_load_b512_skylake_, nk_partial_load_b16x32_skylake_,
                        nk_dot_bf16x32_update_skylake, nk_dot_bf16x32_finalize_skylake, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=16 (16 f16s = 32 bytes = half cache line), F32 accumulator */
nk_define_cross_pack_size_(dots, f16, skylake, f16, f32, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, skylake, f16, f32, nk_b512_vec_t, nk_load_f16x16_to_f32x16_skylake_,
                      nk_partial_load_f16x16_to_f32x16_skylake_, nk_store_b512_skylake_,
                      nk_partial_store_b32x16_skylake_, /*simd_width=*/16, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_f16_, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, skylake, f16, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_load_f16x16_to_f32x16_skylake_,
                           nk_partial_load_f16x16_to_f32x16_skylake_, nk_dot_through_f32_update_skylake_,
                           nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, skylake, f16, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_load_f16x16_to_f32x16_skylake_,
                        nk_partial_load_f16x16_to_f32x16_skylake_, nk_load_b512_skylake_,
                        nk_partial_load_b32x16_skylake_, nk_dot_through_f32_update_skylake_,
                        nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E4M3 GEMM: F16-pack with asymmetric A/B representations at compute time. Pack converts
 * E4M3 → F16 once (~10 ops/16 elements, 2 bytes/elt stored). A-stream uses the Giesen E4M3→F32
 * cast (identical cost to F32-pack path). B-loader widens F16 → F32 inline (1 vcvtph2ps per 16
 * lanes). Update takes both as F32 → plain fmadd. Saves 2 bytes/elt vs F32-pack; inner loop
 * adds one cvtph2ps per B-read. Symmetric uses E4M3→F32 for both sides (no pack involved). */
nk_define_cross_pack_size_(dots, e4m3, skylake, e4m3, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e4m3, skylake, e4m3, f16, nk_b256_vec_t, nk_load_e4m3x16_to_f16x16_skylake_,
                      nk_partial_load_e4m3x16_to_f16x16_skylake_, nk_store_b256_haswell_,
                      nk_partial_store_b16x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e4m3, skylake, e4m3, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_load_e4m3x16_to_f32x16_skylake_,
                           nk_partial_load_e4m3x16_to_f32x16_skylake_, nk_dot_through_f32_update_skylake_,
                           nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, skylake, e4m3, f16, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_load_e4m3x16_to_f32x16_skylake_,
                        nk_partial_load_e4m3x16_to_f32x16_skylake_, nk_load_f16x16_to_f32x16_skylake_,
                        nk_partial_load_f16x16_to_f32x16_skylake_, nk_dot_through_f32_update_skylake_,
                        nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=64 (byte-level batch; widen inside the update helper) */
nk_define_cross_pack_size_(dots, e5m2, skylake, e5m2, f32, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e5m2, skylake, e5m2, f32, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e5m2, skylake, e5m2, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_load_b512_skylake_,
                           nk_partial_load_b8x64_skylake_, nk_dot_e5m2x64_update_skylake_,
                           nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, skylake, e5m2, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_load_b512_skylake_,
                        nk_partial_load_b8x64_skylake_, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                        nk_dot_e5m2x64_update_skylake_, nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

/* E2M3 GEMM: integer LUT path, depth_simd_dimensions=64 (64 e2m3s = 64 bytes = AVX-512 register width) */
nk_define_cross_pack_size_(dots, e2m3, skylake, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e2m3, skylake, e2m3, e2m3, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, skylake, e2m3, f32, nk_b512_vec_t, nk_dot_e2m3x64_state_skylake_t, nk_b128_vec_t,
                           nk_dot_e2m3x64_init_skylake, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                           nk_dot_e2m3x64_update_skylake, nk_dot_e2m3x64_finalize_skylake, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, skylake, e2m3, e2m3, f32, nk_b512_vec_t, nk_dot_e2m3x64_state_skylake_t,
                        nk_b128_vec_t, nk_dot_e2m3x64_init_skylake, nk_load_b512_skylake_,
                        nk_partial_load_b8x64_skylake_, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                        nk_dot_e2m3x64_update_skylake, nk_dot_e2m3x64_finalize_skylake, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

/* E3M2 GEMM: integer LUT path, depth_simd_dimensions=64 (64 e3m2s = 64 bytes = AVX-512 register width) */
nk_define_cross_pack_size_(dots, e3m2, skylake, e3m2, e3m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e3m2, skylake, e3m2, e3m2, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e3m2_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e3m2, skylake, e3m2, f32, nk_b512_vec_t, nk_dot_e3m2x64_state_skylake_t, nk_b128_vec_t,
                           nk_dot_e3m2x64_init_skylake, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                           nk_dot_e3m2x64_update_skylake, nk_dot_e3m2x64_finalize_skylake, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e3m2, skylake, e3m2, e3m2, f32, nk_b512_vec_t, nk_dot_e3m2x64_state_skylake_t,
                        nk_b128_vec_t, nk_dot_e3m2x64_init_skylake, nk_load_b512_skylake_,
                        nk_partial_load_b8x64_skylake_, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                        nk_dot_e3m2x64_update_skylake, nk_dot_e3m2x64_finalize_skylake, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X8664_
#endif // NK_DOTS_SKYLAKE_H
