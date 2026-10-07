/**
 *  @file include/numkong/dots/skylake.h
 *  @author Ash Vardanian
 *  @date September 14, 2024
 *  @brief SIMD-accelerated Batched Dot Products for Skylake.
 *
 *  @sa include/numkong/dots.h
 *
 *  @section skylake_dots_instructions Relevant Instructions
 *
 *  @verbatim
 *  Intrinsic        Instruction                  SKL        ICL        Genoa
 *  _mm512_fmadd_ps  VFMADD132PS (ZMM, ZMM, ZMM)  4cy @ p05  4cy @ p05  4cy @ p01
 *  _mm512_fmadd_pd  VFMADD132PD (ZMM, ZMM, ZMM)  4cy @ p05  4cy @ p05  4cy @ p01
 *  _mm512_cvtph_ps  VCVTPH2PS (ZMM, YMM)         5cy @ p05  5cy @ p05  5cy @ p01
 *  _mm512_loadu_ps  VMOVUPS (ZMM, M512)          7cy @ p23  7cy @ p23  7cy @ p23
 *  @endverbatim
 *
 *  GEMM micro-kernels tile the K dimension to maximize FMA throughput. Skylake-X server chips with
 *  dual FMA units achieve 0.5cy throughput, enabling 32 FLOPs/cycle for f32 or 16 FLOPs/cycle for
 *  f64. FP8 types, E4M3 and E5M2, convert to f32 first, costing ~5cy of extra latency each.
 */
#ifndef NUMKONG_DOTS_SKYLAKE_H
#define NUMKONG_DOTS_SKYLAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SKYLAKE_

#include "numkong/dot/skylake.h"
#include "numkong/cast/skylake.h"   // `nk_partial_load_b32x16_skylake_`
#include "numkong/reduce/skylake.h" // `nk_reduce_moments_f32_skylake_chunked_`
#include "numkong/dots/serial.h"    // `nk_define_cross_pack_size_`

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

#pragma region Norms

NUMKONG_INLINE nk_f64_t nk_dots_reduce_sumsq_f64_skylake_(nk_f64_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f64_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_f64_skylake_contiguous_(data, count, &sum, &sumsq);
    return sumsq;
}

NUMKONG_INLINE nk_f64_t nk_dots_reduce_sumsq_f32_skylake_(nk_f32_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f64_t sum, sumsq;
    nk_reduce_moments_f32_skylake_chunked_(data, count, stride, &sum, &sumsq);
    return sumsq;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_f16_skylake_(nk_f16_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_f16_skylake_contiguous_(data, count, &sum, &sumsq);
    return sumsq;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_bf16_skylake_(nk_bf16_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_bf16_skylake_chunked_(data, count, stride, &sum, &sumsq);
    return sumsq;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e4m3_skylake_(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e4m3_skylake_chunked_(data, count, stride, &sum, &sumsq);
    return sumsq;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e5m2_skylake_(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_e5m2_skylake_contiguous_(data, count, &sum, &sumsq);
    return sumsq;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m3_skylake_(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_e2m3_skylake_contiguous_(data, count, &sum, &sumsq);
    return sumsq;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e3m2_skylake_(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_e3m2_skylake_contiguous_(data, count, &sum, &sumsq);
    return sumsq;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i8_skylake_(nk_i8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_i64_t sum;
    nk_u64_t sumsq;
    nk_unused_(stride);
    nk_reduce_moments_i8_skylake_contiguous_(data, count, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u8_skylake_(nk_u8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_u64_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_u8_skylake_contiguous_(data, count, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i4_skylake_(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_i64_t sum;
    nk_u64_t sumsq;
    nk_unused_(stride);
    nk_reduce_moments_i4_skylake_contiguous_(data, count, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u4_skylake_(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_u64_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_u4_skylake_contiguous_(data, count, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sum_u1_skylake_(nk_u1x8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_u64_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_u1_skylake_contiguous_(data, count, &sum, &sumsq);
    return (nk_u32_t)sum;
}

NUMKONG_INLINE void nk_dots_reduce_moments_i4_skylake_(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride,
                                                       nk_i32_t *sum, nk_u32_t *norm) {
    nk_i64_t row_sum;
    nk_u64_t row_sumsq;
    nk_unused_(stride);
    nk_reduce_moments_i4_skylake_contiguous_(data, count, &row_sum, &row_sumsq);
    *sum = (nk_i32_t)row_sum, *norm = (nk_u32_t)row_sumsq;
}

#pragma endregion Norms

#if NUMKONG_TARGET_SKYLAKE

/* F64 GEMM: depth_simd_dimensions=8 (8 f64s = 64 bytes = 1 cache line) */
nk_define_cross_pack_size_(dots, f64, skylake, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, f64, skylake)
nk_define_cross_pack_(dots, f64, skylake, f64, f64, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b64x8_skylake_, nk_store_b512_skylake_, nk_partial_store_b64x8_skylake_,
                      /*simd_width=*/8, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f64_skylake_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f64, skylake, f64, f64, nk_b512_vec_t, nk_dot_f64x8_state_skylake_t, nk_b256_vec_t,
                           nk_dot_f64x8_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                           nk_partial_load_b64x8_skylake_, nk_dot_f64x8_update_skylake, nk_dot_f64x8_finalize_skylake,
                           nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, skylake, f64, f64, f64, nk_b512_vec_t, nk_dot_f64x8_state_skylake_t, nk_b256_vec_t,
                        nk_dot_f64x8_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                        nk_partial_load_b64x8_skylake_, nk_load_b512_skylake_, nk_partial_load_b64x8_skylake_,
                        nk_dot_f64x8_update_skylake, nk_dot_f64x8_finalize_skylake, nk_store_b256_haswell_,
                        nk_partial_store_b64x4_skylake_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* F32 GEMM: depth_simd_dimensions=8 (8 f32s = 32 bytes = half cache line) */
nk_define_cross_pack_size_(dots, f32, skylake, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, f32, skylake)
nk_define_cross_pack_(dots, f32, skylake, f32, f32, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b32x16_skylake_, nk_store_b512_skylake_, nk_partial_store_b32x16_skylake_,
                      /*simd_width=*/16, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f32_skylake_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f32, skylake, f32, f64, nk_b256_vec_t, nk_dot_f32x8_state_skylake_t, nk_b256_vec_t,
                           nk_dot_f32x8_init_skylake, nk_cross_unscaled_, nk_load_b256_haswell_,
                           nk_partial_load_b32x8_skylake_, nk_dot_f32x8_update_skylake, nk_dot_f32x8_finalize_skylake,
                           nk_store_b256_haswell_, nk_partial_store_b64x4_skylake_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, skylake, f32, f32, f64, nk_b256_vec_t, nk_dot_f32x8_state_skylake_t, nk_b256_vec_t,
                        nk_dot_f32x8_init_skylake, nk_cross_unscaled_, nk_load_b256_haswell_,
                        nk_partial_load_b32x8_skylake_, nk_load_b256_haswell_, nk_partial_load_b32x8_skylake_,
                        nk_dot_f32x8_update_skylake, nk_dot_f32x8_finalize_skylake, nk_store_b256_haswell_,
                        nk_partial_store_b64x4_skylake_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* BF16 GEMM: depth_simd_dimensions=16 (16 bf16s = 32 bytes = half cache line), F32 accumulator */
/* BF16 GEMM: depth_simd_dimensions=32, raw bf16 storage, unpack(zero, bf16) → f32 inline */
nk_define_cross_pack_size_(dots, bf16, skylake, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, bf16, skylake)
nk_define_cross_pack_(dots, bf16, skylake, bf16, bf16, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b16x32_skylake_, nk_store_b512_skylake_, nk_partial_store_b16x32_skylake_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_bf16_skylake_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, skylake, bf16, f32, nk_b512_vec_t, nk_dot_bf16x32_state_skylake_t, nk_b128_vec_t,
                           nk_dot_bf16x32_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                           nk_partial_load_b16x32_skylake_, nk_dot_bf16x32_update_skylake,
                           nk_dot_bf16x32_finalize_skylake, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, skylake, bf16, bf16, f32, nk_b512_vec_t, nk_dot_bf16x32_state_skylake_t,
                        nk_b128_vec_t, nk_dot_bf16x32_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                        nk_partial_load_b16x32_skylake_, nk_load_b512_skylake_, nk_partial_load_b16x32_skylake_,
                        nk_dot_bf16x32_update_skylake, nk_dot_bf16x32_finalize_skylake, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=16 (16 f16s = 32 bytes = half cache line), F32 accumulator */
nk_define_cross_pack_size_(dots, f16, skylake, f16, f32, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, f16, skylake)
nk_define_cross_pack_(dots, f16, skylake, f16, f32, nk_b512_vec_t, nk_load_f16x16_to_f32x16_skylake_,
                      nk_partial_load_f16x16_to_f32x16_skylake_, nk_store_b512_skylake_,
                      nk_partial_store_b32x16_skylake_, /*simd_width=*/16,
                      /*norm_value_type=*/f32, nk_dots_reduce_sumsq_f16_skylake_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, skylake, f16, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_cross_unscaled_,
                           nk_load_f16x16_to_f32x16_skylake_, nk_partial_load_f16x16_to_f32x16_skylake_,
                           nk_dot_through_f32_update_skylake_, nk_dot_through_f32_finalize_skylake_,
                           nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, skylake, f16, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_cross_unscaled_,
                        nk_load_f16x16_to_f32x16_skylake_, nk_partial_load_f16x16_to_f32x16_skylake_,
                        nk_load_b512_skylake_, nk_partial_load_b32x16_skylake_, nk_dot_through_f32_update_skylake_,
                        nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/*  E4M3 GEMM: F16-pack with asymmetric A and B representations at compute time. Packing converts
 *  E4M3 → F16 once, at ~10 ops per 16 elements and 2 bytes stored per element. The A stream uses
 *  the Giesen E4M3 → F32 cast, at the same cost as the F32-pack path. The B loader widens F16 → F32
 *  inline, with 1 vcvtph2ps per 16 lanes. Update takes both as F32 into a plain fmadd. This saves 2
 *  bytes per element against F32-pack, while the inner loop adds one cvtph2ps per B read. Symmetric
 *  uses E4M3 → F32 for both sides, with no pack involved. */
nk_define_cross_pack_size_(dots, e4m3, skylake, e4m3, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, e4m3, skylake)
nk_define_cross_pack_(dots, e4m3, skylake, e4m3, f16, nk_b256_vec_t, nk_load_e4m3x16_to_f16x16_skylake_,
                      nk_partial_load_e4m3x16_to_f16x16_skylake_, nk_store_b256_haswell_,
                      nk_partial_store_b16x16_skylake_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_skylake_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e4m3, skylake, e4m3, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_cross_unscaled_,
                           nk_load_e4m3x16_to_f32x16_skylake_, nk_partial_load_e4m3x16_to_f32x16_skylake_,
                           nk_dot_through_f32_update_skylake_, nk_dot_through_f32_finalize_skylake_,
                           nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, skylake, e4m3, f16, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_cross_unscaled_,
                        nk_load_e4m3x16_to_f32x16_skylake_, nk_partial_load_e4m3x16_to_f32x16_skylake_,
                        nk_load_f16x16_to_f32x16_skylake_, nk_partial_load_f16x16_to_f32x16_skylake_,
                        nk_dot_through_f32_update_skylake_, nk_dot_through_f32_finalize_skylake_,
                        nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=64 (byte-level batch; widen inside the update helper) */
nk_define_cross_pack_size_(dots, e5m2, skylake, e5m2, f32, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, e5m2, skylake)
nk_define_cross_pack_(dots, e5m2, skylake, e5m2, f32, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_skylake_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e5m2, skylake, e5m2, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_cross_unscaled_, nk_load_b512_skylake_,
                           nk_partial_load_b8x64_skylake_, nk_dot_e5m2x64_update_skylake_,
                           nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, skylake, e5m2, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_cross_unscaled_, nk_load_b512_skylake_,
                        nk_partial_load_b8x64_skylake_, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                        nk_dot_e5m2x64_update_skylake_, nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

/*  E2M3 GEMM via the integer LUT path: depth_simd_dimensions = 64, as 64 e2m3s span the 64 bytes of
 *  an AVX-512 register. */
nk_define_cross_pack_size_(dots, e2m3, skylake, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, e2m3, skylake)
nk_define_cross_pack_(dots, e2m3, skylake, e2m3, e2m3, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_skylake_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, skylake, e2m3, f32, nk_b512_vec_t, nk_dot_e2m3x64_state_skylake_t, nk_b128_vec_t,
                           nk_dot_e2m3x64_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                           nk_partial_load_b8x64_skylake_, nk_dot_e2m3x64_update_skylake,
                           nk_dot_e2m3x64_finalize_skylake, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, skylake, e2m3, e2m3, f32, nk_b512_vec_t, nk_dot_e2m3x64_state_skylake_t,
                        nk_b128_vec_t, nk_dot_e2m3x64_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                        nk_partial_load_b8x64_skylake_, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                        nk_dot_e2m3x64_update_skylake, nk_dot_e2m3x64_finalize_skylake, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

/*  E2M1 GEMM via the integer LUT path: depth_simd_dimensions = 128, as 128 nibbles span the 64
 *  bytes of an AVX-512 register. */
nk_define_cross_pack_size_(dots, e2m1, skylake, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_(dots, e2m1, skylake)
nk_define_cross_pack_(dots, e2m1, skylake, e2m1x2, e2m1x2, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, e2m1, skylake, e2m1x2, f32, nk_b512_vec_t, nk_dot_e2m1x128_state_skylake_t,
                           nk_b128_vec_t, nk_dot_e2m1x128_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                           nk_partial_load_b4x128_skylake_, nk_dot_e2m1x128_update_skylake,
                           nk_dot_e2m1x128_finalize_skylake, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, e2m1, skylake, e2m1x2, e2m1x2, f32, nk_b512_vec_t, nk_dot_e2m1x128_state_skylake_t,
                        nk_b128_vec_t, nk_dot_e2m1x128_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                        nk_partial_load_b4x128_skylake_, nk_load_b512_skylake_, nk_partial_load_b4x128_skylake_,
                        nk_dot_e2m1x128_update_skylake, nk_dot_e2m1x128_finalize_skylake, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_, /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)

/* Whole blocks are decoded and scaled once per operand, then reused across each output tile. */
nk_define_cross_pack_size_(dots, nvfp4, skylake, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_(dots, nvfp4, skylake)
nk_define_cross_pack_(dots, nvfp4, skylake, e2m1x2, e2m1x2, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, nvfp4, skylake, e2m1x2, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_cross_scaled_,
                           nk_load_nvfp4x1_to_f32x16_skylake_, nk_partial_load_nvfp4x1_to_f32x16_skylake_,
                           nk_dot_through_f32_update_skylake_, nk_dot_through_f32_finalize_skylake_,
                           nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, nvfp4, skylake, e2m1x2, e2m1x2, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_skylake_, nk_cross_scaled_,
                        nk_load_nvfp4x1_to_f32x16_skylake_, nk_partial_load_nvfp4x1_to_f32x16_skylake_,
                        nk_load_nvfp4x1_to_f32x16_skylake_, nk_partial_load_nvfp4x1_to_f32x16_skylake_,
                        nk_dot_through_f32_update_skylake_, nk_dot_through_f32_finalize_skylake_,
                        nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/2)

nk_define_cross_pack_size_(dots, mxfp4, skylake, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_(dots, mxfp4, skylake)
nk_define_cross_pack_(dots, mxfp4, skylake, e2m1x2, e2m1x2, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, mxfp4, skylake, e2m1x2, f32, nk_dot_f32x32_operand_skylake_t,
                           nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                           nk_cross_scaled_, nk_load_mxfp4x1_to_f32x32_skylake_,
                           nk_partial_load_mxfp4x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                           nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, mxfp4, skylake, e2m1x2, e2m1x2, f32, nk_dot_f32x32_operand_skylake_t,
                        nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                        nk_cross_scaled_, nk_load_mxfp4x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp4x1_to_f32x32_skylake_, nk_load_mxfp4x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp4x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                        nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32,
                        /*dimensions_per_value=*/2)

nk_define_cross_pack_size_(dots, mxfp6e2m3, skylake, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, mxfp6e2m3, skylake)
nk_define_cross_pack_(dots, mxfp6e2m3, skylake, e2m3, e2m3, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, mxfp6e2m3, skylake, e2m3, f32, nk_dot_f32x32_operand_skylake_t,
                           nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                           nk_cross_scaled_, nk_load_mxfp6e2m3x1_to_f32x32_skylake_,
                           nk_partial_load_mxfp6e2m3x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                           nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, mxfp6e2m3, skylake, e2m3, e2m3, f32, nk_dot_f32x32_operand_skylake_t,
                        nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                        nk_cross_scaled_, nk_load_mxfp6e2m3x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp6e2m3x1_to_f32x32_skylake_, nk_load_mxfp6e2m3x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp6e2m3x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                        nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

nk_define_cross_pack_size_(dots, mxfp6e3m2, skylake, e3m2, e3m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, mxfp6e3m2, skylake)
nk_define_cross_pack_(dots, mxfp6e3m2, skylake, e3m2, e3m2, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e3m2_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, mxfp6e3m2, skylake, e3m2, f32, nk_dot_f32x32_operand_skylake_t,
                           nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                           nk_cross_scaled_, nk_load_mxfp6e3m2x1_to_f32x32_skylake_,
                           nk_partial_load_mxfp6e3m2x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                           nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, mxfp6e3m2, skylake, e3m2, e3m2, f32, nk_dot_f32x32_operand_skylake_t,
                        nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                        nk_cross_scaled_, nk_load_mxfp6e3m2x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp6e3m2x1_to_f32x32_skylake_, nk_load_mxfp6e3m2x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp6e3m2x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                        nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

nk_define_cross_pack_size_(dots, mxfp8e4m3, skylake, e4m3, e4m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, mxfp8e4m3, skylake)
nk_define_cross_pack_(dots, mxfp8e4m3, skylake, e4m3, e4m3, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, mxfp8e4m3, skylake, e4m3, f32, nk_dot_f32x32_operand_skylake_t,
                           nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                           nk_cross_scaled_, nk_load_mxfp8e4m3x1_to_f32x32_skylake_,
                           nk_partial_load_mxfp8e4m3x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                           nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, mxfp8e4m3, skylake, e4m3, e4m3, f32, nk_dot_f32x32_operand_skylake_t,
                        nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                        nk_cross_scaled_, nk_load_mxfp8e4m3x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp8e4m3x1_to_f32x32_skylake_, nk_load_mxfp8e4m3x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp8e4m3x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                        nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

nk_define_cross_pack_size_(dots, mxfp8e5m2, skylake, e5m2, e5m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, mxfp8e5m2, skylake)
nk_define_cross_pack_(dots, mxfp8e5m2, skylake, e5m2, e5m2, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, mxfp8e5m2, skylake, e5m2, f32, nk_dot_f32x32_operand_skylake_t,
                           nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                           nk_cross_scaled_, nk_load_mxfp8e5m2x1_to_f32x32_skylake_,
                           nk_partial_load_mxfp8e5m2x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                           nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_skylake_, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, mxfp8e5m2, skylake, e5m2, e5m2, f32, nk_dot_f32x32_operand_skylake_t,
                        nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t, nk_dot_through_f32_init_skylake_,
                        nk_cross_scaled_, nk_load_mxfp8e5m2x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp8e5m2x1_to_f32x32_skylake_, nk_load_mxfp8e5m2x1_to_f32x32_skylake_,
                        nk_partial_load_mxfp8e5m2x1_to_f32x32_skylake_, nk_dot_f32x32_update_skylake_,
                        nk_dot_through_f32_finalize_skylake_, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/*  E3M2 GEMM via the integer LUT path: depth_simd_dimensions = 64, as 64 e3m2s span the 64 bytes of
 *  an AVX-512 register. */
nk_define_cross_pack_size_(dots, e3m2, skylake, e3m2, e3m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, e3m2, skylake)
nk_define_cross_pack_(dots, e3m2, skylake, e3m2, e3m2, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e3m2_skylake_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e3m2, skylake, e3m2, f32, nk_b512_vec_t, nk_dot_e3m2x64_state_skylake_t, nk_b128_vec_t,
                           nk_dot_e3m2x64_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                           nk_partial_load_b8x64_skylake_, nk_dot_e3m2x64_update_skylake,
                           nk_dot_e3m2x64_finalize_skylake, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e3m2, skylake, e3m2, e3m2, f32, nk_b512_vec_t, nk_dot_e3m2x64_state_skylake_t,
                        nk_b128_vec_t, nk_dot_e3m2x64_init_skylake, nk_cross_unscaled_, nk_load_b512_skylake_,
                        nk_partial_load_b8x64_skylake_, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                        nk_dot_e3m2x64_update_skylake, nk_dot_e3m2x64_finalize_skylake, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

#endif // NUMKONG_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_SKYLAKE_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_DOTS_SKYLAKE_H
