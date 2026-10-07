/**
 *  @file include/numkong/dots/icelake.h
 *  @author Ash Vardanian
 *  @date September 14, 2024
 *  @brief SIMD-accelerated Batched Dot Products for Ice Lake.
 *
 *  @sa include/numkong/dots.h
 *
 *  @section ice_dots_instructions Relevant Instructions
 *
 *  @verbatim
 *  Intrinsic             Instruction               Icelake    Genoa
 *  _mm512_dpbusd_epi32   VPDPBUSD (ZMM, ZMM, ZMM)  5cy @ p0   4cy @ p01
 *  _mm512_dpwssd_epi32   VPDPWSSD (ZMM, ZMM, ZMM)  5cy @ p0   4cy @ p01
 *  _mm512_cvtepi8_epi32  VPMOVSXBD (ZMM, XMM)      3cy @ p5   3cy @ p12
 *  _mm512_loadu_si512    VMOVDQU64 (ZMM, M512)     7cy @ p23  7cy @ p23
 *  @endverbatim
 *
 *  Ice Lake's VNNI instructions accelerate int8 GEMM by computing 4-element dot products per lane.
 *  VPDPBUSD/VPDPWSSD bottleneck on port 0, limiting throughput to 1/cy. AMD Genoa achieves 0.5/cy
 *  via dual-issue on ports 0-1, making it significantly faster for quantized inference workloads.
 */
#ifndef NUMKONG_DOTS_ICELAKE_H
#define NUMKONG_DOTS_ICELAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_ICELAKE_

#include "numkong/dot/icelake.h"
#include "numkong/cast/haswell.h"   // `nk_load_b128_haswell_`
#include "numkong/cast/skylake.h"   // `nk_load_b512_skylake_`
#include "numkong/dots/serial.h"    // `nk_define_cross_pack_size_`
#include "numkong/dots/skylake.h"   // `nk_dots_reduce_sumsq_u4_skylake_`
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`
#include "numkong/reduce/icelake.h" // `nk_reduce_moments_i8_icelake_contiguous_`

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                                        \
    __attribute__((                                                                                                  \
        target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512vbmi,avx512vpopcntdq,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512vbmi", \
                   "avx512vpopcntdq", "f16c", "fma", "bmi", "bmi2")
#endif

#pragma region Norms

NUMKONG_INLINE void nk_dots_reduce_moments_i8_icelake_(nk_i8_t const *data, nk_size_t count, nk_size_t stride,
                                                       nk_i32_t *sum, nk_u32_t *norm) {
    nk_i64_t row_sum;
    nk_u64_t row_sumsq;
    nk_unused_(stride);
    nk_reduce_moments_i8_icelake_contiguous_(data, count, &row_sum, &row_sumsq);
    *sum = (nk_i32_t)row_sum, *norm = (nk_u32_t)row_sumsq;
}

NUMKONG_INLINE void nk_dots_reduce_moments_u8_icelake_(nk_u8_t const *data, nk_size_t count, nk_size_t stride,
                                                       nk_u32_t *sum, nk_u32_t *norm) {
    nk_u64_t row_sum, row_sumsq;
    nk_unused_(stride);
    nk_reduce_moments_u8_icelake_contiguous_(data, count, &row_sum, &row_sumsq);
    *sum = (nk_u32_t)row_sum, *norm = (nk_u32_t)row_sumsq;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i8_icelake_(nk_i8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_i32_t sum;
    nk_u32_t norm;
    nk_dots_reduce_moments_i8_icelake_(data, count, stride, &sum, &norm);
    return norm;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u8_icelake_(nk_u8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_u32_t sum, norm;
    nk_dots_reduce_moments_u8_icelake_(data, count, stride, &sum, &norm);
    return norm;
}

#pragma endregion Norms

/* I8 GEMM: depth_simd_dimensions=64 — compensated (B sums precomputed in pack) */
nk_define_cross_packed_shape_(dots, i8, icelake)
nk_define_cross_compensated_pack_size_(dots, i8, icelake, i8, i8,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_compensated_pack_(dots, i8, icelake, i8, i8, nk_b512_vec_t, nk_load_b512_skylake_,
                                  nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_,
                                  nk_partial_store_b8x64_skylake_, /*simd_width=*/64, /*sum_value_type=*/i32,
                                  /*norm_value_type=*/u32, nk_dots_reduce_moments_i8_icelake_,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_compensated_symmetric_(dots, i8, icelake, i8, i32,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b512_vec_t,
                                       nk_dot_i8x64_state_icelake_t, nk_b128_vec_t, nk_dot_i8x64_init_icelake,
                                       nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                                       nk_dot_i8x64_update_icelake, nk_dot_i8x64_finalize_icelake,
                                       nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, nk_load_b128_haswell_,
                                       nk_partial_load_b32x4_skylake_, nk_sum_i8x64_state_icelake_t,
                                       nk_sum_i8x64_init_icelake, nk_sum_i8x64_update_icelake,
                                       nk_sum_i8x64_finalize_icelake,
                                       /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_compensated_packed_(dots, i8, icelake, i8, i8, i32,
                                    /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b512_vec_t,
                                    nk_dot_i8x64_state_icelake_t, nk_b128_vec_t, nk_dot_i8x64_init_icelake,
                                    nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_, nk_load_b512_skylake_,
                                    nk_partial_load_b8x64_skylake_, nk_dot_i8x64_update_icelake,
                                    nk_dot_i8x64_finalize_icelake, nk_store_b128_haswell_,
                                    nk_partial_store_b32x4_skylake_, nk_load_b128_haswell_,
                                    nk_partial_load_b32x4_skylake_, nk_dots_reduce_sum_i8_stub_,
                                    /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=64 — compensated (operand swap, B sums precomputed) */
nk_define_cross_packed_shape_(dots, u8, icelake)
nk_define_cross_compensated_pack_size_(dots, u8, icelake, u8, u8,
                                       /*sum_value_type=*/u32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_compensated_pack_(dots, u8, icelake, u8, u8, nk_b512_vec_t, nk_load_b512_skylake_,
                                  nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_,
                                  nk_partial_store_b8x64_skylake_, /*simd_width=*/64, /*sum_value_type=*/u32,
                                  /*norm_value_type=*/u32, nk_dots_reduce_moments_u8_icelake_,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_compensated_symmetric_(dots, u8, icelake, u8, u32,
                                       /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_b512_vec_t,
                                       nk_dot_u8x64_state_icelake_t, nk_b128_vec_t, nk_dot_u8x64_init_icelake,
                                       nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                                       nk_dot_u8x64_update_icelake, nk_dot_u8x64_finalize_icelake,
                                       nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, nk_load_b128_haswell_,
                                       nk_partial_load_b32x4_skylake_, nk_sum_u8x64_state_icelake_t,
                                       nk_sum_u8x64_init_icelake, nk_sum_u8x64_update_icelake,
                                       nk_sum_u8x64_finalize_icelake,
                                       /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_compensated_packed_(dots, u8, icelake, u8, u8, u32,
                                    /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_b512_vec_t,
                                    nk_dot_u8x64_state_icelake_t, nk_b128_vec_t, nk_dot_u8x64_init_icelake,
                                    nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_, nk_load_b512_skylake_,
                                    nk_partial_load_b8x64_skylake_, nk_dot_u8x64_update_icelake,
                                    nk_dot_u8x64_finalize_icelake, nk_store_b128_haswell_,
                                    nk_partial_store_b32x4_skylake_, nk_load_b128_haswell_,
                                    nk_partial_load_b32x4_skylake_, nk_dots_reduce_sum_u8_stub_,
                                    /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

/* I4 GEMM: depth_simd_dimensions=128 — compensated (A+B sums precomputed) */
nk_define_cross_packed_shape_(dots, i4, icelake)
nk_define_cross_compensated_pack_size_(dots, i4, icelake, i4x2, i4x2,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_compensated_pack_(dots, i4, icelake, i4x2, i4x2, nk_b512_vec_t, nk_load_b512_skylake_,
                                  nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_,
                                  nk_partial_store_b8x64_skylake_, /*simd_width=*/64, /*sum_value_type=*/i32,
                                  /*norm_value_type=*/u32, nk_dots_reduce_moments_i4_skylake_,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2)
nk_define_cross_compensated_symmetric_(dots, i4, icelake, i4x2, i32,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b512_vec_t,
                                       nk_dot_i4x128_state_icelake_t, nk_b128_vec_t, nk_dot_i4x128_init_icelake,
                                       nk_load_b512_skylake_, nk_partial_load_b4x128_skylake_,
                                       nk_dot_i4x128_update_icelake, nk_dot_i4x128_finalize_icelake,
                                       nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_, nk_load_b128_haswell_,
                                       nk_partial_load_b32x4_skylake_, nk_sum_i4x128_state_icelake_t,
                                       nk_sum_i4x128_init_icelake, nk_sum_i4x128_update_icelake,
                                       nk_sum_i4x128_finalize_icelake,
                                       /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_compensated_packed_(dots, i4, icelake, i4x2, i4x2, i32,
                                    /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b512_vec_t,
                                    nk_dot_i4x128_state_icelake_t, nk_b128_vec_t, nk_dot_i4x128_init_icelake,
                                    nk_load_b512_skylake_, nk_partial_load_b4x128_skylake_, nk_load_b512_skylake_,
                                    nk_partial_load_b4x128_skylake_, nk_dot_i4x128_update_icelake,
                                    nk_dot_i4x128_finalize_icelake, nk_store_b128_haswell_,
                                    nk_partial_store_b32x4_skylake_, nk_load_b128_haswell_,
                                    nk_partial_load_b32x4_skylake_, nk_dots_reduce_sum_i4_,
                                    /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)

/* U4 GEMM: depth_simd_dimensions=128 (128 nibbles = 64 bytes = full cache line) */
nk_define_cross_pack_size_(dots, u4, icelake, u4x2, u4x2, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_(dots, u4, icelake)
nk_define_cross_pack_(dots, u4, icelake, u4x2, u4x2, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u4_skylake_,
                      /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)

nk_define_cross_symmetric_(dots, u4, icelake, u4x2, u32, nk_b512_vec_t, nk_dot_u4x128_state_icelake_t, nk_b128_vec_t,
                           nk_dot_u4x128_init_icelake, nk_cross_unscaled_, nk_load_b512_skylake_,
                           nk_partial_load_b4x128_skylake_, nk_dot_u4x128_update_icelake,
                           nk_dot_u4x128_finalize_icelake, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, u4, icelake, u4x2, u4x2, u32, nk_b512_vec_t, nk_dot_u4x128_state_icelake_t, nk_b128_vec_t,
                        nk_dot_u4x128_init_icelake, nk_cross_unscaled_, nk_load_b512_skylake_,
                        nk_partial_load_b4x128_skylake_, nk_load_b512_skylake_, nk_partial_load_b4x128_skylake_,
                        nk_dot_u4x128_update_icelake, nk_dot_u4x128_finalize_icelake, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)

/* U1 GEMM: depth_simd_dimensions=512 (512 bits = 64 bytes = full cache line) */
nk_define_cross_pack_size_(dots, u1, icelake, u1x8, u1x8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/512,
                           /*dimensions_per_value=*/8)
nk_define_cross_packed_shape_(dots, u1, icelake)
nk_define_cross_pack_(dots, u1, icelake, u1x8, u1x8, nk_b512_vec_t, nk_load_b512_skylake_,
                      nk_partial_load_b8x64_skylake_, nk_store_b512_skylake_, nk_partial_store_b8x64_skylake_,
                      /*simd_width=*/64, /*norm_value_type=*/u32, nk_dots_reduce_sum_u1_skylake_,
                      /*depth_simd_dimensions=*/512, /*dimensions_per_value=*/8)
nk_define_cross_symmetric_(dots, u1, icelake, u1x8, u32, nk_b512_vec_t, nk_dot_u1x512_state_icelake_t, nk_b128_vec_t,
                           nk_dot_u1x512_init_icelake, nk_cross_unscaled_, nk_load_b512_skylake_,
                           nk_partial_load_b1x512_skylake_, nk_dot_u1x512_update_icelake,
                           nk_dot_u1x512_finalize_icelake, nk_store_b128_haswell_, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/512, /*dimensions_per_value=*/8)
nk_define_cross_packed_(dots, u1, icelake, u1x8, u1x8, u32, nk_b512_vec_t, nk_dot_u1x512_state_icelake_t, nk_b128_vec_t,
                        nk_dot_u1x512_init_icelake, nk_cross_unscaled_, nk_load_b512_skylake_,
                        nk_partial_load_b1x512_skylake_, nk_load_b512_skylake_, nk_partial_load_b1x512_skylake_,
                        nk_dot_u1x512_update_icelake, nk_dot_u1x512_finalize_icelake, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/512, /*dimensions_per_value=*/8)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_ICELAKE

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_ICELAKE_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_DOTS_ICELAKE_H
