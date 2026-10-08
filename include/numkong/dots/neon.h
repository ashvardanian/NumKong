/**
 *  @file include/numkong/dots/neon.h
 *  @author Ash Vardanian
 *  @date September 14, 2024
 *  @brief SIMD-accelerated Batched Dot Products for NEON.
 *
 *  @sa include/numkong/dots.h
 */
#ifndef NUMKONG_DOTS_NEON_H
#define NUMKONG_DOTS_NEON_H

/*  Block-scaled packs hold B lifted to integers or widened to F16, in the layout the A panels
 *  share, and scale rows of F32 scales, rebased per column and times the lift, followed by the raw
 *  codes. The writers and exact-path readers serve every Arm tier that shares these packs. */

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/dots/serial.h"
#include "numkong/dot/neon.h"

NUMKONG_INLINE void nk_pack_nvfp4_scales_neon_(nk_u8_t const *source, nk_size_t blocks, nk_u8_t *destination,
                                               nk_size_t stride, nk_i32_t base) {
    nk_unused_(base);
    nk_cross_pack_scales_ue4m3_f32_(source, blocks, destination, stride, 0.5f, 1);
}

NUMKONG_INLINE void nk_pack_mxfp4_scales_neon_(nk_u8_t const *source, nk_size_t blocks, nk_u8_t *destination,
                                               nk_size_t stride, nk_i32_t base) {
    nk_cross_pack_scales_ue8m0_f32_(source, blocks, destination, stride, base, 0.5f, 2);
}

NUMKONG_INLINE void nk_pack_mxfp6e2m3_scales_neon_(nk_u8_t const *source, nk_size_t blocks, nk_u8_t *destination,
                                                   nk_size_t stride, nk_i32_t base) {
    nk_cross_pack_scales_ue8m0_f32_(source, blocks, destination, stride, base, 0.125f, 2);
}

NUMKONG_INLINE void nk_pack_mxfp6e3m2_scales_neon_(nk_u8_t const *source, nk_size_t blocks, nk_u8_t *destination,
                                                   nk_size_t stride, nk_i32_t base) {
    nk_cross_pack_scales_ue8m0_f32_(source, blocks, destination, stride, base, 0.0625f, 1);
}

NUMKONG_INLINE void nk_pack_mxfp8_scales_neon_(nk_u8_t const *source, nk_size_t blocks, nk_u8_t *destination,
                                               nk_size_t stride, nk_i32_t base) {
    nk_cross_pack_scales_ue8m0_f32_(source, blocks, destination, stride, base, 1.0f, 1);
}

/** Element @p index of an E2M1 pack row, read back from where @c nk_e2m1x64_to_i8x64_neon_ puts it:
 *  odd elements in the first two registers, even ones in the last two. */
NUMKONG_INLINE nk_f32_t nk_e2m1_lanes_load_f32_neon_(nk_u8_t const *row, nk_size_t index) NUMKONG_STREAMABLE_ {
    nk_size_t const within = index % 64, byte = within % 32 / 2, word = byte / 4;
    nk_size_t const lane_register = (within & 1 ? 0 : 2) + (word & 1), slot = within / 32 * 2 + word / 2;
    return 0.5f * ((nk_i8_t const *)row)[index / 64 * 64 + lane_register * 16 + slot * 4 + byte % 4];
}

/** Element @p index of an E2M3 pack row, read back from where @c nk_e2m3x64_to_i8x64_neon_ puts
 *  it. */
NUMKONG_INLINE nk_f32_t nk_e2m3_lanes_load_f32_neon_(nk_u8_t const *row, nk_size_t index) NUMKONG_STREAMABLE_ {
    nk_size_t const within = index % 64, quarter = within / 16, word = within % 16 / 4;
    nk_size_t const lane_register = (quarter & 1) * 2 + (word & 1), slot = (quarter >> 1) * 2 + (word >> 1);
    return 0.125f * ((nk_i8_t const *)row)[index / 64 * 64 + lane_register * 16 + slot * 4 + within % 4];
}

/** Element @p index of an E3M2 pack row, lifted by sixteen to I16 in element order. */
NUMKONG_INLINE nk_f32_t nk_e3m2_lifted_load_f32_neon_(nk_u8_t const *row, nk_size_t index) NUMKONG_STREAMABLE_ {
    return 0.0625f * ((nk_i16_t const *)row)[index];
}

/** Element @p index of an FP8 pack row, widened to F16 in element order. */
NUMKONG_INLINE nk_f32_t nk_f16_load_f32_neon_(nk_u8_t const *row, nk_size_t index) NUMKONG_STREAMABLE_ {
    nk_f32_t value;
    nk_f16_to_f32_((nk_f16_t const *)row + index, &value);
    return value;
}

/*  The exact paths over these packs, which every Arm tier sharing them passes to its kernels */
nk_define_cross_scaled_exact_(nvfp4, neon, nk_e2m1_load_f32_serial_, nk_e2m1_lanes_load_f32_neon_,
                              nk_ue4m3_split_serial_, /*block_size=*/16)
nk_define_cross_scaled_exact_(mxfp4, neon, nk_e2m1_load_f32_serial_, nk_e2m1_lanes_load_f32_neon_,
                              nk_ue8m0_split_serial_, /*block_size=*/32)
nk_define_cross_scaled_exact_(mxfp6e2m3, neon, nk_e2m3_load_f32_serial_, nk_e2m3_lanes_load_f32_neon_,
                              nk_ue8m0_split_serial_, /*block_size=*/32)
nk_define_cross_scaled_exact_(mxfp6e3m2, neon, nk_e3m2_load_f32_serial_, nk_e3m2_lifted_load_f32_neon_,
                              nk_ue8m0_split_serial_, /*block_size=*/32)
nk_define_cross_scaled_exact_(mxfp8e4m3, neon, nk_e4m3_load_f32_serial_, nk_f16_load_f32_neon_, nk_ue8m0_split_serial_,
                              /*block_size=*/32)
nk_define_cross_scaled_exact_(mxfp8e5m2, neon, nk_e5m2_load_f32_serial_, nk_f16_load_f32_neon_, nk_ue8m0_split_serial_,
                              /*block_size=*/32)

#if NUMKONG_TARGET_NEON
#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/*  F32 GEMM: depth_simd_dimensions = 2, as 2 f32s span an 8-byte, 64-bit input for f64 upcast
 *  accumulation. */
nk_define_cross_pack_size_(dots, f32, neon, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, f32, neon)
nk_define_cross_pack_(dots, f32, neon, f32, f32, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, /*simd_width=*/4,
                      /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f32_, /*depth_simd_dimensions=*/2,
                      /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, f32, neon, f32, f64, nk_b64_vec_t, nk_dot_f32x2_state_neon_t, nk_b256_vec_t,
                           nk_dot_f32x2_init_neon, nk_load_b64_neon_, nk_partial_load_b32x2_serial_,
                           nk_dot_f32x2_update_neon, nk_dot_f32x2_finalize_neon, nk_store_b256_neon_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, neon, f32, f32, f64, nk_b64_vec_t, nk_dot_f32x2_state_neon_t, nk_b256_vec_t,
                        nk_dot_f32x2_init_neon, nk_load_b64_neon_, nk_partial_load_b32x2_serial_, nk_load_b64_neon_,
                        nk_partial_load_b32x2_serial_, nk_dot_f32x2_update_neon, nk_dot_f32x2_finalize_neon,
                        nk_store_b256_neon_, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

/* U1 GEMM: depth_simd_dimensions=128 (128 bits = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, u1, neon, u1x8, u1x8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/8, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, u1, neon)
nk_define_cross_pack_(dots, u1, neon, u1x8, u1x8, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_neon_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sum_u1_, /*depth_simd_dimensions=*/128,
                      /*dimensions_per_value=*/8, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, u1, neon, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_neon_t, nk_b128_vec_t,
                           nk_dot_u1x128_init_neon, nk_load_b128_neon_, nk_partial_load_b1x128_serial_,
                           nk_dot_u1x128_update_neon, nk_dot_u1x128_finalize_neon, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_packed_(dots, u1, neon, u1x8, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_neon_t, nk_b128_vec_t,
                        nk_dot_u1x128_init_neon, nk_load_b128_neon_, nk_partial_load_b1x128_serial_, nk_load_b128_neon_,
                        nk_partial_load_b1x128_serial_, nk_dot_u1x128_update_neon, nk_dot_u1x128_finalize_neon,
                        nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)

/* BF16 GEMM: depth_simd_dimensions=8 (8 bf16s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, bf16, neon, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, bf16, neon)
nk_define_cross_pack_(dots, bf16, neon, bf16, bf16, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                      nk_store_b128_neon_, nk_partial_store_b16x8_serial_,
                      /*simd_width=*/8, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_bf16_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, bf16, neon, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_neon_t, nk_b128_vec_t,
                           nk_dot_bf16x8_init_neon, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                           nk_dot_bf16x8_update_neon, nk_dot_bf16x8_finalize_neon, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, neon, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_neon_t, nk_b128_vec_t,
                        nk_dot_bf16x8_init_neon, nk_load_b128_neon_, nk_partial_load_b16x8_serial_, nk_load_b128_neon_,
                        nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_neon, nk_dot_bf16x8_finalize_neon,
                        nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8,
                        /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=8 (8 f16s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, f16, neon, f16, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, f16, neon)
nk_define_cross_pack_(dots, f16, neon, f16, f16, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                      nk_store_b128_neon_, nk_partial_store_b16x8_serial_, /*simd_width=*/8,
                      /*norm_value_type=*/f32, nk_dots_reduce_sumsq_f16_, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, f16, neon, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neon_t, nk_b128_vec_t,
                           nk_dot_f16x8_init_neon, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                           nk_dot_f16x8_update_neon, nk_dot_f16x8_finalize_neon, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, neon, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neon_t, nk_b128_vec_t,
                        nk_dot_f16x8_init_neon, nk_load_b128_neon_, nk_partial_load_b16x8_serial_, nk_load_b128_neon_,
                        nk_partial_load_b16x8_serial_, nk_dot_f16x8_update_neon, nk_dot_f16x8_finalize_neon,
                        nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8,
                        /*dimensions_per_value=*/1)

/* F64 GEMM: depth_simd_dimensions=2 (2 f64s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, f64, neon, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, f64, neon)
nk_define_cross_pack_(dots, f64, neon, f64, f64, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b64x2_serial_,
                      nk_store_b128_neon_, nk_partial_store_b64x2_serial_, /*simd_width=*/2,
                      /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f64_, /*depth_simd_dimensions=*/2,
                      /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, f64, neon, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_neon_t, nk_b256_vec_t,
                           nk_dot_f64x2_init_neon, nk_load_b128_neon_, nk_partial_load_b64x2_serial_,
                           nk_dot_f64x2_update_neon, nk_dot_f64x2_finalize_neon, nk_store_b256_neon_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, neon, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_neon_t, nk_b256_vec_t,
                        nk_dot_f64x2_init_neon, nk_load_b128_neon_, nk_partial_load_b64x2_serial_, nk_load_b128_neon_,
                        nk_partial_load_b64x2_serial_, nk_dot_f64x2_update_neon, nk_dot_f64x2_finalize_neon,
                        nk_store_b256_neon_, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

nk_define_cross_pack_size_(dots, nvfp4, neon, e2m1x2, u16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/5)
nk_define_cross_packed_shape_(dots, nvfp4, neon)
nk_define_cross_pack_(dots, nvfp4, neon, e2m1x2, u16, nk_b512_vec_t, nk_e2m1x64_to_i8x64_neon_,
                      nk_partial_e2m1x64_to_i8x64_neon_, nk_store_b512_neon_, nk_partial_store_i8x64_neon_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, nk_pack_nvfp4_scales_neon_,
                      /*scale_bytes=*/5)
nk_define_cross_scaled_packed_(dots, nvfp4, neon, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_dot_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_nvfp4_neon_,
                               nk_dot_f32_from_wide_, /*normalized=*/0, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(dots, nvfp4, neon, u16, nk_dot_f32_from_wide_, /*normalized=*/0,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_pack_size_(dots, mxfp4, neon, e2m1x2, u16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/9)
nk_define_cross_packed_shape_(dots, mxfp4, neon)
nk_define_cross_pack_(dots, mxfp4, neon, e2m1x2, u16, nk_b512_vec_t, nk_e2m1x64_to_i8x64_neon_,
                      nk_partial_e2m1x64_to_i8x64_neon_, nk_store_b512_neon_, nk_partial_store_i8x64_neon_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, nk_pack_mxfp4_scales_neon_,
                      /*scale_bytes=*/9)
nk_define_cross_scaled_packed_(dots, mxfp4, neon, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_dot_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp4_neon_,
                               nk_dot_f32_from_wide_, /*normalized=*/0, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(dots, mxfp4, neon, u16, nk_dot_f32_from_wide_, /*normalized=*/0,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_pack_size_(dots, mxfp6e2m3, neon, e2m3, i8, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/9)
nk_define_cross_packed_shape_(dots, mxfp6e2m3, neon)
nk_define_cross_pack_(dots, mxfp6e2m3, neon, e2m3, i8, nk_b512_vec_t, nk_e2m3x64_to_i8x64_neon_,
                      nk_partial_e2m3x64_to_i8x64_neon_, nk_store_b512_neon_, nk_partial_store_i8x64_neon_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1, nk_pack_mxfp6e2m3_scales_neon_,
                      /*scale_bytes=*/9)
nk_define_cross_scaled_packed_(dots, mxfp6e2m3, neon, i8, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_dot_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp6e2m3_neon_,
                               nk_dot_f32_from_wide_, /*normalized=*/0, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(dots, mxfp6e2m3, neon, i8, nk_dot_f32_from_wide_, /*normalized=*/0,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_pack_size_(dots, mxfp6e3m2, neon, e3m2, i16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/5)
nk_define_cross_packed_shape_(dots, mxfp6e3m2, neon)
nk_define_cross_pack_(dots, mxfp6e3m2, neon, e3m2, i16, nk_b256_vec_t, nk_load_e3m2x16_to_i16x16_neon_,
                      nk_partial_load_e3m2x16_to_i16x16_neon_, nk_store_b256_neon_, nk_partial_store_b16x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e3m2_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_pack_mxfp6e3m2_scales_neon_,
                      /*scale_bytes=*/5)
nk_define_cross_scaled_packed_(dots, mxfp6e3m2, neon, i16, nk_dot_scaled_i16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i16x32_neon_,
                               nk_dot_scaled_i16x32_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_dot_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp6e3m2_neon_,
                               nk_dot_f32_from_wide_, /*normalized=*/0, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(dots, mxfp6e3m2, neon, i16, nk_dot_f32_from_wide_, /*normalized=*/0,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_pack_size_(dots, mxfp8e4m3, neon, e4m3, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/5)
nk_define_cross_packed_shape_(dots, mxfp8e4m3, neon)
nk_define_cross_pack_(dots, mxfp8e4m3, neon, e4m3, f16, nk_b256_vec_t, nk_load_e4m3x16_to_f16x16_neon_,
                      nk_partial_load_e4m3x16_to_f16x16_neon_, nk_store_b256_neon_, nk_partial_store_b16x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_pack_mxfp8_scales_neon_,
                      /*scale_bytes=*/5)
nk_define_cross_scaled_packed_(dots, mxfp8e4m3, neon, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_dot_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e4m3_neon_,
                               nk_dot_f32_from_wide_, /*normalized=*/0, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(dots, mxfp8e4m3, neon, f16, nk_dot_f32_from_wide_, /*normalized=*/0,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_pack_size_(dots, mxfp8e5m2, neon, e5m2, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/5)
nk_define_cross_packed_shape_(dots, mxfp8e5m2, neon)
nk_define_cross_pack_(dots, mxfp8e5m2, neon, e5m2, f16, nk_b256_vec_t, nk_load_e5m2x16_to_f16x16_neon_,
                      nk_partial_load_e5m2x16_to_f16x16_neon_, nk_store_b256_neon_, nk_partial_store_b16x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_pack_mxfp8_scales_neon_,
                      /*scale_bytes=*/5)
nk_define_cross_scaled_packed_(dots, mxfp8e5m2, neon, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_dot_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e5m2_neon_,
                               nk_dot_f32_from_wide_, /*normalized=*/0, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(dots, mxfp8e5m2, neon, f16, nk_dot_f32_from_wide_, /*normalized=*/0,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_TARGET_NEON

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_DOTS_NEON_H
