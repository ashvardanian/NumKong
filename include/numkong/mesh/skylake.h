/**
 *  @brief SIMD-accelerated Point Cloud Alignment for Skylake.
 *  @file include/numkong/mesh/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section skylake_mesh_instructions Key AVX-512 Mesh Instructions
 *
 *      Intrinsic               Instruction                   Skylake-X  Genoa
 *      _mm512_fmadd_ps         VFMADD132PS (ZMM, ZMM, ZMM)   4cy @ p05  4cy @ p01
 *      _mm512_permutexvar_ps   VPERMPS (ZMM, ZMM, ZMM)       3cy @ p5   4cy @ p12
 *      _mm512_permutex2var_ps  VPERMT2PS (ZMM, ZMM, ZMM)     3cy @ p5   4cy @ p12
 *      _mm512_extractf32x8_ps  VEXTRACTF32X8 (YMM, ZMM, I8)  3cy @ p5   1cy @ p0123
 *
 *  Most `*_f32` mesh kernels use a 15-lane stride-3 chunk layout: 5 xyz triplets per ZMM (lane 15
 *  masked to zero) so the xyz phase is identical across all chunks and no per-chunk deinterleave is
 *  needed. The 9 cross-covariance cells come from three accumulators a*b, a*rot1(b), a*rot2(b)
 *  demuxed per channel post-loop, where rot1/rot2 are cheap within-triplet permutexvar rotations.
 *  `*_f64`, `*_f16`, `*_bf16` kernels still use VPERMT2PS deinterleave (helpers retained below).
 *  Dual FMA accumulators on Skylake-X hide the 4cy latency for centroid and covariance computation.
 */
#ifndef NK_MESH_SKYLAKE_H
#define NK_MESH_SKYLAKE_H

#if NK_TARGET_X8664_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/dot/skylake.h"
#include "numkong/mesh/serial.h"
#include "numkong/spatial/haswell.h"
#include "numkong/cast/skylake.h"

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

/*  Deinterleave 48 floats (16 xyz triplets) into separate x, y, z vectors.
 *  Uses permutex2var shuffles instead of gather for ~1.8x speedup.
 *
 *  Input: 48 contiguous floats [x0,y0,z0, x1,y1,z1, ..., x15,y15,z15]
 *  Output: x[16], y[16], z[16] vectors
 *
 *  Implementation: Load 3 registers (r0,r1,r2), use 6 permutex2var ops to separate.
 *  Phase analysis: r0 starts at float 0 (phase 0), r1 at float 16 (phase 1), r2 at float 32 (phase 2)
 *
 *  X elements at memory positions: 0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45
 *    = r0[0,3,6,9,12,15], r1[2,5,8,11,14], r2[1,4,7,10,13]
 *  Y elements at memory positions: 1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46
 *    = r0[1,4,7,10,13], r1[0,3,6,9,12,15], r2[2,5,8,11,14]
 *  Z elements at memory positions: 2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47
 *    = r0[2,5,8,11,14], r1[1,4,7,10,13], r2[0,3,6,9,12,15]
 */
NK_INTERNAL void nk_deinterleave_f32x16_skylake_(                                            //
    nk_f32_t const *ptr, __m512 *x_f32x16_out, __m512 *y_f32x16_out, __m512 *z_f32x16_out) { //
    __m512 reg0_f32x16 = _mm512_loadu_ps(ptr);
    __m512 reg1_f32x16 = _mm512_loadu_ps(ptr + 16);
    __m512 reg2_f32x16 = _mm512_loadu_ps(ptr + 32);

    // X: reg0[0,3,6,9,12,15] + reg1[2,5,8,11,14] → 11 elements, then + reg2[1,4,7,10,13] → 16 elements
    // Indices for permutex2var: 0-15 = from first operand, 16-31 = from second operand
    __m512i idx_x_01_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
    __m512i idx_x_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    __m512 x01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_x_01_i32x16, reg1_f32x16);
    *x_f32x16_out = _mm512_permutex2var_ps(x01_f32x16, idx_x_2_i32x16, reg2_f32x16);

    // Y: reg0[1,4,7,10,13] + reg1[0,3,6,9,12,15] → 11 elements, then + reg2[2,5,8,11,14] → 16 elements
    __m512i idx_y_01_i32x16 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
    __m512i idx_y_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    __m512 y01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_y_01_i32x16, reg1_f32x16);
    *y_f32x16_out = _mm512_permutex2var_ps(y01_f32x16, idx_y_2_i32x16, reg2_f32x16);

    // Z: reg0[2,5,8,11,14] + reg1[1,4,7,10,13] → 10 elements, then + reg2[0,3,6,9,12,15] → 16 elements
    __m512i idx_z_01_i32x16 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
    __m512i idx_z_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    __m512 z01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_z_01_i32x16, reg1_f32x16);
    *z_f32x16_out = _mm512_permutex2var_ps(z01_f32x16, idx_z_2_i32x16, reg2_f32x16);
}

/*  Deinterleave 8 f64 3D points from xyz,xyz,xyz... to separate x,y,z vectors.
 *  Input: 24 consecutive f64 values (8 points * 3 coordinates)
 *  Output: Three __m512d vectors containing the x, y, z coordinates separately.
 */
NK_INTERNAL void nk_deinterleave_f64x8_skylake_(                                             //
    nk_f64_t const *ptr, __m512d *x_f64x8_out, __m512d *y_f64x8_out, __m512d *z_f64x8_out) { //
    __m512d reg0_f64x8 = _mm512_loadu_pd(ptr);                                               // elements 0-7
    __m512d reg1_f64x8 = _mm512_loadu_pd(ptr + 8);                                           // elements 8-15
    __m512d reg2_f64x8 = _mm512_loadu_pd(ptr + 16);                                          // elements 16-23

    // X: positions 0,3,6,9,12,15,18,21 → reg0[0,3,6] + reg1[1,4,7] + reg2[2,5]
    __m512i idx_x_01_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0);
    __m512i idx_x_2_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 10, 13);
    __m512d x01_f64x8 = _mm512_permutex2var_pd(reg0_f64x8, idx_x_01_i64x8, reg1_f64x8);
    *x_f64x8_out = _mm512_permutex2var_pd(x01_f64x8, idx_x_2_i64x8, reg2_f64x8);

    // Y: positions 1,4,7,10,13,16,19,22 → reg0[1,4,7] + reg1[2,5] + reg2[0,3,6]
    __m512i idx_y_01_i64x8 = _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0);
    __m512i idx_y_2_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 8, 11, 14);
    __m512d y01_f64x8 = _mm512_permutex2var_pd(reg0_f64x8, idx_y_01_i64x8, reg1_f64x8);
    *y_f64x8_out = _mm512_permutex2var_pd(y01_f64x8, idx_y_2_i64x8, reg2_f64x8);

    // Z: positions 2,5,8,11,14,17,20,23 → reg0[2,5] + reg1[0,3,6] + reg2[1,4,7]
    __m512i idx_z_01_i64x8 = _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0);
    __m512i idx_z_2_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 9, 12, 15);
    __m512d z01_f64x8 = _mm512_permutex2var_pd(reg0_f64x8, idx_z_01_i64x8, reg1_f64x8);
    *z_f64x8_out = _mm512_permutex2var_pd(z01_f64x8, idx_z_2_i64x8, reg2_f64x8);
}

/*  Deinterleave 16 f16 3D points from xyz,xyz,xyz... to separate x,y,z vectors in f32.
 *  Input: 48 consecutive f16 values (16 points * 3 coordinates)
 *  Output: Three __m512 vectors containing the x, y, z coordinates separately (as f32).
 */
NK_INTERNAL void nk_deinterleave_f16x16_to_f32x16_skylake_(                                  //
    nk_f16_t const *ptr, __m512 *x_f32x16_out, __m512 *y_f32x16_out, __m512 *z_f32x16_out) { //
    __m512 reg0_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(ptr)));
    __m512 reg1_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(ptr + 16)));
    __m512 reg2_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(ptr + 32)));

    __m512i idx_x_01_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
    __m512i idx_x_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    __m512 x01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_x_01_i32x16, reg1_f32x16);
    *x_f32x16_out = _mm512_permutex2var_ps(x01_f32x16, idx_x_2_i32x16, reg2_f32x16);

    __m512i idx_y_01_i32x16 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
    __m512i idx_y_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    __m512 y01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_y_01_i32x16, reg1_f32x16);
    *y_f32x16_out = _mm512_permutex2var_ps(y01_f32x16, idx_y_2_i32x16, reg2_f32x16);

    __m512i idx_z_01_i32x16 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
    __m512i idx_z_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    __m512 z01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_z_01_i32x16, reg1_f32x16);
    *z_f32x16_out = _mm512_permutex2var_ps(z01_f32x16, idx_z_2_i32x16, reg2_f32x16);
}

/*  Deinterleave 16 bf16 3D points from xyz,xyz,xyz... to separate x,y,z vectors in f32.
 *  Input: 48 consecutive bf16 values (16 points * 3 coordinates)
 *  Output: Three __m512 vectors containing the x, y, z coordinates separately (as f32).
 */
NK_INTERNAL void nk_deinterleave_bf16x16_to_f32x16_skylake_(                                  //
    nk_bf16_t const *ptr, __m512 *x_f32x16_out, __m512 *y_f32x16_out, __m512 *z_f32x16_out) { //
    __m512 reg0_f32x16 = nk_bf16x16_to_f32x16_skylake_(_mm256_loadu_si256((__m256i const *)(ptr)));
    __m512 reg1_f32x16 = nk_bf16x16_to_f32x16_skylake_(_mm256_loadu_si256((__m256i const *)(ptr + 16)));
    __m512 reg2_f32x16 = nk_bf16x16_to_f32x16_skylake_(_mm256_loadu_si256((__m256i const *)(ptr + 32)));

    __m512i idx_x_01_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
    __m512i idx_x_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    __m512 x01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_x_01_i32x16, reg1_f32x16);
    *x_f32x16_out = _mm512_permutex2var_ps(x01_f32x16, idx_x_2_i32x16, reg2_f32x16);

    __m512i idx_y_01_i32x16 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
    __m512i idx_y_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    __m512 y01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_y_01_i32x16, reg1_f32x16);
    *y_f32x16_out = _mm512_permutex2var_ps(y01_f32x16, idx_y_2_i32x16, reg2_f32x16);

    __m512i idx_z_01_i32x16 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
    __m512i idx_z_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    __m512 z01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_z_01_i32x16, reg1_f32x16);
    *z_f32x16_out = _mm512_permutex2var_ps(z01_f32x16, idx_z_2_i32x16, reg2_f32x16);
}

/*  Masked-tail deinterleave for f16: loads up to 16 xyz points using AVX-512 masked loads,
 *  converts f16→f32, and deinterleaves into separate x,y,z vectors.
 *  Unused lanes are zero. Uses the same permutex2var shuffle as the full-width version.
 */
NK_INTERNAL void nk_deinterleave_f16_tail_to_f32x16_skylake_(                                                 //
    nk_f16_t const *ptr, nk_size_t count, __m512 *x_f32x16_out, __m512 *y_f32x16_out, __m512 *z_f32x16_out) { //
    nk_size_t total = count * 3;
    __mmask16 mask0_i16x16 = (__mmask16)_bzhi_u32(0xFFFF, total >= 16 ? 16 : total);
    __mmask16 mask1_i16x16 = total > 16 ? (__mmask16)_bzhi_u32(0xFFFF, total >= 32 ? 16 : total - 16) : 0;
    __mmask16 mask2_i16x16 = total > 32 ? (__mmask16)_bzhi_u32(0xFFFF, total - 32) : 0;
    __m512 reg0_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask0_i16x16, ptr));
    __m512 reg1_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask1_i16x16, ptr + 16));
    __m512 reg2_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask2_i16x16, ptr + 32));

    __m512i idx_x_01_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
    __m512i idx_x_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    *x_f32x16_out = _mm512_permutex2var_ps(_mm512_permutex2var_ps(reg0_f32x16, idx_x_01_i32x16, reg1_f32x16),
                                           idx_x_2_i32x16, reg2_f32x16);

    __m512i idx_y_01_i32x16 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
    __m512i idx_y_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    *y_f32x16_out = _mm512_permutex2var_ps(_mm512_permutex2var_ps(reg0_f32x16, idx_y_01_i32x16, reg1_f32x16),
                                           idx_y_2_i32x16, reg2_f32x16);

    __m512i idx_z_01_i32x16 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
    __m512i idx_z_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    *z_f32x16_out = _mm512_permutex2var_ps(_mm512_permutex2var_ps(reg0_f32x16, idx_z_01_i32x16, reg1_f32x16),
                                           idx_z_2_i32x16, reg2_f32x16);
}

/*  Masked-tail deinterleave for bf16: same as f16 but with bf16→f32 conversion. */
NK_INTERNAL void nk_deinterleave_bf16_tail_to_f32x16_skylake_(                                                 //
    nk_bf16_t const *ptr, nk_size_t count, __m512 *x_f32x16_out, __m512 *y_f32x16_out, __m512 *z_f32x16_out) { //
    nk_size_t total = count * 3;
    __mmask16 mask0_i16x16 = (__mmask16)_bzhi_u32(0xFFFF, total >= 16 ? 16 : total);
    __mmask16 mask1_i16x16 = total > 16 ? (__mmask16)_bzhi_u32(0xFFFF, total >= 32 ? 16 : total - 16) : 0;
    __mmask16 mask2_i16x16 = total > 32 ? (__mmask16)_bzhi_u32(0xFFFF, total - 32) : 0;
    __m512 reg0_f32x16 = nk_bf16x16_to_f32x16_skylake_(_mm256_maskz_loadu_epi16(mask0_i16x16, ptr));
    __m512 reg1_f32x16 = nk_bf16x16_to_f32x16_skylake_(_mm256_maskz_loadu_epi16(mask1_i16x16, ptr + 16));
    __m512 reg2_f32x16 = nk_bf16x16_to_f32x16_skylake_(_mm256_maskz_loadu_epi16(mask2_i16x16, ptr + 32));

    __m512i idx_x_01_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
    __m512i idx_x_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    *x_f32x16_out = _mm512_permutex2var_ps(_mm512_permutex2var_ps(reg0_f32x16, idx_x_01_i32x16, reg1_f32x16),
                                           idx_x_2_i32x16, reg2_f32x16);

    __m512i idx_y_01_i32x16 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
    __m512i idx_y_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    *y_f32x16_out = _mm512_permutex2var_ps(_mm512_permutex2var_ps(reg0_f32x16, idx_y_01_i32x16, reg1_f32x16),
                                           idx_y_2_i32x16, reg2_f32x16);

    __m512i idx_z_01_i32x16 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
    __m512i idx_z_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    *z_f32x16_out = _mm512_permutex2var_ps(_mm512_permutex2var_ps(reg0_f32x16, idx_z_01_i32x16, reg1_f32x16),
                                           idx_z_2_i32x16, reg2_f32x16);
}

NK_INTERNAL nk_f64_t nk_reduce_stable_f64x8_skylake_(__m512d values_f64x8) {
    nk_b512_vec_t values;
    values.zmm_pd = values_f64x8;
    nk_f64_t sum = 0.0, compensation = 0.0;
    for (nk_size_t lane_index = 0; lane_index != 8; ++lane_index)
        nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[lane_index]);
    return sum + compensation;
}

NK_INTERNAL void nk_accumulate_square_f64x8_skylake_(__m512d *sum_f64x8, __m512d *compensation_f64x8,
                                                     __m512d values_f64x8) {
    __m512d product_f64x8 = _mm512_mul_pd(values_f64x8, values_f64x8);
    __m512d product_error_f64x8 = _mm512_fmsub_pd(values_f64x8, values_f64x8, product_f64x8);
    __m512d tentative_sum_f64x8 = _mm512_add_pd(*sum_f64x8, product_f64x8);
    __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, *sum_f64x8);
    __m512d sum_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(*sum_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
        _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));
    *sum_f64x8 = tentative_sum_f64x8;
    *compensation_f64x8 = _mm512_add_pd(*compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
}

/*  Single-pass streaming statistics over an f32 xyz point-cloud pair.
 *  Processes 5 xyz triplets per chunk (15 fp32 lanes, lane 15 masked to zero) so the stride-3
 *  phase is identical across all chunks and no deinterleave is needed. All accumulators are f64.
 *  Outputs via pointers:
 *    sum_a_out[3] / sum_b_out[3]     - per-channel Sum(a), Sum(b)
 *    raw_covarianceariance_out[9]                  - row-major uncentered Sum(a_j * b_k)
 *    norm_squared_a_out / norm_squared_b_out       - Sum(||a||^2), Sum(||b||^2) across all three channels
 *
 *  The 9 H-cells come from three product accumulators prod_{diag,rot1,rot2} demuxed post-loop
 *  by a-channel. Rotations of b happen in fp64 via permutex2var_pd on the already-widened
 *  halves: widening the rotated fp32 vector would add two extra cvtps_pd per chunk, which we
 *  skip. Post-loop, each (accumulator, channel) pair is gathered into a single 8-lane vector
 *  via one maskz-permutex2var_pd and reduced once — 17 horizontal reductions total (the
 *  theoretical minimum for 17 scalar outputs) instead of 32 masked ones.
 */
NK_INTERNAL void nk_mesh_streaming_stats_f32_skylake_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *sum_a_out, nk_f64_t *sum_b_out,
    nk_f64_t *raw_covarianceariance_out, nk_f64_t *norm_squared_a_out, nk_f64_t *norm_squared_b_out) {

    // Within-triplet rotation indices for fp64 permutex2var across (b_low, b_high) as the two
    // sources. Indices 0..7 pull from b_low, 8..15 pull from b_high. Derived from the fp32
    // rotation pattern {1,2,0,4,5,3,7,8,6,10,11,9,13,14,12,15} (rot1) and {2,0,1,5,3,4,8,6,
    // 7,11,9,10,14,12,13,15} (rot2) split at fp32 lane 8.
    __m512i const idx_rotation_1_low_i64x8 = _mm512_setr_epi64(1, 2, 0, 4, 5, 3, 7, 8);
    __m512i const idx_rotation_1_high_i64x8 = _mm512_setr_epi64(6, 10, 11, 9, 13, 14, 12, 15);
    __m512i const idx_rotation_2_low_i64x8 = _mm512_setr_epi64(2, 0, 1, 5, 3, 4, 8, 6);
    __m512i const idx_rotation_2_high_i64x8 = _mm512_setr_epi64(7, 11, 9, 10, 14, 12, 13, 15);

    // Per-channel gather indices packing the 5 contributing fp64 lanes (across both halves)
    // into lanes 0..4 of the output, with lanes 5..7 zeroed by maskz so the subsequent
    // _mm512_reduce_add_pd is exact without needing a mask-reduce variant.
    //    x -> low {0,3,6} + high {1,4}  = idx [0, 3, 6, 9, 12, _, _, _]
    //    y -> low {1,4,7} + high {2,5}  = idx [1, 4, 7, 10, 13, _, _, _]
    //    z -> low {2,5}   + high {0,3,6} = idx [2, 5, 8, 11, 14, _, _, _]
    __m512i const idx_channel_x_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 0, 0, 0);
    __m512i const idx_channel_y_i64x8 = _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0);
    __m512i const idx_channel_z_i64x8 = _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0);
    __mmask8 const channel_lanes_mask = 0x1F;

    __m512d const zeros_f64x8 = _mm512_setzero_pd();
    __m512d sum_a_low_f64x8 = zeros_f64x8, sum_a_high_f64x8 = zeros_f64x8;
    __m512d sum_b_low_f64x8 = zeros_f64x8, sum_b_high_f64x8 = zeros_f64x8;
    __m512d norm_squared_a_low_f64x8 = zeros_f64x8, norm_squared_a_high_f64x8 = zeros_f64x8;
    __m512d norm_squared_b_low_f64x8 = zeros_f64x8, norm_squared_b_high_f64x8 = zeros_f64x8;
    __m512d product_diagonal_low_f64x8 = zeros_f64x8, product_diagonal_high_f64x8 = zeros_f64x8;
    __m512d product_rotation_1_low_f64x8 = zeros_f64x8, product_rotation_1_high_f64x8 = zeros_f64x8;
    __m512d product_rotation_2_low_f64x8 = zeros_f64x8, product_rotation_2_high_f64x8 = zeros_f64x8;

    nk_size_t index = 0;
    // Main loop: 5 points (15 fp32) per chunk, lane 15 zeroed by mask 0x7FFF.
    for (; index + 5 <= n; index += 5) {
        __m512 a_f32x16 = _mm512_maskz_loadu_ps(0x7FFF, a + index * 3);
        __m512 b_f32x16 = _mm512_maskz_loadu_ps(0x7FFF, b + index * 3);

        __m512d a_low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_f32x16));
        __m512d a_high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_f32x16, 1));
        __m512d b_low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_f32x16));
        __m512d b_high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_f32x16, 1));

        __m512d b_rot1_low_f64x8 = _mm512_permutex2var_pd(b_low_f64x8, idx_rotation_1_low_i64x8, b_high_f64x8);
        __m512d b_rot1_high_f64x8 = _mm512_permutex2var_pd(b_low_f64x8, idx_rotation_1_high_i64x8, b_high_f64x8);
        __m512d b_rot2_low_f64x8 = _mm512_permutex2var_pd(b_low_f64x8, idx_rotation_2_low_i64x8, b_high_f64x8);
        __m512d b_rot2_high_f64x8 = _mm512_permutex2var_pd(b_low_f64x8, idx_rotation_2_high_i64x8, b_high_f64x8);

        sum_a_low_f64x8 = _mm512_add_pd(sum_a_low_f64x8, a_low_f64x8);
        sum_a_high_f64x8 = _mm512_add_pd(sum_a_high_f64x8, a_high_f64x8);
        sum_b_low_f64x8 = _mm512_add_pd(sum_b_low_f64x8, b_low_f64x8);
        sum_b_high_f64x8 = _mm512_add_pd(sum_b_high_f64x8, b_high_f64x8);

        norm_squared_a_low_f64x8 = _mm512_fmadd_pd(a_low_f64x8, a_low_f64x8, norm_squared_a_low_f64x8);
        norm_squared_a_high_f64x8 = _mm512_fmadd_pd(a_high_f64x8, a_high_f64x8, norm_squared_a_high_f64x8);
        norm_squared_b_low_f64x8 = _mm512_fmadd_pd(b_low_f64x8, b_low_f64x8, norm_squared_b_low_f64x8);
        norm_squared_b_high_f64x8 = _mm512_fmadd_pd(b_high_f64x8, b_high_f64x8, norm_squared_b_high_f64x8);

        product_diagonal_low_f64x8 = _mm512_fmadd_pd(a_low_f64x8, b_low_f64x8, product_diagonal_low_f64x8);
        product_diagonal_high_f64x8 = _mm512_fmadd_pd(a_high_f64x8, b_high_f64x8, product_diagonal_high_f64x8);
        product_rotation_1_low_f64x8 = _mm512_fmadd_pd(a_low_f64x8, b_rot1_low_f64x8, product_rotation_1_low_f64x8);
        product_rotation_1_high_f64x8 = _mm512_fmadd_pd(a_high_f64x8, b_rot1_high_f64x8, product_rotation_1_high_f64x8);
        product_rotation_2_low_f64x8 = _mm512_fmadd_pd(a_low_f64x8, b_rot2_low_f64x8, product_rotation_2_low_f64x8);
        product_rotation_2_high_f64x8 = _mm512_fmadd_pd(a_high_f64x8, b_rot2_high_f64x8, product_rotation_2_high_f64x8);
    }

    // Tail: 1..4 points (3..12 fp32) via narrower mask; identical body.
    if (index < n) {
        nk_size_t tail_floats = (n - index) * 3;
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0x7FFF, tail_floats);
        __m512 a_f32x16 = _mm512_maskz_loadu_ps(tail_mask, a + index * 3);
        __m512 b_f32x16 = _mm512_maskz_loadu_ps(tail_mask, b + index * 3);

        __m512d a_low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_f32x16));
        __m512d a_high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_f32x16, 1));
        __m512d b_low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_f32x16));
        __m512d b_high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_f32x16, 1));

        __m512d b_rot1_low_f64x8 = _mm512_permutex2var_pd(b_low_f64x8, idx_rotation_1_low_i64x8, b_high_f64x8);
        __m512d b_rot1_high_f64x8 = _mm512_permutex2var_pd(b_low_f64x8, idx_rotation_1_high_i64x8, b_high_f64x8);
        __m512d b_rot2_low_f64x8 = _mm512_permutex2var_pd(b_low_f64x8, idx_rotation_2_low_i64x8, b_high_f64x8);
        __m512d b_rot2_high_f64x8 = _mm512_permutex2var_pd(b_low_f64x8, idx_rotation_2_high_i64x8, b_high_f64x8);

        sum_a_low_f64x8 = _mm512_add_pd(sum_a_low_f64x8, a_low_f64x8);
        sum_a_high_f64x8 = _mm512_add_pd(sum_a_high_f64x8, a_high_f64x8);
        sum_b_low_f64x8 = _mm512_add_pd(sum_b_low_f64x8, b_low_f64x8);
        sum_b_high_f64x8 = _mm512_add_pd(sum_b_high_f64x8, b_high_f64x8);

        norm_squared_a_low_f64x8 = _mm512_fmadd_pd(a_low_f64x8, a_low_f64x8, norm_squared_a_low_f64x8);
        norm_squared_a_high_f64x8 = _mm512_fmadd_pd(a_high_f64x8, a_high_f64x8, norm_squared_a_high_f64x8);
        norm_squared_b_low_f64x8 = _mm512_fmadd_pd(b_low_f64x8, b_low_f64x8, norm_squared_b_low_f64x8);
        norm_squared_b_high_f64x8 = _mm512_fmadd_pd(b_high_f64x8, b_high_f64x8, norm_squared_b_high_f64x8);

        product_diagonal_low_f64x8 = _mm512_fmadd_pd(a_low_f64x8, b_low_f64x8, product_diagonal_low_f64x8);
        product_diagonal_high_f64x8 = _mm512_fmadd_pd(a_high_f64x8, b_high_f64x8, product_diagonal_high_f64x8);
        product_rotation_1_low_f64x8 = _mm512_fmadd_pd(a_low_f64x8, b_rot1_low_f64x8, product_rotation_1_low_f64x8);
        product_rotation_1_high_f64x8 = _mm512_fmadd_pd(a_high_f64x8, b_rot1_high_f64x8, product_rotation_1_high_f64x8);
        product_rotation_2_low_f64x8 = _mm512_fmadd_pd(a_low_f64x8, b_rot2_low_f64x8, product_rotation_2_low_f64x8);
        product_rotation_2_high_f64x8 = _mm512_fmadd_pd(a_high_f64x8, b_rot2_high_f64x8, product_rotation_2_high_f64x8);
    }

    // Post-loop: gather each (accumulator, a-channel) pair into a single 8-lane vector via one
    // maskz-permutex2var_pd across (low, high) halves, then one _mm512_reduce_add_pd per scalar
    // output. 17 reductions total (6 sums + 9 H cells + 2 norms) = the scalar-output floor.

    __m512d sum_a_x_f64x8 = _mm512_maskz_permutex2var_pd(channel_lanes_mask, sum_a_low_f64x8, idx_channel_x_i64x8,
                                                         sum_a_high_f64x8);
    __m512d sum_a_y_f64x8 = _mm512_maskz_permutex2var_pd(channel_lanes_mask, sum_a_low_f64x8, idx_channel_y_i64x8,
                                                         sum_a_high_f64x8);
    __m512d sum_a_z_f64x8 = _mm512_maskz_permutex2var_pd(channel_lanes_mask, sum_a_low_f64x8, idx_channel_z_i64x8,
                                                         sum_a_high_f64x8);
    sum_a_out[0] = _mm512_reduce_add_pd(sum_a_x_f64x8);
    sum_a_out[1] = _mm512_reduce_add_pd(sum_a_y_f64x8);
    sum_a_out[2] = _mm512_reduce_add_pd(sum_a_z_f64x8);

    __m512d sum_b_x_f64x8 = _mm512_maskz_permutex2var_pd(channel_lanes_mask, sum_b_low_f64x8, idx_channel_x_i64x8,
                                                         sum_b_high_f64x8);
    __m512d sum_b_y_f64x8 = _mm512_maskz_permutex2var_pd(channel_lanes_mask, sum_b_low_f64x8, idx_channel_y_i64x8,
                                                         sum_b_high_f64x8);
    __m512d sum_b_z_f64x8 = _mm512_maskz_permutex2var_pd(channel_lanes_mask, sum_b_low_f64x8, idx_channel_z_i64x8,
                                                         sum_b_high_f64x8);
    sum_b_out[0] = _mm512_reduce_add_pd(sum_b_x_f64x8);
    sum_b_out[1] = _mm512_reduce_add_pd(sum_b_y_f64x8);
    sum_b_out[2] = _mm512_reduce_add_pd(sum_b_z_f64x8);

    // H cells: a-channel picks which demux mask applies; prod-vector picks which b-channel the
    // product pairs a with (diag -> same, rot1 -> +1, rot2 -> +2 mod 3).
    __m512d product_diagonal_x_f64x8 = _mm512_maskz_permutex2var_pd( //
        channel_lanes_mask, product_diagonal_low_f64x8, idx_channel_x_i64x8, product_diagonal_high_f64x8);
    __m512d product_diagonal_y_f64x8 = _mm512_maskz_permutex2var_pd( //
        channel_lanes_mask, product_diagonal_low_f64x8, idx_channel_y_i64x8, product_diagonal_high_f64x8);
    __m512d product_diagonal_z_f64x8 = _mm512_maskz_permutex2var_pd( //
        channel_lanes_mask, product_diagonal_low_f64x8, idx_channel_z_i64x8, product_diagonal_high_f64x8);
    __m512d product_rotation_1_x_f64x8 = _mm512_maskz_permutex2var_pd( //
        channel_lanes_mask, product_rotation_1_low_f64x8, idx_channel_x_i64x8, product_rotation_1_high_f64x8);
    __m512d product_rotation_1_y_f64x8 = _mm512_maskz_permutex2var_pd( //
        channel_lanes_mask, product_rotation_1_low_f64x8, idx_channel_y_i64x8, product_rotation_1_high_f64x8);
    __m512d product_rotation_1_z_f64x8 = _mm512_maskz_permutex2var_pd( //
        channel_lanes_mask, product_rotation_1_low_f64x8, idx_channel_z_i64x8, product_rotation_1_high_f64x8);
    __m512d product_rotation_2_x_f64x8 = _mm512_maskz_permutex2var_pd( //
        channel_lanes_mask, product_rotation_2_low_f64x8, idx_channel_x_i64x8, product_rotation_2_high_f64x8);
    __m512d product_rotation_2_y_f64x8 = _mm512_maskz_permutex2var_pd( //
        channel_lanes_mask, product_rotation_2_low_f64x8, idx_channel_y_i64x8, product_rotation_2_high_f64x8);
    __m512d product_rotation_2_z_f64x8 = _mm512_maskz_permutex2var_pd( //
        channel_lanes_mask, product_rotation_2_low_f64x8, idx_channel_z_i64x8, product_rotation_2_high_f64x8);

    raw_covarianceariance_out[0] = _mm512_reduce_add_pd(product_diagonal_x_f64x8);   // H[x,x]
    raw_covarianceariance_out[1] = _mm512_reduce_add_pd(product_rotation_1_x_f64x8); // H[x,y]
    raw_covarianceariance_out[2] = _mm512_reduce_add_pd(product_rotation_2_x_f64x8); // H[x,z]
    raw_covarianceariance_out[3] = _mm512_reduce_add_pd(product_rotation_2_y_f64x8); // H[y,x]
    raw_covarianceariance_out[4] = _mm512_reduce_add_pd(product_diagonal_y_f64x8);   // H[y,y]
    raw_covarianceariance_out[5] = _mm512_reduce_add_pd(product_rotation_1_y_f64x8); // H[y,z]
    raw_covarianceariance_out[6] = _mm512_reduce_add_pd(product_rotation_1_z_f64x8); // H[z,x]
    raw_covarianceariance_out[7] = _mm512_reduce_add_pd(product_rotation_2_z_f64x8); // H[z,y]
    raw_covarianceariance_out[8] = _mm512_reduce_add_pd(product_diagonal_z_f64x8);   // H[z,z]

    // Norms collapse all three channels, no demux.
    *norm_squared_a_out = _mm512_reduce_add_pd(_mm512_add_pd(norm_squared_a_low_f64x8, norm_squared_a_high_f64x8));
    *norm_squared_b_out = _mm512_reduce_add_pd(_mm512_add_pd(norm_squared_b_low_f64x8, norm_squared_b_high_f64x8));
}

NK_PUBLIC void nk_rmsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    // 15-lane stride-3 chunks: 5 points (15 fp32) per iteration, lane 15 zeroed by mask.
    // Identity rotation + zero centroid (per commit 1a83ab4f) make this a single (a-b)^2 sum.
    __m512d sum_squared_low_f64x8 = _mm512_setzero_pd();
    __m512d sum_squared_high_f64x8 = _mm512_setzero_pd();
    nk_size_t index = 0;

    for (; index + 5 <= n; index += 5) {
        __m512 a_f32x16 = _mm512_maskz_loadu_ps(0x7FFF, a + index * 3);
        __m512 b_f32x16 = _mm512_maskz_loadu_ps(0x7FFF, b + index * 3);
        // Widen before subtracting: fp32 subtraction catastrophically cancels when a ~ b.
        __m512d a_low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_f32x16));
        __m512d a_high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_f32x16, 1));
        __m512d b_low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_f32x16));
        __m512d b_high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_f32x16, 1));
        __m512d delta_low_f64x8 = _mm512_sub_pd(a_low_f64x8, b_low_f64x8);
        __m512d delta_high_f64x8 = _mm512_sub_pd(a_high_f64x8, b_high_f64x8);
        sum_squared_low_f64x8 = _mm512_fmadd_pd(delta_low_f64x8, delta_low_f64x8, sum_squared_low_f64x8);
        sum_squared_high_f64x8 = _mm512_fmadd_pd(delta_high_f64x8, delta_high_f64x8, sum_squared_high_f64x8);
    }

    if (index < n) {
        nk_size_t tail_floats = (n - index) * 3;
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0x7FFF, tail_floats);
        __m512 a_f32x16 = _mm512_maskz_loadu_ps(tail_mask, a + index * 3);
        __m512 b_f32x16 = _mm512_maskz_loadu_ps(tail_mask, b + index * 3);
        __m512d a_low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_f32x16));
        __m512d a_high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_f32x16, 1));
        __m512d b_low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_f32x16));
        __m512d b_high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_f32x16, 1));
        __m512d delta_low_f64x8 = _mm512_sub_pd(a_low_f64x8, b_low_f64x8);
        __m512d delta_high_f64x8 = _mm512_sub_pd(a_high_f64x8, b_high_f64x8);
        sum_squared_low_f64x8 = _mm512_fmadd_pd(delta_low_f64x8, delta_low_f64x8, sum_squared_low_f64x8);
        sum_squared_high_f64x8 = _mm512_fmadd_pd(delta_high_f64x8, delta_high_f64x8, sum_squared_high_f64x8);
    }

    nk_f64_t sum_squared = _mm512_reduce_add_pd(_mm512_add_pd(sum_squared_low_f64x8, sum_squared_high_f64x8));
    *result = nk_f64_sqrt_haswell(sum_squared / (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    // Single pass over (a, b) via streaming-stats helper — no deinterleave, no second SSD pass.
    nk_f64_t sum_a[3], sum_b[3], raw_covariance[9], norm_squared_a, norm_squared_b;
    nk_mesh_streaming_stats_f32_skylake_(a, b, n, sum_a, sum_b, raw_covariance, &norm_squared_a, &norm_squared_b);

    nk_f64_t n_f64 = (nk_f64_t)n;
    nk_f64_t inv_n = 1.0 / n_f64;
    nk_f64_t centroid_a_x = sum_a[0] * inv_n, centroid_a_y = sum_a[1] * inv_n, centroid_a_z = sum_a[2] * inv_n;
    nk_f64_t centroid_b_x = sum_b[0] * inv_n, centroid_b_y = sum_b[1] * inv_n, centroid_b_z = sum_b[2] * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;
    if (scale) *scale = 1.0f;

    // Parallel-axis correction: H_centered[j,k] = Sum(a_j * b_k) - n * centroid_a[j] * centroid_b[k].
    nk_f64_t cross_covariance[9];
    cross_covariance[0] = raw_covariance[0] - n_f64 * centroid_a_x * centroid_b_x;
    cross_covariance[1] = raw_covariance[1] - n_f64 * centroid_a_x * centroid_b_y;
    cross_covariance[2] = raw_covariance[2] - n_f64 * centroid_a_x * centroid_b_z;
    cross_covariance[3] = raw_covariance[3] - n_f64 * centroid_a_y * centroid_b_x;
    cross_covariance[4] = raw_covariance[4] - n_f64 * centroid_a_y * centroid_b_y;
    cross_covariance[5] = raw_covariance[5] - n_f64 * centroid_a_y * centroid_b_z;
    cross_covariance[6] = raw_covariance[6] - n_f64 * centroid_a_z * centroid_b_x;
    cross_covariance[7] = raw_covariance[7] - n_f64 * centroid_a_z * centroid_b_y;
    cross_covariance[8] = raw_covariance[8] - n_f64 * centroid_a_z * centroid_b_z;

    // Identity-dominant short-circuit: skip SVD + rotation_from_svd when H is near-diagonal
    // positive-definite. `r` is set to identity and trace(R * H) collapses to H[0]+H[4]+H[8].
    // Saves ~500 cycles on aligned/pre-registered inputs; zero cost when inputs are random
    //  (branch is well-predicted in practice).
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        if (nk_det3x3_f64_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f64_t centered_norm_squared_a = norm_squared_a -
                                       n_f64 * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b -
                                       n_f64 * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;

    nk_f64_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0 * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_haswell(sum_squared / n_f64);
}

NK_PUBLIC void nk_rmsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    __m512i const gather_idx_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros_f64x8 = _mm512_setzero_pd();
    __m512d sum_squared_x_f64x8 = zeros_f64x8, sum_squared_y_f64x8 = zeros_f64x8, sum_squared_z_f64x8 = zeros_f64x8;

    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;
    nk_size_t i = 0;

    // Main loop with 2x unrolling for better latency hiding
    for (; i + 16 <= n; i += 16) {
        // Iteration 0
        nk_deinterleave_f64x8_skylake_(a + i * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + i * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        __m512d delta_x_f64x8 = _mm512_sub_pd(a_x_f64x8, b_x_f64x8),
                delta_y_f64x8 = _mm512_sub_pd(a_y_f64x8, b_y_f64x8),
                delta_z_f64x8 = _mm512_sub_pd(a_z_f64x8, b_z_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_f64x8, delta_x_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8, sum_squared_z_f64x8);

        // Iteration 1
        __m512d a_x1_f64x8, a_y1_f64x8, a_z1_f64x8, b_x1_f64x8, b_y1_f64x8, b_z1_f64x8;
        nk_deinterleave_f64x8_skylake_(a + (i + 8) * 3, &a_x1_f64x8, &a_y1_f64x8, &a_z1_f64x8);
        nk_deinterleave_f64x8_skylake_(b + (i + 8) * 3, &b_x1_f64x8, &b_y1_f64x8, &b_z1_f64x8);

        __m512d delta_x1_f64x8 = _mm512_sub_pd(a_x1_f64x8, b_x1_f64x8),
                delta_y1_f64x8 = _mm512_sub_pd(a_y1_f64x8, b_y1_f64x8),
                delta_z1_f64x8 = _mm512_sub_pd(a_z1_f64x8, b_z1_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x1_f64x8, delta_x1_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y1_f64x8, delta_y1_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z1_f64x8, delta_z1_f64x8, sum_squared_z_f64x8);
    }

    // Handle 8-point remainder
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f64x8_skylake_(a + i * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + i * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        __m512d delta_x_f64x8 = _mm512_sub_pd(a_x_f64x8, b_x_f64x8),
                delta_y_f64x8 = _mm512_sub_pd(a_y_f64x8, b_y_f64x8),
                delta_z_f64x8 = _mm512_sub_pd(a_z_f64x8, b_z_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_f64x8, delta_x_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8, sum_squared_z_f64x8);
    }

    // Tail: use masked gather
    if (i < n) {
        nk_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        nk_f64_t const *a_tail = a + i * 3;
        nk_f64_t const *b_tail = b + i * 3;

        a_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 0, 8);
        a_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 1, 8);
        a_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 2, 8);
        b_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 0, 8);
        b_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 1, 8);
        b_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 2, 8);

        __m512d delta_x_f64x8 = _mm512_sub_pd(a_x_f64x8, b_x_f64x8),
                delta_y_f64x8 = _mm512_sub_pd(a_y_f64x8, b_y_f64x8),
                delta_z_f64x8 = _mm512_sub_pd(a_z_f64x8, b_z_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_f64x8, delta_x_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8, sum_squared_z_f64x8);
        i = n;
    }

    nk_f64_t total_squared_x = nk_reduce_stable_f64x8_skylake_(sum_squared_x_f64x8), total_squared_x_compensation = 0.0;
    nk_f64_t total_squared_y = nk_reduce_stable_f64x8_skylake_(sum_squared_y_f64x8), total_squared_y_compensation = 0.0;
    nk_f64_t total_squared_z = nk_reduce_stable_f64x8_skylake_(sum_squared_z_f64x8), total_squared_z_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_f64_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        nk_accumulate_square_f64_(&total_squared_x, &total_squared_x_compensation, delta_x);
        nk_accumulate_square_f64_(&total_squared_y, &total_squared_y_compensation, delta_y);
        nk_accumulate_square_f64_(&total_squared_z, &total_squared_z_compensation, delta_z);
    }

    total_squared_x += total_squared_x_compensation, total_squared_y += total_squared_y_compensation,
        total_squared_z += total_squared_z_compensation;
    *result = nk_f64_sqrt_haswell((total_squared_x + total_squared_y + total_squared_z) / (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Optimized fused single-pass implementation for f64.
    // Computes centroids and covariance matrix in one pass using the identity:
    //   Hᵢⱼ = Σ((aᵢ - ā) × (bⱼ - b̄))
    //       = Σ(aᵢ × bⱼ) - Σaᵢ × Σbⱼ / n
    __m512i const gather_idx_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros_f64x8 = _mm512_setzero_pd();

    // Accumulators for centroids
    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;

    // Accumulators for covariance matrix (sum of outer products)
    __m512d covariance_xx_f64x8 = zeros_f64x8, covariance_xy_f64x8 = zeros_f64x8, covariance_xz_f64x8 = zeros_f64x8;
    __m512d covariance_yx_f64x8 = zeros_f64x8, covariance_yy_f64x8 = zeros_f64x8, covariance_yz_f64x8 = zeros_f64x8;
    __m512d covariance_zx_f64x8 = zeros_f64x8, covariance_zy_f64x8 = zeros_f64x8, covariance_zz_f64x8 = zeros_f64x8;
    __m512d norm_squared_a_f64x8 = zeros_f64x8, norm_squared_b_f64x8 = zeros_f64x8;

    nk_size_t i = 0;
    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;

    // Fused single-pass: accumulate sums, outer products, and norms^2 together
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f64x8_skylake_(a + i * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + i * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        // Accumulate centroids
        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        // Accumulate outer products (raw, not centered)
        covariance_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, covariance_xx_f64x8),
        covariance_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, covariance_xy_f64x8),
        covariance_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, covariance_xz_f64x8);
        covariance_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, covariance_yx_f64x8),
        covariance_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, covariance_yy_f64x8),
        covariance_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, covariance_yz_f64x8);
        covariance_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, covariance_zx_f64x8),
        covariance_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, covariance_zy_f64x8),
        covariance_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, covariance_zz_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_x_f64x8, a_x_f64x8, norm_squared_a_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_y_f64x8, a_y_f64x8, norm_squared_a_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_z_f64x8, a_z_f64x8, norm_squared_a_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_x_f64x8, b_x_f64x8, norm_squared_b_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_y_f64x8, b_y_f64x8, norm_squared_b_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_z_f64x8, b_z_f64x8, norm_squared_b_f64x8);
    }

    // Tail: masked gather for remaining points
    if (i < n) {
        nk_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        nk_f64_t const *a_tail = a + i * 3;
        nk_f64_t const *b_tail = b + i * 3;

        a_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 0, 8);
        a_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 1, 8);
        a_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 2, 8);
        b_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 0, 8);
        b_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 1, 8);
        b_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 2, 8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        covariance_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, covariance_xx_f64x8),
        covariance_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, covariance_xy_f64x8),
        covariance_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, covariance_xz_f64x8);
        covariance_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, covariance_yx_f64x8),
        covariance_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, covariance_yy_f64x8),
        covariance_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, covariance_yz_f64x8);
        covariance_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, covariance_zx_f64x8),
        covariance_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, covariance_zy_f64x8),
        covariance_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, covariance_zz_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_x_f64x8, a_x_f64x8, norm_squared_a_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_y_f64x8, a_y_f64x8, norm_squared_a_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_z_f64x8, a_z_f64x8, norm_squared_a_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_x_f64x8, b_x_f64x8, norm_squared_b_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_y_f64x8, b_y_f64x8, norm_squared_b_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_z_f64x8, b_z_f64x8, norm_squared_b_f64x8);
        i = n;
    }

    // Reduce centroids and covariance.
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t sum_a_x = nk_reduce_stable_f64x8_skylake_(sum_a_x_f64x8), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x8_skylake_(sum_a_y_f64x8), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x8_skylake_(sum_a_z_f64x8), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x8_skylake_(sum_b_x_f64x8), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x8_skylake_(sum_b_y_f64x8), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x8_skylake_(sum_b_z_f64x8), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x8_skylake_(covariance_xx_f64x8), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x8_skylake_(covariance_xy_f64x8), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x8_skylake_(covariance_xz_f64x8), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x8_skylake_(covariance_yx_f64x8), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x8_skylake_(covariance_yy_f64x8), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x8_skylake_(covariance_yz_f64x8), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x8_skylake_(covariance_zx_f64x8), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x8_skylake_(covariance_zy_f64x8), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x8_skylake_(covariance_zz_f64x8), covariance_z_z_compensation = 0.0;
    nk_f64_t norm_squared_a_sum = nk_reduce_stable_f64x8_skylake_(norm_squared_a_f64x8),
             norm_squared_a_compensation = 0.0;
    nk_f64_t norm_squared_b_sum = nk_reduce_stable_f64x8_skylake_(norm_squared_b_f64x8),
             norm_squared_b_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax);
        nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay);
        nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx);
        nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by);
        nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx);
        nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by);
        nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx);
        nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by);
        nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx);
        nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by);
        nk_accumulate_product_f64_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ax);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ay);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, az);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bx);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, by);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bz);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    norm_squared_a_sum += norm_squared_a_compensation;
    norm_squared_b_sum += norm_squared_b_compensation;

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance matrix: Hᵢⱼ = Σ(aᵢ×bⱼ) - Σaᵢ × Σbⱼ / n.
    nk_f64_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = covariance_x_y - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = covariance_x_z - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = covariance_y_x - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = covariance_y_y - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = covariance_y_z - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = covariance_z_x - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = covariance_z_y - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = covariance_z_z - sum_a_z * sum_b_z * inv_n;

    // Identity-dominant short-circuit: if H_centered is near-diagonal positive-definite,
    // R = I and trace(R * H) = H[0] + H[4] + H[8]. Saves ~500 cycles on aligned inputs.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        if (nk_det3x3_f64_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    // Output rotation matrix and scale=1.0.
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)optimal_rotation[j];
    if (scale) *scale = 1.0;

    // Folded SSD via trace identity - no second pass over the buffers:
    //   SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f64_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f64_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f64_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;
    nk_f64_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0 * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    // Single pass over (a, b) via streaming-stats helper — no deinterleave, no second SSD pass.
    nk_f64_t sum_a[3], sum_b[3], raw_covariance[9], norm_squared_a, norm_squared_b;
    nk_mesh_streaming_stats_f32_skylake_(a, b, n, sum_a, sum_b, raw_covariance, &norm_squared_a, &norm_squared_b);

    nk_f64_t n_f64 = (nk_f64_t)n;
    nk_f64_t inv_n = 1.0 / n_f64;
    nk_f64_t centroid_a_x = sum_a[0] * inv_n, centroid_a_y = sum_a[1] * inv_n, centroid_a_z = sum_a[2] * inv_n;
    nk_f64_t centroid_b_x = sum_b[0] * inv_n, centroid_b_y = sum_b[1] * inv_n, centroid_b_z = sum_b[2] * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    // Centered norms and centered covariance via parallel-axis identity.
    nk_f64_t centered_norm_squared_a = norm_squared_a -
                                       n_f64 * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b -
                                       n_f64 * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;
    nk_f64_t variance_a = centered_norm_squared_a * inv_n;

    nk_f64_t cross_covariance[9];
    cross_covariance[0] = raw_covariance[0] - n_f64 * centroid_a_x * centroid_b_x;
    cross_covariance[1] = raw_covariance[1] - n_f64 * centroid_a_x * centroid_b_y;
    cross_covariance[2] = raw_covariance[2] - n_f64 * centroid_a_x * centroid_b_z;
    cross_covariance[3] = raw_covariance[3] - n_f64 * centroid_a_y * centroid_b_x;
    cross_covariance[4] = raw_covariance[4] - n_f64 * centroid_a_y * centroid_b_y;
    cross_covariance[5] = raw_covariance[5] - n_f64 * centroid_a_y * centroid_b_z;
    cross_covariance[6] = raw_covariance[6] - n_f64 * centroid_a_z * centroid_b_x;
    cross_covariance[7] = raw_covariance[7] - n_f64 * centroid_a_z * centroid_b_y;
    cross_covariance[8] = raw_covariance[8] - n_f64 * centroid_a_z * centroid_b_z;

    // Identity-dominant short-circuit: when H_centered is near-diagonal positive-definite, R = I
    // and trace(R * H) collapses to H[0]+H[4]+H[8]. Also d3 = +1, so trace_ds = sum of diagonal,
    // and applied_scale = trace_ds / (n * variance_a). Skips SVD + two rotation_from_svd calls.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t applied_scale;
    nk_f64_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        applied_scale = trace_rotation_covariance / (n_f64 * variance_a);
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);

        // Scale factor: c = trace(D · S) / (n * variance_a), with reflection sign via d3.
        nk_f64_t det = nk_det3x3_f64_(optimal_rotation);
        nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
        nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_diagonal[0], 1.0, svd_diagonal[4], 1.0, svd_diagonal[8], d3);
        applied_scale = trace_ds / (n_f64 * variance_a);

        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }
    if (scale) *scale = (nk_f32_t)applied_scale;
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];

    // Folded SSD with scale: sum(|| s*R*(a-abar) - (b-bbar) ||^2)
    //    = s²·‖a-ā‖² + ‖b-b̄‖² − 2s·trace(R · H_centered).
    nk_f64_t sum_squared = applied_scale * applied_scale * centered_norm_squared_a + centered_norm_squared_b -
                           2.0 * applied_scale * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_haswell(sum_squared / n_f64);
}

NK_PUBLIC void nk_umeyama_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m512i const gather_idx_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros_f64x8 = _mm512_setzero_pd();

    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;
    __m512d covariance_xx_f64x8 = zeros_f64x8, covariance_xy_f64x8 = zeros_f64x8, covariance_xz_f64x8 = zeros_f64x8;
    __m512d covariance_yx_f64x8 = zeros_f64x8, covariance_yy_f64x8 = zeros_f64x8, covariance_yz_f64x8 = zeros_f64x8;
    __m512d covariance_zx_f64x8 = zeros_f64x8, covariance_zy_f64x8 = zeros_f64x8, covariance_zz_f64x8 = zeros_f64x8;
    __m512d norm_squared_a_f64x8 = zeros_f64x8, norm_squared_b_f64x8 = zeros_f64x8;

    nk_size_t i = 0;
    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f64x8_skylake_(a + i * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + i * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8);
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8);
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        covariance_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, covariance_xx_f64x8),
        covariance_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, covariance_xy_f64x8);
        covariance_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, covariance_xz_f64x8);
        covariance_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, covariance_yx_f64x8),
        covariance_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, covariance_yy_f64x8);
        covariance_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, covariance_yz_f64x8);
        covariance_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, covariance_zx_f64x8),
        covariance_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, covariance_zy_f64x8);
        covariance_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, covariance_zz_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_x_f64x8, a_x_f64x8, norm_squared_a_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_y_f64x8, a_y_f64x8, norm_squared_a_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_z_f64x8, a_z_f64x8, norm_squared_a_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_x_f64x8, b_x_f64x8, norm_squared_b_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_y_f64x8, b_y_f64x8, norm_squared_b_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_z_f64x8, b_z_f64x8, norm_squared_b_f64x8);
    }

    if (i < n) {
        nk_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        nk_f64_t const *a_tail = a + i * 3;
        nk_f64_t const *b_tail = b + i * 3;

        a_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 0, 8);
        a_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 1, 8);
        a_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 2, 8);
        b_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 0, 8);
        b_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 1, 8);
        b_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 2, 8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8);
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8);
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        covariance_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, covariance_xx_f64x8),
        covariance_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, covariance_xy_f64x8);
        covariance_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, covariance_xz_f64x8);
        covariance_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, covariance_yx_f64x8),
        covariance_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, covariance_yy_f64x8);
        covariance_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, covariance_yz_f64x8);
        covariance_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, covariance_zx_f64x8),
        covariance_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, covariance_zy_f64x8);
        covariance_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, covariance_zz_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_x_f64x8, a_x_f64x8, norm_squared_a_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_y_f64x8, a_y_f64x8, norm_squared_a_f64x8);
        norm_squared_a_f64x8 = _mm512_fmadd_pd(a_z_f64x8, a_z_f64x8, norm_squared_a_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_x_f64x8, b_x_f64x8, norm_squared_b_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_y_f64x8, b_y_f64x8, norm_squared_b_f64x8);
        norm_squared_b_f64x8 = _mm512_fmadd_pd(b_z_f64x8, b_z_f64x8, norm_squared_b_f64x8);
        i = n;
    }

    // Reduce centroids, covariance, and variance.
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t sum_a_x = nk_reduce_stable_f64x8_skylake_(sum_a_x_f64x8), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x8_skylake_(sum_a_y_f64x8), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x8_skylake_(sum_a_z_f64x8), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x8_skylake_(sum_b_x_f64x8), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x8_skylake_(sum_b_y_f64x8), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x8_skylake_(sum_b_z_f64x8), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x8_skylake_(covariance_xx_f64x8), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x8_skylake_(covariance_xy_f64x8), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x8_skylake_(covariance_xz_f64x8), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x8_skylake_(covariance_yx_f64x8), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x8_skylake_(covariance_yy_f64x8), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x8_skylake_(covariance_yz_f64x8), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x8_skylake_(covariance_zx_f64x8), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x8_skylake_(covariance_zy_f64x8), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x8_skylake_(covariance_zz_f64x8), covariance_z_z_compensation = 0.0;
    nk_f64_t norm_squared_a_sum = nk_reduce_stable_f64x8_skylake_(norm_squared_a_f64x8),
             norm_squared_a_compensation = 0.0;
    nk_f64_t norm_squared_b_sum = nk_reduce_stable_f64x8_skylake_(norm_squared_b_f64x8),
             norm_squared_b_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax);
        nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay);
        nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx);
        nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by);
        nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx);
        nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by);
        nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx);
        nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by);
        nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx);
        nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by);
        nk_accumulate_product_f64_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ax);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, ay);
        nk_accumulate_square_f64_(&norm_squared_a_sum, &norm_squared_a_compensation, az);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bx);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, by);
        nk_accumulate_square_f64_(&norm_squared_b_sum, &norm_squared_b_compensation, bz);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    norm_squared_a_sum += norm_squared_a_compensation;
    norm_squared_b_sum += norm_squared_b_compensation;

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Centered norm squared via parallel-axis identity (clamped for numerical safety).
    nk_f64_t centered_norm_squared_a = norm_squared_a_sum -
                                       (nk_f64_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f64_t centered_norm_squared_b = norm_squared_b_sum -
                                       (nk_f64_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0) centered_norm_squared_a = 0.0;
    if (centered_norm_squared_b < 0.0) centered_norm_squared_b = 0.0;

    // Compute centered covariance matrix: Hᵢⱼ = Σ(aᵢ×bⱼ) - Σaᵢ × Σbⱼ / n.
    nk_f64_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = covariance_x_y - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = covariance_x_z - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = covariance_y_x - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = covariance_y_y - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = covariance_y_z - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = covariance_z_x - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = covariance_z_y - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = covariance_z_z - sum_a_z * sum_b_z * inv_n;

    // SVD using f64 for full precision
    // Identity-dominant short-circuit: when H_centered is near-diagonal positive-definite,
    // R = I, trace(R * H) = H[0]+H[4]+H[8] (also == trace_ds with d3=+1), and the scale
    // derivation collapses. Skips SVD + two rotation_from_svd calls.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t c;
    nk_f64_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        c = centered_norm_squared_a > 0.0 ? trace_rotation_covariance / centered_norm_squared_a : 0.0;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);

        // Scale factor: c = trace(D · S) / (n * variance(a)), with reflection sign via d3.
        nk_f64_t det = nk_det3x3_f64_(optimal_rotation);
        nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
        nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_diagonal[0], 1.0, svd_diagonal[4], 1.0, svd_diagonal[8], d3);
        c = centered_norm_squared_a > 0.0 ? trace_ds / centered_norm_squared_a : 0.0;

        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }
    if (scale) *scale = c;
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)optimal_rotation[j];

    // Folded SSD with scale: Sum(|| c*R*(a-abar) - (b-bbar) ||^2)
    //   = c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f64_t sum_squared = c * c * centered_norm_squared_a + centered_norm_squared_b -
                           2.0 * c * trace_rotation_covariance;
    if (sum_squared < 0.0) sum_squared = 0.0;
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_rmsd_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    // 15-lane stride-3 layout: mask at the f16 level so the last chunk stays in-bounds.
    __m512 sum_squared_f32x16 = _mm512_setzero_ps();
    nk_size_t index = 0;

    for (; index + 5 <= n; index += 5) {
        __m256i a_f16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(a + index * 3));
        __m256i b_f16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
        __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
        __m512 delta_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_f32x16, delta_f32x16, sum_squared_f32x16);
    }

    if (index < n) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0x7FFF, (nk_u32_t)((n - index) * 3));
        __m256i a_f16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(a + index * 3));
        __m256i b_f16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
        __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
        __m512 delta_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_f32x16, delta_f32x16, sum_squared_f32x16);
    }

    nk_f32_t sum_squared = _mm512_reduce_add_ps(sum_squared_f32x16);
    *result = nk_f32_sqrt_haswell(sum_squared / (nk_f32_t)n);
}

NK_PUBLIC void nk_rmsd_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    // 15-lane stride-3 layout: mask at the bf16 level so the last chunk stays in-bounds.
    __m512 sum_squared_f32x16 = _mm512_setzero_ps();
    nk_size_t index = 0;

    for (; index + 5 <= n; index += 5) {
        __m256i a_bf16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(a + index * 3));
        __m256i b_bf16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
        __m512 b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
        __m512 delta_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_f32x16, delta_f32x16, sum_squared_f32x16);
    }

    if (index < n) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0x7FFF, (nk_u32_t)((n - index) * 3));
        __m256i a_bf16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(a + index * 3));
        __m256i b_bf16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
        __m512 b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
        __m512 delta_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_f32x16, delta_f32x16, sum_squared_f32x16);
    }

    nk_f32_t sum_squared = _mm512_reduce_add_ps(sum_squared_f32x16);
    *result = nk_f32_sqrt_haswell(sum_squared / (nk_f32_t)n);
}

NK_PUBLIC void nk_kabsch_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // 15-lane stride-3 layout: one masked epi16 load + widen gives {a_f32x16, b_f32x16} with
    // channel phase [x,y,z, x,y,z, x,y,z, x,y,z, x,y,z, _] constant across all chunks. The 9
    // H-cells come from three product accumulators a*b, a*rot1(b), a*rot2(b) demuxed per channel.
    __m512i const idx_rotation_1_i32x16 = _mm512_setr_epi32(1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9, 13, 14, 12, 15);
    __m512i const idx_rotation_2_i32x16 = _mm512_setr_epi32(2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10, 14, 12, 13, 15);

    __m512 const zeros_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_f32x16 = zeros_f32x16, sum_b_f32x16 = zeros_f32x16;
    __m512 norm_squared_a_f32x16 = zeros_f32x16, norm_squared_b_f32x16 = zeros_f32x16;
    __m512 product_diagonal_f32x16 = zeros_f32x16;
    __m512 product_rotation_1_f32x16 = zeros_f32x16;
    __m512 product_rotation_2_f32x16 = zeros_f32x16;

    nk_size_t index = 0;
    for (; index + 5 <= n; index += 5) {
        __m256i a_f16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(a + index * 3));
        __m256i b_f16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
        __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
        __m512 b_rotation_1_f32x16 = _mm512_permutexvar_ps(idx_rotation_1_i32x16, b_f32x16);
        __m512 b_rotation_2_f32x16 = _mm512_permutexvar_ps(idx_rotation_2_i32x16, b_f32x16);
        sum_a_f32x16 = _mm512_add_ps(sum_a_f32x16, a_f32x16);
        sum_b_f32x16 = _mm512_add_ps(sum_b_f32x16, b_f32x16);
        norm_squared_a_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, norm_squared_a_f32x16);
        norm_squared_b_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, norm_squared_b_f32x16);
        product_diagonal_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, product_diagonal_f32x16);
        product_rotation_1_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_1_f32x16, product_rotation_1_f32x16);
        product_rotation_2_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_2_f32x16, product_rotation_2_f32x16);
    }

    if (index < n) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0x7FFF, (nk_u32_t)((n - index) * 3));
        __m256i a_f16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(a + index * 3));
        __m256i b_f16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
        __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
        __m512 b_rotation_1_f32x16 = _mm512_permutexvar_ps(idx_rotation_1_i32x16, b_f32x16);
        __m512 b_rotation_2_f32x16 = _mm512_permutexvar_ps(idx_rotation_2_i32x16, b_f32x16);
        sum_a_f32x16 = _mm512_add_ps(sum_a_f32x16, a_f32x16);
        sum_b_f32x16 = _mm512_add_ps(sum_b_f32x16, b_f32x16);
        norm_squared_a_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, norm_squared_a_f32x16);
        norm_squared_b_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, norm_squared_b_f32x16);
        product_diagonal_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, product_diagonal_f32x16);
        product_rotation_1_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_1_f32x16, product_rotation_1_f32x16);
        product_rotation_2_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_2_f32x16, product_rotation_2_f32x16);
    }

    // Per-channel demux via mask-reduce on the fp32 accumulators (lane i carries channel i%3).
    __mmask16 const mask_channel_x_f32 = 0x1249; // lanes {0, 3, 6, 9, 12}
    __mmask16 const mask_channel_y_f32 = 0x2492; // lanes {1, 4, 7, 10, 13}
    __mmask16 const mask_channel_z_f32 = 0x4924; // lanes {2, 5, 8, 11, 14}

    nk_f32_t sum_a_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_a_f32x16);
    nk_f32_t sum_a_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_a_f32x16);
    nk_f32_t sum_a_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_a_f32x16);
    nk_f32_t sum_b_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_b_f32x16);
    nk_f32_t sum_b_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_b_f32x16);
    nk_f32_t sum_b_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_b_f32x16);
    nk_f32_t norm_squared_a = _mm512_reduce_add_ps(norm_squared_a_f32x16);
    nk_f32_t norm_squared_b = _mm512_reduce_add_ps(norm_squared_b_f32x16);

    nk_f32_t covariance_x_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_diagonal_f32x16);
    nk_f32_t covariance_x_y = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_x_z = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_x = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_diagonal_f32x16);
    nk_f32_t covariance_y_z = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_x = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_y = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_z_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_diagonal_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Parallel-axis correction.
    nk_f32_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - (nk_f32_t)n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - (nk_f32_t)n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - (nk_f32_t)n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - (nk_f32_t)n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - (nk_f32_t)n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - (nk_f32_t)n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - (nk_f32_t)n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - (nk_f32_t)n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
    nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
    nk_f32_t optimal_rotation[9];
    nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    if (nk_det3x3_f32_(optimal_rotation) < 0) {
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    // Folded SSD via trace identity:
    //    SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered)
    //    trace(R · H_centered) = Σⱼₖ R[j,k] · H[k,j]  (note transpose on H).
    nk_f32_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;
    nk_f32_t trace_rotation_covariance =
        optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
        optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
        optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
        optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
        optimal_rotation[8] * cross_covariance[8];
    nk_f32_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0f * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_kabsch_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // 15-lane stride-3 layout: one masked epi16 load + widen gives {a_f32x16, b_f32x16} with
    // channel phase [x,y,z, x,y,z, x,y,z, x,y,z, x,y,z, _] constant across all chunks. The 9
    // H-cells come from three product accumulators a*b, a*rot1(b), a*rot2(b) demuxed per channel.
    __m512i const idx_rotation_1_i32x16 = _mm512_setr_epi32(1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9, 13, 14, 12, 15);
    __m512i const idx_rotation_2_i32x16 = _mm512_setr_epi32(2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10, 14, 12, 13, 15);

    __m512 const zeros_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_f32x16 = zeros_f32x16, sum_b_f32x16 = zeros_f32x16;
    __m512 norm_squared_a_f32x16 = zeros_f32x16, norm_squared_b_f32x16 = zeros_f32x16;
    __m512 product_diagonal_f32x16 = zeros_f32x16;
    __m512 product_rotation_1_f32x16 = zeros_f32x16;
    __m512 product_rotation_2_f32x16 = zeros_f32x16;

    nk_size_t index = 0;
    for (; index + 5 <= n; index += 5) {
        __m256i a_bf16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(a + index * 3));
        __m256i b_bf16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
        __m512 b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
        __m512 b_rotation_1_f32x16 = _mm512_permutexvar_ps(idx_rotation_1_i32x16, b_f32x16);
        __m512 b_rotation_2_f32x16 = _mm512_permutexvar_ps(idx_rotation_2_i32x16, b_f32x16);
        sum_a_f32x16 = _mm512_add_ps(sum_a_f32x16, a_f32x16);
        sum_b_f32x16 = _mm512_add_ps(sum_b_f32x16, b_f32x16);
        norm_squared_a_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, norm_squared_a_f32x16);
        norm_squared_b_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, norm_squared_b_f32x16);
        product_diagonal_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, product_diagonal_f32x16);
        product_rotation_1_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_1_f32x16, product_rotation_1_f32x16);
        product_rotation_2_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_2_f32x16, product_rotation_2_f32x16);
    }

    if (index < n) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0x7FFF, (nk_u32_t)((n - index) * 3));
        __m256i a_bf16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(a + index * 3));
        __m256i b_bf16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
        __m512 b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
        __m512 b_rotation_1_f32x16 = _mm512_permutexvar_ps(idx_rotation_1_i32x16, b_f32x16);
        __m512 b_rotation_2_f32x16 = _mm512_permutexvar_ps(idx_rotation_2_i32x16, b_f32x16);
        sum_a_f32x16 = _mm512_add_ps(sum_a_f32x16, a_f32x16);
        sum_b_f32x16 = _mm512_add_ps(sum_b_f32x16, b_f32x16);
        norm_squared_a_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, norm_squared_a_f32x16);
        norm_squared_b_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, norm_squared_b_f32x16);
        product_diagonal_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, product_diagonal_f32x16);
        product_rotation_1_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_1_f32x16, product_rotation_1_f32x16);
        product_rotation_2_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_2_f32x16, product_rotation_2_f32x16);
    }

    // Per-channel demux via mask-reduce on the fp32 accumulators (lane i carries channel i%3).
    __mmask16 const mask_channel_x_f32 = 0x1249; // lanes {0, 3, 6, 9, 12}
    __mmask16 const mask_channel_y_f32 = 0x2492; // lanes {1, 4, 7, 10, 13}
    __mmask16 const mask_channel_z_f32 = 0x4924; // lanes {2, 5, 8, 11, 14}

    nk_f32_t sum_a_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_a_f32x16);
    nk_f32_t sum_a_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_a_f32x16);
    nk_f32_t sum_a_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_a_f32x16);
    nk_f32_t sum_b_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_b_f32x16);
    nk_f32_t sum_b_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_b_f32x16);
    nk_f32_t sum_b_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_b_f32x16);
    nk_f32_t norm_squared_a = _mm512_reduce_add_ps(norm_squared_a_f32x16);
    nk_f32_t norm_squared_b = _mm512_reduce_add_ps(norm_squared_b_f32x16);

    nk_f32_t covariance_x_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_diagonal_f32x16);
    nk_f32_t covariance_x_y = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_x_z = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_x = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_diagonal_f32x16);
    nk_f32_t covariance_y_z = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_x = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_y = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_z_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_diagonal_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Parallel-axis correction.
    nk_f32_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - (nk_f32_t)n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - (nk_f32_t)n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - (nk_f32_t)n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - (nk_f32_t)n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - (nk_f32_t)n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - (nk_f32_t)n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - (nk_f32_t)n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - (nk_f32_t)n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
    nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
    nk_f32_t optimal_rotation[9];
    nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    if (nk_det3x3_f32_(optimal_rotation) < 0) {
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    // Folded SSD via trace identity:
    //    SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered)
    //    trace(R · H_centered) = Σⱼₖ R[j,k] · H[k,j]  (note transpose on H).
    nk_f32_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;
    nk_f32_t trace_rotation_covariance =
        optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
        optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
        optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
        optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
        optimal_rotation[8] * cross_covariance[8];
    nk_f32_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0f * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Same 15-lane streaming-stats pattern as kabsch_f16_skylake; adds the Umeyama scale.
    __m512i const idx_rotation_1_i32x16 = _mm512_setr_epi32(1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9, 13, 14, 12, 15);
    __m512i const idx_rotation_2_i32x16 = _mm512_setr_epi32(2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10, 14, 12, 13, 15);

    __m512 const zeros_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_f32x16 = zeros_f32x16, sum_b_f32x16 = zeros_f32x16;
    __m512 norm_squared_a_f32x16 = zeros_f32x16, norm_squared_b_f32x16 = zeros_f32x16;
    __m512 product_diagonal_f32x16 = zeros_f32x16;
    __m512 product_rotation_1_f32x16 = zeros_f32x16;
    __m512 product_rotation_2_f32x16 = zeros_f32x16;

    nk_size_t index = 0;
    for (; index + 5 <= n; index += 5) {
        __m256i a_f16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(a + index * 3));
        __m256i b_f16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
        __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
        __m512 b_rotation_1_f32x16 = _mm512_permutexvar_ps(idx_rotation_1_i32x16, b_f32x16);
        __m512 b_rotation_2_f32x16 = _mm512_permutexvar_ps(idx_rotation_2_i32x16, b_f32x16);
        sum_a_f32x16 = _mm512_add_ps(sum_a_f32x16, a_f32x16);
        sum_b_f32x16 = _mm512_add_ps(sum_b_f32x16, b_f32x16);
        norm_squared_a_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, norm_squared_a_f32x16);
        norm_squared_b_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, norm_squared_b_f32x16);
        product_diagonal_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, product_diagonal_f32x16);
        product_rotation_1_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_1_f32x16, product_rotation_1_f32x16);
        product_rotation_2_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_2_f32x16, product_rotation_2_f32x16);
    }

    if (index < n) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0x7FFF, (nk_u32_t)((n - index) * 3));
        __m256i a_f16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(a + index * 3));
        __m256i b_f16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
        __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
        __m512 b_rotation_1_f32x16 = _mm512_permutexvar_ps(idx_rotation_1_i32x16, b_f32x16);
        __m512 b_rotation_2_f32x16 = _mm512_permutexvar_ps(idx_rotation_2_i32x16, b_f32x16);
        sum_a_f32x16 = _mm512_add_ps(sum_a_f32x16, a_f32x16);
        sum_b_f32x16 = _mm512_add_ps(sum_b_f32x16, b_f32x16);
        norm_squared_a_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, norm_squared_a_f32x16);
        norm_squared_b_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, norm_squared_b_f32x16);
        product_diagonal_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, product_diagonal_f32x16);
        product_rotation_1_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_1_f32x16, product_rotation_1_f32x16);
        product_rotation_2_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_2_f32x16, product_rotation_2_f32x16);
    }

    __mmask16 const mask_channel_x_f32 = 0x1249;
    __mmask16 const mask_channel_y_f32 = 0x2492;
    __mmask16 const mask_channel_z_f32 = 0x4924;

    nk_f32_t sum_a_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_a_f32x16);
    nk_f32_t sum_a_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_a_f32x16);
    nk_f32_t sum_a_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_a_f32x16);
    nk_f32_t sum_b_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_b_f32x16);
    nk_f32_t sum_b_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_b_f32x16);
    nk_f32_t sum_b_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_b_f32x16);
    nk_f32_t norm_squared_a = _mm512_reduce_add_ps(norm_squared_a_f32x16);
    nk_f32_t norm_squared_b = _mm512_reduce_add_ps(norm_squared_b_f32x16);

    nk_f32_t covariance_x_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_diagonal_f32x16);
    nk_f32_t covariance_x_y = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_x_z = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_x = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_diagonal_f32x16);
    nk_f32_t covariance_y_z = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_x = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_y = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_z_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_diagonal_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    nk_f32_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

    nk_f32_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - (nk_f32_t)n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - (nk_f32_t)n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - (nk_f32_t)n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - (nk_f32_t)n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - (nk_f32_t)n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - (nk_f32_t)n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - (nk_f32_t)n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - (nk_f32_t)n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
    nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
    nk_f32_t optimal_rotation[9];
    nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);

    // Scale factor: c = trace(D · S) / ‖a-ā‖², with reflection sign via d3.
    nk_f32_t det = nk_det3x3_f32_(optimal_rotation);
    nk_f32_t d3 = det < 0.0f ? -1.0f : 1.0f;
    nk_f32_t trace_ds = nk_sum_three_products_f32_(svd_diagonal[0], 1.0f, svd_diagonal[4], 1.0f, svd_diagonal[8], d3);
    nk_f32_t c = centered_norm_squared_a > 0.0f ? trace_ds / centered_norm_squared_a : 0.0f;
    if (scale) *scale = c;

    if (det < 0.0f) {
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    // Folded SSD with scale:
    //    SSD = c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f32_t trace_rotation_covariance =
        optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
        optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
        optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
        optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
        optimal_rotation[8] * cross_covariance[8];
    nk_f32_t sum_squared = c * c * centered_norm_squared_a + centered_norm_squared_b -
                           2.0f * c * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Same 15-lane streaming-stats pattern as kabsch_bf16_skylake; adds the Umeyama scale.
    __m512i const idx_rotation_1_i32x16 = _mm512_setr_epi32(1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9, 13, 14, 12, 15);
    __m512i const idx_rotation_2_i32x16 = _mm512_setr_epi32(2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10, 14, 12, 13, 15);

    __m512 const zeros_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_f32x16 = zeros_f32x16, sum_b_f32x16 = zeros_f32x16;
    __m512 norm_squared_a_f32x16 = zeros_f32x16, norm_squared_b_f32x16 = zeros_f32x16;
    __m512 product_diagonal_f32x16 = zeros_f32x16;
    __m512 product_rotation_1_f32x16 = zeros_f32x16;
    __m512 product_rotation_2_f32x16 = zeros_f32x16;

    nk_size_t index = 0;
    for (; index + 5 <= n; index += 5) {
        __m256i a_bf16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(a + index * 3));
        __m256i b_bf16x16 = _mm256_maskz_loadu_epi16(0x7FFF, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
        __m512 b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
        __m512 b_rotation_1_f32x16 = _mm512_permutexvar_ps(idx_rotation_1_i32x16, b_f32x16);
        __m512 b_rotation_2_f32x16 = _mm512_permutexvar_ps(idx_rotation_2_i32x16, b_f32x16);
        sum_a_f32x16 = _mm512_add_ps(sum_a_f32x16, a_f32x16);
        sum_b_f32x16 = _mm512_add_ps(sum_b_f32x16, b_f32x16);
        norm_squared_a_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, norm_squared_a_f32x16);
        norm_squared_b_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, norm_squared_b_f32x16);
        product_diagonal_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, product_diagonal_f32x16);
        product_rotation_1_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_1_f32x16, product_rotation_1_f32x16);
        product_rotation_2_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_2_f32x16, product_rotation_2_f32x16);
    }

    if (index < n) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0x7FFF, (nk_u32_t)((n - index) * 3));
        __m256i a_bf16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(a + index * 3));
        __m256i b_bf16x16 = _mm256_maskz_loadu_epi16(tail_mask, (__m256i const *)(b + index * 3));
        __m512 a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
        __m512 b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
        __m512 b_rotation_1_f32x16 = _mm512_permutexvar_ps(idx_rotation_1_i32x16, b_f32x16);
        __m512 b_rotation_2_f32x16 = _mm512_permutexvar_ps(idx_rotation_2_i32x16, b_f32x16);
        sum_a_f32x16 = _mm512_add_ps(sum_a_f32x16, a_f32x16);
        sum_b_f32x16 = _mm512_add_ps(sum_b_f32x16, b_f32x16);
        norm_squared_a_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, norm_squared_a_f32x16);
        norm_squared_b_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, norm_squared_b_f32x16);
        product_diagonal_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, product_diagonal_f32x16);
        product_rotation_1_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_1_f32x16, product_rotation_1_f32x16);
        product_rotation_2_f32x16 = _mm512_fmadd_ps(a_f32x16, b_rotation_2_f32x16, product_rotation_2_f32x16);
    }

    __mmask16 const mask_channel_x_f32 = 0x1249;
    __mmask16 const mask_channel_y_f32 = 0x2492;
    __mmask16 const mask_channel_z_f32 = 0x4924;

    nk_f32_t sum_a_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_a_f32x16);
    nk_f32_t sum_a_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_a_f32x16);
    nk_f32_t sum_a_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_a_f32x16);
    nk_f32_t sum_b_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, sum_b_f32x16);
    nk_f32_t sum_b_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, sum_b_f32x16);
    nk_f32_t sum_b_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, sum_b_f32x16);
    nk_f32_t norm_squared_a = _mm512_reduce_add_ps(norm_squared_a_f32x16);
    nk_f32_t norm_squared_b = _mm512_reduce_add_ps(norm_squared_b_f32x16);

    nk_f32_t covariance_x_x = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_diagonal_f32x16);
    nk_f32_t covariance_x_y = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_x_z = _mm512_mask_reduce_add_ps(mask_channel_x_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_x = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_y_y = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_diagonal_f32x16);
    nk_f32_t covariance_y_z = _mm512_mask_reduce_add_ps(mask_channel_y_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_x = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_1_f32x16);
    nk_f32_t covariance_z_y = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_rotation_2_f32x16);
    nk_f32_t covariance_z_z = _mm512_mask_reduce_add_ps(mask_channel_z_f32, product_diagonal_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    nk_f32_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

    nk_f32_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - (nk_f32_t)n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - (nk_f32_t)n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - (nk_f32_t)n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - (nk_f32_t)n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - (nk_f32_t)n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - (nk_f32_t)n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - (nk_f32_t)n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - (nk_f32_t)n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - (nk_f32_t)n * centroid_a_z * centroid_b_z;

    nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
    nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);
    nk_f32_t optimal_rotation[9];
    nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);

    // Scale factor: c = trace(D · S) / ‖a-ā‖², with reflection sign via d3.
    nk_f32_t det = nk_det3x3_f32_(optimal_rotation);
    nk_f32_t d3 = det < 0.0f ? -1.0f : 1.0f;
    nk_f32_t trace_ds = nk_sum_three_products_f32_(svd_diagonal[0], 1.0f, svd_diagonal[4], 1.0f, svd_diagonal[8], d3);
    nk_f32_t c = centered_norm_squared_a > 0.0f ? trace_ds / centered_norm_squared_a : 0.0f;
    if (scale) *scale = c;

    if (det < 0.0f) {
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, optimal_rotation);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    // Folded SSD with scale:
    //    SSD = c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f32_t trace_rotation_covariance =
        optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
        optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
        optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
        optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
        optimal_rotation[8] * cross_covariance[8];
    nk_f32_t sum_squared = c * c * centered_norm_squared_a + centered_norm_squared_b -
                           2.0f * c * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

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
#endif // NK_MESH_SKYLAKE_H
