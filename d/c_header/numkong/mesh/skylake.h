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
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm512_fmadd_ps             VFMADD132PS (ZMM, ZMM, ZMM)     4cy         0.5/cy      p05
 *      _mm512_permutexvar_ps       VPERMPS (ZMM, ZMM, ZMM)         3cy         1/cy        p5
 *      _mm512_permutex2var_ps      VPERMT2PS (ZMM, ZMM, ZMM)       3cy         1/cy        p5
 *      _mm512_extractf32x8_ps      VEXTRACTF32X8 (YMM, ZMM, I8)    3cy         1/cy        p5
 *
 *  Point cloud operations use VPERMT2PS for stride-3 deinterleaving of xyz coordinates, avoiding
 *  expensive gather instructions. This achieves ~1.8x speedup over scalar deinterleaving. Dual FMA
 *  accumulators on Skylake-X server chips hide the 4cy latency for centroid and covariance computation.
 */
#ifndef NK_MESH_SKYLAKE_H
#define NK_MESH_SKYLAKE_H

#if NK_TARGET_X86_
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

NK_INTERNAL void nk_rotation_from_svd_f64_skylake_(nk_f64_t const *svd_u, nk_f64_t const *svd_v, nk_f64_t *rotation) {
    nk_rotation_from_svd_f64_serial_(svd_u, svd_v, rotation);
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

/*  Compute sum of squared distances after applying rotation (and optional scale).
 *  Used by kabsch (scale=1.0) and umeyama (scale=computed_scale).
 *  Returns sum_squared, caller computes √(sum_squared / n).
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f32_skylake_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                     nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                     nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                     nk_f64_t centroid_b_z) {
    __m512d scaled_rotation_x_x_f64x8 = _mm512_set1_pd(scale * r[0]);
    __m512d scaled_rotation_x_y_f64x8 = _mm512_set1_pd(scale * r[1]);
    __m512d scaled_rotation_x_z_f64x8 = _mm512_set1_pd(scale * r[2]);
    __m512d scaled_rotation_y_x_f64x8 = _mm512_set1_pd(scale * r[3]);
    __m512d scaled_rotation_y_y_f64x8 = _mm512_set1_pd(scale * r[4]);
    __m512d scaled_rotation_y_z_f64x8 = _mm512_set1_pd(scale * r[5]);
    __m512d scaled_rotation_z_x_f64x8 = _mm512_set1_pd(scale * r[6]);
    __m512d scaled_rotation_z_y_f64x8 = _mm512_set1_pd(scale * r[7]);
    __m512d scaled_rotation_z_z_f64x8 = _mm512_set1_pd(scale * r[8]);
    __m512d centroid_a_x_f64x8 = _mm512_set1_pd(centroid_a_x), centroid_a_y_f64x8 = _mm512_set1_pd(centroid_a_y);
    __m512d centroid_a_z_f64x8 = _mm512_set1_pd(centroid_a_z), centroid_b_x_f64x8 = _mm512_set1_pd(centroid_b_x);
    __m512d centroid_b_y_f64x8 = _mm512_set1_pd(centroid_b_y), centroid_b_z_f64x8 = _mm512_set1_pd(centroid_b_z);
    __m512d sum_squared_f64x8 = _mm512_setzero_pd();
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t index = 0;

    for (; index + 16 <= n; index += 16) {
        nk_deinterleave_f32x16_skylake_(a + index * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16),
            nk_deinterleave_f32x16_skylake_(b + index * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);
        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        __m512d centered_a_x_lower_f64x8 = _mm512_sub_pd(a_x_lower_f64x8, centroid_a_x_f64x8);
        __m512d centered_a_x_upper_f64x8 = _mm512_sub_pd(a_x_upper_f64x8, centroid_a_x_f64x8);
        __m512d centered_a_y_lower_f64x8 = _mm512_sub_pd(a_y_lower_f64x8, centroid_a_y_f64x8);
        __m512d centered_a_y_upper_f64x8 = _mm512_sub_pd(a_y_upper_f64x8, centroid_a_y_f64x8);
        __m512d centered_a_z_lower_f64x8 = _mm512_sub_pd(a_z_lower_f64x8, centroid_a_z_f64x8);
        __m512d centered_a_z_upper_f64x8 = _mm512_sub_pd(a_z_upper_f64x8, centroid_a_z_f64x8);
        __m512d centered_b_x_lower_f64x8 = _mm512_sub_pd(b_x_lower_f64x8, centroid_b_x_f64x8);
        __m512d centered_b_x_upper_f64x8 = _mm512_sub_pd(b_x_upper_f64x8, centroid_b_x_f64x8);
        __m512d centered_b_y_lower_f64x8 = _mm512_sub_pd(b_y_lower_f64x8, centroid_b_y_f64x8);
        __m512d centered_b_y_upper_f64x8 = _mm512_sub_pd(b_y_upper_f64x8, centroid_b_y_f64x8);
        __m512d centered_b_z_lower_f64x8 = _mm512_sub_pd(b_z_lower_f64x8, centroid_b_z_f64x8);
        __m512d centered_b_z_upper_f64x8 = _mm512_sub_pd(b_z_upper_f64x8, centroid_b_z_f64x8);

        __m512d rotated_a_x_lower_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_x_z_f64x8, centered_a_z_lower_f64x8,
            _mm512_fmadd_pd(scaled_rotation_x_y_f64x8, centered_a_y_lower_f64x8,
                            _mm512_mul_pd(scaled_rotation_x_x_f64x8, centered_a_x_lower_f64x8)));
        __m512d rotated_a_x_upper_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_x_z_f64x8, centered_a_z_upper_f64x8,
            _mm512_fmadd_pd(scaled_rotation_x_y_f64x8, centered_a_y_upper_f64x8,
                            _mm512_mul_pd(scaled_rotation_x_x_f64x8, centered_a_x_upper_f64x8)));
        __m512d rotated_a_y_lower_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_y_z_f64x8, centered_a_z_lower_f64x8,
            _mm512_fmadd_pd(scaled_rotation_y_y_f64x8, centered_a_y_lower_f64x8,
                            _mm512_mul_pd(scaled_rotation_y_x_f64x8, centered_a_x_lower_f64x8)));
        __m512d rotated_a_y_upper_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_y_z_f64x8, centered_a_z_upper_f64x8,
            _mm512_fmadd_pd(scaled_rotation_y_y_f64x8, centered_a_y_upper_f64x8,
                            _mm512_mul_pd(scaled_rotation_y_x_f64x8, centered_a_x_upper_f64x8)));
        __m512d rotated_a_z_lower_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_z_z_f64x8, centered_a_z_lower_f64x8,
            _mm512_fmadd_pd(scaled_rotation_z_y_f64x8, centered_a_y_lower_f64x8,
                            _mm512_mul_pd(scaled_rotation_z_x_f64x8, centered_a_x_lower_f64x8)));
        __m512d rotated_a_z_upper_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_z_z_f64x8, centered_a_z_upper_f64x8,
            _mm512_fmadd_pd(scaled_rotation_z_y_f64x8, centered_a_y_upper_f64x8,
                            _mm512_mul_pd(scaled_rotation_z_x_f64x8, centered_a_x_upper_f64x8)));

        __m512d delta_x_lower_f64x8 = _mm512_sub_pd(rotated_a_x_lower_f64x8, centered_b_x_lower_f64x8);
        __m512d delta_x_upper_f64x8 = _mm512_sub_pd(rotated_a_x_upper_f64x8, centered_b_x_upper_f64x8);
        __m512d delta_y_lower_f64x8 = _mm512_sub_pd(rotated_a_y_lower_f64x8, centered_b_y_lower_f64x8);
        __m512d delta_y_upper_f64x8 = _mm512_sub_pd(rotated_a_y_upper_f64x8, centered_b_y_upper_f64x8);
        __m512d delta_z_lower_f64x8 = _mm512_sub_pd(rotated_a_z_lower_f64x8, centered_b_z_lower_f64x8);
        __m512d delta_z_upper_f64x8 = _mm512_sub_pd(rotated_a_z_upper_f64x8, centered_b_z_upper_f64x8);

        __m512d batch_sum_squared_f64x8 = _mm512_add_pd(_mm512_mul_pd(delta_x_lower_f64x8, delta_x_lower_f64x8),
                                                        _mm512_mul_pd(delta_x_upper_f64x8, delta_x_upper_f64x8));
        batch_sum_squared_f64x8 = _mm512_fmadd_pd(delta_y_lower_f64x8, delta_y_lower_f64x8, batch_sum_squared_f64x8);
        batch_sum_squared_f64x8 = _mm512_fmadd_pd(delta_y_upper_f64x8, delta_y_upper_f64x8, batch_sum_squared_f64x8);
        batch_sum_squared_f64x8 = _mm512_fmadd_pd(delta_z_lower_f64x8, delta_z_lower_f64x8, batch_sum_squared_f64x8);
        batch_sum_squared_f64x8 = _mm512_fmadd_pd(delta_z_upper_f64x8, delta_z_upper_f64x8, batch_sum_squared_f64x8);
        sum_squared_f64x8 = _mm512_add_pd(sum_squared_f64x8, batch_sum_squared_f64x8);
    }

    nk_f64_t sum_squared = _mm512_reduce_add_pd(sum_squared_f64x8);
    for (; index < n; ++index) {
        nk_f64_t centered_a_x = (nk_f64_t)a[index * 3 + 0] - centroid_a_x;
        nk_f64_t centered_a_y = (nk_f64_t)a[index * 3 + 1] - centroid_a_y;
        nk_f64_t centered_a_z = (nk_f64_t)a[index * 3 + 2] - centroid_a_z;
        nk_f64_t centered_b_x = (nk_f64_t)b[index * 3 + 0] - centroid_b_x;
        nk_f64_t centered_b_y = (nk_f64_t)b[index * 3 + 1] - centroid_b_y;
        nk_f64_t centered_b_z = (nk_f64_t)b[index * 3 + 2] - centroid_b_z;
        nk_f64_t rotated_a_x = scale * (r[0] * centered_a_x + r[1] * centered_a_y + r[2] * centered_a_z);
        nk_f64_t rotated_a_y = scale * (r[3] * centered_a_x + r[4] * centered_a_y + r[5] * centered_a_z);
        nk_f64_t rotated_a_z = scale * (r[6] * centered_a_x + r[7] * centered_a_y + r[8] * centered_a_z);
        nk_f64_t delta_x = rotated_a_x - centered_b_x, delta_y = rotated_a_y - centered_b_y,
                 delta_z = rotated_a_z - centered_b_z;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    return sum_squared;
}

/*  Compute sum of squared distances for f64 after applying rotation (and optional scale).
 *  Rotation matrix, scale and data are all f64 for full precision.
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_skylake_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                     nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                     nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                     nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m512d scaled_rotation_x_x_f64x8 = _mm512_set1_pd(scale * r[0]);
    __m512d scaled_rotation_x_y_f64x8 = _mm512_set1_pd(scale * r[1]);
    __m512d scaled_rotation_x_z_f64x8 = _mm512_set1_pd(scale * r[2]);
    __m512d scaled_rotation_y_x_f64x8 = _mm512_set1_pd(scale * r[3]);
    __m512d scaled_rotation_y_y_f64x8 = _mm512_set1_pd(scale * r[4]);
    __m512d scaled_rotation_y_z_f64x8 = _mm512_set1_pd(scale * r[5]);
    __m512d scaled_rotation_z_x_f64x8 = _mm512_set1_pd(scale * r[6]);
    __m512d scaled_rotation_z_y_f64x8 = _mm512_set1_pd(scale * r[7]);
    __m512d scaled_rotation_z_z_f64x8 = _mm512_set1_pd(scale * r[8]);

    // Broadcast centroids
    __m512d centroid_a_x_f64x8 = _mm512_set1_pd(centroid_a_x);
    __m512d centroid_a_y_f64x8 = _mm512_set1_pd(centroid_a_y);
    __m512d centroid_a_z_f64x8 = _mm512_set1_pd(centroid_a_z);
    __m512d centroid_b_x_f64x8 = _mm512_set1_pd(centroid_b_x);
    __m512d centroid_b_y_f64x8 = _mm512_set1_pd(centroid_b_y);
    __m512d centroid_b_z_f64x8 = _mm512_set1_pd(centroid_b_z);

    __m512d sum_squared_f64x8 = _mm512_setzero_pd();
    __m512d sum_squared_compensation_f64x8 = _mm512_setzero_pd();
    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;
    nk_size_t j = 0;

    for (; j + 8 <= n; j += 8) {
        nk_deinterleave_f64x8_skylake_(a + j * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + j * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        // Center points
        __m512d pa_x_f64x8 = _mm512_sub_pd(a_x_f64x8, centroid_a_x_f64x8);
        __m512d pa_y_f64x8 = _mm512_sub_pd(a_y_f64x8, centroid_a_y_f64x8);
        __m512d pa_z_f64x8 = _mm512_sub_pd(a_z_f64x8, centroid_a_z_f64x8);
        __m512d pb_x_f64x8 = _mm512_sub_pd(b_x_f64x8, centroid_b_x_f64x8);
        __m512d pb_y_f64x8 = _mm512_sub_pd(b_y_f64x8, centroid_b_y_f64x8);
        __m512d pb_z_f64x8 = _mm512_sub_pd(b_z_f64x8, centroid_b_z_f64x8);

        // Rotate and scale: ra = scale * R * pa
        __m512d ra_x_f64x8 = _mm512_fmadd_pd(scaled_rotation_x_z_f64x8, pa_z_f64x8,
                                             _mm512_fmadd_pd(scaled_rotation_x_y_f64x8, pa_y_f64x8,
                                                             _mm512_mul_pd(scaled_rotation_x_x_f64x8, pa_x_f64x8)));
        __m512d ra_y_f64x8 = _mm512_fmadd_pd(scaled_rotation_y_z_f64x8, pa_z_f64x8,
                                             _mm512_fmadd_pd(scaled_rotation_y_y_f64x8, pa_y_f64x8,
                                                             _mm512_mul_pd(scaled_rotation_y_x_f64x8, pa_x_f64x8)));
        __m512d ra_z_f64x8 = _mm512_fmadd_pd(scaled_rotation_z_z_f64x8, pa_z_f64x8,
                                             _mm512_fmadd_pd(scaled_rotation_z_y_f64x8, pa_y_f64x8,
                                                             _mm512_mul_pd(scaled_rotation_z_x_f64x8, pa_x_f64x8)));

        // Delta and accumulate
        __m512d delta_x_f64x8 = _mm512_sub_pd(ra_x_f64x8, pb_x_f64x8);
        __m512d delta_y_f64x8 = _mm512_sub_pd(ra_y_f64x8, pb_y_f64x8);
        __m512d delta_z_f64x8 = _mm512_sub_pd(ra_z_f64x8, pb_z_f64x8);

        nk_accumulate_square_f64x8_skylake_(&sum_squared_f64x8, &sum_squared_compensation_f64x8, delta_x_f64x8);
        nk_accumulate_square_f64x8_skylake_(&sum_squared_f64x8, &sum_squared_compensation_f64x8, delta_y_f64x8);
        nk_accumulate_square_f64x8_skylake_(&sum_squared_f64x8, &sum_squared_compensation_f64x8, delta_z_f64x8);
    }

    nk_f64_t sum_squared = nk_dot_stable_sum_f64x8_skylake_(sum_squared_f64x8, sum_squared_compensation_f64x8);
    nk_f64_t sum_squared_compensation = 0.0;

    // Scalar tail
    for (; j < n; ++j) {
        nk_f64_t pa_x = a[j * 3 + 0] - centroid_a_x;
        nk_f64_t pa_y = a[j * 3 + 1] - centroid_a_y;
        nk_f64_t pa_z = a[j * 3 + 2] - centroid_a_z;
        nk_f64_t pb_x = b[j * 3 + 0] - centroid_b_x;
        nk_f64_t pb_y = b[j * 3 + 1] - centroid_b_y;
        nk_f64_t pb_z = b[j * 3 + 2] - centroid_b_z;

        nk_f64_t ra_x = scale * (r[0] * pa_x + r[1] * pa_y + r[2] * pa_z);
        nk_f64_t ra_y = scale * (r[3] * pa_x + r[4] * pa_y + r[5] * pa_z);
        nk_f64_t ra_z = scale * (r[6] * pa_x + r[7] * pa_y + r[8] * pa_z);

        nk_f64_t delta_x = ra_x - pb_x;
        nk_f64_t delta_y = ra_y - pb_y;
        nk_f64_t delta_z = ra_z - pb_z;
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_x);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_y);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_z);
    }

    return sum_squared + sum_squared_compensation;
}

/*  Compute sum of squared distances for f16 data after applying rotation (and optional scale).
 *  Loads f16, converts to f32 for computation. Rotation matrix, scale, and centroids are f32.
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_f16_skylake_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                     nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                     nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                     nk_f32_t centroid_b_z) {
    __m512 scaled_rotation_x_x_f32x16 = _mm512_set1_ps(scale * r[0]);
    __m512 scaled_rotation_x_y_f32x16 = _mm512_set1_ps(scale * r[1]);
    __m512 scaled_rotation_x_z_f32x16 = _mm512_set1_ps(scale * r[2]);
    __m512 scaled_rotation_y_x_f32x16 = _mm512_set1_ps(scale * r[3]);
    __m512 scaled_rotation_y_y_f32x16 = _mm512_set1_ps(scale * r[4]);
    __m512 scaled_rotation_y_z_f32x16 = _mm512_set1_ps(scale * r[5]);
    __m512 scaled_rotation_z_x_f32x16 = _mm512_set1_ps(scale * r[6]);
    __m512 scaled_rotation_z_y_f32x16 = _mm512_set1_ps(scale * r[7]);
    __m512 scaled_rotation_z_z_f32x16 = _mm512_set1_ps(scale * r[8]);

    __m512 centroid_a_x_f32x16 = _mm512_set1_ps(centroid_a_x);
    __m512 centroid_a_y_f32x16 = _mm512_set1_ps(centroid_a_y);
    __m512 centroid_a_z_f32x16 = _mm512_set1_ps(centroid_a_z);
    __m512 centroid_b_x_f32x16 = _mm512_set1_ps(centroid_b_x);
    __m512 centroid_b_y_f32x16 = _mm512_set1_ps(centroid_b_y);
    __m512 centroid_b_z_f32x16 = _mm512_set1_ps(centroid_b_z);

    __m512 sum_squared_f32x16 = _mm512_setzero_ps();
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t j = 0;

    for (; j + 16 <= n; j += 16) {
        nk_deinterleave_f16x16_to_f32x16_skylake_(a + j * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f16x16_to_f32x16_skylake_(b + j * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        __m512 pa_x_f32x16 = _mm512_sub_ps(a_x_f32x16, centroid_a_x_f32x16);
        __m512 pa_y_f32x16 = _mm512_sub_ps(a_y_f32x16, centroid_a_y_f32x16);
        __m512 pa_z_f32x16 = _mm512_sub_ps(a_z_f32x16, centroid_a_z_f32x16);
        __m512 pb_x_f32x16 = _mm512_sub_ps(b_x_f32x16, centroid_b_x_f32x16);
        __m512 pb_y_f32x16 = _mm512_sub_ps(b_y_f32x16, centroid_b_y_f32x16);
        __m512 pb_z_f32x16 = _mm512_sub_ps(b_z_f32x16, centroid_b_z_f32x16);

        __m512 ra_x_f32x16 = _mm512_fmadd_ps(scaled_rotation_x_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_x_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_x_x_f32x16, pa_x_f32x16)));
        __m512 ra_y_f32x16 = _mm512_fmadd_ps(scaled_rotation_y_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_y_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_y_x_f32x16, pa_x_f32x16)));
        __m512 ra_z_f32x16 = _mm512_fmadd_ps(scaled_rotation_z_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_z_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_z_x_f32x16, pa_x_f32x16)));

        __m512 delta_x_f32x16 = _mm512_sub_ps(ra_x_f32x16, pb_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(ra_y_f32x16, pb_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(ra_z_f32x16, pb_z_f32x16);

        sum_squared_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_f32x16);
    }

    // Tail: deinterleave remaining points into zero-initialized vectors
    if (j < n) {
        nk_size_t tail = n - j;
        nk_deinterleave_f16_tail_to_f32x16_skylake_(a + j * 3, tail, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f16_tail_to_f32x16_skylake_(b + j * 3, tail, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        __m512 pa_x_f32x16 = _mm512_sub_ps(a_x_f32x16, centroid_a_x_f32x16);
        __m512 pa_y_f32x16 = _mm512_sub_ps(a_y_f32x16, centroid_a_y_f32x16);
        __m512 pa_z_f32x16 = _mm512_sub_ps(a_z_f32x16, centroid_a_z_f32x16);
        __m512 pb_x_f32x16 = _mm512_sub_ps(b_x_f32x16, centroid_b_x_f32x16);
        __m512 pb_y_f32x16 = _mm512_sub_ps(b_y_f32x16, centroid_b_y_f32x16);
        __m512 pb_z_f32x16 = _mm512_sub_ps(b_z_f32x16, centroid_b_z_f32x16);

        __m512 ra_x_f32x16 = _mm512_fmadd_ps(scaled_rotation_x_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_x_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_x_x_f32x16, pa_x_f32x16)));
        __m512 ra_y_f32x16 = _mm512_fmadd_ps(scaled_rotation_y_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_y_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_y_x_f32x16, pa_x_f32x16)));
        __m512 ra_z_f32x16 = _mm512_fmadd_ps(scaled_rotation_z_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_z_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_z_x_f32x16, pa_x_f32x16)));

        __m512 delta_x_f32x16 = _mm512_sub_ps(ra_x_f32x16, pb_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(ra_y_f32x16, pb_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(ra_z_f32x16, pb_z_f32x16);

        sum_squared_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_f32x16);
    }

    return _mm512_reduce_add_ps(sum_squared_f32x16);
}

/*  Compute sum of squared distances for bf16 data after applying rotation (and optional scale).
 *  Loads bf16, converts to f32 for computation. Rotation matrix, scale, and centroids are f32.
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_bf16_skylake_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                      nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                      nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                      nk_f32_t centroid_b_z) {
    __m512 scaled_rotation_x_x_f32x16 = _mm512_set1_ps(scale * r[0]);
    __m512 scaled_rotation_x_y_f32x16 = _mm512_set1_ps(scale * r[1]);
    __m512 scaled_rotation_x_z_f32x16 = _mm512_set1_ps(scale * r[2]);
    __m512 scaled_rotation_y_x_f32x16 = _mm512_set1_ps(scale * r[3]);
    __m512 scaled_rotation_y_y_f32x16 = _mm512_set1_ps(scale * r[4]);
    __m512 scaled_rotation_y_z_f32x16 = _mm512_set1_ps(scale * r[5]);
    __m512 scaled_rotation_z_x_f32x16 = _mm512_set1_ps(scale * r[6]);
    __m512 scaled_rotation_z_y_f32x16 = _mm512_set1_ps(scale * r[7]);
    __m512 scaled_rotation_z_z_f32x16 = _mm512_set1_ps(scale * r[8]);

    __m512 centroid_a_x_f32x16 = _mm512_set1_ps(centroid_a_x);
    __m512 centroid_a_y_f32x16 = _mm512_set1_ps(centroid_a_y);
    __m512 centroid_a_z_f32x16 = _mm512_set1_ps(centroid_a_z);
    __m512 centroid_b_x_f32x16 = _mm512_set1_ps(centroid_b_x);
    __m512 centroid_b_y_f32x16 = _mm512_set1_ps(centroid_b_y);
    __m512 centroid_b_z_f32x16 = _mm512_set1_ps(centroid_b_z);

    __m512 sum_squared_f32x16 = _mm512_setzero_ps();
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t j = 0;

    for (; j + 16 <= n; j += 16) {
        nk_deinterleave_bf16x16_to_f32x16_skylake_(a + j * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_bf16x16_to_f32x16_skylake_(b + j * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        __m512 pa_x_f32x16 = _mm512_sub_ps(a_x_f32x16, centroid_a_x_f32x16);
        __m512 pa_y_f32x16 = _mm512_sub_ps(a_y_f32x16, centroid_a_y_f32x16);
        __m512 pa_z_f32x16 = _mm512_sub_ps(a_z_f32x16, centroid_a_z_f32x16);
        __m512 pb_x_f32x16 = _mm512_sub_ps(b_x_f32x16, centroid_b_x_f32x16);
        __m512 pb_y_f32x16 = _mm512_sub_ps(b_y_f32x16, centroid_b_y_f32x16);
        __m512 pb_z_f32x16 = _mm512_sub_ps(b_z_f32x16, centroid_b_z_f32x16);

        __m512 ra_x_f32x16 = _mm512_fmadd_ps(scaled_rotation_x_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_x_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_x_x_f32x16, pa_x_f32x16)));
        __m512 ra_y_f32x16 = _mm512_fmadd_ps(scaled_rotation_y_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_y_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_y_x_f32x16, pa_x_f32x16)));
        __m512 ra_z_f32x16 = _mm512_fmadd_ps(scaled_rotation_z_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_z_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_z_x_f32x16, pa_x_f32x16)));

        __m512 delta_x_f32x16 = _mm512_sub_ps(ra_x_f32x16, pb_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(ra_y_f32x16, pb_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(ra_z_f32x16, pb_z_f32x16);

        sum_squared_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_f32x16);
    }

    // Tail: deinterleave remaining points into zero-initialized vectors
    if (j < n) {
        nk_size_t tail = n - j;
        nk_deinterleave_bf16_tail_to_f32x16_skylake_(a + j * 3, tail, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_bf16_tail_to_f32x16_skylake_(b + j * 3, tail, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        __m512 pa_x_f32x16 = _mm512_sub_ps(a_x_f32x16, centroid_a_x_f32x16);
        __m512 pa_y_f32x16 = _mm512_sub_ps(a_y_f32x16, centroid_a_y_f32x16);
        __m512 pa_z_f32x16 = _mm512_sub_ps(a_z_f32x16, centroid_a_z_f32x16);
        __m512 pb_x_f32x16 = _mm512_sub_ps(b_x_f32x16, centroid_b_x_f32x16);
        __m512 pb_y_f32x16 = _mm512_sub_ps(b_y_f32x16, centroid_b_y_f32x16);
        __m512 pb_z_f32x16 = _mm512_sub_ps(b_z_f32x16, centroid_b_z_f32x16);

        __m512 ra_x_f32x16 = _mm512_fmadd_ps(scaled_rotation_x_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_x_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_x_x_f32x16, pa_x_f32x16)));
        __m512 ra_y_f32x16 = _mm512_fmadd_ps(scaled_rotation_y_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_y_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_y_x_f32x16, pa_x_f32x16)));
        __m512 ra_z_f32x16 = _mm512_fmadd_ps(scaled_rotation_z_z_f32x16, pa_z_f32x16,
                                             _mm512_fmadd_ps(scaled_rotation_z_y_f32x16, pa_y_f32x16,
                                                             _mm512_mul_ps(scaled_rotation_z_x_f32x16, pa_x_f32x16)));

        __m512 delta_x_f32x16 = _mm512_sub_ps(ra_x_f32x16, pb_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(ra_y_f32x16, pb_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(ra_z_f32x16, pb_z_f32x16);

        sum_squared_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_f32x16);
    }

    return _mm512_reduce_add_ps(sum_squared_f32x16);
}

NK_PUBLIC void nk_rmsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;

    // Fused single-pass: centroids + squared differences in f64, using the identity:
    //   RMSD = √(E[(a-b)²] - (ā - b̄)²)
    __m512d const zeros_f64x8 = _mm512_setzero_pd();
    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;
    __m512d sum_squared_x_f64x8 = zeros_f64x8, sum_squared_y_f64x8 = zeros_f64x8, sum_squared_z_f64x8 = zeros_f64x8;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t i = 0;

    // Main loop with 2x unrolling (32 points per iteration)
    for (; i + 32 <= n; i += 32) {
        // Iteration 0: points i..i+15
        nk_deinterleave_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);
        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, _mm512_add_pd(a_x_lower_f64x8, a_x_upper_f64x8));
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, _mm512_add_pd(a_y_lower_f64x8, a_y_upper_f64x8));
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, _mm512_add_pd(a_z_lower_f64x8, a_z_upper_f64x8));
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, _mm512_add_pd(b_x_lower_f64x8, b_x_upper_f64x8));
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, _mm512_add_pd(b_y_lower_f64x8, b_y_upper_f64x8));
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, _mm512_add_pd(b_z_lower_f64x8, b_z_upper_f64x8));

        __m512d delta_x_lower_f64x8 = _mm512_sub_pd(a_x_lower_f64x8, b_x_lower_f64x8);
        __m512d delta_x_upper_f64x8 = _mm512_sub_pd(a_x_upper_f64x8, b_x_upper_f64x8);
        __m512d delta_y_lower_f64x8 = _mm512_sub_pd(a_y_lower_f64x8, b_y_lower_f64x8);
        __m512d delta_y_upper_f64x8 = _mm512_sub_pd(a_y_upper_f64x8, b_y_upper_f64x8);
        __m512d delta_z_lower_f64x8 = _mm512_sub_pd(a_z_lower_f64x8, b_z_lower_f64x8);
        __m512d delta_z_upper_f64x8 = _mm512_sub_pd(a_z_upper_f64x8, b_z_upper_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_lower_f64x8, delta_x_lower_f64x8, sum_squared_x_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_upper_f64x8, delta_x_upper_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_lower_f64x8, delta_y_lower_f64x8, sum_squared_y_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_upper_f64x8, delta_y_upper_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_lower_f64x8, delta_z_lower_f64x8, sum_squared_z_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_upper_f64x8, delta_z_upper_f64x8, sum_squared_z_f64x8);

        // Iteration 1: points i+16..i+31
        nk_deinterleave_f32x16_skylake_(a + (i + 16) * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + (i + 16) * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);
        a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, _mm512_add_pd(a_x_lower_f64x8, a_x_upper_f64x8));
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, _mm512_add_pd(a_y_lower_f64x8, a_y_upper_f64x8));
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, _mm512_add_pd(a_z_lower_f64x8, a_z_upper_f64x8));
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, _mm512_add_pd(b_x_lower_f64x8, b_x_upper_f64x8));
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, _mm512_add_pd(b_y_lower_f64x8, b_y_upper_f64x8));
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, _mm512_add_pd(b_z_lower_f64x8, b_z_upper_f64x8));

        delta_x_lower_f64x8 = _mm512_sub_pd(a_x_lower_f64x8, b_x_lower_f64x8);
        delta_x_upper_f64x8 = _mm512_sub_pd(a_x_upper_f64x8, b_x_upper_f64x8);
        delta_y_lower_f64x8 = _mm512_sub_pd(a_y_lower_f64x8, b_y_lower_f64x8);
        delta_y_upper_f64x8 = _mm512_sub_pd(a_y_upper_f64x8, b_y_upper_f64x8);
        delta_z_lower_f64x8 = _mm512_sub_pd(a_z_lower_f64x8, b_z_lower_f64x8);
        delta_z_upper_f64x8 = _mm512_sub_pd(a_z_upper_f64x8, b_z_upper_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_lower_f64x8, delta_x_lower_f64x8, sum_squared_x_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_upper_f64x8, delta_x_upper_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_lower_f64x8, delta_y_lower_f64x8, sum_squared_y_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_upper_f64x8, delta_y_upper_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_lower_f64x8, delta_z_lower_f64x8, sum_squared_z_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_upper_f64x8, delta_z_upper_f64x8, sum_squared_z_f64x8);
    }

    // Handle 16-point remainder
    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);
        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, _mm512_add_pd(a_x_lower_f64x8, a_x_upper_f64x8));
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, _mm512_add_pd(a_y_lower_f64x8, a_y_upper_f64x8));
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, _mm512_add_pd(a_z_lower_f64x8, a_z_upper_f64x8));
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, _mm512_add_pd(b_x_lower_f64x8, b_x_upper_f64x8));
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, _mm512_add_pd(b_y_lower_f64x8, b_y_upper_f64x8));
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, _mm512_add_pd(b_z_lower_f64x8, b_z_upper_f64x8));

        __m512d delta_x_lower_f64x8 = _mm512_sub_pd(a_x_lower_f64x8, b_x_lower_f64x8);
        __m512d delta_x_upper_f64x8 = _mm512_sub_pd(a_x_upper_f64x8, b_x_upper_f64x8);
        __m512d delta_y_lower_f64x8 = _mm512_sub_pd(a_y_lower_f64x8, b_y_lower_f64x8);
        __m512d delta_y_upper_f64x8 = _mm512_sub_pd(a_y_upper_f64x8, b_y_upper_f64x8);
        __m512d delta_z_lower_f64x8 = _mm512_sub_pd(a_z_lower_f64x8, b_z_lower_f64x8);
        __m512d delta_z_upper_f64x8 = _mm512_sub_pd(a_z_upper_f64x8, b_z_upper_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_lower_f64x8, delta_x_lower_f64x8, sum_squared_x_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_upper_f64x8, delta_x_upper_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_lower_f64x8, delta_y_lower_f64x8, sum_squared_y_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_upper_f64x8, delta_y_upper_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_lower_f64x8, delta_z_lower_f64x8, sum_squared_z_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_upper_f64x8, delta_z_upper_f64x8, sum_squared_z_f64x8);
    }

    // Tail: use masked gather for remaining < 16 points
    if (i < n) {
        nk_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, tail);
        __m512i const gather_idx_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
        __m512 zeros_f32x16 = _mm512_setzero_ps();
        nk_f32_t const *a_tail = a + i * 3;
        nk_f32_t const *b_tail = b + i * 3;

        a_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 0, 4);
        a_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 1, 4);
        a_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 2, 4);
        b_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 0, 4);
        b_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 1, 4);
        b_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 2, 4);

        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, _mm512_add_pd(a_x_lower_f64x8, a_x_upper_f64x8));
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, _mm512_add_pd(a_y_lower_f64x8, a_y_upper_f64x8));
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, _mm512_add_pd(a_z_lower_f64x8, a_z_upper_f64x8));
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, _mm512_add_pd(b_x_lower_f64x8, b_x_upper_f64x8));
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, _mm512_add_pd(b_y_lower_f64x8, b_y_upper_f64x8));
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, _mm512_add_pd(b_z_lower_f64x8, b_z_upper_f64x8));

        __m512d delta_x_lower_f64x8 = _mm512_sub_pd(a_x_lower_f64x8, b_x_lower_f64x8);
        __m512d delta_x_upper_f64x8 = _mm512_sub_pd(a_x_upper_f64x8, b_x_upper_f64x8);
        __m512d delta_y_lower_f64x8 = _mm512_sub_pd(a_y_lower_f64x8, b_y_lower_f64x8);
        __m512d delta_y_upper_f64x8 = _mm512_sub_pd(a_y_upper_f64x8, b_y_upper_f64x8);
        __m512d delta_z_lower_f64x8 = _mm512_sub_pd(a_z_lower_f64x8, b_z_lower_f64x8);
        __m512d delta_z_upper_f64x8 = _mm512_sub_pd(a_z_upper_f64x8, b_z_upper_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_lower_f64x8, delta_x_lower_f64x8, sum_squared_x_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_upper_f64x8, delta_x_upper_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_lower_f64x8, delta_y_lower_f64x8, sum_squared_y_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_upper_f64x8, delta_y_upper_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_lower_f64x8, delta_z_lower_f64x8, sum_squared_z_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_upper_f64x8, delta_z_upper_f64x8, sum_squared_z_f64x8);
    }

    // Reduce and compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t total_ax = _mm512_reduce_add_pd(sum_a_x_f64x8);
    nk_f64_t total_ay = _mm512_reduce_add_pd(sum_a_y_f64x8);
    nk_f64_t total_az = _mm512_reduce_add_pd(sum_a_z_f64x8);
    nk_f64_t total_bx = _mm512_reduce_add_pd(sum_b_x_f64x8);
    nk_f64_t total_by = _mm512_reduce_add_pd(sum_b_y_f64x8);
    nk_f64_t total_bz = _mm512_reduce_add_pd(sum_b_z_f64x8);
    nk_f64_t total_sq_x = _mm512_reduce_add_pd(sum_squared_x_f64x8);
    nk_f64_t total_sq_y = _mm512_reduce_add_pd(sum_squared_y_f64x8);
    nk_f64_t total_sq_z = _mm512_reduce_add_pd(sum_squared_z_f64x8);

    nk_f64_t centroid_a_x = total_ax * inv_n, centroid_a_y = total_ay * inv_n, centroid_a_z = total_az * inv_n;
    nk_f64_t centroid_b_x = total_bx * inv_n, centroid_b_y = total_by * inv_n, centroid_b_z = total_bz * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x, mean_diff_y = centroid_a_y - centroid_b_y,
             mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    // Fused single-pass: centroids + covariance in f64
    __m512d const zeros_f64x8 = _mm512_setzero_pd();
    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;
    __m512d cov_xx_f64x8 = zeros_f64x8, cov_xy_f64x8 = zeros_f64x8, cov_xz_f64x8 = zeros_f64x8;
    __m512d cov_yx_f64x8 = zeros_f64x8, cov_yy_f64x8 = zeros_f64x8, cov_yz_f64x8 = zeros_f64x8;
    __m512d cov_zx_f64x8 = zeros_f64x8, cov_zy_f64x8 = zeros_f64x8, cov_zz_f64x8 = zeros_f64x8;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);
        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, _mm512_add_pd(a_x_lower_f64x8, a_x_upper_f64x8));
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, _mm512_add_pd(a_y_lower_f64x8, a_y_upper_f64x8));
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, _mm512_add_pd(a_z_lower_f64x8, a_z_upper_f64x8));
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, _mm512_add_pd(b_x_lower_f64x8, b_x_upper_f64x8));
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, _mm512_add_pd(b_y_lower_f64x8, b_y_upper_f64x8));
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, _mm512_add_pd(b_z_lower_f64x8, b_z_upper_f64x8));

        cov_xx_f64x8 = _mm512_add_pd(cov_xx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_x_upper_f64x8)));
        cov_xy_f64x8 = _mm512_add_pd(cov_xy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_y_upper_f64x8)));
        cov_xz_f64x8 = _mm512_add_pd(cov_xz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_z_upper_f64x8)));
        cov_yx_f64x8 = _mm512_add_pd(cov_yx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_x_upper_f64x8)));
        cov_yy_f64x8 = _mm512_add_pd(cov_yy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_y_upper_f64x8)));
        cov_yz_f64x8 = _mm512_add_pd(cov_yz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_z_upper_f64x8)));
        cov_zx_f64x8 = _mm512_add_pd(cov_zx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_x_upper_f64x8)));
        cov_zy_f64x8 = _mm512_add_pd(cov_zy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_y_upper_f64x8)));
        cov_zz_f64x8 = _mm512_add_pd(cov_zz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_z_upper_f64x8)));
    }

    // Tail: use masked gather for remaining < 16 points
    if (i < n) {
        nk_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, tail);
        __m512i const gather_idx_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
        __m512 zeros_f32x16 = _mm512_setzero_ps();
        nk_f32_t const *a_tail = a + i * 3;
        nk_f32_t const *b_tail = b + i * 3;

        a_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 0, 4);
        a_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 1, 4);
        a_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 2, 4);
        b_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 0, 4);
        b_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 1, 4);
        b_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 2, 4);

        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, _mm512_add_pd(a_x_lower_f64x8, a_x_upper_f64x8));
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, _mm512_add_pd(a_y_lower_f64x8, a_y_upper_f64x8));
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, _mm512_add_pd(a_z_lower_f64x8, a_z_upper_f64x8));
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, _mm512_add_pd(b_x_lower_f64x8, b_x_upper_f64x8));
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, _mm512_add_pd(b_y_lower_f64x8, b_y_upper_f64x8));
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, _mm512_add_pd(b_z_lower_f64x8, b_z_upper_f64x8));

        cov_xx_f64x8 = _mm512_add_pd(cov_xx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_x_upper_f64x8)));
        cov_xy_f64x8 = _mm512_add_pd(cov_xy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_y_upper_f64x8)));
        cov_xz_f64x8 = _mm512_add_pd(cov_xz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_z_upper_f64x8)));
        cov_yx_f64x8 = _mm512_add_pd(cov_yx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_x_upper_f64x8)));
        cov_yy_f64x8 = _mm512_add_pd(cov_yy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_y_upper_f64x8)));
        cov_yz_f64x8 = _mm512_add_pd(cov_yz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_z_upper_f64x8)));
        cov_zx_f64x8 = _mm512_add_pd(cov_zx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_x_upper_f64x8)));
        cov_zy_f64x8 = _mm512_add_pd(cov_zy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_y_upper_f64x8)));
        cov_zz_f64x8 = _mm512_add_pd(cov_zz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_z_upper_f64x8)));
    }

    nk_f64_t sum_a_x = _mm512_reduce_add_pd(sum_a_x_f64x8), sum_a_y = _mm512_reduce_add_pd(sum_a_y_f64x8),
             sum_a_z = _mm512_reduce_add_pd(sum_a_z_f64x8);
    nk_f64_t sum_b_x = _mm512_reduce_add_pd(sum_b_x_f64x8), sum_b_y = _mm512_reduce_add_pd(sum_b_y_f64x8),
             sum_b_z = _mm512_reduce_add_pd(sum_b_z_f64x8);
    nk_f64_t covariance_x_x = _mm512_reduce_add_pd(cov_xx_f64x8), covariance_x_y = _mm512_reduce_add_pd(cov_xy_f64x8),
             covariance_x_z = _mm512_reduce_add_pd(cov_xz_f64x8);
    nk_f64_t covariance_y_x = _mm512_reduce_add_pd(cov_yx_f64x8), covariance_y_y = _mm512_reduce_add_pd(cov_yy_f64x8),
             covariance_y_z = _mm512_reduce_add_pd(cov_yz_f64x8);
    nk_f64_t covariance_z_x = _mm512_reduce_add_pd(cov_zx_f64x8), covariance_z_y = _mm512_reduce_add_pd(cov_zy_f64x8),
             covariance_z_z = _mm512_reduce_add_pd(cov_zz_f64x8);

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;
    if (scale) *scale = 1.0f;

    nk_f64_t n_f64 = (nk_f64_t)n;
    nk_f64_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - n_f64 * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - n_f64 * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - n_f64 * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - n_f64 * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - n_f64 * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - n_f64 * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - n_f64 * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - n_f64 * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - n_f64 * centroid_a_z * centroid_b_z;

    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);
    nk_f64_t r[9];
    nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)r[j];
    *result = nk_f64_sqrt_haswell(nk_transformed_ssd_f32_skylake_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y,
                                                                  centroid_a_z, centroid_b_x, centroid_b_y,
                                                                  centroid_b_z) /
                                  n_f64);
}

NK_PUBLIC void nk_rmsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // RMSD uses identity rotation and scale=1.0.
    if (rotation) {
        rotation[0] = 1;
        rotation[1] = 0;
        rotation[2] = 0;
        rotation[3] = 0;
        rotation[4] = 1;
        rotation[5] = 0;
        rotation[6] = 0;
        rotation[7] = 0;
        rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    // Optimized fused single-pass implementation for f64.
    // Computes centroids and squared differences in one pass using the identity:
    //   RMSD = √(E[(a-ā) - (b-b̄)]²)
    //        = √(E[(a-b)²] - (ā - b̄)²)
    __m512i const gather_idx_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros_f64x8 = _mm512_setzero_pd();

    // Accumulators for centroids and squared differences
    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;
    __m512d sum_squared_x_f64x8 = zeros_f64x8, sum_squared_y_f64x8 = zeros_f64x8, sum_squared_z_f64x8 = zeros_f64x8;

    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;
    nk_size_t i = 0;

    // Main loop with 2x unrolling for better latency hiding
    for (; i + 16 <= n; i += 16) {
        // Iteration 0
        nk_deinterleave_f64x8_skylake_(a + i * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + i * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

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

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x1_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y1_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z1_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x1_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y1_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z1_f64x8);

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

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

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

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        __m512d delta_x_f64x8 = _mm512_sub_pd(a_x_f64x8, b_x_f64x8),
                delta_y_f64x8 = _mm512_sub_pd(a_y_f64x8, b_y_f64x8),
                delta_z_f64x8 = _mm512_sub_pd(a_z_f64x8, b_z_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_f64x8, delta_x_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8, sum_squared_z_f64x8);
    }

    // Reduce and compute centroids.
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t total_ax = nk_reduce_stable_f64x8_skylake_(sum_a_x_f64x8), total_ax_compensation = 0.0;
    nk_f64_t total_ay = nk_reduce_stable_f64x8_skylake_(sum_a_y_f64x8), total_ay_compensation = 0.0;
    nk_f64_t total_az = nk_reduce_stable_f64x8_skylake_(sum_a_z_f64x8), total_az_compensation = 0.0;
    nk_f64_t total_bx = nk_reduce_stable_f64x8_skylake_(sum_b_x_f64x8), total_bx_compensation = 0.0;
    nk_f64_t total_by = nk_reduce_stable_f64x8_skylake_(sum_b_y_f64x8), total_by_compensation = 0.0;
    nk_f64_t total_bz = nk_reduce_stable_f64x8_skylake_(sum_b_z_f64x8), total_bz_compensation = 0.0;
    nk_f64_t total_squared_x = nk_reduce_stable_f64x8_skylake_(sum_squared_x_f64x8), total_squared_x_compensation = 0.0;
    nk_f64_t total_squared_y = nk_reduce_stable_f64x8_skylake_(sum_squared_y_f64x8), total_squared_y_compensation = 0.0;
    nk_f64_t total_squared_z = nk_reduce_stable_f64x8_skylake_(sum_squared_z_f64x8), total_squared_z_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&total_ax, &total_ax_compensation, ax);
        nk_accumulate_sum_f64_(&total_ay, &total_ay_compensation, ay);
        nk_accumulate_sum_f64_(&total_az, &total_az_compensation, az);
        nk_accumulate_sum_f64_(&total_bx, &total_bx_compensation, bx);
        nk_accumulate_sum_f64_(&total_by, &total_by_compensation, by);
        nk_accumulate_sum_f64_(&total_bz, &total_bz_compensation, bz);
        nk_f64_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        nk_accumulate_square_f64_(&total_squared_x, &total_squared_x_compensation, delta_x);
        nk_accumulate_square_f64_(&total_squared_y, &total_squared_y_compensation, delta_y);
        nk_accumulate_square_f64_(&total_squared_z, &total_squared_z_compensation, delta_z);
    }

    total_ax += total_ax_compensation, total_ay += total_ay_compensation, total_az += total_az_compensation;
    total_bx += total_bx_compensation, total_by += total_by_compensation, total_bz += total_bz_compensation;
    total_squared_x += total_squared_x_compensation, total_squared_y += total_squared_y_compensation,
        total_squared_z += total_squared_z_compensation;

    nk_f64_t centroid_a_x = total_ax * inv_n, centroid_a_y = total_ay * inv_n, centroid_a_z = total_az * inv_n;
    nk_f64_t centroid_b_x = total_bx * inv_n, centroid_b_y = total_by * inv_n, centroid_b_z = total_bz * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute RMSD using the formula:
    // RMSD = √(E[(a-b)²] - (ā - b̄)²).
    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x, mean_diff_y = centroid_a_y - centroid_b_y,
             mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t sum_squared = total_squared_x + total_squared_y + total_squared_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f64_sqrt_haswell(sum_squared * inv_n - mean_diff_sq);
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
    __m512d cov_xx_f64x8 = zeros_f64x8, cov_xy_f64x8 = zeros_f64x8, cov_xz_f64x8 = zeros_f64x8;
    __m512d cov_yx_f64x8 = zeros_f64x8, cov_yy_f64x8 = zeros_f64x8, cov_yz_f64x8 = zeros_f64x8;
    __m512d cov_zx_f64x8 = zeros_f64x8, cov_zy_f64x8 = zeros_f64x8, cov_zz_f64x8 = zeros_f64x8;

    nk_size_t i = 0;
    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;

    // Fused single-pass: accumulate sums and outer products together
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
        cov_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, cov_xx_f64x8),
        cov_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, cov_xy_f64x8),
        cov_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, cov_xz_f64x8);
        cov_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, cov_yx_f64x8),
        cov_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, cov_yy_f64x8),
        cov_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, cov_yz_f64x8);
        cov_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, cov_zx_f64x8),
        cov_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, cov_zy_f64x8),
        cov_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, cov_zz_f64x8);
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

        cov_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, cov_xx_f64x8),
        cov_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, cov_xy_f64x8),
        cov_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, cov_xz_f64x8);
        cov_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, cov_yx_f64x8),
        cov_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, cov_yy_f64x8),
        cov_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, cov_yz_f64x8);
        cov_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, cov_zx_f64x8),
        cov_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, cov_zy_f64x8),
        cov_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, cov_zz_f64x8);
    }

    // Reduce centroids and covariance.
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t sum_a_x = nk_reduce_stable_f64x8_skylake_(sum_a_x_f64x8), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x8_skylake_(sum_a_y_f64x8), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x8_skylake_(sum_a_z_f64x8), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x8_skylake_(sum_b_x_f64x8), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x8_skylake_(sum_b_y_f64x8), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x8_skylake_(sum_b_z_f64x8), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x8_skylake_(cov_xx_f64x8), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x8_skylake_(cov_xy_f64x8), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x8_skylake_(cov_xz_f64x8), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x8_skylake_(cov_yx_f64x8), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x8_skylake_(cov_yy_f64x8), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x8_skylake_(cov_yz_f64x8), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x8_skylake_(cov_zx_f64x8), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x8_skylake_(cov_zy_f64x8), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x8_skylake_(cov_zz_f64x8), covariance_z_z_compensation = 0.0;

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
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;

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

    // SVD using f64 for full precision
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);

    // Handle reflection
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);
    }

    // Output rotation matrix and scale=1.0.
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)r[j];
    }
    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_skylake_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    // Fused single-pass: centroids + covariance + variance of A, all in f64
    __m512d const zeros_f64x8 = _mm512_setzero_pd();
    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;
    __m512d cov_xx_f64x8 = zeros_f64x8, cov_xy_f64x8 = zeros_f64x8, cov_xz_f64x8 = zeros_f64x8;
    __m512d cov_yx_f64x8 = zeros_f64x8, cov_yy_f64x8 = zeros_f64x8, cov_yz_f64x8 = zeros_f64x8;
    __m512d cov_zx_f64x8 = zeros_f64x8, cov_zy_f64x8 = zeros_f64x8, cov_zz_f64x8 = zeros_f64x8;
    __m512d variance_a_f64x8 = zeros_f64x8;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);
        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, _mm512_add_pd(a_x_lower_f64x8, a_x_upper_f64x8));
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, _mm512_add_pd(a_y_lower_f64x8, a_y_upper_f64x8));
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, _mm512_add_pd(a_z_lower_f64x8, a_z_upper_f64x8));
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, _mm512_add_pd(b_x_lower_f64x8, b_x_upper_f64x8));
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, _mm512_add_pd(b_y_lower_f64x8, b_y_upper_f64x8));
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, _mm512_add_pd(b_z_lower_f64x8, b_z_upper_f64x8));

        cov_xx_f64x8 = _mm512_add_pd(cov_xx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_x_upper_f64x8)));
        cov_xy_f64x8 = _mm512_add_pd(cov_xy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_y_upper_f64x8)));
        cov_xz_f64x8 = _mm512_add_pd(cov_xz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_z_upper_f64x8)));
        cov_yx_f64x8 = _mm512_add_pd(cov_yx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_x_upper_f64x8)));
        cov_yy_f64x8 = _mm512_add_pd(cov_yy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_y_upper_f64x8)));
        cov_yz_f64x8 = _mm512_add_pd(cov_yz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_z_upper_f64x8)));
        cov_zx_f64x8 = _mm512_add_pd(cov_zx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_x_upper_f64x8)));
        cov_zy_f64x8 = _mm512_add_pd(cov_zy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_y_upper_f64x8)));
        cov_zz_f64x8 = _mm512_add_pd(cov_zz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_z_upper_f64x8)));

        variance_a_f64x8 = _mm512_add_pd(variance_a_f64x8,
                                         _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, a_x_lower_f64x8),
                                                       _mm512_mul_pd(a_x_upper_f64x8, a_x_upper_f64x8)));
        variance_a_f64x8 = _mm512_add_pd(variance_a_f64x8,
                                         _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, a_y_lower_f64x8),
                                                       _mm512_mul_pd(a_y_upper_f64x8, a_y_upper_f64x8)));
        variance_a_f64x8 = _mm512_add_pd(variance_a_f64x8,
                                         _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, a_z_lower_f64x8),
                                                       _mm512_mul_pd(a_z_upper_f64x8, a_z_upper_f64x8)));
    }

    // Tail: use masked gather for remaining < 16 points
    if (i < n) {
        nk_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, tail);
        __m512i const gather_idx_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
        __m512 zeros_f32x16 = _mm512_setzero_ps();
        nk_f32_t const *a_tail = a + i * 3;
        nk_f32_t const *b_tail = b + i * 3;

        a_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 0, 4);
        a_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 1, 4);
        a_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 2, 4);
        b_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 0, 4);
        b_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 1, 4);
        b_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 2, 4);

        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, _mm512_add_pd(a_x_lower_f64x8, a_x_upper_f64x8));
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, _mm512_add_pd(a_y_lower_f64x8, a_y_upper_f64x8));
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, _mm512_add_pd(a_z_lower_f64x8, a_z_upper_f64x8));
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, _mm512_add_pd(b_x_lower_f64x8, b_x_upper_f64x8));
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, _mm512_add_pd(b_y_lower_f64x8, b_y_upper_f64x8));
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, _mm512_add_pd(b_z_lower_f64x8, b_z_upper_f64x8));

        cov_xx_f64x8 = _mm512_add_pd(cov_xx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_x_upper_f64x8)));
        cov_xy_f64x8 = _mm512_add_pd(cov_xy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_y_upper_f64x8)));
        cov_xz_f64x8 = _mm512_add_pd(cov_xz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_x_upper_f64x8, b_z_upper_f64x8)));
        cov_yx_f64x8 = _mm512_add_pd(cov_yx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_x_upper_f64x8)));
        cov_yy_f64x8 = _mm512_add_pd(cov_yy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_y_upper_f64x8)));
        cov_yz_f64x8 = _mm512_add_pd(cov_yz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_y_upper_f64x8, b_z_upper_f64x8)));
        cov_zx_f64x8 = _mm512_add_pd(cov_zx_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_x_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_x_upper_f64x8)));
        cov_zy_f64x8 = _mm512_add_pd(cov_zy_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_y_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_y_upper_f64x8)));
        cov_zz_f64x8 = _mm512_add_pd(cov_zz_f64x8, _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_z_lower_f64x8),
                                                                 _mm512_mul_pd(a_z_upper_f64x8, b_z_upper_f64x8)));

        variance_a_f64x8 = _mm512_add_pd(variance_a_f64x8,
                                         _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, a_x_lower_f64x8),
                                                       _mm512_mul_pd(a_x_upper_f64x8, a_x_upper_f64x8)));
        variance_a_f64x8 = _mm512_add_pd(variance_a_f64x8,
                                         _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, a_y_lower_f64x8),
                                                       _mm512_mul_pd(a_y_upper_f64x8, a_y_upper_f64x8)));
        variance_a_f64x8 = _mm512_add_pd(variance_a_f64x8,
                                         _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, a_z_lower_f64x8),
                                                       _mm512_mul_pd(a_z_upper_f64x8, a_z_upper_f64x8)));
    }

    nk_f64_t sum_a_x = _mm512_reduce_add_pd(sum_a_x_f64x8), sum_a_y = _mm512_reduce_add_pd(sum_a_y_f64x8),
             sum_a_z = _mm512_reduce_add_pd(sum_a_z_f64x8);
    nk_f64_t sum_b_x = _mm512_reduce_add_pd(sum_b_x_f64x8), sum_b_y = _mm512_reduce_add_pd(sum_b_y_f64x8),
             sum_b_z = _mm512_reduce_add_pd(sum_b_z_f64x8);
    nk_f64_t covariance_x_x = _mm512_reduce_add_pd(cov_xx_f64x8), covariance_x_y = _mm512_reduce_add_pd(cov_xy_f64x8),
             covariance_x_z = _mm512_reduce_add_pd(cov_xz_f64x8);
    nk_f64_t covariance_y_x = _mm512_reduce_add_pd(cov_yx_f64x8), covariance_y_y = _mm512_reduce_add_pd(cov_yy_f64x8),
             covariance_y_z = _mm512_reduce_add_pd(cov_yz_f64x8);
    nk_f64_t covariance_z_x = _mm512_reduce_add_pd(cov_zx_f64x8), covariance_z_y = _mm512_reduce_add_pd(cov_zy_f64x8),
             covariance_z_z = _mm512_reduce_add_pd(cov_zz_f64x8);
    nk_f64_t variance_a_sum = _mm512_reduce_add_pd(variance_a_f64x8);

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    // Compute centered covariance and variance
    nk_f64_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    // Compute centered covariance matrix: Hᵢⱼ = Σ(aᵢ×bⱼ) - Σaᵢ × Σbⱼ / n
    nk_f64_t n_f64 = (nk_f64_t)n;
    nk_f64_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - n_f64 * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - n_f64 * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - n_f64 * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - n_f64 * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - n_f64 * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - n_f64 * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - n_f64 * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - n_f64 * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - n_f64 * centroid_a_z * centroid_b_z;

    // SVD using f64 for full precision
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);

    // Scale factor: c = trace(D × S) / (n × variance(a))
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
    nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_s[0], 1.0, svd_s[4], 1.0, svd_s[8], d3);
    nk_f64_t applied_scale = trace_ds / (n * variance_a);
    if (scale) *scale = (nk_f32_t)applied_scale;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);
    }

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)r[j];
    *result = nk_f64_sqrt_haswell(nk_transformed_ssd_f32_skylake_(a, b, n, r, applied_scale, centroid_a_x, centroid_a_y,
                                                                  centroid_a_z, centroid_b_x, centroid_b_y,
                                                                  centroid_b_z) /
                                  n_f64);
}

NK_PUBLIC void nk_umeyama_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m512i const gather_idx_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros_f64x8 = _mm512_setzero_pd();

    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;
    __m512d cov_xx_f64x8 = zeros_f64x8, cov_xy_f64x8 = zeros_f64x8, cov_xz_f64x8 = zeros_f64x8;
    __m512d cov_yx_f64x8 = zeros_f64x8, cov_yy_f64x8 = zeros_f64x8, cov_yz_f64x8 = zeros_f64x8;
    __m512d cov_zx_f64x8 = zeros_f64x8, cov_zy_f64x8 = zeros_f64x8, cov_zz_f64x8 = zeros_f64x8;
    __m512d variance_a_f64x8 = zeros_f64x8;

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

        cov_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, cov_xx_f64x8),
        cov_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, cov_xy_f64x8);
        cov_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, cov_xz_f64x8);
        cov_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, cov_yx_f64x8),
        cov_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, cov_yy_f64x8);
        cov_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, cov_yz_f64x8);
        cov_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, cov_zx_f64x8),
        cov_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, cov_zy_f64x8);
        cov_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, cov_zz_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_x_f64x8, a_x_f64x8, variance_a_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_y_f64x8, a_y_f64x8, variance_a_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_z_f64x8, a_z_f64x8, variance_a_f64x8);
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

        cov_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, cov_xx_f64x8),
        cov_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, cov_xy_f64x8);
        cov_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, cov_xz_f64x8);
        cov_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, cov_yx_f64x8),
        cov_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, cov_yy_f64x8);
        cov_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, cov_yz_f64x8);
        cov_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, cov_zx_f64x8),
        cov_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, cov_zy_f64x8);
        cov_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, cov_zz_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_x_f64x8, a_x_f64x8, variance_a_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_y_f64x8, a_y_f64x8, variance_a_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_z_f64x8, a_z_f64x8, variance_a_f64x8);
    }

    // Reduce centroids, covariance, and variance.
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t sum_a_x = nk_reduce_stable_f64x8_skylake_(sum_a_x_f64x8), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x8_skylake_(sum_a_y_f64x8), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x8_skylake_(sum_a_z_f64x8), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x8_skylake_(sum_b_x_f64x8), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x8_skylake_(sum_b_y_f64x8), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x8_skylake_(sum_b_z_f64x8), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x8_skylake_(cov_xx_f64x8), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x8_skylake_(cov_xy_f64x8), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x8_skylake_(cov_xz_f64x8), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x8_skylake_(cov_yx_f64x8), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x8_skylake_(cov_yy_f64x8), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x8_skylake_(cov_yz_f64x8), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x8_skylake_(cov_zx_f64x8), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x8_skylake_(cov_zy_f64x8), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x8_skylake_(cov_zz_f64x8), covariance_z_z_compensation = 0.0;
    nk_f64_t variance_a_sum = nk_reduce_stable_f64x8_skylake_(variance_a_f64x8), variance_a_compensation = 0.0;

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
        nk_accumulate_square_f64_(&variance_a_sum, &variance_a_compensation, ax);
        nk_accumulate_square_f64_(&variance_a_sum, &variance_a_compensation, ay);
        nk_accumulate_square_f64_(&variance_a_sum, &variance_a_compensation, az);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    variance_a_sum += variance_a_compensation;

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance.
    nk_f64_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

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
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);

    // Scale factor: c = trace(D × S) / (n × variance(a))
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
    nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_s[0], 1.0, svd_s[4], 1.0, svd_s[8], d3);
    nk_f64_t c = trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);
    }

    // Output rotation matrix.
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)r[j];
    }

    // Compute RMSD with scaling
    nk_f64_t sum_squared = nk_transformed_ssd_f64_skylake_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_rmsd_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    __m512 const zeros_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_x_f32x16 = zeros_f32x16, sum_a_y_f32x16 = zeros_f32x16, sum_a_z_f32x16 = zeros_f32x16;
    __m512 sum_b_x_f32x16 = zeros_f32x16, sum_b_y_f32x16 = zeros_f32x16, sum_b_z_f32x16 = zeros_f32x16;
    __m512 sum_squared_x_f32x16 = zeros_f32x16, sum_squared_y_f32x16 = zeros_f32x16;
    __m512 sum_squared_z_f32x16 = zeros_f32x16;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_f16x16_to_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f16x16_to_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        __m512 delta_x_f32x16 = _mm512_sub_ps(a_x_f32x16, b_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(a_y_f32x16, b_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(a_z_f32x16, b_z_f32x16);

        sum_squared_x_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_x_f32x16);
        sum_squared_y_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_y_f32x16);
        sum_squared_z_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_z_f32x16);
    }

    // Tail: deinterleave remaining points into zero-initialized vectors
    if (i < n) {
        nk_size_t tail = n - i;
        nk_deinterleave_f16_tail_to_f32x16_skylake_(a + i * 3, tail, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f16_tail_to_f32x16_skylake_(b + i * 3, tail, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        __m512 delta_x_f32x16 = _mm512_sub_ps(a_x_f32x16, b_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(a_y_f32x16, b_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(a_z_f32x16, b_z_f32x16);

        sum_squared_x_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_x_f32x16);
        sum_squared_y_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_y_f32x16);
        sum_squared_z_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_z_f32x16);
    }

    nk_f32_t total_ax = _mm512_reduce_add_ps(sum_a_x_f32x16);
    nk_f32_t total_ay = _mm512_reduce_add_ps(sum_a_y_f32x16);
    nk_f32_t total_az = _mm512_reduce_add_ps(sum_a_z_f32x16);
    nk_f32_t total_bx = _mm512_reduce_add_ps(sum_b_x_f32x16);
    nk_f32_t total_by = _mm512_reduce_add_ps(sum_b_y_f32x16);
    nk_f32_t total_bz = _mm512_reduce_add_ps(sum_b_z_f32x16);
    nk_f32_t total_sq_x = _mm512_reduce_add_ps(sum_squared_x_f32x16);
    nk_f32_t total_sq_y = _mm512_reduce_add_ps(sum_squared_y_f32x16);
    nk_f32_t total_sq_z = _mm512_reduce_add_ps(sum_squared_z_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = total_ax * inv_n;
    nk_f32_t centroid_a_y = total_ay * inv_n;
    nk_f32_t centroid_a_z = total_az * inv_n;
    nk_f32_t centroid_b_x = total_bx * inv_n;
    nk_f32_t centroid_b_y = total_by * inv_n;
    nk_f32_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    nk_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f32_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f32_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    nk_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f32_sqrt_haswell(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_rmsd_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    __m512 const zeros_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_x_f32x16 = zeros_f32x16, sum_a_y_f32x16 = zeros_f32x16, sum_a_z_f32x16 = zeros_f32x16;
    __m512 sum_b_x_f32x16 = zeros_f32x16, sum_b_y_f32x16 = zeros_f32x16, sum_b_z_f32x16 = zeros_f32x16;
    __m512 sum_squared_x_f32x16 = zeros_f32x16, sum_squared_y_f32x16 = zeros_f32x16;
    __m512 sum_squared_z_f32x16 = zeros_f32x16;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_bf16x16_to_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_bf16x16_to_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        __m512 delta_x_f32x16 = _mm512_sub_ps(a_x_f32x16, b_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(a_y_f32x16, b_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(a_z_f32x16, b_z_f32x16);

        sum_squared_x_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_x_f32x16);
        sum_squared_y_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_y_f32x16);
        sum_squared_z_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_z_f32x16);
    }

    // Tail: deinterleave remaining points into zero-initialized vectors
    if (i < n) {
        nk_size_t tail = n - i;
        nk_deinterleave_bf16_tail_to_f32x16_skylake_(a + i * 3, tail, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_bf16_tail_to_f32x16_skylake_(b + i * 3, tail, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        __m512 delta_x_f32x16 = _mm512_sub_ps(a_x_f32x16, b_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(a_y_f32x16, b_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(a_z_f32x16, b_z_f32x16);

        sum_squared_x_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_x_f32x16);
        sum_squared_y_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_y_f32x16);
        sum_squared_z_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_z_f32x16);
    }

    nk_f32_t total_ax = _mm512_reduce_add_ps(sum_a_x_f32x16);
    nk_f32_t total_ay = _mm512_reduce_add_ps(sum_a_y_f32x16);
    nk_f32_t total_az = _mm512_reduce_add_ps(sum_a_z_f32x16);
    nk_f32_t total_bx = _mm512_reduce_add_ps(sum_b_x_f32x16);
    nk_f32_t total_by = _mm512_reduce_add_ps(sum_b_y_f32x16);
    nk_f32_t total_bz = _mm512_reduce_add_ps(sum_b_z_f32x16);
    nk_f32_t total_sq_x = _mm512_reduce_add_ps(sum_squared_x_f32x16);
    nk_f32_t total_sq_y = _mm512_reduce_add_ps(sum_squared_y_f32x16);
    nk_f32_t total_sq_z = _mm512_reduce_add_ps(sum_squared_z_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = total_ax * inv_n;
    nk_f32_t centroid_a_y = total_ay * inv_n;
    nk_f32_t centroid_a_z = total_az * inv_n;
    nk_f32_t centroid_b_x = total_bx * inv_n;
    nk_f32_t centroid_b_y = total_by * inv_n;
    nk_f32_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    nk_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f32_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f32_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    nk_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f32_sqrt_haswell(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    __m512 const zeros_f32x16 = _mm512_setzero_ps();

    __m512 sum_a_x_f32x16 = zeros_f32x16, sum_a_y_f32x16 = zeros_f32x16, sum_a_z_f32x16 = zeros_f32x16;
    __m512 sum_b_x_f32x16 = zeros_f32x16, sum_b_y_f32x16 = zeros_f32x16, sum_b_z_f32x16 = zeros_f32x16;
    __m512 cov_xx_f32x16 = zeros_f32x16, cov_xy_f32x16 = zeros_f32x16, cov_xz_f32x16 = zeros_f32x16;
    __m512 cov_yx_f32x16 = zeros_f32x16, cov_yy_f32x16 = zeros_f32x16, cov_yz_f32x16 = zeros_f32x16;
    __m512 cov_zx_f32x16 = zeros_f32x16, cov_zy_f32x16 = zeros_f32x16, cov_zz_f32x16 = zeros_f32x16;

    nk_size_t i = 0;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;

    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_f16x16_to_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f16x16_to_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);
    }

    // Tail: deinterleave remaining points into zero-initialized vectors
    if (i < n) {
        nk_size_t tail = n - i;
        nk_deinterleave_f16_tail_to_f32x16_skylake_(a + i * 3, tail, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f16_tail_to_f32x16_skylake_(b + i * 3, tail, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);
    }

    nk_f32_t sum_a_x = _mm512_reduce_add_ps(sum_a_x_f32x16);
    nk_f32_t sum_a_y = _mm512_reduce_add_ps(sum_a_y_f32x16);
    nk_f32_t sum_a_z = _mm512_reduce_add_ps(sum_a_z_f32x16);
    nk_f32_t sum_b_x = _mm512_reduce_add_ps(sum_b_x_f32x16);
    nk_f32_t sum_b_y = _mm512_reduce_add_ps(sum_b_y_f32x16);
    nk_f32_t sum_b_z = _mm512_reduce_add_ps(sum_b_z_f32x16);
    nk_f32_t covariance_x_x = _mm512_reduce_add_ps(cov_xx_f32x16);
    nk_f32_t covariance_x_y = _mm512_reduce_add_ps(cov_xy_f32x16);
    nk_f32_t covariance_x_z = _mm512_reduce_add_ps(cov_xz_f32x16);
    nk_f32_t covariance_y_x = _mm512_reduce_add_ps(cov_yx_f32x16);
    nk_f32_t covariance_y_y = _mm512_reduce_add_ps(cov_yy_f32x16);
    nk_f32_t covariance_y_z = _mm512_reduce_add_ps(cov_yz_f32x16);
    nk_f32_t covariance_z_x = _mm512_reduce_add_ps(cov_zx_f32x16);
    nk_f32_t covariance_z_y = _mm512_reduce_add_ps(cov_zy_f32x16);
    nk_f32_t covariance_z_z = _mm512_reduce_add_ps(cov_zz_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    if (nk_det3x3_f32_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0f;

    nk_f32_t sum_squared = nk_transformed_ssd_f16_skylake_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_kabsch_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    __m512 const zeros_f32x16 = _mm512_setzero_ps();

    __m512 sum_a_x_f32x16 = zeros_f32x16, sum_a_y_f32x16 = zeros_f32x16, sum_a_z_f32x16 = zeros_f32x16;
    __m512 sum_b_x_f32x16 = zeros_f32x16, sum_b_y_f32x16 = zeros_f32x16, sum_b_z_f32x16 = zeros_f32x16;
    __m512 cov_xx_f32x16 = zeros_f32x16, cov_xy_f32x16 = zeros_f32x16, cov_xz_f32x16 = zeros_f32x16;
    __m512 cov_yx_f32x16 = zeros_f32x16, cov_yy_f32x16 = zeros_f32x16, cov_yz_f32x16 = zeros_f32x16;
    __m512 cov_zx_f32x16 = zeros_f32x16, cov_zy_f32x16 = zeros_f32x16, cov_zz_f32x16 = zeros_f32x16;

    nk_size_t i = 0;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;

    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_bf16x16_to_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_bf16x16_to_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);
    }

    // Tail: deinterleave remaining points into zero-initialized vectors
    if (i < n) {
        nk_size_t tail = n - i;
        nk_deinterleave_bf16_tail_to_f32x16_skylake_(a + i * 3, tail, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_bf16_tail_to_f32x16_skylake_(b + i * 3, tail, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);
    }

    nk_f32_t sum_a_x = _mm512_reduce_add_ps(sum_a_x_f32x16);
    nk_f32_t sum_a_y = _mm512_reduce_add_ps(sum_a_y_f32x16);
    nk_f32_t sum_a_z = _mm512_reduce_add_ps(sum_a_z_f32x16);
    nk_f32_t sum_b_x = _mm512_reduce_add_ps(sum_b_x_f32x16);
    nk_f32_t sum_b_y = _mm512_reduce_add_ps(sum_b_y_f32x16);
    nk_f32_t sum_b_z = _mm512_reduce_add_ps(sum_b_z_f32x16);
    nk_f32_t covariance_x_x = _mm512_reduce_add_ps(cov_xx_f32x16);
    nk_f32_t covariance_x_y = _mm512_reduce_add_ps(cov_xy_f32x16);
    nk_f32_t covariance_x_z = _mm512_reduce_add_ps(cov_xz_f32x16);
    nk_f32_t covariance_y_x = _mm512_reduce_add_ps(cov_yx_f32x16);
    nk_f32_t covariance_y_y = _mm512_reduce_add_ps(cov_yy_f32x16);
    nk_f32_t covariance_y_z = _mm512_reduce_add_ps(cov_yz_f32x16);
    nk_f32_t covariance_z_x = _mm512_reduce_add_ps(cov_zx_f32x16);
    nk_f32_t covariance_z_y = _mm512_reduce_add_ps(cov_zy_f32x16);
    nk_f32_t covariance_z_z = _mm512_reduce_add_ps(cov_zz_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    if (nk_det3x3_f32_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0f;

    nk_f32_t sum_squared = nk_transformed_ssd_bf16_skylake_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                            centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    __m512 const zeros_f32x16 = _mm512_setzero_ps();

    __m512 sum_a_x_f32x16 = zeros_f32x16, sum_a_y_f32x16 = zeros_f32x16, sum_a_z_f32x16 = zeros_f32x16;
    __m512 sum_b_x_f32x16 = zeros_f32x16, sum_b_y_f32x16 = zeros_f32x16, sum_b_z_f32x16 = zeros_f32x16;
    __m512 cov_xx_f32x16 = zeros_f32x16, cov_xy_f32x16 = zeros_f32x16, cov_xz_f32x16 = zeros_f32x16;
    __m512 cov_yx_f32x16 = zeros_f32x16, cov_yy_f32x16 = zeros_f32x16, cov_yz_f32x16 = zeros_f32x16;
    __m512 cov_zx_f32x16 = zeros_f32x16, cov_zy_f32x16 = zeros_f32x16, cov_zz_f32x16 = zeros_f32x16;
    __m512 variance_a_f32x16 = zeros_f32x16;

    nk_size_t i = 0;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;

    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_f16x16_to_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f16x16_to_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);

        variance_a_f32x16 = _mm512_fmadd_ps(a_x_f32x16, a_x_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_y_f32x16, a_y_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_z_f32x16, a_z_f32x16, variance_a_f32x16);
    }

    // Tail: deinterleave remaining points into zero-initialized vectors
    if (i < n) {
        nk_size_t tail = n - i;
        nk_deinterleave_f16_tail_to_f32x16_skylake_(a + i * 3, tail, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f16_tail_to_f32x16_skylake_(b + i * 3, tail, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);

        variance_a_f32x16 = _mm512_fmadd_ps(a_x_f32x16, a_x_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_y_f32x16, a_y_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_z_f32x16, a_z_f32x16, variance_a_f32x16);
    }

    nk_f32_t sum_a_x = _mm512_reduce_add_ps(sum_a_x_f32x16);
    nk_f32_t sum_a_y = _mm512_reduce_add_ps(sum_a_y_f32x16);
    nk_f32_t sum_a_z = _mm512_reduce_add_ps(sum_a_z_f32x16);
    nk_f32_t sum_b_x = _mm512_reduce_add_ps(sum_b_x_f32x16);
    nk_f32_t sum_b_y = _mm512_reduce_add_ps(sum_b_y_f32x16);
    nk_f32_t sum_b_z = _mm512_reduce_add_ps(sum_b_z_f32x16);
    nk_f32_t covariance_x_x = _mm512_reduce_add_ps(cov_xx_f32x16);
    nk_f32_t covariance_x_y = _mm512_reduce_add_ps(cov_xy_f32x16);
    nk_f32_t covariance_x_z = _mm512_reduce_add_ps(cov_xz_f32x16);
    nk_f32_t covariance_y_x = _mm512_reduce_add_ps(cov_yx_f32x16);
    nk_f32_t covariance_y_y = _mm512_reduce_add_ps(cov_yy_f32x16);
    nk_f32_t covariance_y_z = _mm512_reduce_add_ps(cov_yz_f32x16);
    nk_f32_t covariance_z_x = _mm512_reduce_add_ps(cov_zx_f32x16);
    nk_f32_t covariance_z_y = _mm512_reduce_add_ps(cov_zy_f32x16);
    nk_f32_t covariance_z_z = _mm512_reduce_add_ps(cov_zz_f32x16);
    nk_f32_t variance_a_sum = _mm512_reduce_add_ps(variance_a_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    nk_f32_t det = nk_det3x3_f32_(r);
    nk_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    nk_f32_t c = trace_ds / (n * variance_a);
    if (scale) *scale = c;

    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    nk_f32_t sum_squared = nk_transformed_ssd_f16_skylake_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_haswell(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    __m512 const zeros_f32x16 = _mm512_setzero_ps();

    __m512 sum_a_x_f32x16 = zeros_f32x16, sum_a_y_f32x16 = zeros_f32x16, sum_a_z_f32x16 = zeros_f32x16;
    __m512 sum_b_x_f32x16 = zeros_f32x16, sum_b_y_f32x16 = zeros_f32x16, sum_b_z_f32x16 = zeros_f32x16;
    __m512 cov_xx_f32x16 = zeros_f32x16, cov_xy_f32x16 = zeros_f32x16, cov_xz_f32x16 = zeros_f32x16;
    __m512 cov_yx_f32x16 = zeros_f32x16, cov_yy_f32x16 = zeros_f32x16, cov_yz_f32x16 = zeros_f32x16;
    __m512 cov_zx_f32x16 = zeros_f32x16, cov_zy_f32x16 = zeros_f32x16, cov_zz_f32x16 = zeros_f32x16;
    __m512 variance_a_f32x16 = zeros_f32x16;

    nk_size_t i = 0;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;

    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_bf16x16_to_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_bf16x16_to_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);

        variance_a_f32x16 = _mm512_fmadd_ps(a_x_f32x16, a_x_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_y_f32x16, a_y_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_z_f32x16, a_z_f32x16, variance_a_f32x16);
    }

    // Tail: deinterleave remaining points into zero-initialized vectors
    if (i < n) {
        nk_size_t tail = n - i;
        nk_deinterleave_bf16_tail_to_f32x16_skylake_(a + i * 3, tail, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_bf16_tail_to_f32x16_skylake_(b + i * 3, tail, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);

        variance_a_f32x16 = _mm512_fmadd_ps(a_x_f32x16, a_x_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_y_f32x16, a_y_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_z_f32x16, a_z_f32x16, variance_a_f32x16);
    }

    nk_f32_t sum_a_x = _mm512_reduce_add_ps(sum_a_x_f32x16);
    nk_f32_t sum_a_y = _mm512_reduce_add_ps(sum_a_y_f32x16);
    nk_f32_t sum_a_z = _mm512_reduce_add_ps(sum_a_z_f32x16);
    nk_f32_t sum_b_x = _mm512_reduce_add_ps(sum_b_x_f32x16);
    nk_f32_t sum_b_y = _mm512_reduce_add_ps(sum_b_y_f32x16);
    nk_f32_t sum_b_z = _mm512_reduce_add_ps(sum_b_z_f32x16);
    nk_f32_t covariance_x_x = _mm512_reduce_add_ps(cov_xx_f32x16);
    nk_f32_t covariance_x_y = _mm512_reduce_add_ps(cov_xy_f32x16);
    nk_f32_t covariance_x_z = _mm512_reduce_add_ps(cov_xz_f32x16);
    nk_f32_t covariance_y_x = _mm512_reduce_add_ps(cov_yx_f32x16);
    nk_f32_t covariance_y_y = _mm512_reduce_add_ps(cov_yy_f32x16);
    nk_f32_t covariance_y_z = _mm512_reduce_add_ps(cov_yz_f32x16);
    nk_f32_t covariance_z_x = _mm512_reduce_add_ps(cov_zx_f32x16);
    nk_f32_t covariance_z_y = _mm512_reduce_add_ps(cov_zy_f32x16);
    nk_f32_t covariance_z_z = _mm512_reduce_add_ps(cov_zz_f32x16);
    nk_f32_t variance_a_sum = _mm512_reduce_add_ps(variance_a_f32x16);

    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};

    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    nk_f32_t det = nk_det3x3_f32_(r);
    nk_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    nk_f32_t c = trace_ds / (n * variance_a);
    if (scale) *scale = c;

    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    nk_f32_t sum_squared = nk_transformed_ssd_bf16_skylake_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                            centroid_b_x, centroid_b_y, centroid_b_z);
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
#endif // NK_TARGET_X86_
#endif // NK_MESH_SKYLAKE_H
