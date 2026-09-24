/**
 *  @brief SIMD-accelerated Batched Dot Products for RISC-V.
 *  @file include/numkong/dots/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Custom RVV-native register-tiled GEMM implementation, analogous to how AMX
 *  (dots/sapphireamx.h) and SME (dots/sme.h) each have their own unique implementations
 *  independent of the cross-product macros.
 *
 *  RVV's variable-length vectors and widening multiply-accumulate (`vfwmacc`) make it
 *  fundamentally different from fixed-width SIMD. Key design choices:
 *
 *  - f32 GEMM: Uses `vfwmacc_vv_f64m4` for f64 accumulation (vector-vector widened FMA),
 *    Process 4 rows per tile (rows_per_tile=4). Narrowed to f32 on store.
 *  - f64 GEMM: Uses `vfmul`+Kahan with Kahan compensation,
 *    Process 2 rows per tile (rows_per_tile=2, tighter register budget at LMUL=4).
 *  - B packing: Column-panel layout with cache-line padding. Each depth step stores
 *    contiguous elements along depth — one `vle32`/`vle64` per vectorized chunk.
 *  - Edge handling: RVV's `vsetvl` returns actual VL for partial vectors — no separate
 *    edge kernel needed.
 *  - Vectorization axis: depth (k dimension). Each inner loop iteration loads a chunk of
 *    both A and B along depth, computing element-wise widened FMA.
 *
 *  - e2m3 GEMM: Integer arithmetic via LUT (5-bit magnitude → i8 value×16).
 *    B is pre-packed as signed i8. A is converted on-the-fly via `vluxei8` gather.
 *    Uses `vwmul` (i8→i16) then `vwadd_wv` (i32+=i16) for K-vectorized accumulation.
 *    Final result scaled by 1/256. Process 4 rows per tile (rows_per_tile=4).
 *  - e3m2 GEMM: Integer arithmetic via LUT (5-bit magnitude → i16 value×16).
 *    B is pre-packed as signed i16. A is converted on-the-fly via `vluxei16` gather.
 *    Uses `vwmacc` (i16×i16→i32) for K-vectorized widening MAC.
 *    Final result scaled by 1/256. Process 2 rows per tile (rows_per_tile=2, wider accumulator elements).
 *  - e4m3 GEMM: f32 LUT gather (7-bit magnitude → f32 bit pattern, 128 entries).
 *    B is pre-packed as f32. A is converted on-the-fly via `vluxei32` gather with
 *    sign injection (bit 7 → bit 31). Uses `vfwmacc_vv_f64m4` for f64 accumulation.
 *    Process 2 rows per tile (rows_per_tile=2, u32m2 gather + f64m4 accumulator is register-heavy).
 *  - e5m2 GEMM: Same f32 LUT gather approach as e4m3, different LUT contents.
 *    E5M2 has 5 exponent bits (wider range, lower precision than e4m3).
 *    Process 2 rows per tile (rows_per_tile=2).
 */
#ifndef NK_DOTS_RVV_H
#define NK_DOTS_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/dots/serial.h"
#include "numkong/cast/rvv.h" // `nk_bf16m1_to_f32m2_rvv_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  E2M3 magnitude LUT: 5-bit magnitude → unsigned value×16 (u8).
 *          Shared across scalar helper, packed kernel, and symmetric kernel.
 */
static nk_u8_t const nk_e2m3_magnitude_lut_rvv_[32] = {0,  2,  4,  6,  8,  10, 12, 14,  16,  18, 20,
                                                       22, 24, 26, 28, 30, 32, 36, 40,  44,  48, 52,
                                                       56, 60, 64, 72, 80, 88, 96, 104, 112, 120};

/**
 *  @brief  E3M2 magnitude LUT: 5-bit magnitude → unsigned value×16 (u16).
 *          Shared across scalar helper, packed kernel, and symmetric kernel.
 */
static nk_u16_t const nk_e3m2_magnitude_lut_rvv_[32] = {0,  1,   2,   3,   4,   5,   6,   7,   8,   10, 12,
                                                        14, 16,  20,  24,  28,  32,  40,  48,  56,  64, 80,
                                                        96, 112, 128, 160, 192, 224, 256, 320, 384, 448};

#pragma region Single Precision Floats

NK_PUBLIC nk_size_t nk_dots_packed_size_f32_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    // Break power-of-2 strides for cache associativity
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_f32_t) +
           column_count * sizeof(nk_f64_t); // per-column norms
}

NK_PUBLIC void nk_dots_pack_f32_rvv(nk_f32_t const *b, nk_size_t column_count, nk_size_t depth,
                                    nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_f32_t *packed = (nk_f32_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_f32_t const *src = (nk_f32_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_f32_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) dst[k] = src[k];
    }

    // Append per-column norms after packed data
    nk_f64_t *norms = (nk_f64_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_f32_t const *src = (nk_f32_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_f32_(src, depth);
    }
}

/**
 *  @brief  f32 packed GEMM kernel: C += A * B_packed^T with f64 widened accumulation.
 *
 *  Vectorizes over the depth dimension (k). For each (row, column) pair:
 *    acc_f64 = sum_k  f64(a[row][k]) * f64(b_packed[column][k])
 *  using `vfwmacc_vv_f64m4` which widens both operands from f32m2 to f64m4.
 *
 *  Register tile: process 4 rows per iteration (rows_per_tile=4).
 *  Each row loads its own A vector; B vector is shared across rows per depth chunk.
 */
NK_INTERNAL void nk_dots_packed_f32_rvv_aligned_(nk_f32_t const *a_matrix, void const *b_packed_buffer,
                                                 nk_f64_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                 nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                 nk_size_t c_stride_in_bytes) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_f32_t const *packed_data = (nk_f32_t const *)((char const *)b_packed_buffer +
                                                     sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_f64_t *c_row = (nk_f64_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // mr=4 register tile over rows
    nk_size_t row = 0;
    for (; row + 4 <= row_count; row += 4) {
        nk_f32_t const *a_row_0 = (nk_f32_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_f32_t const *a_row_1 = (nk_f32_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_f32_t const *a_row_2 = (nk_f32_t const *)((char const *)a_matrix + (row + 2) * a_stride_in_bytes);
        nk_f32_t const *a_row_3 = (nk_f32_t const *)((char const *)a_matrix + (row + 3) * a_stride_in_bytes);
        nk_f64_t *c_row_0 = (nk_f64_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_f64_t *c_row_1 = (nk_f64_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);
        nk_f64_t *c_row_2 = (nk_f64_t *)((char *)c_matrix + (row + 2) * c_stride_in_bytes);
        nk_f64_t *c_row_3 = (nk_f64_t *)((char *)c_matrix + (row + 3) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_0_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_1_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_2_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_3_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);
                vfloat32m2_t a_vector_0_f32m2 = __riscv_vle32_v_f32m2(a_row_0 + k, vector_length);
                vfloat32m2_t a_vector_1_f32m2 = __riscv_vle32_v_f32m2(a_row_1 + k, vector_length);
                vfloat32m2_t a_vector_2_f32m2 = __riscv_vle32_v_f32m2(a_row_2 + k, vector_length);
                vfloat32m2_t a_vector_3_f32m2 = __riscv_vle32_v_f32m2(a_row_3 + k, vector_length);
                accumulator_0_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_0_f64m4, a_vector_0_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_1_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_1_f64m4, a_vector_1_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_2_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_2_f64m4, a_vector_2_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_3_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_3_f64m4, a_vector_3_f32m2, b_vector_f32m2,
                                                                  vector_length);
            }

            // Horizontal reduce directly to f64
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row_0[column] = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_0_f64m4, zero_f64m1, vlmax));
            c_row_1[column] = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_1_f64m4, zero_f64m1, vlmax));
            c_row_2[column] = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_2_f64m4, zero_f64m1, vlmax));
            c_row_3[column] = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_3_f64m4, zero_f64m1, vlmax));
        }
    }
    // Remainder rows (mr < 4)
    for (; row < row_count; ++row) {
        nk_f32_t const *a_row = (nk_f32_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_f64_t *c_row = (nk_f64_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);
                vfloat32m2_t a_vector_f32m2 = __riscv_vle32_v_f32m2(a_row + k, vector_length);
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, a_vector_f32m2, b_vector_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row[column] = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
        }
    }
}

/**
 *  @brief  Public f32 packed GEMM wrapper matching the declared signature in dots.h.
 *
 *  Dispatches to the aligned kernel for all cases — RVV's `vsetvl` handles partial
 *  vectors naturally, so no separate edge kernel is needed.
 */
NK_PUBLIC void nk_dots_packed_f32_rvv(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t m, nk_size_t n,
                                      nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_f32_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric f32 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses f64 widened accumulation via `vfwmacc_vv_f64m4` for precision.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_f32_rvv(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                         nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_f32_t const *a_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_f32_t const *a_j = vectors + j * stride_elements;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vfloat32m2_t a_vector_f32m2 = __riscv_vle32_v_f32m2(a_i + k, vector_length);
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(a_j + k, vector_length);
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, a_vector_f32m2, b_vector_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            nk_f64_t dot = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Single Precision Floats

#pragma region Double Precision Floats

NK_PUBLIC nk_size_t nk_dots_packed_size_f64_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e64m4();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f64_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_f64_t) +
           column_count * sizeof(nk_f64_t); // per-column norms
}

NK_PUBLIC void nk_dots_pack_f64_rvv(nk_f64_t const *b, nk_size_t column_count, nk_size_t depth,
                                    nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e64m4();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f64_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_f64_t *packed = (nk_f64_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_f64_t const *src = (nk_f64_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_f64_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) dst[k] = src[k];
    }

    // Append per-column norms after packed data
    nk_f64_t *norms = (nk_f64_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_f64_t const *src = (nk_f64_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_f64_(src, depth);
    }
}

/**
 *  @brief  f64 packed GEMM kernel: C += A * B_packed^T with Kahan compensation.
 *
 *  Vectorizes over depth dimension k using `vfmul`+Kahan (vector-vector multiply).
 *  Uses Kahan summation over full depth to maintain precision.
 *  Register tile: process 2 rows per iteration (rows_per_tile=2, budget: 32 regs at LMUL=4).
 */
NK_INTERNAL void nk_dots_packed_f64_rvv_aligned_(nk_f64_t const *a_matrix, void const *b_packed_buffer,
                                                 nk_f64_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                 nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                 nk_size_t c_stride_in_bytes) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_f64_t const *packed_data = (nk_f64_t const *)((char const *)b_packed_buffer +
                                                     sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_f64_t *c_row = (nk_f64_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // Process 2 rows per tile (rows_per_tile=2, tighter register budget for f64 at LMUL=4)
    nk_size_t row = 0;
    for (; row + 2 <= row_count; row += 2) {
        nk_f64_t const *a_row_0 = (nk_f64_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_f64_t const *a_row_1 = (nk_f64_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_f64_t *c_row_0 = (nk_f64_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_f64_t *c_row_1 = (nk_f64_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f64_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
            vfloat64m4_t accumulator_0_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_1_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t compensation_0_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t compensation_1_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e64m4(remaining);
                vfloat64m4_t b_vector_f64m4 = __riscv_vle64_v_f64m4(b_column + k, vector_length);
                vfloat64m4_t a_vector_0_f64m4 = __riscv_vle64_v_f64m4(a_row_0 + k, vector_length);
                vfloat64m4_t a_vector_1_f64m4 = __riscv_vle64_v_f64m4(a_row_1 + k, vector_length);

                // Kahan step for row 0: product = a*b; corrected = product - comp; running = acc + corrected; comp =
                // (running - acc) - corrected; acc = running
                vfloat64m4_t product_0_f64m4 = __riscv_vfmul_vv_f64m4(a_vector_0_f64m4, b_vector_f64m4, vector_length);
                vfloat64m4_t corrected_term_0_f64m4 = __riscv_vfsub_vv_f64m4(product_0_f64m4, compensation_0_f64m4,
                                                                             vector_length);
                vfloat64m4_t running_sum_0_f64m4 = __riscv_vfadd_vv_f64m4_tu(accumulator_0_f64m4, accumulator_0_f64m4,
                                                                             corrected_term_0_f64m4, vector_length);
                compensation_0_f64m4 = __riscv_vfsub_vv_f64m4_tu(
                    compensation_0_f64m4,
                    __riscv_vfsub_vv_f64m4(running_sum_0_f64m4, accumulator_0_f64m4, vector_length),
                    corrected_term_0_f64m4, vector_length);
                accumulator_0_f64m4 = running_sum_0_f64m4;

                // Kahan step for row 1
                vfloat64m4_t product_1_f64m4 = __riscv_vfmul_vv_f64m4(a_vector_1_f64m4, b_vector_f64m4, vector_length);
                vfloat64m4_t corrected_term_1_f64m4 = __riscv_vfsub_vv_f64m4(product_1_f64m4, compensation_1_f64m4,
                                                                             vector_length);
                vfloat64m4_t running_sum_1_f64m4 = __riscv_vfadd_vv_f64m4_tu(accumulator_1_f64m4, accumulator_1_f64m4,
                                                                             corrected_term_1_f64m4, vector_length);
                compensation_1_f64m4 = __riscv_vfsub_vv_f64m4_tu(
                    compensation_1_f64m4,
                    __riscv_vfsub_vv_f64m4(running_sum_1_f64m4, accumulator_1_f64m4, vector_length),
                    corrected_term_1_f64m4, vector_length);
                accumulator_1_f64m4 = running_sum_1_f64m4;
            }

            // Horizontal reduce
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row_0[column] = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_0_f64m4, zero_f64m1, vlmax));
            c_row_1[column] = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_1_f64m4, zero_f64m1, vlmax));
        }
    }
    // Remainder rows
    for (; row < row_count; ++row) {
        nk_f64_t const *a_row = (nk_f64_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_f64_t *c_row = (nk_f64_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f64_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t compensation_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e64m4(remaining);
                vfloat64m4_t b_vector_f64m4 = __riscv_vle64_v_f64m4(b_column + k, vector_length);
                vfloat64m4_t a_vector_f64m4 = __riscv_vle64_v_f64m4(a_row + k, vector_length);

                vfloat64m4_t product_f64m4 = __riscv_vfmul_vv_f64m4(a_vector_f64m4, b_vector_f64m4, vector_length);
                vfloat64m4_t corrected_term_f64m4 = __riscv_vfsub_vv_f64m4(product_f64m4, compensation_f64m4,
                                                                           vector_length);
                vfloat64m4_t running_sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(accumulator_f64m4, accumulator_f64m4,
                                                                           corrected_term_f64m4, vector_length);
                compensation_f64m4 = __riscv_vfsub_vv_f64m4_tu(
                    compensation_f64m4, __riscv_vfsub_vv_f64m4(running_sum_f64m4, accumulator_f64m4, vector_length),
                    corrected_term_f64m4, vector_length);
                accumulator_f64m4 = running_sum_f64m4;
            }

            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row[column] = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
        }
    }
}

/**
 *  @brief  Public f64 packed GEMM wrapper matching the declared signature in dots.h.
 */
NK_PUBLIC void nk_dots_packed_f64_rvv(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t m, nk_size_t n,
                                      nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_f64_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric f64 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses Kahan compensation over full depth for precision.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_f64_rvv(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                         nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_f64_t const *a_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_f64_t const *a_j = vectors + j * stride_elements;
            nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t compensation_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e64m4(remaining);
                vfloat64m4_t a_vector_f64m4 = __riscv_vle64_v_f64m4(a_i + k, vector_length);
                vfloat64m4_t b_vector_f64m4 = __riscv_vle64_v_f64m4(a_j + k, vector_length);

                vfloat64m4_t product_f64m4 = __riscv_vfmul_vv_f64m4(a_vector_f64m4, b_vector_f64m4, vector_length);
                vfloat64m4_t corrected_term_f64m4 = __riscv_vfsub_vv_f64m4(product_f64m4, compensation_f64m4,
                                                                           vector_length);
                vfloat64m4_t running_sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(accumulator_f64m4, accumulator_f64m4,
                                                                           corrected_term_f64m4, vector_length);
                compensation_f64m4 = __riscv_vfsub_vv_f64m4_tu(
                    compensation_f64m4, __riscv_vfsub_vv_f64m4(running_sum_f64m4, accumulator_f64m4, vector_length),
                    corrected_term_f64m4, vector_length);
                accumulator_f64m4 = running_sum_f64m4;
            }

            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            nk_f64_t dot = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Double Precision Floats

#pragma region Micro Precision E2M3

/**
 *  @brief  Scalar conversion helper: e2m3 byte → signed i8 (value × 16).
 *
 *  Extracts 5-bit magnitude, looks up in LUT, applies sign from bit 5.
 *  Every e2m3 value × 16 is an exact integer in [-120, +120], fitting in i8.
 */
NK_INTERNAL nk_i8_t nk_e2m3_to_i8_rvv_(nk_u8_t raw) {
    nk_u8_t magnitude = raw & 0x1Fu;
    nk_i8_t val = (nk_i8_t)nk_e2m3_magnitude_lut_rvv_[magnitude];
    return (raw & 0x20u) ? (nk_i8_t)(-val) : val;
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e8m1();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_i8_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_i8_t) +
           column_count * sizeof(nk_f32_t); // per-column norms
}

/**
 *  @brief  Pack B matrix from e2m3 to signed i8 (value × 16) for integer dot product.
 *
 *  Each e2m3 byte is converted to a signed i8 via scalar LUT lookup.
 *  Padding values are zeroed. Column-panel layout with depth-contiguous storage.
 */
NK_PUBLIC void nk_dots_pack_e2m3_rvv(nk_e2m3_t const *b, nk_size_t column_count, nk_size_t depth,
                                     nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e8m1();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_i8_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_i8_t *packed = (nk_i8_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_u8_t const *src = (nk_u8_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_i8_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) dst[k] = nk_e2m3_to_i8_rvv_(src[k]);
    }

    // Append per-column norms after packed data
    nk_f32_t *norms = (nk_f32_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_e2m3_t const *src = (nk_e2m3_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_e2m3_(src, depth);
    }
}

/**
 *  @brief  e2m3 packed GEMM kernel: C += A * B_packed^T with integer i8 LUT arithmetic.
 *
 *  Vectorizes over the depth dimension (k). For each (row, column) pair:
 *    - Load raw e2m3 bytes from A, extract magnitude via `vluxei8` gather LUT
 *    - Apply sign from bit 5 via masked negate to produce signed i8 A values
 *    - Load pre-packed signed i8 values from B
 *    - Widening multiply i8×i8 → i16, then widen-accumulate i32 += i16
 *    - Final result = i32_sum / 256.0f
 *
 *  Register tile: process 4 rows per iteration (rows_per_tile=4).
 *  The LUT gather on A magnitudes uses `vluxei8_v_u8m1` (byte-indexed byte gather).
 */
NK_INTERNAL void nk_dots_packed_e2m3_rvv_aligned_(nk_e2m3_t const *a_matrix, void const *b_packed_buffer,
                                                  nk_f32_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                  nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                  nk_size_t c_stride_in_bytes) {
    nk_f32_t const lut_scale_reciprocal = 1.0f / 256.0f;

    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_i8_t const *packed_data = (nk_i8_t const *)((char const *)b_packed_buffer +
                                                   sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // mr=4 register tile over rows
    nk_size_t row = 0;
    for (; row + 4 <= row_count; row += 4) {
        nk_u8_t const *a_row_0 = (nk_u8_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_u8_t const *a_row_1 = (nk_u8_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_u8_t const *a_row_2 = (nk_u8_t const *)((char const *)a_matrix + (row + 2) * a_stride_in_bytes);
        nk_u8_t const *a_row_3 = (nk_u8_t const *)((char const *)a_matrix + (row + 3) * a_stride_in_bytes);
        nk_f32_t *c_row_0 = (nk_f32_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_f32_t *c_row_1 = (nk_f32_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);
        nk_f32_t *c_row_2 = (nk_f32_t *)((char *)c_matrix + (row + 2) * c_stride_in_bytes);
        nk_f32_t *c_row_3 = (nk_f32_t *)((char *)c_matrix + (row + 3) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_i8_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vint32m4_t accumulator_0_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            vint32m4_t accumulator_1_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            vint32m4_t accumulator_2_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            vint32m4_t accumulator_3_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e8m1(remaining);

                // Load pre-packed i8 B values
                vint8m1_t b_vector_i8m1 = __riscv_vle8_v_i8m1(b_column + k, vector_length);

                // Load raw e2m3 bytes from each A row and convert via LUT
                vuint8m1_t raw0_u8m1 = __riscv_vle8_v_u8m1(a_row_0 + k, vector_length);
                vuint8m1_t raw1_u8m1 = __riscv_vle8_v_u8m1(a_row_1 + k, vector_length);
                vuint8m1_t raw2_u8m1 = __riscv_vle8_v_u8m1(a_row_2 + k, vector_length);
                vuint8m1_t raw3_u8m1 = __riscv_vle8_v_u8m1(a_row_3 + k, vector_length);

                // Extract magnitudes and gather from LUT
                vuint8m1_t mag0_u8m1 = __riscv_vand_vx_u8m1(raw0_u8m1, 0x1F, vector_length);
                vuint8m1_t mag1_u8m1 = __riscv_vand_vx_u8m1(raw1_u8m1, 0x1F, vector_length);
                vuint8m1_t mag2_u8m1 = __riscv_vand_vx_u8m1(raw2_u8m1, 0x1F, vector_length);
                vuint8m1_t mag3_u8m1 = __riscv_vand_vx_u8m1(raw3_u8m1, 0x1F, vector_length);
                vuint8m1_t uval0_u8m1 = __riscv_vluxei8_v_u8m1(nk_e2m3_magnitude_lut_rvv_, mag0_u8m1, vector_length);
                vuint8m1_t uval1_u8m1 = __riscv_vluxei8_v_u8m1(nk_e2m3_magnitude_lut_rvv_, mag1_u8m1, vector_length);
                vuint8m1_t uval2_u8m1 = __riscv_vluxei8_v_u8m1(nk_e2m3_magnitude_lut_rvv_, mag2_u8m1, vector_length);
                vuint8m1_t uval3_u8m1 = __riscv_vluxei8_v_u8m1(nk_e2m3_magnitude_lut_rvv_, mag3_u8m1, vector_length);

                // Apply sign to A: negate where bit 5 is set.
                // B is already signed from packing, so A sign completes the product sign.
                vint8m1_t a_vector_0_i8m1 = __riscv_vreinterpret_v_u8m1_i8m1(uval0_u8m1);
                vbool8_t negated_0_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw0_u8m1, 0x20, vector_length),
                                                                 0, vector_length);
                a_vector_0_i8m1 = __riscv_vneg_v_i8m1_mu(negated_0_b8, a_vector_0_i8m1, a_vector_0_i8m1, vector_length);

                vint8m1_t a_vector_1_i8m1 = __riscv_vreinterpret_v_u8m1_i8m1(uval1_u8m1);
                vbool8_t negated_1_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw1_u8m1, 0x20, vector_length),
                                                                 0, vector_length);
                a_vector_1_i8m1 = __riscv_vneg_v_i8m1_mu(negated_1_b8, a_vector_1_i8m1, a_vector_1_i8m1, vector_length);

                vint8m1_t a_vector_2_i8m1 = __riscv_vreinterpret_v_u8m1_i8m1(uval2_u8m1);
                vbool8_t negated_2_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw2_u8m1, 0x20, vector_length),
                                                                 0, vector_length);
                a_vector_2_i8m1 = __riscv_vneg_v_i8m1_mu(negated_2_b8, a_vector_2_i8m1, a_vector_2_i8m1, vector_length);

                vint8m1_t a_vector_3_i8m1 = __riscv_vreinterpret_v_u8m1_i8m1(uval3_u8m1);
                vbool8_t negated_3_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw3_u8m1, 0x20, vector_length),
                                                                 0, vector_length);
                a_vector_3_i8m1 = __riscv_vneg_v_i8m1_mu(negated_3_b8, a_vector_3_i8m1, a_vector_3_i8m1, vector_length);

                // Widening multiply: i8×i8 → i16, then accumulate: i32 += i16
                vint16m2_t product_0_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_0_i8m1, b_vector_i8m1, vector_length);
                vint16m2_t product_1_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_1_i8m1, b_vector_i8m1, vector_length);
                vint16m2_t product_2_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_2_i8m1, b_vector_i8m1, vector_length);
                vint16m2_t product_3_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_3_i8m1, b_vector_i8m1, vector_length);
                accumulator_0_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_0_i32m4, accumulator_0_i32m4,
                                                                product_0_i16m2, vector_length);
                accumulator_1_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_1_i32m4, accumulator_1_i32m4,
                                                                product_1_i16m2, vector_length);
                accumulator_2_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_2_i32m4, accumulator_2_i32m4,
                                                                product_2_i16m2, vector_length);
                accumulator_3_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_3_i32m4, accumulator_3_i32m4,
                                                                product_3_i16m2, vector_length);
            }

            // Horizontal reduce and convert to f32 with scaling
            vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            c_row_0[column] = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                                  __riscv_vredsum_vs_i32m4_i32m1(accumulator_0_i32m4, zero_i32m1, vlmax)) *
                              lut_scale_reciprocal;
            c_row_1[column] = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                                  __riscv_vredsum_vs_i32m4_i32m1(accumulator_1_i32m4, zero_i32m1, vlmax)) *
                              lut_scale_reciprocal;
            c_row_2[column] = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                                  __riscv_vredsum_vs_i32m4_i32m1(accumulator_2_i32m4, zero_i32m1, vlmax)) *
                              lut_scale_reciprocal;
            c_row_3[column] = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                                  __riscv_vredsum_vs_i32m4_i32m1(accumulator_3_i32m4, zero_i32m1, vlmax)) *
                              lut_scale_reciprocal;
        }
    }
    // Remainder rows (mr < 4)
    for (; row < row_count; ++row) {
        nk_u8_t const *a_row = (nk_u8_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_i8_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vint32m4_t accumulator_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e8m1(remaining);
                vint8m1_t b_vector_i8m1 = __riscv_vle8_v_i8m1(b_column + k, vector_length);
                vuint8m1_t raw_a_u8m1 = __riscv_vle8_v_u8m1(a_row + k, vector_length);
                vuint8m1_t mag_a_u8m1 = __riscv_vand_vx_u8m1(raw_a_u8m1, 0x1F, vector_length);
                vuint8m1_t uval_a_u8m1 = __riscv_vluxei8_v_u8m1(nk_e2m3_magnitude_lut_rvv_, mag_a_u8m1, vector_length);
                vint8m1_t a_vector_i8m1 = __riscv_vreinterpret_v_u8m1_i8m1(uval_a_u8m1);
                vbool8_t negated_a_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw_a_u8m1, 0x20, vector_length),
                                                                 0, vector_length);
                a_vector_i8m1 = __riscv_vneg_v_i8m1_mu(negated_a_b8, a_vector_i8m1, a_vector_i8m1, vector_length);
                vint16m2_t product_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_i8m1, b_vector_i8m1, vector_length);
                accumulator_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_i32m4, accumulator_i32m4, product_i16m2,
                                                              vector_length);
            }
            vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            c_row[column] = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                                __riscv_vredsum_vs_i32m4_i32m1(accumulator_i32m4, zero_i32m1, vlmax)) *
                            lut_scale_reciprocal;
        }
    }
}

/**
 *  @brief  Public e2m3 packed GEMM wrapper matching the declared signature in dots.h.
 */
NK_PUBLIC void nk_dots_packed_e2m3_rvv(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_e2m3_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric e2m3 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses integer i8 LUT arithmetic with i32 accumulation, scaled by 1/256.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_e2m3_rvv(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
    nk_f32_t const lut_scale_reciprocal = 1.0f / 256.0f;

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_u8_t const *a_i = (nk_u8_t const *)vectors + i * stride;
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_u8_t const *a_j = (nk_u8_t const *)vectors + j * stride;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vint32m4_t accumulator_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e8m1(remaining);
                vuint8m1_t raw_i_u8m1 = __riscv_vle8_v_u8m1(a_i + k, vector_length);
                vuint8m1_t raw_j_u8m1 = __riscv_vle8_v_u8m1(a_j + k, vector_length);

                // Extract magnitudes and gather from LUT
                vuint8m1_t mag_i_u8m1 = __riscv_vand_vx_u8m1(raw_i_u8m1, 0x1F, vector_length);
                vuint8m1_t mag_j_u8m1 = __riscv_vand_vx_u8m1(raw_j_u8m1, 0x1F, vector_length);
                vuint8m1_t uval_i_u8m1 = __riscv_vluxei8_v_u8m1(nk_e2m3_magnitude_lut_rvv_, mag_i_u8m1, vector_length);
                vuint8m1_t uval_j_u8m1 = __riscv_vluxei8_v_u8m1(nk_e2m3_magnitude_lut_rvv_, mag_j_u8m1, vector_length);

                // Combined sign: XOR sign bits → conditional negate on B side
                vuint8m1_t sign_xor_u8m1 = __riscv_vand_vx_u8m1(
                    __riscv_vxor_vv_u8m1(raw_i_u8m1, raw_j_u8m1, vector_length), 0x20, vector_length);
                vbool8_t negate_b8 = __riscv_vmsne_vx_u8m1_b8(sign_xor_u8m1, 0, vector_length);
                vint8m1_t val_i_i8m1 = __riscv_vreinterpret_v_u8m1_i8m1(uval_i_u8m1);
                vint8m1_t val_j_i8m1 = __riscv_vreinterpret_v_u8m1_i8m1(uval_j_u8m1);
                val_j_i8m1 = __riscv_vneg_v_i8m1_mu(negate_b8, val_j_i8m1, val_j_i8m1, vector_length);

                // Widening multiply: i8×i8 → i16, then accumulate: i32 += i16
                vint16m2_t product_i16m2 = __riscv_vwmul_vv_i16m2(val_i_i8m1, val_j_i8m1, vector_length);
                accumulator_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_i32m4, accumulator_i32m4, product_i16m2,
                                                              vector_length);
            }
            vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            nk_f32_t dot = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                               __riscv_vredsum_vs_i32m4_i32m1(accumulator_i32m4, zero_i32m1, vlmax)) *
                           lut_scale_reciprocal;
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Micro Precision E2M3

#pragma region Micro Precision E3M2

/**
 *  @brief  Scalar conversion helper: e3m2 byte → signed i16 (value × 16).
 *
 *  Extracts 5-bit magnitude, looks up in LUT, applies sign from bit 5.
 *  Every e3m2 value × 16 is an exact integer in [-448, +448], requiring i16.
 */
NK_INTERNAL nk_i16_t nk_e3m2_to_i16_rvv_(nk_u8_t raw) {
    nk_u8_t magnitude = raw & 0x1Fu;
    nk_i16_t val = (nk_i16_t)nk_e3m2_magnitude_lut_rvv_[magnitude];
    return (raw & 0x20u) ? (nk_i16_t)(-val) : val;
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e16m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_i16_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_i16_t) +
           column_count * sizeof(nk_f32_t); // per-column norms
}

/**
 *  @brief  Pack B matrix from e3m2 to signed i16 (value × 16) for integer dot product.
 *
 *  Each e3m2 byte is converted to a signed i16 via scalar LUT lookup.
 *  Padding values are zeroed. Column-panel layout with depth-contiguous storage.
 */
NK_PUBLIC void nk_dots_pack_e3m2_rvv(nk_e3m2_t const *b, nk_size_t column_count, nk_size_t depth,
                                     nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e16m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_i16_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_i16_t *packed = (nk_i16_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_u8_t const *src = (nk_u8_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_i16_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) dst[k] = nk_e3m2_to_i16_rvv_(src[k]);
    }

    // Append per-column norms after packed data
    nk_f32_t *norms = (nk_f32_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_e3m2_t const *src = (nk_e3m2_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_e3m2_(src, depth);
    }
}

/**
 *  @brief  e3m2 packed GEMM kernel: C += A * B_packed^T with integer i16 LUT arithmetic.
 *
 *  Vectorizes over the depth dimension (k). For each (row, column) pair:
 *    - Load raw e3m2 bytes from A, convert to signed i16 via `vluxei16` gather LUT
 *    - Load pre-packed i16 values from B
 *    - Widening multiply-accumulate: i16×i16 → i32 via `vwmacc`
 *    - Final result = i32_sum / 256.0f
 *
 *  Register tile: process 2 rows per iteration (rows_per_tile=2, wider i16/i32 elements reduce VL).
 *  The LUT gather on A magnitudes uses `vluxei16_v_u16m2` (16-bit indexed 16-bit gather).
 */
NK_INTERNAL void nk_dots_packed_e3m2_rvv_aligned_(nk_e3m2_t const *a_matrix, void const *b_packed_buffer,
                                                  nk_f32_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                  nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                  nk_size_t c_stride_in_bytes) {
    nk_f32_t const lut_scale_reciprocal = 1.0f / 256.0f;

    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_i16_t const *packed_data = (nk_i16_t const *)((char const *)b_packed_buffer +
                                                     sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // mr=2 register tile (i16 at LMUL=2 and i32 at LMUL=4 leaves fewer spare registers)
    nk_size_t row = 0;
    for (; row + 2 <= row_count; row += 2) {
        nk_u8_t const *a_row_0 = (nk_u8_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_u8_t const *a_row_1 = (nk_u8_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_f32_t *c_row_0 = (nk_f32_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_f32_t *c_row_1 = (nk_f32_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_i16_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vint32m4_t accumulator_0_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            vint32m4_t accumulator_1_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e16m2(remaining);

                // Load pre-packed i16 B values
                vint16m2_t b_vector_i16m2 = __riscv_vle16_v_i16m2(b_column + k, vector_length);

                // Load raw e3m2 bytes from each A row
                vuint8m1_t raw0_u8m1 = __riscv_vle8_v_u8m1(a_row_0 + k, vector_length);
                vuint8m1_t raw1_u8m1 = __riscv_vle8_v_u8m1(a_row_1 + k, vector_length);

                // Extract magnitudes, zero-extend to u16, compute byte offsets for i16 LUT gather
                vuint8m1_t mag0_u8m1 = __riscv_vand_vx_u8m1(raw0_u8m1, 0x1F, vector_length);
                vuint8m1_t mag1_u8m1 = __riscv_vand_vx_u8m1(raw1_u8m1, 0x1F, vector_length);
                vuint16m2_t idx0_u16m2 = __riscv_vzext_vf2_u16m2(mag0_u8m1, vector_length);
                vuint16m2_t idx1_u16m2 = __riscv_vzext_vf2_u16m2(mag1_u8m1, vector_length);
                vuint16m2_t off0_u16m2 = __riscv_vsll_vx_u16m2(idx0_u16m2, 1,
                                                               vector_length); // byte offsets = index × 2
                vuint16m2_t off1_u16m2 = __riscv_vsll_vx_u16m2(idx1_u16m2, 1, vector_length);

                // Gather unsigned magnitudes from i16 LUT
                vuint16m2_t uval0_u16m2 = __riscv_vluxei16_v_u16m2(nk_e3m2_magnitude_lut_rvv_, off0_u16m2,
                                                                   vector_length);
                vuint16m2_t uval1_u16m2 = __riscv_vluxei16_v_u16m2(nk_e3m2_magnitude_lut_rvv_, off1_u16m2,
                                                                   vector_length);

                // Apply sign: negate where bit 5 is set
                vuint8m1_t sign0_u8m1 = __riscv_vand_vx_u8m1(raw0_u8m1, 0x20, vector_length);
                vuint8m1_t sign1_u8m1 = __riscv_vand_vx_u8m1(raw1_u8m1, 0x20, vector_length);
                vbool8_t negated_0_b8 = __riscv_vmsne_vx_u8m1_b8(sign0_u8m1, 0, vector_length);
                vbool8_t negated_1_b8 = __riscv_vmsne_vx_u8m1_b8(sign1_u8m1, 0, vector_length);

                vint16m2_t a_vector_0_i16m2 = __riscv_vreinterpret_v_u16m2_i16m2(uval0_u16m2);
                a_vector_0_i16m2 = __riscv_vneg_v_i16m2_mu(negated_0_b8, a_vector_0_i16m2, a_vector_0_i16m2,
                                                           vector_length);
                vint16m2_t a_vector_1_i16m2 = __riscv_vreinterpret_v_u16m2_i16m2(uval1_u16m2);
                a_vector_1_i16m2 = __riscv_vneg_v_i16m2_mu(negated_1_b8, a_vector_1_i16m2, a_vector_1_i16m2,
                                                           vector_length);

                // Widening multiply-accumulate: i16×i16 → i32
                accumulator_0_i32m4 = __riscv_vwmacc_vv_i32m4_tu(accumulator_0_i32m4, a_vector_0_i16m2, b_vector_i16m2,
                                                                 vector_length);
                accumulator_1_i32m4 = __riscv_vwmacc_vv_i32m4_tu(accumulator_1_i32m4, a_vector_1_i16m2, b_vector_i16m2,
                                                                 vector_length);
            }

            // Horizontal reduce and convert to f32 with scaling
            vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            c_row_0[column] = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                                  __riscv_vredsum_vs_i32m4_i32m1(accumulator_0_i32m4, zero_i32m1, vlmax)) *
                              lut_scale_reciprocal;
            c_row_1[column] = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                                  __riscv_vredsum_vs_i32m4_i32m1(accumulator_1_i32m4, zero_i32m1, vlmax)) *
                              lut_scale_reciprocal;
        }
    }
    // Remainder rows
    for (; row < row_count; ++row) {
        nk_u8_t const *a_row = (nk_u8_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_i16_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vint32m4_t accumulator_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e16m2(remaining);
                vint16m2_t b_vector_i16m2 = __riscv_vle16_v_i16m2(b_column + k, vector_length);
                vuint8m1_t raw_a_u8m1 = __riscv_vle8_v_u8m1(a_row + k, vector_length);
                vuint8m1_t mag_a_u8m1 = __riscv_vand_vx_u8m1(raw_a_u8m1, 0x1F, vector_length);
                vuint16m2_t idx_a_u16m2 = __riscv_vzext_vf2_u16m2(mag_a_u8m1, vector_length);
                vuint16m2_t off_a_u16m2 = __riscv_vsll_vx_u16m2(idx_a_u16m2, 1, vector_length);
                vuint16m2_t uval_a_u16m2 = __riscv_vluxei16_v_u16m2(nk_e3m2_magnitude_lut_rvv_, off_a_u16m2,
                                                                    vector_length);
                vint16m2_t a_vector_i16m2 = __riscv_vreinterpret_v_u16m2_i16m2(uval_a_u16m2);
                vbool8_t negated_a_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw_a_u8m1, 0x20, vector_length),
                                                                 0, vector_length);
                a_vector_i16m2 = __riscv_vneg_v_i16m2_mu(negated_a_b8, a_vector_i16m2, a_vector_i16m2, vector_length);
                accumulator_i32m4 = __riscv_vwmacc_vv_i32m4_tu(accumulator_i32m4, a_vector_i16m2, b_vector_i16m2,
                                                               vector_length);
            }
            vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            c_row[column] = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                                __riscv_vredsum_vs_i32m4_i32m1(accumulator_i32m4, zero_i32m1, vlmax)) *
                            lut_scale_reciprocal;
        }
    }
}

/**
 *  @brief  Public e3m2 packed GEMM wrapper matching the declared signature in dots.h.
 */
NK_PUBLIC void nk_dots_packed_e3m2_rvv(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_e3m2_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric e3m2 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses integer i16 LUT arithmetic with i32 widening MAC, scaled by 1/256.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_e3m2_rvv(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
    nk_f32_t const lut_scale_reciprocal = 1.0f / 256.0f;

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_u8_t const *a_i = (nk_u8_t const *)vectors + i * stride;
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_u8_t const *a_j = (nk_u8_t const *)vectors + j * stride;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vint32m4_t accumulator_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e16m2(remaining);
                vuint8m1_t raw_i_u8m1 = __riscv_vle8_v_u8m1(a_i + k, vector_length);
                vuint8m1_t raw_j_u8m1 = __riscv_vle8_v_u8m1(a_j + k, vector_length);

                // Extract magnitudes, zero-extend to u16, compute byte offsets
                vuint8m1_t mag_i_u8m1 = __riscv_vand_vx_u8m1(raw_i_u8m1, 0x1F, vector_length);
                vuint8m1_t mag_j_u8m1 = __riscv_vand_vx_u8m1(raw_j_u8m1, 0x1F, vector_length);
                vuint16m2_t idx_i_u16m2 = __riscv_vzext_vf2_u16m2(mag_i_u8m1, vector_length);
                vuint16m2_t idx_j_u16m2 = __riscv_vzext_vf2_u16m2(mag_j_u8m1, vector_length);
                vuint16m2_t off_i_u16m2 = __riscv_vsll_vx_u16m2(idx_i_u16m2, 1, vector_length);
                vuint16m2_t off_j_u16m2 = __riscv_vsll_vx_u16m2(idx_j_u16m2, 1, vector_length);

                // Gather unsigned magnitudes
                vuint16m2_t uval_i_u16m2 = __riscv_vluxei16_v_u16m2(nk_e3m2_magnitude_lut_rvv_, off_i_u16m2,
                                                                    vector_length);
                vuint16m2_t uval_j_u16m2 = __riscv_vluxei16_v_u16m2(nk_e3m2_magnitude_lut_rvv_, off_j_u16m2,
                                                                    vector_length);

                // Apply individual signs
                vuint8m1_t sign_i_u8m1 = __riscv_vand_vx_u8m1(raw_i_u8m1, 0x20, vector_length);
                vuint8m1_t sign_j_u8m1 = __riscv_vand_vx_u8m1(raw_j_u8m1, 0x20, vector_length);
                vbool8_t negated_i_b8 = __riscv_vmsne_vx_u8m1_b8(sign_i_u8m1, 0, vector_length);
                vbool8_t negated_j_b8 = __riscv_vmsne_vx_u8m1_b8(sign_j_u8m1, 0, vector_length);

                vint16m2_t val_i_i16m2 = __riscv_vreinterpret_v_u16m2_i16m2(uval_i_u16m2);
                val_i_i16m2 = __riscv_vneg_v_i16m2_mu(negated_i_b8, val_i_i16m2, val_i_i16m2, vector_length);
                vint16m2_t val_j_i16m2 = __riscv_vreinterpret_v_u16m2_i16m2(uval_j_u16m2);
                val_j_i16m2 = __riscv_vneg_v_i16m2_mu(negated_j_b8, val_j_i16m2, val_j_i16m2, vector_length);

                // Widening multiply-accumulate: i16×i16 → i32
                accumulator_i32m4 = __riscv_vwmacc_vv_i32m4_tu(accumulator_i32m4, val_i_i16m2, val_j_i16m2,
                                                               vector_length);
            }
            vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            nk_f32_t dot = (nk_f32_t)__riscv_vmv_x_s_i32m1_i32(
                               __riscv_vredsum_vs_i32m4_i32m1(accumulator_i32m4, zero_i32m1, vlmax)) *
                           lut_scale_reciprocal;
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Micro Precision E3M2

#pragma region Brain Float 16

/**
 *  @brief  Compute the packed buffer size for bf16 GEMM (B stored as f32).
 *
 *  VL is determined by `__riscv_vsetvlmax_e32m2()` since B is stored as f32.
 *  Layout: column-panel with depth-contiguous f32 values, cache-line padding.
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    // Break power-of-2 strides for cache associativity
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_f32_t) +
           column_count * sizeof(nk_f32_t); // per-column norms
}

/**
 *  @brief  Pack B matrix from bf16 to f32 for widened dot product.
 *
 *  Each bf16 value is converted to f32 via bit shift (bf16 is the upper 16 bits of f32).
 *  Padding values are zeroed. Column-panel layout with depth-contiguous storage.
 */
NK_PUBLIC void nk_dots_pack_bf16_rvv(nk_bf16_t const *b, nk_size_t column_count, nk_size_t depth,
                                     nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_f32_t *packed = (nk_f32_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_u16_t const *src = (nk_u16_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_f32_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) {
            union {
                nk_u32_t u;
                nk_f32_t f;
            } conv;
            conv.u = (nk_u32_t)src[k] << 16;
            dst[k] = conv.f;
        }
    }

    // Append per-column norms after packed data
    nk_f32_t *norms = (nk_f32_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_bf16_t const *src = (nk_bf16_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_bf16_(src, depth);
    }
}

/**
 *  @brief  bf16 packed GEMM kernel: C += A * B_packed^T with f64 widened accumulation.
 *
 *  Vectorizes over the depth dimension (k). For each (row, column) pair:
 *    - Load A as u16m1 and convert to f32m2 via `nk_bf16m1_to_f32m2_rvv_`
 *    - Load B as f32m2 directly (pre-packed)
 *    - Accumulate via `vfwmacc_vv_f64m4` which widens both f32 operands to f64
 *    - Horizontal reduce and narrow to f32 on store
 *
 *  Register tile: process 4 rows per iteration (rows_per_tile=4).
 */
NK_INTERNAL void nk_dots_packed_bf16_rvv_aligned_(nk_bf16_t const *a_matrix, void const *b_packed_buffer,
                                                  nk_f32_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                  nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                  nk_size_t c_stride_in_bytes) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_f32_t const *packed_data = (nk_f32_t const *)((char const *)b_packed_buffer +
                                                     sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // mr=4 register tile over rows
    nk_size_t row = 0;
    for (; row + 4 <= row_count; row += 4) {
        nk_u16_t const *a_row_0 = (nk_u16_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_u16_t const *a_row_1 = (nk_u16_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_u16_t const *a_row_2 = (nk_u16_t const *)((char const *)a_matrix + (row + 2) * a_stride_in_bytes);
        nk_u16_t const *a_row_3 = (nk_u16_t const *)((char const *)a_matrix + (row + 3) * a_stride_in_bytes);
        nk_f32_t *c_row_0 = (nk_f32_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_f32_t *c_row_1 = (nk_f32_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);
        nk_f32_t *c_row_2 = (nk_f32_t *)((char *)c_matrix + (row + 2) * c_stride_in_bytes);
        nk_f32_t *c_row_3 = (nk_f32_t *)((char *)c_matrix + (row + 3) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_0_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_1_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_2_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_3_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);
                // Load A as u16m1 and convert to f32m2
                vuint16m1_t a_raw_0_u16m1 = __riscv_vle16_v_u16m1(a_row_0 + k, vector_length);
                vfloat32m2_t a_vector_0_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_raw_0_u16m1, vector_length);
                vuint16m1_t a_raw_1_u16m1 = __riscv_vle16_v_u16m1(a_row_1 + k, vector_length);
                vfloat32m2_t a_vector_1_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_raw_1_u16m1, vector_length);
                vuint16m1_t a_raw_2_u16m1 = __riscv_vle16_v_u16m1(a_row_2 + k, vector_length);
                vfloat32m2_t a_vector_2_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_raw_2_u16m1, vector_length);
                vuint16m1_t a_raw_3_u16m1 = __riscv_vle16_v_u16m1(a_row_3 + k, vector_length);
                vfloat32m2_t a_vector_3_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_raw_3_u16m1, vector_length);
                accumulator_0_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_0_f64m4, a_vector_0_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_1_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_1_f64m4, a_vector_1_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_2_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_2_f64m4, a_vector_2_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_3_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_3_f64m4, a_vector_3_f32m2, b_vector_f32m2,
                                                                  vector_length);
            }

            // Horizontal reduce and narrow to f32
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row_0[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_0_f64m4, zero_f64m1, vlmax));
            c_row_1[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_1_f64m4, zero_f64m1, vlmax));
            c_row_2[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_2_f64m4, zero_f64m1, vlmax));
            c_row_3[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_3_f64m4, zero_f64m1, vlmax));
        }
    }
    // Remainder rows (mr < 4)
    for (; row < row_count; ++row) {
        nk_u16_t const *a_row = (nk_u16_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);
                vuint16m1_t a_raw_u16m1 = __riscv_vle16_v_u16m1(a_row + k, vector_length);
                vfloat32m2_t a_vector_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_raw_u16m1, vector_length);
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, a_vector_f32m2, b_vector_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
        }
    }
}

/**
 *  @brief  Public bf16 packed GEMM wrapper matching the declared signature in dots.h.
 *
 *  Dispatches to the aligned kernel for all cases — RVV's `vsetvl` handles partial
 *  vectors naturally, so no separate edge kernel is needed.
 */
NK_PUBLIC void nk_dots_packed_bf16_rvv(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_bf16_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric bf16 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses f64 widened accumulation via `vfwmacc_vv_f64m4` for precision.
 *  Both inputs are bf16, loaded as u16 and converted to f32 via `nk_bf16m1_to_f32m2_rvv_`.
 *  Stride is in bytes.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_bf16_rvv(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_u16_t const *a_i = (nk_u16_t const *)((char const *)vectors + i * stride);
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_u16_t const *a_j = (nk_u16_t const *)((char const *)vectors + j * stride);
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vuint16m1_t a_raw_u16m1 = __riscv_vle16_v_u16m1(a_i + k, vector_length);
                vfloat32m2_t a_vector_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_raw_u16m1, vector_length);
                vuint16m1_t b_raw_u16m1 = __riscv_vle16_v_u16m1(a_j + k, vector_length);
                vfloat32m2_t b_vector_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_raw_u16m1, vector_length);
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, a_vector_f32m2, b_vector_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            nk_f32_t dot = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Brain Float 16

#pragma region Half Precision Floats

/**
 *  @brief  Compute the packed buffer size for f16 GEMM (B stored as f32).
 *
 *  VL is determined by `__riscv_vsetvlmax_e32m2()` since B is stored as f32.
 *  Layout: column-panel with depth-contiguous f32 values, cache-line padding.
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    // Break power-of-2 strides for cache associativity
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_f32_t) +
           column_count * sizeof(nk_f32_t); // per-column norms
}

/**
 *  @brief  Pack B matrix from f16 to f32 for widened dot product.
 *
 *  Each f16 value is converted to f32 via `nk_f16_to_f32_serial`.
 *  Padding values are zeroed. Column-panel layout with depth-contiguous storage.
 */
NK_PUBLIC void nk_dots_pack_f16_rvv(nk_f16_t const *b, nk_size_t column_count, nk_size_t depth,
                                    nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_f32_t *packed = (nk_f32_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_f16_t const *src = (nk_f16_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_f32_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) nk_f16_to_f32_serial(&src[k], &dst[k]);
    }

    // Append per-column norms after packed data
    nk_f32_t *norms = (nk_f32_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_f16_t const *src = (nk_f16_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_f16_(src, depth);
    }
}

/**
 *  @brief  f16 packed GEMM kernel: C += A * B_packed^T with f64 widened accumulation.
 *
 *  Vectorizes over the depth dimension (k). For each (row, column) pair:
 *    - Load A as u16m1 and convert to f32m2 via `nk_f16m1_to_f32m2_rvv_`
 *    - Load B as f32m2 directly (pre-packed)
 *    - Accumulate via `vfwmacc_vv_f64m4` which widens both f32 operands to f64
 *    - Horizontal reduce and narrow to f32 on store
 *
 *  Register tile: process 4 rows per iteration (rows_per_tile=4).
 */
NK_INTERNAL void nk_dots_packed_f16_rvv_aligned_(nk_f16_t const *a_matrix, void const *b_packed_buffer,
                                                 nk_f32_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                 nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                 nk_size_t c_stride_in_bytes) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_f32_t const *packed_data = (nk_f32_t const *)((char const *)b_packed_buffer +
                                                     sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // mr=4 register tile over rows
    nk_size_t row = 0;
    for (; row + 4 <= row_count; row += 4) {
        nk_u16_t const *a_row_0 = (nk_u16_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_u16_t const *a_row_1 = (nk_u16_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_u16_t const *a_row_2 = (nk_u16_t const *)((char const *)a_matrix + (row + 2) * a_stride_in_bytes);
        nk_u16_t const *a_row_3 = (nk_u16_t const *)((char const *)a_matrix + (row + 3) * a_stride_in_bytes);
        nk_f32_t *c_row_0 = (nk_f32_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_f32_t *c_row_1 = (nk_f32_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);
        nk_f32_t *c_row_2 = (nk_f32_t *)((char *)c_matrix + (row + 2) * c_stride_in_bytes);
        nk_f32_t *c_row_3 = (nk_f32_t *)((char *)c_matrix + (row + 3) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_0_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_1_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_2_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_3_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);
                // Load A as u16m1 and convert to f32m2
                vuint16m1_t a_raw_0_u16m1 = __riscv_vle16_v_u16m1(a_row_0 + k, vector_length);
                vfloat32m2_t a_vector_0_f32m2 = nk_f16m1_to_f32m2_rvv_(a_raw_0_u16m1, vector_length);
                vuint16m1_t a_raw_1_u16m1 = __riscv_vle16_v_u16m1(a_row_1 + k, vector_length);
                vfloat32m2_t a_vector_1_f32m2 = nk_f16m1_to_f32m2_rvv_(a_raw_1_u16m1, vector_length);
                vuint16m1_t a_raw_2_u16m1 = __riscv_vle16_v_u16m1(a_row_2 + k, vector_length);
                vfloat32m2_t a_vector_2_f32m2 = nk_f16m1_to_f32m2_rvv_(a_raw_2_u16m1, vector_length);
                vuint16m1_t a_raw_3_u16m1 = __riscv_vle16_v_u16m1(a_row_3 + k, vector_length);
                vfloat32m2_t a_vector_3_f32m2 = nk_f16m1_to_f32m2_rvv_(a_raw_3_u16m1, vector_length);
                accumulator_0_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_0_f64m4, a_vector_0_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_1_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_1_f64m4, a_vector_1_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_2_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_2_f64m4, a_vector_2_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_3_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_3_f64m4, a_vector_3_f32m2, b_vector_f32m2,
                                                                  vector_length);
            }

            // Horizontal reduce and narrow to f32
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row_0[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_0_f64m4, zero_f64m1, vlmax));
            c_row_1[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_1_f64m4, zero_f64m1, vlmax));
            c_row_2[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_2_f64m4, zero_f64m1, vlmax));
            c_row_3[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_3_f64m4, zero_f64m1, vlmax));
        }
    }
    // Remainder rows (mr < 4)
    for (; row < row_count; ++row) {
        nk_u16_t const *a_row = (nk_u16_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);
                vuint16m1_t a_raw_u16m1 = __riscv_vle16_v_u16m1(a_row + k, vector_length);
                vfloat32m2_t a_vector_f32m2 = nk_f16m1_to_f32m2_rvv_(a_raw_u16m1, vector_length);
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, a_vector_f32m2, b_vector_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
        }
    }
}

/**
 *  @brief  Public f16 packed GEMM wrapper matching the declared signature in dots.h.
 *
 *  Dispatches to the aligned kernel for all cases — RVV's `vsetvl` handles partial
 *  vectors naturally, so no separate edge kernel is needed.
 */
NK_PUBLIC void nk_dots_packed_f16_rvv(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                      nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_f16_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric f16 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses f64 widened accumulation via `vfwmacc_vv_f64m4` for precision.
 *  Both inputs are f16, loaded as u16 and converted to f32 via `nk_f16m1_to_f32m2_rvv_`.
 *  Stride is in bytes.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_f16_rvv(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_u16_t const *a_i = (nk_u16_t const *)((char const *)vectors + i * stride);
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_u16_t const *a_j = (nk_u16_t const *)((char const *)vectors + j * stride);
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vuint16m1_t a_raw_u16m1 = __riscv_vle16_v_u16m1(a_i + k, vector_length);
                vfloat32m2_t a_vector_f32m2 = nk_f16m1_to_f32m2_rvv_(a_raw_u16m1, vector_length);
                vuint16m1_t b_raw_u16m1 = __riscv_vle16_v_u16m1(a_j + k, vector_length);
                vfloat32m2_t b_vector_f32m2 = nk_f16m1_to_f32m2_rvv_(b_raw_u16m1, vector_length);
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, a_vector_f32m2, b_vector_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            nk_f32_t dot = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Half Precision Floats

#pragma region Signed 8-bit Integers

/**
 *  @brief  Compute the packed buffer size for i8 GEMM (B stored as i8).
 *
 *  VL is determined by `__riscv_vsetvlmax_e8m1()` since B is stored as i8.
 *  Layout: column-panel with depth-contiguous i8 values, cache-line padding.
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e8m1();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    // Break power-of-2 strides for cache associativity
    nk_size_t stride_bytes = depth_padded * sizeof(nk_i8_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_i8_t) +
           column_count * sizeof(nk_u32_t); // per-column norms
}

/**
 *  @brief  Pack B matrix from i8 to i8 (direct copy) for integer dot product.
 *
 *  No conversion needed — values are copied directly.
 *  Padding values are zeroed. Column-panel layout with depth-contiguous storage.
 */
NK_PUBLIC void nk_dots_pack_i8_rvv(nk_i8_t const *b, nk_size_t column_count, nk_size_t depth,
                                   nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e8m1();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_i8_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_i8_t *packed = (nk_i8_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_i8_t const *src = (nk_i8_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_i8_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) dst[k] = src[k];
    }

    // Append per-column norms after packed data
    nk_u32_t *norms = (nk_u32_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_i8_t const *src = (nk_i8_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_i8_(src, depth);
    }
}

/**
 *  @brief  i8 packed GEMM kernel: C += A * B_packed^T with i32 accumulation.
 *
 *  Vectorizes over the depth dimension (k). For each (row, column) pair:
 *    - Load i8 values from A and pre-packed i8 values from B
 *    - Widening multiply: i8 x i8 -> i16 via `vwmul`
 *    - Widen-accumulate: i32 += i16 via `vwadd_wv`
 *    - Horizontal reduce via `vredsum`
 *
 *  Register tile: process 4 rows per iteration (rows_per_tile=4).
 *  Output is nk_i32_t (integer result, no scaling).
 */
NK_INTERNAL void nk_dots_packed_i8_rvv_aligned_(nk_i8_t const *a_matrix, void const *b_packed_buffer,
                                                nk_i32_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                nk_size_t c_stride_in_bytes) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_i8_t const *packed_data = (nk_i8_t const *)((char const *)b_packed_buffer +
                                                   sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_i32_t *c_row = (nk_i32_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // mr=4 register tile over rows
    nk_size_t row = 0;
    for (; row + 4 <= row_count; row += 4) {
        nk_i8_t const *a_row_0 = (nk_i8_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_i8_t const *a_row_1 = (nk_i8_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_i8_t const *a_row_2 = (nk_i8_t const *)((char const *)a_matrix + (row + 2) * a_stride_in_bytes);
        nk_i8_t const *a_row_3 = (nk_i8_t const *)((char const *)a_matrix + (row + 3) * a_stride_in_bytes);
        nk_i32_t *c_row_0 = (nk_i32_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_i32_t *c_row_1 = (nk_i32_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);
        nk_i32_t *c_row_2 = (nk_i32_t *)((char *)c_matrix + (row + 2) * c_stride_in_bytes);
        nk_i32_t *c_row_3 = (nk_i32_t *)((char *)c_matrix + (row + 3) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_i8_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vint32m4_t accumulator_0_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            vint32m4_t accumulator_1_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            vint32m4_t accumulator_2_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            vint32m4_t accumulator_3_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e8m1(remaining);
                vint8m1_t b_vector_i8m1 = __riscv_vle8_v_i8m1(b_column + k, vector_length);
                vint8m1_t a_vector_0_i8m1 = __riscv_vle8_v_i8m1(a_row_0 + k, vector_length);
                vint8m1_t a_vector_1_i8m1 = __riscv_vle8_v_i8m1(a_row_1 + k, vector_length);
                vint8m1_t a_vector_2_i8m1 = __riscv_vle8_v_i8m1(a_row_2 + k, vector_length);
                vint8m1_t a_vector_3_i8m1 = __riscv_vle8_v_i8m1(a_row_3 + k, vector_length);
                vint16m2_t product_0_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_0_i8m1, b_vector_i8m1, vector_length);
                vint16m2_t product_1_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_1_i8m1, b_vector_i8m1, vector_length);
                vint16m2_t product_2_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_2_i8m1, b_vector_i8m1, vector_length);
                vint16m2_t product_3_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_3_i8m1, b_vector_i8m1, vector_length);
                accumulator_0_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_0_i32m4, accumulator_0_i32m4,
                                                                product_0_i16m2, vector_length);
                accumulator_1_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_1_i32m4, accumulator_1_i32m4,
                                                                product_1_i16m2, vector_length);
                accumulator_2_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_2_i32m4, accumulator_2_i32m4,
                                                                product_2_i16m2, vector_length);
                accumulator_3_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_3_i32m4, accumulator_3_i32m4,
                                                                product_3_i16m2, vector_length);
            }

            // Horizontal reduce
            vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            c_row_0[column] = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(
                __riscv_vredsum_vs_i32m4_i32m1(accumulator_0_i32m4, zero_i32m1, vlmax));
            c_row_1[column] = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(
                __riscv_vredsum_vs_i32m4_i32m1(accumulator_1_i32m4, zero_i32m1, vlmax));
            c_row_2[column] = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(
                __riscv_vredsum_vs_i32m4_i32m1(accumulator_2_i32m4, zero_i32m1, vlmax));
            c_row_3[column] = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(
                __riscv_vredsum_vs_i32m4_i32m1(accumulator_3_i32m4, zero_i32m1, vlmax));
        }
    }
    // Remainder rows (mr < 4)
    for (; row < row_count; ++row) {
        nk_i8_t const *a_row = (nk_i8_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_i32_t *c_row = (nk_i32_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_i8_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vint32m4_t accumulator_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e8m1(remaining);
                vint8m1_t b_vector_i8m1 = __riscv_vle8_v_i8m1(b_column + k, vector_length);
                vint8m1_t a_vector_i8m1 = __riscv_vle8_v_i8m1(a_row + k, vector_length);
                vint16m2_t product_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_i8m1, b_vector_i8m1, vector_length);
                accumulator_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_i32m4, accumulator_i32m4, product_i16m2,
                                                              vector_length);
            }
            vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            c_row[column] = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(
                __riscv_vredsum_vs_i32m4_i32m1(accumulator_i32m4, zero_i32m1, vlmax));
        }
    }
}

/**
 *  @brief  Public i8 packed GEMM wrapper matching the declared signature in dots.h.
 *
 *  Dispatches to the aligned kernel for all cases — RVV's `vsetvl` handles partial
 *  vectors naturally, so no separate edge kernel is needed.
 */
NK_PUBLIC void nk_dots_packed_i8_rvv(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                     nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_i8_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric i8 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses integer i8 arithmetic with i32 accumulation.
 *  Both inputs are i8, widened via i8 x i8 -> i16 -> i32 accumulation.
 *  Stride is in bytes.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_i8_rvv(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_i8_t const *a_i = (nk_i8_t const *)((char const *)vectors + i * stride);
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_i8_t const *a_j = (nk_i8_t const *)((char const *)vectors + j * stride);
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vint32m4_t accumulator_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e8m1(remaining);
                vint8m1_t a_vector_i8m1 = __riscv_vle8_v_i8m1(a_i + k, vector_length);
                vint8m1_t b_vector_i8m1 = __riscv_vle8_v_i8m1(a_j + k, vector_length);
                vint16m2_t product_i16m2 = __riscv_vwmul_vv_i16m2(a_vector_i8m1, b_vector_i8m1, vector_length);
                accumulator_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_i32m4, accumulator_i32m4, product_i16m2,
                                                              vector_length);
            }
            vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            nk_i32_t dot = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(
                __riscv_vredsum_vs_i32m4_i32m1(accumulator_i32m4, zero_i32m1, vlmax));
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Signed 8-bit Integers

#pragma region Unsigned 8-bit Integers

/**
 *  @brief  Compute the packed buffer size for u8 GEMM (B stored as u8).
 *
 *  VL is determined by `__riscv_vsetvlmax_e8m1()` since B is stored as u8.
 *  Layout: column-panel with depth-contiguous u8 values, cache-line padding.
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e8m1();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    // Break power-of-2 strides for cache associativity
    nk_size_t stride_bytes = depth_padded * sizeof(nk_u8_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_u8_t) +
           column_count * sizeof(nk_u32_t); // per-column norms
}

/**
 *  @brief  Pack B matrix from u8 to u8 (direct copy) for integer dot product.
 *
 *  No conversion needed — values are copied directly.
 *  Padding values are zeroed. Column-panel layout with depth-contiguous storage.
 */
NK_PUBLIC void nk_dots_pack_u8_rvv(nk_u8_t const *b, nk_size_t column_count, nk_size_t depth,
                                   nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e8m1();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_u8_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_u8_t *packed = (nk_u8_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_u8_t const *src = (nk_u8_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_u8_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) dst[k] = src[k];
    }

    // Append per-column norms after packed data
    nk_u32_t *norms = (nk_u32_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_u8_t const *src = (nk_u8_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_u8_(src, depth);
    }
}

/**
 *  @brief  u8 packed GEMM kernel: C += A * B_packed^T with u32 accumulation.
 *
 *  Vectorizes over the depth dimension (k). For each (row, column) pair:
 *    - Load u8 values from A and pre-packed u8 values from B
 *    - Widening multiply: u8 x u8 -> u16 via `vwmulu`
 *    - Widen-accumulate: u32 += u16 via `vwaddu_wv`
 *    - Horizontal reduce via `vredsum`
 *
 *  Register tile: process 4 rows per iteration (rows_per_tile=4).
 *  Output is nk_u32_t (unsigned integer result, no scaling).
 */
NK_INTERNAL void nk_dots_packed_u8_rvv_aligned_(nk_u8_t const *a_matrix, void const *b_packed_buffer,
                                                nk_u32_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                nk_size_t c_stride_in_bytes) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_u8_t const *packed_data = (nk_u8_t const *)((char const *)b_packed_buffer +
                                                   sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_u32_t *c_row = (nk_u32_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // mr=4 register tile over rows
    nk_size_t row = 0;
    for (; row + 4 <= row_count; row += 4) {
        nk_u8_t const *a_row_0 = (nk_u8_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_u8_t const *a_row_1 = (nk_u8_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_u8_t const *a_row_2 = (nk_u8_t const *)((char const *)a_matrix + (row + 2) * a_stride_in_bytes);
        nk_u8_t const *a_row_3 = (nk_u8_t const *)((char const *)a_matrix + (row + 3) * a_stride_in_bytes);
        nk_u32_t *c_row_0 = (nk_u32_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_u32_t *c_row_1 = (nk_u32_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);
        nk_u32_t *c_row_2 = (nk_u32_t *)((char *)c_matrix + (row + 2) * c_stride_in_bytes);
        nk_u32_t *c_row_3 = (nk_u32_t *)((char *)c_matrix + (row + 3) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_u8_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vuint32m4_t accumulator_0_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
            vuint32m4_t accumulator_1_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
            vuint32m4_t accumulator_2_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
            vuint32m4_t accumulator_3_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e8m1(remaining);
                vuint8m1_t b_vector_u8m1 = __riscv_vle8_v_u8m1(b_column + k, vector_length);
                vuint8m1_t a_vector_0_u8m1 = __riscv_vle8_v_u8m1(a_row_0 + k, vector_length);
                vuint8m1_t a_vector_1_u8m1 = __riscv_vle8_v_u8m1(a_row_1 + k, vector_length);
                vuint8m1_t a_vector_2_u8m1 = __riscv_vle8_v_u8m1(a_row_2 + k, vector_length);
                vuint8m1_t a_vector_3_u8m1 = __riscv_vle8_v_u8m1(a_row_3 + k, vector_length);
                vuint16m2_t product_0_u16m2 = __riscv_vwmulu_vv_u16m2(a_vector_0_u8m1, b_vector_u8m1, vector_length);
                vuint16m2_t product_1_u16m2 = __riscv_vwmulu_vv_u16m2(a_vector_1_u8m1, b_vector_u8m1, vector_length);
                vuint16m2_t product_2_u16m2 = __riscv_vwmulu_vv_u16m2(a_vector_2_u8m1, b_vector_u8m1, vector_length);
                vuint16m2_t product_3_u16m2 = __riscv_vwmulu_vv_u16m2(a_vector_3_u8m1, b_vector_u8m1, vector_length);
                accumulator_0_u32m4 = __riscv_vwaddu_wv_u32m4_tu(accumulator_0_u32m4, accumulator_0_u32m4,
                                                                 product_0_u16m2, vector_length);
                accumulator_1_u32m4 = __riscv_vwaddu_wv_u32m4_tu(accumulator_1_u32m4, accumulator_1_u32m4,
                                                                 product_1_u16m2, vector_length);
                accumulator_2_u32m4 = __riscv_vwaddu_wv_u32m4_tu(accumulator_2_u32m4, accumulator_2_u32m4,
                                                                 product_2_u16m2, vector_length);
                accumulator_3_u32m4 = __riscv_vwaddu_wv_u32m4_tu(accumulator_3_u32m4, accumulator_3_u32m4,
                                                                 product_3_u16m2, vector_length);
            }

            // Horizontal reduce
            vuint32m1_t zero_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
            c_row_0[column] = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(
                __riscv_vredsum_vs_u32m4_u32m1(accumulator_0_u32m4, zero_u32m1, vlmax));
            c_row_1[column] = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(
                __riscv_vredsum_vs_u32m4_u32m1(accumulator_1_u32m4, zero_u32m1, vlmax));
            c_row_2[column] = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(
                __riscv_vredsum_vs_u32m4_u32m1(accumulator_2_u32m4, zero_u32m1, vlmax));
            c_row_3[column] = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(
                __riscv_vredsum_vs_u32m4_u32m1(accumulator_3_u32m4, zero_u32m1, vlmax));
        }
    }
    // Remainder rows (mr < 4)
    for (; row < row_count; ++row) {
        nk_u8_t const *a_row = (nk_u8_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_u32_t *c_row = (nk_u32_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_u8_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vuint32m4_t accumulator_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e8m1(remaining);
                vuint8m1_t b_vector_u8m1 = __riscv_vle8_v_u8m1(b_column + k, vector_length);
                vuint8m1_t a_vector_u8m1 = __riscv_vle8_v_u8m1(a_row + k, vector_length);
                vuint16m2_t product_u16m2 = __riscv_vwmulu_vv_u16m2(a_vector_u8m1, b_vector_u8m1, vector_length);
                accumulator_u32m4 = __riscv_vwaddu_wv_u32m4_tu(accumulator_u32m4, accumulator_u32m4, product_u16m2,
                                                               vector_length);
            }
            vuint32m1_t zero_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
            c_row[column] = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(
                __riscv_vredsum_vs_u32m4_u32m1(accumulator_u32m4, zero_u32m1, vlmax));
        }
    }
}

/**
 *  @brief  Public u8 packed GEMM wrapper matching the declared signature in dots.h.
 *
 *  Dispatches to the aligned kernel for all cases — RVV's `vsetvl` handles partial
 *  vectors naturally, so no separate edge kernel is needed.
 */
NK_PUBLIC void nk_dots_packed_u8_rvv(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t m, nk_size_t n,
                                     nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_u8_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric u8 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses unsigned integer u8 arithmetic with u32 accumulation.
 *  Both inputs are u8, widened via u8 x u8 -> u16 -> u32 accumulation.
 *  Stride is in bytes.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_u8_rvv(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_u32_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_u8_t const *a_i = (nk_u8_t const *)((char const *)vectors + i * stride);
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_u8_t const *a_j = (nk_u8_t const *)((char const *)vectors + j * stride);
            nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
            vuint32m4_t accumulator_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e8m1(remaining);
                vuint8m1_t a_vector_u8m1 = __riscv_vle8_v_u8m1(a_i + k, vector_length);
                vuint8m1_t b_vector_u8m1 = __riscv_vle8_v_u8m1(a_j + k, vector_length);
                vuint16m2_t product_u16m2 = __riscv_vwmulu_vv_u16m2(a_vector_u8m1, b_vector_u8m1, vector_length);
                accumulator_u32m4 = __riscv_vwaddu_wv_u32m4_tu(accumulator_u32m4, accumulator_u32m4, product_u16m2,
                                                               vector_length);
            }
            vuint32m1_t zero_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
            nk_u32_t dot = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(
                __riscv_vredsum_vs_u32m4_u32m1(accumulator_u32m4, zero_u32m1, vlmax));
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Unsigned 8-bit Integers

#pragma region Quarter Precision E4M3

/**
 *  @brief  E4M3 magnitude LUT: 7-bit magnitude -> f32 bit pattern (u32).
 *          nk_e4m3_magnitude_lut_rvv_[i] = float_to_bits(e4m3_to_f32(i)) for i=0..127.
 *          E4M3FN: 4 exponent bits (bias=7), 3 mantissa bits, no infinity,
 *          NaN = magnitude 0x7F only.
 */
static nk_u32_t const nk_e4m3_magnitude_lut_rvv_[128] = {
    0x00000000u, 0x3B000000u, 0x3B800000u, 0x3BC00000u,
    0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u, /* [  0..  7] */
    0x3C800000u, 0x3C900000u, 0x3CA00000u, 0x3CB00000u,
    0x3CC00000u, 0x3CD00000u, 0x3CE00000u, 0x3CF00000u, /* [  8.. 15] */
    0x3D000000u, 0x3D100000u, 0x3D200000u, 0x3D300000u,
    0x3D400000u, 0x3D500000u, 0x3D600000u, 0x3D700000u, /* [ 16.. 23] */
    0x3D800000u, 0x3D900000u, 0x3DA00000u, 0x3DB00000u,
    0x3DC00000u, 0x3DD00000u, 0x3DE00000u, 0x3DF00000u, /* [ 24.. 31] */
    0x3E000000u, 0x3E100000u, 0x3E200000u, 0x3E300000u,
    0x3E400000u, 0x3E500000u, 0x3E600000u, 0x3E700000u, /* [ 32.. 39] */
    0x3E800000u, 0x3E900000u, 0x3EA00000u, 0x3EB00000u,
    0x3EC00000u, 0x3ED00000u, 0x3EE00000u, 0x3EF00000u, /* [ 40.. 47] */
    0x3F000000u, 0x3F100000u, 0x3F200000u, 0x3F300000u,
    0x3F400000u, 0x3F500000u, 0x3F600000u, 0x3F700000u, /* [ 48.. 55] */
    0x3F800000u, 0x3F900000u, 0x3FA00000u, 0x3FB00000u,
    0x3FC00000u, 0x3FD00000u, 0x3FE00000u, 0x3FF00000u, /* [ 56.. 63] */
    0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u,
    0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u, /* [ 64.. 71] */
    0x40800000u, 0x40900000u, 0x40A00000u, 0x40B00000u,
    0x40C00000u, 0x40D00000u, 0x40E00000u, 0x40F00000u, /* [ 72.. 79] */
    0x41000000u, 0x41100000u, 0x41200000u, 0x41300000u,
    0x41400000u, 0x41500000u, 0x41600000u, 0x41700000u, /* [ 80.. 87] */
    0x41800000u, 0x41900000u, 0x41A00000u, 0x41B00000u,
    0x41C00000u, 0x41D00000u, 0x41E00000u, 0x41F00000u, /* [ 88.. 95] */
    0x42000000u, 0x42100000u, 0x42200000u, 0x42300000u,
    0x42400000u, 0x42500000u, 0x42600000u, 0x42700000u, /* [ 96..103] */
    0x42800000u, 0x42900000u, 0x42A00000u, 0x42B00000u,
    0x42C00000u, 0x42D00000u, 0x42E00000u, 0x42F00000u, /* [104..111] */
    0x43000000u, 0x43100000u, 0x43200000u, 0x43300000u,
    0x43400000u, 0x43500000u, 0x43600000u, 0x43700000u, /* [112..119] */
    0x43800000u, 0x43900000u, 0x43A00000u, 0x43B00000u,
    0x43C00000u, 0x43D00000u, 0x43E00000u, 0x7FC00000u /* [120..127] */
};

NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_f32_t) +
           column_count * sizeof(nk_f32_t); // per-column norms
}

/**
 *  @brief  Pack B matrix from e4m3 to f32 for floating-point dot product.
 *
 *  Each e4m3 byte is converted to f32 via `nk_e4m3_to_f32_serial`.
 *  Padding values are zeroed. Column-panel layout with depth-contiguous storage.
 */
NK_PUBLIC void nk_dots_pack_e4m3_rvv(nk_e4m3_t const *b, nk_size_t column_count, nk_size_t depth,
                                     nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_f32_t *packed = (nk_f32_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_e4m3_t const *src = (nk_e4m3_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_f32_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) nk_e4m3_to_f32_serial(&src[k], &dst[k]);
    }

    // Append per-column norms after packed data
    nk_f32_t *norms = (nk_f32_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_e4m3_t const *src = (nk_e4m3_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_e4m3_(src, depth);
    }
}

/**
 *  @brief  e4m3 packed GEMM kernel: C += A * B_packed^T with f64 widened accumulation.
 *
 *  Vectorizes over the depth dimension (k). For each (row, column) pair:
 *    - Load pre-packed f32 values from B
 *    - Load raw e4m3 bytes from A, convert on-the-fly via 128-entry f32 LUT gather:
 *      extract 7-bit magnitude, zero-extend to u32, compute byte offsets (x4),
 *      gather f32 bit patterns, inject sign bit from bit 7 (<<24), reinterpret as f32
 *    - Widening FMA: f32xf32 -> f64 via `vfwmacc_vv_f64m4`
 *
 *  Register tile: process 2 rows per iteration (rows_per_tile=2, u32m2 gather + f64m4 accumulator is register-heavy).
 */
NK_INTERNAL void nk_dots_packed_e4m3_rvv_aligned_(nk_e4m3_t const *a_matrix, void const *b_packed_buffer,
                                                  nk_f32_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                  nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                  nk_size_t c_stride_in_bytes) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_f32_t const *packed_data = (nk_f32_t const *)((char const *)b_packed_buffer +
                                                     sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // mr=2 register tile over rows
    nk_size_t row = 0;
    for (; row + 2 <= row_count; row += 2) {
        nk_u8_t const *a_row_0 = (nk_u8_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_u8_t const *a_row_1 = (nk_u8_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_f32_t *c_row_0 = (nk_f32_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_f32_t *c_row_1 = (nk_f32_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_0_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_1_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);

                // Load pre-packed f32 B values
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);

                // Load raw e4m3 bytes from each A row
                vuint8mf2_t raw0_u8mf2 = __riscv_vle8_v_u8mf2(a_row_0 + k, vector_length);
                vuint8mf2_t raw1_u8mf2 = __riscv_vle8_v_u8mf2(a_row_1 + k, vector_length);

                // Extract 7-bit magnitudes, zero-extend to u32, compute byte offsets for f32 LUT
                vuint8mf2_t mag0_u8mf2 = __riscv_vand_vx_u8mf2(raw0_u8mf2, 0x7F, vector_length);
                vuint8mf2_t mag1_u8mf2 = __riscv_vand_vx_u8mf2(raw1_u8mf2, 0x7F, vector_length);
                vuint32m2_t idx0_u32m2 = __riscv_vzext_vf4_u32m2(mag0_u8mf2, vector_length);
                vuint32m2_t idx1_u32m2 = __riscv_vzext_vf4_u32m2(mag1_u8mf2, vector_length);
                vuint32m2_t off0_u32m2 = __riscv_vsll_vx_u32m2(idx0_u32m2, 2,
                                                               vector_length); // byte offsets = index * 4
                vuint32m2_t off1_u32m2 = __riscv_vsll_vx_u32m2(idx1_u32m2, 2, vector_length);

                // Gather f32 bit patterns from magnitude LUT
                vuint32m2_t bits0_u32m2 = __riscv_vluxei32_v_u32m2(nk_e4m3_magnitude_lut_rvv_, off0_u32m2,
                                                                   vector_length);
                vuint32m2_t bits1_u32m2 = __riscv_vluxei32_v_u32m2(nk_e4m3_magnitude_lut_rvv_, off1_u32m2,
                                                                   vector_length);

                // Extract sign bit 7, shift to f32 sign position (bit 31)
                vuint8mf2_t sign0_u8mf2 = __riscv_vand_vx_u8mf2(raw0_u8mf2, 0x80, vector_length);
                vuint8mf2_t sign1_u8mf2 = __riscv_vand_vx_u8mf2(raw1_u8mf2, 0x80, vector_length);
                vuint32m2_t sign0_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign0_u8mf2, vector_length), 24,
                                                                vector_length);
                vuint32m2_t sign1_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign1_u8mf2, vector_length), 24,
                                                                vector_length);

                // Apply sign and reinterpret as f32
                vfloat32m2_t a_vector_0_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits0_u32m2, sign0_u32m2, vector_length));
                vfloat32m2_t a_vector_1_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits1_u32m2, sign1_u32m2, vector_length));

                // Widening FMA: f32xf32 -> f64
                accumulator_0_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_0_f64m4, a_vector_0_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_1_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_1_f64m4, a_vector_1_f32m2, b_vector_f32m2,
                                                                  vector_length);
            }

            // Horizontal reduce and narrow to f32
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row_0[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_0_f64m4, zero_f64m1, vlmax));
            c_row_1[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_1_f64m4, zero_f64m1, vlmax));
        }
    }
    // Remainder rows
    for (; row < row_count; ++row) {
        nk_u8_t const *a_row = (nk_u8_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);
                vuint8mf2_t raw_a_u8mf2 = __riscv_vle8_v_u8mf2(a_row + k, vector_length);
                vuint8mf2_t mag_a_u8mf2 = __riscv_vand_vx_u8mf2(raw_a_u8mf2, 0x7F, vector_length);
                vuint32m2_t idx_a_u32m2 = __riscv_vzext_vf4_u32m2(mag_a_u8mf2, vector_length);
                vuint32m2_t off_a_u32m2 = __riscv_vsll_vx_u32m2(idx_a_u32m2, 2, vector_length);
                vuint32m2_t bits_a_u32m2 = __riscv_vluxei32_v_u32m2(nk_e4m3_magnitude_lut_rvv_, off_a_u32m2,
                                                                    vector_length);
                vuint8mf2_t sign_a_u8mf2 = __riscv_vand_vx_u8mf2(raw_a_u8mf2, 0x80, vector_length);
                vuint32m2_t sign_a_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign_a_u8mf2, vector_length),
                                                                 24, vector_length);
                vfloat32m2_t a_vector_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits_a_u32m2, sign_a_u32m2, vector_length));
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, a_vector_f32m2, b_vector_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
        }
    }
}

/**
 *  @brief  Public e4m3 packed GEMM wrapper matching the declared signature in dots.h.
 */
NK_PUBLIC void nk_dots_packed_e4m3_rvv(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_e4m3_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric e4m3 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses f32 LUT gather with f64 widened accumulation for precision.
 *  Both operands are converted from e4m3 on-the-fly via magnitude LUT.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_e4m3_rvv(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_u8_t const *a_i = (nk_u8_t const *)vectors + i * stride;
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_u8_t const *a_j = (nk_u8_t const *)vectors + j * stride;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vuint8mf2_t raw_i_u8mf2 = __riscv_vle8_v_u8mf2(a_i + k, vector_length);
                vuint8mf2_t raw_j_u8mf2 = __riscv_vle8_v_u8mf2(a_j + k, vector_length);

                // Convert i-vector via LUT gather
                vuint8mf2_t mag_i_u8mf2 = __riscv_vand_vx_u8mf2(raw_i_u8mf2, 0x7F, vector_length);
                vuint32m2_t idx_i_u32m2 = __riscv_vzext_vf4_u32m2(mag_i_u8mf2, vector_length);
                vuint32m2_t off_i_u32m2 = __riscv_vsll_vx_u32m2(idx_i_u32m2, 2, vector_length);
                vuint32m2_t bits_i_u32m2 = __riscv_vluxei32_v_u32m2(nk_e4m3_magnitude_lut_rvv_, off_i_u32m2,
                                                                    vector_length);
                vuint8mf2_t sign_i_u8mf2 = __riscv_vand_vx_u8mf2(raw_i_u8mf2, 0x80, vector_length);
                vuint32m2_t sign_i_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign_i_u8mf2, vector_length),
                                                                 24, vector_length);
                vfloat32m2_t val_i_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits_i_u32m2, sign_i_u32m2, vector_length));

                // Convert j-vector via LUT gather
                vuint8mf2_t mag_j_u8mf2 = __riscv_vand_vx_u8mf2(raw_j_u8mf2, 0x7F, vector_length);
                vuint32m2_t idx_j_u32m2 = __riscv_vzext_vf4_u32m2(mag_j_u8mf2, vector_length);
                vuint32m2_t off_j_u32m2 = __riscv_vsll_vx_u32m2(idx_j_u32m2, 2, vector_length);
                vuint32m2_t bits_j_u32m2 = __riscv_vluxei32_v_u32m2(nk_e4m3_magnitude_lut_rvv_, off_j_u32m2,
                                                                    vector_length);
                vuint8mf2_t sign_j_u8mf2 = __riscv_vand_vx_u8mf2(raw_j_u8mf2, 0x80, vector_length);
                vuint32m2_t sign_j_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign_j_u8mf2, vector_length),
                                                                 24, vector_length);
                vfloat32m2_t val_j_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits_j_u32m2, sign_j_u32m2, vector_length));

                // Widening FMA: f32xf32 -> f64
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, val_i_f32m2, val_j_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            nk_f32_t dot = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Quarter Precision E4M3

#pragma region Quarter Precision E5M2

/**
 *  @brief  E5M2 magnitude LUT: 7-bit magnitude -> f32 bit pattern (u32).
 *          nk_e5m2_magnitude_lut_rvv_[i] = float_to_bits(e5m2_to_f32(i)) for i=0..127.
 *          E5M2: 5 exponent bits (bias=15), 2 mantissa bits, has infinity (0x7C) and
 *          NaN (magnitudes 0x7D..0x7F).
 */
static nk_u32_t const nk_e5m2_magnitude_lut_rvv_[128] = {
    0x00000000u, 0x37800000u, 0x38000000u, 0x38400000u,
    0x38800000u, 0x38A00000u, 0x38C00000u, 0x38E00000u, /* [  0..  7] */
    0x39000000u, 0x39200000u, 0x39400000u, 0x39600000u,
    0x39800000u, 0x39A00000u, 0x39C00000u, 0x39E00000u, /* [  8.. 15] */
    0x3A000000u, 0x3A200000u, 0x3A400000u, 0x3A600000u,
    0x3A800000u, 0x3AA00000u, 0x3AC00000u, 0x3AE00000u, /* [ 16.. 23] */
    0x3B000000u, 0x3B200000u, 0x3B400000u, 0x3B600000u,
    0x3B800000u, 0x3BA00000u, 0x3BC00000u, 0x3BE00000u, /* [ 24.. 31] */
    0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u,
    0x3C800000u, 0x3CA00000u, 0x3CC00000u, 0x3CE00000u, /* [ 32.. 39] */
    0x3D000000u, 0x3D200000u, 0x3D400000u, 0x3D600000u,
    0x3D800000u, 0x3DA00000u, 0x3DC00000u, 0x3DE00000u, /* [ 40.. 47] */
    0x3E000000u, 0x3E200000u, 0x3E400000u, 0x3E600000u,
    0x3E800000u, 0x3EA00000u, 0x3EC00000u, 0x3EE00000u, /* [ 48.. 55] */
    0x3F000000u, 0x3F200000u, 0x3F400000u, 0x3F600000u,
    0x3F800000u, 0x3FA00000u, 0x3FC00000u, 0x3FE00000u, /* [ 56.. 63] */
    0x40000000u, 0x40200000u, 0x40400000u, 0x40600000u,
    0x40800000u, 0x40A00000u, 0x40C00000u, 0x40E00000u, /* [ 64.. 71] */
    0x41000000u, 0x41200000u, 0x41400000u, 0x41600000u,
    0x41800000u, 0x41A00000u, 0x41C00000u, 0x41E00000u, /* [ 72.. 79] */
    0x42000000u, 0x42200000u, 0x42400000u, 0x42600000u,
    0x42800000u, 0x42A00000u, 0x42C00000u, 0x42E00000u, /* [ 80.. 87] */
    0x43000000u, 0x43200000u, 0x43400000u, 0x43600000u,
    0x43800000u, 0x43A00000u, 0x43C00000u, 0x43E00000u, /* [ 88.. 95] */
    0x44000000u, 0x44200000u, 0x44400000u, 0x44600000u,
    0x44800000u, 0x44A00000u, 0x44C00000u, 0x44E00000u, /* [ 96..103] */
    0x45000000u, 0x45200000u, 0x45400000u, 0x45600000u,
    0x45800000u, 0x45A00000u, 0x45C00000u, 0x45E00000u, /* [104..111] */
    0x46000000u, 0x46200000u, 0x46400000u, 0x46600000u,
    0x46800000u, 0x46A00000u, 0x46C00000u, 0x46E00000u, /* [112..119] */
    0x47000000u, 0x47200000u, 0x47400000u, 0x47600000u,
    0x7F800000u, 0x7FC00000u, 0x7FC00000u, 0x7FC00000u /* [120..127] */
};

NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_rvv(nk_size_t column_count, nk_size_t depth) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;
    return sizeof(nk_cross_packed_buffer_header_t) + column_count * depth_padded * sizeof(nk_f32_t) +
           column_count * sizeof(nk_f32_t); // per-column norms
}

/**
 *  @brief  Pack B matrix from e5m2 to f32 for floating-point dot product.
 *
 *  Each e5m2 byte is converted to f32 via `nk_e5m2_to_f32_serial`.
 *  Padding values are zeroed. Column-panel layout with depth-contiguous storage.
 */
NK_PUBLIC void nk_dots_pack_e5m2_rvv(nk_e5m2_t const *b, nk_size_t column_count, nk_size_t depth,
                                     nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t vector_length = __riscv_vsetvlmax_e32m2();
    nk_size_t depth_padded = nk_size_round_up_to_multiple_(depth, vector_length);
    nk_size_t stride_bytes = depth_padded * sizeof(nk_f32_t);
    if (stride_bytes > 0 && (stride_bytes & (stride_bytes - 1)) == 0) depth_padded += vector_length;

    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded;

    nk_f32_t *packed = (nk_f32_t *)((char *)b_packed + sizeof(nk_cross_packed_buffer_header_t));
    nk_size_t total = column_count * depth_padded;
    for (nk_size_t i = 0; i < total; ++i) packed[i] = 0;

    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_e5m2_t const *src = (nk_e5m2_t const *)((char const *)b + column * b_stride_in_bytes);
        nk_f32_t *dst = packed + column * depth_padded;
        for (nk_size_t k = 0; k < depth; ++k) nk_e5m2_to_f32_serial(&src[k], &dst[k]);
    }

    // Append per-column norms after packed data
    nk_f32_t *norms = (nk_f32_t *)(packed + total);
    for (nk_size_t column = 0; column < column_count; ++column) {
        nk_e5m2_t const *src = (nk_e5m2_t const *)((char const *)b + column * b_stride_in_bytes);
        norms[column] = nk_dots_reduce_sumsq_e5m2_(src, depth);
    }
}

/**
 *  @brief  e5m2 packed GEMM kernel: C += A * B_packed^T with f64 widened accumulation.
 *
 *  Vectorizes over the depth dimension (k). For each (row, column) pair:
 *    - Load pre-packed f32 values from B
 *    - Load raw e5m2 bytes from A, convert on-the-fly via 128-entry f32 LUT gather:
 *      extract 7-bit magnitude, zero-extend to u32, compute byte offsets (x4),
 *      gather f32 bit patterns, inject sign bit from bit 7 (<<24), reinterpret as f32
 *    - Widening FMA: f32xf32 -> f64 via `vfwmacc_vv_f64m4`
 *
 *  Register tile: process 2 rows per iteration (rows_per_tile=2, u32m2 gather + f64m4 accumulator is register-heavy).
 */
NK_INTERNAL void nk_dots_packed_e5m2_rvv_aligned_(nk_e5m2_t const *a_matrix, void const *b_packed_buffer,
                                                  nk_f32_t *c_matrix, nk_size_t row_count, nk_size_t column_count,
                                                  nk_size_t depth, nk_size_t a_stride_in_bytes,
                                                  nk_size_t c_stride_in_bytes) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;
    nk_size_t const depth_padded = header->depth_padded_values;
    nk_f32_t const *packed_data = (nk_f32_t const *)((char const *)b_packed_buffer +
                                                     sizeof(nk_cross_packed_buffer_header_t));

    // Zero output matrix
    for (nk_size_t i = 0; i < row_count; ++i) {
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + i * c_stride_in_bytes);
        for (nk_size_t j = 0; j < column_count; ++j) c_row[j] = 0;
    }

    // mr=2 register tile over rows
    nk_size_t row = 0;
    for (; row + 2 <= row_count; row += 2) {
        nk_u8_t const *a_row_0 = (nk_u8_t const *)((char const *)a_matrix + (row + 0) * a_stride_in_bytes);
        nk_u8_t const *a_row_1 = (nk_u8_t const *)((char const *)a_matrix + (row + 1) * a_stride_in_bytes);
        nk_f32_t *c_row_0 = (nk_f32_t *)((char *)c_matrix + (row + 0) * c_stride_in_bytes);
        nk_f32_t *c_row_1 = (nk_f32_t *)((char *)c_matrix + (row + 1) * c_stride_in_bytes);

        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_0_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            vfloat64m4_t accumulator_1_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);

                // Load pre-packed f32 B values
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);

                // Load raw e5m2 bytes from each A row
                vuint8mf2_t raw0_u8mf2 = __riscv_vle8_v_u8mf2(a_row_0 + k, vector_length);
                vuint8mf2_t raw1_u8mf2 = __riscv_vle8_v_u8mf2(a_row_1 + k, vector_length);

                // Extract 7-bit magnitudes, zero-extend to u32, compute byte offsets for f32 LUT
                vuint8mf2_t mag0_u8mf2 = __riscv_vand_vx_u8mf2(raw0_u8mf2, 0x7F, vector_length);
                vuint8mf2_t mag1_u8mf2 = __riscv_vand_vx_u8mf2(raw1_u8mf2, 0x7F, vector_length);
                vuint32m2_t idx0_u32m2 = __riscv_vzext_vf4_u32m2(mag0_u8mf2, vector_length);
                vuint32m2_t idx1_u32m2 = __riscv_vzext_vf4_u32m2(mag1_u8mf2, vector_length);
                vuint32m2_t off0_u32m2 = __riscv_vsll_vx_u32m2(idx0_u32m2, 2,
                                                               vector_length); // byte offsets = index * 4
                vuint32m2_t off1_u32m2 = __riscv_vsll_vx_u32m2(idx1_u32m2, 2, vector_length);

                // Gather f32 bit patterns from magnitude LUT
                vuint32m2_t bits0_u32m2 = __riscv_vluxei32_v_u32m2(nk_e5m2_magnitude_lut_rvv_, off0_u32m2,
                                                                   vector_length);
                vuint32m2_t bits1_u32m2 = __riscv_vluxei32_v_u32m2(nk_e5m2_magnitude_lut_rvv_, off1_u32m2,
                                                                   vector_length);

                // Extract sign bit 7, shift to f32 sign position (bit 31)
                vuint8mf2_t sign0_u8mf2 = __riscv_vand_vx_u8mf2(raw0_u8mf2, 0x80, vector_length);
                vuint8mf2_t sign1_u8mf2 = __riscv_vand_vx_u8mf2(raw1_u8mf2, 0x80, vector_length);
                vuint32m2_t sign0_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign0_u8mf2, vector_length), 24,
                                                                vector_length);
                vuint32m2_t sign1_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign1_u8mf2, vector_length), 24,
                                                                vector_length);

                // Apply sign and reinterpret as f32
                vfloat32m2_t a_vector_0_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits0_u32m2, sign0_u32m2, vector_length));
                vfloat32m2_t a_vector_1_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits1_u32m2, sign1_u32m2, vector_length));

                // Widening FMA: f32xf32 -> f64
                accumulator_0_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_0_f64m4, a_vector_0_f32m2, b_vector_f32m2,
                                                                  vector_length);
                accumulator_1_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_1_f64m4, a_vector_1_f32m2, b_vector_f32m2,
                                                                  vector_length);
            }

            // Horizontal reduce and narrow to f32
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row_0[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_0_f64m4, zero_f64m1, vlmax));
            c_row_1[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_1_f64m4, zero_f64m1, vlmax));
        }
    }
    // Remainder rows
    for (; row < row_count; ++row) {
        nk_u8_t const *a_row = (nk_u8_t const *)((char const *)a_matrix + row * a_stride_in_bytes);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c_matrix + row * c_stride_in_bytes);
        for (nk_size_t column = 0; column < column_count; ++column) {
            nk_f32_t const *b_column = packed_data + column * depth_padded;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vfloat32m2_t b_vector_f32m2 = __riscv_vle32_v_f32m2(b_column + k, vector_length);
                vuint8mf2_t raw_a_u8mf2 = __riscv_vle8_v_u8mf2(a_row + k, vector_length);
                vuint8mf2_t mag_a_u8mf2 = __riscv_vand_vx_u8mf2(raw_a_u8mf2, 0x7F, vector_length);
                vuint32m2_t idx_a_u32m2 = __riscv_vzext_vf4_u32m2(mag_a_u8mf2, vector_length);
                vuint32m2_t off_a_u32m2 = __riscv_vsll_vx_u32m2(idx_a_u32m2, 2, vector_length);
                vuint32m2_t bits_a_u32m2 = __riscv_vluxei32_v_u32m2(nk_e5m2_magnitude_lut_rvv_, off_a_u32m2,
                                                                    vector_length);
                vuint8mf2_t sign_a_u8mf2 = __riscv_vand_vx_u8mf2(raw_a_u8mf2, 0x80, vector_length);
                vuint32m2_t sign_a_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign_a_u8mf2, vector_length),
                                                                 24, vector_length);
                vfloat32m2_t a_vector_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits_a_u32m2, sign_a_u32m2, vector_length));
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, a_vector_f32m2, b_vector_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            c_row[column] = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
        }
    }
}

/**
 *  @brief  Public e5m2 packed GEMM wrapper matching the declared signature in dots.h.
 */
NK_PUBLIC void nk_dots_packed_e5m2_rvv(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                       nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_packed_e5m2_rvv_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/**
 *  @brief  Symmetric e5m2 GEMM: C = A * A^T, upper triangle + mirror.
 *
 *  Uses f32 LUT gather with f64 widened accumulation for precision.
 *  Both operands are converted from e5m2 on-the-fly via magnitude LUT.
 *  Processes only the rows in [row_start, row_start + row_count) for parallelism.
 */
NK_PUBLIC void nk_dots_symmetric_e5m2_rvv(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_size_t const row_end = (row_start + row_count < n_vectors) ? (row_start + row_count) : n_vectors;

    for (nk_size_t i = row_start; i < row_end; ++i) {
        nk_u8_t const *a_i = (nk_u8_t const *)vectors + i * stride;
        for (nk_size_t j = i; j < n_vectors; ++j) {
            nk_u8_t const *a_j = (nk_u8_t const *)vectors + j * stride;
            nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
            vfloat64m4_t accumulator_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
            nk_size_t remaining = depth;
            nk_size_t k = 0;
            for (nk_size_t vector_length = 0; remaining > 0; remaining -= vector_length, k += vector_length) {
                vector_length = __riscv_vsetvl_e32m2(remaining);
                vuint8mf2_t raw_i_u8mf2 = __riscv_vle8_v_u8mf2(a_i + k, vector_length);
                vuint8mf2_t raw_j_u8mf2 = __riscv_vle8_v_u8mf2(a_j + k, vector_length);

                // Convert i-vector via LUT gather
                vuint8mf2_t mag_i_u8mf2 = __riscv_vand_vx_u8mf2(raw_i_u8mf2, 0x7F, vector_length);
                vuint32m2_t idx_i_u32m2 = __riscv_vzext_vf4_u32m2(mag_i_u8mf2, vector_length);
                vuint32m2_t off_i_u32m2 = __riscv_vsll_vx_u32m2(idx_i_u32m2, 2, vector_length);
                vuint32m2_t bits_i_u32m2 = __riscv_vluxei32_v_u32m2(nk_e5m2_magnitude_lut_rvv_, off_i_u32m2,
                                                                    vector_length);
                vuint8mf2_t sign_i_u8mf2 = __riscv_vand_vx_u8mf2(raw_i_u8mf2, 0x80, vector_length);
                vuint32m2_t sign_i_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign_i_u8mf2, vector_length),
                                                                 24, vector_length);
                vfloat32m2_t val_i_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits_i_u32m2, sign_i_u32m2, vector_length));

                // Convert j-vector via LUT gather
                vuint8mf2_t mag_j_u8mf2 = __riscv_vand_vx_u8mf2(raw_j_u8mf2, 0x7F, vector_length);
                vuint32m2_t idx_j_u32m2 = __riscv_vzext_vf4_u32m2(mag_j_u8mf2, vector_length);
                vuint32m2_t off_j_u32m2 = __riscv_vsll_vx_u32m2(idx_j_u32m2, 2, vector_length);
                vuint32m2_t bits_j_u32m2 = __riscv_vluxei32_v_u32m2(nk_e5m2_magnitude_lut_rvv_, off_j_u32m2,
                                                                    vector_length);
                vuint8mf2_t sign_j_u8mf2 = __riscv_vand_vx_u8mf2(raw_j_u8mf2, 0x80, vector_length);
                vuint32m2_t sign_j_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf4_u32m2(sign_j_u8mf2, vector_length),
                                                                 24, vector_length);
                vfloat32m2_t val_j_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
                    __riscv_vor_vv_u32m2(bits_j_u32m2, sign_j_u32m2, vector_length));

                // Widening FMA: f32xf32 -> f64
                accumulator_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(accumulator_f64m4, val_i_f32m2, val_j_f32m2,
                                                                vector_length);
            }
            vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
            nk_f32_t dot = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredusum_vs_f64m4_f64m1(accumulator_f64m4, zero_f64m1, vlmax));
            result[i * result_stride_elements + j] = dot;
        }
    }
}

#pragma endregion // Quarter Precision E5M2

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_DOTS_RVV_H
