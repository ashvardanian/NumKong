/**
 *  @brief SIMD-accelerated Batched Dot Products for SME.
 *  @file include/numkong/dots/sme.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses ARM Scalable Matrix Extension (SME) for efficient matrix multiplication
 *  with `ZA32` tiles supporting `f16`, `bf16`, `i8`, `u8`, and `e4m3` input types:
 *
 *  - `svmopa_za32_f16_m`: `f16` × `f16` outer product accumulate to `f32`
 *  - `svmopa_za32_bf16_m`: `bf16` × `bf16` outer product accumulate to `f32`
 *  - `svmopa_za32_s8_m`: `i8` × `i8` outer product accumulate to `i32`
 *  - `svmopa_za32_u8_m`: `u8` × `u8` outer product accumulate to `u32`
 *
 *  SME tile dimensions (for SVL=512, i.e., Apple M4):
 *
 *  - `ZA32` tile: 16 × 16 `f32`/`i32` elements (1KB)
 *  - `f16`/`bf16` vectors: 32 elements per SVE vector
 *  - `i8`/`u8` vectors: 64 elements per SVE vector
 *  - `f32`/`i32` vectors: 16 elements per SVE vector
 *
 *  Output pattern: Each `svmopa` accumulates a 16 × 16 tile from input vectors.
 *  We process multiple ZA tiles (0-3) to form larger output blocks.
 *
 *  Performance characteristics (Apple M4):
 *
 *  - `f16` → `f32` peak: ~2 TFLOPS per core
 *  - `bf16` → `f32` peak: ~2 TFLOPS per core
 *  - `i8` → `i32` peak: ~2 TOPS per core
 *  - Streaming mode has different register set from normal NEON
 *
 *  Acceleration opportunities:
 *
 *  - Pre-pack B matrix for column-major access: avoids transpose overhead
 *  - Tile along M/N dimensions: cache efficiency
 *  - Use multiple ZA tiles: 2×2 output blocking
 *
 *  @section dots_sme_instructions ARM SME Instructions
 *
 *      Intrinsic                       Instruction                         Latency     Throughput
 *      `svmopa_za32_f16_m`             `FMOPA` (ZA.S, P/M, Z.H, Z.H)       16cy        amortized
 *      `svmopa_za32_bf16_m`            `BFMOPA` (ZA.S, P/M, Z.H, Z.H)      16cy        amortized
 *      `svmopa_za32_s8_m`              `SMOPA` (ZA.S, P/M, Z.B, Z.B)       16cy        amortized
 *      `svmopa_za32_u8_m`              `UMOPA` (ZA.S, P/M, Z.B, Z.B)       16cy        amortized
 *      `svzero_za`                     `ZERO` (ZA)                         2cy         1/cy
 *      `svld1_hor_za32`                `LD1W` (ZA.S[Ws, #imm], P/Z)        4-6cy       1/cy
 *      `svst1_hor_za32`                `ST1W` (ZA.S[Ws, #imm], P)          4cy         1/cy
 *      `svwrite_hor_za32_f32_m`        `MOVA` (ZA.S[Ws, #imm], P/M, Z.S)   2cy         1/cy
 *      `svread_ver_za32_f32_m`         `MOVA` (Z.S, P/M, ZA.S[Ws, #imm])   2cy         1/cy
 *      `__arm_streaming`               `SMSTART`                           ~50-100cy
 *      `__arm_streaming` (exit)        `SMSTOP`                            ~50-100cy
 *      `__arm_new("za")`               ZA tile allocation                  0cy
 *      `svcntw`                        `CNTW` (Xd)                         1cy         2/cy
 *      `svcnth`                        `CNTH` (Xd)                         1cy         2/cy
 */
#ifndef NK_DOTS_SME_H
#define NK_DOTS_SME_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_e4m3_to_f16_serial`, `nk_e5m2_to_f16_serial`
#include "numkong/dots/serial.h" // `nk_dots_reduce_sumsq_f16_`, `nk_dots_reduce_sumsq_i8_`, etc.

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

/**
 *  SME-specific packed buffer header (64-byte aligned).
 *  Layout optimized for SME outer product access patterns with predicate-based edge handling.
 */
typedef struct {
    nk_u32_t column_tile_count; // ⌈columns/tile_dimension⌉: number of column tiles
    nk_u32_t depth_tile_count;  // ⌈depth/depth_tile_size⌉: number of depth tiles
    nk_u32_t columns;           // actual N dimension for predicates
    nk_u32_t depth;             // actual K dimension for predicates
    nk_u32_t svl_bytes;         // SVL in bytes at pack time: for validation
    nk_u32_t norms_offset;      // byte offset from buffer start to per-column norms
    nk_u32_t reserved[10];      // padding to 64 bytes
} nk_dots_sme_packed_header_t;

/*  Selective ZA tile zeroing masks for `svzero_mask_za(mask)`.
 *  These zero individual ZA.S tiles without destroying other accumulators.
 *  ZA.D tiles (8 tiles, 8x8 each): ZA0.D = mask bit 0, ..., ZA7.D = mask bit 7.
 */
enum {
    nk_sme_zero_za32_tile_0_ = 0x11,
    nk_sme_zero_za32_tile_1_ = 0x22,
    nk_sme_zero_za32_tile_2_ = 0x44,
    nk_sme_zero_za32_tile_3_ = 0x88,
    nk_sme_zero_za32_tiles_123_ = 0xEE, /* Accumulators only (preserves ZA0 staging) */
    nk_sme_zero_za64_tile_0_ = 0x01,
    nk_sme_zero_za64_tile_1_ = 0x02,
    nk_sme_zero_za64_tile_2_ = 0x04,    /* ZA2.D only */
    nk_sme_zero_za64_tiles_1_3_ = 0x0E, /* ZA1-3.D (Ozaki 1-col path) */
    nk_sme_zero_za64_tiles_1_6_ = 0x7E, /* ZA1-6.D (Ozaki 2-col path) */
    nk_sme_zero_za64_tiles_1_7_ = 0xFE, /* Accumulators ZA1-7.D (preserves ZA0.D staging) */
};

/*
 *  f16/bf16 → f32 GEMM using FMOPA/BFMOPA with ZA32 tiles.
 *
 *  Tile layout (SVL=512, Apple M4):
 *  - ZA32 output tile: 16 × 16 f32 elements (1 KB)
 *  - Input vectors: 32 f16/bf16 elements (SVL/16)
 *  - Depth per FMOPA: 2 f16 pairs → 1 f32 (widening 2:1)
 *  - FMOPA predicates: b16 (input granularity), not b32
 *  - 4-tile path: ZA0-ZA3 process 4 column tiles simultaneously
 */
#pragma region Half Precision Floats

NK_PUBLIC nk_size_t nk_dots_packed_size_f16_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const expansion = 2;               // FMOPA f16 → f32: 2 f16 pairs per f32 output
    nk_size_t const tile_dimension = svcntsw();  // ZA32 tile dim: 16
    nk_size_t const vector_elements = svcntsh(); // f16 elements per SVE vector: 32
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_step_count * vector_elements * sizeof(nk_f16_t);
    size += n * sizeof(nk_f32_t); // per-column squared norms
    return size;
}

NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_f16_sme(n, k); }

NK_PUBLIC void nk_dots_pack_f16_sme(             //
    nk_f16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;               // FMOPA f16 → f32: 2 f16 pairs per f32 output
    nk_size_t const tile_dimension = svcntsw();  // ZA32 tile dim: 16
    nk_size_t const vector_elements = svcntsh(); // f16 per SVE vector: 32 = tile_dimension * expansion
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f16_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all packed data (partial vectors stay zero-padded)
    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) { tiles_ptr[i] = 0; }

    // Pack in interleaved format for FMOPA:
    // For each column tile and depth step, produce one SVE vector where:
    //   packed_vec[expansion * d + k] = B[column_tile*tile_dim + d, depth_step*expansion + k]
    // d = 0..tile_dim-1 (B row within column tile), k = 0..expansion-1 (depth sub-element)
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        vec_output[expansion * column_in_tile + sub_element] = b[src_idx];
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_elements * sizeof(nk_f16_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_f32_t *norms_ptr = (nk_f32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_f16_t const *col_data = (nk_f16_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_f16_(col_data, k);
    }
}

NK_PUBLIC void nk_dots_pack_bf16_sme(             //
    nk_bf16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsh();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) { tiles_ptr[i] = 0; }

    // Interleaved packing: packed_vec[expansion * d + k] = B[column_tile*tile_dim + d, depth_step*expansion + k]
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_bf16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        vec_output[expansion * column_in_tile + sub_element] = b[src_idx];
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_elements * sizeof(nk_bf16_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_f32_t *norms_ptr = (nk_f32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_bf16_t const *col_data = (nk_bf16_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_bf16_(col_data, k);
    }
}

/**
 *  `f16` → `f32` GEMM core kernel using SME outer products.
 *
 *  FMOPA f16 → f32 semantics: ZA[s][d] += Σ(k=0..1) Zn[2s+k] * Zm[2d+k]
 *  So Zn is interpreted as a 16×2 sub-matrix and Zm as a 2×16 sub-matrix.
 *
 *  For correct GEMM C[i][j] = Σ_k A[i][k]*B[j][k]:
 *  - Zn[2*s+k] = A[row_start+s, depth_base+k]  (gather 2 depth elements per A row)
 *  - Zm[2*d+k] = B[column_start+d, depth_base+k]   (pre-packed interleaved)
 *  - Loop over depth in steps of 2 (expansion factor)
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_f16_sme_streaming_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,                              //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();              // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcnth();             // 32: f16 elements per SVE vector
    nk_size_t const depth_steps_per_batch = tile_dimension; // 16 depth steps per ZA0 load

    nk_f16_t const *b_packed_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    // ZA tile assignment:
    //   ZA0.S = A data staging (horizontal load, vertical read)
    //   ZA1.S, ZA2.S, ZA3.S = MOPA accumulation (3 column tiles simultaneously)

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles at a time using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_); // zero accumulators only

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svzero_mask_za(nk_sme_zero_za32_tile_0_); // clear staging ZA0 only

                // Load A rows into ZA0.S horizontally (f32 words = 2 packed f16 each)
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr);
                }

                // For each depth step in batch: vertical read = transpose = perfect interleaving
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t a_interleaved_vector_f16x = svreinterpret_f16_f32(a_column_f32x);

                    // Load B vectors for 3 column tiles
                    nk_f16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    svfloat16_t b_packed_vector_0_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_0);
                    svfloat16_t b_packed_vector_1_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_1);
                    svfloat16_t b_packed_vector_2_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_2);

                    // 3 MOPA calls into ZA1, ZA2, ZA3
                    svmopa_za32_f16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_0_f16x);
                    svmopa_za32_f16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_1_f16x);
                    svmopa_za32_f16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_2_f16x);
                }
            }

            // Extract from ZA1, ZA2, ZA3 (accumulated across ALL batches)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile at a time using ZA1 only
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_); // zero ZA1 accumulator only

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_); // clear staging ZA0

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t a_interleaved_vector_f16x = svreinterpret_f16_f32(a_column_f32x);

                    nk_f16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_elements;
                    svfloat16_t b_packed_vector_f16x = svld1_f16(column_predicate_f16x,
                                                                 (float16_t const *)b_packed_ptr);

                    svmopa_za32_f16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_f32x, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

/**
 *  `bf16` → `f32` GEMM core kernel using SME outer products.
 *  Same interleaved algorithm as f16 kernel, using BFMOPA bf16 → f32.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_bf16_sme_streaming_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,                              //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                 //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_bf16_t const *b_packed_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svbfloat16_t a_interleaved_vector_bf16x = svreinterpret_bf16_f32(a_column_f32x);

                    nk_bf16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                       depth_batch_start + depth_step) *
                                                                          vector_elements;
                    nk_bf16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                       depth_batch_start + depth_step) *
                                                                          vector_elements;
                    nk_bf16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                       depth_batch_start + depth_step) *
                                                                          vector_elements;
                    svbfloat16_t b_packed_vector_0_bf16x = svld1_bf16(predicate_all_f16x,
                                                                      (bfloat16_t const *)b_packed_ptr_0);
                    svbfloat16_t b_packed_vector_1_bf16x = svld1_bf16(predicate_all_f16x,
                                                                      (bfloat16_t const *)b_packed_ptr_1);
                    svbfloat16_t b_packed_vector_2_bf16x = svld1_bf16(predicate_all_f16x,
                                                                      (bfloat16_t const *)b_packed_ptr_2);

                    svmopa_za32_bf16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_bf16x,
                                       b_packed_vector_0_bf16x);
                    svmopa_za32_bf16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_bf16x,
                                       b_packed_vector_1_bf16x);
                    svmopa_za32_bf16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_bf16x,
                                       b_packed_vector_2_bf16x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svbfloat16_t a_interleaved_vector_bf16x = svreinterpret_bf16_f32(a_column_f32x);

                    nk_bf16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    svbfloat16_t b_packed_vector_bf16x = svld1_bf16(column_predicate_f16x,
                                                                    (bfloat16_t const *)b_packed_ptr);

                    svmopa_za32_bf16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_vector_bf16x,
                                       b_packed_vector_bf16x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_f32x, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_f16_sme(                    //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_f16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_PUBLIC void nk_dots_packed_bf16_sme(                    //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_bf16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *   `f16` × `f16` → `f32` symmetric kernel using MOPA self-GEMM.
 *   Time-shares ZA0 for both A and B transposition: loads A horizontally,
 *   pre-reads A columns into Z registers, then reloads ZA0 with B data
 *   per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_f16_sme_streaming_(
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw(); // 16
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension; // 16

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const depth_remainder = depth % expansion;
    nk_u32_t const element_mask_bits = depth_remainder ? (1u << (depth_remainder * (32u / expansion))) - 1u
                                                       : 0xFFFFFFFFu;
    svuint32_t const element_mask_u32x = svdup_u32(element_mask_bits);
    svuint32_t const lane_indices_u32x = svindex_u32(0, 1);

    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_clamped * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_clamped);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles at a time
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // Per-batch cleanup: predicate true only at the last f32 word when depth has remainder
                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();
                // ZA transpose for A rows (identical to packed kernel)
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    if (clean_last_word) {
                        svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)a_row_ptr);
                        row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                        svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_f32x, row_u32x);
                    }
                    else { svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr); }
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                // Load B column tile 0 into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }

                // Load B column tile 1 into ZA0, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }

                // Load B column tile 2 into ZA0, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }
            }

            // Extract results from ZA1-3
            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        // Remainder: 1 column tile at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    if (clean_last_word) {
                        svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)a_row_ptr);
                        row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                        svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_f32x, row_u32x);
                    }
                    else { svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr); }
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                // Load B column tile into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_f32x,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
}

__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_bf16_sme_streaming_(
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const depth_remainder = depth % expansion;
    nk_u32_t const element_mask_bits = depth_remainder ? (1u << (depth_remainder * (32u / expansion))) - 1u
                                                       : 0xFFFFFFFFu;
    svuint32_t const element_mask_u32x = svdup_u32(element_mask_bits);
    svuint32_t const lane_indices_u32x = svindex_u32(0, 1);

    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_clamped * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    if (clean_last_word) {
                        svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)a_row_ptr);
                        row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                        svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_f32x, row_u32x);
                    }
                    else { svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr); }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svbfloat16_t a_interleaved_bf16x = svreinterpret_bf16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svbfloat16_t b_interleaved_bf16x = svreinterpret_bf16_f32(b_column_f32x);
                    svmopa_za32_bf16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_bf16x,
                                       b_interleaved_bf16x);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svbfloat16_t a_interleaved_bf16x = svreinterpret_bf16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svbfloat16_t b_interleaved_bf16x = svreinterpret_bf16_f32(b_column_f32x);
                    svmopa_za32_bf16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_bf16x,
                                       b_interleaved_bf16x);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svbfloat16_t a_interleaved_bf16x = svreinterpret_bf16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svbfloat16_t b_interleaved_bf16x = svreinterpret_bf16_f32(b_column_f32x);
                    svmopa_za32_bf16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_bf16x,
                                       b_interleaved_bf16x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    if (clean_last_word) {
                        svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)a_row_ptr);
                        row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                        svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_f32x, row_u32x);
                    }
                    else { svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr); }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svbfloat16_t a_interleaved_bf16x = svreinterpret_bf16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_f32x, 0,
                                                                      depth_step);
                    svbfloat16_t b_interleaved_bf16x = svreinterpret_bf16_f32(b_column_f32x);
                    svmopa_za32_bf16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_bf16x,
                                       b_interleaved_bf16x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_f32x,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
}

#pragma endregion

/*
 *  `i8` × `i8` → `i32` GEMM using SME outer products.
 *
 *  Uses `svmopa_za32_s8_m` for signed 8-bit integer outer product accumulate.
 *  Available on Apple M4 (SME_I8I32 = 1).
 *
 *  Tile dimensions for `i8` → `i32` (512-bit SVL):
 *  - Input vectors: 64 `i8` elements (SVL/8 = 64)
 *  - Output tile: 16 × 16 `i32` elements (`ZA32`)
 *  - Each output `i32` is a dot product of 4 `i8` pairs
 *
 *  Expected performance: ~2 TOPS (4× `f16` due to 4:1 element packing)
 */

#pragma region Signed 8-bit Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const expansion = 4;               // SMOPA i8→i32: 4 i8 pairs per i32 output
    nk_size_t const tile_dimension = svcntsw();  // ZA32 tile dim: 16
    nk_size_t const vector_elements = svcntsb(); // i8 elements per SVE vector: 64
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_step_count * vector_elements * sizeof(nk_i8_t);
    size += n * sizeof(nk_u32_t); // per-column squared norms
    return size;
}

NK_PUBLIC void nk_dots_pack_i8_sme(             //
    nk_i8_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb(); // 64 = tile_dimension * expansion
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_i8_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_i32_t));

    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) { tiles_ptr[i] = 0; }

    // Interleaved packing: packed_vec[expansion * d + k] = B[column_tile*tile_dim + d, depth_step*expansion + k]
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_i8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        vec_output[expansion * column_in_tile + sub_element] = b[src_idx];
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_elements * sizeof(nk_i8_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_u32_t *norms_ptr = (nk_u32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_i8_t const *col_data = (nk_i8_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_i8_(col_data, k);
    }
}

/**
 *  `i8` × `i8` → `i32` GEMM core kernel using SME outer products.
 *
 *  SMOPA i8→i32 semantics: ZA[s][d] += Σ(k=0..3) Zn[4s+k] * Zm[4d+k]
 *  So Zn is interpreted as a 16×4 sub-matrix and Zm as a 4×16 sub-matrix.
 *
 *  For correct GEMM C[i][j] = Σ_k A[i][k]*B[j][k]:
 *  - Zn[4*s+k] = A[row_start+s, depth_base+k]  (gather 4 depth elements per A row)
 *  - Zm[4*d+k] = B[column_start+d, depth_base+k]   (pre-packed interleaved)
 *  - Loop over depth in steps of 4 (expansion factor)
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_i8_sme_streaming_( //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c,                              //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                               //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();              // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcntb();             // 64: i8 elements per SVE vector
    nk_size_t const depth_steps_per_batch = tile_dimension; // 16 steps = 64 depth elements per ZA0 load

    nk_i8_t const *b_packed_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                // Load A rows into ZA0.S (each f32 word = 4 packed i8 bytes)
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    // Vertical read at f32 granularity produces [row0_k0..k3, row1_k0..k3, ...]
                    svint32_t a_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, depth_step);
                    svint8_t a_interleaved_vector_i8x = svreinterpret_s8_s32(a_column_i32x);

                    nk_i8_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_i8_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_i8_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    svint8_t b_packed_vector_0_i8x = svld1_s8(predicate_all_i8x, b_packed_ptr_0);
                    svint8_t b_packed_vector_1_i8x = svld1_s8(predicate_all_i8x, b_packed_ptr_1);
                    svint8_t b_packed_vector_2_i8x = svld1_s8(predicate_all_i8x, b_packed_ptr_2);

                    svmopa_za32_s8_m(1, row_predicate_i8x, predicate_all_i8x, a_interleaved_vector_i8x,
                                     b_packed_vector_0_i8x);
                    svmopa_za32_s8_m(2, row_predicate_i8x, predicate_all_i8x, a_interleaved_vector_i8x,
                                     b_packed_vector_1_i8x);
                    svmopa_za32_s8_m(3, row_predicate_i8x, predicate_all_i8x, a_interleaved_vector_i8x,
                                     b_packed_vector_2_i8x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svint32_t a_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, depth_step);
                    svint8_t a_interleaved_vector_i8x = svreinterpret_s8_s32(a_column_i32x);

                    nk_i8_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                   depth_batch_start + depth_step) *
                                                                      vector_elements;
                    svint8_t b_packed_vector_i8x = svld1_s8(column_predicate_i8x, b_packed_ptr);

                    svmopa_za32_s8_m(1, row_predicate_i8x, column_predicate_i8x, a_interleaved_vector_i8x,
                                     b_packed_vector_i8x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_f32x, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_i8_sme(                    //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);

    nk_dots_packed_i8_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

// expansion=4: each i32 word packs 4 i8 values
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_i8_sme_streaming_(
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_i32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const depth_remainder = depth % expansion;
    nk_u32_t const element_mask_bits = depth_remainder ? (1u << (depth_remainder * (32u / expansion))) - 1u
                                                       : 0xFFFFFFFFu;
    svuint32_t const element_mask_u32x = svdup_u32(element_mask_bits);
    svuint32_t const lane_indices_u32x = svindex_u32(0, 1);

    NK_ALIGN64 nk_i32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_clamped * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    if (clean_last_word) {
                        svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)a_row_ptr);
                        row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                        svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_f32x, row_u32x);
                    }
                    else { svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr); }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved_i8x = svreinterpret_s8_s32(
                        svld1_s32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_f32x, 0, depth_step);
                    svint8_t b_interleaved_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svmopa_za32_s8_m(1, row_predicate_i8x, predicate_all_i8x, a_interleaved_i8x, b_interleaved_i8x);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved_i8x = svreinterpret_s8_s32(
                        svld1_s32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_f32x, 0, depth_step);
                    svint8_t b_interleaved_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svmopa_za32_s8_m(2, row_predicate_i8x, predicate_all_i8x, a_interleaved_i8x, b_interleaved_i8x);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved_i8x = svreinterpret_s8_s32(
                        svld1_s32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_f32x, 0, depth_step);
                    svint8_t b_interleaved_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svmopa_za32_s8_m(3, row_predicate_i8x, predicate_all_i8x, a_interleaved_i8x, b_interleaved_i8x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = (nk_f32_t *)(result + row_abs * result_stride_elements);
                svst1_hor_za32(1, row, predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    if (clean_last_word) {
                        svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)a_row_ptr);
                        row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                        svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_f32x, row_u32x);
                    }
                    else { svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr); }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved_i8x = svreinterpret_s8_s32(
                        svld1_s32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), column_predicate_f32x, 0, depth_step);
                    svint8_t b_interleaved_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svmopa_za32_s8_m(1, row_predicate_i8x, column_predicate_i8x, a_interleaved_i8x, b_interleaved_i8x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_f32x,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);
    nk_dots_symmetric_i8_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                        row_start, row_count);
}

#pragma endregion

/*
 *  e4m3 × e4m3 → f32 GEMM using inline SSVE conversion + FMOPA.
 *
 *  Pipeline: e4m3 bytes → svunpklo → arithmetic → f16 → FMOPA → f32
 *  - Load: 64 bytes via svld1_u8, convert lower 32 to f16 inline
 *  - Accumulate: FMOPA f16 → f32 into ZA32 tiles
 *  - No memory round-trip for format conversion
 *  - FMOPA predicates: b16 (f16 input granularity)
 */
#pragma region Quarter Precision E4M3

/**
 *  E4M3 subnormal → F16 lookup table (8 entries).
 *  Index by 3-bit mantissa (0..7) to get the F16 bit pattern for mant/512.
 *  For indices >= 8 (normal values), SVE TBL returns 0 which is ignored by SEL.
 */
static nk_u16_t const nk_e4m3_subnorm_f16_lut_[8] = {
    0x0000, // mant=0: 0.0
    0x1800, // mant=1: 1/512
    0x1C00, // mant=2: 2/512
    0x1E00, // mant=3: 3/512
    0x2000, // mant=4: 4/512
    0x2100, // mant=5: 5/512
    0x2200, // mant=6: 6/512
    0x2300, // mant=7: 7/512
};

/**
 *  Inline `e4m3` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *  This avoids memory round-trip when used inside a streaming kernel.
 *
 *  Uses an 8-entry TBL lookup for subnormals instead of UCVTF+FMUL, saving 3 instructions
 *  (18 → 15). The `subnorm_lut_u16x` parameter should be loaded once per kernel entry via:
 *    svuint16_t subnorm_lut_u16x = svld1_u16(svwhilelt_b16(0u, 8u), nk_e4m3_subnorm_f16_lut_);
 *
 *  @param predicate_f16x Predicate for 16-bit elements: use `svptrue_b16()`
 *  @param bytes_u8x Pre-loaded 64 bytes: `svuint8_t` from `svld1_u8`
 *  @param subnorm_lut_u16x Pre-loaded 8-entry subnormal LUT as `svuint16_t`
 *  @return 32 `f16` values as `svfloat16_t`: from lower 32 bytes
 */
NK_PUBLIC svfloat16_t nk_e4m3x_to_f16x_ssve_(svbool_t predicate_f16x, svuint8_t bytes_u8x,
                                             svuint16_t subnorm_lut_u16x) NK_STREAMING_COMPATIBLE_ {
    svuint16_t vals_u16x = svunpklo_u16(bytes_u8x); // 1: UUNPKLO

    svuint16_t sign_u16x = svlsl_n_u16_x(predicate_f16x, svand_n_u16_x(predicate_f16x, vals_u16x, 0x80),
                                         8);                              // 2-3: AND+LSL
    svuint16_t mag_u16x = svand_n_u16_x(predicate_f16x, vals_u16x, 0x7F); // 4: AND

    // Normal path: sign | ((mag << 7) + 0x2000)
    svuint16_t normal_u16x = svadd_n_u16_x(predicate_f16x, svlsl_n_u16_x(predicate_f16x, mag_u16x, 7),
                                           0x2000);                    // 5-6: LSL+ADD
    normal_u16x = svorr_u16_x(predicate_f16x, normal_u16x, sign_u16x); // 7: ORR

    // Subnormal path: 8-entry TBL (mag 0..7 for subnorms; mag>7 → TBL returns 0, ignored by SEL)
    svuint16_t subnorm_u16x = svorr_u16_x(predicate_f16x, svtbl_u16(subnorm_lut_u16x, mag_u16x),
                                          sign_u16x); // 8-9: TBL+ORR

    svbool_t is_subnorm_u16x = svcmpeq_n_u16(predicate_f16x, svand_n_u16_x(predicate_f16x, vals_u16x, 0x78),
                                             0); // 10-11: AND+CMPEQ
    // NaN path: mag == 0x7F → f16 NaN
    svbool_t is_nan_u16x = svcmpeq_n_u16(predicate_f16x, mag_u16x, 0x7F);           // 12: CMPEQ
    svuint16_t nan_val_u16x = svorr_n_u16_x(predicate_f16x, sign_u16x, 0x7E00);     // 13: ORR
    svuint16_t result_u16x = svsel_u16(is_subnorm_u16x, subnorm_u16x, normal_u16x); // 14: SEL
    result_u16x = svsel_u16(is_nan_u16x, nan_val_u16x, result_u16x);                // 15: SEL

    return svreinterpret_f16_u16(result_u16x);
}

/**
 *  Inline `e5m2` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *  This avoids memory round-trip when used inside a streaming kernel.
 *
 *  E5M2 format: S EEEEE MM (1+5+2 bits, bias=15, range [-57344, 57344])
 *  F16 format:  S EEEEE MMMMMMMMMM (1+5+10 bits, bias=15)
 *
 *  Since E5M2 and F16 share the same exponent bias (15), normal values convert
 *  by simply shifting the magnitude left by 8 bits.
 *
 *  @param predicate_f16x Predicate for 16-bit elements (use svptrue_b16())
 *  @param bytes_u8x Pre-loaded 64 bytes (svuint8_t from svld1_u8)
 *  @return 32 F16 values as svfloat16_t (from lower 32 bytes)
 */
NK_PUBLIC svfloat16_t nk_e5m2x_to_f16x_ssve_(svbool_t predicate_f16x, svuint8_t bytes_u8x) NK_STREAMING_COMPATIBLE_ {
    // E5M2 and F16 share the same exponent bias (15), sign position, exponent width,
    // and mantissa field alignment. The conversion f16 = byte << 8 is exact for ALL
    // 256 values including subnormals, infinity, and NaN.
    return svreinterpret_f16_u16(svlsl_n_u16_x(predicate_f16x, svunpklo_u16(bytes_u8x), 8));
}

/**
 *  Fused `e4m3` × `e4m3` → `f32` GEMM kernel using interleaved FMOPA.
 *  Converts `e4m3` → `f16` on-the-fly for A, B is pre-converted during packing.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_packed_e4m3_sme_streaming_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,                                                        //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                                           //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_f16_t const *b_packed_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    // Load e4m3 subnormal LUT once for all conversions in this kernel
    svuint16_t const subnorm_lut_u16x = svld1_u16(svwhilelt_b16(0u, 8u), nk_e4m3_subnorm_f16_lut_);

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                // Convert e4m3 → f16 for each A row in this batch, then load into ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    // Load raw e4m3 bytes and convert to f16 using vectorized conversion
                    nk_e4m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e4m3x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x,
                                                                        subnorm_lut_u16x);
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, svreinterpret_f32_f16(converted_f16x));
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t a_interleaved_vector_f16x = svreinterpret_f16_f32(a_column_f32x);

                    nk_f16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    svfloat16_t b_packed_vector_0_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_0);
                    svfloat16_t b_packed_vector_1_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_1);
                    svfloat16_t b_packed_vector_2_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_2);

                    svmopa_za32_f16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_0_f16x);
                    svmopa_za32_f16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_1_f16x);
                    svmopa_za32_f16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_2_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_e4m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e4m3x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x,
                                                                        subnorm_lut_u16x);
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, svreinterpret_f32_f16(converted_f16x));
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t a_interleaved_vector_f16x = svreinterpret_f16_f32(a_column_f32x);

                    nk_f16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_elements;
                    svfloat16_t b_packed_vector_f16x = svld1_f16(column_predicate_f16x,
                                                                 (float16_t const *)b_packed_ptr);

                    svmopa_za32_f16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_f32x, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sme(nk_size_t n, nk_size_t k) {
    // Uses `f16` format for packed data
    return nk_dots_packed_size_f16_sme(n, k);
}

NK_PUBLIC void nk_dots_pack_e4m3_sme(             //
    nk_e4m3_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsh();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e4m3_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) tiles_ptr[i] = 0;

    // Interleaved packing with e4m3 → f16 conversion
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        nk_e4m3_to_f16_serial(&b[src_idx], &vec_output[expansion * column_in_tile + sub_element]);
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_elements * sizeof(nk_f16_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_f32_t *norms_ptr = (nk_f32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_e4m3_t const *col_data = (nk_e4m3_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_e4m3_(col_data, k);
    }
}

NK_PUBLIC void nk_dots_packed_e4m3_sme(                    //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_e4m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 * `e4m3` × `e4m3` → `f32` symmetric kernel using MOPA self-GEMM.
 *  Time-shares ZA0 for both A and B transposition with e4m3 → f16 conversion.
 *  Pre-reads A columns into Z registers, then reloads ZA0 with converted B data
 *  per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_e4m3_sme_streaming_(
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    // Load e4m3 subnormal LUT once for all conversions in this kernel
    svuint16_t const subnorm_lut_u16x = svld1_u16(svwhilelt_b16(0u, 8u), nk_e4m3_subnorm_f16_lut_);

    nk_size_t const depth_remainder = depth % expansion;
    nk_u32_t const element_mask_bits = depth_remainder ? (1u << (depth_remainder * (32u / expansion))) - 1u
                                                       : 0xFFFFFFFFu;
    svuint32_t const element_mask_u32x = svdup_u32(element_mask_bits);
    svuint32_t const lane_indices_u32x = svindex_u32(0, 1);

    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_actual * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_actual);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                // ZA transpose for A rows: convert e4m3 → f16, MOVA directly into ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e4m3_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e4m3x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x,
                                                                        subnorm_lut_u16x);
                    svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                    if (clean_last_word)
                        write_f32x = svreinterpret_f32_u32(
                            svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, write_f32x);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                // Load B column tile 0 into ZA0 via MOVA, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e4m3x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x,
                                                                            subnorm_lut_u16x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }

                // Load B column tile 1 into ZA0 via MOVA, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e4m3x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x,
                                                                            subnorm_lut_u16x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }

                // Load B column tile 2 into ZA0 via MOVA, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e4m3x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x,
                                                                            subnorm_lut_u16x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e4m3_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e4m3x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x,
                                                                        subnorm_lut_u16x);
                    svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                    if (clean_last_word)
                        write_f32x = svreinterpret_f32_u32(
                            svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, write_f32x);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                // Load B column tile into ZA0 via MOVA, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e4m3x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x,
                                                                            subnorm_lut_u16x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_f32x,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
}

#pragma endregion

/*
 *  e5m2 × e5m2 → f32 GEMM using inline SSVE conversion + FMOPA.
 *
 *  Pipeline: e5m2 bytes → svunpklo → arithmetic → f16 → FMOPA → f32
 *  - Same tile layout as e4m3 (both convert to f16 before FMOPA)
 *  - E5M2 shares F16 exponent bias (15), so normal conversion is a shift
 *  - Handles infinity (mag=0x7C) and NaN (mag>0x7C)
 */
#pragma region Quarter Precision E5M2

/**
 *  Fused `e5m2` × `e5m2` → `f32` GEMM kernel using interleaved FMOPA.
 *  Converts `e5m2` → `f16` on-the-fly for A, B is pre-converted during packing.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_packed_e5m2_sme_streaming_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,                                                        //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                                           //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_f16_t const *b_packed_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    // Vectorized e5m2 → f16 conversion (2 instructions)
                    nk_e5m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e5m2x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x);
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, svreinterpret_f32_f16(converted_f16x));
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t a_interleaved_vector_f16x = svreinterpret_f16_f32(a_column_f32x);

                    nk_f16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    svfloat16_t b_packed_vector_0_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_0);
                    svfloat16_t b_packed_vector_1_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_1);
                    svfloat16_t b_packed_vector_2_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_2);

                    svmopa_za32_f16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_0_f16x);
                    svmopa_za32_f16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_1_f16x);
                    svmopa_za32_f16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_2_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_e5m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e5m2x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x);
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, svreinterpret_f32_f16(converted_f16x));
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t a_interleaved_vector_f16x = svreinterpret_f16_f32(a_column_f32x);

                    nk_f16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_elements;
                    svfloat16_t b_packed_vector_f16x = svld1_f16(column_predicate_f16x,
                                                                 (float16_t const *)b_packed_ptr);

                    svmopa_za32_f16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_f32x, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_f16_sme(n, k); }

NK_PUBLIC void nk_dots_pack_e5m2_sme(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsh();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e5m2_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) tiles_ptr[i] = 0;

    // Interleaved packing with e5m2 → f16 conversion
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        nk_e5m2_to_f16_serial(&b[src_idx], &vec_output[expansion * column_in_tile + sub_element]);
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_elements * sizeof(nk_f16_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_f32_t *norms_ptr = (nk_f32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_e5m2_t const *col_data = (nk_e5m2_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_e5m2_(col_data, k);
    }
}

/*  `e5m2` × `e5m2` → `f32` GEMM: public interface.
 *  Predicate-based edge handling eliminates scalar fallbacks.
 */
NK_PUBLIC void nk_dots_packed_e5m2_sme(                    //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_e5m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 * `e5m2` × `e5m2` → `f32` symmetric kernel using MOPA self-GEMM.
 *  Time-shares ZA0 for both A and B transposition with e5m2 → f16 conversion.
 *  Pre-reads A columns into Z registers, then reloads ZA0 with converted B data
 *  per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_e5m2_sme_streaming_(
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const depth_remainder = depth % expansion;
    nk_u32_t const element_mask_bits = depth_remainder ? (1u << (depth_remainder * (32u / expansion))) - 1u
                                                       : 0xFFFFFFFFu;
    svuint32_t const element_mask_u32x = svdup_u32(element_mask_bits);
    svuint32_t const lane_indices_u32x = svindex_u32(0, 1);

    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_actual * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_actual);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                // ZA transpose for A rows: convert e5m2 → f16, MOVA directly into ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e5m2_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e5m2x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x);
                    svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                    if (clean_last_word)
                        write_f32x = svreinterpret_f32_u32(
                            svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, write_f32x);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                // Load B column tile 0 into ZA0 via MOVA, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e5m2x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }

                // Load B column tile 1 into ZA0 via MOVA, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e5m2x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }

                // Load B column tile 2 into ZA0 via MOVA, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e5m2x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e5m2_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e5m2x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x);
                    svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                    if (clean_last_word)
                        write_f32x = svreinterpret_f32_u32(
                            svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, write_f32x);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                // Load B column tile into ZA0 via MOVA, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e5m2x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_f32x,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
}

#pragma endregion

/*
 *  `e2m3` × `e2m3` → `f32` GEMM using `i8` SME outer products.
 *
 *  E2M3 format: S MMMMM (1+2+3 = 6 bits, stored in a byte with upper 2 bits unused).
 *  Values are converted to signed `i8` via a 32-entry magnitude LUT, then processed
 *  with `svmopa_za32_s8_m`. The `i32` accumulator is converted to `f32` and scaled
 *  by 1/256 to recover the true floating-point dot product.
 *
 *  This gives 2x throughput over the `f16` path since `i8` SMOPA processes 4 elements
 *  per `i32` word vs 2 `f16` elements per `f32` word.
 *
 *  Tile dimensions (SVL=512, Apple M4):
 *  - Input vectors: 64 `i8` elements (after conversion)
 *  - Output tile: 16 × 16 `i32` → `f32` elements (`ZA32`)
 *  - Each output word accumulates 4 `i8` pairs, scaled by 1/256
 */

#pragma region Micro Precision E2M3

/**
 *  Inline `e2m3` → signed `i8` conversion returning `svint8_t` for direct use in GEMM.
 *  Uses a 32-entry magnitude LUT via SVE TBL instruction.
 *
 *  E2M3 encoding: bit 5 = sign, bits 4:0 = magnitude index (0..27 used).
 *  LUT maps magnitude index → unsigned integer value, then sign is applied.
 *
 *  @param predicate_i8x Predicate for 8-bit elements
 *  @param raw_bytes_u8x Pre-loaded e2m3 bytes as `svuint8_t`
 *  @return              Signed `i8` values as `svint8_t`
 */
NK_PUBLIC svint8_t nk_e2m3x_to_i8x_ssve_(svbool_t predicate_i8x, svuint8_t raw_bytes_u8x) NK_STREAMING_COMPATIBLE_ {
    // 32-entry magnitude LUT, replicated for SVE TBL (handles SVL > 256 bits)
    static NK_ALIGN64 nk_u8_t const lut_data[64] = {
        0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,  28,  30,  //
        32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, //
        0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,  28,  30,  //
        32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, //
    };
    svuint8_t magnitude_lut_u8x = svld1_u8(svptrue_b8(), lut_data);
    svuint8_t magnitude_u8x = svand_n_u8_x(predicate_i8x, raw_bytes_u8x, 0x1F);
    svuint8_t unsigned_value_u8x = svtbl_u8(magnitude_lut_u8x, magnitude_u8x);
    svuint8_t sign_bits_u8x = svand_n_u8_x(predicate_i8x, raw_bytes_u8x, 0x20);
    svbool_t negate_mask_b8x = svcmpne_n_u8(predicate_i8x, sign_bits_u8x, 0);
    svint8_t positive_value_i8x = svreinterpret_s8_u8(unsigned_value_u8x);
    svint8_t negated_value_i8x = svneg_s8_x(predicate_i8x, positive_value_i8x);
    return svsel_s8(negate_mask_b8x, negated_value_i8x, positive_value_i8x);
}

/**
 *  Fused `e2m3` × `e2m3` → `f32` GEMM kernel using interleaved SMOPA.
 *  Converts `e2m3` → `i8` on-the-fly for A, B is pre-converted during packing.
 *  Accumulates in `i32` via `svmopa_za32_s8_m`, then converts to `f32` with 1/256 scaling.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_packed_e2m3_sme_streaming_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,                                                        //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                                           //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;              // SMOPA i8→i32: 4 i8 pairs per i32 output
    nk_size_t const tile_dimension = svcntw();  // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcntb(); // 64: i8 elements per SVE vector
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_i8_t const *b_packed_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);

                // Convert e2m3 → i8 for each A row in this batch, then load into ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    // Load raw e2m3 bytes and convert to i8 using vectorized conversion
                    nk_e2m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svint8_t converted_i8x = nk_e2m3x_to_i8x_ssve_(depth_predicate_i8x, raw_bytes_u8x);
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, svreinterpret_f32_s8(converted_i8x));
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    // Vertical read at f32 granularity produces [row0_k0..k3, row1_k0..k3, ...]
                    svint32_t a_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, depth_step);
                    svint8_t a_interleaved_vector_i8x = svreinterpret_s8_s32(a_column_i32x);

                    nk_i8_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_i8_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_i8_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    svint8_t b_packed_vector_0_i8x = svld1_s8(predicate_all_i8x, b_packed_ptr_0);
                    svint8_t b_packed_vector_1_i8x = svld1_s8(predicate_all_i8x, b_packed_ptr_1);
                    svint8_t b_packed_vector_2_i8x = svld1_s8(predicate_all_i8x, b_packed_ptr_2);

                    svmopa_za32_s8_m(1, row_predicate_i8x, predicate_all_i8x, a_interleaved_vector_i8x,
                                     b_packed_vector_0_i8x);
                    svmopa_za32_s8_m(2, row_predicate_i8x, predicate_all_i8x, a_interleaved_vector_i8x,
                                     b_packed_vector_1_i8x);
                    svmopa_za32_s8_m(3, row_predicate_i8x, predicate_all_i8x, a_interleaved_vector_i8x,
                                     b_packed_vector_2_i8x);
                }
            }

            // Store results: convert i32 → f32 with 1/256 scaling
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const row_offset = (row_start + row) * c_stride_elements;
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;

                svint32_t row_1_i32x = svread_hor_za32_s32_m(svdup_s32(0), predicate_all_f32x, 1, row);
                svint32_t row_2_i32x = svread_hor_za32_s32_m(svdup_s32(0), predicate_all_f32x, 2, row);
                svint32_t row_3_i32x = svread_hor_za32_s32_m(svdup_s32(0), predicate_all_f32x, 3, row);
                svfloat32_t row_1_f32x = svmul_n_f32_x(predicate_all_f32x,
                                                       svcvt_f32_s32_x(predicate_all_f32x, row_1_i32x), 1.0f / 256.0f);
                svfloat32_t row_2_f32x = svmul_n_f32_x(predicate_all_f32x,
                                                       svcvt_f32_s32_x(predicate_all_f32x, row_2_i32x), 1.0f / 256.0f);
                svfloat32_t row_3_f32x = svmul_n_f32_x(predicate_all_f32x,
                                                       svcvt_f32_s32_x(predicate_all_f32x, row_3_i32x), 1.0f / 256.0f);
                svst1_f32(predicate_all_f32x, c + row_offset + cs0, row_1_f32x);
                svst1_f32(predicate_all_f32x, c + row_offset + cs1, row_2_f32x);
                svst1_f32(predicate_all_f32x, c + row_offset + cs2, row_3_f32x);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_e2m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svint8_t converted_i8x = nk_e2m3x_to_i8x_ssve_(depth_predicate_i8x, raw_bytes_u8x);
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, svreinterpret_f32_s8(converted_i8x));
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svint32_t a_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, depth_step);
                    svint8_t a_interleaved_vector_i8x = svreinterpret_s8_s32(a_column_i32x);

                    nk_i8_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                   depth_batch_start + depth_step) *
                                                                      vector_elements;
                    svint8_t b_packed_vector_i8x = svld1_s8(column_predicate_i8x, b_packed_ptr);

                    svmopa_za32_s8_m(1, row_predicate_i8x, column_predicate_i8x, a_interleaved_vector_i8x,
                                     b_packed_vector_i8x);
                }
            }

            // Store results: convert i32 → f32 with 1/256 scaling
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svint32_t row_i32x = svread_hor_za32_s32_m(svdup_s32(0), column_predicate_f32x, 1, row);
                svfloat32_t row_f32x = svmul_n_f32_x(column_predicate_f32x,
                                                     svcvt_f32_s32_x(column_predicate_f32x, row_i32x), 1.0f / 256.0f);
                svst1_f32(column_predicate_f32x, c + (row_start + row) * c_stride_elements + column_start, row_f32x);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_sme(nk_size_t n, nk_size_t k) {
    // Uses `i8` format for packed data (same tile geometry as i8)
    return nk_dots_packed_size_i8_sme(n, k);
}

NK_PUBLIC void nk_dots_pack_e2m3_sme(             //
    nk_e2m3_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb(); // 64 = tile_dimension * expansion
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e2m3_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_i32_t));

    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) { tiles_ptr[i] = 0; }

    // Scalar e2m3 → signed i8 magnitude LUT
    static nk_u8_t const lut_magnitude[32] = {
        0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,  28,  30,  //
        32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, //
    };

    // Interleaved packing with e2m3 → i8 conversion
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_i8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        nk_u8_t raw = b[src_idx];
                        nk_i8_t val = (nk_i8_t)lut_magnitude[raw & 0x1F];
                        if (raw & 0x20) val = -val;
                        vec_output[expansion * column_in_tile + sub_element] = val;
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_elements * sizeof(nk_i8_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_f32_t *norms_ptr = (nk_f32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_e2m3_t const *col_data = (nk_e2m3_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_e2m3_(col_data, k);
    }
}

NK_PUBLIC void nk_dots_packed_e2m3_sme(                    //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_e2m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *  `e2m3` × `e2m3` → `f32` symmetric kernel using SMOPA self-GEMM.
 *  Time-shares ZA0 for both A and B transposition with e2m3 → i8 conversion.
 *  Pre-reads A columns into Z registers, then reloads ZA0 with converted B data
 *  per column tile. Accumulates in i32, converts to f32 with 1/256 scaling.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_e2m3_sme_streaming_(
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const depth_remainder = depth % expansion;
    nk_u32_t const element_mask_bits = depth_remainder ? (1u << (depth_remainder * (32u / expansion))) - 1u
                                                       : 0xFFFFFFFFu;
    svuint32_t const element_mask_u32x = svdup_u32(element_mask_bits);
    svuint32_t const lane_indices_u32x = svindex_u32(0, 1);

    NK_ALIGN64 nk_i32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_actual * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_actual);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();

                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);

                // ZA transpose for A rows: convert e2m3 → i8 then load
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e2m3_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svint8_t converted_i8x = nk_e2m3x_to_i8x_ssve_(depth_predicate_i8x, raw_bytes_u8x);
                    svfloat32_t write_f32x = svreinterpret_f32_s8(converted_i8x);
                    if (clean_last_word)
                        write_f32x = svreinterpret_f32_u32(
                            svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, write_f32x);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, s));

                // Load B column tile 0 into ZA0, vertical read + SMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e2m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svint8_t converted_i8x = nk_e2m3x_to_i8x_ssve_(depth_predicate_i8x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_s8(converted_i8x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved_i8x = svreinterpret_s8_s32(
                        svld1_s32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_f32x, 0, depth_step);
                    svint8_t b_interleaved_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svmopa_za32_s8_m(1, row_predicate_i8x, predicate_all_i8x, a_interleaved_i8x, b_interleaved_i8x);
                }

                // Load B column tile 1 into ZA0, vertical read + SMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e2m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svint8_t converted_i8x = nk_e2m3x_to_i8x_ssve_(depth_predicate_i8x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_s8(converted_i8x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved_i8x = svreinterpret_s8_s32(
                        svld1_s32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_f32x, 0, depth_step);
                    svint8_t b_interleaved_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svmopa_za32_s8_m(2, row_predicate_i8x, predicate_all_i8x, a_interleaved_i8x, b_interleaved_i8x);
                }

                // Load B column tile 2 into ZA0, vertical read + SMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e2m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svint8_t converted_i8x = nk_e2m3x_to_i8x_ssve_(depth_predicate_i8x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_s8(converted_i8x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved_i8x = svreinterpret_s8_s32(
                        svld1_s32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_f32x, 0, depth_step);
                    svint8_t b_interleaved_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svmopa_za32_s8_m(3, row_predicate_i8x, predicate_all_i8x, a_interleaved_i8x, b_interleaved_i8x);
                }
            }

            // Store results: convert i32 → f32 with 1/256 scaling
            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;

                svint32_t row_1_i32x = svread_hor_za32_s32_m(svdup_s32(0), predicate_all_f32x, 1, row);
                svint32_t row_2_i32x = svread_hor_za32_s32_m(svdup_s32(0), predicate_all_f32x, 2, row);
                svint32_t row_3_i32x = svread_hor_za32_s32_m(svdup_s32(0), predicate_all_f32x, 3, row);
                svfloat32_t row_1_f32x = svmul_n_f32_x(predicate_all_f32x,
                                                       svcvt_f32_s32_x(predicate_all_f32x, row_1_i32x), 1.0f / 256.0f);
                svfloat32_t row_2_f32x = svmul_n_f32_x(predicate_all_f32x,
                                                       svcvt_f32_s32_x(predicate_all_f32x, row_2_i32x), 1.0f / 256.0f);
                svfloat32_t row_3_f32x = svmul_n_f32_x(predicate_all_f32x,
                                                       svcvt_f32_s32_x(predicate_all_f32x, row_3_i32x), 1.0f / 256.0f);
                svst1_f32(predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension, row_1_f32x);
                svst1_f32(predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension, row_2_f32x);
                svst1_f32(predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension, row_3_f32x);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();

                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e2m3_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svint8_t converted_i8x = nk_e2m3x_to_i8x_ssve_(depth_predicate_i8x, raw_bytes_u8x);
                    svfloat32_t write_f32x = svreinterpret_f32_s8(converted_i8x);
                    if (clean_last_word)
                        write_f32x = svreinterpret_f32_u32(
                            svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, write_f32x);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, s));

                // Load B column tile into ZA0, vertical read + SMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_e2m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svint8_t converted_i8x = nk_e2m3x_to_i8x_ssve_(depth_predicate_i8x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_s8(converted_i8x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved_i8x = svreinterpret_s8_s32(
                        svld1_s32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), column_predicate_f32x, 0, depth_step);
                    svint8_t b_interleaved_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svmopa_za32_s8_m(1, row_predicate_i8x, column_predicate_i8x, a_interleaved_i8x, b_interleaved_i8x);
                }
            }

            // Store results: convert i32 → f32 with 1/256 scaling
            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svint32_t row_i32x = svread_hor_za32_s32_m(svdup_s32(0), column_predicate_f32x, 1, row);
                svfloat32_t row_f32x = svmul_n_f32_x(column_predicate_f32x,
                                                     svcvt_f32_s32_x(column_predicate_f32x, row_i32x), 1.0f / 256.0f);
                svst1_f32(column_predicate_f32x, result + row_abs * result_stride_elements + column_tile_start,
                          row_f32x);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
}

#pragma endregion

/*
 *  e3m2 × e3m2 → f32 GEMM using inline SSVE conversion + FMOPA.
 *
 *  Pipeline: e3m2 bytes → extract sign/exp/mant → rebuild f16 → FMOPA → f32
 *  - Same tile layout as e5m2 (both convert to f16 before FMOPA)
 *  - E3M2 has 3-bit exponent (bias=3), 2-bit mantissa; requires rebiasing to f16
 *  - Handles subnormals (exp=0, mant!=0) via integer → float conversion + scaling
 *  - No infinity or NaN in E3M2FN format
 *
 *  SME I16I32 alternative - the `svmopa_za32_s16_m` (SMOPA 2-way i16×i16 → i32) was considered for
 *  e3m2 GEMM via an integer LUT (e3m2 → i16, max magnitude 448). Same ZA32 tile geometry (16×16)
 *  and same 2-way expansion as the f16 → f32 path used here — no throughput benefit. Worse, i32
 *  accumulation overflows at depth ~10,698 elements (max product per MOPA = 2 × 448² = 401,408,
 *  i32 max = 2,147,483,647). The f16 → f32 path has no such depth constraint (f32 range ~3.4e38).
 *  Apple M4 has `hw.optional.arm.SME_I16I32: 1` but the feature offers no advantage here.
 */
#pragma region Micro Precision E3M2

/**
 *  Inline `e3m2` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *
 *  The packed-B path already uses the serial `e3m2 → f32 → f16` LUT semantics. Mirror those bit
 *  patterns here with a byte TBL over the 5-bit magnitude, then widen the selected high bytes into
 *  16-bit lanes and apply the sign bit. All representable `e3m2` values map to `f16` bit patterns
 *  with a zero low byte, so a single-byte lookup is sufficient.
 *
 *  @param predicate_f16x Predicate for 16-bit elements
 *  @param bytes_u8x      Pre-loaded bytes (svuint8_t from svld1_u8)
 *  @return               F16 values as svfloat16_t (from lower half of bytes via unpack)
 */
NK_PUBLIC svfloat16_t nk_e3m2x_to_f16x_ssve_(svbool_t predicate_f16x, svuint8_t bytes_u8x) NK_STREAMING_COMPATIBLE_ {
    static NK_ALIGN64 nk_u8_t const magnitude_hi_lut[64] = {
        0x00, 0x2C, 0x30, 0x32, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
        0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
        0x00, 0x2C, 0x30, 0x32, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
        0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
    };

    svuint8_t magnitude_hi_lut_u8x = svld1_u8(svptrue_b8(), magnitude_hi_lut);
    svuint8_t magnitude_u8x = svand_n_u8_x(svptrue_b8(), bytes_u8x, 0x1F);
    svuint8_t magnitude_hi_u8x = svtbl_u8(magnitude_hi_lut_u8x, magnitude_u8x);

    svuint16_t values_u16x = svunpklo_u16(bytes_u8x);
    svuint16_t magnitude_hi_u16x = svunpklo_u16(magnitude_hi_u8x);
    svuint16_t sign_u16x = svlsl_n_u16_x(predicate_f16x, svand_n_u16_x(predicate_f16x, values_u16x, 0x20), 10);
    svuint16_t magnitude_bits_u16x = svlsl_n_u16_x(predicate_f16x, magnitude_hi_u16x, 8);
    return svreinterpret_f16_u16(svorr_u16_x(predicate_f16x, magnitude_bits_u16x, sign_u16x));
}

/**
 *  Fused `e3m2` × `e3m2` → `f32` GEMM kernel using interleaved FMOPA.
 *  Converts `e3m2` → `f16` on-the-fly for A, B is pre-converted during packing.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_packed_e3m2_sme_streaming_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,                                                        //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                                           //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_f16_t const *b_packed_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    // Vectorized e3m2 → f16 conversion
                    nk_e3m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e3m2x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x);
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, svreinterpret_f32_f16(converted_f16x));
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t a_interleaved_vector_f16x = svreinterpret_f16_f32(a_column_f32x);

                    nk_f16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    svfloat16_t b_packed_vector_0_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_0);
                    svfloat16_t b_packed_vector_1_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_1);
                    svfloat16_t b_packed_vector_2_f16x = svld1_f16(predicate_all_f16x,
                                                                   (float16_t const *)b_packed_ptr_2);

                    svmopa_za32_f16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_0_f16x);
                    svmopa_za32_f16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_1_f16x);
                    svmopa_za32_f16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_2_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, predicate_all_f32x, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_e3m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e3m2x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x);
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, svreinterpret_f32_f16(converted_f16x));
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t a_interleaved_vector_f16x = svreinterpret_f16_f32(a_column_f32x);

                    nk_f16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_elements;
                    svfloat16_t b_packed_vector_f16x = svld1_f16(column_predicate_f16x,
                                                                 (float16_t const *)b_packed_ptr);

                    svmopa_za32_f16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_vector_f16x,
                                      b_packed_vector_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_f32x, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_f16_sme(n, k); }

NK_PUBLIC void nk_dots_pack_e3m2_sme(nk_e3m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsh();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e3m2_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) tiles_ptr[i] = 0;

    // Interleaved packing with e3m2 → f16 conversion (via f32 intermediate)
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        nk_f32_t temp_f32;
                        nk_e3m2_to_f32_serial(&b[src_idx], &temp_f32);
                        nk_f32_to_f16_serial(&temp_f32, &vec_output[expansion * column_in_tile + sub_element]);
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_elements * sizeof(nk_f16_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_f32_t *norms_ptr = (nk_f32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_e3m2_t const *col_data = (nk_e3m2_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_e3m2_(col_data, k);
    }
}

/*  `e3m2` × `e3m2` → `f32` GEMM: public interface.
 *  Predicate-based edge handling eliminates scalar fallbacks.
 */
NK_PUBLIC void nk_dots_packed_e3m2_sme(                    //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_e3m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 * `e3m2` × `e3m2` → `f32` symmetric kernel using MOPA self-GEMM.
 *  Time-shares ZA0 for both A and B transposition with e3m2 → f16 conversion.
 *  Pre-reads A columns into Z registers, then reloads ZA0 with converted B data
 *  per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_e3m2_sme_streaming_(
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_f16x = svptrue_b16();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const depth_remainder = depth % expansion;
    nk_u32_t const element_mask_bits = depth_remainder ? (1u << (depth_remainder * (32u / expansion))) - 1u
                                                       : 0xFFFFFFFFu;
    svuint32_t const element_mask_u32x = svdup_u32(element_mask_bits);
    svuint32_t const lane_indices_u32x = svindex_u32(0, 1);

    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_f16x = svwhilelt_b16_u64(0u, rows_actual * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_actual);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                // ZA transpose for A rows: convert e3m2 → f16 then load
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e3m2_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e3m2x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x);
                    svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                    if (clean_last_word)
                        write_f32x = svreinterpret_f32_u32(
                            svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, write_f32x);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                // Load B column tile 0 into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e3m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e3m2x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(1, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }

                // Load B column tile 1 into ZA0, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e3m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e3m2x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(2, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }

                // Load B column tile 2 into ZA0, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e3m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e3m2x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(3, row_predicate_f16x, predicate_all_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_f16x = svwhilelt_b16_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();
                svbool_t const depth_predicate_i8x = svwhilelt_b8_u64(0u, batch_size * expansion);
                svbool_t const depth_predicate_f16x = svwhilelt_b16_u64(0u, batch_size * expansion);

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e3m2_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)a_src);
                    svfloat16_t converted_f16x = nk_e3m2x_to_f16x_ssve_(depth_predicate_f16x, raw_bytes_u8x);
                    svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                    if (clean_last_word)
                        write_f32x = svreinterpret_f32_u32(
                            svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                    svwrite_hor_za32_f32_m(0, row_in_tile, batch_predicate_f32x, write_f32x);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_f32x, 0, s));

                // Load B column tile into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_e3m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw_u8x = svld1_u8(depth_predicate_i8x, (uint8_t const *)src);
                        svfloat16_t converted_f16x = nk_e3m2x_to_f16x_ssve_(depth_predicate_f16x, raw_u8x);
                        svfloat32_t write_f32x = svreinterpret_f32_f16(converted_f16x);
                        if (clean_last_word)
                            write_f32x = svreinterpret_f32_u32(
                                svand_u32_m(cleanup_pred_f32x, svreinterpret_u32_f32(write_f32x), element_mask_u32x));
                        svwrite_hor_za32_f32_m(0, column, batch_predicate_f32x, write_f32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved_f16x = svreinterpret_f16_f32(
                        svld1_f32(predicate_all_f32x, a_buffer[depth_step]));
                    svfloat32_t b_column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_f32x, 0,
                                                                      depth_step);
                    svfloat16_t b_interleaved_f16x = svreinterpret_f16_f32(b_column_f32x);
                    svmopa_za32_f16_m(1, row_predicate_f16x, column_predicate_f16x, a_interleaved_f16x,
                                      b_interleaved_f16x);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_f32x,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
}

#pragma endregion // Signed 8-bit Integers

/*
 *  `u8` × `u8` → `u32` GEMM using SME outer products.
 *
 *  Uses `svmopa_za32_u8_m` for unsigned 8-bit integer outer product accumulate.
 *  Available on Apple M4 (SME_I8I32 = 1, covers both signed and unsigned).
 *
 *  Tile dimensions identical to `i8` → `i32` (512-bit SVL):
 *  - Input vectors: 64 `u8` elements (SVL/8 = 64)
 *  - Output tile: 16 × 16 `u32` elements (`ZA32`)
 *  - Each output `u32` is a dot product of 4 `u8` pairs
 */

#pragma region Unsigned 8-bit Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_i8_sme(n, k); }

NK_PUBLIC void nk_dots_pack_u8_sme(             //
    nk_u8_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_u8_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_u32_t));

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) tiles_ptr[i] = 0;

    // Interleaved packing: packed_vec[expansion * d + k] = B[column_tile*tile_dim + d, depth_step*expansion + k]
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_u8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        vec_output[expansion * column_in_tile + sub_element] = b[src_idx];
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_elements * sizeof(nk_u8_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_u32_t *norms_ptr = (nk_u32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_u8_t const *col_data = (nk_u8_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_u8_(col_data, k);
    }
}

/**
 *  `u8` × `u8` → `u32` GEMM core kernel using SME outer products.
 *  Same interleaved algorithm as i8 kernel, using UMOPA u8→u32.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_u8_sme_streaming_( //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c,                              //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                               //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcntb();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_u8_t const *b_packed_base = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svuint32_t a_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, depth_step);
                    svuint8_t a_interleaved_vector_u8x = svreinterpret_u8_u32(a_column_u32x);

                    nk_u8_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_u8_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_u8_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    svuint8_t b_packed_vector_0_u8x = svld1_u8(predicate_all_i8x, b_packed_ptr_0);
                    svuint8_t b_packed_vector_1_u8x = svld1_u8(predicate_all_i8x, b_packed_ptr_1);
                    svuint8_t b_packed_vector_2_u8x = svld1_u8(predicate_all_i8x, b_packed_ptr_2);

                    svmopa_za32_u8_m(1, row_predicate_i8x, predicate_all_i8x, a_interleaved_vector_u8x,
                                     b_packed_vector_0_u8x);
                    svmopa_za32_u8_m(2, row_predicate_i8x, predicate_all_i8x, a_interleaved_vector_u8x,
                                     b_packed_vector_1_u8x);
                    svmopa_za32_u8_m(3, row_predicate_i8x, predicate_all_i8x, a_interleaved_vector_u8x,
                                     b_packed_vector_2_u8x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, predicate_all_f32x,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + cs0));
                svst1_hor_za32(2, row, predicate_all_f32x,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + cs1));
                svst1_hor_za32(3, row, predicate_all_f32x,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + cs2));
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svuint32_t a_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, depth_step);
                    svuint8_t a_interleaved_vector_u8x = svreinterpret_u8_u32(a_column_u32x);

                    nk_u8_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                   depth_batch_start + depth_step) *
                                                                      vector_elements;
                    svuint8_t b_packed_vector_u8x = svld1_u8(column_predicate_i8x, b_packed_ptr);

                    svmopa_za32_u8_m(1, row_predicate_i8x, column_predicate_i8x, a_interleaved_vector_u8x,
                                     b_packed_vector_u8x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_f32x,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + column_start));
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_u8_sme(                    //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_u32_t);

    nk_dots_packed_u8_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

// expansion=4: each u32 word packs 4 u8 values
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_u8_sme_streaming_(
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_u32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const depth_remainder = depth % expansion;
    nk_u32_t const element_mask_bits = depth_remainder ? (1u << (depth_remainder * (32u / expansion))) - 1u
                                                       : 0xFFFFFFFFu;
    svuint32_t const element_mask_u32x = svdup_u32(element_mask_bits);
    svuint32_t const lane_indices_u32x = svindex_u32(0, 1);

    NK_ALIGN64 nk_u32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_clamped * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    if (clean_last_word) {
                        svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)a_row_ptr);
                        row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                        svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_f32x, row_u32x);
                    }
                    else { svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr); }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_interleaved_u8x = svreinterpret_u8_u32(
                        svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svuint32_t b_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, depth_step);
                    svuint8_t b_interleaved_u8x = svreinterpret_u8_u32(b_column_u32x);
                    svmopa_za32_u8_m(1, row_predicate_i8x, predicate_all_i8x, a_interleaved_u8x, b_interleaved_u8x);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_interleaved_u8x = svreinterpret_u8_u32(
                        svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svuint32_t b_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, depth_step);
                    svuint8_t b_interleaved_u8x = svreinterpret_u8_u32(b_column_u32x);
                    svmopa_za32_u8_m(2, row_predicate_i8x, predicate_all_i8x, a_interleaved_u8x, b_interleaved_u8x);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_interleaved_u8x = svreinterpret_u8_u32(
                        svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svuint32_t b_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, depth_step);
                    svuint8_t b_interleaved_u8x = svreinterpret_u8_u32(b_column_u32x);
                    svmopa_za32_u8_m(3, row_predicate_i8x, predicate_all_i8x, a_interleaved_u8x, b_interleaved_u8x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = (nk_f32_t *)(result + row_abs * result_stride_elements);
                svst1_hor_za32(1, row, predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                nk_size_t const clean_last_word = depth_remainder &&
                                                  (depth_batch_start + batch_size >= depth_step_count);
                svbool_t const cleanup_pred_f32x = clean_last_word ? svcmpeq_n_u32(svptrue_b32(), lane_indices_u32x,
                                                                                   (nk_u32_t)(batch_size - 1))
                                                                   : svpfalse_b();

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    if (clean_last_word) {
                        svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)a_row_ptr);
                        row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                        svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_f32x, row_u32x);
                    }
                    else { svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, a_row_ptr); }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        if (clean_last_word) {
                            svuint32_t row_u32x = svld1_u32(batch_predicate_f32x, (nk_u32_t const *)b_row);
                            row_u32x = svand_u32_m(cleanup_pred_f32x, row_u32x, element_mask_u32x);
                            svwrite_hor_za32_u32_m(0, column, batch_predicate_f32x, row_u32x);
                        }
                        else { svld1_hor_za32(0, column, batch_predicate_f32x, b_row); }
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_interleaved_u8x = svreinterpret_u8_u32(
                        svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svuint32_t b_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_f32x, 0,
                                                                     depth_step);
                    svuint8_t b_interleaved_u8x = svreinterpret_u8_u32(b_column_u32x);
                    svmopa_za32_u8_m(1, row_predicate_i8x, column_predicate_i8x, a_interleaved_u8x, b_interleaved_u8x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_f32x,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_u32_t);
    nk_dots_symmetric_u8_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                        row_start, row_count);
}

#pragma endregion // Unsigned 8-bit Integers

/*
 *  4-bit integer GEMM (u4, i4) using direct mask-and-double-MOPA.
 *
 *  Each byte packs two 4-bit values (nk_u4x2_t / nk_i4x2_t): byte = (hi << 4) | lo.
 *  Pack functions copy raw packed bytes into the tiled layout (no nibble→byte expansion).
 *  Kernels load packed bytes directly into ZA0, then split into low/high nibbles in
 *  registers and issue 2 MOPAs per depth step:
 *
 *    u4: a_low = a & 0x0F,  a_high = a >> 4   → UMOPA(tile, a_low, b_low) + UMOPA(tile, a_high, b_high)
 *    i4: a_low = (a<<4)>>4, a_high = a >> 4    → SMOPA(tile, a_low, b_low) + SMOPA(tile, a_high, b_high)
 *
 *  This eliminates the bounce buffer entirely, halves memory bandwidth, and matches
 *  i8/u8 throughput since the dominant cost is 2 MOPAs per step × ceil(depth/8) steps
 *  = ceil(depth/4) total MOPAs — same as the unpacked path.
 */

#pragma region Nibble Unsigned Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_u4_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const packed_depth = nk_size_divide_round_up_(k, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_step_count * vector_pair_stride * sizeof(nk_u8_t);
    size += n * sizeof(nk_u32_t); // per-column squared norms
    return size;
}

NK_PUBLIC void nk_dots_pack_u4_sme(               //
    nk_u4x2_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t const b_stride_bytes = b_stride / sizeof(nk_u4x2_t);
    nk_size_t const packed_depth = nk_size_divide_round_up_(k, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_u32_t));

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_pair_stride; i++) tiles_ptr[i] = 0;

    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_u8_t *vec_low = tiles_ptr + vec_index * vector_pair_stride;
            nk_u8_t *vec_high = vec_low + vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_byte_base = depth_step * 4;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub = 0; sub < 4; sub++) {
                    nk_size_t const byte_idx = depth_byte_base + sub;
                    if (byte_idx < packed_depth) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_bytes + byte_idx;
                        nk_u8_t val = ((nk_u8_t const *)b)[src_idx];
                        nk_u8_t low_nibble = val & 0x0F;
                        nk_u8_t high_nibble = val >> 4;
                        if (byte_idx == packed_depth - 1 && (k & 1)) high_nibble = 0;
                        nk_size_t const slot = 4 * column_in_tile + sub;
                        vec_low[slot] = low_nibble;
                        vec_high[slot] = high_nibble;
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_pair_stride * sizeof(nk_u8_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_u32_t *norms_ptr = (nk_u32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_u4x2_t const *col_data = (nk_u4x2_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_u4_(col_data, k);
    }
}

/**
 *  `u4` × `u4` → `u32` packed GEMM kernel using SME UMOPA with direct mask-and-double-MOPA.
 *  A input is nibble-packed — loaded directly into ZA0, split into low/high nibbles in registers.
 *  B input is pre-split low/high nibble vectors from nk_dots_pack_u4_sme.
 *  Two UMOPAs per depth step: one for low nibbles, one for high nibbles.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_u4_sme_streaming_( //
    nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c,                            //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                               //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcntb();
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t const depth_steps_per_batch = tile_dimension;
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);

    nk_u8_t const *b_packed_base = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup: mask invalid bytes and odd-depth high nibble
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                            svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0,
                                                                           last_step);
                            column_u32x = svand_n_u32_x(row_predicate_f32x, column_u32x, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_f32x, column_u32x);
                        }
                    }
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svuint32_t a_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, depth_step);
                    svuint8_t a_packed_nibbles_u8x = svreinterpret_u8_u32(a_column_u32x);
                    svuint8_t a_low_nibbles_u8x = svand_n_u8_x(predicate_all_i8x, a_packed_nibbles_u8x, 0x0F);
                    svuint8_t a_high_nibbles_u8x = svlsr_n_u8_x(predicate_all_i8x, a_packed_nibbles_u8x, 4);

                    nk_u8_t const *b_column_pair_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_pair_stride;
                    nk_u8_t const *b_column_pair_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_pair_stride;
                    nk_u8_t const *b_column_pair_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_pair_stride;

                    svuint8_t b_low_nibbles_0_u8x = svld1_u8(predicate_all_i8x, b_column_pair_0);
                    svuint8_t b_high_nibbles_0_u8x = svld1_u8(predicate_all_i8x, b_column_pair_0 + vector_elements);
                    svuint8_t b_low_nibbles_1_u8x = svld1_u8(predicate_all_i8x, b_column_pair_1);
                    svuint8_t b_high_nibbles_1_u8x = svld1_u8(predicate_all_i8x, b_column_pair_1 + vector_elements);
                    svuint8_t b_low_nibbles_2_u8x = svld1_u8(predicate_all_i8x, b_column_pair_2);
                    svuint8_t b_high_nibbles_2_u8x = svld1_u8(predicate_all_i8x, b_column_pair_2 + vector_elements);

                    svmopa_za32_u8_m(1, row_predicate_i8x, predicate_all_i8x, a_low_nibbles_u8x, b_low_nibbles_0_u8x);
                    svmopa_za32_u8_m(1, row_predicate_i8x, predicate_all_i8x, a_high_nibbles_u8x, b_high_nibbles_0_u8x);
                    svmopa_za32_u8_m(2, row_predicate_i8x, predicate_all_i8x, a_low_nibbles_u8x, b_low_nibbles_1_u8x);
                    svmopa_za32_u8_m(2, row_predicate_i8x, predicate_all_i8x, a_high_nibbles_u8x, b_high_nibbles_1_u8x);
                    svmopa_za32_u8_m(3, row_predicate_i8x, predicate_all_i8x, a_low_nibbles_u8x, b_low_nibbles_2_u8x);
                    svmopa_za32_u8_m(3, row_predicate_i8x, predicate_all_i8x, a_high_nibbles_u8x, b_high_nibbles_2_u8x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_u32_t *c_row = c + (row_start + row) * c_stride_elements;
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, predicate_all_f32x, (nk_f32_t *)(c_row + cs0));
                svst1_hor_za32(2, row, predicate_all_f32x, (nk_f32_t *)(c_row + cs1));
                svst1_hor_za32(3, row, predicate_all_f32x, (nk_f32_t *)(c_row + cs2));
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                            svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0,
                                                                           last_step);
                            column_u32x = svand_n_u32_x(row_predicate_f32x, column_u32x, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_f32x, column_u32x);
                        }
                    }
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svuint32_t a_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, depth_step);
                    svuint8_t a_packed_nibbles_u8x = svreinterpret_u8_u32(a_column_u32x);
                    svuint8_t a_low_nibbles_u8x = svand_n_u8_x(predicate_all_i8x, a_packed_nibbles_u8x, 0x0F);
                    svuint8_t a_high_nibbles_u8x = svlsr_n_u8_x(predicate_all_i8x, a_packed_nibbles_u8x, 4);

                    nk_u8_t const *b_column_pair = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_pair_stride;
                    svuint8_t b_low_nibbles_u8x = svld1_u8(column_predicate_i8x, b_column_pair);
                    svuint8_t b_high_nibbles_u8x = svld1_u8(column_predicate_i8x, b_column_pair + vector_elements);

                    svmopa_za32_u8_m(1, row_predicate_i8x, column_predicate_i8x, a_low_nibbles_u8x, b_low_nibbles_u8x);
                    svmopa_za32_u8_m(1, row_predicate_i8x, column_predicate_i8x, a_high_nibbles_u8x,
                                     b_high_nibbles_u8x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_f32x,
                               (nk_f32_t *)(c + (row_start + row) * c_stride_elements + column_start));
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_u4_sme(                      //
    nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_u32_t);
    nk_dots_packed_u4_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

#pragma endregion // Nibble Unsigned Integers

#pragma region Nibble Signed Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_i4_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const packed_depth = nk_size_divide_round_up_(k, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_step_count * vector_pair_stride * sizeof(nk_u8_t);
    size += n * sizeof(nk_u32_t); // per-column squared norms
    return size;
}

NK_PUBLIC void nk_dots_pack_i4_sme(               //
    nk_i4x2_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t const b_stride_bytes = b_stride / sizeof(nk_i4x2_t);
    nk_size_t const packed_depth = nk_size_divide_round_up_(k, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_i32_t));

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_pair_stride; i++) tiles_ptr[i] = 0;

    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_u8_t *vec_low = tiles_ptr + vec_index * vector_pair_stride;
            nk_u8_t *vec_high = vec_low + vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_byte_base = depth_step * 4;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub = 0; sub < 4; sub++) {
                    nk_size_t const byte_idx = depth_byte_base + sub;
                    if (byte_idx < packed_depth) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_bytes + byte_idx;
                        nk_u8_t val = ((nk_u8_t const *)b)[src_idx];
                        nk_u8_t low_nibble = val & 0x0F;
                        nk_u8_t high_nibble = (val >> 4) & 0x0F;
                        if (byte_idx == packed_depth - 1 && (k & 1)) high_nibble = 0;
                        // Sign-extend 4-bit to 8-bit: (nibble ^ 8) - 8
                        nk_i8_t low_extended = (nk_i8_t)((low_nibble ^ 8) - 8);
                        nk_i8_t high_extended = (nk_i8_t)((high_nibble ^ 8) - 8);
                        nk_size_t const slot = 4 * column_in_tile + sub;
                        vec_low[slot] = (nk_u8_t)low_extended;
                        vec_high[slot] = (nk_u8_t)high_extended;
                    }
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_vectors * vector_pair_stride * sizeof(nk_u8_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_u32_t *norms_ptr = (nk_u32_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < n; col++) {
        nk_i4x2_t const *col_data = (nk_i4x2_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_i4_(col_data, k);
    }
}

/**
 *  `i4` × `i4` → `i32` packed GEMM kernel using SME SMOPA with direct mask-and-double-MOPA.
 *  A input is nibble-packed — loaded directly into ZA0, sign-extended via LSL+ASR in registers.
 *  B input is pre-split sign-extended nibble vectors from nk_dots_pack_i4_sme.
 *  Two SMOPAs per depth step: one for low nibbles, one for high nibbles.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_i4_sme_streaming_( //
    nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c,                            //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                               //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcntb();
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t const depth_steps_per_batch = tile_dimension;
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);

    nk_u8_t const *b_packed_base = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup: mask invalid bytes and odd-depth high nibble
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                            svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0,
                                                                           last_step);
                            column_u32x = svand_n_u32_x(row_predicate_f32x, column_u32x, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_f32x, column_u32x);
                        }
                    }
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svint32_t a_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, depth_step);
                    svint8_t a_packed_nibbles_i8x = svreinterpret_s8_s32(a_column_i32x);
                    svint8_t a_low_nibbles_i8x = svasr_n_s8_x(
                        predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, a_packed_nibbles_i8x, 4), 4);
                    svint8_t a_high_nibbles_i8x = svasr_n_s8_x(predicate_all_i8x, a_packed_nibbles_i8x, 4);

                    nk_i8_t const *b_column_pair_0 = (nk_i8_t const *)(b_packed_base +
                                                                       ((column_tile_index + 0) * depth_step_count +
                                                                        depth_batch_start + depth_step) *
                                                                           vector_pair_stride);
                    nk_i8_t const *b_column_pair_1 = (nk_i8_t const *)(b_packed_base +
                                                                       ((column_tile_index + 1) * depth_step_count +
                                                                        depth_batch_start + depth_step) *
                                                                           vector_pair_stride);
                    nk_i8_t const *b_column_pair_2 = (nk_i8_t const *)(b_packed_base +
                                                                       ((column_tile_index + 2) * depth_step_count +
                                                                        depth_batch_start + depth_step) *
                                                                           vector_pair_stride);

                    svint8_t b_low_nibbles_0_i8x = svld1_s8(predicate_all_i8x, b_column_pair_0);
                    svint8_t b_high_nibbles_0_i8x = svld1_s8(predicate_all_i8x, b_column_pair_0 + vector_elements);
                    svint8_t b_low_nibbles_1_i8x = svld1_s8(predicate_all_i8x, b_column_pair_1);
                    svint8_t b_high_nibbles_1_i8x = svld1_s8(predicate_all_i8x, b_column_pair_1 + vector_elements);
                    svint8_t b_low_nibbles_2_i8x = svld1_s8(predicate_all_i8x, b_column_pair_2);
                    svint8_t b_high_nibbles_2_i8x = svld1_s8(predicate_all_i8x, b_column_pair_2 + vector_elements);

                    svmopa_za32_s8_m(1, row_predicate_i8x, predicate_all_i8x, a_low_nibbles_i8x, b_low_nibbles_0_i8x);
                    svmopa_za32_s8_m(1, row_predicate_i8x, predicate_all_i8x, a_high_nibbles_i8x, b_high_nibbles_0_i8x);
                    svmopa_za32_s8_m(2, row_predicate_i8x, predicate_all_i8x, a_low_nibbles_i8x, b_low_nibbles_1_i8x);
                    svmopa_za32_s8_m(2, row_predicate_i8x, predicate_all_i8x, a_high_nibbles_i8x, b_high_nibbles_1_i8x);
                    svmopa_za32_s8_m(3, row_predicate_i8x, predicate_all_i8x, a_low_nibbles_i8x, b_low_nibbles_2_i8x);
                    svmopa_za32_s8_m(3, row_predicate_i8x, predicate_all_i8x, a_high_nibbles_i8x, b_high_nibbles_2_i8x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_i32_t *c_row = c + (row_start + row) * c_stride_elements;
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, predicate_all_f32x, (nk_f32_t *)(c_row + cs0));
                svst1_hor_za32(2, row, predicate_all_f32x, (nk_f32_t *)(c_row + cs1));
                svst1_hor_za32(3, row, predicate_all_f32x, (nk_f32_t *)(c_row + cs2));
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                            svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0,
                                                                           last_step);
                            column_u32x = svand_n_u32_x(row_predicate_f32x, column_u32x, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_f32x, column_u32x);
                        }
                    }
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svint32_t a_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_f32x, 0, depth_step);
                    svint8_t a_packed_nibbles_i8x = svreinterpret_s8_s32(a_column_i32x);
                    svint8_t a_low_nibbles_i8x = svasr_n_s8_x(
                        predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, a_packed_nibbles_i8x, 4), 4);
                    svint8_t a_high_nibbles_i8x = svasr_n_s8_x(predicate_all_i8x, a_packed_nibbles_i8x, 4);

                    nk_i8_t const *b_column_pair =
                        (nk_i8_t const *)(b_packed_base +
                                          (column_tile_index * depth_step_count + depth_batch_start + depth_step) *
                                              vector_pair_stride);
                    svint8_t b_low_nibbles_i8x = svld1_s8(column_predicate_i8x, b_column_pair);
                    svint8_t b_high_nibbles_i8x = svld1_s8(column_predicate_i8x, b_column_pair + vector_elements);

                    svmopa_za32_s8_m(1, row_predicate_i8x, column_predicate_i8x, a_low_nibbles_i8x, b_low_nibbles_i8x);
                    svmopa_za32_s8_m(1, row_predicate_i8x, column_predicate_i8x, a_high_nibbles_i8x,
                                     b_high_nibbles_i8x);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_f32x,
                               (nk_f32_t *)(c + (row_start + row) * c_stride_elements + column_start));
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_i4_sme(                      //
    nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);
    nk_dots_packed_i4_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *  `u4` × `u4` → `u32` symmetric kernel using SME UMOPA with direct mask-and-double-MOPA.
 *  Loads packed nibble bytes directly into ZA0, splits into low/high nibbles in registers,
 *  issues 2 UMOPAs per depth step. ZA0 = staging tile, ZA1-ZA3 = accumulators.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_u4_sme_streaming_(
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_u32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    NK_ALIGN64 nk_u32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_clamped * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // Load A rows into ZA0 (packed bytes directly from source)
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup for A
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                            svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0,
                                                                           last_step);
                            column_u32x = svand_n_u32_x(row_predicate_f32x, column_u32x, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_f32x, column_u32x);
                        }
                    }
                }

                // Cache A columns
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_f32x, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, last_step);
                        column_u32x = svand_n_u32_x(predicate_all_f32x, column_u32x, mask);
                        svwrite_ver_za32_u32_m(0, last_step, predicate_all_f32x, column_u32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_u8x = svreinterpret_u8_u32(svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svuint32_t b_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, depth_step);
                    svuint8_t b_u8x = svreinterpret_u8_u32(b_column_u32x);
                    svuint8_t a_low_u8x = svand_n_u8_x(predicate_all_i8x, a_u8x, 0x0F);
                    svuint8_t a_high_u8x = svlsr_n_u8_x(predicate_all_i8x, a_u8x, 4);
                    svuint8_t b_low_u8x = svand_n_u8_x(predicate_all_i8x, b_u8x, 0x0F);
                    svuint8_t b_high_u8x = svlsr_n_u8_x(predicate_all_i8x, b_u8x, 4);
                    svmopa_za32_u8_m(1, row_predicate_i8x, predicate_all_i8x, a_low_u8x, b_low_u8x);
                    svmopa_za32_u8_m(1, row_predicate_i8x, predicate_all_i8x, a_high_u8x, b_high_u8x);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_f32x, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, last_step);
                        column_u32x = svand_n_u32_x(predicate_all_f32x, column_u32x, mask);
                        svwrite_ver_za32_u32_m(0, last_step, predicate_all_f32x, column_u32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_u8x = svreinterpret_u8_u32(svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svuint32_t b_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, depth_step);
                    svuint8_t b_u8x = svreinterpret_u8_u32(b_column_u32x);
                    svuint8_t a_low_u8x = svand_n_u8_x(predicate_all_i8x, a_u8x, 0x0F);
                    svuint8_t a_high_u8x = svlsr_n_u8_x(predicate_all_i8x, a_u8x, 4);
                    svuint8_t b_low_u8x = svand_n_u8_x(predicate_all_i8x, b_u8x, 0x0F);
                    svuint8_t b_high_u8x = svlsr_n_u8_x(predicate_all_i8x, b_u8x, 4);
                    svmopa_za32_u8_m(2, row_predicate_i8x, predicate_all_i8x, a_low_u8x, b_low_u8x);
                    svmopa_za32_u8_m(2, row_predicate_i8x, predicate_all_i8x, a_high_u8x, b_high_u8x);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_f32x, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, last_step);
                        column_u32x = svand_n_u32_x(predicate_all_f32x, column_u32x, mask);
                        svwrite_ver_za32_u32_m(0, last_step, predicate_all_f32x, column_u32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_u8x = svreinterpret_u8_u32(svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svuint32_t b_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, depth_step);
                    svuint8_t b_u8x = svreinterpret_u8_u32(b_column_u32x);
                    svuint8_t a_low_u8x = svand_n_u8_x(predicate_all_i8x, a_u8x, 0x0F);
                    svuint8_t a_high_u8x = svlsr_n_u8_x(predicate_all_i8x, a_u8x, 4);
                    svuint8_t b_low_u8x = svand_n_u8_x(predicate_all_i8x, b_u8x, 0x0F);
                    svuint8_t b_high_u8x = svlsr_n_u8_x(predicate_all_i8x, b_u8x, 4);
                    svmopa_za32_u8_m(3, row_predicate_i8x, predicate_all_i8x, a_low_u8x, b_low_u8x);
                    svmopa_za32_u8_m(3, row_predicate_i8x, predicate_all_i8x, a_high_u8x, b_high_u8x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = (nk_f32_t *)(result + row_abs * result_stride_elements);
                svst1_hor_za32(1, row, predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup for A
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                            svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0,
                                                                           last_step);
                            column_u32x = svand_n_u32_x(row_predicate_f32x, column_u32x, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_f32x, column_u32x);
                        }
                    }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_f32x, (nk_f32_t const *)src);
                    }
                }
                // Last-step ZA0 fixup for B
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_f32x, 0,
                                                                       last_step);
                        column_u32x = svand_n_u32_x(column_predicate_f32x, column_u32x, mask);
                        svwrite_ver_za32_u32_m(0, last_step, column_predicate_f32x, column_u32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_u8x = svreinterpret_u8_u32(svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svuint32_t b_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_f32x, 0,
                                                                     depth_step);
                    svuint8_t b_u8x = svreinterpret_u8_u32(b_column_u32x);
                    svuint8_t a_low_u8x = svand_n_u8_x(predicate_all_i8x, a_u8x, 0x0F);
                    svuint8_t a_high_u8x = svlsr_n_u8_x(predicate_all_i8x, a_u8x, 4);
                    svuint8_t b_low_u8x = svand_n_u8_x(predicate_all_i8x, b_u8x, 0x0F);
                    svuint8_t b_high_u8x = svlsr_n_u8_x(predicate_all_i8x, b_u8x, 4);
                    svmopa_za32_u8_m(1, row_predicate_i8x, column_predicate_i8x, a_low_u8x, b_low_u8x);
                    svmopa_za32_u8_m(1, row_predicate_i8x, column_predicate_i8x, a_high_u8x, b_high_u8x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_f32x,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                        nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_u32_t);
    nk_dots_symmetric_u4_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                        row_start, row_count);
}

/**
 *  `i4` × `i4` → `i32` symmetric kernel using SME SMOPA with direct mask-and-double-MOPA.
 *  Loads packed nibble bytes directly into ZA0, sign-extends via LSL+ASR in registers,
 *  issues 2 SMOPAs per depth step. ZA0 = staging tile, ZA1-ZA3 = accumulators.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_i4_sme_streaming_(
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_i32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_i8x = svptrue_b8();
    svbool_t const predicate_all_f32x = svptrue_b32();

    NK_ALIGN64 nk_u32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_i8x = svwhilelt_b8_u64(0u, rows_clamped * expansion);
        svbool_t const row_predicate_f32x = svwhilelt_b32_u64(0u, rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // Load A rows into ZA0 (packed bytes directly from source)
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup for A
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                            svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0,
                                                                           last_step);
                            column_u32x = svand_n_u32_x(row_predicate_f32x, column_u32x, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_f32x, column_u32x);
                        }
                    }
                }

                // Cache A columns
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_f32x, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, last_step);
                        column_u32x = svand_n_u32_x(predicate_all_f32x, column_u32x, mask);
                        svwrite_ver_za32_u32_m(0, last_step, predicate_all_f32x, column_u32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_i8x = svreinterpret_s8_u32(svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_f32x, 0, depth_step);
                    svint8_t b_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svint8_t a_low_i8x = svasr_n_s8_x(predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, a_i8x, 4), 4);
                    svint8_t a_high_i8x = svasr_n_s8_x(predicate_all_i8x, a_i8x, 4);
                    svint8_t b_low_i8x = svasr_n_s8_x(predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, b_i8x, 4), 4);
                    svint8_t b_high_i8x = svasr_n_s8_x(predicate_all_i8x, b_i8x, 4);
                    svmopa_za32_s8_m(1, row_predicate_i8x, predicate_all_i8x, a_low_i8x, b_low_i8x);
                    svmopa_za32_s8_m(1, row_predicate_i8x, predicate_all_i8x, a_high_i8x, b_high_i8x);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_f32x, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, last_step);
                        column_u32x = svand_n_u32_x(predicate_all_f32x, column_u32x, mask);
                        svwrite_ver_za32_u32_m(0, last_step, predicate_all_f32x, column_u32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_i8x = svreinterpret_s8_u32(svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_f32x, 0, depth_step);
                    svint8_t b_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svint8_t a_low_i8x = svasr_n_s8_x(predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, a_i8x, 4), 4);
                    svint8_t a_high_i8x = svasr_n_s8_x(predicate_all_i8x, a_i8x, 4);
                    svint8_t b_low_i8x = svasr_n_s8_x(predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, b_i8x, 4), 4);
                    svint8_t b_high_i8x = svasr_n_s8_x(predicate_all_i8x, b_i8x, 4);
                    svmopa_za32_s8_m(2, row_predicate_i8x, predicate_all_i8x, a_low_i8x, b_low_i8x);
                    svmopa_za32_s8_m(2, row_predicate_i8x, predicate_all_i8x, a_high_i8x, b_high_i8x);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_f32x, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_f32x, 0, last_step);
                        column_u32x = svand_n_u32_x(predicate_all_f32x, column_u32x, mask);
                        svwrite_ver_za32_u32_m(0, last_step, predicate_all_f32x, column_u32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_i8x = svreinterpret_s8_u32(svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_f32x, 0, depth_step);
                    svint8_t b_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svint8_t a_low_i8x = svasr_n_s8_x(predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, a_i8x, 4), 4);
                    svint8_t a_high_i8x = svasr_n_s8_x(predicate_all_i8x, a_i8x, 4);
                    svint8_t b_low_i8x = svasr_n_s8_x(predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, b_i8x, 4), 4);
                    svint8_t b_high_i8x = svasr_n_s8_x(predicate_all_i8x, b_i8x, 4);
                    svmopa_za32_s8_m(3, row_predicate_i8x, predicate_all_i8x, a_low_i8x, b_low_i8x);
                    svmopa_za32_s8_m(3, row_predicate_i8x, predicate_all_i8x, a_high_i8x, b_high_i8x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = (nk_f32_t *)(result + row_abs * result_stride_elements);
                svst1_hor_za32(1, row, predicate_all_f32x, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, predicate_all_f32x, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, predicate_all_f32x, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f32x = svwhilelt_b32_u64(0u, cols_remaining);
            svbool_t const column_predicate_i8x = svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_f32x = svwhilelt_b32_u64(0u, batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_f32x, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup for A
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                            svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0,
                                                                           last_step);
                            column_u32x = svand_n_u32_x(row_predicate_f32x, column_u32x, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_f32x, column_u32x);
                        }
                    }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(predicate_all_f32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_f32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_f32x, (nk_f32_t const *)src);
                    }
                }
                // Last-step ZA0 fixup for B
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column_u32x = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_f32x, 0,
                                                                       last_step);
                        column_u32x = svand_n_u32_x(column_predicate_f32x, column_u32x, mask);
                        svwrite_ver_za32_u32_m(0, last_step, column_predicate_f32x, column_u32x);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_i8x = svreinterpret_s8_u32(svld1_u32(predicate_all_f32x, a_buffer[depth_step]));
                    svint32_t b_column_i32x = svread_ver_za32_s32_m(svdup_s32(0), column_predicate_f32x, 0, depth_step);
                    svint8_t b_i8x = svreinterpret_s8_s32(b_column_i32x);
                    svint8_t a_low_i8x = svasr_n_s8_x(predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, a_i8x, 4), 4);
                    svint8_t a_high_i8x = svasr_n_s8_x(predicate_all_i8x, a_i8x, 4);
                    svint8_t b_low_i8x = svasr_n_s8_x(predicate_all_i8x, svlsl_n_s8_x(predicate_all_i8x, b_i8x, 4), 4);
                    svint8_t b_high_i8x = svasr_n_s8_x(predicate_all_i8x, b_i8x, 4);
                    svmopa_za32_s8_m(1, row_predicate_i8x, column_predicate_i8x, a_low_i8x, b_low_i8x);
                    svmopa_za32_s8_m(1, row_predicate_i8x, column_predicate_i8x, a_high_i8x, b_high_i8x);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_f32x,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                        nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);
    nk_dots_symmetric_i4_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                        row_start, row_count);
}

#pragma endregion // Nibble Signed Integers
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_
#endif // NK_DOTS_SME_H
