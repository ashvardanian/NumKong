/**
 *  @brief SIMD-accelerated Batched Dot Products for SME F64.
 *  @file include/numkong/dots/smef64.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses ARM SME with `FEAT_SME_F64F64` for high-precision GEMM.
 *  Requires Apple M4 or equivalent with `f64` outer product support.
 *
 *  Provides `f32` and `f64` GEMM using `ZA64` tiles:
 *  - `f32` inputs with `f64` accumulation: higher precision than `ZA32`
 *  - Native `f64` GEMM via 3-way Ozaki splitting (19+17+17 mantissa bits)
 *
 *  Ozaki splitting for `f64`:
 *  Each `f64` value is decomposed into 3 non-overlapping mantissa-masked slices
 *  that each fit in `f32` (max 19 significant bits < 24). All cross-products with
 *  index sum i+j <= 2 are accumulated via 6 FMOPAs into 3 merged accumulators.
 *  Products are exact in `f64` (max 19+19 = 38 < 53 mantissa bits).
 *  B is pre-split at pack time into interleaved `f32` slices; A is split in-register.
 *  A 2-column-tile fast path halves A memory traffic.
 *
 *  Tile dimensions for SVL=512 (Apple M4):
 *  - `ZA64` tile: 8 × 8 `f64` elements (512B)
 *  - `f64` vectors: 8 elements per SVE vector
 *  - `f32` vectors: 16 elements per SVE vector, converted to `f64`
 *
 *  Key instructions:
 *  - `svmopa_za64_f64_m` / `FMOPA`: `f64` outer product, 16cy amortized
 *  - `svcvt_f64_f32_x` / `FCVT`: `f32` → `f64` conversion
 *  - `svwrite_hor_za64_f64_m` / `MOVA`: direct Z → ZA tile write (no bounce buffer)
 */
#ifndef NK_DOTS_SMEF64_H
#define NK_DOTS_SMEF64_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/types.h"
#include "numkong/dots/sme.h" // `nk_dots_sme_packed_header_t`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sve,sme-f64f64"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#endif

/*
 *  f32 → f64 GEMM using FMOPA with ZA64 tiles (FEAT_SME_F64F64).
 *
 *  Tile layout (SVL=512, Apple M4):
 *  - ZA64 output tile: 8 × 8 f64 elements (512 B)
 *  - f32 input vectors: 16 elements (SVL/32), converted to f64 in chunks of 8
 *  - Depth sub-loop: processes 8 f32 values per iteration (→ 8 f64)
 *  - FMOPA predicates: b64 (f64 output granularity)
 *  - f32 load predicates: b32 (f32 input granularity)
 *  - 4-tile path: ZA0-ZA3 process 4 column tiles simultaneously
 *  - Output: native f64 results written directly from ZA64 tiles
 *
 *  Non-widening alternative (FEAT_SME_F32F32, `svmopa_za32_f32_m`): ZA32 tiles are 16×16
 *  (4× area vs ZA64 8×8) with no f32↔f64 conversion, offering ~3-4× raw throughput. However,
 *  ZA32 and ZA64 tiles alias physically (ZA0.S overlaps ZA0.D+ZA1.D), so a periodic flush to
 *  f64 stack accumulators would be needed for precision above f32 — erasing most speedup.
 *  Pure f32 accumulation (no flush) provides only f32 precision, which is already served by
 *  the f16 → f32 GEMM path for reduced-precision workloads. This f64 path exists specifically
 *  for higher-than-f32 accumulation precision; replacing it with f32 FMOPA would be
 *  counterproductive. Apple M4 has `hw.optional.arm.SME_F32F32: 1` but we don't use it here.
 */
#pragma region Single Precision Floats

NK_PUBLIC nk_size_t nk_dots_packed_size_f32_smef64(nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = svcntsd();  // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsw(); // `f32` depth elements per tile (16 for SVL=512)

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_tile_count * tile_dimension * depth_tile_size * sizeof(nk_f32_t);
    size += columns * sizeof(nk_f64_t); // per-column squared norms

    return size;
}

NK_PUBLIC void nk_dots_pack_f32_smef64(nk_f32_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed) {

    nk_size_t const tile_dimension = svcntsd();                       // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsw();                      // `f32` depth elements per tile (16 for SVL=512)
    nk_size_t const tile_elements = tile_dimension * depth_tile_size; // 128
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f32_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    // Store actual dimensions and tile counts in header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)columns;
    header->depth = (nk_u32_t)depth;
    header->svl_bytes = (nk_u32_t)svcntsb(); // streaming vector length in bytes

    nk_f32_t *tiles = (nk_f32_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles (handles partial tile padding)
    nk_size_t const total_elements = total_tiles * tile_elements;
    for (nk_size_t i = 0; i < total_elements; i++) tiles[i] = 0.0f;

    // Pack data into tiles with depth-major layout within each tile:
    // dst_idx = depth_idx * tile_dimension + column_idx
    // This allows loading one B vector per depth step: svld1(b_tile + k * tile_dimension)
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tile_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tile_count + depth_tile_idx;
            nk_f32_t *tile_output = tiles + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tile_dimension;
            nk_size_t const src_column_start = depth_tile_idx * depth_tile_size;

            // Handle partial tiles at edges
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= columns) ? tile_dimension
                                                                                       : (columns - src_row_start);
            nk_size_t const columns_to_pack = (src_column_start + depth_tile_size <= depth)
                                                  ? depth_tile_size
                                                  : (depth - src_column_start);

            for (nk_size_t column_idx = 0; column_idx < rows_to_pack; column_idx++) {
                for (nk_size_t depth_idx = 0; depth_idx < columns_to_pack; depth_idx++) {
                    nk_size_t const src_idx = (src_row_start + column_idx) * b_stride_elements + src_column_start +
                                              depth_idx;
                    nk_size_t const dst_idx = depth_idx * tile_dimension + column_idx;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_tiles * tile_elements * sizeof(nk_f32_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_f64_t *norms_ptr = (nk_f64_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < columns; col++) {
        nk_f32_t const *col_data = (nk_f32_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_f32_(col_data, depth);
    }
}

__arm_locally_streaming __arm_new("za") static void nk_dots_packed_f32_smef64_streaming_(
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntd();  // 8 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 for 512-bit SVL
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const depth_steps_per_batch = tile_dimension; // 8 depth steps per ZA0.D load

    nk_f32_t const *b_tiles = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_f64x = svptrue_b64();

    // ZA0.D = staging, ZA1-7.D = accumulation (7-tile fast path)
    for (nk_size_t row_tile_index = 0; row_tile_index < nk_size_divide_round_up_(rows, tile_dimension);
         row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_f64x = svwhilelt_b64_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 7 column tiles using ZA1-ZA7 (ZA0.D = staging)
        for (; column_tile_index + 7 <= column_tile_count; column_tile_index += 7) {
            svzero_mask_za(nk_sme_zero_za64_tiles_1_7_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                // Process depth_tile_size elements in batches of tile_dimension (8)
                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    // Check if any elements in this batch are valid
                    if (depth_offset + depth_batch_start >= depth) break;

                    svzero_mask_za(nk_sme_zero_za64_tile_0_);

                    // Load A rows into ZA0.D: extending load f32→u64 + convert to f64
                    svbool_t const batch_predicate_f64x = svwhilelt_b64_u64(0u, (uint64_t)batch_size);
                    svbool_t const a_depth_predicate_f64x = svwhilelt_b64_u64(depth_offset + depth_batch_start,
                                                                              (uint64_t)depth);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        // Extending load: svld1uw_u64 loads f32 bits into lower 32 of each u64 lane
                        svfloat64_t a_row_widened_f64x = svcvt_f64_f32_x(
                            batch_predicate_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_predicate_f64x,
                                (nk_u32_t const *)&a[a_row * a_stride_elements + depth_offset + depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_f64x, a_row_widened_f64x);
                    }

                    // Vertical read + MOPA for each depth step in batch
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;

                        svfloat64_t a_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_f64x, 0, step);

                        nk_size_t const b_k = depth_batch_start + step;

                        // Extending load f32→u64 + convert to f64: svld1uw_u64 replaces svld1_f32 + svunpklo_u64
                        svfloat64_t b_column_tile_1_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 0) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_2_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 1) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_3_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 2) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_4_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 3) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_5_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 4) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_6_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 5) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_7_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 6) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));

                        svmopa_za64_f64_m(1, row_predicate_f64x, predicate_all_f64x, a_f64x, b_column_tile_1_f64x);
                        svmopa_za64_f64_m(2, row_predicate_f64x, predicate_all_f64x, a_f64x, b_column_tile_2_f64x);
                        svmopa_za64_f64_m(3, row_predicate_f64x, predicate_all_f64x, a_f64x, b_column_tile_3_f64x);
                        svmopa_za64_f64_m(4, row_predicate_f64x, predicate_all_f64x, a_f64x, b_column_tile_4_f64x);
                        svmopa_za64_f64_m(5, row_predicate_f64x, predicate_all_f64x, a_f64x, b_column_tile_5_f64x);
                        svmopa_za64_f64_m(6, row_predicate_f64x, predicate_all_f64x, a_f64x, b_column_tile_6_f64x);
                        svmopa_za64_f64_m(7, row_predicate_f64x, predicate_all_f64x, a_f64x, b_column_tile_7_f64x);
                    }
                }
            }

            // Extract from ZA1-7 and store native f64 outputs.
            svbool_t const predicate_tile_f64x = svwhilelt_b64_u64(0u, tile_dimension);
            // The 7th tile (index 6) may be partial when it's the last column tile
            nk_size_t const last_fast_col_start = (column_tile_index + 6) * tile_dimension;
            nk_size_t const last_fast_cols = (last_fast_col_start + tile_dimension <= columns)
                                                 ? tile_dimension
                                                 : (columns - last_fast_col_start);
            svbool_t const last_tile_pred_f64x = svwhilelt_b64_u64(0u, last_fast_cols);
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f64_t *c_row = c + (row_start + row_idx) * c_stride_elements;

                svfloat64_t za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 1, row_idx);
                svst1_f64(predicate_tile_f64x, c_row + (column_tile_index + 0) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 2, row_idx);
                svst1_f64(predicate_tile_f64x, c_row + (column_tile_index + 1) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 3, row_idx);
                svst1_f64(predicate_tile_f64x, c_row + (column_tile_index + 2) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 4, row_idx);
                svst1_f64(predicate_tile_f64x, c_row + (column_tile_index + 3) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 5, row_idx);
                svst1_f64(predicate_tile_f64x, c_row + (column_tile_index + 4) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 6, row_idx);
                svst1_f64(predicate_tile_f64x, c_row + (column_tile_index + 5) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 7, row_idx);
                svst1_f64(last_tile_pred_f64x, c_row + (column_tile_index + 6) * tile_dimension, za_row_f64x);
            }
        }

        // Remainder: 1 column tile at a time using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                           : (columns - column_start);
            svbool_t const column_predicate_f64x = svwhilelt_b64_u64(0u, columns_remaining);

            svzero_mask_za(nk_sme_zero_za64_tile_1_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    svzero_mask_za(nk_sme_zero_za64_tile_0_);

                    svbool_t const batch_predicate_f64x = svwhilelt_b64_u64(0u, (uint64_t)batch_size);
                    svbool_t const a_depth_pred_f64x = svwhilelt_b64_u64(depth_offset + depth_batch_start,
                                                                         (uint64_t)depth);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        svfloat64_t a_row_widened_f64x = svcvt_f64_f32_x(
                            batch_predicate_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_pred_f64x,
                                (nk_u32_t const *)&a[a_row * a_stride_elements + depth_offset + depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_f64x, a_row_widened_f64x);
                    }

                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;

                        svfloat64_t a_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_f64x, 0, step);

                        nk_size_t const b_k = depth_batch_start + step;
                        nk_f32_t const *b_tile = b_tiles + (column_tile_index * depth_tile_count + depth_tile_idx) *
                                                               tile_elements;
                        // Extending load f32→u64 + convert to f64
                        svfloat64_t b_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(
                                svld1uw_u64(predicate_all_f64x, (nk_u32_t const *)(b_tile + b_k * tile_dimension))));

                        svmopa_za64_f64_m(1, row_predicate_f64x, column_predicate_f64x, a_f64x, b_f64x);
                    }
                }
            }

            // Store native f64 outputs for the tail column tile.
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                svfloat64_t za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 1, row_idx);
                nk_f64_t *c_row = c + (row_start + row_idx) * c_stride_elements + column_start;
                svst1_f64(column_predicate_f64x, c_row, za_row_f64x);
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                         nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_dots_packed_f32_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *  `f32` × `f32` → `f32` symmetric kernel using MOPA self-GEMM with f64 accumulation.
 *  Time-shares ZA0 for both A and B transposition: loads A horizontally,
 *  pre-reads A columns into Z registers, then reloads ZA0 with widened B data
 *  per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_f32_smef64_streaming_(
    nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const tile_dimension = svcntd();              // 8 for SVL=512
    nk_size_t const depth_tile_size = svcntw();             // 16 for SVL=512
    nk_size_t const depth_steps_per_batch = tile_dimension; // 8

    svbool_t const predicate_all_f64x = svptrue_b64();

    NK_ALIGN64 nk_f64_t a_buffer[8][8];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_f64x = svwhilelt_b64_u64(0u, rows_actual);

        nk_size_t column_tile_index = 0;

        // Fast path: 7 column tiles at a time
        for (; column_tile_index + 7 <= column_tile_count; column_tile_index += 7) {
            svzero_mask_za(nk_sme_zero_za64_tiles_1_7_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                // Process depth_tile_size in batches of depth_steps_per_batch (8)
                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    // ZA transpose for A rows: extending load f32→f64, MOVA directly into ZA0
                    svbool_t const batch_predicate_f64x = svwhilelt_b64_u64(0u, (uint64_t)batch_size);
                    svbool_t const a_depth_predicate_f64x = svwhilelt_b64_u64(depth_offset + depth_batch_start,
                                                                              (uint64_t)depth);
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                        nk_size_t const row_abs = row_tile_start + row_in_tile;
                        svfloat64_t a_row_widened_f64x = svcvt_f64_f32_x(
                            batch_predicate_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_predicate_f64x, (nk_u32_t const *)&vectors[row_abs * stride_elements +
                                                                                   depth_offset + depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_f64x, a_row_widened_f64x);
                    }

                    // Save A columns from ZA0 to stack buffer
                    for (nk_size_t s = 0; s < batch_size; s++)
                        svst1_f64(predicate_all_f64x, a_buffer[s],
                                  svread_ver_za64_f64_m(svdup_f64(0), row_predicate_f64x, 0, s));

                    // Column tile 0 → ZA1 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_f64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_f64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_f64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_f64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 0, step);
                        svmopa_za64_f64_m(1, row_predicate_f64x, predicate_all_f64x, a_f64x, b_f64x);
                    }

                    // Column tile 1 → ZA2 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_f64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_f64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_f64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_f64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 0, step);
                        svmopa_za64_f64_m(2, row_predicate_f64x, predicate_all_f64x, a_f64x, b_f64x);
                    }

                    // Column tile 2 → ZA3 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_f64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_f64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_f64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_f64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 0, step);
                        svmopa_za64_f64_m(3, row_predicate_f64x, predicate_all_f64x, a_f64x, b_f64x);
                    }

                    // Column tile 3 → ZA4 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 3) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_f64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_f64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_f64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_f64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 0, step);
                        svmopa_za64_f64_m(4, row_predicate_f64x, predicate_all_f64x, a_f64x, b_f64x);
                    }

                    // Column tile 4 → ZA5 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 4) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_f64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_f64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_f64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_f64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 0, step);
                        svmopa_za64_f64_m(5, row_predicate_f64x, predicate_all_f64x, a_f64x, b_f64x);
                    }

                    // Column tile 5 → ZA6 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 5) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_f64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_f64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_f64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_f64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 0, step);
                        svmopa_za64_f64_m(6, row_predicate_f64x, predicate_all_f64x, a_f64x, b_f64x);
                    }

                    // Column tile 6 → ZA7 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 6) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_f64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_f64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_f64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_f64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 0, step);
                        svmopa_za64_f64_m(7, row_predicate_f64x, predicate_all_f64x, a_f64x, b_f64x);
                    }
                }
            }

            // Extract results and store native f64 outputs.
            svbool_t const predicate_tile_f64x = svwhilelt_b64_u64(0u, tile_dimension);
            // The 7th tile (index 6) may be partial when it's the last column tile
            nk_size_t const last_fast_col_start = (column_tile_index + 6) * tile_dimension;
            nk_size_t const last_fast_cols = (last_fast_col_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - last_fast_col_start);
            svbool_t const last_tile_pred_f64x = svwhilelt_b64_u64(0u, last_fast_cols);
            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f64_t *result_row = result + row_abs * result_stride_elements;

                svfloat64_t za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 1, row);
                svst1_f64(predicate_tile_f64x, result_row + (column_tile_index + 0) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 2, row);
                svst1_f64(predicate_tile_f64x, result_row + (column_tile_index + 1) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 3, row);
                svst1_f64(predicate_tile_f64x, result_row + (column_tile_index + 2) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 4, row);
                svst1_f64(predicate_tile_f64x, result_row + (column_tile_index + 3) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 5, row);
                svst1_f64(predicate_tile_f64x, result_row + (column_tile_index + 4) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 6, row);
                svst1_f64(predicate_tile_f64x, result_row + (column_tile_index + 5) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 7, row);
                svst1_f64(last_tile_pred_f64x, result_row + (column_tile_index + 6) * tile_dimension, za_row_f64x);
            }
        }

        // Remainder: 1 column tile at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                    ? tile_dimension
                                                    : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f64x = svwhilelt_b64_u64(0u, columns_remaining);

            svzero_mask_za(nk_sme_zero_za64_tile_1_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    svbool_t const batch_predicate_f64x = svwhilelt_b64_u64(0u, (uint64_t)batch_size);
                    svbool_t const a_depth_pred_f64x = svwhilelt_b64_u64(depth_offset + depth_batch_start,
                                                                         (uint64_t)depth);
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                        nk_size_t const row_abs = row_tile_start + row_in_tile;
                        svfloat64_t a_row_widened_f64x = svcvt_f64_f32_x(
                            batch_predicate_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_pred_f64x, (nk_u32_t const *)&vectors[row_abs * stride_elements + depth_offset +
                                                                              depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_f64x, a_row_widened_f64x);
                    }

                    // Save A columns from ZA0 to stack buffer
                    for (nk_size_t s = 0; s < batch_size; s++)
                        svst1_f64(predicate_all_f64x, a_buffer[s],
                                  svread_ver_za64_f64_m(svdup_f64(0), row_predicate_f64x, 0, s));

                    // Load B column tile into ZA0 via MOVA, vertical read + FMOPA into ZA1
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = column_tile_start + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_f64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_pred_f64x, (nk_u32_t const *)&vectors[column_abs * stride_elements +
                                                                                  depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_f64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;
                        svfloat64_t a_f64x = svld1_f64(predicate_all_f64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), column_predicate_f64x, 0, step);
                        svmopa_za64_f64_m(1, row_predicate_f64x, column_predicate_f64x, a_f64x, b_f64x);
                    }
                }
            }

            // Store native f64 outputs for the tail column tile.
            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svfloat64_t za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_f64x, 1, row);
                svst1_f64(column_predicate_f64x, result + row_abs * result_stride_elements + column_tile_start,
                          za_row_f64x);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f32_smef64_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                            row_start, row_count);
}

#pragma endregion // Single Precision Floats

/*
 *  f64 GEMM via 3-way Ozaki splitting using FMOPA with ZA64 tiles.
 *  Uses ZA transpose for A-vector construction (expansion=1, no interleaving needed).
 *
 *  Each f64 is split into 3 non-overlapping mantissa-masked slices (19+17+17 bits).
 *  All slices fit in f32 (max 19 significant bits < 24-bit significand).
 *  Cross-products with index sum i+j ≤ 2 are accumulated via 6 FMOPAs into 3 merged
 *  accumulators. Products are exact in f64 (max 19+19 = 38 < 53 mantissa bits).
 *  Accumulation is exact for K ≤ 16,384 (ceil(log2(K)) + 38 + 1 ≤ 53).
 *
 *  Packed GEMM tile allocation (2-column fast path):
 *  - ZA0.D: A-staging (horizontal load, vertical read, shared by both col tiles)
 *  - ZA1.D: col0 acc0 — a₀×b₀              (i+j=0, dominant)
 *  - ZA2.D: col0 acc1 — a₀×b₁ + a₁×b₀      (i+j=1)
 *  - ZA3.D: col0 acc2 — a₀×b₂ + a₁×b₁ + a₂×b₀  (i+j=2, smallest)
 *  - ZA4.D: col1 acc0 — a₀×b₀              (i+j=0)
 *  - ZA5.D: col1 acc1 — a₀×b₁ + a₁×b₀      (i+j=1)
 *  - ZA6.D: col1 acc2 — a₀×b₂ + a₁×b₁ + a₂×b₀  (i+j=2)
 *  - ZA7.D: unused
 *
 *  1-column remainder uses ZA1-3 only.
 *  B is pre-split at pack time into interleaved f32 slices.
 *  A is split in-register per depth step via SVE integer AND.
 *
 *  Symmetric GEMM tile allocation:
 *  - ZA0.D: staging (A rows, then B columns via horizontal load)
 *  - ZA1-3.D: merged Ozaki accumulators (i+j=0,1,2)
 *  Both A and B are split in-register per depth step.
 *
 *  Tile dimensions for SVL=512 (Apple M4):
 *  - ZA64 tile: 8 × 8 f64 elements (512B)
 *  - f64 input vectors: 8 elements (SVL/64)
 *  - FMOPA predicates: b64 (native f64 granularity)
 */
#pragma region Double Precision Floats

/*  Mantissa bit masks for 3-way Ozaki splitting of f64 values.
 *
 *  f64 layout: [63]=sign, [62:52]=exponent (11 bits), [51:0]=mantissa (52 bits).
 *  Significand = implicit 1 + mantissa = 53 significant bits.
 *
 *  Slice 0 (19 significant bits): keep sign + exponent + top 18 mantissa bits.
 *    Zeroes mantissa bits [33:0] (34 bits). Mask = 0xFFFFFFFC00000000.
 *  Slice 1 (17 significant bits): keep sign + exponent + top 16 mantissa bits of residual.
 *    Zeroes mantissa bits [35:0] (36 bits). Mask = 0xFFFFFFF000000000.
 *  Slice 2 = residual of residual (at most 17 significant bits, fits f32).
 *
 *  All slices fit in f32 (24-bit significand). Products: max 19+19 = 38 ≤ 53, exact in f64.
 */
NK_PUBLIC nk_u64_t nk_f64_smef64_ozaki_mask_19_bits_(void) NK_STREAMING_COMPATIBLE_ {
    return 0xFFFFFFFC00000000ULL; // keep top 19 sig bits
}
NK_PUBLIC nk_u64_t nk_f64_smef64_ozaki_mask_17_bits_(void) NK_STREAMING_COMPATIBLE_ {
    return 0xFFFFFFF000000000ULL; // keep top 17 sig bits
}

/*  Split a scalar f64 into 3 non-overlapping Ozaki slices (19+17+17 mantissa bits).
 *  Each slice fits in f32. Outputs stored via pointers. */
NK_PUBLIC void nk_f64_smef64_ozaki_split_f64_(nk_f64_t val, nk_f64_t *slice_0, nk_f64_t *slice_1,
                                              nk_f64_t *slice_2) NK_STREAMING_COMPATIBLE_ {
    nk_fui64_t pun;
    pun.f = val;
    pun.u &= nk_f64_smef64_ozaki_mask_19_bits_();
    *slice_0 = pun.f;
    nk_f64_t residual = val - *slice_0;
    pun.f = residual;
    pun.u &= nk_f64_smef64_ozaki_mask_17_bits_();
    *slice_1 = pun.f;
    *slice_2 = residual - *slice_1;
}

__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_f64_smef64_streaming_(
    nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const tile_dimension = svcntd();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const predicate_all_f64x = svptrue_b64();
    svuint64_t const ozaki_mask_19_u64x = svdup_u64(nk_f64_smef64_ozaki_mask_19_bits_());
    svuint64_t const ozaki_mask_17_u64x = svdup_u64(nk_f64_smef64_ozaki_mask_17_bits_());

    NK_ALIGN64 nk_f64_t a_buffer[8][8]; // save A columns before reusing ZA0 for B

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    // ZA0.D = staging (A then B), ZA1-3.D = merged Ozaki accumulators (i+j=0,1,2)
    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_f64x = svwhilelt_b64_u64(0u, rows_clamped);

        for (nk_size_t column_tile_index = 0; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                    ? tile_dimension
                                                    : (n_vectors - column_tile_start);
            svbool_t const column_predicate_f64x = svwhilelt_b64_u64(0u, columns_remaining);

            // Zero ZA1-3 (3 merged Ozaki accumulators)
            svzero_mask_za(nk_sme_zero_za64_tiles_1_3_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;
                svbool_t const batch_predicate_f64x = svwhilelt_b64_u64(0u, batch_size);

                // Load A rows into ZA0
                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    svld1_hor_za64(0, row_in_tile, batch_predicate_f64x,
                                   vectors + row_abs * stride_elements + depth_batch_start);
                }

                // Save A columns to buffer before reusing ZA0 for B
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f64(predicate_all_f64x, a_buffer[s],
                              svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_f64x, 0, s));

                // Load B columns into ZA0 (reuse)
                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors)
                        svld1_hor_za64(0, column, batch_predicate_f64x,
                                       vectors + column_abs * stride_elements + depth_batch_start);
                }

                // Split both A and B into 3 Ozaki slices, 6 FMOPAs per step
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat64_t a_f64x = svld1_f64(predicate_all_f64x, a_buffer[step]);
                    svuint64_t a_bits_u64x = svreinterpret_u64_f64(a_f64x);
                    svfloat64_t a_slice_0_f64x = svreinterpret_f64_u64(
                        svand_u64_x(predicate_all_f64x, a_bits_u64x, ozaki_mask_19_u64x));
                    svfloat64_t residual_a_f64x = svsub_f64_x(predicate_all_f64x, a_f64x, a_slice_0_f64x);
                    svuint64_t residual_a_bits_u64x = svreinterpret_u64_f64(residual_a_f64x);
                    svfloat64_t a_slice_1_f64x = svreinterpret_f64_u64(
                        svand_u64_x(predicate_all_f64x, residual_a_bits_u64x, ozaki_mask_17_u64x));
                    svfloat64_t a_slice_2_f64x = svsub_f64_x(predicate_all_f64x, residual_a_f64x, a_slice_1_f64x);

                    svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), column_predicate_f64x, 0, step);
                    svuint64_t b_bits_u64x = svreinterpret_u64_f64(b_f64x);
                    svfloat64_t b_slice_0_f64x = svreinterpret_f64_u64(
                        svand_u64_x(predicate_all_f64x, b_bits_u64x, ozaki_mask_19_u64x));
                    svfloat64_t residual_b_f64x = svsub_f64_x(predicate_all_f64x, b_f64x, b_slice_0_f64x);
                    svuint64_t residual_b_bits_u64x = svreinterpret_u64_f64(residual_b_f64x);
                    svfloat64_t b_slice_1_f64x = svreinterpret_f64_u64(
                        svand_u64_x(predicate_all_f64x, residual_b_bits_u64x, ozaki_mask_17_u64x));
                    svfloat64_t b_slice_2_f64x = svsub_f64_x(predicate_all_f64x, residual_b_f64x, b_slice_1_f64x);

                    // 6 FMOPAs reordered to minimize WAW pipeline stalls on 3 tiles.
                    // Same-tile accumulation order preserved (bit-identical output).
                    // Tile schedule: ZA3(0), ZA2(1), ZA1(2), ZA3(4), ZA2(5), ZA3(8).
                    // 9 cycles vs 15 original (3 unavoidable bubbles with only 3 tiles).
                    svmopa_za64_f64_m(3, row_predicate_f64x, column_predicate_f64x, a_slice_0_f64x,
                                      b_slice_2_f64x); // ZA3: i+j=2 (1/3)
                    svmopa_za64_f64_m(2, row_predicate_f64x, column_predicate_f64x, a_slice_0_f64x,
                                      b_slice_1_f64x); // ZA2: i+j=1 (1/2)
                    svmopa_za64_f64_m(1, row_predicate_f64x, column_predicate_f64x, a_slice_0_f64x,
                                      b_slice_0_f64x); // ZA1: i+j=0
                    svmopa_za64_f64_m(3, row_predicate_f64x, column_predicate_f64x, a_slice_1_f64x,
                                      b_slice_1_f64x); // ZA3: i+j=2 (2/3)
                    svmopa_za64_f64_m(2, row_predicate_f64x, column_predicate_f64x, a_slice_1_f64x,
                                      b_slice_0_f64x); // ZA2: i+j=1 (2/2)
                    svmopa_za64_f64_m(3, row_predicate_f64x, column_predicate_f64x, a_slice_2_f64x,
                                      b_slice_0_f64x); // ZA3: i+j=2 (3/3)
                }
            }

            // Sum ZA3 + ZA2 + ZA1 (smallest to largest)
            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svfloat64_t result_f64x = svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 3, row);
                result_f64x = svadd_f64_x(predicate_all_f64x, result_f64x,
                                          svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 2, row));
                result_f64x = svadd_f64_x(predicate_all_f64x, result_f64x,
                                          svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 1, row));
                svst1_f64(column_predicate_f64x, result + row_abs * result_stride_elements + column_tile_start,
                          result_f64x);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f64_smef64_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                            row_start, row_count);
}

NK_PUBLIC nk_size_t nk_dots_packed_size_f64_smef64(nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = svcntsd();
    nk_size_t const depth_tile_size = svcntsw();
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);
    // Single header + interleaved 3-slice data (3× tile_dimension elements per depth step)
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_tile_count * 3 * tile_dimension * depth_tile_size * sizeof(nk_f32_t);
    size += columns * sizeof(nk_f64_t); // per-column squared norms
    return size;
}

NK_PUBLIC void nk_dots_pack_f64_smef64(nk_f64_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed) {

    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f64_t);

    nk_size_t const tile_dimension = svcntsd();
    nk_size_t const depth_tile_size = svcntsw();
    nk_size_t const interleaved_stride = 3 * tile_dimension;
    nk_size_t const interleaved_tile_elements = depth_tile_size * interleaved_stride;

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    // Write single header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)columns;
    header->depth = (nk_u32_t)depth;
    header->svl_bytes = (nk_u32_t)svcntsb();

    nk_f32_t *tiles = (nk_f32_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles (handles partial tile padding)
    nk_size_t const total_elements = total_tiles * interleaved_tile_elements;
    for (nk_size_t i = 0; i < total_elements; i++) tiles[i] = 0.0f;

    // Inline tiling + 3-way mantissa-mask split with interleaved slice layout.
    // Per depth step depth_idx, 3 slices are stored contiguously:
    //   tiles[tile_output + depth_idx * interleaved_stride + slice * tile_dimension + column_idx]
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tile_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
            nk_f32_t *tile_output = tiles +
                                    (column_tile_idx * depth_tile_count + depth_tile_idx) * interleaved_tile_elements;

            nk_size_t const column_start = column_tile_idx * tile_dimension;
            nk_size_t const k_start = depth_tile_idx * depth_tile_size;
            nk_size_t const columns_to_pack = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                         : (columns - column_start);
            nk_size_t const depth_to_pack = (k_start + depth_tile_size <= depth) ? depth_tile_size : (depth - k_start);

            for (nk_size_t column_idx = 0; column_idx < columns_to_pack; column_idx++) {
                for (nk_size_t depth_idx = 0; depth_idx < depth_to_pack; depth_idx++) {
                    nk_f64_t val = b[(column_start + column_idx) * b_stride_elements + k_start + depth_idx];
                    nk_f64_t slice_0, slice_1, slice_2;
                    nk_f64_smef64_ozaki_split_f64_(val, &slice_0, &slice_1, &slice_2);

                    tile_output[depth_idx * interleaved_stride + 0 * tile_dimension + column_idx] = (nk_f32_t)slice_0;
                    tile_output[depth_idx * interleaved_stride + 1 * tile_dimension + column_idx] = (nk_f32_t)slice_1;
                    tile_output[depth_idx * interleaved_stride + 2 * tile_dimension + column_idx] = (nk_f32_t)slice_2;
                }
            }
        }
    }

    // Compute per-column squared norms and store after packed data
    nk_size_t const data_size = total_tiles * interleaved_tile_elements * sizeof(nk_f32_t);
    header->norms_offset = (nk_u32_t)(sizeof(nk_dots_sme_packed_header_t) + data_size);
    nk_f64_t *norms_ptr = (nk_f64_t *)((char *)b_packed + header->norms_offset);
    for (nk_size_t col = 0; col < columns; col++) {
        nk_f64_t const *col_data = (nk_f64_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_f64_(col_data, depth);
    }
}

__arm_locally_streaming __arm_new("za") static void nk_dots_packed_f64_smef64_streaming_(
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    // Read header
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntd();                                        // 8 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw();                                       // 16 (f32 packing granularity)
    nk_size_t const interleaved_stride = 3 * tile_dimension;                          // 24
    nk_size_t const interleaved_tile_elements = depth_tile_size * interleaved_stride; // 384
    nk_size_t const depth_steps_per_batch = tile_dimension;                           // 8 f64 steps per ZA0 load

    // B tile data pointer (f32, interleaved slices)
    nk_f32_t const *b_tiles = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_f64x = svptrue_b64();

    // Mantissa masks for in-register Ozaki splitting (19+17+17 bits)
    svuint64_t const ozaki_mask_19_u64x = svdup_u64(nk_f64_smef64_ozaki_mask_19_bits_());
    svuint64_t const ozaki_mask_17_u64x = svdup_u64(nk_f64_smef64_ozaki_mask_17_bits_());

    // ZA0.D = A staging
    // ZA1-3.D = col0 merged accumulators (i+j=0,1,2)
    // ZA4-6.D = col1 merged accumulators (i+j=0,1,2) [2-col path only]
    for (nk_size_t row_tile_index = 0; row_tile_index < nk_size_divide_round_up_(rows, tile_dimension);
         row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_f64x = svwhilelt_b64_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // 2-column fast path: process 2 column tiles per iteration, A staged once
        for (; column_tile_index + 2 <= column_tile_count; column_tile_index += 2) {
            nk_size_t const column_start_0 = column_tile_index * tile_dimension;
            nk_size_t const column_start_1 = (column_tile_index + 1) * tile_dimension;
            nk_size_t const columns_remaining_0 = (column_start_0 + tile_dimension <= columns)
                                                      ? tile_dimension
                                                      : (columns - column_start_0);
            nk_size_t const columns_remaining_1 = (column_start_1 + tile_dimension <= columns)
                                                      ? tile_dimension
                                                      : (columns - column_start_1);
            svbool_t const column_predicate_0_f64x = svwhilelt_b64_u64(0u, columns_remaining_0);
            svbool_t const column_predicate_1_f64x = svwhilelt_b64_u64(0u, columns_remaining_1);

            // Zero ZA1-6 (3 accumulators × 2 column tiles)
            svzero_mask_za(nk_sme_zero_za64_tiles_1_6_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    // Load A rows into ZA0.D (shared for both column tiles)
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        svbool_t const a_depth_predicate_f64x = svwhilelt_b64_u64(depth_offset + depth_batch_start,
                                                                                  (uint64_t)depth);
                        svld1_hor_za64(0, row_in_tile, a_depth_predicate_f64x,
                                       &a[a_row * a_stride_elements + depth_offset + depth_batch_start]);
                    }

                    // B base offsets for both column tiles
                    nk_size_t const b_batch_offset_0 = (column_tile_index * depth_tile_count + depth_tile_idx) *
                                                           interleaved_tile_elements +
                                                       depth_batch_start * interleaved_stride;
                    nk_size_t const b_batch_offset_1 = ((column_tile_index + 1) * depth_tile_count + depth_tile_idx) *
                                                           interleaved_tile_elements +
                                                       depth_batch_start * interleaved_stride;

                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;

                        // Read A column from ZA0 and split into 3 Ozaki slices
                        svfloat64_t a_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_f64x, 0, step);
                        svuint64_t a_bits_u64x = svreinterpret_u64_f64(a_f64x);
                        svfloat64_t a_slice_0_f64x = svreinterpret_f64_u64(
                            svand_u64_x(predicate_all_f64x, a_bits_u64x, ozaki_mask_19_u64x));
                        svfloat64_t residual_a_f64x = svsub_f64_x(predicate_all_f64x, a_f64x, a_slice_0_f64x);
                        svuint64_t residual_a_bits_u64x = svreinterpret_u64_f64(residual_a_f64x);
                        svfloat64_t a_slice_1_f64x = svreinterpret_f64_u64(
                            svand_u64_x(predicate_all_f64x, residual_a_bits_u64x, ozaki_mask_17_u64x));
                        svfloat64_t a_slice_2_f64x = svsub_f64_x(predicate_all_f64x, residual_a_f64x, a_slice_1_f64x);

                        // Load all 6 B slices upfront (3 per column tile) for pipeline interleaving
                        nk_size_t const b_tile_offset_0 = b_batch_offset_0 + step * interleaved_stride;
                        nk_size_t const b_tile_offset_1 = b_batch_offset_1 + step * interleaved_stride;
                        svfloat64_t b_column_0_slice_0_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(
                                svld1uw_u64(predicate_all_f64x, (nk_u32_t const *)(b_tiles + b_tile_offset_0))));
                        svfloat64_t b_column_0_slice_1_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x, (nk_u32_t const *)(b_tiles + b_tile_offset_0 + tile_dimension))));
                        svfloat64_t b_column_0_slice_2_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x, svreinterpret_f32_u64(svld1uw_u64(
                                                    predicate_all_f64x, (nk_u32_t const *)(b_tiles + b_tile_offset_0 +
                                                                                           2 * tile_dimension))));
                        svfloat64_t b_column_1_slice_0_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(
                                svld1uw_u64(predicate_all_f64x, (nk_u32_t const *)(b_tiles + b_tile_offset_1))));
                        svfloat64_t b_column_1_slice_1_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x, (nk_u32_t const *)(b_tiles + b_tile_offset_1 + tile_dimension))));
                        svfloat64_t b_column_1_slice_2_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x, svreinterpret_f32_u64(svld1uw_u64(
                                                    predicate_all_f64x, (nk_u32_t const *)(b_tiles + b_tile_offset_1 +
                                                                                           2 * tile_dimension))));

                        // 12 FMOPAs interleaved across 6 tiles to eliminate WAW pipeline stalls.
                        // Same-tile accumulation order preserved (bit-identical output).
                        // Tile gaps: ZA3 at 0,6,10 (6,4); ZA6 at 1,7,11 (6,4); ZA2 at 4,8 (4);
                        //            ZA5 at 5,9 (4); ZA1 at 2; ZA4 at 3. All gaps >= 4-cycle latency.
                        svmopa_za64_f64_m(3, row_predicate_f64x, column_predicate_0_f64x, a_slice_0_f64x,
                                          b_column_0_slice_2_f64x); // ZA3: i+j=2 (1/3)
                        svmopa_za64_f64_m(6, row_predicate_f64x, column_predicate_1_f64x, a_slice_0_f64x,
                                          b_column_1_slice_2_f64x); // ZA6: i+j=2 (1/3)
                        svmopa_za64_f64_m(1, row_predicate_f64x, column_predicate_0_f64x, a_slice_0_f64x,
                                          b_column_0_slice_0_f64x); // ZA1: i+j=0
                        svmopa_za64_f64_m(4, row_predicate_f64x, column_predicate_1_f64x, a_slice_0_f64x,
                                          b_column_1_slice_0_f64x); // ZA4: i+j=0
                        svmopa_za64_f64_m(2, row_predicate_f64x, column_predicate_0_f64x, a_slice_0_f64x,
                                          b_column_0_slice_1_f64x); // ZA2: i+j=1 (1/2)
                        svmopa_za64_f64_m(5, row_predicate_f64x, column_predicate_1_f64x, a_slice_0_f64x,
                                          b_column_1_slice_1_f64x); // ZA5: i+j=1 (1/2)
                        svmopa_za64_f64_m(3, row_predicate_f64x, column_predicate_0_f64x, a_slice_1_f64x,
                                          b_column_0_slice_1_f64x); // ZA3: i+j=2 (2/3)
                        svmopa_za64_f64_m(6, row_predicate_f64x, column_predicate_1_f64x, a_slice_1_f64x,
                                          b_column_1_slice_1_f64x); // ZA6: i+j=2 (2/3)
                        svmopa_za64_f64_m(2, row_predicate_f64x, column_predicate_0_f64x, a_slice_1_f64x,
                                          b_column_0_slice_0_f64x); // ZA2: i+j=1 (2/2)
                        svmopa_za64_f64_m(5, row_predicate_f64x, column_predicate_1_f64x, a_slice_1_f64x,
                                          b_column_1_slice_0_f64x); // ZA5: i+j=1 (2/2)
                        svmopa_za64_f64_m(3, row_predicate_f64x, column_predicate_0_f64x, a_slice_2_f64x,
                                          b_column_0_slice_0_f64x); // ZA3: i+j=2 (3/3)
                        svmopa_za64_f64_m(6, row_predicate_f64x, column_predicate_1_f64x, a_slice_2_f64x,
                                          b_column_1_slice_0_f64x); // ZA6: i+j=2 (3/3)
                    }
                }
            }

            // Simple summation for col tile 0: ZA3 + ZA2 + ZA1 (smallest to largest)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_f64_t *c_row = c + (row_start + row) * c_stride_elements + column_start_0;
                svfloat64_t result_f64x = svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 3, row);
                result_f64x = svadd_f64_x(predicate_all_f64x, result_f64x,
                                          svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 2, row));
                result_f64x = svadd_f64_x(predicate_all_f64x, result_f64x,
                                          svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 1, row));
                svst1_f64(column_predicate_0_f64x, c_row, result_f64x);
            }

            // Simple summation for col tile 1: ZA6 + ZA5 + ZA4 (smallest to largest)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_f64_t *c_row = c + (row_start + row) * c_stride_elements + column_start_1;
                svfloat64_t result_f64x = svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 6, row);
                result_f64x = svadd_f64_x(predicate_all_f64x, result_f64x,
                                          svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 5, row));
                result_f64x = svadd_f64_x(predicate_all_f64x, result_f64x,
                                          svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 4, row));
                svst1_f64(column_predicate_1_f64x, c_row, result_f64x);
            }
        }

        // 1-column remainder (when column_tile_count is odd)
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                           : (columns - column_start);
            svbool_t const column_predicate_f64x = svwhilelt_b64_u64(0u, columns_remaining);

            // Zero ZA1-3 (3 merged accumulators)
            svzero_mask_za(nk_sme_zero_za64_tiles_1_3_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    // Load A rows into ZA0.D
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        svbool_t const a_depth_predicate_f64x = svwhilelt_b64_u64(depth_offset + depth_batch_start,
                                                                                  (uint64_t)depth);
                        svld1_hor_za64(0, row_in_tile, a_depth_predicate_f64x,
                                       &a[a_row * a_stride_elements + depth_offset + depth_batch_start]);
                    }

                    nk_size_t const b_batch_offset = (column_tile_index * depth_tile_count + depth_tile_idx) *
                                                         interleaved_tile_elements +
                                                     depth_batch_start * interleaved_stride;

                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;

                        // Read A column from ZA0 and split into 3 Ozaki slices
                        svfloat64_t a_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_f64x, 0, step);
                        svuint64_t a_bits_u64x = svreinterpret_u64_f64(a_f64x);
                        svfloat64_t a_slice_0_f64x = svreinterpret_f64_u64(
                            svand_u64_x(predicate_all_f64x, a_bits_u64x, ozaki_mask_19_u64x));
                        svfloat64_t residual_a_f64x = svsub_f64_x(predicate_all_f64x, a_f64x, a_slice_0_f64x);
                        svuint64_t residual_a_bits_u64x = svreinterpret_u64_f64(residual_a_f64x);
                        svfloat64_t a_slice_1_f64x = svreinterpret_f64_u64(
                            svand_u64_x(predicate_all_f64x, residual_a_bits_u64x, ozaki_mask_17_u64x));
                        svfloat64_t a_slice_2_f64x = svsub_f64_x(predicate_all_f64x, residual_a_f64x, a_slice_1_f64x);

                        // Load 3 B slices (contiguous in interleaved layout)
                        nk_size_t const b_tile_offset = b_batch_offset + step * interleaved_stride;
                        svfloat64_t b_slice_0_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x, svreinterpret_f32_u64(svld1uw_u64(
                                                    predicate_all_f64x, (nk_u32_t const *)(b_tiles + b_tile_offset))));
                        svfloat64_t b_slice_1_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x, (nk_u32_t const *)(b_tiles + b_tile_offset + tile_dimension))));
                        svfloat64_t b_slice_2_f64x = svcvt_f64_f32_x(
                            predicate_all_f64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_f64x, (nk_u32_t const *)(b_tiles + b_tile_offset + 2 * tile_dimension))));

                        // 6 FMOPAs reordered to minimize WAW pipeline stalls on 3 tiles.
                        // Same-tile accumulation order preserved (bit-identical output).
                        // Tile schedule: ZA3(0), ZA2(1), ZA1(2), ZA3(4), ZA2(5), ZA3(8).
                        // 9 cycles vs 15 original (3 unavoidable bubbles with only 3 tiles).
                        svmopa_za64_f64_m(3, row_predicate_f64x, column_predicate_f64x, a_slice_0_f64x,
                                          b_slice_2_f64x); // ZA3: i+j=2 (1/3)
                        svmopa_za64_f64_m(2, row_predicate_f64x, column_predicate_f64x, a_slice_0_f64x,
                                          b_slice_1_f64x); // ZA2: i+j=1 (1/2)
                        svmopa_za64_f64_m(1, row_predicate_f64x, column_predicate_f64x, a_slice_0_f64x,
                                          b_slice_0_f64x); // ZA1: i+j=0
                        svmopa_za64_f64_m(3, row_predicate_f64x, column_predicate_f64x, a_slice_1_f64x,
                                          b_slice_1_f64x); // ZA3: i+j=2 (2/3)
                        svmopa_za64_f64_m(2, row_predicate_f64x, column_predicate_f64x, a_slice_1_f64x,
                                          b_slice_0_f64x); // ZA2: i+j=1 (2/2)
                        svmopa_za64_f64_m(3, row_predicate_f64x, column_predicate_f64x, a_slice_2_f64x,
                                          b_slice_0_f64x); // ZA3: i+j=2 (3/3)
                    }
                }
            }

            // Simple summation: ZA3 + ZA2 + ZA1 (smallest to largest)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_f64_t *c_row = c + (row_start + row) * c_stride_elements + column_start;
                svfloat64_t result_f64x = svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 3, row);
                result_f64x = svadd_f64_x(predicate_all_f64x, result_f64x,
                                          svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 2, row));
                result_f64x = svadd_f64_x(predicate_all_f64x, result_f64x,
                                          svread_hor_za64_f64_m(svdup_f64(0.0), predicate_all_f64x, 1, row));
                svst1_f64(column_predicate_f64x, c_row, result_f64x);
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                         nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_dots_packed_f64_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

#pragma endregion // Double Precision Floats

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
#endif // NK_DOTS_SMEF64_H
