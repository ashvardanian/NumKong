/**
 *  @file include/numkong/dots/smef64.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *  @brief SIMD-accelerated Batched Dot Products for SME F64.
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses ARM SME with @c FEAT_SME_F64F64 for high-precision GEMM.
 *  Requires Apple M4 or equivalent with @c f64 outer product support.
 *
 *  Provides @c f32 and @c f64 GEMM using @c ZA64 tiles:
 *  - @c f32 inputs with @c f64 accumulation: higher precision than @c ZA32
 *  - @c f64 inputs compensated like Dot2, via Ozaki slices on shared exponents
 *
 *  Ozaki splitting for @c f64:
 *  Rows of A and columns of B are scaled by powers of two and cut into 4 slices on 20-bit grids and
 *  a remainder. The 13 grid products with index sums up to 4 add exactly into 5 tiles, the 2 with a
 *  remainder round in a sixth, and TwoSum folds them every 4096 depth steps. B is pre-split at pack
 *  time into @c f64 slices, 5× its size; A is split in-register with integer and rounding ops.
 *
 *  Tile dimensions for SVL=512 (Apple M4):
 *  - @c ZA64 tile: @b [8,8] @c f64 elements, 512B total
 *  - @c f64 vectors: 8 elements per SVE vector
 *  - @c f32 vectors: 16 elements per SVE vector, converted to @c f64
 *
 *  Key instructions:
 *  - @c svmopa_za64_f64_m or @c FMOPA: f64 outer product, 16cy amortized
 *  - @c svcvt_f64_f32_x or @c FCVT: f32 → f64 conversion
 *  - @c svwrite_hor_za64_f64_m or @c MOVA: direct Z → ZA tile write, without a bounce buffer
 */
#ifndef NUMKONG_DOTS_SMEF64_H
#define NUMKONG_DOTS_SMEF64_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_SMEF64

#include "numkong/types.h"
#include "numkong/dots/sme.h" // `nk_dots_sme_packed_header_t`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sme-f64f64"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#endif

/*  f32 → f64 GEMM using FMOPA with ZA64 tiles (FEAT_SME_F64F64).
 *
 *  Tile layout (SVL=512, Apple M4):
 *  - ZA64 output tile: 8 × 8 f64 elements (512 B)
 *  - f32 input vectors: 16 elements (SVL/32), converted to f64 in chunks of 8
 *  - Depth sub-loop: processes 8 f32 values per iteration, widened into 8 f64
 *  - FMOPA predicates: b64 (f64 output granularity)
 *  - f32 load predicates: b32 (f32 input granularity)
 *  - 4-tile path: ZA0-ZA3 process 4 column tiles simultaneously
 *  - Output: native f64 results written directly from ZA64 tiles
 *
 *  Non-widening alternative (FEAT_SME_F32F32, @c svmopa_za32_f32_m): ZA32 tiles are 16×16, 4× the
 *  area of ZA64 8×8, with no f32 ↔ f64 conversion, offering ~3-4× raw throughput. However, ZA32
 *  and ZA64 tiles alias physically, as ZA0.S overlaps ZA0.D+ZA1.D, so a periodic flush to f64
 *  stack accumulators would be needed for precision above f32, erasing most of the speedup. Pure
 *  f32 accumulation without a flush provides only f32 precision, which is already served by the
 *  f16 → f32 GEMM path for reduced-precision workloads. This f64 path exists specifically for
 *  higher-than-f32 accumulation precision; replacing it with f32 FMOPA would be counterproductive.
 *  Apple M4 has `hw.optional.arm.SME_F32F32: 1` but we don't use it here. */
#pragma region F32 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_f32_smef64(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    nk_size_t const tile_dimension = nk_sme_cntd_();  // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = nk_sme_cntw_(); // `f32` depth elements per tile (16 for SVL=512)

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_tile_count * tile_dimension * depth_tile_size * sizeof(nk_f32_t);
    size += columns * sizeof(nk_f64_t); // per-column squared norms

    *bytes = size;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_f32_smef64(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_smef64_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_f32_smef64(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const tile_dimension = nk_sme_cntd_();                  // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = nk_sme_cntw_();                 // `f32` depth elements per tile (16 for SVL=512)
    nk_size_t const tile_elements = tile_dimension * depth_tile_size; // 128
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f32_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;
    nk_size_t const tile_column_begin = nk_size_divide_round_up_(columns_begin, tile_dimension);
    nk_size_t tile_column_end = nk_size_divide_round_up_(columns_end, tile_dimension);
    if (tile_column_end > column_tile_count) tile_column_end = column_tile_count;

    // Store actual dimensions and tile counts in header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    if (columns_begin == 0) {
        for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
            ((nk_u32_t *)header)[word_index] = 0;
        header->column_tile_count = (nk_u32_t)column_tile_count;
        header->depth_tile_count = (nk_u32_t)depth_tile_count;
        header->columns = (nk_u32_t)columns;
        header->depth = (nk_u32_t)depth;
        header->svl_bytes = (nk_u32_t)nk_sme_cntb_(); // streaming vector length in bytes
        header->capability = nk_cap_smef64_k;
    }

    nk_f32_t *tiles = (nk_f32_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles (handles partial tile padding)
    for (nk_size_t i = tile_column_begin * depth_tile_count * tile_elements;
         i < tile_column_end * depth_tile_count * tile_elements; i++)
        tiles[i] = 0.0f;

    // Pack data into tiles with depth-major layout within each tile:
    // dst_idx = depth_idx * tile_dimension + column_idx
    // This allows loading one B vector per depth step: svld1(b_tile + k * tile_dimension)
    for (nk_size_t column_tile_idx = tile_column_begin; column_tile_idx < tile_column_end; column_tile_idx++) {
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
    nk_size_t const norms_offset = sizeof(nk_dots_sme_packed_header_t) + data_size;
    if (columns_begin == 0) header->norms_offset = (nk_u32_t)norms_offset;
    nk_f64_t *norms_ptr = (nk_f64_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = columns_begin; col < columns_end; col++) {
        nk_f32_t const *col_data = (nk_f32_t const *)((char const *)b + col * b_stride);
        norms_ptr[col] = nk_dots_reduce_sumsq_f32_(col_data, depth, nk_cap_smef64_k);
    }
    return nk_success_k;
}

__arm_new("za") static void nk_dots_packed_f32_smef64_streaming_( //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntd();  // 8 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 for 512-bit SVL
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const depth_steps_per_batch = tile_dimension; // 8 depth steps per ZA0.D load

    nk_f32_t const *b_tiles = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_all_b64x = svptrue_b64();

    // ZA0.D = staging, ZA1-7.D = accumulation (7-tile fast path)
    for (nk_size_t row_tile_index = 0; row_tile_index < nk_size_divide_round_up_(rows, tile_dimension);
         row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b64x = svwhilelt_b64_u64(0u, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 7 column tiles using ZA1-ZA7 (ZA0.D = staging)
        for (; column_tile_index + 7 <= column_tile_count; column_tile_index += 7) {
            svzero_mask_za(nk_sme_zero_za64_tiles_1_7_k);

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

                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);

                    // Load A rows into ZA0.D: extending load f32→u64 + convert to f64
                    svbool_t const batch_predicate_b64x = svwhilelt_b64_u64(0u, batch_size);
                    svbool_t const a_depth_predicate_b64x = svwhilelt_b64_u64(depth_offset + depth_batch_start, depth);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        // Extending load: svld1uw_u64 loads f32 bits into lower 32 of each u64 lane
                        svfloat64_t a_row_widened_f64x = svcvt_f64_f32_x(
                            batch_predicate_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_predicate_b64x,
                                (nk_u32_t const *)&a[a_row * a_stride_elements + depth_offset + depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_b64x, a_row_widened_f64x);
                    }

                    // Vertical read + MOPA for each depth step in batch
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;

                        svfloat64_t a_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 0, step);

                        nk_size_t const b_k = depth_batch_start + step;

                        // Extending load f32→u64 + convert to f64: svld1uw_u64 replaces svld1_f32 + svunpklo_u64
                        svfloat64_t b_column_tile_1_f64x = svcvt_f64_f32_x(
                            predicate_all_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_b64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 0) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_2_f64x = svcvt_f64_f32_x(
                            predicate_all_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_b64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 1) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_3_f64x = svcvt_f64_f32_x(
                            predicate_all_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_b64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 2) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_4_f64x = svcvt_f64_f32_x(
                            predicate_all_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_b64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 3) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_5_f64x = svcvt_f64_f32_x(
                            predicate_all_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_b64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 4) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_6_f64x = svcvt_f64_f32_x(
                            predicate_all_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_b64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 5) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_7_f64x = svcvt_f64_f32_x(
                            predicate_all_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                predicate_all_b64x,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 6) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));

                        svmopa_za64_f64_m(1, row_predicate_b64x, predicate_all_b64x, a_f64x, b_column_tile_1_f64x);
                        svmopa_za64_f64_m(2, row_predicate_b64x, predicate_all_b64x, a_f64x, b_column_tile_2_f64x);
                        svmopa_za64_f64_m(3, row_predicate_b64x, predicate_all_b64x, a_f64x, b_column_tile_3_f64x);
                        svmopa_za64_f64_m(4, row_predicate_b64x, predicate_all_b64x, a_f64x, b_column_tile_4_f64x);
                        svmopa_za64_f64_m(5, row_predicate_b64x, predicate_all_b64x, a_f64x, b_column_tile_5_f64x);
                        svmopa_za64_f64_m(6, row_predicate_b64x, predicate_all_b64x, a_f64x, b_column_tile_6_f64x);
                        svmopa_za64_f64_m(7, row_predicate_b64x, predicate_all_b64x, a_f64x, b_column_tile_7_f64x);
                    }
                }
            }

            // Extract from ZA1-7 and store native f64 outputs.
            svbool_t const predicate_tile_b64x = svwhilelt_b64_u64(0u, tile_dimension);
            // The 7th tile (index 6) may be partial when it's the last column tile
            nk_size_t const last_fast_col_start = (column_tile_index + 6) * tile_dimension;
            nk_size_t const last_fast_columns = (last_fast_col_start + tile_dimension <= columns)
                                                    ? tile_dimension
                                                    : (columns - last_fast_col_start);
            svbool_t const last_tile_pred_b64x = svwhilelt_b64_u64(0u, last_fast_columns);
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f64_t *c_row = c + (row_start + row_idx) * c_stride_elements;

                svfloat64_t za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 1, row_idx);
                svst1_f64(predicate_tile_b64x, c_row + (column_tile_index + 0) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 2, row_idx);
                svst1_f64(predicate_tile_b64x, c_row + (column_tile_index + 1) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 3, row_idx);
                svst1_f64(predicate_tile_b64x, c_row + (column_tile_index + 2) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 4, row_idx);
                svst1_f64(predicate_tile_b64x, c_row + (column_tile_index + 3) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 5, row_idx);
                svst1_f64(predicate_tile_b64x, c_row + (column_tile_index + 4) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 6, row_idx);
                svst1_f64(predicate_tile_b64x, c_row + (column_tile_index + 5) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 7, row_idx);
                svst1_f64(last_tile_pred_b64x, c_row + (column_tile_index + 6) * tile_dimension, za_row_f64x);
            }
        }

        // Remainder: 1 column tile at a time using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                           : (columns - column_start);
            svbool_t const column_predicate_b64x = svwhilelt_b64_u64(0u, columns_remaining);

            svzero_mask_za(nk_sme_zero_za64_tile_1_k);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);

                    svbool_t const batch_predicate_b64x = svwhilelt_b64_u64(0u, batch_size);
                    svbool_t const a_depth_pred_b64x = svwhilelt_b64_u64(depth_offset + depth_batch_start, depth);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        svfloat64_t a_row_widened_f64x = svcvt_f64_f32_x(
                            batch_predicate_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_pred_b64x,
                                (nk_u32_t const *)&a[a_row * a_stride_elements + depth_offset + depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_b64x, a_row_widened_f64x);
                    }

                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;

                        svfloat64_t a_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 0, step);

                        nk_size_t const b_k = depth_batch_start + step;
                        nk_f32_t const *b_tile = b_tiles + (column_tile_index * depth_tile_count + depth_tile_idx) *
                                                               tile_elements;
                        // Extending load f32→u64 + convert to f64
                        svfloat64_t b_f64x = svcvt_f64_f32_x(
                            predicate_all_b64x,
                            svreinterpret_f32_u64(
                                svld1uw_u64(predicate_all_b64x, (nk_u32_t const *)(b_tile + b_k * tile_dimension))));

                        svmopa_za64_f64_m(1, row_predicate_b64x, column_predicate_b64x, a_f64x, b_f64x);
                    }
                }
            }

            // Store native f64 outputs for the tail column tile.
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                svfloat64_t za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 1, row_idx);
                nk_f64_t *c_row = c + (row_start + row_idx) * c_stride_elements + column_start;
                svst1_f64(column_predicate_b64x, c_row, za_row_f64x);
            }
        }
    }
}

NUMKONG_API nk_status_t nk_dots_packed_f32_smef64( //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_smef64_k) return nk_pack_mismatch_k;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_sme_start_streaming_();
    nk_dots_packed_f32_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

/** f32 × f32 → f32 symmetric kernel using MOPA self-GEMM with f64 accumulation. Time-shares ZA0 for
 *  both A and B transposition: loads A horizontally, pre-reads A columns into Z registers, then
 *  reloads ZA0 with widened B data per column tile. Eliminates all scalar B-packing loops. */
__arm_new("za") static void nk_dots_symmetric_f32_smef64_streaming_( //
    nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntd();              // 8 for SVL=512
    nk_size_t const depth_tile_size = svcntw();             // 16 for SVL=512
    nk_size_t const depth_steps_per_batch = tile_dimension; // 8

    svbool_t const predicate_all_b64x = svptrue_b64();

    nk_align_(64) nk_f64_t a_buffer[8][8];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(vectors_count, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < vectors_count;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= vectors_count)
                                          ? rows_clamped
                                          : (vectors_count - row_tile_start);
        svbool_t const row_predicate_b64x = svwhilelt_b64_u64(0u, rows_actual);

        // Upper triangle: start from this row tile's column
        nk_size_t column_tile_index = row_tile_start / tile_dimension;

        // Fast path: 7 column tiles at a time
        for (; column_tile_index + 7 <= column_tile_count; column_tile_index += 7) {
            svzero_mask_za(nk_sme_zero_za64_tiles_1_7_k);

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
                    svbool_t const batch_predicate_b64x = svwhilelt_b64_u64(0u, batch_size);
                    svbool_t const a_depth_predicate_b64x = svwhilelt_b64_u64(depth_offset + depth_batch_start, depth);
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                        nk_size_t const row_abs = row_tile_start + row_in_tile;
                        svfloat64_t a_row_widened_f64x = svcvt_f64_f32_x(
                            batch_predicate_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_predicate_b64x, (nk_u32_t const *)&vectors[row_abs * stride_elements +
                                                                                   depth_offset + depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_b64x, a_row_widened_f64x);
                    }

                    // Save A columns from ZA0 to stack buffer
                    for (nk_size_t s = 0; s < batch_size; s++)
                        svst1_f64(predicate_all_b64x, a_buffer[s],
                                  svread_ver_za64_f64_m(svdup_f64(0), row_predicate_b64x, 0, s));

                    // Column tile 0 → ZA1 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                        if (column_abs < vectors_count) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_b64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_b64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_b64x, 0, step);
                        svmopa_za64_f64_m(1, row_predicate_b64x, predicate_all_b64x, a_f64x, b_f64x);
                    }

                    // Column tile 1 → ZA2 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                        if (column_abs < vectors_count) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_b64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_b64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_b64x, 0, step);
                        svmopa_za64_f64_m(2, row_predicate_b64x, predicate_all_b64x, a_f64x, b_f64x);
                    }

                    // Column tile 2 → ZA3 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                        if (column_abs < vectors_count) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_b64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_b64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_b64x, 0, step);
                        svmopa_za64_f64_m(3, row_predicate_b64x, predicate_all_b64x, a_f64x, b_f64x);
                    }

                    // Column tile 3 → ZA4 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 3) * tile_dimension + column;
                        if (column_abs < vectors_count) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_b64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_b64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_b64x, 0, step);
                        svmopa_za64_f64_m(4, row_predicate_b64x, predicate_all_b64x, a_f64x, b_f64x);
                    }

                    // Column tile 4 → ZA5 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 4) * tile_dimension + column;
                        if (column_abs < vectors_count) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_b64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_b64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_b64x, 0, step);
                        svmopa_za64_f64_m(5, row_predicate_b64x, predicate_all_b64x, a_f64x, b_f64x);
                    }

                    // Column tile 5 → ZA6 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 5) * tile_dimension + column;
                        if (column_abs < vectors_count) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_b64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_b64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_b64x, 0, step);
                        svmopa_za64_f64_m(6, row_predicate_b64x, predicate_all_b64x, a_f64x, b_f64x);
                    }

                    // Column tile 6 → ZA7 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 6) * tile_dimension + column;
                        if (column_abs < vectors_count) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_b64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64x,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_f64x = svld1_f64(predicate_all_b64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), predicate_all_b64x, 0, step);
                        svmopa_za64_f64_m(7, row_predicate_b64x, predicate_all_b64x, a_f64x, b_f64x);
                    }
                }
            }

            nk_size_t const first_column_start = (column_tile_index + 0) * tile_dimension;
            nk_size_t const second_column_start = (column_tile_index + 1) * tile_dimension;
            svbool_t const first_bound_b64x = svwhilelt_b64_u64(first_column_start, vectors_count);
            svbool_t const second_bound_b64x = svwhilelt_b64_u64(second_column_start, vectors_count);
            svbool_t const third_bound_b64x = svwhilelt_b64_u64((column_tile_index + 2) * tile_dimension,
                                                                vectors_count);
            svbool_t const fourth_bound_b64x = svwhilelt_b64_u64((column_tile_index + 3) * tile_dimension,
                                                                 vectors_count);
            svbool_t const fifth_bound_b64x = svwhilelt_b64_u64((column_tile_index + 4) * tile_dimension,
                                                                vectors_count);
            svbool_t const sixth_bound_b64x = svwhilelt_b64_u64((column_tile_index + 5) * tile_dimension,
                                                                vectors_count);
            svbool_t const seventh_bound_b64x = svwhilelt_b64_u64((column_tile_index + 6) * tile_dimension,
                                                                  vectors_count);
            int const first_crosses_diagonal = first_column_start < row_tile_start + tile_dimension;
            int const second_crosses_diagonal = second_column_start < row_tile_start + tile_dimension;
            // Extract results and store native f64 outputs.
            for (nk_size_t row = 0; row < rows_actual; row++) {
                svbool_t const first_b64x = first_crosses_diagonal
                                                ? nk_sme_diagonal_cut_b64x_(first_bound_b64x, first_column_start,
                                                                            row_tile_start + row)
                                                : first_bound_b64x;
                svbool_t const second_b64x = second_crosses_diagonal
                                                 ? nk_sme_diagonal_cut_b64x_(second_bound_b64x, second_column_start,
                                                                             row_tile_start + row)
                                                 : second_bound_b64x;
                nk_size_t const row_abs = row_tile_start + row;
                nk_f64_t *result_row = result + row_abs * result_stride_elements;

                svfloat64_t za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 1, row);
                svst1_f64(first_b64x, result_row + (column_tile_index + 0) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 2, row);
                svst1_f64(second_b64x, result_row + (column_tile_index + 1) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 3, row);
                svst1_f64(third_bound_b64x, result_row + (column_tile_index + 2) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 4, row);
                svst1_f64(fourth_bound_b64x, result_row + (column_tile_index + 3) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 5, row);
                svst1_f64(fifth_bound_b64x, result_row + (column_tile_index + 4) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 6, row);
                svst1_f64(sixth_bound_b64x, result_row + (column_tile_index + 5) * tile_dimension, za_row_f64x);

                za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 7, row);
                svst1_f64(seventh_bound_b64x, result_row + (column_tile_index + 6) * tile_dimension, za_row_f64x);
            }
        }

        // Remainder: 1 column tile at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_tile_start + tile_dimension <= vectors_count)
                                                    ? tile_dimension
                                                    : (vectors_count - column_tile_start);
            svbool_t const column_predicate_b64x = svwhilelt_b64_u64(0u, columns_remaining);

            svzero_mask_za(nk_sme_zero_za64_tile_1_k);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    svbool_t const batch_predicate_b64x = svwhilelt_b64_u64(0u, batch_size);
                    svbool_t const a_depth_pred_b64x = svwhilelt_b64_u64(depth_offset + depth_batch_start, depth);
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                        nk_size_t const row_abs = row_tile_start + row_in_tile;
                        svfloat64_t a_row_widened_f64x = svcvt_f64_f32_x(
                            batch_predicate_b64x,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_pred_b64x, (nk_u32_t const *)&vectors[row_abs * stride_elements + depth_offset +
                                                                              depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_b64x, a_row_widened_f64x);
                    }

                    // Save A columns from ZA0 to stack buffer
                    for (nk_size_t s = 0; s < batch_size; s++)
                        svst1_f64(predicate_all_b64x, a_buffer[s],
                                  svread_ver_za64_f64_m(svdup_f64(0), row_predicate_b64x, 0, s));

                    // Load B column tile into ZA0 via MOVA, vertical read + FMOPA into ZA1
                    svzero_mask_za(nk_sme_zero_za64_tile_0_k);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = column_tile_start + column;
                        if (column_abs < vectors_count) {
                            svfloat64_t widened_f64x = svcvt_f64_f32_x(
                                batch_predicate_b64x,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_pred_b64x, (nk_u32_t const *)&vectors[column_abs * stride_elements +
                                                                                  depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64x, widened_f64x);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;
                        svfloat64_t a_f64x = svld1_f64(predicate_all_b64x, a_buffer[step]);
                        svfloat64_t b_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), column_predicate_b64x, 0, step);
                        svmopa_za64_f64_m(1, row_predicate_b64x, column_predicate_b64x, a_f64x, b_f64x);
                    }
                }
            }

            int const crosses_diagonal = column_tile_start < row_tile_start + tile_dimension;
            // Store native f64 outputs for the tail column tile.
            for (nk_size_t row = 0; row < rows_actual; row++) {
                svbool_t const store_b64x = crosses_diagonal
                                                ? nk_sme_diagonal_cut_b64x_(column_predicate_b64x, column_tile_start,
                                                                            row_tile_start + row)
                                                : column_predicate_b64x;
                nk_size_t const row_abs = row_tile_start + row;
                svfloat64_t za_row_f64x = svread_hor_za64_f64_m(svdup_f64(0), predicate_all_b64x, 1, row);
                svst1_f64(store_b64x, result + row_abs * result_stride_elements + column_tile_start, za_row_f64x);
            }
        }
    }
}

NUMKONG_API nk_status_t nk_dots_symmetric_f32_smef64( //
    nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f64_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));

    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_f32_smef64_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                            result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion F32 Floats

/*  f64 GEMM via Ozaki splitting on shared exponents, using FMOPA with ZA64 tiles.
 *
 *  Each row of A and column of B is scaled by a power of two into (-1, 1) and cut into 5 slices.
 *  Four sit on the grids 2⁻²⁰, 2⁻⁴⁰, 2⁻⁶⁰ and 2⁻⁸⁰, counted in units of their grid, all but the
 *  first at most 2¹⁹ units, and the fifth is a remainder of at most 2⁻⁸¹. Products of grid slices
 *  whose indices sum to c are integers in units of 2⁻²⁰⁽ᶜ⁺²⁾, so the 13 with c ≤ 4 add up exactly
 *  for 4096 depth steps, at most 1.5 × 2⁴⁰ units per step against the 2⁵³ an f64 holds. The 2
 *  products of a remainder with a first slice round, which only happens far below the largest
 *  magnitudes of their row and column.
 *
 *  Every 4096 steps and at the end, TwoSum folds the tiles into a running sum, so results are
 *  compensated like Dot2, missing only the products with c > 4, each below 2⁻¹⁰⁰ of the largest
 *  magnitudes in its row and column.
 *
 *  Streaming mode issues FRINTN, conversions and integer ops every cycle, but f64 arithmetic only
 *  every 4 cycles, so splitting takes 3 multiplies and does the rest on exponents and integers.
 *
 *  Tile allocation:
 *  - ZA0.D: staging, rows loaded horizontally and read back vertically as depth steps
 *  - ZA1-5.D: grid products with index sums 0..4
 *  - ZA7.D: products with a remainder
 *
 *  Between flushes the running sum waits on the stack and its compensation in the output tile.
 *
 *  Packed B keeps the slices in f64, loading without conversion and keeping remainders exact. It
 *  takes 40 bytes per element, 5× the input with depth padded to the tile, plus an exponent and a
 *  norm per column. */
#pragma region F64 Floats

/** Depth steps between flushes, as 4096 × 1.5 × 2⁴⁰ units fit in the 2⁵³ an f64 holds exactly. */
enum { nk_dots_f64_smef64_flush_steps_ = 4096 };

/** SVE Dot2 accumulator: sum += a × b with error compensation. Uses TwoProd, whose svnmls gives
 *  a × b - product in one rounding, and TwoSum error-free transformations. Always inlined, as an
 *  outlined call passes the accumulators through the stack, where streaming stores are slow. */
NUMKONG_INLINE void nk_dot2_f64_sve_accumulate_(svbool_t predicate_b64x, svfloat64_t *sum, svfloat64_t *comp,
                                                svfloat64_t a_f64x, svfloat64_t b_f64x) NUMKONG_STREAMING_ {
    svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_f64x, b_f64x);
    svfloat64_t product_error_f64x = svnmls_f64_x(predicate_b64x, product_f64x, a_f64x, b_f64x);
    svfloat64_t running_sum_f64x = svadd_f64_m(predicate_b64x, *sum, product_f64x);
    svfloat64_t recovered_addend_f64x = svsub_f64_x(predicate_b64x, running_sum_f64x, *sum);
    svfloat64_t sum_error_f64x = svadd_f64_x(
        predicate_b64x,
        svsub_f64_x(predicate_b64x, *sum, svsub_f64_x(predicate_b64x, running_sum_f64x, recovered_addend_f64x)),
        svsub_f64_x(predicate_b64x, product_f64x, recovered_addend_f64x));
    *sum = running_sum_f64x;
    *comp = svadd_f64_m(predicate_b64x, *comp, svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
}

/** Folds the lanes of a Dot2 @p sum and @p compensation with TwoSum into @p high and @p low, whose
 *  total keeps the compensation, unlike a plain horizontal add. */
NUMKONG_INLINE void nk_dot2_f64_sve_reduce_(svfloat64_t sum_f64x, svfloat64_t compensation_f64x, nk_f64_t *high,
                                            nk_f64_t *low) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b64x = svptrue_b64();
    svfloat64_t total_f64x = svadd_f64_x(predicate_all_b64x, sum_f64x, compensation_f64x);
    svfloat64_t bent_f64x = svsub_f64_x(predicate_all_b64x, total_f64x, sum_f64x);
    svfloat64_t error_f64x = svadd_f64_x(
        predicate_all_b64x,
        svsub_f64_x(predicate_all_b64x, sum_f64x, svsub_f64_x(predicate_all_b64x, total_f64x, bent_f64x)),
        svsub_f64_x(predicate_all_b64x, compensation_f64x, bent_f64x));
    // Streaming vector lengths are powers of two, and out-of-range table indices read as zeros
    for (nk_size_t half = svcntd() / 2; half != 0; half /= 2) {
        svuint64_t const upper_u64x = svindex_u64(half, 1);
        svfloat64_t const upper_total_f64x = svtbl_f64(total_f64x, upper_u64x);
        svfloat64_t const folded_f64x = svadd_f64_x(predicate_all_b64x, total_f64x, upper_total_f64x);
        svfloat64_t const folded_bent_f64x = svsub_f64_x(predicate_all_b64x, folded_f64x, total_f64x);
        svfloat64_t const folded_error_f64x = svadd_f64_x(
            predicate_all_b64x,
            svsub_f64_x(predicate_all_b64x, total_f64x, svsub_f64_x(predicate_all_b64x, folded_f64x, folded_bent_f64x)),
            svsub_f64_x(predicate_all_b64x, upper_total_f64x, folded_bent_f64x));
        error_f64x = svadd_f64_x(predicate_all_b64x,
                                 svadd_f64_x(predicate_all_b64x, error_f64x, svtbl_f64(error_f64x, upper_u64x)),
                                 folded_error_f64x);
        total_f64x = folded_f64x;
    }
    svbool_t const predicate_first_b64x = svwhilelt_b64_u64(0u, 1u);
    *high = svlastb_f64(predicate_first_b64x, total_f64x);
    *low = svlastb_f64(predicate_first_b64x, error_f64x);
}

/** Returns the smallest e ≥ -1022 with magnitudes below 2ᵉ, from the bits of the largest magnitude.
 *  Subnormals are multiples of 2⁻¹⁰⁷⁴, so scaling them by 2¹⁰²² keeps them on the 2⁻⁸⁰ grid. */
NUMKONG_INLINE nk_i64_t nk_f64_smef64_exponent_above_(nk_u64_t magnitude_bits) NUMKONG_STREAMABLE_ {
    nk_i64_t const biased_exponent = (nk_i64_t)(magnitude_bits >> 52);
    return (biased_exponent ? biased_exponent : 1) - 1022;
}

/** Returns @p value × 2ⁿ for n = @p exponent, exact whenever the result is normal, through two
 *  powers of two that are normal themselves. */
NUMKONG_INLINE nk_f64_t nk_f64_smef64_scaled_(nk_f64_t value, nk_i64_t exponent) {
    nk_i64_t const half = exponent / 2;
    nk_fui64_t first, second;
    first.u = (nk_u64_t)(1023 + half) << 52, second.u = (nk_u64_t)(1023 + exponent - half) << 52;
    return value * first.f * second.f;
}

/** Cuts @p value from (-1, 1) into 5 slices @p stride apart: 4 on 20-bit grids in units of their
 *  grid, then the remainder. */
NUMKONG_INLINE void nk_f64_smef64_ozaki_split_f64_(nk_f64_t value, nk_f64_t *slices, nk_size_t stride) {
    nk_f64_t sigma = 0x1.8p32, unit = 0x1p20; // adding 1.5 × 2³² rounds to a multiple of 2⁻²⁰
    for (nk_size_t slice = 0; slice != 4; ++slice, sigma *= 0x1p-20, unit *= 0x1p20) {
        nk_f64_t const rounded = (value + sigma) - sigma;
        slices[slice * stride] = rounded * unit, value -= rounded;
    }
    slices[4 * stride] = value;
}

/** Cuts every lane of @p values × @p multipliers, landing in (-1, 1), into the slices of
 *  @c nk_f64_smef64_ozaki_split_f64_, though halfway cases of the first two round up. */
NUMKONG_INLINE void nk_f64_smef64_ozaki_split_f64x_( //
    svfloat64_t values_f64x, svfloat64_t multipliers_f64x, svfloat64_t *slice_0_f64x, svfloat64_t *slice_1_f64x,
    svfloat64_t *slice_2_f64x, svfloat64_t *slice_3_f64x, svfloat64_t *slice_4_f64x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b64x = svptrue_b64();
    svfloat64_t const scaled_f64x = svmul_f64_x(predicate_all_b64x, values_f64x, multipliers_f64x);
    // Raising the exponent field scales by 2⁶⁰ or 2⁸⁰, and turns zeros, subnormals and non-finite
    // values, which it breaks, into magnitudes that round to zero
    svuint64_t const scaled_u64x = svreinterpret_u64_f64(scaled_f64x);
    svfloat64_t const units_60_f64x = svrintn_f64_x(
        predicate_all_b64x, svreinterpret_f64_u64(svadd_n_u64_x(predicate_all_b64x, scaled_u64x, 60ull << 52)));
    svfloat64_t const units_80_f64x = svrintn_f64_x(
        predicate_all_b64x, svreinterpret_f64_u64(svadd_n_u64_x(predicate_all_b64x, scaled_u64x, 80ull << 52)));
    svint64_t const units_60_i64x = svcvt_s64_f64_x(predicate_all_b64x, units_60_f64x);
    svint64_t const units_40_i64x = svrshr_n_s64_x(predicate_all_b64x, units_60_i64x, 20);
    *slice_0_f64x = svcvt_f64_s64_x(predicate_all_b64x, svrshr_n_s64_x(predicate_all_b64x, units_40_i64x, 20));
    // Sign-extending the low 20 bits leaves exactly what the rounding shift moved above them
    *slice_1_f64x = svcvt_f64_s64_x(
        predicate_all_b64x,
        svasr_n_s64_x(predicate_all_b64x, svlsl_n_s64_x(predicate_all_b64x, units_40_i64x, 44), 44));
    *slice_2_f64x = svcvt_f64_s64_x(
        predicate_all_b64x,
        svasr_n_s64_x(predicate_all_b64x, svlsl_n_s64_x(predicate_all_b64x, units_60_i64x, 44), 44));
    *slice_3_f64x = svmls_n_f64_x(predicate_all_b64x, units_80_f64x, units_60_f64x, 0x1p20);
    *slice_4_f64x = svmls_n_f64_x(predicate_all_b64x, scaled_f64x, units_80_f64x, 0x1p-80);
}

/** Returns @c nk_f64_smef64_exponent_above_ of each of the first @p count vectors, which lie
 *  @p stride_elements apart, in the lane of its index, and 0 in the lanes past them. */
NUMKONG_INLINE svint64_t nk_f64_smef64_exponents_(nk_f64_t const *vectors, nk_size_t stride_elements, nk_size_t count,
                                                  nk_size_t depth) NUMKONG_STREAMING_ {
    svint64_t exponents_i64x = svdup_s64(0);
    for (nk_size_t index = svcntd(); index-- != 0;) {
        nk_u64_t magnitude_bits = 0;
        if (index < count) {
            svuint64_t magnitudes_u64x = svdup_u64(0);
            for (nk_size_t depth_index = 0; depth_index < depth; depth_index += svcntd()) {
                svbool_t const predicate_b64x = svwhilelt_b64_u64(depth_index, depth);
                svuint64_t const bits_u64x = svreinterpret_u64_f64(
                    svld1_f64(predicate_b64x, vectors + index * stride_elements + depth_index));
                magnitudes_u64x = svmax_u64_m(predicate_b64x, magnitudes_u64x,
                                              svand_n_u64_x(predicate_b64x, bits_u64x, 0x7FFFFFFFFFFFFFFFull));
            }
            magnitude_bits = svmaxv_u64(svptrue_b64(), magnitudes_u64x);
        }
        exponents_i64x = svinsr_n_s64(exponents_i64x, nk_f64_smef64_exponent_above_(magnitude_bits));
    }
    return exponents_i64x;
}

/** Transposes the batch of @p count vectors starting at depth @p batch_start through ZA0.D into
 *  @p steps, one depth step per vector, zeros past the vectors and the depth. */
NUMKONG_INLINE void nk_f64_smef64_stage_(nk_f64_t const *vectors, nk_size_t stride_elements, nk_size_t count,
                                         nk_size_t batch_start, nk_size_t depth, nk_f64_t *steps) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const tile_dimension = svcntd();
    svbool_t const batch_predicate_b64x = svwhilelt_b64_u64(batch_start, depth);
    if (count != tile_dimension) svzero_mask_za(nk_sme_zero_za64_tile_0_k);
    for (nk_size_t index = 0; index < count; index++)
        svld1_hor_za64(0, index, batch_predicate_b64x, vectors + index * stride_elements + batch_start);
    for (nk_size_t step = 0; step < tile_dimension; step++)
        svst1_f64(svptrue_b64(), steps + step * tile_dimension,
                  svread_ver_za64_f64_m(svundef_f64(), svptrue_b64(), 0, step));
}

/** Adds the 15 slice products: the 13 of grid slices into ZA1-5.D by index sum, the 2 with a
 *  remainder into ZA7.D, spacing out the FMOPAs into one tile. */
NUMKONG_INLINE void nk_dots_f64_smef64_accumulate_( //
    svfloat64_t a_slice_0_f64x, svfloat64_t a_slice_1_f64x, svfloat64_t a_slice_2_f64x, svfloat64_t a_slice_3_f64x,
    svfloat64_t a_slice_4_f64x, svfloat64_t b_slice_0_f64x, svfloat64_t b_slice_1_f64x, svfloat64_t b_slice_2_f64x,
    svfloat64_t b_slice_3_f64x, svfloat64_t b_slice_4_f64x) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b64x = svptrue_b64();
    svmopa_za64_f64_m(4, predicate_all_b64x, predicate_all_b64x, a_slice_0_f64x, b_slice_3_f64x);
    svmopa_za64_f64_m(5, predicate_all_b64x, predicate_all_b64x, a_slice_1_f64x, b_slice_3_f64x);
    svmopa_za64_f64_m(3, predicate_all_b64x, predicate_all_b64x, a_slice_0_f64x, b_slice_2_f64x);
    svmopa_za64_f64_m(2, predicate_all_b64x, predicate_all_b64x, a_slice_0_f64x, b_slice_1_f64x);
    svmopa_za64_f64_m(4, predicate_all_b64x, predicate_all_b64x, a_slice_1_f64x, b_slice_2_f64x);
    svmopa_za64_f64_m(5, predicate_all_b64x, predicate_all_b64x, a_slice_2_f64x, b_slice_2_f64x);
    svmopa_za64_f64_m(3, predicate_all_b64x, predicate_all_b64x, a_slice_1_f64x, b_slice_1_f64x);
    svmopa_za64_f64_m(7, predicate_all_b64x, predicate_all_b64x, a_slice_0_f64x, b_slice_4_f64x);
    svmopa_za64_f64_m(4, predicate_all_b64x, predicate_all_b64x, a_slice_2_f64x, b_slice_1_f64x);
    svmopa_za64_f64_m(5, predicate_all_b64x, predicate_all_b64x, a_slice_3_f64x, b_slice_1_f64x);
    svmopa_za64_f64_m(3, predicate_all_b64x, predicate_all_b64x, a_slice_2_f64x, b_slice_0_f64x);
    svmopa_za64_f64_m(2, predicate_all_b64x, predicate_all_b64x, a_slice_1_f64x, b_slice_0_f64x);
    svmopa_za64_f64_m(4, predicate_all_b64x, predicate_all_b64x, a_slice_3_f64x, b_slice_0_f64x);
    svmopa_za64_f64_m(7, predicate_all_b64x, predicate_all_b64x, a_slice_4_f64x, b_slice_0_f64x);
    svmopa_za64_f64_m(1, predicate_all_b64x, predicate_all_b64x, a_slice_0_f64x, b_slice_0_f64x);
}

/** Folds @p row of ZA1-5.D and ZA7.D, scaled from their units, into @p sum and @p compensation with
 *  TwoSum. */
NUMKONG_INLINE void nk_dots_f64_smef64_fold_(nk_size_t row, svfloat64_t *sum,
                                             svfloat64_t *compensation) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b64x = svptrue_b64();
    svfloat64_t const zeros_f64x = svdup_f64(0.0);
    // Multiplying by a power of two leaves TwoProd exact, so each accumulation is a plain TwoSum
    nk_dot2_f64_sve_accumulate_(predicate_all_b64x, sum, compensation,
                                svread_hor_za64_f64_m(zeros_f64x, predicate_all_b64x, 1, row), svdup_f64(0x1p-40));
    nk_dot2_f64_sve_accumulate_(predicate_all_b64x, sum, compensation,
                                svread_hor_za64_f64_m(zeros_f64x, predicate_all_b64x, 2, row), svdup_f64(0x1p-60));
    nk_dot2_f64_sve_accumulate_(predicate_all_b64x, sum, compensation,
                                svread_hor_za64_f64_m(zeros_f64x, predicate_all_b64x, 3, row), svdup_f64(0x1p-80));
    nk_dot2_f64_sve_accumulate_(predicate_all_b64x, sum, compensation,
                                svread_hor_za64_f64_m(zeros_f64x, predicate_all_b64x, 4, row), svdup_f64(0x1p-100));
    nk_dot2_f64_sve_accumulate_(predicate_all_b64x, sum, compensation,
                                svread_hor_za64_f64_m(zeros_f64x, predicate_all_b64x, 5, row), svdup_f64(0x1p-120));
    nk_dot2_f64_sve_accumulate_(predicate_all_b64x, sum, compensation,
                                svread_hor_za64_f64_m(zeros_f64x, predicate_all_b64x, 7, row), svdup_f64(0x1p-20));
}

/** Byte offset of the column tiles in packed B, after the header and the per-column norms. */
NUMKONG_INLINE nk_size_t nk_dots_f64_smef64_tiles_offset_(nk_size_t columns) NUMKONG_STREAMABLE_ {
    return nk_size_round_up_to_multiple_(sizeof(nk_dots_sme_packed_header_t) + columns * sizeof(nk_f64_t), 64);
}

__arm_new("za") static void nk_dots_symmetric_f64_smef64_streaming_( //
    nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntd();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(vectors_count, tile_dimension);
    svbool_t const predicate_all_b64x = svptrue_b64();
    svfloat64_t const zeros_f64x = svdup_f64(0.0);
    nk_size_t const ring_steps = 2 * tile_dimension; // two batches, one staged ahead
    nk_align_(64) nk_f64_t sums[8][8], row_steps[2 * 8 * 8], column_steps[2 * 8 * 8];

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < vectors_count;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= vectors_count)
                                           ? rows_remaining
                                           : (vectors_count - row_tile_start);
        nk_f64_t const *row_vectors = vectors + row_tile_start * stride_elements;
        svint64_t const row_exponents_i64x = nk_f64_smef64_exponents_(row_vectors, stride_elements, rows_clamped,
                                                                      depth);
        svfloat64_t const row_multipliers_f64x = svscale_f64_x(predicate_all_b64x, svdup_f64(1.0),
                                                               svneg_s64_x(predicate_all_b64x, row_exponents_i64x));

        // Upper triangle: start from this row tile's column
        for (nk_size_t column_tile_index = row_tile_start / tile_dimension; column_tile_index < column_tile_count;
             column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_tile_start + tile_dimension <= vectors_count)
                                                    ? tile_dimension
                                                    : (vectors_count - column_tile_start);
            nk_f64_t const *column_vectors = vectors + column_tile_start * stride_elements;
            svbool_t const column_predicate_b64x = svwhilelt_b64_u64(0u, columns_remaining);
            svint64_t const column_exponents_i64x = nk_f64_smef64_exponents_(column_vectors, stride_elements,
                                                                             columns_remaining, depth);
            svfloat64_t const column_multipliers_f64x = svscale_f64_x(
                predicate_all_b64x, svdup_f64(1.0), svneg_s64_x(predicate_all_b64x, column_exponents_i64x));
            int const crosses_diagonal = column_tile_start < row_tile_start + tile_dimension;
            svzero_za();
            nk_f64_smef64_stage_(row_vectors, stride_elements, rows_clamped, 0, depth, row_steps);
            nk_f64_smef64_stage_(column_vectors, stride_elements, columns_remaining, 0, depth, column_steps);

            svfloat64_t a_slice_0_f64x, a_slice_1_f64x, a_slice_2_f64x, a_slice_3_f64x, a_slice_4_f64x;
            nk_f64_smef64_ozaki_split_f64x_(svld1_f64(predicate_all_b64x, row_steps), row_multipliers_f64x,
                                            &a_slice_0_f64x, &a_slice_1_f64x, &a_slice_2_f64x, &a_slice_3_f64x,
                                            &a_slice_4_f64x);
            for (nk_size_t step = 0; step < depth_padded; step++) {
                // Staging the next batch first leaves its ZA reads waiting only on older FMOPAs
                nk_size_t const next_batch = step + tile_dimension;
                if (step % tile_dimension == 0 && next_batch < depth_padded) {
                    nk_size_t const slot = next_batch % ring_steps * tile_dimension;
                    nk_f64_smef64_stage_(row_vectors, stride_elements, rows_clamped, next_batch, depth,
                                         row_steps + slot);
                    nk_f64_smef64_stage_(column_vectors, stride_elements, columns_remaining, next_batch, depth,
                                         column_steps + slot);
                }
                svfloat64_t b_slice_0_f64x, b_slice_1_f64x, b_slice_2_f64x, b_slice_3_f64x, b_slice_4_f64x;
                nk_f64_smef64_ozaki_split_f64x_(
                    svld1_f64(predicate_all_b64x, column_steps + step % ring_steps * tile_dimension),
                    column_multipliers_f64x, &b_slice_0_f64x, &b_slice_1_f64x, &b_slice_2_f64x, &b_slice_3_f64x,
                    &b_slice_4_f64x);
                // Splitting the next row step here keeps its chain in flight behind the FMOPAs
                svfloat64_t next_0_f64x, next_1_f64x, next_2_f64x, next_3_f64x, next_4_f64x;
                nk_f64_smef64_ozaki_split_f64x_(
                    svld1_f64(predicate_all_b64x, row_steps + (step + 1) % ring_steps * tile_dimension),
                    row_multipliers_f64x, &next_0_f64x, &next_1_f64x, &next_2_f64x, &next_3_f64x, &next_4_f64x);
                nk_dots_f64_smef64_accumulate_(a_slice_0_f64x, a_slice_1_f64x, a_slice_2_f64x, a_slice_3_f64x,
                                               a_slice_4_f64x, b_slice_0_f64x, b_slice_1_f64x, b_slice_2_f64x,
                                               b_slice_3_f64x, b_slice_4_f64x);
                a_slice_0_f64x = next_0_f64x, a_slice_1_f64x = next_1_f64x, a_slice_2_f64x = next_2_f64x;
                a_slice_3_f64x = next_3_f64x, a_slice_4_f64x = next_4_f64x;

                nk_size_t const steps_done = step + 1;
                if (steps_done % nk_dots_f64_smef64_flush_steps_ != 0 || steps_done >= depth) continue;
                for (nk_size_t row = 0; row < rows_clamped; row++) {
                    svbool_t const store_b64x = crosses_diagonal
                                                    ? nk_sme_diagonal_cut_b64x_(column_predicate_b64x,
                                                                                column_tile_start, row_tile_start + row)
                                                    : column_predicate_b64x;
                    nk_f64_t *result_row = result + (row_tile_start + row) * result_stride_elements + column_tile_start;
                    int const folded_before = steps_done > nk_dots_f64_smef64_flush_steps_;
                    svfloat64_t sum_f64x = folded_before ? svld1_f64(predicate_all_b64x, sums[row]) : zeros_f64x;
                    svfloat64_t compensation_f64x = folded_before ? svld1_f64(store_b64x, result_row) : zeros_f64x;
                    nk_dots_f64_smef64_fold_(row, &sum_f64x, &compensation_f64x);
                    svst1_f64(predicate_all_b64x, sums[row], sum_f64x);
                    svst1_f64(store_b64x, result_row, compensation_f64x);
                }
                svzero_mask_za(nk_sme_zero_za64_tiles_1_5_7_k);
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                svbool_t const store_b64x = crosses_diagonal
                                                ? nk_sme_diagonal_cut_b64x_(column_predicate_b64x, column_tile_start,
                                                                            row_tile_start + row)
                                                : column_predicate_b64x;
                nk_f64_t *result_row = result + (row_tile_start + row) * result_stride_elements + column_tile_start;
                int const folded_before = depth > nk_dots_f64_smef64_flush_steps_;
                svfloat64_t sum_f64x = folded_before ? svld1_f64(predicate_all_b64x, sums[row]) : zeros_f64x;
                svfloat64_t compensation_f64x = folded_before ? svld1_f64(store_b64x, result_row) : zeros_f64x;
                nk_dots_f64_smef64_fold_(row, &sum_f64x, &compensation_f64x);
                svint64_t const exponents_i64x = svadd_s64_x(predicate_all_b64x, column_exponents_i64x,
                                                             svdup_lane_s64(row_exponents_i64x, row));
                svst1_f64(store_b64x, result_row,
                          svscale_f64_x(predicate_all_b64x,
                                        svadd_f64_x(predicate_all_b64x, sum_f64x, compensation_f64x), exponents_i64x));
            }
        }
    }
}

NUMKONG_API nk_status_t nk_dots_symmetric_f64_smef64( //
    nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f64_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));

    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_f64_smef64_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                            result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_f64_smef64(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    nk_size_t const tile_dimension = nk_sme_cntd_();
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    // Each column tile holds its exponents, then 5 slices of each column per depth step
    *bytes = nk_dots_f64_smef64_tiles_offset_(columns) +
             column_tile_count * tile_dimension * (1 + 5 * depth_padded) * sizeof(nk_f64_t);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_f64_smef64(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_smef64_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_f64_smef64(nk_f64_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f64_t);
    nk_size_t const tile_dimension = nk_sme_cntd_();
    nk_size_t const tile_elements = tile_dimension * (1 + 5 * nk_size_round_up_to_multiple_(depth, tile_dimension));
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const tile_column_begin = nk_size_divide_round_up_(columns_begin, tile_dimension);
    nk_size_t tile_column_end = nk_size_divide_round_up_(columns_end, tile_dimension);
    if (tile_column_end > column_tile_count) tile_column_end = column_tile_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    if (columns_begin == 0) {
        for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
            ((nk_u32_t *)header)[word_index] = 0;
        // The norms end short of the 64-byte aligned tiles, so the gap between them is zeroed too.
        for (nk_size_t byte_index = sizeof(*header) + columns * sizeof(nk_f64_t);
             byte_index < nk_dots_f64_smef64_tiles_offset_(columns); byte_index++)
            ((char *)b_packed)[byte_index] = 0;
        header->column_tile_count = (nk_u32_t)column_tile_count;
        header->depth_tile_count = (nk_u32_t)nk_size_divide_round_up_(depth, tile_dimension);
        header->columns = (nk_u32_t)columns;
        header->depth = (nk_u32_t)depth;
        header->svl_bytes = (nk_u32_t)nk_sme_cntb_();
        header->capability = nk_cap_smef64_k;
        header->norms_offset = (nk_u32_t)sizeof(nk_dots_sme_packed_header_t);
    }

    nk_f64_t *tiles = (nk_f64_t *)((char *)b_packed + nk_dots_f64_smef64_tiles_offset_(columns));
    for (nk_size_t i = tile_column_begin * tile_elements; i < tile_column_end * tile_elements; i++) tiles[i] = 0.0;

    for (nk_size_t column_tile_index = tile_column_begin; column_tile_index < tile_column_end; column_tile_index++) {
        nk_f64_t *tile = tiles + column_tile_index * tile_elements;
        nk_size_t const column_start = column_tile_index * tile_dimension;
        nk_size_t const columns_in_tile = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - column_start);
        for (nk_size_t column = 0; column < columns_in_tile; column++) {
            nk_f64_t const *values = b + (column_start + column) * b_stride_elements;
            nk_u64_t magnitude_bits = 0;
            for (nk_size_t depth_index = 0; depth_index < depth; depth_index++) {
                nk_fui64_t pun;
                pun.f = values[depth_index];
                if ((pun.u & 0x7FFFFFFFFFFFFFFFull) > magnitude_bits) magnitude_bits = pun.u & 0x7FFFFFFFFFFFFFFFull;
            }
            nk_i64_t const exponent = nk_f64_smef64_exponent_above_(magnitude_bits);
            tile[column] = (nk_f64_t)exponent;
            for (nk_size_t depth_index = 0; depth_index < depth; depth_index++)
                nk_f64_smef64_ozaki_split_f64_(nk_f64_smef64_scaled_(values[depth_index], -exponent),
                                               tile + tile_dimension * (1 + 5 * depth_index) + column, tile_dimension);
        }
    }

    nk_f64_t *norms = (nk_f64_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));
    for (nk_size_t column = columns_begin; column < columns_end; column++)
        norms[column] = nk_dots_reduce_sumsq_f64_(b + column * b_stride_elements, depth, nk_cap_smef64_k);
    return nk_success_k;
}

__arm_new("za") static void nk_dots_packed_f64_smef64_streaming_( //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntd();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const tile_elements = tile_dimension * (1 + 5 * depth_padded);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_f64_t const *tiles = (nk_f64_t const *)((char const *)b_packed + nk_dots_f64_smef64_tiles_offset_(columns));
    svbool_t const predicate_all_b64x = svptrue_b64();
    svfloat64_t const zeros_f64x = svdup_f64(0.0);
    nk_align_(64) nk_f64_t sums[8][8];

    for (nk_size_t row_start = 0; row_start < rows; row_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        nk_f64_t const *row_vectors = a + row_start * a_stride_elements;
        svint64_t const row_exponents_i64x = nk_f64_smef64_exponents_(row_vectors, a_stride_elements, rows_remaining,
                                                                      depth);
        svfloat64_t const row_multipliers_f64x = svscale_f64_x(predicate_all_b64x, svdup_f64(1.0),
                                                               svneg_s64_x(predicate_all_b64x, row_exponents_i64x));

        for (nk_size_t column_tile_index = 0; column_tile_index < column_tile_count; column_tile_index++) {
            nk_f64_t const *tile = tiles + column_tile_index * tile_elements;
            nk_size_t const column_start = column_tile_index * tile_dimension;
            svbool_t const column_predicate_b64x = svwhilelt_b64_u64(column_start, columns);
            svint64_t const column_exponents_i64x = svcvt_s64_f64_x(predicate_all_b64x,
                                                                    svld1_f64(predicate_all_b64x, tile));
            nk_f64_t *c_tile = c + row_start * c_stride_elements + column_start;
            svzero_za(); // staged rows past the tile's edge then read back as zeros

            for (nk_size_t row = 0; row < rows_remaining; row++)
                svld1_hor_za64(0, row, svwhilelt_b64_u64(0u, depth), row_vectors + row * a_stride_elements);
            svfloat64_t column_f64x = svread_ver_za64_f64_m(svundef_f64(), predicate_all_b64x, 0, 0);
            svfloat64_t next_column_f64x = svread_ver_za64_f64_m(svundef_f64(), predicate_all_b64x, 0, 1);

            nk_f64_t const *b_slices = tile + tile_dimension;
            for (nk_size_t step = 0; step < depth_padded; step++, b_slices += 5 * tile_dimension) {
                // Reading 2 steps ahead leaves each ZA read waiting only on the last step's FMOPAs
                nk_size_t const ahead = step + 2;
                if (ahead % tile_dimension == 0 && ahead < depth_padded)
                    for (nk_size_t row = 0; row < rows_remaining; row++)
                        svld1_hor_za64(0, row, svwhilelt_b64_u64(ahead, depth),
                                       row_vectors + row * a_stride_elements + ahead);
                svfloat64_t const ahead_column_f64x = svread_ver_za64_f64_m(svundef_f64(), predicate_all_b64x, 0,
                                                                            ahead & (tile_dimension - 1));
                svfloat64_t a_slice_0_f64x, a_slice_1_f64x, a_slice_2_f64x, a_slice_3_f64x, a_slice_4_f64x;
                nk_f64_smef64_ozaki_split_f64x_(column_f64x, row_multipliers_f64x, &a_slice_0_f64x, &a_slice_1_f64x,
                                                &a_slice_2_f64x, &a_slice_3_f64x, &a_slice_4_f64x);
                nk_dots_f64_smef64_accumulate_(a_slice_0_f64x, a_slice_1_f64x, a_slice_2_f64x, a_slice_3_f64x,
                                               a_slice_4_f64x, svld1_f64(predicate_all_b64x, b_slices),
                                               svld1_f64(predicate_all_b64x, b_slices + tile_dimension),
                                               svld1_f64(predicate_all_b64x, b_slices + 2 * tile_dimension),
                                               svld1_f64(predicate_all_b64x, b_slices + 3 * tile_dimension),
                                               svld1_f64(predicate_all_b64x, b_slices + 4 * tile_dimension));
                column_f64x = next_column_f64x, next_column_f64x = ahead_column_f64x;

                nk_size_t const steps_done = step + 1;
                if (steps_done % nk_dots_f64_smef64_flush_steps_ != 0 || steps_done >= depth) continue;
                for (nk_size_t row = 0; row < rows_remaining; row++) {
                    nk_f64_t *c_row = c_tile + row * c_stride_elements;
                    int const folded_before = steps_done > nk_dots_f64_smef64_flush_steps_;
                    svfloat64_t sum_f64x = folded_before ? svld1_f64(predicate_all_b64x, sums[row]) : zeros_f64x;
                    svfloat64_t compensation_f64x = folded_before ? svld1_f64(column_predicate_b64x, c_row)
                                                                  : zeros_f64x;
                    nk_dots_f64_smef64_fold_(row, &sum_f64x, &compensation_f64x);
                    svst1_f64(predicate_all_b64x, sums[row], sum_f64x);
                    svst1_f64(column_predicate_b64x, c_row, compensation_f64x);
                }
                svzero_mask_za(nk_sme_zero_za64_tiles_1_5_7_k);
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_f64_t *c_row = c_tile + row * c_stride_elements;
                int const folded_before = depth > nk_dots_f64_smef64_flush_steps_;
                svfloat64_t sum_f64x = folded_before ? svld1_f64(predicate_all_b64x, sums[row]) : zeros_f64x;
                svfloat64_t compensation_f64x = folded_before ? svld1_f64(column_predicate_b64x, c_row) : zeros_f64x;
                nk_dots_f64_smef64_fold_(row, &sum_f64x, &compensation_f64x);
                svint64_t const exponents_i64x = svadd_s64_x(predicate_all_b64x, column_exponents_i64x,
                                                             svdup_lane_s64(row_exponents_i64x, row));
                svst1_f64(column_predicate_b64x, c_row,
                          svscale_f64_x(predicate_all_b64x,
                                        svadd_f64_x(predicate_all_b64x, sum_f64x, compensation_f64x), exponents_i64x));
            }
        }
    }
}

NUMKONG_API nk_status_t nk_dots_packed_f64_smef64( //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_smef64_k) return nk_pack_mismatch_k;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_sme_start_streaming_();
    nk_dots_packed_f64_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion F64 Floats

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_SMEF64
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_DOTS_SMEF64_H
