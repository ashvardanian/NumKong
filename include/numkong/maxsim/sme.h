/**
 *  @file include/numkong/maxsim/sme.h
 *  @author Ash Vardanian
 *  @date February 10, 2026
 *  @brief SIMD-accelerated MaxSim, ColBERT late-interaction, for SME.
 *
 *  Computes MaxSim(Q, D) = Σᵢ minⱼ angular(qᵢ, dⱼ) using ARM SME outer products, ranking
 *  documents by dot / ‖d‖, which orders them by cosine.
 *
 *  Both Q and D are pre-packed with @c nk_dots_pack_bf16_sme from `dots/sme.h`, which frees all 4
 *  ZA tiles for accumulation, versus 3 with A-side staging.
 *
 *  Key optimization: vertical column reads for max reduction. Traditional extraction reads tile
 *  rows, then calls @c svmaxv, a horizontal max at ~8cy. Our approach reads tile columns with
 *  @c svread_ver_za32_f32_m:
 *
 *  @verbatim
 *  - Each column read gives dot products of all query tokens vs one doc token.
 *  - Element-wise svmax   (~1cy) updates a running max vector across doc tokens.
 *  - Only svaddv   at the very end: ⌈n_q/16⌉ = 2 horizontal reductions total.
 *  @endverbatim
 *
 *  This is ~100x fewer horizontal reductions for typical ColBERT dimensions.
 *
 *  ZA tile layout after BFMOPA accumulation, 16x16 f32:
 *
 *  - Row i, Column j = dot(q_{tile_row_start + i}, d_{tile_col_start + j})
 *  - Vertical column read of column j → similarities of all 16 q tokens to doc token j
 *  - Element-wise max across columns → per-query-token max over doc tokens in this tile group
 *
 *  Benchmark results, Apple M4, SVL=512:
 *
 *  @verbatim
 *  Dimensions              GEMM GFLOPS         fused GFLOPS    GEMM ×          end-to-end ×
 *  32×128×128 (ColBERT)    840 GFLOPS          1516 GFLOPS     1.81×           5.10×
 *  32×256×128              1037 GFLOPS         1591 GFLOPS     1.53×           5.17×
 *  64×512×128              1016 GFLOPS         1651 GFLOPS     1.62×           5.42×
 *  32×128×256              859 GFLOPS          1725 GFLOPS     2.01×           4.06×
 *  32×1024×768 (BERT)      1124 GFLOPS         1932 GFLOPS     1.72×           2.61×
 *  @endverbatim
 *
 *  Speedup sources:
 *
 *  1. Pre-packing both sides → 4 ZA tiles, vs 3 with A-staging: +33% MOPA throughput
 *  2. No output matrix materialization → eliminates M × N f32 memory round-trip
 *  3. Vertical column reads → ~128 element-wise svmax, 1cy, vs ~256 svmaxv reductions, 8cy
 */
#ifndef NUMKONG_MAXSIM_SME_H
#define NUMKONG_MAXSIM_SME_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_SME_

#include "numkong/dots/sme.h"      // `nk_dots_pack_bf16_tiles_sme_`, `nk_sme_zero_za32_*`
#include "numkong/maxsim/serial.h" // `nk_maxsim_screen_error_`
#include "numkong/reduce/sve.h"    // `nk_svaddv_f64_`
#include "numkong/scalar/neon.h"   // `nk_f64_rsqrt_refined_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

/**
 *  @brief Packed header for MaxSim SME kernels, shared by the f32 kernels, which screen in i8 and
 *      refine in f32, and the bf16 and f16 kernels, which use BFMOPA or FMOPA and normalize angles.
 *
 *  For f32 it stores i8 tile-interleaved data, f64 inverse norms, f32 screening weights absmax /
 *  (127 · ‖v‖), and the f32 originals. For bf16 and f16 it stores tile-interleaved data and f32
 *  inverse norms 1 / ‖v‖, leaving @c originals_offset, @c original_stride and
 *  @c screen_weights_offset unused at 0.
 */
typedef struct {

    /** Column tiles, ⌈vectors / tile dimension⌉. */
    nk_u32_t column_tile_count;

    /** Depth tiles, ⌈depth / expansion⌉. */
    nk_u32_t depth_tile_count;

    /** Vectors, not padded, for the predicates. */
    nk_u32_t columns;

    /** Depth, not padded. */
    nk_u32_t depth;

    /** Streaming vector length in bytes at pack time, which every consumer validates. */
    nk_u32_t svl_bytes;

    /** Byte offset to the per-vector inverse norms: f64 for f32, f32 for bf16 and f16. */
    nk_u32_t norms_offset;

    /** Byte offset to the f32 original vectors, or 0 for bf16 and f16. */
    nk_u32_t originals_offset;

    /** Row stride in bytes of the originals, 64-byte aligned, or 0 for bf16 and f16. */
    nk_u32_t original_stride;

    /** Byte offset to the f32 screening weights, or 0 for bf16 and f16. */
    nk_u32_t screen_weights_offset;

    /** Zeroed; pads the header to 64 bytes. */
    nk_u32_t reserved[5];

    /** The capability that packed the buffer, which every consumer checks. */
    nk_capability_t capability;
} nk_maxsim_sme_packed_header_t;

nk_static_assert_(sizeof(nk_maxsim_sme_packed_header_t) == 64, nk_maxsim_sme_packed_header_must_be_64_bytes);

/**
 *  @brief MaxSim f16 kernel with both Q and D pre-packed, extracting through vertical column reads.
 *
 *  The 4-tile fast path processes 4 document column tiles simultaneously using ZA0-ZA3. Each depth
 *  step of the inner loop costs 1 Q load, 4 D loads and 4 FMOPA, 9 ops in all. Extraction per
 *  4-tile group costs 4 × 16 = 64 vertical reads and 64 svmax, about 128 cycles.
 *
 *  The 1-tile remainder uses ZA0 only, with predicated loads for partial tiles.
 */
__arm_new("za") static void nk_maxsim_packed_f16_streaming_( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_f32_t *result) NUMKONG_STREAMING_ {

    nk_maxsim_sme_packed_header_t const *query_header = (nk_maxsim_sme_packed_header_t const *)query_packed;
    nk_maxsim_sme_packed_header_t const *document_header = (nk_maxsim_sme_packed_header_t const *)document_packed;
    nk_size_t const depth_step_count = query_header->depth_tile_count;
    nk_size_t const query_row_tiles = query_header->column_tile_count;
    nk_size_t const document_col_tiles = document_header->column_tile_count;

    nk_size_t const tile_dimension = svcntw();  // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcnth(); // 32: f16 elements per SVE vector

    nk_f16_t const *query_vecs = (nk_f16_t const *)((char const *)query_packed + sizeof(nk_maxsim_sme_packed_header_t));
    nk_f16_t const *document_vecs = (nk_f16_t const *)((char const *)document_packed +
                                                       sizeof(nk_maxsim_sme_packed_header_t));

    nk_f32_t const *query_inverse_norms = (nk_f32_t const *)((char const *)query_packed + query_header->norms_offset);
    nk_f32_t const *document_inverse_norms = (nk_f32_t const *)((char const *)document_packed +
                                                                document_header->norms_offset);

    svbool_t const predicate_all_b16x = svptrue_b16();
    svbool_t const predicate_all_b32x = svptrue_b32();

    nk_f32_t total_angular_distance = 0.0f;

    for (nk_size_t row_tile_index = 0; row_tile_index < query_row_tiles; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= query_count) ? tile_dimension
                                                                                     : (query_count - row_start);
        svbool_t const row_predicate_b16x = (rows_remaining == tile_dimension)
                                                ? svptrue_b16()
                                                : svwhilelt_b16_u64(0u, rows_remaining * 2);
        svbool_t const row_predicate_b32x = (rows_remaining == tile_dimension) ? svptrue_b32()
                                                                               : svwhilelt_b32_u64(0u, rows_remaining);

        // Running max of dot / ‖d‖ per query for angular distance finalization
        svfloat32_t running_maximum_f32x = svdup_f32(NUMKONG_F32_MIN);

        nk_size_t column_tile_index = 0;

        // Fast path: 4 doc column tiles at a time using ZA0-ZA3
        for (; column_tile_index + 4 <= document_col_tiles; column_tile_index += 4) {
            svzero_za(); // Zero all 4 tiles

            // Accumulate: for each depth step, load Q vector and 4 D vectors, issue 4 FMOPAs
            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svfloat16_t query_packed_f16x = svld1_f16(
                    row_predicate_b16x,
                    (float16_t const *)(query_vecs +
                                        (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svfloat16_t document_packed_0_f16x = svld1_f16(
                    predicate_all_b16x,
                    (float16_t const *)(document_vecs +
                                        ((column_tile_index + 0) * depth_step_count + depth_step) * vector_elements));
                svfloat16_t document_packed_1_f16x = svld1_f16(
                    predicate_all_b16x,
                    (float16_t const *)(document_vecs +
                                        ((column_tile_index + 1) * depth_step_count + depth_step) * vector_elements));
                svfloat16_t document_packed_2_f16x = svld1_f16(
                    predicate_all_b16x,
                    (float16_t const *)(document_vecs +
                                        ((column_tile_index + 2) * depth_step_count + depth_step) * vector_elements));
                svfloat16_t document_packed_3_f16x = svld1_f16(
                    predicate_all_b16x,
                    (float16_t const *)(document_vecs +
                                        ((column_tile_index + 3) * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_f16_m(0, row_predicate_b16x, predicate_all_b16x, query_packed_f16x, document_packed_0_f16x);
                svmopa_za32_f16_m(1, row_predicate_b16x, predicate_all_b16x, query_packed_f16x, document_packed_1_f16x);
                svmopa_za32_f16_m(2, row_predicate_b16x, predicate_all_b16x, query_packed_f16x, document_packed_2_f16x);
                svmopa_za32_f16_m(3, row_predicate_b16x, predicate_all_b16x, query_packed_f16x, document_packed_3_f16x);
            }

            // Vertical column extraction + max update (manually unrolled over 4 tiles)
            nk_size_t const last_col_start = (column_tile_index + 3) * tile_dimension;
            nk_size_t const last_columns_remaining = (last_col_start + tile_dimension <= document_count)
                                                         ? tile_dimension
                                                         : (document_count - last_col_start);
            for (nk_size_t column_within_tile = 0; column_within_tile < tile_dimension; column_within_tile++) {
                // Tile 0
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 0) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x,
                                                                         0, column_within_tile);
                    running_maximum_f32x = svmax_f32_x(
                        predicate_all_b32x, running_maximum_f32x,
                        svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
                }
                // Tile 1
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 1) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x,
                                                                         1, column_within_tile);
                    running_maximum_f32x = svmax_f32_x(
                        predicate_all_b32x, running_maximum_f32x,
                        svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
                }
                // Tile 2
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 2) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x,
                                                                         2, column_within_tile);
                    running_maximum_f32x = svmax_f32_x(
                        predicate_all_b32x, running_maximum_f32x,
                        svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
                }
                // Tile 3, whose zero-padded columns would outscore all-negative dots
                if (column_within_tile < last_columns_remaining) {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 3) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x,
                                                                         3, column_within_tile);
                    running_maximum_f32x = svmax_f32_x(
                        predicate_all_b32x, running_maximum_f32x,
                        svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
                }
            }
        }

        // Remainder: 1 doc column tile at a time using ZA0 only
        for (; column_tile_index < document_col_tiles; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (col_start + tile_dimension <= document_count)
                                                    ? tile_dimension
                                                    : (document_count - col_start);
            svbool_t const column_predicate_b16x = (columns_remaining == tile_dimension)
                                                       ? svptrue_b16()
                                                       : svwhilelt_b16_u64(0u, columns_remaining * 2);

            svzero_mask_za(nk_sme_zero_za32_tile_0_k); // Zero ZA0 only

            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svfloat16_t query_packed_f16x = svld1_f16(
                    row_predicate_b16x,
                    (float16_t const *)(query_vecs +
                                        (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svfloat16_t document_packed_f16x = svld1_f16(
                    column_predicate_b16x,
                    (float16_t const *)(document_vecs +
                                        (column_tile_index * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_f16_m(0, row_predicate_b16x, column_predicate_b16x, query_packed_f16x,
                                  document_packed_f16x);
            }

            // Vertical column extraction from ZA0 + max update
            for (nk_size_t column_within_tile = 0; column_within_tile < columns_remaining; column_within_tile++) {
                nk_u32_t document_index = (nk_u32_t)(col_start + column_within_tile);
                svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x, 0,
                                                                     column_within_tile);
                running_maximum_f32x = svmax_f32_x(
                    predicate_all_b32x, running_maximum_f32x,
                    svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
            }
        }

        // SVE-width: cosine = max(dot * inv_norm_d) * inv_norm_q, angular = max(1 - cosine, 0)
        svfloat32_t query_inverse_norms_f32x = svld1_f32(row_predicate_b32x, query_inverse_norms + row_start);
        svfloat32_t cosine_f32x = svmul_f32_x(row_predicate_b32x, running_maximum_f32x, query_inverse_norms_f32x);
        svfloat32_t angular_distance_f32x = svmax_f32_x(
            row_predicate_b32x, svsub_f32_x(row_predicate_b32x, svdup_f32(1.0f), cosine_f32x), svdup_f32(0.0f));
        total_angular_distance += nk_svaddv_f32_(row_predicate_b32x, angular_distance_f32x);
    }

    *result = total_angular_distance;
}

#if NUMKONG_TARGET_SME
NUMKONG_API nk_status_t nk_maxsim_packed_f16_sme( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_maxsim_sme_packed_header_t const *)query_packed)->capability != nk_cap_sme_k ||
        ((nk_maxsim_sme_packed_header_t const *)document_packed)->capability != nk_cap_sme_k)
        return nk_pack_mismatch_k;
    nk_unused_(depth);

    nk_sme_start_streaming_();
    nk_maxsim_packed_f16_streaming_(query_packed, document_packed, query_count, document_count, result);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
#endif // NUMKONG_TARGET_SME

/**
 *  @brief MaxSim bf16 kernel with Q and D pre-packed, extracting through vertical column reads.
 *
 *  The 4-tile fast path processes 4 document column tiles simultaneously using ZA0-ZA3. Each depth
 *  step of the inner loop costs 1 Q load, 4 D loads and 4 BFMOPA, 9 ops in all. Extraction per
 *  4-tile group costs 4 × 16 = 64 vertical reads and 64 svmax, about 128 cycles.
 *
 *  The 1-tile remainder uses ZA0 only, with predicated loads for partial tiles.
 */
__arm_new("za") static void nk_maxsim_packed_bf16_streaming_( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_f32_t *result) NUMKONG_STREAMING_ {

    nk_maxsim_sme_packed_header_t const *query_header = (nk_maxsim_sme_packed_header_t const *)query_packed;
    nk_maxsim_sme_packed_header_t const *document_header = (nk_maxsim_sme_packed_header_t const *)document_packed;
    nk_size_t const depth_step_count = query_header->depth_tile_count;
    nk_size_t const query_row_tiles = query_header->column_tile_count;
    nk_size_t const document_col_tiles = document_header->column_tile_count;

    nk_size_t const tile_dimension = svcntw();  // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcnth(); // 32: bf16 elements per SVE vector

    nk_bf16_t const *query_vecs = (nk_bf16_t const *)((char const *)query_packed +
                                                      sizeof(nk_maxsim_sme_packed_header_t));
    nk_bf16_t const *document_vecs = (nk_bf16_t const *)((char const *)document_packed +
                                                         sizeof(nk_maxsim_sme_packed_header_t));

    nk_f32_t const *query_inverse_norms = (nk_f32_t const *)((char const *)query_packed + query_header->norms_offset);
    nk_f32_t const *document_inverse_norms = (nk_f32_t const *)((char const *)document_packed +
                                                                document_header->norms_offset);

    svbool_t const predicate_all_b16x = svptrue_b16();
    svbool_t const predicate_all_b32x = svptrue_b32();

    nk_f32_t total_angular_distance = 0.0f;

    for (nk_size_t row_tile_index = 0; row_tile_index < query_row_tiles; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= query_count) ? tile_dimension
                                                                                     : (query_count - row_start);
        svbool_t const row_predicate_b16x = (rows_remaining == tile_dimension)
                                                ? svptrue_b16()
                                                : svwhilelt_b16_u64(0u, rows_remaining * 2);
        svbool_t const row_predicate_b32x = (rows_remaining == tile_dimension) ? svptrue_b32()
                                                                               : svwhilelt_b32_u64(0u, rows_remaining);

        // Running max of dot / ‖d‖ per query for angular distance finalization
        svfloat32_t running_maximum_f32x = svdup_f32(NUMKONG_F32_MIN);

        nk_size_t column_tile_index = 0;

        // Fast path: 4 doc column tiles at a time using ZA0-ZA3
        for (; column_tile_index + 4 <= document_col_tiles; column_tile_index += 4) {
            svzero_za(); // Zero all 4 tiles

            // Accumulate: for each depth step, load Q vector and 4 D vectors, issue 4 BFMOPAs
            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svbfloat16_t query_packed_bf16x = svld1_bf16(
                    row_predicate_b16x,
                    (bfloat16_t const *)(query_vecs +
                                         (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t document_packed_0_bf16x = svld1_bf16(
                    predicate_all_b16x,
                    (bfloat16_t const *)(document_vecs +
                                         ((column_tile_index + 0) * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t document_packed_1_bf16x = svld1_bf16(
                    predicate_all_b16x,
                    (bfloat16_t const *)(document_vecs +
                                         ((column_tile_index + 1) * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t document_packed_2_bf16x = svld1_bf16(
                    predicate_all_b16x,
                    (bfloat16_t const *)(document_vecs +
                                         ((column_tile_index + 2) * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t document_packed_3_bf16x = svld1_bf16(
                    predicate_all_b16x,
                    (bfloat16_t const *)(document_vecs +
                                         ((column_tile_index + 3) * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_bf16_m(0, row_predicate_b16x, predicate_all_b16x, query_packed_bf16x,
                                   document_packed_0_bf16x);
                svmopa_za32_bf16_m(1, row_predicate_b16x, predicate_all_b16x, query_packed_bf16x,
                                   document_packed_1_bf16x);
                svmopa_za32_bf16_m(2, row_predicate_b16x, predicate_all_b16x, query_packed_bf16x,
                                   document_packed_2_bf16x);
                svmopa_za32_bf16_m(3, row_predicate_b16x, predicate_all_b16x, query_packed_bf16x,
                                   document_packed_3_bf16x);
            }

            // Vertical column extraction + max update (manually unrolled over 4 tiles)
            nk_size_t const last_col_start = (column_tile_index + 3) * tile_dimension;
            nk_size_t const last_columns_remaining = (last_col_start + tile_dimension <= document_count)
                                                         ? tile_dimension
                                                         : (document_count - last_col_start);
            for (nk_size_t column_within_tile = 0; column_within_tile < tile_dimension; column_within_tile++) {
                // Tile 0
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 0) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x,
                                                                         0, column_within_tile);
                    running_maximum_f32x = svmax_f32_x(
                        predicate_all_b32x, running_maximum_f32x,
                        svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
                }
                // Tile 1
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 1) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x,
                                                                         1, column_within_tile);
                    running_maximum_f32x = svmax_f32_x(
                        predicate_all_b32x, running_maximum_f32x,
                        svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
                }
                // Tile 2
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 2) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x,
                                                                         2, column_within_tile);
                    running_maximum_f32x = svmax_f32_x(
                        predicate_all_b32x, running_maximum_f32x,
                        svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
                }
                // Tile 3, whose zero-padded columns would outscore all-negative dots
                if (column_within_tile < last_columns_remaining) {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 3) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x,
                                                                         3, column_within_tile);
                    running_maximum_f32x = svmax_f32_x(
                        predicate_all_b32x, running_maximum_f32x,
                        svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
                }
            }
        }

        // Remainder: 1 doc column tile at a time using ZA0 only
        for (; column_tile_index < document_col_tiles; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (col_start + tile_dimension <= document_count)
                                                    ? tile_dimension
                                                    : (document_count - col_start);
            svbool_t const column_predicate_b16x = (columns_remaining == tile_dimension)
                                                       ? svptrue_b16()
                                                       : svwhilelt_b16_u64(0u, columns_remaining * 2);

            svzero_mask_za(nk_sme_zero_za32_tile_0_k); // Zero ZA0 only

            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svbfloat16_t query_packed_bf16x = svld1_bf16(
                    row_predicate_b16x,
                    (bfloat16_t const *)(query_vecs +
                                         (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t document_packed_bf16x = svld1_bf16(
                    column_predicate_b16x,
                    (bfloat16_t const *)(document_vecs +
                                         (column_tile_index * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_bf16_m(0, row_predicate_b16x, column_predicate_b16x, query_packed_bf16x,
                                   document_packed_bf16x);
            }

            // Vertical column extraction from ZA0 + max update
            for (nk_size_t column_within_tile = 0; column_within_tile < columns_remaining; column_within_tile++) {
                nk_u32_t document_index = (nk_u32_t)(col_start + column_within_tile);
                svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NUMKONG_F32_MIN), predicate_all_b32x, 0,
                                                                     column_within_tile);
                running_maximum_f32x = svmax_f32_x(
                    predicate_all_b32x, running_maximum_f32x,
                    svmul_n_f32_x(predicate_all_b32x, column_dots_f32x, document_inverse_norms[document_index]));
            }
        }

        // SVE-width: cosine = max(dot * inv_norm_d) * inv_norm_q, angular = max(1 - cosine, 0)
        svfloat32_t query_inverse_norms_f32x = svld1_f32(row_predicate_b32x, query_inverse_norms + row_start);
        svfloat32_t cosine_f32x = svmul_f32_x(row_predicate_b32x, running_maximum_f32x, query_inverse_norms_f32x);
        svfloat32_t angular_distance_f32x = svmax_f32_x(
            row_predicate_b32x, svsub_f32_x(row_predicate_b32x, svdup_f32(1.0f), cosine_f32x), svdup_f32(0.0f));
        total_angular_distance += nk_svaddv_f32_(row_predicate_b32x, angular_distance_f32x);
    }

    *result = total_angular_distance;
}

#if NUMKONG_TARGET_SME
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_sme( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_maxsim_sme_packed_header_t const *)query_packed)->capability != nk_cap_sme_k ||
        ((nk_maxsim_sme_packed_header_t const *)document_packed)->capability != nk_cap_sme_k)
        return nk_pack_mismatch_k;
    nk_unused_(depth);

    nk_sme_start_streaming_();
    nk_maxsim_packed_bf16_streaming_(query_packed, document_packed, query_count, document_count, result);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) { //
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_sme(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                        void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_sme_packed_header_t const *header = (nk_maxsim_sme_packed_header_t const *)packed;
    if (header->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    *vectors = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) { //
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_sme(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                       void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_sme_packed_header_t const *header = (nk_maxsim_sme_packed_header_t const *)packed;
    if (header->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    *vectors = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_sme( //
    nk_bf16_t const *vectors, nk_size_t columns, nk_size_t depth, nk_size_t stride, void *packed,
    void *stream) { //
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const blob_bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    for (nk_size_t byte_index = 0; byte_index < blob_bytes; byte_index++) ((char *)packed)[byte_index] = 0;

    // Delegate tile interleaving and squared norms computation to dots pack.
    // Both headers are 64 bytes with identical layout for the first 6 fields.
    nk_dots_pack_bf16_tiles_sme_(vectors, columns, depth, stride, packed, 0, columns);

    // Set maxsim-specific header fields (overlaps dots reserved area)
    nk_maxsim_sme_packed_header_t *header = (nk_maxsim_sme_packed_header_t *)packed;
    header->originals_offset = 0;      // not used for bf16
    header->original_stride = 0;       // not used for bf16
    header->screen_weights_offset = 0; // not used for bf16
    header->capability = nk_cap_sme_k;
    for (nk_size_t i = 0; i < 5; i++) header->reserved[i] = 0;

    // Convert squared norms → inverse norms in-place
    nk_f32_t *norms = (nk_f32_t *)((char *)packed + header->norms_offset);
    for (nk_size_t i = 0; i < columns; i++) {
        nk_f32_t norm_sq = norms[i];
        norms[i] = (norm_sq > 0.0f) ? (nk_f32_t)nk_f64_rsqrt_refined_neon_((nk_f64_t)norm_sq) : 0.0f;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_sme( //
    nk_f16_t const *vectors, nk_size_t columns, nk_size_t depth, nk_size_t stride, void *packed,
    void *stream) { //
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const blob_bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    for (nk_size_t byte_index = 0; byte_index < blob_bytes; byte_index++) ((char *)packed)[byte_index] = 0;

    // Delegate tile interleaving and squared norms computation to dots pack.
    // Both headers are 64 bytes with identical layout for the first 6 fields.
    nk_dots_pack_f16_tiles_sme_(vectors, columns, depth, stride, packed, 0, columns);

    // Set maxsim-specific header fields (overlaps dots reserved area)
    nk_maxsim_sme_packed_header_t *header = (nk_maxsim_sme_packed_header_t *)packed;
    header->originals_offset = 0;      // not used for f16
    header->original_stride = 0;       // not used for f16
    header->screen_weights_offset = 0; // not used for f16
    header->capability = nk_cap_sme_k;
    for (nk_size_t i = 0; i < 5; i++) header->reserved[i] = 0;

    // Convert squared norms → inverse norms in-place
    nk_f32_t *norms = (nk_f32_t *)((char *)packed + header->norms_offset);
    for (nk_size_t i = 0; i < columns; i++) {
        nk_f32_t norm_sq = norms[i];
        norms[i] = (norm_sq > 0.0f) ? (nk_f32_t)nk_f64_rsqrt_refined_neon_((nk_f64_t)norm_sq) : 0.0f;
    }
    return nk_success_k;
}
#endif // NUMKONG_TARGET_SME

/** Bytes of an F32 pack: I8 tiles, f64 inverse norms, screening weights, aligned originals. */
NUMKONG_INLINE nk_size_t nk_maxsim_pack_bytes_f32_sme_(nk_size_t columns, nk_size_t depth) {
    nk_size_t const expansion = 4;                    // i8 → i32 SMOPA
    nk_size_t const tile_dimension = nk_sme_cntw_();  // 16 for SVL=512
    nk_size_t const vector_elements = nk_sme_cntb_(); // 64 for SVL=512
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(depth * sizeof(nk_f32_t), 64);

    nk_size_t size = sizeof(nk_maxsim_sme_packed_header_t);         // 64 B header
    size += column_tile_count * depth_step_count * vector_elements; // i8 tiles
    size += columns * sizeof(nk_f64_t);                             // f64 inverse norms
    size += columns * sizeof(nk_f32_t);                             // f32 screening weights
    size += columns * original_stride;                              // f32 originals
    return size;
}

#if NUMKONG_TARGET_SME
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) { //
    *bytes = nk_maxsim_pack_bytes_f32_sme_(columns, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_sme(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                       void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_sme_packed_header_t const *header = (nk_maxsim_sme_packed_header_t const *)packed;
    if (header->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    *vectors = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_sme( //
    nk_f32_t const *vectors, nk_size_t columns, nk_size_t depth, nk_size_t stride, void *packed,
    void *stream) { //
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const blob_bytes = nk_maxsim_pack_bytes_f32_sme_(columns, depth);
    for (nk_size_t byte_index = 0; byte_index < blob_bytes; byte_index++) ((char *)packed)[byte_index] = 0;

    nk_size_t const expansion = 4;                    // i8 → i32 SMOPA
    nk_size_t const tile_dimension = nk_sme_cntw_();  // 16 for SVL=512
    nk_size_t const vector_elements = nk_sme_cntb_(); // 64 for SVL=512

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(depth * sizeof(nk_f32_t), 64);

    // Set up header
    nk_maxsim_sme_packed_header_t *header = (nk_maxsim_sme_packed_header_t *)packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)columns;
    header->depth = (nk_u32_t)depth;
    header->svl_bytes = (nk_u32_t)(tile_dimension * sizeof(nk_f32_t));

    nk_size_t const tiles_size = total_vectors * vector_elements;
    nk_size_t const norms_offset = sizeof(nk_maxsim_sme_packed_header_t) + tiles_size;
    nk_size_t const screen_weights_offset = norms_offset + columns * sizeof(nk_f64_t);
    nk_size_t const originals_offset = screen_weights_offset + columns * sizeof(nk_f32_t);

    header->norms_offset = (nk_u32_t)norms_offset;
    header->originals_offset = (nk_u32_t)originals_offset;
    header->original_stride = (nk_u32_t)original_stride;
    header->screen_weights_offset = (nk_u32_t)screen_weights_offset;
    header->capability = nk_cap_sme_k;
    for (nk_size_t i = 0; i < 5; i++) header->reserved[i] = 0;

    nk_i8_t *tiles = (nk_i8_t *)((char *)packed + sizeof(nk_maxsim_sme_packed_header_t));
    nk_f64_t *inverse_norms = (nk_f64_t *)((char *)packed + norms_offset);
    nk_f32_t *screen_weights = (nk_f32_t *)((char *)packed + screen_weights_offset);
    char *originals = (char *)packed + originals_offset;

    // Zero-initialize tile data (partial vectors stay zero-padded)
    for (nk_size_t i = 0; i < tiles_size; i++) tiles[i] = 0;

    // For each vector: quantize metadata, quantize+interleave into tiles, copy originals
    for (nk_size_t vector_index = 0; vector_index < columns; vector_index++) {
        nk_f32_t const *source = (nk_f32_t const *)((char const *)vectors + vector_index * stride);

        // Pass 1: Compute absmax and norm_sq simultaneously
        nk_f32_t absmax = 0.0f;
        nk_f64_t norm_sq = 0.0;
        for (nk_size_t dim = 0; dim < depth; dim++) {
            nk_f32_t val = source[dim];
            nk_f32_t abs_val = nk_f32_abs_(val);
            if (abs_val > absmax) absmax = abs_val;
            norm_sq += (nk_f64_t)val * val;
        }
        inverse_norms[vector_index] = norm_sq > 0.0 ? nk_f64_rsqrt_refined_neon_(norm_sq) : 0.0;

        nk_f32_t scale = absmax / 127.0f;
        if (scale == 0.0f) scale = 1.0f;
        screen_weights[vector_index] = scale * (nk_f32_t)inverse_norms[vector_index];

        // Pass 2: Quantize and scatter into tile-interleaved positions
        nk_size_t const column_tile = vector_index / tile_dimension;
        nk_size_t const column_in_tile = vector_index % tile_dimension;

        for (nk_size_t dim = 0; dim < depth; dim++) {
            nk_size_t const depth_step = dim / expansion;
            nk_size_t const sub_element = dim % expansion;
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_size_t const offset = vec_index * vector_elements + expansion * column_in_tile + sub_element;

            nk_f32_t scaled = source[dim] / scale;
            nk_i32_t quantized;
            if (scaled >= 0.0f) quantized = (nk_i32_t)(scaled + 0.5f);
            else quantized = (nk_i32_t)(scaled - 0.5f);
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;

            tiles[offset] = (nk_i8_t)quantized;
        }

        // Pass 3: Copy originals (64B-aligned stride, zero-pad tail)
        char *dest_original = originals + vector_index * original_stride;
        nk_copy_bytes_(dest_original, source, depth * sizeof(nk_f32_t));
        for (nk_size_t byte = depth * sizeof(nk_f32_t); byte < original_stride; byte++) dest_original[byte] = 0;
    }
    return nk_success_k;
}
#endif // NUMKONG_TARGET_SME

/** Streaming-compatible f32 dot product with f64 accumulation, following the svcntd() stride and
 *  svcvt_f64_f32_x widening of @c nk_dots_reduce_sumsq_f32_ssve_. */
NUMKONG_INLINE nk_f64_t nk_maxsim_reduce_dot_f32_ssve_(                         //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t count) NUMKONG_STREAMING_ { //
    svfloat64_t accumulator_even_f64x = svdup_f64(0.0);
    svfloat64_t accumulator_odd_f64x = svdup_f64(0.0);
    nk_size_t const vector_length = svcntw();
    // Lanes past `count` load as zeros, so every widened lane may accumulate.
    svbool_t const widened_b64x = svptrue_b64();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b32x = svwhilelt_b32_u64(i, count);
        svfloat32_t a_f32x = svld1_f32(predicate_b32x, a + i);
        svfloat32_t b_f32x = svld1_f32(predicate_b32x, b + i);

        svfloat64_t a_even_f64x = svcvt_f64_f32_x(widened_b64x, a_f32x);
        svfloat64_t b_even_f64x = svcvt_f64_f32_x(widened_b64x, b_f32x);
        accumulator_even_f64x = svmla_f64_m(widened_b64x, accumulator_even_f64x, a_even_f64x, b_even_f64x);

        svfloat64_t a_odd_f64x = svcvtlt_f64_f32_x(widened_b64x, a_f32x);
        svfloat64_t b_odd_f64x = svcvtlt_f64_f32_x(widened_b64x, b_f32x);
        accumulator_odd_f64x = svmla_f64_m(widened_b64x, accumulator_odd_f64x, a_odd_f64x, b_odd_f64x);
    }
    return nk_svaddv_f64_(svptrue_b64(), accumulator_even_f64x) + nk_svaddv_f64_(svptrue_b64(), accumulator_odd_f64x);
}

/** What the f32 refinement reads besides the screened columns. */
typedef struct {
    nk_f32_t const *query_originals;
    nk_f32_t const *document_originals;
    nk_size_t query_stride_elements;
    nk_size_t document_stride_elements;
    nk_size_t depth;
    nk_f64_t const *query_inverse_norms;
    nk_f64_t const *document_inverse_norms;
    nk_f32_t const *document_screen_weights;
} nk_maxsim_refine_f32_sme_t;

/** Raises each query's lower bound with a column, as @c nk_maxsim_screen_lower_bound_ does. */
NUMKONG_INLINE svfloat32_t nk_maxsim_fold_column_sme_(                          //
    svfloat32_t lower_bounds_f32x, svint32_t dots_i32x, nk_f32_t screen_weight, //
    svfloat32_t error_bases_f32x, svfloat32_t error_per_weights_f32x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svfloat32_t scores_f32x = svmul_n_f32_x(predicate_all_b32x, svcvt_f32_s32_x(predicate_all_b32x, dots_i32x),
                                            screen_weight);
    svfloat32_t errors_f32x = svmla_n_f32_x(predicate_all_b32x, error_bases_f32x, error_per_weights_f32x,
                                            screen_weight);
    return svmaxnm_f32_x(predicate_all_b32x, lower_bounds_f32x,
                         svsub_f32_x(predicate_all_b32x, scores_f32x, errors_f32x));
}

/** Refines against @p document_index every query whose screened score plus error reaches its lower
 *  bound, as @c nk_maxsim_screen_candidates_ selects, keeping each query's best cosine. */
NUMKONG_INLINE void nk_maxsim_refine_column_sme_(                                                    //
    nk_maxsim_refine_f32_sme_t const *refine, svint32_t dots_i32x, nk_size_t document_index,         //
    svfloat32_t error_bases_f32x, svfloat32_t error_per_weights_f32x, svfloat32_t lower_bounds_f32x, //
    svbool_t row_predicate_b32x, nk_size_t row_start, nk_f64_t *best_cosines) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    nk_f32_t const screen_weight = refine->document_screen_weights[document_index];
    svfloat32_t scores_f32x = svmul_n_f32_x(predicate_all_b32x, svcvt_f32_s32_x(predicate_all_b32x, dots_i32x),
                                            screen_weight);
    svfloat32_t highs_f32x = svadd_f32_x(
        predicate_all_b32x, scores_f32x,
        svmla_n_f32_x(predicate_all_b32x, error_bases_f32x, error_per_weights_f32x, screen_weight));
    svbool_t candidates_b32x = svcmpge_f32(row_predicate_b32x, highs_f32x, lower_bounds_f32x);
    if (!svptest_any(row_predicate_b32x, candidates_b32x)) return;

    nk_u32_t candidate_flags[64]; // max tile_dimension across all SVL values
    svst1_u32(predicate_all_b32x, candidate_flags, svdup_n_u32_z(candidates_b32x, 1));
    nk_f32_t const *document_original = refine->document_originals + document_index * refine->document_stride_elements;
    for (nk_size_t row_in_tile = 0; row_in_tile < svcntw(); row_in_tile++) {
        if (!candidate_flags[row_in_tile]) continue;
        nk_size_t const query_index = row_start + row_in_tile;
        nk_f64_t const cosine =
            nk_maxsim_reduce_dot_f32_ssve_(refine->query_originals + query_index * refine->query_stride_elements,
                                           document_original, refine->depth) *
            refine->query_inverse_norms[query_index] * refine->document_inverse_norms[document_index];
        if (cosine > best_cosines[row_in_tile]) best_cosines[row_in_tile] = cosine;
    }
}

/**
 *  @brief MaxSim f32 kernel: i8 SMOPA screening, then f32 and f64 refinement into angular distance.
 *
 *  Screening uses i8 SMOPA with an expansion of 4, covering 4× more depth per instruction than f32
 *  FMOPA. With 4 ZA tiles the fast path processes 64 document columns per iteration.
 *
 *  Each 4-tile group is read from ZA twice: once to raise every query's lower bound, then to refine
 *  in f64 every (query, document) pair the screen cannot rule out, so the compensated sum of
 *  angular distances 1 − dot / (‖q‖ · ‖d‖) matches an exhaustive search.
 */
__arm_new("za") static void nk_maxsim_packed_f32_streaming_( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t residue, nk_f64_t *result) NUMKONG_STREAMING_ {

    nk_maxsim_sme_packed_header_t const *query_header = (nk_maxsim_sme_packed_header_t const *)query_packed;
    nk_maxsim_sme_packed_header_t const *document_header = (nk_maxsim_sme_packed_header_t const *)document_packed;

    nk_size_t const depth_step_count = query_header->depth_tile_count;
    nk_size_t const query_row_tiles = query_header->column_tile_count;
    nk_size_t const document_col_tiles = document_header->column_tile_count;

    nk_size_t const tile_dimension = svcntw();  // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcntb(); // 64: i8 elements per SVE vector

    // Tile data pointers (i8)
    nk_i8_t const *query_tiles = (nk_i8_t const *)((char const *)query_packed + sizeof(nk_maxsim_sme_packed_header_t));
    nk_i8_t const *document_tiles = (nk_i8_t const *)((char const *)document_packed +
                                                      sizeof(nk_maxsim_sme_packed_header_t));

    nk_f32_t const *query_screen_weights = (nk_f32_t const *)((char const *)query_packed +
                                                              query_header->screen_weights_offset);
    nk_f32_t const *document_screen_weights = (nk_f32_t const *)((char const *)document_packed +
                                                                 document_header->screen_weights_offset);
    nk_maxsim_refine_f32_sme_t refine;
    refine.query_originals = (nk_f32_t const *)((char const *)query_packed + query_header->originals_offset);
    refine.document_originals = (nk_f32_t const *)((char const *)document_packed + document_header->originals_offset);
    refine.query_stride_elements = query_header->original_stride / sizeof(nk_f32_t);
    refine.document_stride_elements = document_header->original_stride / sizeof(nk_f32_t);
    refine.depth = depth;
    refine.query_inverse_norms = (nk_f64_t const *)((char const *)query_packed + query_header->norms_offset);
    refine.document_inverse_norms = (nk_f64_t const *)((char const *)document_packed + document_header->norms_offset);
    refine.document_screen_weights = document_screen_weights;

    nk_size_t const expansion = 4; // i8 → i32 SMOPA

    svbool_t const predicate_all_b8x = svptrue_b8();
    svbool_t const predicate_all_b32x = svptrue_b32();

    nk_f64_t total_angular_distance_f64 = 0.0, total_compensation_f64 = 0.0;

    for (nk_size_t row_tile_index = 0; row_tile_index < query_row_tiles; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= query_count) ? tile_dimension
                                                                                     : (query_count - row_start);
        svbool_t const row_predicate_b8x = (rows_remaining == tile_dimension)
                                               ? svptrue_b8()
                                               : svwhilelt_b8_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_b32x = (rows_remaining == tile_dimension) ? svptrue_b32()
                                                                               : svwhilelt_b32_u64(0u, rows_remaining);

        nk_f32_t error_bases[64], error_per_weights[64]; // max tile_dimension across all SVL values
        nk_f64_t best_cosines[64];
        for (nk_size_t row_in_tile = 0; row_in_tile < tile_dimension; row_in_tile++) {
            nk_maxsim_screen_error_t error = {0.0f, 0.0f};
            if (row_in_tile < rows_remaining)
                error = nk_maxsim_screen_error_(query_screen_weights[row_start + row_in_tile], residue, depth);
            error_bases[row_in_tile] = error.base;
            error_per_weights[row_in_tile] = error.per_weight;
            best_cosines[row_in_tile] = NUMKONG_F32_MIN;
        }
        svfloat32_t const error_bases_f32x = svld1_f32(predicate_all_b32x, error_bases);
        svfloat32_t const error_per_weights_f32x = svld1_f32(predicate_all_b32x, error_per_weights);
        svfloat32_t lower_bounds_f32x = svdup_f32(NUMKONG_F32_MIN);

        nk_size_t column_tile_index = 0;

        // 4-tile fast path: ZA0-ZA3 process 4 document column tiles simultaneously
        for (; column_tile_index + 4 <= document_col_tiles; column_tile_index += 4) {
            svzero_za();

            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svint8_t query_packed_i8x = svld1_s8(
                    row_predicate_b8x,
                    (nk_i8_t const *)(query_tiles +
                                      (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svint8_t document_packed_0_i8x = svld1_s8(
                    predicate_all_b8x,
                    (nk_i8_t const *)(document_tiles +
                                      ((column_tile_index + 0) * depth_step_count + depth_step) * vector_elements));
                svint8_t document_packed_1_i8x = svld1_s8(
                    predicate_all_b8x,
                    (nk_i8_t const *)(document_tiles +
                                      ((column_tile_index + 1) * depth_step_count + depth_step) * vector_elements));
                svint8_t document_packed_2_i8x = svld1_s8(
                    predicate_all_b8x,
                    (nk_i8_t const *)(document_tiles +
                                      ((column_tile_index + 2) * depth_step_count + depth_step) * vector_elements));
                svint8_t document_packed_3_i8x = svld1_s8(
                    predicate_all_b8x,
                    (nk_i8_t const *)(document_tiles +
                                      ((column_tile_index + 3) * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_s8_m(0, row_predicate_b8x, predicate_all_b8x, query_packed_i8x, document_packed_0_i8x);
                svmopa_za32_s8_m(1, row_predicate_b8x, predicate_all_b8x, query_packed_i8x, document_packed_1_i8x);
                svmopa_za32_s8_m(2, row_predicate_b8x, predicate_all_b8x, query_packed_i8x, document_packed_2_i8x);
                svmopa_za32_s8_m(3, row_predicate_b8x, predicate_all_b8x, query_packed_i8x, document_packed_3_i8x);
            }

            // Vertical column reads, manually unrolled over 4 tiles, first raising the lower bounds
            nk_size_t const first_column = column_tile_index * tile_dimension;
            nk_size_t const last_col_start = (column_tile_index + 3) * tile_dimension;
            nk_size_t const last_columns_remaining = (last_col_start + tile_dimension <= document_count)
                                                         ? tile_dimension
                                                         : (document_count - last_col_start);
            for (nk_size_t column_within_tile = 0; column_within_tile < tile_dimension; column_within_tile++) {
                nk_size_t const document_index = first_column + column_within_tile;
                lower_bounds_f32x = nk_maxsim_fold_column_sme_(
                    lower_bounds_f32x, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 0, column_within_tile),
                    document_screen_weights[document_index + 0 * tile_dimension], error_bases_f32x,
                    error_per_weights_f32x);
                lower_bounds_f32x = nk_maxsim_fold_column_sme_(
                    lower_bounds_f32x, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 1, column_within_tile),
                    document_screen_weights[document_index + 1 * tile_dimension], error_bases_f32x,
                    error_per_weights_f32x);
                lower_bounds_f32x = nk_maxsim_fold_column_sme_(
                    lower_bounds_f32x, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 2, column_within_tile),
                    document_screen_weights[document_index + 2 * tile_dimension], error_bases_f32x,
                    error_per_weights_f32x);
                // Tile 3 may end in zero-padded columns
                if (column_within_tile < last_columns_remaining)
                    lower_bounds_f32x = nk_maxsim_fold_column_sme_(
                        lower_bounds_f32x,
                        svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 3, column_within_tile),
                        document_screen_weights[document_index + 3 * tile_dimension], error_bases_f32x,
                        error_per_weights_f32x);
            }
            // ... then refining every pair they cannot rule out
            for (nk_size_t column_within_tile = 0; column_within_tile < tile_dimension; column_within_tile++) {
                nk_size_t const document_index = first_column + column_within_tile;
                nk_maxsim_refine_column_sme_(
                    &refine, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 0, column_within_tile),
                    document_index + 0 * tile_dimension, error_bases_f32x, error_per_weights_f32x, lower_bounds_f32x,
                    row_predicate_b32x, row_start, best_cosines);
                nk_maxsim_refine_column_sme_(
                    &refine, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 1, column_within_tile),
                    document_index + 1 * tile_dimension, error_bases_f32x, error_per_weights_f32x, lower_bounds_f32x,
                    row_predicate_b32x, row_start, best_cosines);
                nk_maxsim_refine_column_sme_(
                    &refine, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 2, column_within_tile),
                    document_index + 2 * tile_dimension, error_bases_f32x, error_per_weights_f32x, lower_bounds_f32x,
                    row_predicate_b32x, row_start, best_cosines);
                if (column_within_tile < last_columns_remaining)
                    nk_maxsim_refine_column_sme_(
                        &refine, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 3, column_within_tile),
                        document_index + 3 * tile_dimension, error_bases_f32x, error_per_weights_f32x,
                        lower_bounds_f32x, row_predicate_b32x, row_start, best_cosines);
            }
        }

        // 1-tile remainder: ZA0 only
        for (; column_tile_index < document_col_tiles; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (col_start + tile_dimension <= document_count)
                                                    ? tile_dimension
                                                    : (document_count - col_start);
            svbool_t const column_predicate_b8x = (columns_remaining == tile_dimension)
                                                      ? svptrue_b8()
                                                      : svwhilelt_b8_u64(0u, columns_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_0_k);

            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svint8_t query_packed_i8x = svld1_s8(
                    row_predicate_b8x,
                    (nk_i8_t const *)(query_tiles +
                                      (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svint8_t document_packed_i8x = svld1_s8(
                    column_predicate_b8x,
                    (nk_i8_t const *)(document_tiles +
                                      (column_tile_index * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_s8_m(0, row_predicate_b8x, column_predicate_b8x, query_packed_i8x, document_packed_i8x);
            }

            for (nk_size_t column_within_tile = 0; column_within_tile < columns_remaining; column_within_tile++)
                lower_bounds_f32x = nk_maxsim_fold_column_sme_(
                    lower_bounds_f32x, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 0, column_within_tile),
                    document_screen_weights[col_start + column_within_tile], error_bases_f32x, error_per_weights_f32x);
            for (nk_size_t column_within_tile = 0; column_within_tile < columns_remaining; column_within_tile++)
                nk_maxsim_refine_column_sme_(
                    &refine, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 0, column_within_tile),
                    col_start + column_within_tile, error_bases_f32x, error_per_weights_f32x, lower_bounds_f32x,
                    row_predicate_b32x, row_start, best_cosines);
        }

        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
            nk_f64_t angular = 1.0 - best_cosines[row_in_tile];
            if (angular < 0.0) angular = 0.0;
            nk_f64_dot2_(&total_angular_distance_f64, &total_compensation_f64, angular, 1.0);
        }
    }

    *result = total_angular_distance_f64 + total_compensation_f64;
}

#if NUMKONG_TARGET_SME
NUMKONG_API nk_status_t nk_maxsim_packed_f32_sme( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_maxsim_sme_packed_header_t const *)query_packed)->capability != nk_cap_sme_k ||
        ((nk_maxsim_sme_packed_header_t const *)document_packed)->capability != nk_cap_sme_k)
        return nk_pack_mismatch_k;

    nk_f32_t const residue = 0.5f * nk_f32_sqrt_((nk_f32_t)depth);
    nk_sme_start_streaming_();
    nk_maxsim_packed_f32_streaming_(query_packed, document_packed, query_count, document_count, depth, residue, result);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
#endif // NUMKONG_TARGET_SME

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_SME_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_MAXSIM_SME_H
