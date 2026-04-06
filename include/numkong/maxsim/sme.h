/**
 *  @brief SIMD-accelerated MaxSim (ColBERT late-interaction) for SME.
 *  @file include/numkong/maxsim/sme.h
 *  @author Ash Vardanian
 *  @date February 10, 2026
 *
 *  Computes MaxSim(Q, D) = Σᵢ maxⱼ dot(qᵢ, dⱼ) using ARM SME outer products.
 *
 *  Both Q and D are pre-packed with `nk_dots_pack_bf16_sme` from `dots/sme.h`.
 *  This frees all 4 ZA tiles for accumulation (vs 3 with A-side staging).
 *
 *  Key optimization: vertical column reads for max reduction.
 *  Traditional extraction reads tile rows then calls `svmaxv` (horizontal max, ~8cy).
 *  Our approach reads tile columns with `svread_ver_za32_f32_m`:
 *
 *    - Each column read gives dot products of all query tokens vs one doc token.
 *    - Element-wise `svmax` (~1cy) updates a running max vector across doc tokens.
 *    - Only `svaddv` at the very end: ⌈n_q/16⌉ = 2 horizontal reductions total.
 *
 *  This is ~100x fewer horizontal reductions for typical ColBERT dimensions.
 *
 *  ZA tile layout after BFMOPA accumulation (16x16 f32):
 *
 *  - Row i, Column j = dot(q_{tile_row_start + i}, d_{tile_col_start + j})
 *  - Vertical column read of column j → similarities of all 16 q tokens to doc token j
 *  - Element-wise max across columns → per-query-token max over doc tokens in this tile group
 *
 *  Benchmark results (Apple M4, SVL=512):
 *
 *      Dimensions              dots_packed GEMM    maxsim fused    GEMM speedup    End-to-end speedup
 *      32×128×128 (ColBERT)    840 GFLOPS          1516 GFLOPS     1.81×           5.10×
 *      32×256×128              1037 GFLOPS         1591 GFLOPS     1.53×           5.17×
 *      64×512×128              1016 GFLOPS         1651 GFLOPS     1.62×           5.42×
 *      32×128×256              859 GFLOPS          1725 GFLOPS     2.01×           4.06×
 *      32×1024×768 (BERT)      1124 GFLOPS         1932 GFLOPS     1.72×           2.61×
 *
 *  Speedup sources:
 *
 *  1. Pre-packing both sides → 4 ZA tiles for accumulation (vs 3 with A-staging): +33% MOPA throughput
 *  2. No output matrix materialization → eliminates M×N f32 memory round-trip
 *  3. Vertical column reads → ~128 element-wise svmax (1cy) vs ~256 svmaxv horizontal reductions (8cy)
 */
#ifndef NK_MAXSIM_SME_H
#define NK_MAXSIM_SME_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SME

#include "numkong/dots/sme.h" // nk_dots_sme_packed_header_t, nk_dots_pack_{f16,bf16}_sme, nk_dots_packed_size_{f16,bf16}_sme

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
 *  Packed header for MaxSim SME kernels. Used by f32 (i8 screening + f32 refinement)
 *  and bf16/f16 (BFMOPA/FMOPA + angular normalization) kernels.
 *
 *  For f32: stores i8 tile-interleaved data, f32 squared norms, AND f32 originals.
 *  For bf16/f16: stores tile-interleaved data and f32 inverse norms (1/||v||).
 *    originals_offset and original_stride are 0 (unused).
 */
typedef struct {
    nk_u32_t column_tile_count; // ceil(n / tile_dimension)
    nk_u32_t depth_tile_count;  // ceil(depth / expansion)
    nk_u32_t columns;           // actual vector count (for predicates)
    nk_u32_t depth;             // actual depth
    nk_u32_t svl_bytes;         // SVL in bytes at pack time (validation)
    nk_u32_t norms_offset;      // byte offset -> per-vector norms (squared for f32, inverse for bf16/f16)
    nk_u32_t originals_offset;  // byte offset -> f32 original vectors (0 for bf16/f16)
    nk_u32_t original_stride;   // row stride in bytes for originals (64B-aligned, 0 for bf16/f16)
    nk_u32_t reserved[8];       // padding to 64 bytes
} nk_maxsim_sme_packed_header_t;

NK_STATIC_ASSERT(sizeof(nk_maxsim_sme_packed_header_t) == 64, nk_maxsim_sme_packed_header_must_be_64_bytes);

/**
 *  MaxSim f16 kernel: both Q and D pre-packed, vertical column read extraction.
 *
 *  4-tile fast path: processes 4 doc column tiles simultaneously using ZA0-ZA3.
 *  Inner loop per depth_step: 1 Q load + 4 D loads + 4 FMOPA = 9 ops.
 *  Extraction per 4-tile group: 4×16 = 64 vertical reads + 64 svmax = ~128 cycles.
 *
 *  1-tile remainder: uses ZA0 only, with predicated loads for partial tiles.
 */
__arm_locally_streaming __arm_new("za") static void nk_maxsim_packed_f16_streaming_( //
    void const *query_packed, void const *document_packed,                           //
    nk_size_t query_count, nk_size_t document_count,                                 //
    nk_size_t depth, nk_f32_t *result) {

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

        // Running max + argmax vectors for angular distance finalization
        svfloat32_t running_maximum_f32x = svdup_f32(NK_F32_MIN);
        svuint32_t running_argmax_u32x = svdup_u32(0);

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

            // Vertical column extraction + argmax update (manually unrolled over 4 tiles)
            for (nk_size_t column_within_tile = 0; column_within_tile < tile_dimension; column_within_tile++) {
                // Tile 0
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 0) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 0,
                                                                         column_within_tile);
                    svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                    running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
                // Tile 1
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 1) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 1,
                                                                         column_within_tile);
                    svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                    running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
                // Tile 2
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 2) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 2,
                                                                         column_within_tile);
                    svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                    running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
                // Tile 3
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 3) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 3,
                                                                         column_within_tile);
                    svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                    running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
            }
        }

        // Remainder: 1 doc column tile at a time using ZA0 only
        for (; column_tile_index < document_col_tiles; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= document_count)
                                                 ? tile_dimension
                                                 : (document_count - col_start);
            svbool_t const column_predicate_b16x = (cols_remaining == tile_dimension)
                                                       ? svptrue_b16()
                                                       : svwhilelt_b16_u64(0u, cols_remaining * 2);

            svzero_mask_za(nk_sme_zero_za32_tile_0_); // Zero ZA0 only

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

            // Vertical column extraction from ZA0 + argmax update
            for (nk_size_t column_within_tile = 0; column_within_tile < cols_remaining; column_within_tile++) {
                nk_u32_t document_index = (nk_u32_t)(col_start + column_within_tile);
                svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 0,
                                                                     column_within_tile);
                svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
            }
        }

        // Angular distance finalization — SVE-width vector ops
        // Gather document inverse norms via argmax indices (no SVE gather in streaming mode)
        nk_u32_t best_document_indices[64];
        nk_f32_t document_inverse_norms_gathered[64];
        svst1_u32(row_predicate_b32x, best_document_indices, running_argmax_u32x);
        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++)
            document_inverse_norms_gathered[row_in_tile] = document_inverse_norms[best_document_indices[row_in_tile]];

        // SVE-width: cosine = dot * inv_norm_q * inv_norm_d, angular = max(1 - cosine, 0)
        svfloat32_t query_inverse_norms_f32x = svld1_f32(row_predicate_b32x, query_inverse_norms + row_start);
        svfloat32_t document_inverse_norms_f32x = svld1_f32(row_predicate_b32x, document_inverse_norms_gathered);
        svfloat32_t cosine_f32x = svmul_f32_x(
            row_predicate_b32x, svmul_f32_x(row_predicate_b32x, running_maximum_f32x, query_inverse_norms_f32x),
            document_inverse_norms_f32x);
        svfloat32_t angular_distance_f32x = svmax_f32_x(
            row_predicate_b32x, svsub_f32_x(row_predicate_b32x, svdup_f32(1.0f), cosine_f32x), svdup_f32(0.0f));
        total_angular_distance += svaddv_f32(row_predicate_b32x, angular_distance_f32x);
        NK_UNPOISON(&total_angular_distance, sizeof(total_angular_distance));
    }

    *result = total_angular_distance;
}

NK_PUBLIC void nk_maxsim_packed_f16_sme(                              //
    void const *query_packed, void const *document_packed,            //
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth, //
    nk_f32_t *result) {                                               //

    nk_maxsim_packed_f16_streaming_(query_packed, document_packed, query_count, document_count, depth, result);
}

/**
 *  MaxSim bf16 kernel: both Q and D pre-packed, vertical column read extraction.
 *
 *  4-tile fast path: processes 4 doc column tiles simultaneously using ZA0-ZA3.
 *  Inner loop per depth_step: 1 Q load + 4 D loads + 4 BFMOPA = 9 ops.
 *  Extraction per 4-tile group: 4×16 = 64 vertical reads + 64 svmax = ~128 cycles.
 *
 *  1-tile remainder: uses ZA0 only, with predicated loads for partial tiles.
 */
__arm_locally_streaming __arm_new("za") static void nk_maxsim_packed_bf16_streaming_( //
    void const *query_packed, void const *document_packed,                            //
    nk_size_t query_count, nk_size_t document_count,                                  //
    nk_size_t depth, nk_f32_t *result) {

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

        // Running max + argmax vectors for angular distance finalization
        svfloat32_t running_maximum_f32x = svdup_f32(NK_F32_MIN);
        svuint32_t running_argmax_u32x = svdup_u32(0);

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

            // Vertical column extraction + argmax update (manually unrolled over 4 tiles)
            for (nk_size_t column_within_tile = 0; column_within_tile < tile_dimension; column_within_tile++) {
                // Tile 0
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 0) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 0,
                                                                         column_within_tile);
                    svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                    running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
                // Tile 1
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 1) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 1,
                                                                         column_within_tile);
                    svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                    running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
                // Tile 2
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 2) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 2,
                                                                         column_within_tile);
                    svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                    running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
                // Tile 3
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 3) * tile_dimension + column_within_tile);
                    svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 3,
                                                                         column_within_tile);
                    svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                    running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
            }
        }

        // Remainder: 1 doc column tile at a time using ZA0 only
        for (; column_tile_index < document_col_tiles; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= document_count)
                                                 ? tile_dimension
                                                 : (document_count - col_start);
            svbool_t const column_predicate_b16x = (cols_remaining == tile_dimension)
                                                       ? svptrue_b16()
                                                       : svwhilelt_b16_u64(0u, cols_remaining * 2);

            svzero_mask_za(nk_sme_zero_za32_tile_0_); // Zero ZA0 only

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

            // Vertical column extraction from ZA0 + argmax update
            for (nk_size_t column_within_tile = 0; column_within_tile < cols_remaining; column_within_tile++) {
                nk_u32_t document_index = (nk_u32_t)(col_start + column_within_tile);
                svfloat32_t column_dots_f32x = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), predicate_all_b32x, 0,
                                                                     column_within_tile);
                svbool_t is_better_bx = svcmpgt_f32(predicate_all_b32x, column_dots_f32x, running_maximum_f32x);
                running_maximum_f32x = svsel_f32(is_better_bx, column_dots_f32x, running_maximum_f32x);
                running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
            }
        }

        // Angular distance finalization — SVE-width vector ops
        // Gather document inverse norms via argmax indices (no SVE gather in streaming mode)
        nk_u32_t best_document_indices[64];
        nk_f32_t document_inverse_norms_gathered[64];
        svst1_u32(row_predicate_b32x, best_document_indices, running_argmax_u32x);
        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++)
            document_inverse_norms_gathered[row_in_tile] = document_inverse_norms[best_document_indices[row_in_tile]];

        // SVE-width: cosine = dot * inv_norm_q * inv_norm_d, angular = max(1 - cosine, 0)
        svfloat32_t query_inverse_norms_f32x = svld1_f32(row_predicate_b32x, query_inverse_norms + row_start);
        svfloat32_t document_inverse_norms_f32x = svld1_f32(row_predicate_b32x, document_inverse_norms_gathered);
        svfloat32_t cosine_f32x = svmul_f32_x(
            row_predicate_b32x, svmul_f32_x(row_predicate_b32x, running_maximum_f32x, query_inverse_norms_f32x),
            document_inverse_norms_f32x);
        svfloat32_t angular_distance_f32x = svmax_f32_x(
            row_predicate_b32x, svsub_f32_x(row_predicate_b32x, svdup_f32(1.0f), cosine_f32x), svdup_f32(0.0f));
        total_angular_distance += svaddv_f32(row_predicate_b32x, angular_distance_f32x);
        NK_UNPOISON(&total_angular_distance, sizeof(total_angular_distance));
    }

    *result = total_angular_distance;
}

NK_PUBLIC void nk_maxsim_packed_bf16_sme(                             //
    void const *query_packed, void const *document_packed,            //
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth, //
    nk_f32_t *result) {                                               //

    nk_maxsim_packed_bf16_streaming_(query_packed, document_packed, query_count, document_count, depth, result);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_sme(nk_size_t columns, nk_size_t depth) { //
    return nk_dots_packed_size_bf16_sme(columns, depth);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_sme(nk_size_t columns, nk_size_t depth) { //
    return nk_dots_packed_size_f16_sme(columns, depth);
}

NK_PUBLIC void nk_maxsim_pack_bf16_sme(                                                                      //
    nk_bf16_t const *vectors, nk_size_t columns, nk_size_t depth, nk_size_t stride_in_bytes, void *packed) { //

    // Delegate tile interleaving and squared norms computation to dots pack.
    // Both headers are 64 bytes with identical layout for the first 6 fields.
    nk_dots_pack_bf16_sme(vectors, columns, depth, stride_in_bytes, packed);

    // Set maxsim-specific header fields (overlaps dots reserved area)
    nk_maxsim_sme_packed_header_t *header = (nk_maxsim_sme_packed_header_t *)packed;
    header->originals_offset = 0; // not used for bf16
    header->original_stride = 0;  // not used for bf16
    for (nk_size_t i = 0; i < 8; i++) header->reserved[i] = 0;

    // Convert squared norms → inverse norms in-place
    nk_f32_t *norms = (nk_f32_t *)((char *)packed + header->norms_offset);
    for (nk_size_t i = 0; i < columns; i++) {
        nk_f32_t norm_sq = norms[i];
        norms[i] = (norm_sq > 0.0f) ? (nk_f32_t)nk_f64_rsqrt_neon((nk_f64_t)norm_sq) : 0.0f;
    }
}

NK_PUBLIC void nk_maxsim_pack_f16_sme(                                                                      //
    nk_f16_t const *vectors, nk_size_t columns, nk_size_t depth, nk_size_t stride_in_bytes, void *packed) { //

    // Delegate tile interleaving and squared norms computation to dots pack.
    // Both headers are 64 bytes with identical layout for the first 6 fields.
    nk_dots_pack_f16_sme(vectors, columns, depth, stride_in_bytes, packed);

    // Set maxsim-specific header fields (overlaps dots reserved area)
    nk_maxsim_sme_packed_header_t *header = (nk_maxsim_sme_packed_header_t *)packed;
    header->originals_offset = 0; // not used for f16
    header->original_stride = 0;  // not used for f16
    for (nk_size_t i = 0; i < 8; i++) header->reserved[i] = 0;

    // Convert squared norms → inverse norms in-place
    nk_f32_t *norms = (nk_f32_t *)((char *)packed + header->norms_offset);
    for (nk_size_t i = 0; i < columns; i++) {
        nk_f32_t norm_sq = norms[i];
        norms[i] = (norm_sq > 0.0f) ? (nk_f32_t)nk_f64_rsqrt_neon((nk_f64_t)norm_sq) : 0.0f;
    }
}

/**
 *  MaxSim f32 kernel: i8 SMOPA screening + f32/f64 refinement + angular distance.
 *
 *  Screening: i8 SMOPA has expansion=4, processing 4x more depth per instruction than f32 FMOPA.
 *  With 4 ZA tiles the fast path processes 64 document columns per iteration.
 *
 *  Refinement: tile-wide interleaved f64 dot products for the winning (query, document) pairs.
 *  Angular distance: 1 - dot / sqrt(||q||^2 * ||d||^2), accumulated with f64.
 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_sme(nk_size_t columns, nk_size_t depth) { //
    nk_size_t const expansion = 4;                                                      // i8->i32 SMOPA
    nk_size_t const tile_dimension = nk_sme_cntw_();                                    // 16 for SVL=512
    nk_size_t const vector_elements = nk_sme_cntb_();                                   // 64 for SVL=512
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(depth * sizeof(nk_f32_t), 64);

    nk_size_t size = sizeof(nk_maxsim_sme_packed_header_t);         // 64 B header
    size += column_tile_count * depth_step_count * vector_elements; // i8 tiles
    size += columns * sizeof(nk_f32_t);                             // f32 squared norms
    size += columns * original_stride;                              // f32 originals
    return size;
}

NK_PUBLIC void nk_maxsim_pack_f32_sme(                                                                      //
    nk_f32_t const *vectors, nk_size_t columns, nk_size_t depth, nk_size_t stride_in_bytes, void *packed) { //

    nk_size_t const expansion = 4;                    // i8->i32 SMOPA
    nk_size_t const tile_dimension = nk_sme_cntw_();  // 16 for SVL=512
    nk_size_t const vector_elements = nk_sme_cntb_(); // 64 for SVL=512
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f32_t);

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
    nk_size_t const originals_offset = norms_offset + columns * sizeof(nk_f32_t);

    header->norms_offset = (nk_u32_t)norms_offset;
    header->originals_offset = (nk_u32_t)originals_offset;
    header->original_stride = (nk_u32_t)original_stride;
    for (nk_size_t i = 0; i < 8; i++) header->reserved[i] = 0;

    nk_i8_t *tiles = (nk_i8_t *)((char *)packed + sizeof(nk_maxsim_sme_packed_header_t));
    nk_f32_t *norms = (nk_f32_t *)((char *)packed + norms_offset);
    char *originals = (char *)packed + originals_offset;

    // Zero-initialize tile data (partial vectors stay zero-padded)
    for (nk_size_t i = 0; i < tiles_size; i++) tiles[i] = 0;

    // For each vector: quantize metadata, quantize+interleave into tiles, copy originals
    for (nk_size_t vector_index = 0; vector_index < columns; vector_index++) {
        nk_f32_t const *source = (nk_f32_t const *)((char const *)vectors + vector_index * stride_in_bytes);

        // Pass 1: Compute absmax and norm_sq simultaneously
        nk_f32_t absmax = 0.0f;
        nk_f32_t norm_sq = 0.0f;
        for (nk_size_t dim = 0; dim < depth; dim++) {
            nk_f32_t val = source[dim];
            nk_f32_t abs_val = nk_f32_abs_(val);
            if (abs_val > absmax) absmax = abs_val;
            norm_sq += val * val;
        }
        norms[vector_index] = norm_sq;

        nk_f32_t scale = absmax / 127.0f;
        if (scale == 0.0f) scale = 1.0f;

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
}

/**
 *  Streaming-compatible f32 dot product with f64 accumulation.
 *  Follows the svcntd()-stride + svcvt_f64_f32_x pattern from nk_dots_reduce_sumsq_f32_ssve_.
 */
NK_PUBLIC nk_f64_t nk_maxsim_reduce_dot_f32_ssve_(                         //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t count) NK_STREAMING_ { //
    svfloat64_t accumulator_even_f64x = svdup_f64(0.0);
    svfloat64_t accumulator_odd_f64x = svdup_f64(0.0);
    nk_size_t const vector_length = svcntw();
    nk_size_t const half_vector_length = svcntd();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b32x = svwhilelt_b32_u64(i, count);
        svfloat32_t a_f32x = svld1_f32(predicate_b32x, a + i);
        svfloat32_t b_f32x = svld1_f32(predicate_b32x, b + i);

        svbool_t predicate_even_b64x = svwhilelt_b64_u64(i, count);
        svfloat64_t a_even_f64x = svcvt_f64_f32_x(predicate_even_b64x, a_f32x);
        svfloat64_t b_even_f64x = svcvt_f64_f32_x(predicate_even_b64x, b_f32x);
        accumulator_even_f64x = svmla_f64_m(predicate_even_b64x, accumulator_even_f64x, a_even_f64x, b_even_f64x);

        svbool_t predicate_odd_b64x = svwhilelt_b64_u64(i + half_vector_length, count);
        svfloat64_t a_odd_f64x = svcvtlt_f64_f32_x(predicate_odd_b64x, a_f32x);
        svfloat64_t b_odd_f64x = svcvtlt_f64_f32_x(predicate_odd_b64x, b_f32x);
        accumulator_odd_f64x = svmla_f64_m(predicate_odd_b64x, accumulator_odd_f64x, a_odd_f64x, b_odd_f64x);
    }
    nk_f64_t sum_even = svaddv_f64(svptrue_b64(), accumulator_even_f64x);
    nk_f64_t sum_odd = svaddv_f64(svptrue_b64(), accumulator_odd_f64x);
    NK_UNPOISON(&sum_even, sizeof(sum_even));
    NK_UNPOISON(&sum_odd, sizeof(sum_odd));
    return sum_even + sum_odd;
}

/**
 *  MaxSim f32 kernel: i8 SMOPA screening + f32/f64 refinement + angular distance.
 *
 *  Screening: i8 SMOPA has expansion=4, processing 4x more depth per instruction than f32 FMOPA.
 *  With 4 ZA tiles the fast path processes 64 document columns per iteration.
 *
 *  Refinement: tile-wide interleaved f64 dot products for the winning (query, document) pairs.
 *  Angular distance: 1 - dot / sqrt(||q||^2 * ||d||^2), accumulated with f64.
 */
__arm_locally_streaming __arm_new("za") static void nk_maxsim_packed_f32_streaming_( //
    void const *query_packed, void const *document_packed,                           //
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,                //
    nk_f64_t *result) {

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

    // Norms and originals pointers
    nk_f32_t const *query_norms = (nk_f32_t const *)((char const *)query_packed + query_header->norms_offset);
    nk_f32_t const *document_norms = (nk_f32_t const *)((char const *)document_packed + document_header->norms_offset);
    nk_f32_t const *query_originals = (nk_f32_t const *)((char const *)query_packed + query_header->originals_offset);
    nk_f32_t const *document_originals = (nk_f32_t const *)((char const *)document_packed +
                                                            document_header->originals_offset);
    nk_size_t const query_original_stride_elements = query_header->original_stride / sizeof(nk_f32_t);
    nk_size_t const document_original_stride_elements = document_header->original_stride / sizeof(nk_f32_t);

    nk_size_t const expansion = 4; // i8->i32 SMOPA

    svbool_t const predicate_all_b8x = svptrue_b8();
    svbool_t const predicate_all_b32x = svptrue_b32();

    nk_f64_t total_angular_distance_f64 = 0.0;

    for (nk_size_t row_tile_index = 0; row_tile_index < query_row_tiles; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= query_count) ? tile_dimension
                                                                                     : (query_count - row_start);
        svbool_t const row_predicate_b8x = (rows_remaining == tile_dimension)
                                               ? svptrue_b8()
                                               : svwhilelt_b8_u64(0u, rows_remaining * expansion);
        svbool_t const row_predicate_b32x = (rows_remaining == tile_dimension) ? svptrue_b32()
                                                                               : svwhilelt_b32_u64(0u, rows_remaining);

        svint32_t running_max_i32x = svdup_s32(NK_I32_MIN);
        svuint32_t running_argmax_u32x = svdup_u32(0);

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

            // Vertical column extraction + argmax update (manually unrolled over 4 tiles)
            for (nk_size_t column_within_tile = 0; column_within_tile < tile_dimension; column_within_tile++) {
                // Tile 0
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 0) * tile_dimension + column_within_tile);
                    svint32_t column_dots_i32x = svread_ver_za32_s32_m(svdup_s32(NK_I32_MIN), predicate_all_b32x, 0,
                                                                       column_within_tile);
                    svbool_t is_better_bx = svcmpgt_s32(predicate_all_b32x, column_dots_i32x, running_max_i32x);
                    running_max_i32x = svsel_s32(is_better_bx, column_dots_i32x, running_max_i32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
                // Tile 1
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 1) * tile_dimension + column_within_tile);
                    svint32_t column_dots_i32x = svread_ver_za32_s32_m(svdup_s32(NK_I32_MIN), predicate_all_b32x, 1,
                                                                       column_within_tile);
                    svbool_t is_better_bx = svcmpgt_s32(predicate_all_b32x, column_dots_i32x, running_max_i32x);
                    running_max_i32x = svsel_s32(is_better_bx, column_dots_i32x, running_max_i32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
                // Tile 2
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 2) * tile_dimension + column_within_tile);
                    svint32_t column_dots_i32x = svread_ver_za32_s32_m(svdup_s32(NK_I32_MIN), predicate_all_b32x, 2,
                                                                       column_within_tile);
                    svbool_t is_better_bx = svcmpgt_s32(predicate_all_b32x, column_dots_i32x, running_max_i32x);
                    running_max_i32x = svsel_s32(is_better_bx, column_dots_i32x, running_max_i32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
                // Tile 3
                {
                    nk_u32_t document_index = (nk_u32_t)((column_tile_index + 3) * tile_dimension + column_within_tile);
                    svint32_t column_dots_i32x = svread_ver_za32_s32_m(svdup_s32(NK_I32_MIN), predicate_all_b32x, 3,
                                                                       column_within_tile);
                    svbool_t is_better_bx = svcmpgt_s32(predicate_all_b32x, column_dots_i32x, running_max_i32x);
                    running_max_i32x = svsel_s32(is_better_bx, column_dots_i32x, running_max_i32x);
                    running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
                }
            }
        }

        // 1-tile remainder: ZA0 only
        for (; column_tile_index < document_col_tiles; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= document_count)
                                                 ? tile_dimension
                                                 : (document_count - col_start);
            svbool_t const column_predicate_b8x = (cols_remaining == tile_dimension)
                                                      ? svptrue_b8()
                                                      : svwhilelt_b8_u64(0u, cols_remaining * expansion);

            svzero_mask_za(nk_sme_zero_za32_tile_0_);

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

            for (nk_size_t column_within_tile = 0; column_within_tile < cols_remaining; column_within_tile++) {
                nk_u32_t document_index = (nk_u32_t)(col_start + column_within_tile);
                svint32_t column_dots_i32x = svread_ver_za32_s32_m(svdup_s32(NK_I32_MIN), predicate_all_b32x, 0,
                                                                   column_within_tile);
                svbool_t is_better_bx = svcmpgt_s32(predicate_all_b32x, column_dots_i32x, running_max_i32x);
                running_max_i32x = svsel_s32(is_better_bx, column_dots_i32x, running_max_i32x);
                running_argmax_u32x = svsel_u32(is_better_bx, svdup_u32(document_index), running_argmax_u32x);
            }
        }

        // Refinement: tile-wide interleaved f64 dot products
        nk_u32_t best_document_indices[64]; // max tile_dimension across all SVL values
        svst1_u32(row_predicate_b32x, best_document_indices, running_argmax_u32x);

        // Pointer setup: one (query, document) pair per row in the tile
        nk_f32_t const *query_original_ptrs[64];
        nk_f32_t const *document_original_ptrs[64];
        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
            nk_size_t query_index = row_start + row_in_tile;
            nk_u32_t best_document_index = best_document_indices[row_in_tile];
            query_original_ptrs[row_in_tile] = query_originals + query_index * query_original_stride_elements;
            document_original_ptrs[row_in_tile] = document_originals +
                                                  best_document_index * document_original_stride_elements;
        }

        // Interleaved f64 dot products in batches of 4 (hides MLA 4-cycle latency)
        nk_size_t row_batch_start = 0;

        // Fast path: 4-wide batches
        for (; row_batch_start + 4 <= rows_remaining; row_batch_start += 4) {
            svfloat64_t accumulator_0_f64x = svdup_f64(0.0);
            svfloat64_t accumulator_1_f64x = svdup_f64(0.0);
            svfloat64_t accumulator_2_f64x = svdup_f64(0.0);
            svfloat64_t accumulator_3_f64x = svdup_f64(0.0);
            nk_size_t const depth_vector_length = svcntw();
            nk_size_t const depth_half_length = svcntd();

            for (nk_size_t depth_index = 0; depth_index < depth; depth_index += depth_vector_length) {
                svbool_t predicate_depth_b32x = svwhilelt_b32_u64(depth_index, depth);
                svbool_t predicate_even_b64x = svwhilelt_b64_u64(depth_index, depth);
                svbool_t predicate_odd_b64x = svwhilelt_b64_u64(depth_index + depth_half_length, depth);

                svfloat32_t query_values_0_f32x = svld1_f32(predicate_depth_b32x,
                                                            query_original_ptrs[row_batch_start + 0] + depth_index);
                svfloat32_t document_values_0_f32x = svld1_f32(
                    predicate_depth_b32x, document_original_ptrs[row_batch_start + 0] + depth_index);
                accumulator_0_f64x = svmla_f64_m(predicate_even_b64x, accumulator_0_f64x,
                                                 svcvt_f64_f32_x(predicate_even_b64x, query_values_0_f32x),
                                                 svcvt_f64_f32_x(predicate_even_b64x, document_values_0_f32x));
                accumulator_0_f64x = svmla_f64_m(predicate_odd_b64x, accumulator_0_f64x,
                                                 svcvtlt_f64_f32_x(predicate_odd_b64x, query_values_0_f32x),
                                                 svcvtlt_f64_f32_x(predicate_odd_b64x, document_values_0_f32x));

                svfloat32_t query_values_1_f32x = svld1_f32(predicate_depth_b32x,
                                                            query_original_ptrs[row_batch_start + 1] + depth_index);
                svfloat32_t document_values_1_f32x = svld1_f32(
                    predicate_depth_b32x, document_original_ptrs[row_batch_start + 1] + depth_index);
                accumulator_1_f64x = svmla_f64_m(predicate_even_b64x, accumulator_1_f64x,
                                                 svcvt_f64_f32_x(predicate_even_b64x, query_values_1_f32x),
                                                 svcvt_f64_f32_x(predicate_even_b64x, document_values_1_f32x));
                accumulator_1_f64x = svmla_f64_m(predicate_odd_b64x, accumulator_1_f64x,
                                                 svcvtlt_f64_f32_x(predicate_odd_b64x, query_values_1_f32x),
                                                 svcvtlt_f64_f32_x(predicate_odd_b64x, document_values_1_f32x));

                svfloat32_t query_values_2_f32x = svld1_f32(predicate_depth_b32x,
                                                            query_original_ptrs[row_batch_start + 2] + depth_index);
                svfloat32_t document_values_2_f32x = svld1_f32(
                    predicate_depth_b32x, document_original_ptrs[row_batch_start + 2] + depth_index);
                accumulator_2_f64x = svmla_f64_m(predicate_even_b64x, accumulator_2_f64x,
                                                 svcvt_f64_f32_x(predicate_even_b64x, query_values_2_f32x),
                                                 svcvt_f64_f32_x(predicate_even_b64x, document_values_2_f32x));
                accumulator_2_f64x = svmla_f64_m(predicate_odd_b64x, accumulator_2_f64x,
                                                 svcvtlt_f64_f32_x(predicate_odd_b64x, query_values_2_f32x),
                                                 svcvtlt_f64_f32_x(predicate_odd_b64x, document_values_2_f32x));

                svfloat32_t query_values_3_f32x = svld1_f32(predicate_depth_b32x,
                                                            query_original_ptrs[row_batch_start + 3] + depth_index);
                svfloat32_t document_values_3_f32x = svld1_f32(
                    predicate_depth_b32x, document_original_ptrs[row_batch_start + 3] + depth_index);
                accumulator_3_f64x = svmla_f64_m(predicate_even_b64x, accumulator_3_f64x,
                                                 svcvt_f64_f32_x(predicate_even_b64x, query_values_3_f32x),
                                                 svcvt_f64_f32_x(predicate_even_b64x, document_values_3_f32x));
                accumulator_3_f64x = svmla_f64_m(predicate_odd_b64x, accumulator_3_f64x,
                                                 svcvtlt_f64_f32_x(predicate_odd_b64x, query_values_3_f32x),
                                                 svcvtlt_f64_f32_x(predicate_odd_b64x, document_values_3_f32x));
            }

            // Reduce accumulators and compute angular distance per row
            svfloat64_t *batch_accumulators[] = {&accumulator_0_f64x, &accumulator_1_f64x, &accumulator_2_f64x,
                                                 &accumulator_3_f64x};
            for (nk_size_t batch_index = 0; batch_index < 4; batch_index++) {
                nk_size_t query_index = row_start + row_batch_start + batch_index;
                nk_u32_t best_document_index = best_document_indices[row_batch_start + batch_index];
                nk_f64_t dot_product_f64 = svaddv_f64(svptrue_b64(), *batch_accumulators[batch_index]);
                NK_UNPOISON(&dot_product_f64, sizeof(dot_product_f64));
                nk_f64_t norm_product_f64 = (nk_f64_t)query_norms[query_index] *
                                            (nk_f64_t)document_norms[best_document_index];
                nk_f64_t cosine_f64 = (norm_product_f64 > 0.0) ? dot_product_f64 * nk_f64_rsqrt_serial(norm_product_f64)
                                                               : 0.0;
                nk_f64_t angular_distance_f64 = 1.0 - cosine_f64;
                if (angular_distance_f64 < 0.0) angular_distance_f64 = 0.0;
                total_angular_distance_f64 += angular_distance_f64;
            }
        }

        // Remainder: 1 row at a time
        for (; row_batch_start < rows_remaining; row_batch_start++) {
            nk_size_t query_index = row_start + row_batch_start;
            nk_u32_t best_document_index = best_document_indices[row_batch_start];
            nk_f64_t dot_product_f64 = nk_maxsim_reduce_dot_f32_ssve_(query_original_ptrs[row_batch_start],
                                                                      document_original_ptrs[row_batch_start], depth);
            nk_f64_t norm_product_f64 = (nk_f64_t)query_norms[query_index] *
                                        (nk_f64_t)document_norms[best_document_index];
            nk_f64_t cosine_f64 = (norm_product_f64 > 0.0) ? dot_product_f64 * nk_f64_rsqrt_serial(norm_product_f64)
                                                           : 0.0;
            nk_f64_t angular_distance_f64 = 1.0 - cosine_f64;
            if (angular_distance_f64 < 0.0) angular_distance_f64 = 0.0;
            total_angular_distance_f64 += angular_distance_f64;
        }
    }

    *result = total_angular_distance_f64;
}

NK_PUBLIC void nk_maxsim_packed_f32_sme(                              //
    void const *query_packed, void const *document_packed,            //
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth, //
    nk_f64_t *result) {                                               //

    nk_maxsim_packed_f32_streaming_(query_packed, document_packed, query_count, document_count, depth, result);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM64_
#endif // NK_MAXSIM_SME_H
