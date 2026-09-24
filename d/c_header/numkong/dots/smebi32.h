/**
 *  @brief SIMD-accelerated Batched Dot Products for SME (u1 binary vectors).
 *  @file include/numkong/dots/smebi32.h
 *  @author Ash Vardanian
 *  @date February 24, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses ARM SME BMOPA instruction for binary dot products:
 *    BMOPA gives: matching = popcount(XNOR(a,b))
 *    dot(a,b) = popcount(a AND b) = (pop_a + pop_b - depth + matching) / 2
 */
#ifndef NK_DOTS_SMEBI32_H
#define NK_DOTS_SMEBI32_H

#if NK_TARGET_ARM_
#if NK_TARGET_SMEBI32

#include "numkong/types.h"
#include "numkong/dots/sme.h"     // nk_sme_zero_za32_* constants
#include "numkong/sets/smebi32.h" // nk_dots_packed_size_u1_smebi32, nk_dots_pack_u1_smebi32

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme2,sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme2")
#endif

/**
 *  SME u1 dot-product kernel using ZA transpose for unpacked A.
 *  ZA0.S = staging (A rows loaded horizontally, read vertically for BMOPA).
 *  ZA1-3.S = BMOPA accumulation (3 B column tiles in fast path).
 *
 *  BMOPA gives matching = popcount(XNOR(a,b)).
 *  dot(a,b) = popcount(a AND b) = (pop_a + pop_b - depth_bits + matching) / 2
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_u1_smebi32_streaming_(
    nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t row_count_a, nk_size_t row_count_b,
    nk_size_t depth_bits, nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {

    nk_sets_smebi32_packed_header_t const *header = (nk_sets_smebi32_packed_header_t const *)b_packed;
    nk_size_t const row_tile_count_b = header->row_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const depth_u32_total = nk_size_divide_round_up_(depth_bits, 32);

    nk_u32_t const *b_tiles = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_sets_smebi32_packed_header_t));
    nk_u32_t const *b_norms = header->norms_offset ? (nk_u32_t const *)((char const *)b_packed + header->norms_offset)
                                                   : (nk_u32_t const *)0;

    svbool_t const predicate_all_u32x = svptrue_b32();
    svuint32_t const depth_u32x = svdup_u32((nk_u32_t)depth_bits);
    nk_size_t const depth_in_bytes = nk_size_divide_round_up_(depth_bits, 8);
    nk_size_t const row_tile_count_a = nk_size_divide_round_up_(row_count_a, tile_dim);

    for (nk_size_t row_tile_a = 0; row_tile_a < row_tile_count_a; row_tile_a++) {
        nk_size_t const row_start_a = row_tile_a * tile_dim;
        nk_size_t const rows_a_remaining = (row_start_a + tile_dim <= row_count_a) ? tile_dim
                                                                                   : (row_count_a - row_start_a);
        svbool_t const row_predicate_u32x = svwhilelt_b32_u64(0u, rows_a_remaining);

        // Compute A row popcounts for this tile
        nk_u32_t a_popcounts[16];
        for (nk_size_t r = 0; r < rows_a_remaining; r++) {
            nk_u1x8_t const *a_row = (nk_u1x8_t const *)((char const *)a + (row_start_a + r) * a_stride_in_bytes);
            a_popcounts[r] = nk_sets_reduce_sumsq_u1_streaming_(a_row, depth_in_bytes);
        }

        // Fast path: 3 B column tiles using ZA1-ZA3 (ZA0.S = staging)
        nk_size_t row_tile_b = 0;
        for (; row_tile_b + 3 <= row_tile_count_b; row_tile_b += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_u32x = svwhilelt_b32_u64(0u, u32s_this_tile);

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_a_remaining; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)a +
                                                                   (row_start_a + row_in_tile) * a_stride_in_bytes) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_u32x, a_row_u32);
                }

                nk_u32_t const *b_tile0 = b_tiles + ((row_tile_b + 0) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile1 = b_tiles + ((row_tile_b + 1) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile2 = b_tiles + ((row_tile_b + 2) * depth_tile_count + d_tile) * tile_elements;

                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_u32x, 0, step);

                    svbmopa_za32_u32_m(1, row_predicate_u32x, predicate_all_u32x, a_column_u32x,
                                       svld1_u32(predicate_all_u32x, b_tile0 + step * tile_dim));
                    svbmopa_za32_u32_m(2, row_predicate_u32x, predicate_all_u32x, a_column_u32x,
                                       svld1_u32(predicate_all_u32x, b_tile1 + step * tile_dim));
                    svbmopa_za32_u32_m(3, row_predicate_u32x, predicate_all_u32x, a_column_u32x,
                                       svld1_u32(predicate_all_u32x, b_tile2 + step * tile_dim));
                }
            }

            // Extract: dot = (pop_a + pop_b - depth + matching) / 2
            // matching = ZA[i][j]
            svuint32_t b_pop0_u32x = svld1_u32(predicate_all_u32x, b_norms + (row_tile_b + 0) * tile_dim);
            svuint32_t b_pop1_u32x = svld1_u32(predicate_all_u32x, b_norms + (row_tile_b + 1) * tile_dim);
            svuint32_t b_pop2_u32x = svld1_u32(predicate_all_u32x, b_norms + (row_tile_b + 2) * tile_dim);

            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);
                svuint32_t pop_a_u32x = svdup_u32(a_popcounts[row]);

                svuint32_t za1_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_u32x, 1, row);
                svuint32_t sum_pops0_u32x = svadd_u32_x(predicate_all_u32x, pop_a_u32x, b_pop0_u32x);
                svuint32_t numerator0_u32x = svadd_u32_x(
                    predicate_all_u32x, svsub_u32_x(predicate_all_u32x, sum_pops0_u32x, depth_u32x), za1_u32x);
                svst1_u32(predicate_all_u32x, c_row + (row_tile_b + 0) * tile_dim,
                          svlsr_n_u32_x(predicate_all_u32x, numerator0_u32x, 1));

                svuint32_t za2_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_u32x, 2, row);
                svuint32_t sum_pops1_u32x = svadd_u32_x(predicate_all_u32x, pop_a_u32x, b_pop1_u32x);
                svuint32_t numerator1_u32x = svadd_u32_x(
                    predicate_all_u32x, svsub_u32_x(predicate_all_u32x, sum_pops1_u32x, depth_u32x), za2_u32x);
                svst1_u32(predicate_all_u32x, c_row + (row_tile_b + 1) * tile_dim,
                          svlsr_n_u32_x(predicate_all_u32x, numerator1_u32x, 1));

                svuint32_t za3_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_u32x, 3, row);
                svuint32_t sum_pops2_u32x = svadd_u32_x(predicate_all_u32x, pop_a_u32x, b_pop2_u32x);
                svuint32_t numerator2_u32x = svadd_u32_x(
                    predicate_all_u32x, svsub_u32_x(predicate_all_u32x, sum_pops2_u32x, depth_u32x), za3_u32x);
                svst1_u32(predicate_all_u32x, c_row + (row_tile_b + 2) * tile_dim,
                          svlsr_n_u32_x(predicate_all_u32x, numerator2_u32x, 1));
            }
        }

        // Remainder: 1 B column tile at a time using ZA1
        for (; row_tile_b < row_tile_count_b; row_tile_b++) {
            nk_size_t const row_start_b = row_tile_b * tile_dim;
            nk_size_t const rows_b_remaining = (row_start_b + tile_dim <= row_count_b) ? tile_dim
                                                                                       : (row_count_b - row_start_b);
            svbool_t const column_predicate_u32x = svwhilelt_b32_u64(0u, rows_b_remaining);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_u32x = svwhilelt_b32_u64(0u, u32s_this_tile);

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_a_remaining; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)a +
                                                                   (row_start_a + row_in_tile) * a_stride_in_bytes) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_u32x, a_row_u32);
                }

                nk_u32_t const *b_tile = b_tiles + (row_tile_b * depth_tile_count + d_tile) * tile_elements;

                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_u32x, 0, step);
                    svuint32_t b_u32x = svld1_u32(predicate_all_u32x, b_tile + step * tile_dim);
                    svbmopa_za32_u32_m(1, row_predicate_u32x, column_predicate_u32x, a_column_u32x, b_u32x);
                }
            }

            // Extract: dot = (pop_a + pop_b - depth + matching) / 2
            svuint32_t b_pop_u32x = svld1_u32(predicate_all_u32x, b_norms + row_start_b);
            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                svuint32_t za1_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_u32x, 1, row);
                svuint32_t pop_a_u32x = svdup_u32(a_popcounts[row]);
                svuint32_t sum_pops_u32x = svadd_u32_x(predicate_all_u32x, pop_a_u32x, b_pop_u32x);
                svuint32_t numerator_u32x = svadd_u32_x(
                    predicate_all_u32x, svsub_u32_x(predicate_all_u32x, sum_pops_u32x, depth_u32x), za1_u32x);
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);
                svst1_u32(column_predicate_u32x, c_row + row_start_b,
                          svlsr_n_u32_x(predicate_all_u32x, numerator_u32x, 1));
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_u1_smebi32(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t row_count_a,
                                         nk_size_t row_count_b, nk_size_t depth_bits, nk_size_t a_stride_in_bytes,
                                         nk_size_t c_stride_in_bytes) {
    nk_dots_packed_u1_smebi32_streaming_(a, b_packed, c, row_count_a, row_count_b, depth_bits, a_stride_in_bytes,
                                         c_stride_in_bytes);
}

/**
 *  Symmetric u1 dot-product using ZA0 time-sharing + 3-tile fast path.
 *  Same ZA transpose pattern as hammings_symmetric, but with dot extraction.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_u1_smebi32_streaming_(
    nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t depth_bits, nk_size_t stride, nk_u32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    nk_size_t const depth_u32_total = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_u32_total, depth_tile_size);
    nk_size_t const depth_in_bytes = nk_size_divide_round_up_(depth_bits, 8);

    svbool_t const predicate_all_u32x = svptrue_b32();
    svuint32_t const depth_u32x = svdup_u32((nk_u32_t)depth_bits);

    NK_ALIGN64 nk_u32_t a_buffer[16][16]; // Stack buffer for A column save

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dim);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dim) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dim <= row_end) ? tile_dim : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_u32x = svwhilelt_b32_u64(0u, rows_clamped);

        // Compute A tile popcounts
        NK_ALIGN64 nk_u32_t a_tile_pops[16];
        for (nk_size_t r = 0; r < rows_clamped; r++) {
            nk_u1x8_t const *a_row = (nk_u1x8_t const *)((char const *)vectors + (row_tile_start + r) * stride);
            a_tile_pops[r] = nk_sets_reduce_sumsq_u1_streaming_(a_row, depth_in_bytes);
        }
        for (nk_size_t r = rows_clamped; r < tile_dim; r++) a_tile_pops[r] = 0;

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_u32x = svwhilelt_b32_u64(0u, u32s_this_tile);

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)vectors +
                                                                   (row_tile_start + row_in_tile) * stride) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_u32x, a_row_u32);
                }

                // Save A columns
                for (nk_size_t s = 0; s < u32s_this_tile; s++)
                    svst1_u32(predicate_all_u32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_u32x, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_u32x, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_u32x = svld1_u32(predicate_all_u32x, a_buffer[step]);
                    svuint32_t b_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_u32x, 0, step);
                    svbmopa_za32_u32_m(1, row_predicate_u32x, predicate_all_u32x, a_u32x, b_u32x);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_u32x, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_u32x = svld1_u32(predicate_all_u32x, a_buffer[step]);
                    svuint32_t b_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_u32x, 0, step);
                    svbmopa_za32_u32_m(2, row_predicate_u32x, predicate_all_u32x, a_u32x, b_u32x);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_u32x, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_u32x = svld1_u32(predicate_all_u32x, a_buffer[step]);
                    svuint32_t b_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_u32x, 0, step);
                    svbmopa_za32_u32_m(3, row_predicate_u32x, predicate_all_u32x, a_u32x, b_u32x);
                }
            }

            // Extract: dot = (pop_a + pop_b - depth + matching) / 2
            // Compute B tile popcounts
            NK_ALIGN64 nk_u32_t b_pops[3][16];
            for (nk_size_t t = 0; t < 3; t++) {
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + t) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u1x8_t const *b_row = (nk_u1x8_t const *)((char const *)vectors + col_abs * stride);
                        b_pops[t][col] = nk_sets_reduce_sumsq_u1_streaming_(b_row, depth_in_bytes);
                    }
                    else { b_pops[t][col] = 0; }
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_u32_t *result_row = (nk_u32_t *)((char *)result + (row_tile_start + row) * result_stride);
                svuint32_t pop_a_u32x = svdup_u32(a_tile_pops[row]);

                svuint32_t za1_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_u32x, 1, row);
                svuint32_t b_popcount_0_u32x = svld1_u32(predicate_all_u32x, b_pops[0]);
                svuint32_t sum_pops0_u32x = svadd_u32_x(predicate_all_u32x, pop_a_u32x, b_popcount_0_u32x);
                svuint32_t numerator0_u32x = svadd_u32_x(
                    predicate_all_u32x, svsub_u32_x(predicate_all_u32x, sum_pops0_u32x, depth_u32x), za1_u32x);
                svst1_u32(predicate_all_u32x, result_row + (column_tile_index + 0) * tile_dim,
                          svlsr_n_u32_x(predicate_all_u32x, numerator0_u32x, 1));

                svuint32_t za2_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_u32x, 2, row);
                svuint32_t b_popcount_1_u32x = svld1_u32(predicate_all_u32x, b_pops[1]);
                svuint32_t sum_pops1_u32x = svadd_u32_x(predicate_all_u32x, pop_a_u32x, b_popcount_1_u32x);
                svuint32_t numerator1_u32x = svadd_u32_x(
                    predicate_all_u32x, svsub_u32_x(predicate_all_u32x, sum_pops1_u32x, depth_u32x), za2_u32x);
                svst1_u32(predicate_all_u32x, result_row + (column_tile_index + 1) * tile_dim,
                          svlsr_n_u32_x(predicate_all_u32x, numerator1_u32x, 1));

                svuint32_t za3_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_u32x, 3, row);
                svuint32_t b_popcount_2_u32x = svld1_u32(predicate_all_u32x, b_pops[2]);
                svuint32_t sum_pops2_u32x = svadd_u32_x(predicate_all_u32x, pop_a_u32x, b_popcount_2_u32x);
                svuint32_t numerator2_u32x = svadd_u32_x(
                    predicate_all_u32x, svsub_u32_x(predicate_all_u32x, sum_pops2_u32x, depth_u32x), za3_u32x);
                svst1_u32(predicate_all_u32x, result_row + (column_tile_index + 2) * tile_dim,
                          svlsr_n_u32_x(predicate_all_u32x, numerator2_u32x, 1));
            }
        }

        // Remainder: 1 column tile at a time using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_tile_start = column_tile_index * tile_dim;
            nk_size_t const cols_remaining = (col_tile_start + tile_dim <= n_vectors) ? tile_dim
                                                                                      : (n_vectors - col_tile_start);
            svbool_t const column_predicate_u32x = svwhilelt_b32_u64(0u, cols_remaining);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_u32x = svwhilelt_b32_u64(0u, u32s_this_tile);

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)vectors +
                                                                   (row_tile_start + row_in_tile) * stride) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_u32x, a_row_u32);
                }

                for (nk_size_t s = 0; s < u32s_this_tile; s++)
                    svst1_u32(predicate_all_u32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_u32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_u32x, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_u32x = svld1_u32(predicate_all_u32x, a_buffer[step]);
                    svuint32_t b_u32x = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_u32x, 0, step);
                    svbmopa_za32_u32_m(1, row_predicate_u32x, column_predicate_u32x, a_u32x, b_u32x);
                }
            }

            // Compute B tile popcounts for remainder
            NK_ALIGN64 nk_u32_t b_pops_r[16];
            for (nk_size_t col = 0; col < tile_dim; col++) {
                nk_size_t const col_abs = col_tile_start + col;
                if (col_abs < n_vectors) {
                    nk_u1x8_t const *b_row = (nk_u1x8_t const *)((char const *)vectors + col_abs * stride);
                    b_pops_r[col] = nk_sets_reduce_sumsq_u1_streaming_(b_row, depth_in_bytes);
                }
                else { b_pops_r[col] = 0; }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                svuint32_t za1_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_u32x, 1, row);
                svuint32_t pop_a_u32x = svdup_u32(a_tile_pops[row]);
                svuint32_t b_popcount_u32x = svld1_u32(predicate_all_u32x, b_pops_r);
                svuint32_t sum_pops_u32x = svadd_u32_x(predicate_all_u32x, pop_a_u32x, b_popcount_u32x);
                svuint32_t numerator_u32x = svadd_u32_x(
                    predicate_all_u32x, svsub_u32_x(predicate_all_u32x, sum_pops_u32x, depth_u32x), za1_u32x);
                nk_u32_t *result_row = (nk_u32_t *)((char *)result + (row_tile_start + row) * result_stride);
                svst1_u32(column_predicate_u32x, result_row + col_tile_start,
                          svlsr_n_u32_x(predicate_all_u32x, numerator_u32x, 1));
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t depth_bits,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {
    nk_dots_symmetric_u1_smebi32_streaming_(vectors, n_vectors, depth_bits, stride, result, result_stride, row_start,
                                            row_count);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SMEBI32
#endif // NK_TARGET_ARM_
#endif // NK_DOTS_SMEBI32_H
