/**
 *  @file include/numkong/dots/smebi32.h
 *  @author Ash Vardanian
 *  @date February 24, 2026
 *  @brief SIMD-accelerated Batched Dot Products for SME, u1 binary vectors.
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses ARM SME BMOPA instruction for binary dot products.
 *
 *  @verbatim
 *  matching = popcount(XNOR(a, b))
 *  dot(a, b) = popcount(a AND b) = (pop_a + pop_b - depth + matching) / 2
 *  @endverbatim
 */
#ifndef NUMKONG_DOTS_SMEBI32_H
#define NUMKONG_DOTS_SMEBI32_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_SMEBI32

#include "numkong/types.h"
#include "numkong/reduce/sve.h" // `nk_svaddv_u32_`
#include "numkong/dots/sme.h"   // `nk_zero_za32_sme_*`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme2")
#endif

/*  Read SVL in bytes from non-streaming context using RDSVL instruction. */
NUMKONG_INLINE nk_size_t nk_svl_bytes_smebi32_(void) {
    nk_size_t svl_bytes;
    __asm__ volatile("rdsvl %0, #1" : "=r"(svl_bytes));
    return svl_bytes;
}

/*  Get ZA32 tile dimension (number of f32/u32 elements per row). */
NUMKONG_INLINE nk_size_t nk_tile_dim_smebi32_(void) { return nk_svl_bytes_smebi32_() / sizeof(nk_u32_t); }

typedef struct {

    /** Row tiles, ⌈rows / tile dimension⌉. */
    nk_u32_t row_tile_count;

    /** Depth tiles, ⌈depth bits / depth tile bits⌉. */
    nk_u32_t depth_tile_count;

    /** Rows, not padded. */
    nk_u32_t rows;

    /** Depth in bits, not padded. */
    nk_u32_t depth_bits;

    /** Streaming vector length in bytes at pack time, which every consumer validates. */
    nk_u32_t svl_bytes;

    /** Byte offset from the buffer start to the norms, or 0 without them. */
    nk_u32_t norms_offset;

    /** Zeroed; pads the header to 64 bytes. */
    nk_u32_t reserved[8];

    /** The capability that packed the buffer, which every consumer checks. */
    nk_capability_t capability;
} nk_dots_smebi32_packed_header_t;

/** Count total set bits across a byte vector using streaming SVE.
 *  Accumulates per-byte popcounts into u32 lanes via svdot; single horizontal reduction at end. */
NUMKONG_INLINE nk_u32_t nk_dots_reduce_sum_u1_smebi32_streaming_(nk_u1x8_t const *data,
                                                                 nk_size_t n_bytes) NUMKONG_STREAMING_ {
    svuint32_t accumulator_u32x = svdup_u32(0);
    svuint8_t const ones_u8x = svdup_u8(1);
    for (nk_size_t offset = 0; offset < n_bytes; offset += svcntb()) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(offset, n_bytes);
        accumulator_u32x = svdot_u32(accumulator_u32x,
                                     svcnt_u8_z(predicate_b8x, svld1_u8(predicate_b8x, data + offset)), ones_u8x);
    }
    return (nk_u32_t)nk_svaddv_u32_(svptrue_b32(), accumulator_u32x);
}

/** Packs U1 rows of a contiguous depth into tiles of 32-bit words, word-major across 16 rows, and
 *  their popcounts, for the row tiles starting from @p columns_begin up to @p columns_end: a batch
 *  of 16 words of every row goes into ZA0.S horizontally and leaves it vertically. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_pack_u1_smebi32_streaming_( //
    nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth_bits, nk_size_t b_row_stride, nk_u32_t *tiles,
    nk_u32_t *norms, nk_size_t columns_begin, nk_size_t columns_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_bytes = depth_bits / NUMKONG_BITS_PER_BYTE;
    nk_size_t const depth_words_padded = nk_size_round_up_to_multiple_(nk_size_divide_round_up_(depth_bits, 32),
                                                                       tile_dimension);
    nk_size_t const row_tile_end = nk_size_divide_round_up_(nk_min_of_two(columns_end, columns), tile_dimension);
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t row_tile = nk_size_divide_round_up_(columns_begin, tile_dimension); row_tile < row_tile_end;
         row_tile++) {
        nk_size_t const row_start = row_tile * tile_dimension;
        nk_size_t const tile_rows = nk_min_of_two(tile_dimension, columns - row_start);
        char const *source = (char const *)b + row_start * b_row_stride;
        nk_u32_t *packed = tiles + row_tile * depth_words_padded * tile_dimension;
        svbool_t const row_predicate_b32x = svwhilelt_b32_u64(0u, tile_rows);
        svuint32_t popcounts_u32x = svdup_n_u32(0);
        for (nk_size_t word = 0; word < depth_words_padded; word++) {
            nk_size_t const slice = word % tile_dimension;
            if (slice == 0)
                for (nk_size_t row = 0; row < tile_rows; row++)
                    svwrite_hor_za32_u32_m(
                        0, row, predicate_all_b32x,
                        svreinterpret_u32_u8(svld1_u8(svwhilelt_b8_u64(4 * word, depth_bytes),
                                                      (nk_u8_t const *)(source + row * b_row_stride) + 4 * word)));
            svuint32_t const words_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32x, 0, slice);
            svst1_u32(predicate_all_b32x, packed + word * tile_dimension, words_u32x);
            popcounts_u32x = svadd_u32_x(predicate_all_b32x, popcounts_u32x,
                                         svcnt_u32_x(predicate_all_b32x, words_u32x));
        }
        svst1_u32(row_predicate_b32x, norms + row_start, popcounts_u32x);
    }
}

NUMKONG_API nk_status_t nk_dots_pack_size_u1_smebi32(nk_size_t columns, nk_size_t depth_bits, nk_size_t *bytes) {
    nk_size_t const tile_dim = nk_tile_dim_smebi32_();        // 16 rows per tile
    nk_size_t const depth_tile_size = nk_tile_dim_smebi32_(); // 16 u32 per depth tile = 512 bits

    nk_size_t const depth_u32 = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const row_tile_count = nk_size_divide_round_up_(columns, tile_dim);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_u32, depth_tile_size);

    nk_size_t const tile_elements = tile_dim * depth_tile_size; // 256 u32 per tile
    nk_size_t size = sizeof(nk_dots_smebi32_packed_header_t);
    size += row_tile_count * depth_tile_count * tile_elements * sizeof(nk_u32_t);
    size += columns * sizeof(nk_u32_t); // per-row population counts

    *bytes = size;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_u1_smebi32(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_smebi32_packed_header_t const *header = (nk_dots_smebi32_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_smebi32_k) return nk_pack_mismatch_k;
    *columns = header->rows;
    *depth = header->depth_bits;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_u1_smebi32(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth_bits,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const svl_bytes = nk_svl_bytes_smebi32_();
    nk_size_t const tile_dim = nk_tile_dim_smebi32_();        // 16 rows per tile
    nk_size_t const depth_tile_size = nk_tile_dim_smebi32_(); // 16 u32 per depth tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    // BMOPA processes binary data in 32-bit words: each svbmopa_za32_u32_m step
    // handles one u32 (32 bits) across all row × column pairs simultaneously.
    nk_size_t const depth_words = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const row_tile_count = nk_size_divide_round_up_(columns, tile_dim);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_words, depth_tile_size);
    nk_size_t const total_tiles = row_tile_count * depth_tile_count;
    nk_size_t const data_size = total_tiles * tile_elements * sizeof(nk_u32_t);

    nk_size_t const norms_offset = sizeof(nk_dots_smebi32_packed_header_t) + data_size;
    nk_dots_smebi32_packed_header_t *header = (nk_dots_smebi32_packed_header_t *)b_packed;
    if (columns_begin == 0) {
        nk_u32_t *header_words = (nk_u32_t *)header;
        for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
            header_words[word_index] = 0;
        header->row_tile_count = (nk_u32_t)row_tile_count;
        header->depth_tile_count = (nk_u32_t)depth_tile_count;
        header->rows = (nk_u32_t)columns;
        header->depth_bits = (nk_u32_t)depth_bits;
        header->svl_bytes = (nk_u32_t)svl_bytes;
        header->capability = nk_cap_smebi32_k;
        header->norms_offset = (nk_u32_t)norms_offset;
    }

    nk_u32_t *tiles = (nk_u32_t *)((char *)b_packed + sizeof(nk_dots_smebi32_packed_header_t));
    nk_u32_t *norms = (nk_u32_t *)((char *)b_packed + norms_offset);
    nk_start_sme_streaming_();
    nk_dots_pack_u1_smebi32_streaming_(b, columns, depth_bits, b_stride, tiles, norms, columns_begin, columns_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

/**
 *  @brief SME u1 dot-product kernel using ZA transpose for unpacked A.
 *
 *  ZA0.S stages A rows, loaded horizontally and read vertically for BMOPA, while ZA1-3.S hold the
 *  BMOPA accumulation of 3 B column tiles in the fast path. BMOPA gives the matching bit count:
 *
 *  @verbatim
 *  matching  = popcount(XNOR(a, b))
 *  dot(a, b) = popcount(a AND b) = (pop_a + pop_b - depth_bits + matching) / 2
 *  @endverbatim
 */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_packed_u1_smebi32_streaming_( //
    nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t row_count_a, nk_size_t row_count_b,
    nk_size_t depth_bits, nk_size_t a_stride, nk_size_t c_stride) NUMKONG_STREAMING_ {

    nk_dots_smebi32_packed_header_t const *header = (nk_dots_smebi32_packed_header_t const *)b_packed;
    nk_size_t const row_tile_count_b = header->row_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    // BMOPA processes binary data in 32-bit words: each svbmopa_za32_u32_m step
    // handles one u32 (32 bits) across all row × column pairs simultaneously.
    nk_size_t const depth_words = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const depth_bytes = depth_bits / NUMKONG_BITS_PER_BYTE;

    nk_u32_t const *b_tiles = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_dots_smebi32_packed_header_t));
    nk_u32_t const *b_norms = header->norms_offset ? (nk_u32_t const *)((char const *)b_packed + header->norms_offset)
                                                   : (nk_u32_t const *)0;

    svbool_t const predicate_all_b32x = svptrue_b32();
    // Use padded depth (depth_words * 32) for BMOPA: zero-padded bits always match in XNOR,
    // so the effective depth for the matching→intersection conversion is the rounded-up bit count.
    svuint32_t const depth_u32x = svdup_u32((nk_u32_t)(depth_words * 32));
    nk_size_t const row_tile_count_a = nk_size_divide_round_up_(row_count_a, tile_dim);

    for (nk_size_t row_tile_a = 0; row_tile_a < row_tile_count_a; row_tile_a++) {
        nk_size_t const row_start_a = row_tile_a * tile_dim;
        nk_size_t const rows_a_remaining = (row_start_a + tile_dim <= row_count_a) ? tile_dim
                                                                                   : (row_count_a - row_start_a);
        svbool_t const row_predicate_b32x = svwhilelt_b32_u64(0u, rows_a_remaining);

        // Compute A row popcounts for this tile
        nk_u32_t a_popcounts[16];
        for (nk_size_t r = 0; r < rows_a_remaining; r++) {
            nk_u1x8_t const *a_row = (nk_u1x8_t const *)((char const *)a + (row_start_a + r) * a_stride);
            a_popcounts[r] = nk_dots_reduce_sum_u1_smebi32_streaming_(a_row, depth_bytes);
        }

        // Fast path: 3 B column tiles using ZA1-ZA3 (ZA0.S = staging)
        nk_size_t row_tile_b = 0;
        for (; row_tile_b + 3 <= row_tile_count_b; row_tile_b += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_k);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_words)
                                                     ? depth_tile_size
                                                     : (depth_words > d_start_u32 ? depth_words - d_start_u32 : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_k);

                svbool_t const batch_predicate_b32x = svwhilelt_b32_u64(0u, u32s_this_tile);

                svbool_t const depth_predicate_b8x = svwhilelt_b8_u64(d_start_u32 * 4, depth_bytes);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_a_remaining; row_in_tile++) {
                    nk_u8_t const *a_row = (nk_u8_t const *)a + (row_start_a + row_in_tile) * a_stride +
                                           d_start_u32 * 4;
                    svuint8_t row_u8x = svld1_u8(depth_predicate_b8x, a_row);
                    svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_b32x, svreinterpret_u32_u8(row_u8x));
                }

                nk_u32_t const *b_tile0 = b_tiles + ((row_tile_b + 0) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile1 = b_tiles + ((row_tile_b + 1) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile2 = b_tiles + ((row_tile_b + 2) * depth_tile_count + d_tile) * tile_elements;

                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32x, 0, step);

                    svbmopa_za32_u32_m(1, row_predicate_b32x, predicate_all_b32x, a_column_u32x,
                                       svld1_u32(predicate_all_b32x, b_tile0 + step * tile_dim));
                    svbmopa_za32_u32_m(2, row_predicate_b32x, predicate_all_b32x, a_column_u32x,
                                       svld1_u32(predicate_all_b32x, b_tile1 + step * tile_dim));
                    svbmopa_za32_u32_m(3, row_predicate_b32x, predicate_all_b32x, a_column_u32x,
                                       svld1_u32(predicate_all_b32x, b_tile2 + step * tile_dim));
                }
            }

            // Extract: dot = (pop_a + pop_b - depth + matching) / 2
            // matching = ZA[i][j]
            svbool_t const third_bound_b32x = svwhilelt_b32_u64((row_tile_b + 2) * tile_dim, row_count_b);
            svuint32_t b_pop0_u32x = svld1_u32(predicate_all_b32x, b_norms + (row_tile_b + 0) * tile_dim);
            svuint32_t b_pop1_u32x = svld1_u32(predicate_all_b32x, b_norms + (row_tile_b + 1) * tile_dim);
            svuint32_t b_pop2_u32x = svld1_u32(third_bound_b32x, b_norms + (row_tile_b + 2) * tile_dim);

            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride);
                svuint32_t pop_a_u32x = svdup_u32(a_popcounts[row]);

                svuint32_t za1_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_b32x, 1, row);
                svuint32_t sum_pops0_u32x = svadd_u32_x(predicate_all_b32x, pop_a_u32x, b_pop0_u32x);
                svuint32_t numerator0_u32x = svadd_u32_x(
                    predicate_all_b32x, svsub_u32_x(predicate_all_b32x, sum_pops0_u32x, depth_u32x), za1_u32x);
                svst1_u32(predicate_all_b32x, c_row + (row_tile_b + 0) * tile_dim,
                          svlsr_n_u32_x(predicate_all_b32x, numerator0_u32x, 1));

                svuint32_t za2_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_b32x, 2, row);
                svuint32_t sum_pops1_u32x = svadd_u32_x(predicate_all_b32x, pop_a_u32x, b_pop1_u32x);
                svuint32_t numerator1_u32x = svadd_u32_x(
                    predicate_all_b32x, svsub_u32_x(predicate_all_b32x, sum_pops1_u32x, depth_u32x), za2_u32x);
                svst1_u32(predicate_all_b32x, c_row + (row_tile_b + 1) * tile_dim,
                          svlsr_n_u32_x(predicate_all_b32x, numerator1_u32x, 1));

                svuint32_t za3_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_b32x, 3, row);
                svuint32_t sum_pops2_u32x = svadd_u32_x(predicate_all_b32x, pop_a_u32x, b_pop2_u32x);
                svuint32_t numerator2_u32x = svadd_u32_x(
                    predicate_all_b32x, svsub_u32_x(predicate_all_b32x, sum_pops2_u32x, depth_u32x), za3_u32x);
                svst1_u32(third_bound_b32x, c_row + (row_tile_b + 2) * tile_dim,
                          svlsr_n_u32_x(predicate_all_b32x, numerator2_u32x, 1));
            }
        }

        // Remainder: 1 B column tile at a time using ZA1
        for (; row_tile_b < row_tile_count_b; row_tile_b++) {
            nk_size_t const row_start_b = row_tile_b * tile_dim;
            nk_size_t const rows_b_remaining = (row_start_b + tile_dim <= row_count_b) ? tile_dim
                                                                                       : (row_count_b - row_start_b);
            svbool_t const column_predicate_b32x = svwhilelt_b32_u64(0u, rows_b_remaining);

            svzero_mask_za(nk_sme_zero_za32_tile_1_k);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_words)
                                                     ? depth_tile_size
                                                     : (depth_words > d_start_u32 ? depth_words - d_start_u32 : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_k);

                svbool_t const batch_predicate_b32x = svwhilelt_b32_u64(0u, u32s_this_tile);

                svbool_t const depth_predicate_b8x = svwhilelt_b8_u64(d_start_u32 * 4, depth_bytes);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_a_remaining; row_in_tile++) {
                    nk_u8_t const *a_row = (nk_u8_t const *)a + (row_start_a + row_in_tile) * a_stride +
                                           d_start_u32 * 4;
                    svuint8_t row_u8x = svld1_u8(depth_predicate_b8x, a_row);
                    svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_b32x, svreinterpret_u32_u8(row_u8x));
                }

                nk_u32_t const *b_tile = b_tiles + (row_tile_b * depth_tile_count + d_tile) * tile_elements;

                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_column_u32x = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32x, 0, step);
                    svuint32_t b_u32x = svld1_u32(predicate_all_b32x, b_tile + step * tile_dim);
                    svbmopa_za32_u32_m(1, row_predicate_b32x, column_predicate_b32x, a_column_u32x, b_u32x);
                }
            }

            // Extract: dot = (pop_a + pop_b - depth + matching) / 2
            svuint32_t b_pop_u32x = svld1_u32(column_predicate_b32x, b_norms + row_start_b);
            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                svuint32_t za1_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_b32x, 1, row);
                svuint32_t pop_a_u32x = svdup_u32(a_popcounts[row]);
                svuint32_t sum_pops_u32x = svadd_u32_x(predicate_all_b32x, pop_a_u32x, b_pop_u32x);
                svuint32_t numerator_u32x = svadd_u32_x(
                    predicate_all_b32x, svsub_u32_x(predicate_all_b32x, sum_pops_u32x, depth_u32x), za1_u32x);
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride);
                svst1_u32(column_predicate_b32x, c_row + row_start_b,
                          svlsr_n_u32_x(predicate_all_b32x, numerator_u32x, 1));
            }
        }
    }
}

NUMKONG_API nk_status_t nk_dots_packed_u1_smebi32( //
    nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t row_count_a, nk_size_t row_count_b,
    nk_size_t depth_bits, nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_smebi32_packed_header_t const *)b_packed)->capability != nk_cap_smebi32_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_u1_smebi32_streaming_(a, b_packed, c, row_count_a, row_count_b, depth_bits, a_stride, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

/** Symmetric u1 dot-product using ZA0 time-sharing and a 3-tile fast path. Same ZA transpose
 *  pattern as hammings_symmetric, but with dot extraction. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_symmetric_u1_smebi32_streaming_( //
    nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth_bits, nk_size_t stride, nk_u32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    // BMOPA processes binary data in 32-bit words: each svbmopa_za32_u32_m step
    // handles one u32 (32 bits) across all row × column pairs simultaneously.
    nk_size_t const depth_words = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const depth_bytes = depth_bits / NUMKONG_BITS_PER_BYTE;
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_words, depth_tile_size);

    svbool_t const predicate_all_b32x = svptrue_b32();
    // Use padded depth (depth_words * 32) for BMOPA: zero-padded bits always match in XNOR,
    // so the effective depth for the matching→intersection conversion is the rounded-up bit count.
    svuint32_t const depth_u32x = svdup_u32((nk_u32_t)(depth_words * 32));

    nk_align_(64) nk_u32_t a_buffer[16][16]; // Stack buffer for A column save

    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(vector_count, tile_dim);

    for (nk_size_t row_tile_start = rows_begin; row_tile_start < rows_end && row_tile_start < vector_count;
         row_tile_start += tile_dim) {
        nk_size_t const rows_left = rows_end - row_tile_start;
        nk_size_t const rows_remaining = rows_left < tile_dim ? rows_left : tile_dim;
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= vector_count)
                                           ? rows_remaining
                                           : (vector_count - row_tile_start);
        svbool_t const row_predicate_b32x = svwhilelt_b32_u64(0u, rows_clamped);

        // Compute A tile popcounts
        nk_align_(64) nk_u32_t a_tile_pops[16];
        for (nk_size_t r = 0; r < rows_clamped; r++) {
            nk_u1x8_t const *a_row = (nk_u1x8_t const *)((char const *)vectors + (row_tile_start + r) * stride);
            a_tile_pops[r] = nk_dots_reduce_sum_u1_smebi32_streaming_(a_row, depth_bytes);
        }
        for (nk_size_t r = rows_clamped; r < tile_dim; r++) a_tile_pops[r] = 0;

        // Upper triangle: start from this row tile's column
        nk_size_t column_tile_index = row_tile_start / tile_dim;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_k);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_words)
                                                     ? depth_tile_size
                                                     : (depth_words > d_start_u32 ? depth_words - d_start_u32 : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_k);
                svbool_t const batch_predicate_b32x = svwhilelt_b32_u64(0u, u32s_this_tile);

                svbool_t const depth_predicate_b8x = svwhilelt_b8_u64(d_start_u32 * 4, depth_bytes);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_u8_t const *a_row = (nk_u8_t const *)vectors + (row_tile_start + row_in_tile) * stride +
                                           d_start_u32 * 4;
                    svuint8_t row_u8x = svld1_u8(depth_predicate_b8x, a_row);
                    svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_b32x, svreinterpret_u32_u8(row_u8x));
                }

                // Save A columns
                for (nk_size_t s = 0; s < u32s_this_tile; s++)
                    svst1_u32(predicate_all_b32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32x, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_k);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dim + col;
                    if (col_abs < vector_count) {
                        nk_u8_t const *b_row = (nk_u8_t const *)vectors + col_abs * stride + d_start_u32 * 4;
                        svuint8_t col_u8x = svld1_u8(depth_predicate_b8x, b_row);
                        svwrite_hor_za32_u32_m(0, col, batch_predicate_b32x, svreinterpret_u32_u8(col_u8x));
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_u32x = svld1_u32(predicate_all_b32x, a_buffer[step]);
                    svuint32_t b_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_b32x, 0, step);
                    svbmopa_za32_u32_m(1, row_predicate_b32x, predicate_all_b32x, a_u32x, b_u32x);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_k);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dim + col;
                    if (col_abs < vector_count) {
                        nk_u8_t const *b_row = (nk_u8_t const *)vectors + col_abs * stride + d_start_u32 * 4;
                        svuint8_t col_u8x = svld1_u8(depth_predicate_b8x, b_row);
                        svwrite_hor_za32_u32_m(0, col, batch_predicate_b32x, svreinterpret_u32_u8(col_u8x));
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_u32x = svld1_u32(predicate_all_b32x, a_buffer[step]);
                    svuint32_t b_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_b32x, 0, step);
                    svbmopa_za32_u32_m(2, row_predicate_b32x, predicate_all_b32x, a_u32x, b_u32x);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_k);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dim + col;
                    if (col_abs < vector_count) {
                        nk_u8_t const *b_row = (nk_u8_t const *)vectors + col_abs * stride + d_start_u32 * 4;
                        svuint8_t col_u8x = svld1_u8(depth_predicate_b8x, b_row);
                        svwrite_hor_za32_u32_m(0, col, batch_predicate_b32x, svreinterpret_u32_u8(col_u8x));
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_u32x = svld1_u32(predicate_all_b32x, a_buffer[step]);
                    svuint32_t b_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_b32x, 0, step);
                    svbmopa_za32_u32_m(3, row_predicate_b32x, predicate_all_b32x, a_u32x, b_u32x);
                }
            }

            // Extract: dot = (pop_a + pop_b - depth + matching) / 2
            // Compute B tile popcounts
            nk_align_(64) nk_u32_t b_pops[3][16];
            for (nk_size_t t = 0; t < 3; t++) {
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + t) * tile_dim + col;
                    if (col_abs < vector_count) {
                        nk_u1x8_t const *b_row = (nk_u1x8_t const *)((char const *)vectors + col_abs * stride);
                        b_pops[t][col] = nk_dots_reduce_sum_u1_smebi32_streaming_(b_row, depth_bytes);
                    }
                    else { b_pops[t][col] = 0; }
                }
            }

            nk_size_t const first_column_start = (column_tile_index + 0) * tile_dim;
            nk_size_t const second_column_start = (column_tile_index + 1) * tile_dim;
            svbool_t const first_bound_b32x = svwhilelt_b32_u64(first_column_start, vector_count);
            svbool_t const second_bound_b32x = svwhilelt_b32_u64(second_column_start, vector_count);
            svbool_t const third_bound_b32x = svwhilelt_b32_u64((column_tile_index + 2) * tile_dim, vector_count);
            int const first_crosses_diagonal = first_column_start < row_tile_start + tile_dim;
            int const second_crosses_diagonal = second_column_start < row_tile_start + tile_dim;
            for (nk_size_t row = 0; row < rows_clamped; row++) {
                svbool_t const first_b32x = first_crosses_diagonal
                                                ? nk_diagonal_cut_b32x_sme_(first_bound_b32x, first_column_start,
                                                                            row_tile_start + row)
                                                : first_bound_b32x;
                svbool_t const second_b32x = second_crosses_diagonal
                                                 ? nk_diagonal_cut_b32x_sme_(second_bound_b32x, second_column_start,
                                                                             row_tile_start + row)
                                                 : second_bound_b32x;
                nk_u32_t *result_row = (nk_u32_t *)((char *)result + (row_tile_start + row) * result_stride);
                svuint32_t pop_a_u32x = svdup_u32(a_tile_pops[row]);

                svuint32_t za1_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_b32x, 1, row);
                svuint32_t b_popcount_0_u32x = svld1_u32(predicate_all_b32x, b_pops[0]);
                svuint32_t sum_pops0_u32x = svadd_u32_x(predicate_all_b32x, pop_a_u32x, b_popcount_0_u32x);
                svuint32_t numerator0_u32x = svadd_u32_x(
                    predicate_all_b32x, svsub_u32_x(predicate_all_b32x, sum_pops0_u32x, depth_u32x), za1_u32x);
                svst1_u32(first_b32x, result_row + (column_tile_index + 0) * tile_dim,
                          svlsr_n_u32_x(predicate_all_b32x, numerator0_u32x, 1));

                svuint32_t za2_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_b32x, 2, row);
                svuint32_t b_popcount_1_u32x = svld1_u32(predicate_all_b32x, b_pops[1]);
                svuint32_t sum_pops1_u32x = svadd_u32_x(predicate_all_b32x, pop_a_u32x, b_popcount_1_u32x);
                svuint32_t numerator1_u32x = svadd_u32_x(
                    predicate_all_b32x, svsub_u32_x(predicate_all_b32x, sum_pops1_u32x, depth_u32x), za2_u32x);
                svst1_u32(second_b32x, result_row + (column_tile_index + 1) * tile_dim,
                          svlsr_n_u32_x(predicate_all_b32x, numerator1_u32x, 1));

                svuint32_t za3_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_b32x, 3, row);
                svuint32_t b_popcount_2_u32x = svld1_u32(predicate_all_b32x, b_pops[2]);
                svuint32_t sum_pops2_u32x = svadd_u32_x(predicate_all_b32x, pop_a_u32x, b_popcount_2_u32x);
                svuint32_t numerator2_u32x = svadd_u32_x(
                    predicate_all_b32x, svsub_u32_x(predicate_all_b32x, sum_pops2_u32x, depth_u32x), za3_u32x);
                svst1_u32(third_bound_b32x, result_row + (column_tile_index + 2) * tile_dim,
                          svlsr_n_u32_x(predicate_all_b32x, numerator2_u32x, 1));
            }
        }

        // Remainder: 1 column tile at a time using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_tile_start = column_tile_index * tile_dim;
            nk_size_t const columns_remaining = (col_tile_start + tile_dim <= vector_count)
                                                    ? tile_dim
                                                    : (vector_count - col_tile_start);
            svbool_t const column_predicate_b32x = svwhilelt_b32_u64(0u, columns_remaining);

            svzero_mask_za(nk_sme_zero_za32_tile_1_k);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_words)
                                                     ? depth_tile_size
                                                     : (depth_words > d_start_u32 ? depth_words - d_start_u32 : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_k);
                svbool_t const batch_predicate_b32x = svwhilelt_b32_u64(0u, u32s_this_tile);

                svbool_t const depth_predicate_b8x = svwhilelt_b8_u64(d_start_u32 * 4, depth_bytes);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_u8_t const *a_row = (nk_u8_t const *)vectors + (row_tile_start + row_in_tile) * stride +
                                           d_start_u32 * 4;
                    svuint8_t row_u8x = svld1_u8(depth_predicate_b8x, a_row);
                    svwrite_hor_za32_u32_m(0, row_in_tile, batch_predicate_b32x, svreinterpret_u32_u8(row_u8x));
                }

                for (nk_size_t s = 0; s < u32s_this_tile; s++)
                    svst1_u32(predicate_all_b32x, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32x, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_k);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < vector_count) {
                        nk_u8_t const *b_row = (nk_u8_t const *)vectors + col_abs * stride + d_start_u32 * 4;
                        svuint8_t col_u8x = svld1_u8(depth_predicate_b8x, b_row);
                        svwrite_hor_za32_u32_m(0, col, batch_predicate_b32x, svreinterpret_u32_u8(col_u8x));
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_u32x = svld1_u32(predicate_all_b32x, a_buffer[step]);
                    svuint32_t b_u32x = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_b32x, 0, step);
                    svbmopa_za32_u32_m(1, row_predicate_b32x, column_predicate_b32x, a_u32x, b_u32x);
                }
            }

            // Compute B tile popcounts for remainder
            nk_align_(64) nk_u32_t b_pops_r[16];
            for (nk_size_t col = 0; col < tile_dim; col++) {
                nk_size_t const col_abs = col_tile_start + col;
                if (col_abs < vector_count) {
                    nk_u1x8_t const *b_row = (nk_u1x8_t const *)((char const *)vectors + col_abs * stride);
                    b_pops_r[col] = nk_dots_reduce_sum_u1_smebi32_streaming_(b_row, depth_bytes);
                }
                else { b_pops_r[col] = 0; }
            }

            int const crosses_diagonal = col_tile_start < row_tile_start + tile_dim;
            for (nk_size_t row = 0; row < rows_clamped; row++) {
                svbool_t const store_b32x = crosses_diagonal
                                                ? nk_diagonal_cut_b32x_sme_(column_predicate_b32x, col_tile_start,
                                                                            row_tile_start + row)
                                                : column_predicate_b32x;
                svuint32_t za1_u32x = svread_hor_za32_u32_m(svdup_u32(0), predicate_all_b32x, 1, row);
                svuint32_t pop_a_u32x = svdup_u32(a_tile_pops[row]);
                svuint32_t b_popcount_u32x = svld1_u32(predicate_all_b32x, b_pops_r);
                svuint32_t sum_pops_u32x = svadd_u32_x(predicate_all_b32x, pop_a_u32x, b_popcount_u32x);
                svuint32_t numerator_u32x = svadd_u32_x(
                    predicate_all_b32x, svsub_u32_x(predicate_all_b32x, sum_pops_u32x, depth_u32x), za1_u32x);
                nk_u32_t *result_row = (nk_u32_t *)((char *)result + (row_tile_start + row) * result_stride);
                svst1_u32(store_b32x, result_row + col_tile_start,
                          svlsr_n_u32_x(predicate_all_b32x, numerator_u32x, 1));
            }
        }
    }
}

NUMKONG_API nk_status_t nk_dots_symmetric_u1_smebi32( //
    nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth_bits, nk_size_t stride, nk_u32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth_bits, 8) * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_u1_smebi32_streaming_(vectors, vector_count, depth_bits, stride, result, result_stride,
                                            rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_SMEBI32
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_DOTS_SMEBI32_H
