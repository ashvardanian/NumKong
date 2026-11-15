/**
 *  @file include/numkong/dots/powermma.h
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief SIMD-accelerated Batched Dot Products for the Power ISA 3.1 Matrix-Multiply Assist.
 *
 *  @sa c/numkong/dots.h
 *
 *  @section powermma_dots_instructions Key POWER10 MMA GEMM Instructions
 *
 *  @verbatim
 *  Intrinsic                      Instruction     Update of one 512-bit accumulator
 *  __builtin_mma_xxsetaccz        xxsetaccz       zero
 *  __builtin_mma_xvf64gerpp       xvf64gerpp      4 × 2 F64 += 4 F64 ⊗ 2 F64, rank 1
 *  __builtin_mma_xvbf16ger2pp     xvbf16ger2pp    4 × 4 F32 += 4 × 2 BF16 ⊗ 4 × 2 BF16, rank 2
 *  __builtin_mma_xvf16ger2pp      xvf16ger2pp     4 × 4 F32 += 4 × 2 F16 ⊗ 4 × 2 F16, rank 2
 *  __builtin_mma_xvi8ger4pp       xvi8ger4pp      4 × 4 I32 += 4 × 4 I8 ⊗ 4 × 4 U8, rank 4, modulo
 *  __builtin_mma_disassemble_acc  xxmfacc         4 accumulator rows into VSX registers
 *  @endverbatim
 *
 *  An accumulator row holds one row of A against four columns of B, so both operands of every
 *  update are columns: four rows of A at one depth step, and four columns of B at the same step.
 *  The 16-byte depth chunk of four A rows becomes four such operands through one 4 × 4 transpose
 *  of 32-bit words, as a word is one F32, one BF16 or F16 pair, or one quad of bytes. Packed B
 *  stores panels of 8 columns already transposed, two operands per depth step, zero-padded in
 *  depth, so the hot loops load B without shuffles. Symmetric kernels transpose both sides. Edge
 *  tiles repeat the last row or column to fill up, compute the full tile, and store only what lies
 *  inside the output.
 *
 *  Each dtype keeps its serial result type:
 *
 *  - BF16 and F16 accumulate in F32 through the rank-2 updates, 8 × 8 tiles in 4 accumulators.
 *  - F32 accumulates in F64: A widens in registers, B widens at pack time, and @c xvf64gerpp forms
 *    exact products, 4 × 8 tiles in 4 accumulators. @c xvf32gerpp would accumulate in F32.
 *  - I8 multiplies A as signed against B biased by 128 into unsigned, as @c xvi8ger4pp takes signed
 *    by unsigned bytes, and subtracts 128 times each A row's sum, which @c vec_sum4s gathers in the
 *    same loop. The I32 lanes wrap like the serial kernel's.
 *  - U8 biases A by 128 into signed bytes against unsigned B, and adds 128 times each B column's
 *    sum of bytes.
 *  - F64 stays on Power VSX, as @c xvf64gerpp accumulates without the compensation F64 dots carry,
 *    and U1 stays there too, as MMA has no binary update.
 */
#ifndef NUMKONG_DOTS_POWERMMA_H
#define NUMKONG_DOTS_POWERMMA_H

#if NUMKONG_ARCH_PPC64_
#if NUMKONG_ARCH_PPC64_POWERMMA_

#include "numkong/types.h"
#include "numkong/dots/serial.h" // `nk_cross_packed_buffer_header_t`, `nk_dots_reduce_sumsq_bf16_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("cpu=pwr10"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("cpu=power10")
#endif

#pragma region Helpers

/** Loads @p bytes, at most 16, at @p offset of four @p rows, zeroing the rest of each, and
 *  transposes the 4 × 4 words, so that @p columns_u32x4 hold word 0 to word 3 of all four rows. */
NUMKONG_INLINE void nk_load_transposed_b32x4x4_powermma_(nk_u8_t const *const *rows, nk_size_t offset, nk_size_t bytes,
                                                         nk_vu32x4_t *columns_u32x4) {
    nk_vu32x4_t const first_u32x4 = (nk_vu32x4_t)vec_xl_len((nk_u8_t *)rows[0] + offset, bytes);
    nk_vu32x4_t const second_u32x4 = (nk_vu32x4_t)vec_xl_len((nk_u8_t *)rows[1] + offset, bytes);
    nk_vu32x4_t const third_u32x4 = (nk_vu32x4_t)vec_xl_len((nk_u8_t *)rows[2] + offset, bytes);
    nk_vu32x4_t const fourth_u32x4 = (nk_vu32x4_t)vec_xl_len((nk_u8_t *)rows[3] + offset, bytes);
    nk_vu64x2_t const upper_front_u64x2 = (nk_vu64x2_t)vec_mergeh(first_u32x4, second_u32x4);
    nk_vu64x2_t const upper_back_u64x2 = (nk_vu64x2_t)vec_mergel(first_u32x4, second_u32x4);
    nk_vu64x2_t const lower_front_u64x2 = (nk_vu64x2_t)vec_mergeh(third_u32x4, fourth_u32x4);
    nk_vu64x2_t const lower_back_u64x2 = (nk_vu64x2_t)vec_mergel(third_u32x4, fourth_u32x4);
    columns_u32x4[0] = (nk_vu32x4_t)vec_mergeh(upper_front_u64x2, lower_front_u64x2);
    columns_u32x4[1] = (nk_vu32x4_t)vec_mergel(upper_front_u64x2, lower_front_u64x2);
    columns_u32x4[2] = (nk_vu32x4_t)vec_mergeh(upper_back_u64x2, lower_back_u64x2);
    columns_u32x4[3] = (nk_vu32x4_t)vec_mergel(upper_back_u64x2, lower_back_u64x2);
}

/** Points @p rows at the @p count rows from @p first on, repeating row @p end − 1 past @p end, so
 *  edge tiles run the full tile and store only their valid rows. */
NUMKONG_INLINE void nk_dots_clamped_rows_powermma_(void const *base, nk_size_t stride, nk_size_t first, nk_size_t end,
                                                   nk_size_t count, nk_u8_t const **rows) {
    for (nk_size_t index = 0; index != count; ++index)
        rows[index] = (nk_u8_t const *)base + nk_min_of_two(first + index, end - 1) * stride;
}

/** Copies @p count 8-bit values into @p destination, 16 at a time with @c lxvl and @c stxvl. */
NUMKONG_INLINE void nk_dots_copy_row_b8_powermma_(nk_u8_t *destination, nk_u8_t const *source, nk_size_t count) {
    for (nk_size_t index = 0; index < count; index += 16) {
        nk_size_t const chunk = nk_min_of_two(count - index, (nk_size_t)16);
        vec_xst_len(vec_xl_len((nk_u8_t *)source + index, chunk), destination + index, chunk);
    }
}

/** Unloads four accumulators of 32-bit lanes, a 2 × 2 grid of 4 × 4 blocks, into 8 rows of 8. */
NUMKONG_INLINE void nk_dots_unload_b32x8x8_powermma_(__vector_quad *top_left_tile, __vector_quad *top_right_tile,
                                                     __vector_quad *bottom_left_tile, __vector_quad *bottom_right_tile,
                                                     nk_b256_vec_t *rows_vec) {
    nk_vu32x4_t left_u32x4[4], right_u32x4[4];
    __builtin_mma_disassemble_acc(left_u32x4, top_left_tile);
    __builtin_mma_disassemble_acc(right_u32x4, top_right_tile);
    for (unsigned row = 0; row != 4; ++row)
        rows_vec[row].vu32x4s[0] = left_u32x4[row], rows_vec[row].vu32x4s[1] = right_u32x4[row];
    __builtin_mma_disassemble_acc(left_u32x4, bottom_left_tile);
    __builtin_mma_disassemble_acc(right_u32x4, bottom_right_tile);
    for (unsigned row = 0; row != 4; ++row)
        rows_vec[row + 4].vu32x4s[0] = left_u32x4[row], rows_vec[row + 4].vu32x4s[1] = right_u32x4[row];
}

/** Subtracts 128 times each row's sum of signed bytes from its 8 lanes, as I8 kernels bias B. */
NUMKONG_INLINE void nk_dots_unbias_rows_i8_powermma_(nk_vi32x4_t top_sums_i32x4, nk_vi32x4_t bottom_sums_i32x4,
                                                     nk_b256_vec_t *rows_vec) {
    nk_vu32x4_t const shift_u32x4 = vec_splats((nk_u32_t)7);
    nk_b256_vec_t corrections_vec;
    corrections_vec.vi32x4s[0] = vec_sl(top_sums_i32x4, shift_u32x4);
    corrections_vec.vi32x4s[1] = vec_sl(bottom_sums_i32x4, shift_u32x4);
    for (unsigned row = 0; row != 8; ++row) {
        nk_vi32x4_t const correction_i32x4 = vec_splats(corrections_vec.i32s[row]);
        rows_vec[row].vi32x4s[0] = vec_sub(rows_vec[row].vi32x4s[0], correction_i32x4);
        rows_vec[row].vi32x4s[1] = vec_sub(rows_vec[row].vi32x4s[1], correction_i32x4);
    }
}

/** Adds 128 times each column's sum of unsigned bytes to every row, as the U8 kernels bias A. */
NUMKONG_INLINE void nk_dots_unbias_columns_u8_powermma_(nk_vu32x4_t left_sums_u32x4, nk_vu32x4_t right_sums_u32x4,
                                                        nk_b256_vec_t *rows_vec) {
    nk_vu32x4_t const shift_u32x4 = vec_splats((nk_u32_t)7);
    nk_vu32x4_t const left_corrections_u32x4 = vec_sl(left_sums_u32x4, shift_u32x4);
    nk_vu32x4_t const right_corrections_u32x4 = vec_sl(right_sums_u32x4, shift_u32x4);
    for (unsigned row = 0; row != 8; ++row) {
        rows_vec[row].vu32x4s[0] = vec_add(rows_vec[row].vu32x4s[0], left_corrections_u32x4);
        rows_vec[row].vu32x4s[1] = vec_add(rows_vec[row].vu32x4s[1], right_corrections_u32x4);
    }
}

/** Packs columns from @p columns_begin up to @p columns_end of @p b, rows of @p depth_bytes, into
 *  panels of 8 columns: per 16-byte depth chunk, four steps of two transposed operands, each XORed
 *  with @p flip_u8x16. A window packs the panels that start in it, reading on to the panel's end,
 *  so any split into windows packs the same bytes. */
NUMKONG_INLINE void nk_dots_pack_panels_b32_powermma_(void const *b, nk_size_t column_count, nk_size_t depth_bytes,
                                                      nk_size_t b_stride, nk_u8_t *panels, nk_size_t panel_bytes,
                                                      nk_size_t columns_begin, nk_size_t columns_end,
                                                      nk_vu8x16_t flip_u8x16) {
    nk_size_t const panels_begin = nk_size_divide_round_up_(columns_begin, 8);
    nk_size_t const panels_end = nk_size_divide_round_up_(columns_end, 8);
    nk_u8_t const *columns[8];
    nk_vu32x4_t left_u32x4[4], right_u32x4[4];
    for (nk_size_t panel = panels_begin; panel < panels_end; ++panel) {
        nk_u8_t *panel_output = panels + panel * panel_bytes;
        nk_dots_clamped_rows_powermma_(b, b_stride, panel * 8, column_count, 8, columns);
        for (nk_size_t offset = 0; offset < depth_bytes; offset += 16, panel_output += 128) {
            nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
            nk_load_transposed_b32x4x4_powermma_(columns, offset, chunk_bytes, left_u32x4);
            nk_load_transposed_b32x4x4_powermma_(columns + 4, offset, chunk_bytes, right_u32x4);
            for (unsigned step = 0; step != 4; ++step) {
                vec_xst(vec_xor((nk_vu8x16_t)left_u32x4[step], flip_u8x16), 32 * step, panel_output);
                vec_xst(vec_xor((nk_vu8x16_t)right_u32x4[step], flip_u8x16), 32 * step + 16, panel_output);
            }
        }
    }
}

/** Writes the pack header of @p column_count columns of @p depth dimensions, padded to
 *  @p depth_padded_values, with norms at @p norms_offset. */
NUMKONG_INLINE void nk_dots_pack_header_powermma_(void *b_packed, nk_size_t column_count, nk_size_t depth,
                                                  nk_size_t depth_padded_values, nk_size_t norms_offset) {
    nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;
    header->column_count = (nk_u32_t)column_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_padded_values = (nk_u32_t)depth_padded_values;
    header->scales_stride = 0;
    header->tensor_scale = 1;
    header->norms_offset = (nk_u32_t)norms_offset;
    for (nk_size_t reserved_index = 0; reserved_index < 8; reserved_index++) header->reserved[reserved_index] = 0;
    header->capability = nk_cap_powermma_k;
}

/** Bytes of one panel of 8 columns over @p depth_values padded to 16 bytes of @p value_bytes each,
 *  stored as @p packed_value_bytes each. */
NUMKONG_INLINE nk_size_t nk_dots_panel_bytes_powermma_(nk_size_t depth_values, nk_size_t value_bytes,
                                                       nk_size_t packed_value_bytes) {
    return nk_size_round_up_to_multiple_(depth_values, 16 / value_bytes) * 8 * packed_value_bytes;
}

/** Bytes of a packed buffer: its header, one panel per 8 columns, and @p norm_bytes per column. */
NUMKONG_INLINE nk_size_t nk_dots_pack_bytes_powermma_(nk_size_t column_count, nk_size_t depth, nk_size_t value_bytes,
                                                      nk_size_t packed_value_bytes, nk_size_t norm_bytes) {
    return sizeof(nk_cross_packed_buffer_header_t) +
           nk_size_divide_round_up_(column_count, 8) *
               nk_dots_panel_bytes_powermma_(depth, value_bytes, packed_value_bytes) +
           column_count * norm_bytes;
}

/** Stores row @p row of a tile whose first column is @p column, from its lane on the diagonal or
 *  its first lane, whichever is later, up to @p vector_count, @p value_bytes per lane. */
NUMKONG_INLINE void nk_dots_store_upper_row_powermma_(nk_u8_t const *tile_row, nk_size_t row, nk_size_t column,
                                                      nk_size_t vector_count, nk_size_t value_bytes, void *result,
                                                      nk_size_t result_stride) {
    nk_size_t const first = row > column ? row - column : 0;
    nk_size_t const end = nk_min_of_two(vector_count - column, (nk_size_t)8);
    if (first >= end) return;
    nk_dots_copy_row_b8_powermma_((nk_u8_t *)result + row * result_stride + (column + first) * value_bytes,
                                  tile_row + first * value_bytes, (end - first) * value_bytes);
}

#pragma endregion Helpers

/*  Operands stay non-const, as Clang's MMA builtins reject const vectors. */
#pragma region B32 Tiles

NUMKONG_INLINE void nk_dots_packed_tile_bf16_powermma_(nk_u8_t const *const *rows, nk_u8_t const *panel,
                                                       nk_size_t depth_bytes, nk_b256_vec_t *tile_vec) {
    __vector_quad top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile;
    __builtin_mma_xxsetaccz(&top_left_tile), __builtin_mma_xxsetaccz(&top_right_tile);
    __builtin_mma_xxsetaccz(&bottom_left_tile), __builtin_mma_xxsetaccz(&bottom_right_tile);
    nk_vu32x4_t top_u32x4[4], bottom_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16, panel += 128) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, top_u32x4);
        nk_load_transposed_b32x4x4_powermma_(rows + 4, offset, chunk_bytes, bottom_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            nk_vu8x16_t left_u8x16 = vec_xl(32 * step, panel), right_u8x16 = vec_xl(32 * step + 16, panel);
            __builtin_mma_xvbf16ger2pp(&top_left_tile, (nk_vu8x16_t)top_u32x4[step], left_u8x16);
            __builtin_mma_xvbf16ger2pp(&top_right_tile, (nk_vu8x16_t)top_u32x4[step], right_u8x16);
            __builtin_mma_xvbf16ger2pp(&bottom_left_tile, (nk_vu8x16_t)bottom_u32x4[step], left_u8x16);
            __builtin_mma_xvbf16ger2pp(&bottom_right_tile, (nk_vu8x16_t)bottom_u32x4[step], right_u8x16);
        }
    }
    nk_dots_unload_b32x8x8_powermma_(&top_left_tile, &top_right_tile, &bottom_left_tile, &bottom_right_tile, tile_vec);
}

NUMKONG_INLINE void nk_dots_symmetric_tile_bf16_powermma_(nk_u8_t const *const *rows, nk_u8_t const *const *columns,
                                                          nk_size_t depth_bytes, nk_b256_vec_t *tile_vec) {
    __vector_quad top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile;
    __builtin_mma_xxsetaccz(&top_left_tile), __builtin_mma_xxsetaccz(&top_right_tile);
    __builtin_mma_xxsetaccz(&bottom_left_tile), __builtin_mma_xxsetaccz(&bottom_right_tile);
    nk_vu32x4_t top_u32x4[4], bottom_u32x4[4], left_u32x4[4], right_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, top_u32x4);
        nk_load_transposed_b32x4x4_powermma_(rows + 4, offset, chunk_bytes, bottom_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns, offset, chunk_bytes, left_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns + 4, offset, chunk_bytes, right_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            nk_vu8x16_t left_u8x16 = (nk_vu8x16_t)left_u32x4[step];
            nk_vu8x16_t right_u8x16 = (nk_vu8x16_t)right_u32x4[step];
            __builtin_mma_xvbf16ger2pp(&top_left_tile, (nk_vu8x16_t)top_u32x4[step], left_u8x16);
            __builtin_mma_xvbf16ger2pp(&top_right_tile, (nk_vu8x16_t)top_u32x4[step], right_u8x16);
            __builtin_mma_xvbf16ger2pp(&bottom_left_tile, (nk_vu8x16_t)bottom_u32x4[step], left_u8x16);
            __builtin_mma_xvbf16ger2pp(&bottom_right_tile, (nk_vu8x16_t)bottom_u32x4[step], right_u8x16);
        }
    }
    nk_dots_unload_b32x8x8_powermma_(&top_left_tile, &top_right_tile, &bottom_left_tile, &bottom_right_tile, tile_vec);
}

NUMKONG_INLINE void nk_dots_packed_tile_f16_powermma_(nk_u8_t const *const *rows, nk_u8_t const *panel,
                                                      nk_size_t depth_bytes, nk_b256_vec_t *tile_vec) {
    __vector_quad top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile;
    __builtin_mma_xxsetaccz(&top_left_tile), __builtin_mma_xxsetaccz(&top_right_tile);
    __builtin_mma_xxsetaccz(&bottom_left_tile), __builtin_mma_xxsetaccz(&bottom_right_tile);
    nk_vu32x4_t top_u32x4[4], bottom_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16, panel += 128) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, top_u32x4);
        nk_load_transposed_b32x4x4_powermma_(rows + 4, offset, chunk_bytes, bottom_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            nk_vu8x16_t left_u8x16 = vec_xl(32 * step, panel), right_u8x16 = vec_xl(32 * step + 16, panel);
            __builtin_mma_xvf16ger2pp(&top_left_tile, (nk_vu8x16_t)top_u32x4[step], left_u8x16);
            __builtin_mma_xvf16ger2pp(&top_right_tile, (nk_vu8x16_t)top_u32x4[step], right_u8x16);
            __builtin_mma_xvf16ger2pp(&bottom_left_tile, (nk_vu8x16_t)bottom_u32x4[step], left_u8x16);
            __builtin_mma_xvf16ger2pp(&bottom_right_tile, (nk_vu8x16_t)bottom_u32x4[step], right_u8x16);
        }
    }
    nk_dots_unload_b32x8x8_powermma_(&top_left_tile, &top_right_tile, &bottom_left_tile, &bottom_right_tile, tile_vec);
}

NUMKONG_INLINE void nk_dots_symmetric_tile_f16_powermma_(nk_u8_t const *const *rows, nk_u8_t const *const *columns,
                                                         nk_size_t depth_bytes, nk_b256_vec_t *tile_vec) {
    __vector_quad top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile;
    __builtin_mma_xxsetaccz(&top_left_tile), __builtin_mma_xxsetaccz(&top_right_tile);
    __builtin_mma_xxsetaccz(&bottom_left_tile), __builtin_mma_xxsetaccz(&bottom_right_tile);
    nk_vu32x4_t top_u32x4[4], bottom_u32x4[4], left_u32x4[4], right_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, top_u32x4);
        nk_load_transposed_b32x4x4_powermma_(rows + 4, offset, chunk_bytes, bottom_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns, offset, chunk_bytes, left_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns + 4, offset, chunk_bytes, right_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            nk_vu8x16_t left_u8x16 = (nk_vu8x16_t)left_u32x4[step];
            nk_vu8x16_t right_u8x16 = (nk_vu8x16_t)right_u32x4[step];
            __builtin_mma_xvf16ger2pp(&top_left_tile, (nk_vu8x16_t)top_u32x4[step], left_u8x16);
            __builtin_mma_xvf16ger2pp(&top_right_tile, (nk_vu8x16_t)top_u32x4[step], right_u8x16);
            __builtin_mma_xvf16ger2pp(&bottom_left_tile, (nk_vu8x16_t)bottom_u32x4[step], left_u8x16);
            __builtin_mma_xvf16ger2pp(&bottom_right_tile, (nk_vu8x16_t)bottom_u32x4[step], right_u8x16);
        }
    }
    nk_dots_unload_b32x8x8_powermma_(&top_left_tile, &top_right_tile, &bottom_left_tile, &bottom_right_tile, tile_vec);
}

NUMKONG_INLINE void nk_dots_packed_tile_i8_powermma_(nk_u8_t const *const *rows, nk_u8_t const *panel,
                                                     nk_size_t depth_bytes, nk_b256_vec_t *tile_vec) {
    __vector_quad top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile;
    __builtin_mma_xxsetaccz(&top_left_tile), __builtin_mma_xxsetaccz(&top_right_tile);
    __builtin_mma_xxsetaccz(&bottom_left_tile), __builtin_mma_xxsetaccz(&bottom_right_tile);
    nk_vi32x4_t top_sums_i32x4 = vec_splats((nk_i32_t)0), bottom_sums_i32x4 = vec_splats((nk_i32_t)0);
    nk_vu32x4_t top_u32x4[4], bottom_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16, panel += 128) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, top_u32x4);
        nk_load_transposed_b32x4x4_powermma_(rows + 4, offset, chunk_bytes, bottom_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            nk_vu8x16_t left_u8x16 = vec_xl(32 * step, panel), right_u8x16 = vec_xl(32 * step + 16, panel);
            top_sums_i32x4 = vec_sum4s((nk_vi8x16_t)top_u32x4[step], top_sums_i32x4);
            bottom_sums_i32x4 = vec_sum4s((nk_vi8x16_t)bottom_u32x4[step], bottom_sums_i32x4);
            __builtin_mma_xvi8ger4pp(&top_left_tile, (nk_vu8x16_t)top_u32x4[step], left_u8x16);
            __builtin_mma_xvi8ger4pp(&top_right_tile, (nk_vu8x16_t)top_u32x4[step], right_u8x16);
            __builtin_mma_xvi8ger4pp(&bottom_left_tile, (nk_vu8x16_t)bottom_u32x4[step], left_u8x16);
            __builtin_mma_xvi8ger4pp(&bottom_right_tile, (nk_vu8x16_t)bottom_u32x4[step], right_u8x16);
        }
    }
    nk_dots_unload_b32x8x8_powermma_(&top_left_tile, &top_right_tile, &bottom_left_tile, &bottom_right_tile, tile_vec);
    nk_dots_unbias_rows_i8_powermma_(top_sums_i32x4, bottom_sums_i32x4, tile_vec);
}

NUMKONG_INLINE void nk_dots_symmetric_tile_i8_powermma_(nk_u8_t const *const *rows, nk_u8_t const *const *columns,
                                                        nk_size_t depth_bytes, nk_b256_vec_t *tile_vec) {
    __vector_quad top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile;
    __builtin_mma_xxsetaccz(&top_left_tile), __builtin_mma_xxsetaccz(&top_right_tile);
    __builtin_mma_xxsetaccz(&bottom_left_tile), __builtin_mma_xxsetaccz(&bottom_right_tile);
    nk_vi32x4_t top_sums_i32x4 = vec_splats((nk_i32_t)0), bottom_sums_i32x4 = vec_splats((nk_i32_t)0);
    nk_vu8x16_t const bias_u8x16 = vec_splats((nk_u8_t)0x80);
    nk_vu32x4_t top_u32x4[4], bottom_u32x4[4], left_u32x4[4], right_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, top_u32x4);
        nk_load_transposed_b32x4x4_powermma_(rows + 4, offset, chunk_bytes, bottom_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns, offset, chunk_bytes, left_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns + 4, offset, chunk_bytes, right_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            nk_vu8x16_t left_u8x16 = vec_xor((nk_vu8x16_t)left_u32x4[step], bias_u8x16);
            nk_vu8x16_t right_u8x16 = vec_xor((nk_vu8x16_t)right_u32x4[step], bias_u8x16);
            top_sums_i32x4 = vec_sum4s((nk_vi8x16_t)top_u32x4[step], top_sums_i32x4);
            bottom_sums_i32x4 = vec_sum4s((nk_vi8x16_t)bottom_u32x4[step], bottom_sums_i32x4);
            __builtin_mma_xvi8ger4pp(&top_left_tile, (nk_vu8x16_t)top_u32x4[step], left_u8x16);
            __builtin_mma_xvi8ger4pp(&top_right_tile, (nk_vu8x16_t)top_u32x4[step], right_u8x16);
            __builtin_mma_xvi8ger4pp(&bottom_left_tile, (nk_vu8x16_t)bottom_u32x4[step], left_u8x16);
            __builtin_mma_xvi8ger4pp(&bottom_right_tile, (nk_vu8x16_t)bottom_u32x4[step], right_u8x16);
        }
    }
    nk_dots_unload_b32x8x8_powermma_(&top_left_tile, &top_right_tile, &bottom_left_tile, &bottom_right_tile, tile_vec);
    nk_dots_unbias_rows_i8_powermma_(top_sums_i32x4, bottom_sums_i32x4, tile_vec);
}

NUMKONG_INLINE void nk_dots_packed_tile_u8_powermma_(nk_u8_t const *const *rows, nk_u8_t const *panel,
                                                     nk_size_t depth_bytes, nk_b256_vec_t *tile_vec) {
    __vector_quad top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile;
    __builtin_mma_xxsetaccz(&top_left_tile), __builtin_mma_xxsetaccz(&top_right_tile);
    __builtin_mma_xxsetaccz(&bottom_left_tile), __builtin_mma_xxsetaccz(&bottom_right_tile);
    nk_vu32x4_t left_sums_u32x4 = vec_splats((nk_u32_t)0), right_sums_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu8x16_t const bias_u8x16 = vec_splats((nk_u8_t)0x80);
    nk_vu32x4_t top_u32x4[4], bottom_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16, panel += 128) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, top_u32x4);
        nk_load_transposed_b32x4x4_powermma_(rows + 4, offset, chunk_bytes, bottom_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            nk_vu8x16_t left_u8x16 = vec_xl(32 * step, panel), right_u8x16 = vec_xl(32 * step + 16, panel);
            nk_vu8x16_t top_u8x16 = vec_xor((nk_vu8x16_t)top_u32x4[step], bias_u8x16);
            nk_vu8x16_t bottom_u8x16 = vec_xor((nk_vu8x16_t)bottom_u32x4[step], bias_u8x16);
            left_sums_u32x4 = vec_sum4s(left_u8x16, left_sums_u32x4);
            right_sums_u32x4 = vec_sum4s(right_u8x16, right_sums_u32x4);
            __builtin_mma_xvi8ger4pp(&top_left_tile, top_u8x16, left_u8x16);
            __builtin_mma_xvi8ger4pp(&top_right_tile, top_u8x16, right_u8x16);
            __builtin_mma_xvi8ger4pp(&bottom_left_tile, bottom_u8x16, left_u8x16);
            __builtin_mma_xvi8ger4pp(&bottom_right_tile, bottom_u8x16, right_u8x16);
        }
    }
    nk_dots_unload_b32x8x8_powermma_(&top_left_tile, &top_right_tile, &bottom_left_tile, &bottom_right_tile, tile_vec);
    nk_dots_unbias_columns_u8_powermma_(left_sums_u32x4, right_sums_u32x4, tile_vec);
}

NUMKONG_INLINE void nk_dots_symmetric_tile_u8_powermma_(nk_u8_t const *const *rows, nk_u8_t const *const *columns,
                                                        nk_size_t depth_bytes, nk_b256_vec_t *tile_vec) {
    __vector_quad top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile;
    __builtin_mma_xxsetaccz(&top_left_tile), __builtin_mma_xxsetaccz(&top_right_tile);
    __builtin_mma_xxsetaccz(&bottom_left_tile), __builtin_mma_xxsetaccz(&bottom_right_tile);
    nk_vu32x4_t left_sums_u32x4 = vec_splats((nk_u32_t)0), right_sums_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu8x16_t const bias_u8x16 = vec_splats((nk_u8_t)0x80);
    nk_vu32x4_t top_u32x4[4], bottom_u32x4[4], left_u32x4[4], right_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, top_u32x4);
        nk_load_transposed_b32x4x4_powermma_(rows + 4, offset, chunk_bytes, bottom_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns, offset, chunk_bytes, left_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns + 4, offset, chunk_bytes, right_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            nk_vu8x16_t left_u8x16 = (nk_vu8x16_t)left_u32x4[step];
            nk_vu8x16_t right_u8x16 = (nk_vu8x16_t)right_u32x4[step];
            nk_vu8x16_t top_u8x16 = vec_xor((nk_vu8x16_t)top_u32x4[step], bias_u8x16);
            nk_vu8x16_t bottom_u8x16 = vec_xor((nk_vu8x16_t)bottom_u32x4[step], bias_u8x16);
            left_sums_u32x4 = vec_sum4s(left_u8x16, left_sums_u32x4);
            right_sums_u32x4 = vec_sum4s(right_u8x16, right_sums_u32x4);
            __builtin_mma_xvi8ger4pp(&top_left_tile, top_u8x16, left_u8x16);
            __builtin_mma_xvi8ger4pp(&top_right_tile, top_u8x16, right_u8x16);
            __builtin_mma_xvi8ger4pp(&bottom_left_tile, bottom_u8x16, left_u8x16);
            __builtin_mma_xvi8ger4pp(&bottom_right_tile, bottom_u8x16, right_u8x16);
        }
    }
    nk_dots_unload_b32x8x8_powermma_(&top_left_tile, &top_right_tile, &bottom_left_tile, &bottom_right_tile, tile_vec);
    nk_dots_unbias_columns_u8_powermma_(left_sums_u32x4, right_sums_u32x4, tile_vec);
}

#pragma endregion B32 Tiles

#pragma region F64 Tiles

/** Widens four F32 lanes, one per row, into the F64 register pair @c xvf64gerpp takes, as MMA
 *  types never pass by value. */
NUMKONG_INLINE void nk_f32x4_to_f64x4_pair_powermma_(nk_vf32x4_t rows_f32x4, __vector_pair *rows_f64x4) {
    __builtin_vsx_assemble_pair(rows_f64x4, (nk_vu8x16_t)vec_doublel(rows_f32x4), (nk_vu8x16_t)vec_doubleh(rows_f32x4));
}

/** Unloads four accumulators of 4 × 2 F64, side by side, into 4 rows of 8 F64. */
NUMKONG_INLINE void nk_dots_unload_f64x4x8_powermma_(__vector_quad *first_tile, __vector_quad *second_tile,
                                                     __vector_quad *third_tile, __vector_quad *fourth_tile,
                                                     nk_b512_vec_t *rows_vec) {
    nk_vf64x2_t columns_f64x2[4];
    __builtin_mma_disassemble_acc(columns_f64x2, first_tile);
    for (unsigned row = 0; row != 4; ++row) vec_xst(columns_f64x2[row], 0, rows_vec[row].f64s);
    __builtin_mma_disassemble_acc(columns_f64x2, second_tile);
    for (unsigned row = 0; row != 4; ++row) vec_xst(columns_f64x2[row], 0, rows_vec[row].f64s + 2);
    __builtin_mma_disassemble_acc(columns_f64x2, third_tile);
    for (unsigned row = 0; row != 4; ++row) vec_xst(columns_f64x2[row], 0, rows_vec[row].f64s + 4);
    __builtin_mma_disassemble_acc(columns_f64x2, fourth_tile);
    for (unsigned row = 0; row != 4; ++row) vec_xst(columns_f64x2[row], 0, rows_vec[row].f64s + 6);
}

NUMKONG_INLINE void nk_dots_packed_tile_f32_powermma_(nk_u8_t const *const *rows, nk_u8_t const *panel,
                                                      nk_size_t depth_bytes, nk_b512_vec_t *tile_vec) {
    __vector_quad first_tile, second_tile, third_tile, fourth_tile;
    __builtin_mma_xxsetaccz(&first_tile), __builtin_mma_xxsetaccz(&second_tile);
    __builtin_mma_xxsetaccz(&third_tile), __builtin_mma_xxsetaccz(&fourth_tile);
    nk_vu32x4_t rows_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16, panel += 256) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, rows_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            __vector_pair rows_f64x4;
            nk_f32x4_to_f64x4_pair_powermma_((nk_vf32x4_t)rows_u32x4[step], &rows_f64x4);
            nk_f64_t const *columns = (nk_f64_t const *)panel + 8 * step;
            __builtin_mma_xvf64gerpp(&first_tile, rows_f64x4, (nk_vu8x16_t)vec_xl(0, columns));
            __builtin_mma_xvf64gerpp(&second_tile, rows_f64x4, (nk_vu8x16_t)vec_xl(0, columns + 2));
            __builtin_mma_xvf64gerpp(&third_tile, rows_f64x4, (nk_vu8x16_t)vec_xl(0, columns + 4));
            __builtin_mma_xvf64gerpp(&fourth_tile, rows_f64x4, (nk_vu8x16_t)vec_xl(0, columns + 6));
        }
    }
    nk_dots_unload_f64x4x8_powermma_(&first_tile, &second_tile, &third_tile, &fourth_tile, tile_vec);
}

NUMKONG_INLINE void nk_dots_symmetric_tile_f32_powermma_(nk_u8_t const *const *rows, nk_u8_t const *const *columns,
                                                         nk_size_t depth_bytes, nk_b512_vec_t *tile_vec) {
    __vector_quad first_tile, second_tile, third_tile, fourth_tile;
    __builtin_mma_xxsetaccz(&first_tile), __builtin_mma_xxsetaccz(&second_tile);
    __builtin_mma_xxsetaccz(&third_tile), __builtin_mma_xxsetaccz(&fourth_tile);
    nk_vu32x4_t rows_u32x4[4], left_u32x4[4], right_u32x4[4];
    for (nk_size_t offset = 0; offset < depth_bytes; offset += 16) {
        nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
        nk_load_transposed_b32x4x4_powermma_(rows, offset, chunk_bytes, rows_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns, offset, chunk_bytes, left_u32x4);
        nk_load_transposed_b32x4x4_powermma_(columns + 4, offset, chunk_bytes, right_u32x4);
        for (unsigned step = 0; step != 4; ++step) {
            __vector_pair rows_f64x4;
            nk_f32x4_to_f64x4_pair_powermma_((nk_vf32x4_t)rows_u32x4[step], &rows_f64x4);
            nk_vf32x4_t const left_f32x4 = (nk_vf32x4_t)left_u32x4[step];
            nk_vf32x4_t const right_f32x4 = (nk_vf32x4_t)right_u32x4[step];
            __builtin_mma_xvf64gerpp(&first_tile, rows_f64x4, (nk_vu8x16_t)vec_doubleh(left_f32x4));
            __builtin_mma_xvf64gerpp(&second_tile, rows_f64x4, (nk_vu8x16_t)vec_doublel(left_f32x4));
            __builtin_mma_xvf64gerpp(&third_tile, rows_f64x4, (nk_vu8x16_t)vec_doubleh(right_f32x4));
            __builtin_mma_xvf64gerpp(&fourth_tile, rows_f64x4, (nk_vu8x16_t)vec_doublel(right_f32x4));
        }
    }
    nk_dots_unload_f64x4x8_powermma_(&first_tile, &second_tile, &third_tile, &fourth_tile, tile_vec);
}

#pragma endregion F64 Tiles

#pragma region BF16 Floats

#if NUMKONG_TARGET_POWERMMA
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_powermma(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_bytes_powermma_(columns, depth, sizeof(nk_bf16_t), sizeof(nk_bf16_t), sizeof(nk_f32_t));
    return nk_success_k;
}

nk_define_cross_packed_shape_(dots, bf16, powermma)

NUMKONG_API nk_status_t nk_dots_pack_bf16_powermma(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const panel_bytes = nk_dots_panel_bytes_powermma_(depth, sizeof(nk_bf16_t), sizeof(nk_bf16_t));
    nk_size_t const norms_offset = sizeof(nk_cross_packed_buffer_header_t) +
                                   nk_size_divide_round_up_(columns, 8) * panel_bytes;
    if (columns_begin == 0)
        nk_dots_pack_header_powermma_(b_packed, columns, depth, panel_bytes / 8 / sizeof(nk_bf16_t), norms_offset);
    nk_dots_pack_panels_b32_powermma_(b, columns, depth * sizeof(nk_bf16_t), b_stride,
                                      (nk_u8_t *)b_packed + sizeof(nk_cross_packed_buffer_header_t), panel_bytes,
                                      columns_begin, columns_end, vec_splats((nk_u8_t)0));
    nk_f32_t *norms = (nk_f32_t *)((nk_u8_t *)b_packed + norms_offset);
    for (nk_size_t column = columns_begin; column < columns_end; ++column)
        norms[column] = nk_dots_reduce_sumsq_bf16_serial_((nk_bf16_t const *)((nk_u8_t const *)b + column * b_stride),
                                                          depth, sizeof(nk_bf16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_workspace_size_bf16_powermma(nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                    nk_size_t *bytes) {
    nk_unused_(rows), nk_unused_(columns), nk_unused_(depth);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_bf16_powermma(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *workspace,
                                                     nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    if (header->capability != nk_cap_powermma_k) return nk_pack_mismatch_k;
    nk_size_t const panel_bytes = (nk_size_t)header->depth_padded_values * 8 * sizeof(nk_bf16_t);
    nk_u8_t const *panels = (nk_u8_t const *)b_packed + sizeof(nk_cross_packed_buffer_header_t);
    nk_u8_t const *a_rows[8];
    nk_b256_vec_t tile_vec[8];
    for (nk_size_t row = 0; row < rows; row += 8) {
        nk_dots_clamped_rows_powermma_(a, a_stride, row, rows, 8, a_rows);
        nk_size_t const tile_rows = nk_min_of_two(rows - row, (nk_size_t)8);
        for (nk_size_t column = 0; column < columns; column += 8) {
            nk_dots_packed_tile_bf16_powermma_(a_rows, panels + column / 8 * panel_bytes, depth * sizeof(nk_bf16_t),
                                               tile_vec);
            nk_size_t const tile_bytes = nk_min_of_two(columns - column, (nk_size_t)8) * sizeof(nk_f32_t);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_copy_row_b8_powermma_((nk_u8_t *)c + (row + tile_row) * c_stride + column * sizeof(nk_f32_t),
                                              tile_vec[tile_row].u8s, tile_bytes);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_workspace_size_bf16_powermma(nk_size_t vector_count, nk_size_t depth,
                                                                       nk_size_t rows_begin, nk_size_t rows_end,
                                                                       nk_size_t *bytes) {
    nk_unused_(vector_count), nk_unused_(depth), nk_unused_(rows_begin), nk_unused_(rows_end);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_bf16_powermma(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, void *workspace, nk_size_t workspace_bytes,
                                                        nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_u8_t const *row_vectors[8], *column_vectors[8];
    nk_b256_vec_t tile_vec[8];
    for (nk_size_t row = rows_begin; row < rows_end; row += 8) {
        nk_dots_clamped_rows_powermma_(vectors, stride, row, rows_end, 8, row_vectors);
        nk_size_t const tile_rows = nk_min_of_two(rows_end - row, (nk_size_t)8);
        for (nk_size_t column = row; column < vector_count; column += 8) {
            nk_dots_clamped_rows_powermma_(vectors, stride, column, vector_count, 8, column_vectors);
            nk_dots_symmetric_tile_bf16_powermma_(row_vectors, column_vectors, depth * sizeof(nk_bf16_t), tile_vec);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_store_upper_row_powermma_(tile_vec[tile_row].u8s, row + tile_row, column, vector_count,
                                                  sizeof(nk_f32_t), result, result_stride);
        }
    }
    return nk_success_k;
}

#pragma endregion BF16 Floats

#pragma region F16 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_f16_powermma(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_bytes_powermma_(columns, depth, sizeof(nk_f16_t), sizeof(nk_f16_t), sizeof(nk_f32_t));
    return nk_success_k;
}

nk_define_cross_packed_shape_(dots, f16, powermma)

NUMKONG_API nk_status_t nk_dots_pack_f16_powermma(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const panel_bytes = nk_dots_panel_bytes_powermma_(depth, sizeof(nk_f16_t), sizeof(nk_f16_t));
    nk_size_t const norms_offset = sizeof(nk_cross_packed_buffer_header_t) +
                                   nk_size_divide_round_up_(columns, 8) * panel_bytes;
    if (columns_begin == 0)
        nk_dots_pack_header_powermma_(b_packed, columns, depth, panel_bytes / 8 / sizeof(nk_f16_t), norms_offset);
    nk_dots_pack_panels_b32_powermma_(b, columns, depth * sizeof(nk_f16_t), b_stride,
                                      (nk_u8_t *)b_packed + sizeof(nk_cross_packed_buffer_header_t), panel_bytes,
                                      columns_begin, columns_end, vec_splats((nk_u8_t)0));
    nk_f32_t *norms = (nk_f32_t *)((nk_u8_t *)b_packed + norms_offset);
    for (nk_size_t column = columns_begin; column < columns_end; ++column)
        norms[column] = nk_dots_reduce_sumsq_f16_serial_((nk_f16_t const *)((nk_u8_t const *)b + column * b_stride),
                                                         depth, sizeof(nk_f16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_workspace_size_f16_powermma(nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                   nk_size_t *bytes) {
    nk_unused_(rows), nk_unused_(columns), nk_unused_(depth);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_f16_powermma(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *workspace,
                                                    nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    if (header->capability != nk_cap_powermma_k) return nk_pack_mismatch_k;
    nk_size_t const panel_bytes = (nk_size_t)header->depth_padded_values * 8 * sizeof(nk_f16_t);
    nk_u8_t const *panels = (nk_u8_t const *)b_packed + sizeof(nk_cross_packed_buffer_header_t);
    nk_u8_t const *a_rows[8];
    nk_b256_vec_t tile_vec[8];
    for (nk_size_t row = 0; row < rows; row += 8) {
        nk_dots_clamped_rows_powermma_(a, a_stride, row, rows, 8, a_rows);
        nk_size_t const tile_rows = nk_min_of_two(rows - row, (nk_size_t)8);
        for (nk_size_t column = 0; column < columns; column += 8) {
            nk_dots_packed_tile_f16_powermma_(a_rows, panels + column / 8 * panel_bytes, depth * sizeof(nk_f16_t),
                                              tile_vec);
            nk_size_t const tile_bytes = nk_min_of_two(columns - column, (nk_size_t)8) * sizeof(nk_f32_t);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_copy_row_b8_powermma_((nk_u8_t *)c + (row + tile_row) * c_stride + column * sizeof(nk_f32_t),
                                              tile_vec[tile_row].u8s, tile_bytes);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_workspace_size_f16_powermma(nk_size_t vector_count, nk_size_t depth,
                                                                      nk_size_t rows_begin, nk_size_t rows_end,
                                                                      nk_size_t *bytes) {
    nk_unused_(vector_count), nk_unused_(depth), nk_unused_(rows_begin), nk_unused_(rows_end);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_f16_powermma(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t rows_begin, nk_size_t rows_end, void *workspace,
                                                       nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_u8_t const *row_vectors[8], *column_vectors[8];
    nk_b256_vec_t tile_vec[8];
    for (nk_size_t row = rows_begin; row < rows_end; row += 8) {
        nk_dots_clamped_rows_powermma_(vectors, stride, row, rows_end, 8, row_vectors);
        nk_size_t const tile_rows = nk_min_of_two(rows_end - row, (nk_size_t)8);
        for (nk_size_t column = row; column < vector_count; column += 8) {
            nk_dots_clamped_rows_powermma_(vectors, stride, column, vector_count, 8, column_vectors);
            nk_dots_symmetric_tile_f16_powermma_(row_vectors, column_vectors, depth * sizeof(nk_f16_t), tile_vec);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_store_upper_row_powermma_(tile_vec[tile_row].u8s, row + tile_row, column, vector_count,
                                                  sizeof(nk_f32_t), result, result_stride);
        }
    }
    return nk_success_k;
}

#pragma endregion F16 Floats

#pragma region I8 Integers

NUMKONG_API nk_status_t nk_dots_pack_size_i8_powermma(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_bytes_powermma_(columns, depth, sizeof(nk_i8_t), sizeof(nk_i8_t), sizeof(nk_u32_t));
    return nk_success_k;
}

nk_define_cross_packed_shape_(dots, i8, powermma)

NUMKONG_API nk_status_t nk_dots_pack_i8_powermma(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const panel_bytes = nk_dots_panel_bytes_powermma_(depth, sizeof(nk_i8_t), sizeof(nk_i8_t));
    nk_size_t const norms_offset = sizeof(nk_cross_packed_buffer_header_t) +
                                   nk_size_divide_round_up_(columns, 8) * panel_bytes;
    if (columns_begin == 0)
        nk_dots_pack_header_powermma_(b_packed, columns, depth, panel_bytes / 8 / sizeof(nk_i8_t), norms_offset);
    nk_dots_pack_panels_b32_powermma_(b, columns, depth * sizeof(nk_i8_t), b_stride,
                                      (nk_u8_t *)b_packed + sizeof(nk_cross_packed_buffer_header_t), panel_bytes,
                                      columns_begin, columns_end, vec_splats((nk_u8_t)0x80));
    nk_u32_t *norms = (nk_u32_t *)((nk_u8_t *)b_packed + norms_offset);
    for (nk_size_t column = columns_begin; column < columns_end; ++column)
        norms[column] = nk_dots_reduce_sumsq_i8_serial_((nk_i8_t const *)((nk_u8_t const *)b + column * b_stride),
                                                        depth, sizeof(nk_i8_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_workspace_size_i8_powermma(nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t *bytes) {
    nk_unused_(rows), nk_unused_(columns), nk_unused_(depth);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_i8_powermma(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, void *workspace, nk_size_t workspace_bytes,
                                                   nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    if (header->capability != nk_cap_powermma_k) return nk_pack_mismatch_k;
    nk_size_t const panel_bytes = (nk_size_t)header->depth_padded_values * 8 * sizeof(nk_i8_t);
    nk_u8_t const *panels = (nk_u8_t const *)b_packed + sizeof(nk_cross_packed_buffer_header_t);
    nk_u8_t const *a_rows[8];
    nk_b256_vec_t tile_vec[8];
    for (nk_size_t row = 0; row < rows; row += 8) {
        nk_dots_clamped_rows_powermma_(a, a_stride, row, rows, 8, a_rows);
        nk_size_t const tile_rows = nk_min_of_two(rows - row, (nk_size_t)8);
        for (nk_size_t column = 0; column < columns; column += 8) {
            nk_dots_packed_tile_i8_powermma_(a_rows, panels + column / 8 * panel_bytes, depth * sizeof(nk_i8_t),
                                             tile_vec);
            nk_size_t const tile_bytes = nk_min_of_two(columns - column, (nk_size_t)8) * sizeof(nk_i32_t);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_copy_row_b8_powermma_((nk_u8_t *)c + (row + tile_row) * c_stride + column * sizeof(nk_i32_t),
                                              tile_vec[tile_row].u8s, tile_bytes);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_workspace_size_i8_powermma(nk_size_t vector_count, nk_size_t depth,
                                                                     nk_size_t rows_begin, nk_size_t rows_end,
                                                                     nk_size_t *bytes) {
    nk_unused_(vector_count), nk_unused_(depth), nk_unused_(rows_begin), nk_unused_(rows_end);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_i8_powermma(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, void *workspace,
                                                      nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_u8_t const *row_vectors[8], *column_vectors[8];
    nk_b256_vec_t tile_vec[8];
    for (nk_size_t row = rows_begin; row < rows_end; row += 8) {
        nk_dots_clamped_rows_powermma_(vectors, stride, row, rows_end, 8, row_vectors);
        nk_size_t const tile_rows = nk_min_of_two(rows_end - row, (nk_size_t)8);
        for (nk_size_t column = row; column < vector_count; column += 8) {
            nk_dots_clamped_rows_powermma_(vectors, stride, column, vector_count, 8, column_vectors);
            nk_dots_symmetric_tile_i8_powermma_(row_vectors, column_vectors, depth * sizeof(nk_i8_t), tile_vec);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_store_upper_row_powermma_(tile_vec[tile_row].u8s, row + tile_row, column, vector_count,
                                                  sizeof(nk_i32_t), result, result_stride);
        }
    }
    return nk_success_k;
}

#pragma endregion I8 Integers

#pragma region U8 Integers

NUMKONG_API nk_status_t nk_dots_pack_size_u8_powermma(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_bytes_powermma_(columns, depth, sizeof(nk_u8_t), sizeof(nk_u8_t), sizeof(nk_u32_t));
    return nk_success_k;
}

nk_define_cross_packed_shape_(dots, u8, powermma)

NUMKONG_API nk_status_t nk_dots_pack_u8_powermma(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const panel_bytes = nk_dots_panel_bytes_powermma_(depth, sizeof(nk_u8_t), sizeof(nk_u8_t));
    nk_size_t const norms_offset = sizeof(nk_cross_packed_buffer_header_t) +
                                   nk_size_divide_round_up_(columns, 8) * panel_bytes;
    if (columns_begin == 0)
        nk_dots_pack_header_powermma_(b_packed, columns, depth, panel_bytes / 8 / sizeof(nk_u8_t), norms_offset);
    nk_dots_pack_panels_b32_powermma_(b, columns, depth * sizeof(nk_u8_t), b_stride,
                                      (nk_u8_t *)b_packed + sizeof(nk_cross_packed_buffer_header_t), panel_bytes,
                                      columns_begin, columns_end, vec_splats((nk_u8_t)0));
    nk_u32_t *norms = (nk_u32_t *)((nk_u8_t *)b_packed + norms_offset);
    for (nk_size_t column = columns_begin; column < columns_end; ++column)
        norms[column] = nk_dots_reduce_sumsq_u8_serial_(b + column * b_stride, depth, sizeof(nk_u8_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_workspace_size_u8_powermma(nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t *bytes) {
    nk_unused_(rows), nk_unused_(columns), nk_unused_(depth);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_u8_powermma(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, void *workspace, nk_size_t workspace_bytes,
                                                   nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    if (header->capability != nk_cap_powermma_k) return nk_pack_mismatch_k;
    nk_size_t const panel_bytes = (nk_size_t)header->depth_padded_values * 8 * sizeof(nk_u8_t);
    nk_u8_t const *panels = (nk_u8_t const *)b_packed + sizeof(nk_cross_packed_buffer_header_t);
    nk_u8_t const *a_rows[8];
    nk_b256_vec_t tile_vec[8];
    for (nk_size_t row = 0; row < rows; row += 8) {
        nk_dots_clamped_rows_powermma_(a, a_stride, row, rows, 8, a_rows);
        nk_size_t const tile_rows = nk_min_of_two(rows - row, (nk_size_t)8);
        for (nk_size_t column = 0; column < columns; column += 8) {
            nk_dots_packed_tile_u8_powermma_(a_rows, panels + column / 8 * panel_bytes, depth * sizeof(nk_u8_t),
                                             tile_vec);
            nk_size_t const tile_bytes = nk_min_of_two(columns - column, (nk_size_t)8) * sizeof(nk_u32_t);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_copy_row_b8_powermma_((nk_u8_t *)c + (row + tile_row) * c_stride + column * sizeof(nk_u32_t),
                                              tile_vec[tile_row].u8s, tile_bytes);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_workspace_size_u8_powermma(nk_size_t vector_count, nk_size_t depth,
                                                                     nk_size_t rows_begin, nk_size_t rows_end,
                                                                     nk_size_t *bytes) {
    nk_unused_(vector_count), nk_unused_(depth), nk_unused_(rows_begin), nk_unused_(rows_end);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_u8_powermma(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, void *workspace,
                                                      nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_u8_t const *row_vectors[8], *column_vectors[8];
    nk_b256_vec_t tile_vec[8];
    for (nk_size_t row = rows_begin; row < rows_end; row += 8) {
        nk_dots_clamped_rows_powermma_(vectors, stride, row, rows_end, 8, row_vectors);
        nk_size_t const tile_rows = nk_min_of_two(rows_end - row, (nk_size_t)8);
        for (nk_size_t column = row; column < vector_count; column += 8) {
            nk_dots_clamped_rows_powermma_(vectors, stride, column, vector_count, 8, column_vectors);
            nk_dots_symmetric_tile_u8_powermma_(row_vectors, column_vectors, depth * sizeof(nk_u8_t), tile_vec);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_store_upper_row_powermma_(tile_vec[tile_row].u8s, row + tile_row, column, vector_count,
                                                  sizeof(nk_u32_t), result, result_stride);
        }
    }
    return nk_success_k;
}

#pragma endregion U8 Integers

#pragma region F32 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_f32_powermma(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_bytes_powermma_(columns, depth, sizeof(nk_f32_t), sizeof(nk_f64_t), sizeof(nk_f64_t));
    return nk_success_k;
}

nk_define_cross_packed_shape_(dots, f32, powermma)

NUMKONG_API nk_status_t nk_dots_pack_f32_powermma(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const panel_bytes = nk_dots_panel_bytes_powermma_(depth, sizeof(nk_f32_t), sizeof(nk_f64_t));
    nk_size_t const depth_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const norms_offset = sizeof(nk_cross_packed_buffer_header_t) +
                                   nk_size_divide_round_up_(columns, 8) * panel_bytes;
    if (columns_begin == 0)
        nk_dots_pack_header_powermma_(b_packed, columns, depth, panel_bytes / 8 / sizeof(nk_f64_t), norms_offset);
    nk_u8_t *panels = (nk_u8_t *)b_packed + sizeof(nk_cross_packed_buffer_header_t);
    nk_u8_t const *column_rows[8];
    nk_vu32x4_t left_u32x4[4], right_u32x4[4];
    for (nk_size_t panel = nk_size_divide_round_up_(columns_begin, 8); panel < nk_size_divide_round_up_(columns_end, 8);
         ++panel) {
        nk_dots_clamped_rows_powermma_(b, b_stride, panel * 8, columns, 8, column_rows);
        nk_f64_t *panel_output = (nk_f64_t *)(panels + panel * panel_bytes);
        for (nk_size_t offset = 0; offset < depth_bytes; offset += 16, panel_output += 32) {
            nk_size_t const chunk_bytes = nk_min_of_two(depth_bytes - offset, (nk_size_t)16);
            nk_load_transposed_b32x4x4_powermma_(column_rows, offset, chunk_bytes, left_u32x4);
            nk_load_transposed_b32x4x4_powermma_(column_rows + 4, offset, chunk_bytes, right_u32x4);
            for (unsigned step = 0; step != 4; ++step) {
                nk_vf32x4_t const left_f32x4 = (nk_vf32x4_t)left_u32x4[step];
                nk_vf32x4_t const right_f32x4 = (nk_vf32x4_t)right_u32x4[step];
                nk_f64_t *step_output = panel_output + 8 * step;
                vec_xst(vec_doubleh(left_f32x4), 0, step_output);
                vec_xst(vec_doublel(left_f32x4), 0, step_output + 2);
                vec_xst(vec_doubleh(right_f32x4), 0, step_output + 4);
                vec_xst(vec_doublel(right_f32x4), 0, step_output + 6);
            }
        }
    }
    nk_f64_t *norms = (nk_f64_t *)((nk_u8_t *)b_packed + norms_offset);
    for (nk_size_t column = columns_begin; column < columns_end; ++column)
        norms[column] = nk_dots_reduce_sumsq_f32_serial_((nk_f32_t const *)((nk_u8_t const *)b + column * b_stride),
                                                         depth, sizeof(nk_f32_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_workspace_size_f32_powermma(nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                   nk_size_t *bytes) {
    nk_unused_(rows), nk_unused_(columns), nk_unused_(depth);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_f32_powermma(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *workspace,
                                                    nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    if (header->capability != nk_cap_powermma_k) return nk_pack_mismatch_k;
    nk_size_t const panel_bytes = (nk_size_t)header->depth_padded_values * 8 * sizeof(nk_f64_t);
    nk_u8_t const *panels = (nk_u8_t const *)b_packed + sizeof(nk_cross_packed_buffer_header_t);
    nk_u8_t const *a_rows[4];
    nk_b512_vec_t tile_vec[4];
    for (nk_size_t row = 0; row < rows; row += 4) {
        nk_dots_clamped_rows_powermma_(a, a_stride, row, rows, 4, a_rows);
        nk_size_t const tile_rows = nk_min_of_two(rows - row, (nk_size_t)4);
        for (nk_size_t column = 0; column < columns; column += 8) {
            nk_dots_packed_tile_f32_powermma_(a_rows, panels + column / 8 * panel_bytes, depth * sizeof(nk_f32_t),
                                              tile_vec);
            nk_size_t const tile_bytes = nk_min_of_two(columns - column, (nk_size_t)8) * sizeof(nk_f64_t);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_copy_row_b8_powermma_((nk_u8_t *)c + (row + tile_row) * c_stride + column * sizeof(nk_f64_t),
                                              tile_vec[tile_row].u8s, tile_bytes);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_workspace_size_f32_powermma(nk_size_t vector_count, nk_size_t depth,
                                                                      nk_size_t rows_begin, nk_size_t rows_end,
                                                                      nk_size_t *bytes) {
    nk_unused_(vector_count), nk_unused_(depth), nk_unused_(rows_begin), nk_unused_(rows_end);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_f32_powermma(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                       nk_size_t rows_begin, nk_size_t rows_end, void *workspace,
                                                       nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_u8_t const *row_vectors[4], *column_vectors[8];
    nk_b512_vec_t tile_vec[4];
    for (nk_size_t row = rows_begin; row < rows_end; row += 4) {
        nk_dots_clamped_rows_powermma_(vectors, stride, row, rows_end, 4, row_vectors);
        nk_size_t const tile_rows = nk_min_of_two(rows_end - row, (nk_size_t)4);
        for (nk_size_t column = row; column < vector_count; column += 8) {
            nk_dots_clamped_rows_powermma_(vectors, stride, column, vector_count, 8, column_vectors);
            nk_dots_symmetric_tile_f32_powermma_(row_vectors, column_vectors, depth * sizeof(nk_f32_t), tile_vec);
            for (nk_size_t tile_row = 0; tile_row != tile_rows; ++tile_row)
                nk_dots_store_upper_row_powermma_(tile_vec[tile_row].u8s, row + tile_row, column, vector_count,
                                                  sizeof(nk_f64_t), result, result_stride);
        }
    }
    return nk_success_k;
}

#pragma endregion F32 Floats

#endif // NUMKONG_TARGET_POWERMMA

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_PPC64_POWERMMA_
#endif // NUMKONG_ARCH_PPC64_
#endif // NUMKONG_DOTS_POWERMMA_H
