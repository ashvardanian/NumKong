/**
 *  @file include/numkong/simt.cuh
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief Device helpers every GPU family shares on the SIMT cores of CUDA and ROCm devices:
 *      operand swizzles and diagonal-band coverage.
 *
 *  @sa include/numkong/cuda.cuh
 *  @sa include/numkong/rocm.cuh
 */
#ifndef NUMKONG_SIMT_CUH
#define NUMKONG_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/types.h" // `nk_diagonal_band_t`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Addressing

/** Byte @p byte of row @p row of an operand in the 32, 64 or 128-byte swizzle that @p swizzle_bytes
 *  names, rows that wide: each 16-byte chunk XORs with the row's 128-byte line, modulo the chunks
 *  of a row, as the NVIDIA tensor cores and copy engines lay operands out. */
NUMKONG_DEVICE unsigned nk_swizzled_offset_simt_(unsigned row, unsigned byte, unsigned swizzle_bytes) {
    unsigned const line = (row / (128 / swizzle_bytes)) & (swizzle_bytes / 16 - 1);
    return row * swizzle_bytes + (((byte >> 4) ^ line) << 4) + (byte & 15);
}

/** Byte @p byte of row @p row of an operand deeper than one swizzle: blocks of @p panel_bytes, each
 *  holding @p swizzle_bytes of every row in the swizzle of that width. */
NUMKONG_DEVICE unsigned nk_swizzled_panel_offset_simt_(unsigned row, unsigned byte, unsigned swizzle_bytes,
                                                       unsigned panel_bytes) {
    return byte / swizzle_bytes * panel_bytes + nk_swizzled_offset_simt_(row, byte % swizzle_bytes, swizzle_bytes);
}

#pragma endregion Addressing

#pragma region Diagonal Band

/** Writes the half-open range of the first @p columns that @p band shows to @p row, which may sit
 *  before column 0 or past the last column; @p column_begin equals @p column_end when none is. */
NUMKONG_DEVICE void nk_diagonal_band_row_range_simt_(nk_diagonal_band_t band, nk_i64_t row, nk_size_t columns,
                                                     nk_size_t *column_begin, nk_size_t *column_end) {
    nk_size_t const superdiagonals = band.superdiagonals;
    if (row < 0) {
        nk_size_t const lag = (nk_size_t)-row;
        *column_begin = 0;
        *column_end = superdiagonals < lag ? 0 : superdiagonals - lag < columns ? superdiagonals - lag + 1 : columns;
        return;
    }
    nk_size_t const lead = (nk_size_t)row;
    nk_size_t const begin = lead > band.subdiagonals ? lead - band.subdiagonals : 0;
    *column_begin = begin < columns ? begin : columns;
    *column_end = lead < columns && superdiagonals < columns - lead - 1 ? lead + superdiagonals + 1 : columns;
}

/** Classifies the tile of @p rows rows from @p row by @p columns columns from @p column against
 *  @p band, so tiled kernels skip outside tiles and run inside ones without per-cell masks.
 *  Symmetric dots skip the outside of band (0, NUMKONG_SIZE_MAX) as mirrored, while attention skips
 *  it as forbidden. */
NUMKONG_DEVICE nk_diagonal_band_coverage_t nk_diagonal_band_tile_coverage_simt_(nk_diagonal_band_t band, nk_i64_t row,
                                                                                nk_size_t rows, nk_size_t column,
                                                                                nk_size_t columns) {
    if (rows == 0 || columns == 0) return nk_diagonal_band_outside_k;
    // Visible ranges only grow with the row, so the first and last rows bound the whole tile.
    nk_size_t const column_end = column + columns;
    nk_size_t first_begin, first_end, last_begin, last_end;
    nk_diagonal_band_row_range_simt_(band, row, column_end, &first_begin, &first_end);
    nk_diagonal_band_row_range_simt_(band, row + (nk_i64_t)rows - 1, column_end, &last_begin, &last_end);
    if (last_end <= column || first_begin >= column_end) return nk_diagonal_band_outside_k;
    if (last_begin <= column && first_end >= column_end) return nk_diagonal_band_inside_k;
    return nk_diagonal_band_crossing_k;
}

/** Bit i set when key @p first_key + i of a segment of @p key_count keys is visible to @p row, for
 *  the 32 keys a thread holds of a crossing tile, so each score tests one bit instead of
 *  recomputing the band. */
NUMKONG_DEVICE nk_u32_t nk_diagonal_band_row_mask_simt_(nk_diagonal_band_t band, nk_i64_t row, nk_size_t first_key,
                                                        nk_size_t key_count) {
    nk_size_t key_begin, key_end;
    nk_diagonal_band_row_range_simt_(band, row, key_count, &key_begin, &key_end);
    nk_size_t const begin = key_begin > first_key ? key_begin - first_key : 0;
    nk_size_t const end = key_end > first_key + 32 ? 32 : key_end > first_key ? key_end - first_key : 0;
    return begin < end ? (0xFFFFFFFFu >> (32 - (unsigned)(end - begin))) << (unsigned)begin : 0u;
}

#pragma endregion Diagonal Band

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_SIMT_CUH
