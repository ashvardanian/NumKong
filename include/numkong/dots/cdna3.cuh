/**
 *  @file include/numkong/dots/cdna3.cuh
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief SIMD-accelerated Batched Dot Products for AMD Instinct MI300, gfx942.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/rocm.cuh
 *
 *  Four wavefronts own a @b [128,128] output tile, 64 × 64 each, streaming 128-byte depth slabs
 *  through registers into two swizzled shared-memory stages, all 64 KiB of MI300's LDS. Every lane
 *  reads the 32 bytes of one quarter of its row's slab and folds them in four 16 × 16 MFMAs: BF16
 *  and F16 through @c v_mfma_f32_16x16x16 and I8 through @c v_mfma_i32_16x16x32_i8 . U8 is offset
 *  into I8 and restored from the row and column byte sums, and I4 and U4 widen into I8 at twice
 *  the MFMAs. MI300's Float8 MFMA reads the FNUZ encodings rather than OCP's, so the Float8, Float6
 *  and Float4 codes widen into F16 pairs on the @c rocm B32 tile, folded through
 *  @c v_dot2_f32_f16 . Packs keep rows as they are, as on the @c rocm capability. The staging, the
 *  epilogue and the integer norms serve the CDNA4 and CDNA5 tiles too.
 */
#ifndef NUMKONG_DOTS_CDNA3_CUH
#define NUMKONG_DOTS_CDNA3_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_ROCM_CDNA3_

#include "numkong/dots/rocm.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_cross_threads_cdna3_k = 256,
    nk_cross_tile_cdna3_k = 128,
    nk_cross_slab_bytes_cdna3_k = 128,
    nk_cross_stage_bytes_cdna3_k = nk_cross_tile_cdna3_k * nk_cross_slab_bytes_cdna3_k,
    nk_cross_loads_cdna3_k = nk_cross_stage_bytes_cdna3_k / 16 / nk_cross_threads_cdna3_k,
};

/** Shared memory of a tile: two stages, A then B in each, whose first holds the row norms, column
 *  norms, row byte sums and column byte sums, a tile's worth each, once every product is done. */
typedef union {
    unsigned char stages[2][2][nk_cross_stage_bytes_cdna3_k];
    nk_fui32_t statistics[4 * nk_cross_tile_cdna3_k];
} nk_cross_shared_cdna3_t;

#pragma endregion Configuration

#pragma region Instructions

/** Adds the four signed byte products of @p a and @p b to @p sum with CDNA3's @c v_dot4_i32_i8 . */
NUMKONG_DEVICE nk_i32_t nk_dot_i8x4_cdna3_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) {
    return __builtin_amdgcn_sdot4((int)a, (int)b, sum, 0);
}

/** Adds both F16 products of @p a and @p b to @p sum with CDNA3's @c v_dot2_f32_f16 . */
NUMKONG_DEVICE nk_f32_t nk_dot_f16x2_cdna3_(nk_u32_t a, nk_u32_t b, nk_f32_t sum) {
    union {
        nk_u32_t bits;
        _Float16_2 halves;
    } const a_pair = {a}, b_pair = {b};
    return __builtin_amdgcn_fdot2(a_pair.halves, b_pair.halves, sum, 0);
}

typedef float nk_f32x4_cdna3_t __attribute__((vector_size(16)));
typedef int nk_i32x4_cdna3_t __attribute__((vector_size(16)));
typedef short nk_i16x4_cdna3_t __attribute__((vector_size(8)));
typedef _Float16 nk_f16x4_cdna3_t __attribute__((vector_size(8)));

NUMKONG_DEVICE nk_f32x4_cdna3_t nk_f32x4_load_cdna3_(nk_fui32_t const accumulator[4]) {
    nk_f32x4_cdna3_t const sum = {accumulator[0].f, accumulator[1].f, accumulator[2].f, accumulator[3].f};
    return sum;
}

NUMKONG_DEVICE void nk_f32x4_store_cdna3_(nk_fui32_t accumulator[4], nk_f32x4_cdna3_t sum) {
    accumulator[0].f = sum[0], accumulator[1].f = sum[1], accumulator[2].f = sum[2], accumulator[3].f = sum[3];
}

/** The 8 bytes of a lane's MFMA operand, from its 2 words. */
NUMKONG_DEVICE nk_u64_t nk_b64_load_cdna3_(nk_u32_t const words[2]) {
    return (nk_u64_t)words[0] | ((nk_u64_t)words[1] << 32);
}

/*  One 16 × 16 step each, every lane holding 4 BF16 or F16 values or 8 I8 values of its row of A
 *  and its column of B. */

NUMKONG_DEVICE void nk_mfma_bf16_cdna3_(nk_fui32_t accumulator[4], nk_u32_t const a[2], nk_u32_t const b[2]) {
    nk_f32x4_store_cdna3_(accumulator, __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                                           __builtin_bit_cast(nk_i16x4_cdna3_t, nk_b64_load_cdna3_(a)),
                                           __builtin_bit_cast(nk_i16x4_cdna3_t, nk_b64_load_cdna3_(b)),
                                           nk_f32x4_load_cdna3_(accumulator), 0, 0, 0));
}

NUMKONG_DEVICE void nk_mfma_f16_cdna3_(nk_fui32_t accumulator[4], nk_u32_t const a[2], nk_u32_t const b[2]) {
    nk_f32x4_store_cdna3_(
        accumulator, __builtin_amdgcn_mfma_f32_16x16x16f16(__builtin_bit_cast(nk_f16x4_cdna3_t, nk_b64_load_cdna3_(a)),
                                                           __builtin_bit_cast(nk_f16x4_cdna3_t, nk_b64_load_cdna3_(b)),
                                                           nk_f32x4_load_cdna3_(accumulator), 0, 0, 0));
}

NUMKONG_DEVICE void nk_mfma_i8_cdna3_(nk_fui32_t accumulator[4], nk_u32_t const a[2], nk_u32_t const b[2]) {
    nk_i32x4_cdna3_t const sum = {accumulator[0].i, accumulator[1].i, accumulator[2].i, accumulator[3].i};
    nk_i32x4_cdna3_t const result = __builtin_amdgcn_mfma_i32_16x16x32_i8((long)nk_b64_load_cdna3_(a),
                                                                          (long)nk_b64_load_cdna3_(b), sum, 0, 0, 0);
    accumulator[0].i = result[0], accumulator[1].i = result[1], accumulator[2].i = result[2];
    accumulator[3].i = result[3];
}

#pragma endregion Instructions

/*  Each adds the exact squares of one 16-byte chunk of a staged row through the byte dots. */
#pragma region Norms

NUMKONG_DEVICE void nk_i8_norm_update_cdna3_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word)
        *integer_sum = (nk_u32_t)nk_dot_i8x4_cdna3_(words[word], words[word], (nk_i32_t)*integer_sum);
}

NUMKONG_DEVICE void nk_u8_norm_update_cdna3_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) *integer_sum = nk_dot_u8x4_rocm_(words[word], words[word], *integer_sum);
}

NUMKONG_DEVICE void nk_i4_norm_update_cdna3_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_i4x8_to_i8x8_simt_(words[word], &low, &high);
        *integer_sum = (nk_u32_t)nk_dot_i8x4_cdna3_(low, low, (nk_i32_t)*integer_sum);
        *integer_sum = (nk_u32_t)nk_dot_i8x4_cdna3_(high, high, (nk_i32_t)*integer_sum);
    }
}

NUMKONG_DEVICE void nk_u4_norm_update_cdna3_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_u4x8_to_u8x8_simt_(words[word], &low, &high);
        *integer_sum = nk_dot_u8x4_rocm_(low, low, *integer_sum);
        *integer_sum = nk_dot_u8x4_rocm_(high, high, *integer_sum);
    }
}

/** Adds the running squares of the staged chunks of each operand, for the dtype in the name. */
NUMKONG_DEVICE void nk_cross_stage_norms_bf16_cdna3_(int row_squares, int column_squares,
                                                     uint4 const chunks[2][nk_cross_loads_cdna3_k],
                                                     nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k],
                                                     nk_f32_t real_norms[2][nk_cross_loads_cdna3_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step) {
            if (!(operand ? column_squares : row_squares)) continue;
            uint4 const chunk = chunks[operand][step];
            nk_u32_t const words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            nk_bf16_norm_update_simt_(words, &integer_norms[operand][step], &real_norms[operand][step]);
        }
}

NUMKONG_DEVICE void nk_cross_stage_norms_f16_cdna3_(int row_squares, int column_squares,
                                                    uint4 const chunks[2][nk_cross_loads_cdna3_k],
                                                    nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k],
                                                    nk_f32_t real_norms[2][nk_cross_loads_cdna3_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step) {
            if (!(operand ? column_squares : row_squares)) continue;
            uint4 const chunk = chunks[operand][step];
            nk_u32_t const words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            nk_f16_norm_update_simt_(words, &integer_norms[operand][step], &real_norms[operand][step]);
        }
}

NUMKONG_DEVICE void nk_cross_stage_norms_i8_cdna3_(int row_squares, int column_squares,
                                                   uint4 const chunks[2][nk_cross_loads_cdna3_k],
                                                   nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k],
                                                   nk_f32_t real_norms[2][nk_cross_loads_cdna3_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step) {
            if (!(operand ? column_squares : row_squares)) continue;
            uint4 const chunk = chunks[operand][step];
            nk_u32_t const words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            nk_i8_norm_update_cdna3_(words, &integer_norms[operand][step], &real_norms[operand][step]);
        }
}

NUMKONG_DEVICE void nk_cross_stage_norms_u8_cdna3_(int row_squares, int column_squares,
                                                   uint4 const chunks[2][nk_cross_loads_cdna3_k],
                                                   nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k],
                                                   nk_f32_t real_norms[2][nk_cross_loads_cdna3_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step) {
            if (!(operand ? column_squares : row_squares)) continue;
            uint4 const chunk = chunks[operand][step];
            nk_u32_t const words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            nk_u8_norm_update_cdna3_(words, &integer_norms[operand][step], &real_norms[operand][step]);
        }
}

NUMKONG_DEVICE void nk_cross_stage_norms_i4_cdna3_(int row_squares, int column_squares,
                                                   uint4 const chunks[2][nk_cross_loads_cdna3_k],
                                                   nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k],
                                                   nk_f32_t real_norms[2][nk_cross_loads_cdna3_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step) {
            if (!(operand ? column_squares : row_squares)) continue;
            uint4 const chunk = chunks[operand][step];
            nk_u32_t const words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            nk_i4_norm_update_cdna3_(words, &integer_norms[operand][step], &real_norms[operand][step]);
        }
}

NUMKONG_DEVICE void nk_cross_stage_norms_u4_cdna3_(int row_squares, int column_squares,
                                                   uint4 const chunks[2][nk_cross_loads_cdna3_k],
                                                   nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k],
                                                   nk_f32_t real_norms[2][nk_cross_loads_cdna3_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step) {
            if (!(operand ? column_squares : row_squares)) continue;
            uint4 const chunk = chunks[operand][step];
            nk_u32_t const words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            nk_u4_norm_update_cdna3_(words, &integer_norms[operand][step], &real_norms[operand][step]);
        }
}

#pragma endregion Norms

#pragma region Staging

/** Byte offset of 16-byte chunk @p chunk of staged row @p row. The XOR keeps any 16 consecutive
 *  rows' same chunk on distinct banks, whether the LDS serves 8 or 16 lanes of 16 bytes a clock. */
NUMKONG_DEVICE unsigned nk_cross_stage_offset_cdna3_(unsigned row, unsigned chunk) {
    return row * nk_cross_slab_bytes_cdna3_k + ((chunk ^ (row & 7) ^ ((row >> 3) & 1)) << 4);
}

/** Loads the 16 bytes of @p row from @p byte, as zeros past @p depth_bytes, never reading past
 *  them, since A's stride padding may hold anything. */
NUMKONG_DEVICE uint4 nk_cross_load_chunk_cdna3_(unsigned char const *row, nk_size_t byte, nk_size_t depth_bytes) {
    if (byte + 16 <= depth_bytes) return *(uint4 const *)(row + byte);
    nk_u32_t words[4] = {0, 0, 0, 0};
#pragma unroll
    for (unsigned index = 0; index < 16; ++index)
        if (byte + index < depth_bytes) words[index / 4] |= (nk_u32_t)row[byte + index] << (index % 4 * 8);
    return make_uint4(words[0], words[1], words[2], words[3]);
}

/*  A tile stages both operands the same way, A at index 0 and B at index 1: each thread moves one
 *  16-byte chunk of rows threadIdx.x / 8 + 32 × i, chunk threadIdx.x % 8, and keeps the running
 *  squared norms and byte sums of those rows. */

/** Loads this thread's chunks of slab @p slab, A's rows from @p first_row and B's from
 *  @p first_column, as zeros past each operand's last row. */
NUMKONG_DEVICE void nk_cross_load_slab_cdna3_(uint4 chunks[2][nk_cross_loads_cdna3_k],
                                              nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                              nk_size_t first_column, nk_size_t slab) {
    nk_size_t const byte = slab * nk_cross_slab_bytes_cdna3_k + (threadIdx.x & 7) * 16;
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand) {
        unsigned char const *rows = operand ? arguments->b : arguments->a;
        nk_size_t const stride = operand ? arguments->b_stride : arguments->a_stride;
        nk_size_t const first = operand ? first_column : first_row;
        nk_size_t const end = operand ? arguments->column_count : arguments->rows_end;
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step) {
            nk_size_t const row = first + (threadIdx.x >> 3) + step * (nk_cross_threads_cdna3_k / 8);
            chunks[operand][step] = row < end && byte < arguments->depth_bytes
                                        ? nk_cross_load_chunk_cdna3_(rows + row * stride, byte, arguments->depth_bytes)
                                        : make_uint4(0, 0, 0, 0);
        }
    }
}

/** Stores this thread's chunks into @p stage, one swizzled slab per operand. */
NUMKONG_DEVICE void nk_cross_store_slab_cdna3_(unsigned char stage[2][nk_cross_stage_bytes_cdna3_k],
                                               uint4 const chunks[2][nk_cross_loads_cdna3_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step)
            *(uint4 *)(stage[operand] +
                       nk_cross_stage_offset_cdna3_((threadIdx.x >> 3) + step * (nk_cross_threads_cdna3_k / 8),
                                                    threadIdx.x & 7)) = chunks[operand][step];
}

/** Adds both operands' byte sums of this thread's staged chunks into @p sums. */
NUMKONG_DEVICE void nk_cross_stage_byte_sums_cdna3_(uint4 const chunks[2][nk_cross_loads_cdna3_k],
                                                    nk_u32_t sums[2][nk_cross_loads_cdna3_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step) {
            uint4 const chunk = chunks[operand][step];
            nk_u32_t const words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
#pragma unroll
            for (unsigned word = 0; word < 4; ++word)
                sums[operand][step] = nk_dot_u8x4_rocm_(words[word], 0x01010101u, sums[operand][step]);
        }
}

/** Merges the running sums of the 8 threads staging each row, then writes them from its first
 *  thread into @p statistics, a tile's worth each: row norms, column norms, row byte sums and
 *  column byte sums. Without @p column_squares, a metric reads its column norms from @c b_norms. */
NUMKONG_DEVICE void nk_cross_write_statistics_cdna3_(nk_cross_norm_t norm, nk_f32_t norm_scale, int row_squares,
                                                     int column_squares, int offsets,
                                                     nk_cross_tile_arguments_t const *arguments, nk_size_t first_column,
                                                     nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k],
                                                     nk_f32_t real_norms[2][nk_cross_loads_cdna3_k],
                                                     nk_u32_t sums[2][nk_cross_loads_cdna3_k], nk_fui32_t *statistics) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna3_k; ++step) {
            int const squares = operand ? column_squares : row_squares;
#pragma unroll
            for (unsigned offset = 1; offset < 8; offset <<= 1) {
                if (squares)
                    integer_norms[operand][step] += nk_shuffle_xor_u32_rocm_(integer_norms[operand][step], offset),
                        real_norms[operand][step] += nk_shuffle_xor_f32_rocm_(real_norms[operand][step], offset);
                if (offsets) sums[operand][step] += nk_shuffle_xor_u32_rocm_(sums[operand][step], offset);
            }
            if ((threadIdx.x & 7) != 0) continue;
            unsigned const tile_row = (threadIdx.x >> 3) + step * (nk_cross_threads_cdna3_k / 8);
            if (squares)
                statistics[operand * nk_cross_tile_cdna3_k + tile_row] = nk_cross_norm_finalize_simt_(
                    norm, integer_norms[operand][step], real_norms[operand][step], norm_scale);
            if (offsets) statistics[(2 + operand) * nk_cross_tile_cdna3_k + tile_row].u = sums[operand][step];
        }
    if (!row_squares || column_squares || threadIdx.x >= nk_cross_tile_cdna3_k) return;
    nk_size_t const column = first_column + threadIdx.x;
    nk_u32_t const *column_norms = (nk_u32_t const *)arguments->b_norms;
    statistics[nk_cross_tile_cdna3_k + threadIdx.x].u = column < arguments->column_count ? column_norms[column] : 0;
}

/** Writes one output of a tile: the dot product, restored from its U8 offset where it has one, or
 *  the metric from it and both squared norms, with zeros on a Gram matrix's diagonal. @p statistics
 *  holds the row norms, column norms, row byte sums and column byte sums, a tile's worth each. */
NUMKONG_DEVICE void nk_cross_store_cdna3_(nk_cross_epilogue_t epilogue, nk_f32_t output_scale, nk_diagonal_band_t band,
                                          nk_cross_metric_t metric, nk_cross_norm_t norm, nk_fui32_t sum,
                                          nk_fui32_t const *statistics, nk_u32_t offset_correction, unsigned tile_row,
                                          unsigned tile_column, nk_size_t row, nk_size_t column,
                                          unsigned char *output) {
    nk_f32_t *output_f32 = (nk_f32_t *)output;
    nk_u32_t *output_u32 = (nk_u32_t *)output;
    if (epilogue == nk_cross_epilogue_offset_u32_k)
        sum.u += 128u * (statistics[2 * nk_cross_tile_cdna3_k + tile_row].u +
                         statistics[3 * nk_cross_tile_cdna3_k + tile_column].u) -
                 offset_correction;
    if (metric == nk_cross_metric_dot_k) {
        if (epilogue == nk_cross_epilogue_f32_k) output_f32[column] = sum.f * output_scale;
        else if (epilogue == nk_cross_epilogue_i32_to_f32_k) output_f32[column] = (nk_f32_t)sum.i * output_scale;
        else output_u32[column] = sum.u;
        return;
    }
    if (nk_cross_symmetric_simt_(band) && column == row) {
        output_f32[column] = 0.0f;
        return;
    }
    nk_fui32_t const row_norm = statistics[tile_row], column_norm = statistics[nk_cross_tile_cdna3_k + tile_column];
    nk_f32_t const dot = nk_cross_dot_to_f32_simt_(sum, epilogue, output_scale);
    if (norm != nk_cross_norm_f32_k)
        output_f32[column] = nk_cross_integer_metric_simt_(metric, norm, sum.u, row_norm.u, column_norm.u);
    else if (metric == nk_cross_metric_angular_k)
        output_f32[column] = nk_f32_angular_simt_(dot, row_norm.f, column_norm.f);
    else output_f32[column] = nk_f32_euclidean_simt_(dot, row_norm.f, column_norm.f);
}

#pragma endregion Staging

#pragma region Tile

/** The 32 bytes of quarter @p quarter of staged row @p row: an MFMA operand for its lane. */
NUMKONG_DEVICE void nk_cross_load_fragment_cdna3_(unsigned char const *stage, unsigned row, unsigned quarter,
                                                  nk_u32_t fragment[8]) {
    uint4 const first = *(uint4 const *)(stage + nk_cross_stage_offset_cdna3_(row, quarter * 2));
    uint4 const second = *(uint4 const *)(stage + nk_cross_stage_offset_cdna3_(row, quarter * 2 + 1));
    fragment[0] = first.x, fragment[1] = first.y, fragment[2] = first.z, fragment[3] = first.w;
    fragment[4] = second.x, fragment[5] = second.y, fragment[6] = second.z, fragment[7] = second.w;
}

NUMKONG_DEVICE void nk_cross_accumulators_clear_cdna3_(nk_fui32_t accumulators[4][4][4]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element) accumulators[row_tile][column_tile][element].u = 0;
}

/** Loads the next slab's chunks while this one multiplies, then waits for every store of
 *  this one and every read of the stage the next refills, and loads this wavefront's
 *  fragments of @p stage. */
NUMKONG_DEVICE void nk_cross_advance_slab_cdna3_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                 nk_size_t first_column, nk_size_t slab, nk_size_t slabs,
                                                 unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k],
                                                 uint4 chunks[2][nk_cross_loads_cdna3_k], nk_u32_t a_fragments[4][8],
                                                 nk_u32_t b_fragments[4][8]) {
    unsigned const lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    unsigned const wave_row = (wave >> 1) * 64, wave_column = (wave & 1) * 64;
    unsigned const lane_row = lane & 15, quarter = lane >> 4;
    if (slab + 1 < slabs) nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, slab + 1);
    __syncthreads();
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
        nk_cross_load_fragment_cdna3_(stage[0], wave_row + row_tile * 16 + lane_row, quarter, a_fragments[row_tile]);
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
        nk_cross_load_fragment_cdna3_(stage[1], wave_column + column_tile * 16 + lane_row, quarter,
                                      b_fragments[column_tile]);
}

/**
 *  @brief Writes the statistics of one 128 × 128 output tile and its outputs after every slab.
 *  @param[in] epilogue What the accumulators hold and how they reach the output.
 *  @param[in] output_scale Undoes the power of two a widening introduced, or 1.
 *  @param[in] norm How the squared norms are stored and the metric is computed; unused for dots.
 *  @param[in] norm_scale Undoes the power of two the norm update's widening introduced, or 1.
 *
 *  Row norms, and for a Gram matrix the column norms, accumulate from the staged chunks, 8 threads
 *  per row; a packed product reads its column norms from @c b_norms. Lane l of each 16 × 16 MFMA
 *  output holds its column l % 16 in rows 4 × (l / 16) + e.
 */
NUMKONG_DEVICE void nk_cross_finish_tile_cdna3_(
    nk_cross_epilogue_t epilogue, nk_f32_t output_scale, nk_cross_norm_t norm, nk_f32_t norm_scale,
    nk_cross_metric_t metric, nk_diagonal_band_t band, nk_cross_tile_arguments_t const *arguments,
    nk_cross_shared_cdna3_t *shared, nk_size_t first_row, nk_size_t first_column, nk_size_t slabs,
    nk_fui32_t accumulators[4][4][4], nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k],
    nk_f32_t real_norms[2][nk_cross_loads_cdna3_k], nk_u32_t sums[2][nk_cross_loads_cdna3_k]) {
    unsigned const lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    unsigned const wave_row = (wave >> 1) * 64, wave_column = (wave & 1) * 64;
    unsigned const lane_row = lane & 15, quarter = lane >> 4;
    int const offsets = epilogue == nk_cross_epilogue_offset_u32_k, row_squares = metric != nk_cross_metric_dot_k;
    int const column_squares = row_squares && nk_cross_symmetric_simt_(band);

    // Row, then column, norms and byte sums go to the first stage once fragment reads retire.
    nk_fui32_t *statistics = shared->statistics;
    if (row_squares || offsets) {
        __syncthreads();
        nk_cross_write_statistics_cdna3_(norm, norm_scale, row_squares, column_squares, offsets, arguments,
                                         first_column, integer_norms, real_norms, sums, statistics);
        __syncthreads();
    }
    // Every multiplied position of a U8 offset product adds 128² beyond the codes' own product.
    nk_u32_t const offset_correction = (nk_u32_t)(slabs * nk_cross_slab_bytes_cdna3_k * 16384u);

#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            unsigned const tile_row = wave_row + row_tile * 16 + quarter * 4 + element;
            nk_size_t const row = first_row + tile_row;
            if (row >= arguments->rows_end) continue;
            nk_size_t column_begin, column_end;
            nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin, &column_end);
            unsigned char *output = (unsigned char *)arguments->c + row * arguments->c_stride;
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile) {
                unsigned const tile_column = wave_column + column_tile * 16 + lane_row;
                nk_size_t const column = first_column + tile_column;
                if (column < column_begin || column >= column_end) continue;
                nk_cross_store_cdna3_(epilogue, output_scale, band, metric, norm,
                                      accumulators[row_tile][column_tile][element], statistics, offset_correction,
                                      tile_row, tile_column, row, column, output);
            }
        }
}

#pragma endregion Tile

#if NUMKONG_TARGET_CDNA3

/*  The @c rocm B32 tile over the F16 pairs that the Float8, Float6 and Float4 codes widen into. */
#pragma region F16 Pair Tile

enum { nk_cross_threads_b32_cdna3_k = nk_cross_threads_simt_k, nk_cross_tile_b32_cdna3_k = nk_cross_tile_simt_k };

/** Adds the squares of the staged F16 pairs into the running norms of this thread's rows of A, and
 *  for a Gram matrix of B, through CDNA3's @c v_dot2_f32_f16 . */
NUMKONG_DEVICE void nk_cross_stage_norms_f16x2_cdna3_(nk_diagonal_band_t band,
                                                      nk_u32_t const a_words[nk_cross_loads_simt_k],
                                                      nk_u32_t const b_words[nk_cross_loads_simt_k],
                                                      nk_fui32_t a_norms[nk_cross_loads_simt_k],
                                                      nk_fui32_t b_norms[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        a_norms[step].f = nk_dot_f16x2_cdna3_(a_words[step], a_words[step], a_norms[step].f);
        if (nk_cross_symmetric_simt_(band))
            b_norms[step].f = nk_dot_f16x2_cdna3_(b_words[step], b_words[step], b_norms[step].f);
    }
}

/** Folds one staged slab of F16 pairs into this thread's grid of F32 sums. */
NUMKONG_DEVICE void nk_cross_fold_slab_f16x2_cdna3_(
    nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k]) {
#pragma unroll
    for (unsigned offset = 0; offset < nk_cross_slab_simt_k; ++offset) {
        nk_u32_t a_words[nk_cross_thread_tile_simt_k], b_words[nk_cross_thread_tile_simt_k];
        nk_cross_load_words_b32_rocm_(a_slab, b_slab, offset, a_words, b_words);
#pragma unroll
        for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step)
#pragma unroll
            for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step)
                sums[row_step][column_step].f = nk_dot_f16x2_cdna3_(a_words[row_step], b_words[column_step],
                                                                    sums[row_step][column_step].f);
    }
}

/** @c nk_cross_tile_e5m2_rocm_ folding F16 pairs through CDNA3's @c v_dot2_f32_f16 . */
NUMKONG_DEVICE void nk_cross_tile_e5m2_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                              nk_cross_tile_arguments_t const *arguments) {
    // One extra column breaks the 64-word stride that would put a slab's stores on one bank.
    __shared__ nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_fui32_t norms[2][nk_cross_tile_simt_k];
    nk_size_t const words = nk_size_divide_round_up_(arguments->depth, nk_cross_dimensions_f16x2_simt_k);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_simt_k)) continue;
        nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{{0}}};
        nk_fui32_t a_norms[nk_cross_loads_simt_k] = {{0}}, b_norms[nk_cross_loads_simt_k] = {{0}};
        for (nk_size_t slab = 0; slab < words; slab += nk_cross_slab_simt_k) {
            nk_u32_t a_words[nk_cross_loads_simt_k], b_words[nk_cross_loads_simt_k];
            nk_cross_stage_slab_e5m2_f16x2_simt_(arguments, first_row, first_column, slab, words, a_slab, b_slab,
                                                 a_words, b_words);
            if (metric != nk_cross_metric_dot_k)
                nk_cross_stage_norms_f16x2_cdna3_(band, a_words, b_words, a_norms, b_norms);
            __syncthreads();
            nk_cross_fold_slab_f16x2_cdna3_(a_slab, b_slab, sums);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
            nk_cross_merge_norms_f32_rocm_(band, a_norms, b_norms);
            nk_cross_publish_norms_b32_simt_(band, arguments, first_column, a_norms, b_norms, norms);
        }
        nk_cross_store_tile_f32_simt_(band, metric, arguments, first_row, first_column, sums, norms);
    }
}

/** @c nk_cross_tile_e4m3_rocm_ folding F16 pairs through CDNA3's @c v_dot2_f32_f16 . */
NUMKONG_DEVICE void nk_cross_tile_e4m3_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                              nk_cross_tile_arguments_t const *arguments) {
    // One extra column breaks the 64-word stride that would put a slab's stores on one bank.
    __shared__ nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_fui32_t norms[2][nk_cross_tile_simt_k];
    nk_size_t const words = nk_size_divide_round_up_(arguments->depth, nk_cross_dimensions_f16x2_simt_k);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_simt_k)) continue;
        nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{{0}}};
        nk_fui32_t a_norms[nk_cross_loads_simt_k] = {{0}}, b_norms[nk_cross_loads_simt_k] = {{0}};
        for (nk_size_t slab = 0; slab < words; slab += nk_cross_slab_simt_k) {
            nk_u32_t a_words[nk_cross_loads_simt_k], b_words[nk_cross_loads_simt_k];
            nk_cross_stage_slab_e4m3_f16x2_simt_(arguments, first_row, first_column, slab, words, a_slab, b_slab,
                                                 a_words, b_words);
            if (metric != nk_cross_metric_dot_k)
                nk_cross_stage_norms_f16x2_cdna3_(band, a_words, b_words, a_norms, b_norms);
            __syncthreads();
            nk_cross_fold_slab_f16x2_cdna3_(a_slab, b_slab, sums);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
            nk_cross_merge_norms_f32_rocm_(band, a_norms, b_norms);
            nk_cross_publish_norms_b32_simt_(band, arguments, first_column, a_norms, b_norms, norms);
        }
        nk_cross_store_tile_f32_simt_(band, metric, arguments, first_row, first_column, sums, norms);
    }
}

/** @c nk_cross_tile_e3m2_rocm_ folding F16 pairs through CDNA3's @c v_dot2_f32_f16 . */
NUMKONG_DEVICE void nk_cross_tile_e3m2_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                              nk_cross_tile_arguments_t const *arguments) {
    // One extra column breaks the 64-word stride that would put a slab's stores on one bank.
    __shared__ nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_fui32_t norms[2][nk_cross_tile_simt_k];
    nk_size_t const words = nk_size_divide_round_up_(arguments->depth, nk_cross_dimensions_f16x2_simt_k);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_simt_k)) continue;
        nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{{0}}};
        nk_fui32_t a_norms[nk_cross_loads_simt_k] = {{0}}, b_norms[nk_cross_loads_simt_k] = {{0}};
        for (nk_size_t slab = 0; slab < words; slab += nk_cross_slab_simt_k) {
            nk_u32_t a_words[nk_cross_loads_simt_k], b_words[nk_cross_loads_simt_k];
            nk_cross_stage_slab_e3m2_f16x2_simt_(arguments, first_row, first_column, slab, words, a_slab, b_slab,
                                                 a_words, b_words);
            if (metric != nk_cross_metric_dot_k)
                nk_cross_stage_norms_f16x2_cdna3_(band, a_words, b_words, a_norms, b_norms);
            __syncthreads();
            nk_cross_fold_slab_f16x2_cdna3_(a_slab, b_slab, sums);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
            nk_cross_merge_norms_f32_rocm_(band, a_norms, b_norms);
            nk_cross_publish_norms_b32_simt_(band, arguments, first_column, a_norms, b_norms, norms);
        }
        nk_cross_store_tile_f32_simt_(band, metric, arguments, first_row, first_column, sums, norms);
    }
}

/** @c nk_cross_tile_e2m3_rocm_ folding F16 pairs through CDNA3's @c v_dot2_f32_f16 . */
NUMKONG_DEVICE void nk_cross_tile_e2m3_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                              nk_cross_tile_arguments_t const *arguments) {
    // One extra column breaks the 64-word stride that would put a slab's stores on one bank.
    __shared__ nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_fui32_t norms[2][nk_cross_tile_simt_k];
    nk_size_t const words = nk_size_divide_round_up_(arguments->depth, nk_cross_dimensions_f16x2_simt_k);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_simt_k)) continue;
        nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{{0}}};
        nk_fui32_t a_norms[nk_cross_loads_simt_k] = {{0}}, b_norms[nk_cross_loads_simt_k] = {{0}};
        for (nk_size_t slab = 0; slab < words; slab += nk_cross_slab_simt_k) {
            nk_u32_t a_words[nk_cross_loads_simt_k], b_words[nk_cross_loads_simt_k];
            nk_cross_stage_slab_e2m3_f16x2_simt_(arguments, first_row, first_column, slab, words, a_slab, b_slab,
                                                 a_words, b_words);
            if (metric != nk_cross_metric_dot_k)
                nk_cross_stage_norms_f16x2_cdna3_(band, a_words, b_words, a_norms, b_norms);
            __syncthreads();
            nk_cross_fold_slab_f16x2_cdna3_(a_slab, b_slab, sums);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
            nk_cross_merge_norms_f32_rocm_(band, a_norms, b_norms);
            nk_cross_publish_norms_b32_simt_(band, arguments, first_column, a_norms, b_norms, norms);
        }
        nk_cross_store_tile_f32_simt_(band, metric, arguments, first_row, first_column, sums, norms);
    }
}

/** @c nk_cross_tile_e2m1_rocm_ folding F16 pairs through CDNA3's @c v_dot2_f32_f16 . */
NUMKONG_DEVICE void nk_cross_tile_e2m1_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                              nk_cross_tile_arguments_t const *arguments) {
    // One extra column breaks the 64-word stride that would put a slab's stores on one bank.
    __shared__ nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_fui32_t norms[2][nk_cross_tile_simt_k];
    nk_size_t const words = nk_size_divide_round_up_(arguments->depth, nk_cross_dimensions_f16x2_simt_k);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_simt_k)) continue;
        nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{{0}}};
        nk_fui32_t a_norms[nk_cross_loads_simt_k] = {{0}}, b_norms[nk_cross_loads_simt_k] = {{0}};
        for (nk_size_t slab = 0; slab < words; slab += nk_cross_slab_simt_k) {
            nk_u32_t a_words[nk_cross_loads_simt_k], b_words[nk_cross_loads_simt_k];
            nk_cross_stage_slab_e2m1_f16x2_simt_(arguments, first_row, first_column, slab, words, a_slab, b_slab,
                                                 a_words, b_words);
            if (metric != nk_cross_metric_dot_k)
                nk_cross_stage_norms_f16x2_cdna3_(band, a_words, b_words, a_norms, b_norms);
            __syncthreads();
            nk_cross_fold_slab_f16x2_cdna3_(a_slab, b_slab, sums);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
            nk_cross_merge_norms_f32_rocm_(band, a_norms, b_norms);
            nk_cross_publish_norms_b32_simt_(band, arguments, first_column, a_norms, b_norms, norms);
        }
        nk_cross_store_tile_f32_simt_(band, metric, arguments, first_row, first_column, sums, norms);
    }
}

#pragma endregion F16 Pair Tile

/*  Each folds one 128-byte slab in four 16 × 16 MFMAs per output tile, each reading the next 8
 *  bytes of every fragment, since A and B share each lane's depth mapping; the outer loop over
 *  those steps keeps dependent MFMAs 16 apart. */
#pragma region Multiplies

NUMKONG_DEVICE void nk_dots_bf16_multiply_cdna3_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned step = 0; step < 4; ++step)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_bf16_cdna3_(accumulators[row_tile][column_tile], a[row_tile] + step * 2,
                                    b[column_tile] + step * 2);
}

NUMKONG_DEVICE void nk_dots_f16_multiply_cdna3_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned step = 0; step < 4; ++step)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_f16_cdna3_(accumulators[row_tile][column_tile], a[row_tile] + step * 2,
                                   b[column_tile] + step * 2);
}

NUMKONG_DEVICE void nk_dots_i8_multiply_cdna3_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned step = 0; step < 4; ++step)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna3_(accumulators[row_tile][column_tile], a[row_tile] + step * 2,
                                  b[column_tile] + step * 2);
}

/** U8 codes offset by −128 into I8 by flipping each byte's top bit, which the epilogue undoes. */
NUMKONG_DEVICE void nk_dots_u8_multiply_cdna3_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned step = 0; step < 4; ++step) {
        nk_u32_t a_offset[4][2], b_offset[4][2];
#pragma unroll
        for (unsigned tile = 0; tile < 4; ++tile)
#pragma unroll
            for (unsigned word = 0; word < 2; ++word)
                a_offset[tile][word] = a[tile][step * 2 + word] ^ 0x80808080u,
                b_offset[tile][word] = b[tile][step * 2 + word] ^ 0x80808080u;
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna3_(accumulators[row_tile][column_tile], a_offset[row_tile], b_offset[column_tile]);
    }
}

/** Widens nibble fragments into I8, each word of nibbles becoming the 2 words of one MFMA. */
NUMKONG_DEVICE void nk_dots_i4_multiply_cdna3_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned step = 0; step < 8; ++step) {
        nk_u32_t a_widened[4][2], b_widened[4][2];
#pragma unroll
        for (unsigned tile = 0; tile < 4; ++tile)
            nk_i4x8_to_i8x8_simt_(a[tile][step], &a_widened[tile][0], &a_widened[tile][1]),
                nk_i4x8_to_i8x8_simt_(b[tile][step], &b_widened[tile][0], &b_widened[tile][1]);
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna3_(accumulators[row_tile][column_tile], a_widened[row_tile], b_widened[column_tile]);
    }
}

/** Widens nibble fragments into I8, each word of nibbles becoming the 2 words of one MFMA. */
NUMKONG_DEVICE void nk_dots_u4_multiply_cdna3_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned step = 0; step < 8; ++step) {
        nk_u32_t a_widened[4][2], b_widened[4][2];
#pragma unroll
        for (unsigned tile = 0; tile < 4; ++tile)
            nk_u4x8_to_u8x8_simt_(a[tile][step], &a_widened[tile][0], &a_widened[tile][1]),
                nk_u4x8_to_u8x8_simt_(b[tile][step], &b_widened[tile][0], &b_widened[tile][1]);
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna3_(accumulators[row_tile][column_tile], a_widened[row_tile], b_widened[column_tile]);
    }
}

#pragma endregion Multiplies

#pragma region Tiles

/** The GEMM of one 128 × 128 output tile of BF16 inputs on 64-lane wavefronts, streaming
 *  128-byte slabs through two swizzled stages: each slab's global loads land in registers
 *  while the previous slab multiplies, then go to the stage no wavefront is reading, so one
 *  barrier a slab suffices. */
NUMKONG_DEVICE void nk_cross_tile_bf16_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                              nk_cross_tile_arguments_t const *arguments) {
    __shared__ __attribute__((aligned(16))) nk_cross_shared_cdna3_t shared;
    nk_size_t const slabs = nk_size_divide_round_up_(arguments->depth_bytes, nk_cross_slab_bytes_cdna3_k);
    int const row_squares = metric != nk_cross_metric_dot_k;
    int const column_squares = row_squares && nk_cross_symmetric_simt_(band);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's epilogue reads the first stage, which this tile's first slab refills.
        __syncthreads();
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_cdna3_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_cdna3_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_cdna3_k)) continue;

        nk_fui32_t accumulators[4][4][4];
        nk_cross_accumulators_clear_cdna3_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_bf16_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][8], b_fragments[4][8];
            nk_cross_advance_slab_cdna3_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_bf16_multiply_cdna3_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna3_(nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of F16 inputs on 64-lane wavefronts, streaming
 *  128-byte slabs through two swizzled stages: each slab's global loads land in registers
 *  while the previous slab multiplies, then go to the stage no wavefront is reading, so one
 *  barrier a slab suffices. */
NUMKONG_DEVICE void nk_cross_tile_f16_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                             nk_cross_tile_arguments_t const *arguments) {
    __shared__ __attribute__((aligned(16))) nk_cross_shared_cdna3_t shared;
    nk_size_t const slabs = nk_size_divide_round_up_(arguments->depth_bytes, nk_cross_slab_bytes_cdna3_k);
    int const row_squares = metric != nk_cross_metric_dot_k;
    int const column_squares = row_squares && nk_cross_symmetric_simt_(band);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's epilogue reads the first stage, which this tile's first slab refills.
        __syncthreads();
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_cdna3_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_cdna3_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_cdna3_k)) continue;

        nk_fui32_t accumulators[4][4][4];
        nk_cross_accumulators_clear_cdna3_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_f16_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][8], b_fragments[4][8];
            nk_cross_advance_slab_cdna3_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_f16_multiply_cdna3_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna3_(nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of I8 inputs on 64-lane wavefronts, streaming
 *  128-byte slabs through two swizzled stages: each slab's global loads land in registers
 *  while the previous slab multiplies, then go to the stage no wavefront is reading, so one
 *  barrier a slab suffices. */
NUMKONG_DEVICE void nk_cross_tile_i8_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                            nk_cross_tile_arguments_t const *arguments) {
    __shared__ __attribute__((aligned(16))) nk_cross_shared_cdna3_t shared;
    nk_size_t const slabs = nk_size_divide_round_up_(arguments->depth_bytes, nk_cross_slab_bytes_cdna3_k);
    int const row_squares = metric != nk_cross_metric_dot_k;
    int const column_squares = row_squares && nk_cross_symmetric_simt_(band);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's epilogue reads the first stage, which this tile's first slab refills.
        __syncthreads();
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_cdna3_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_cdna3_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_cdna3_k)) continue;

        nk_fui32_t accumulators[4][4][4];
        nk_cross_accumulators_clear_cdna3_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_i8_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][8], b_fragments[4][8];
            nk_cross_advance_slab_cdna3_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_i8_multiply_cdna3_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna3_(nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_i32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of U8 inputs on 64-lane wavefronts, streaming
 *  128-byte slabs through two swizzled stages: each slab's global loads land in registers
 *  while the previous slab multiplies, then go to the stage no wavefront is reading, so one
 *  barrier a slab suffices. */
NUMKONG_DEVICE void nk_cross_tile_u8_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                            nk_cross_tile_arguments_t const *arguments) {
    __shared__ __attribute__((aligned(16))) nk_cross_shared_cdna3_t shared;
    nk_size_t const slabs = nk_size_divide_round_up_(arguments->depth_bytes, nk_cross_slab_bytes_cdna3_k);
    int const row_squares = metric != nk_cross_metric_dot_k;
    int const column_squares = row_squares && nk_cross_symmetric_simt_(band);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's epilogue reads the first stage, which this tile's first slab refills.
        __syncthreads();
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_cdna3_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_cdna3_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_cdna3_k)) continue;

        nk_fui32_t accumulators[4][4][4];
        nk_cross_accumulators_clear_cdna3_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_u8_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_cross_stage_byte_sums_cdna3_(chunks, sums);
            nk_u32_t a_fragments[4][8], b_fragments[4][8];
            nk_cross_advance_slab_cdna3_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_u8_multiply_cdna3_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna3_(nk_cross_epilogue_offset_u32_k, 1.0f, nk_cross_norm_u32_k, 1.0f, metric, band,
                                    arguments, &shared, first_row, first_column, slabs, accumulators, integer_norms,
                                    real_norms, sums);
    }
}

/** The GEMM of one 128 × 128 output tile of I4 inputs on 64-lane wavefronts, streaming
 *  128-byte slabs through two swizzled stages: each slab's global loads land in registers
 *  while the previous slab multiplies, then go to the stage no wavefront is reading, so one
 *  barrier a slab suffices. */
NUMKONG_DEVICE void nk_cross_tile_i4_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                            nk_cross_tile_arguments_t const *arguments) {
    __shared__ __attribute__((aligned(16))) nk_cross_shared_cdna3_t shared;
    nk_size_t const slabs = nk_size_divide_round_up_(arguments->depth_bytes, nk_cross_slab_bytes_cdna3_k);
    int const row_squares = metric != nk_cross_metric_dot_k;
    int const column_squares = row_squares && nk_cross_symmetric_simt_(band);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's epilogue reads the first stage, which this tile's first slab refills.
        __syncthreads();
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_cdna3_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_cdna3_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_cdna3_k)) continue;

        nk_fui32_t accumulators[4][4][4];
        nk_cross_accumulators_clear_cdna3_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_i4_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][8], b_fragments[4][8];
            nk_cross_advance_slab_cdna3_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_i4_multiply_cdna3_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna3_(nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_i32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of U4 inputs on 64-lane wavefronts, streaming
 *  128-byte slabs through two swizzled stages: each slab's global loads land in registers
 *  while the previous slab multiplies, then go to the stage no wavefront is reading, so one
 *  barrier a slab suffices. */
NUMKONG_DEVICE void nk_cross_tile_u4_cdna3_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                            nk_cross_tile_arguments_t const *arguments) {
    __shared__ __attribute__((aligned(16))) nk_cross_shared_cdna3_t shared;
    nk_size_t const slabs = nk_size_divide_round_up_(arguments->depth_bytes, nk_cross_slab_bytes_cdna3_k);
    int const row_squares = metric != nk_cross_metric_dot_k;
    int const column_squares = row_squares && nk_cross_symmetric_simt_(band);

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's epilogue reads the first stage, which this tile's first slab refills.
        __syncthreads();
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_cdna3_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_cdna3_k;
        if (nk_cross_tile_outside_rocm_(band, first_row, first_column, nk_cross_tile_cdna3_k)) continue;

        nk_fui32_t accumulators[4][4][4];
        nk_cross_accumulators_clear_cdna3_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_u4_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][8], b_fragments[4][8];
            nk_cross_advance_slab_cdna3_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_u4_multiply_cdna3_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna3_(nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_u32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

#pragma endregion Tiles

#pragma region BF16

nk_define_cross_pack_rocm_(bf16, cdna3, bf16, bf16, nk_load_b8_simt_, /*norm_value_type=*/f32, nk_bf16_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, bf16, cdna3, cdna3, bf16, bf16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_rocm_(f16, cdna3, f16, f16, nk_load_b8_simt_, /*norm_value_type=*/f32, nk_f16_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, f16, cdna3, cdna3, f16, f16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region Float8, Float6 and Float4

nk_define_cross_pack_rocm_(e5m2, cdna3, e5m2, e5m2, nk_load_b8_simt_, f32, nk_e5m2_lane_sumsq_simt_, 16, 1)
nk_define_cross_rocm_(dot, e5m2, cdna3, b32_cdna3, e5m2, e5m2, f32, 16, 1)
nk_define_cross_pack_rocm_(e4m3, cdna3, e4m3, e4m3, nk_load_b8_simt_, f32, nk_e4m3_lane_sumsq_simt_, 16, 1)
nk_define_cross_rocm_(dot, e4m3, cdna3, b32_cdna3, e4m3, e4m3, f32, 16, 1)
nk_define_cross_pack_rocm_(e3m2, cdna3, e3m2, e3m2, nk_load_b8_simt_, f32, nk_e3m2_lane_sumsq_simt_, 16, 1)
nk_define_cross_rocm_(dot, e3m2, cdna3, b32_cdna3, e3m2, e3m2, f32, 16, 1)
nk_define_cross_pack_rocm_(e2m3, cdna3, e2m3, e2m3, nk_load_b8_simt_, f32, nk_e2m3_lane_sumsq_simt_, 16, 1)
nk_define_cross_rocm_(dot, e2m3, cdna3, b32_cdna3, e2m3, e2m3, f32, 16, 1)
nk_define_cross_pack_rocm_(e2m1, cdna3, e2m1x2, e2m1x2, nk_load_b8_simt_, f32, nk_e2m1_lane_sumsq_simt_, 32, 2)
nk_define_cross_rocm_(dot, e2m1, cdna3, b32_cdna3, e2m1x2, e2m1x2, f32, 32, 2)

#pragma endregion Float8, Float6 and Float4

#pragma region I8

nk_define_cross_pack_rocm_(i8, cdna3, i8, i8, nk_load_b8_simt_, /*norm_value_type=*/u32, nk_i8_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, i8, cdna3, cdna3, i8, i8, i32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_rocm_(i4, cdna3, i4x2, i4x2, nk_load_b8_simt_, /*norm_value_type=*/u32, nk_i4_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, i4, cdna3, cdna3, i4x2, i4x2, i32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_rocm_(u8, cdna3, u8, u8, nk_load_b8_simt_, /*norm_value_type=*/u32, nk_u8_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, u8, cdna3, cdna3, u8, u8, u32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_rocm_(u4, cdna3, u4x2, u4x2, nk_load_b8_simt_, /*norm_value_type=*/u32, nk_u4_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, u4, cdna3, cdna3, u4x2, u4x2, u32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion U4

#endif // NUMKONG_TARGET_CDNA3

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA3_
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_DOTS_CDNA3_CUH
