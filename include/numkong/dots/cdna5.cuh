/**
 *  @file include/numkong/dots/cdna5.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated Batched Dot Products for AMD Instinct MI400, gfx1250 and gfx1251.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/cdna3.cuh
 *  @sa include/numkong/dots/cdna4.cuh
 *
 *  The CDNA3 tile's staging, norms and epilogue on 32-lane wavefronts: eight of them own the
 *  @b [128,128] output tile, 64 × 32 each, and every lane reads the 64 bytes of one half of its
 *  row's 128-byte slab. The 16 × 16 WMMAs take BF16, F16 and 8-bit integers two to a slab, and the
 *  Float8, Float6 and Float4 codes as they are through @c v_wmma_f32_16x16x128_f8f6f4, Float6
 *  packed into its dense 6-bit stream. The integer WMMA takes each operand's signedness, so U8
 *  needs no offset, and I4 and U4 widen into it. The pack stores rows as they are, as the @c rocm
 *  capability does. Signed integer norms square through @c v_dot4_i32_iu8 with both operands
 *  marked signed, as MI400 lacks @c v_dot4_i32_i8 .
 */
#ifndef NUMKONG_DOTS_CDNA5_CUH
#define NUMKONG_DOTS_CDNA5_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_TARGET_CDNA5

#include "numkong/dots/cdna4.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

/** The tile side and block size of CDNA3, which CDNA5's tile keeps. */
enum { nk_cross_tile_cdna5_k = nk_cross_tile_cdna3_k, nk_cross_threads_cdna5_k = nk_cross_threads_cdna3_k };

#pragma endregion Configuration

#pragma region Instructions

typedef float nk_f32x8_cdna5_t __attribute__((vector_size(32)));
typedef int nk_i32x8_cdna5_t __attribute__((vector_size(32)));
typedef int nk_i32x16_cdna5_t __attribute__((vector_size(64)));
typedef __bf16 nk_bf16x16_cdna5_t __attribute__((vector_size(32)));
typedef _Float16 nk_f16x16_cdna5_t __attribute__((vector_size(32)));

NUMKONG_DEVICE nk_f32x8_cdna5_t nk_f32x8_load_cdna5_(nk_fui32_t const accumulator[8]) {
    nk_f32x8_cdna5_t sum;
#pragma unroll
    for (unsigned index = 0; index < 8; ++index) sum[index] = accumulator[index].f;
    return sum;
}

NUMKONG_DEVICE void nk_f32x8_store_cdna5_(nk_fui32_t accumulator[8], nk_f32x8_cdna5_t sum) {
#pragma unroll
    for (unsigned index = 0; index < 8; ++index) accumulator[index].f = sum[index];
}

NUMKONG_DEVICE nk_i32x8_cdna5_t nk_i32x8_load_cdna5_(nk_u32_t const words[8]) {
    nk_i32x8_cdna5_t vector;
#pragma unroll
    for (unsigned index = 0; index < 8; ++index) vector[index] = (int)words[index];
    return vector;
}

/** Up to 16 words of Float8, Float6 or Float4 codes; the WMMA reads as many as the format needs. */
NUMKONG_DEVICE nk_i32x16_cdna5_t nk_i32x16_load_cdna5_(nk_u32_t const *words, unsigned count) {
    nk_i32x16_cdna5_t vector;
#pragma unroll
    for (unsigned index = 0; index < 16; ++index) vector[index] = index < count ? (int)words[index] : 0;
    return vector;
}

NUMKONG_DEVICE void nk_wmma_bf16_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_f32x8_store_cdna5_(accumulator, __builtin_amdgcn_wmma_f32_16x16x32_bf16(
                                           false, __builtin_bit_cast(nk_bf16x16_cdna5_t, nk_i32x8_load_cdna5_(a)),
                                           false, __builtin_bit_cast(nk_bf16x16_cdna5_t, nk_i32x8_load_cdna5_(b)),
                                           (short)0, nk_f32x8_load_cdna5_(accumulator), false, false));
}

NUMKONG_DEVICE void nk_wmma_f16_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_f32x8_store_cdna5_(accumulator, __builtin_amdgcn_wmma_f32_16x16x32_f16(
                                           false, __builtin_bit_cast(nk_f16x16_cdna5_t, nk_i32x8_load_cdna5_(a)), false,
                                           __builtin_bit_cast(nk_f16x16_cdna5_t, nk_i32x8_load_cdna5_(b)), (short)0,
                                           nk_f32x8_load_cdna5_(accumulator), false, false));
}

/*  The integer WMMA's leading flags mark each operand signed: I8 × I8, U8 × U8, and U8 × I8. */

NUMKONG_DEVICE void nk_wmma_i8_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_i32x8_cdna5_t const result = __builtin_amdgcn_wmma_i32_16x16x64_iu8(
        true, nk_i32x8_load_cdna5_(a), true, nk_i32x8_load_cdna5_(b),
        nk_i32x8_load_cdna5_((nk_u32_t const *)accumulator), false, false);
#pragma unroll
    for (unsigned index = 0; index < 8; ++index) accumulator[index].i = result[index];
}

NUMKONG_DEVICE void nk_wmma_u8_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_i32x8_cdna5_t const result = __builtin_amdgcn_wmma_i32_16x16x64_iu8(
        false, nk_i32x8_load_cdna5_(a), false, nk_i32x8_load_cdna5_(b),
        nk_i32x8_load_cdna5_((nk_u32_t const *)accumulator), false, false);
#pragma unroll
    for (unsigned index = 0; index < 8; ++index) accumulator[index].i = result[index];
}

NUMKONG_DEVICE void nk_wmma_u8i8_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_i32x8_cdna5_t const result = __builtin_amdgcn_wmma_i32_16x16x64_iu8(
        false, nk_i32x8_load_cdna5_(a), true, nk_i32x8_load_cdna5_(b),
        nk_i32x8_load_cdna5_((nk_u32_t const *)accumulator), false, false);
#pragma unroll
    for (unsigned index = 0; index < 8; ++index) accumulator[index].i = result[index];
}

/*  One 16 × 16 × 128 step per format, selecting E4M3 as 0, E5M2 as 1, E2M3 as 2, E3M2 as 3 and
 *  E2M1 as 4, for 16, 16, 12, 12 and 8 words a lane. */

NUMKONG_DEVICE void nk_wmma_e4m3_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[16], nk_u32_t const b[16]) {
    nk_f32x8_store_cdna5_(accumulator, __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                           0, nk_i32x16_load_cdna5_(a, 16), 0, nk_i32x16_load_cdna5_(b, 16), (short)0,
                                           nk_f32x8_load_cdna5_(accumulator)));
}

NUMKONG_DEVICE void nk_wmma_e5m2_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[16], nk_u32_t const b[16]) {
    nk_f32x8_store_cdna5_(accumulator, __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                           1, nk_i32x16_load_cdna5_(a, 16), 1, nk_i32x16_load_cdna5_(b, 16), (short)0,
                                           nk_f32x8_load_cdna5_(accumulator)));
}

NUMKONG_DEVICE void nk_wmma_e2m3_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[12], nk_u32_t const b[12]) {
    nk_f32x8_store_cdna5_(accumulator, __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                           2, nk_i32x16_load_cdna5_(a, 12), 2, nk_i32x16_load_cdna5_(b, 12), (short)0,
                                           nk_f32x8_load_cdna5_(accumulator)));
}

NUMKONG_DEVICE void nk_wmma_e3m2_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[12], nk_u32_t const b[12]) {
    nk_f32x8_store_cdna5_(accumulator, __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                           3, nk_i32x16_load_cdna5_(a, 12), 3, nk_i32x16_load_cdna5_(b, 12), (short)0,
                                           nk_f32x8_load_cdna5_(accumulator)));
}

NUMKONG_DEVICE void nk_wmma_e2m1_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_f32x8_store_cdna5_(accumulator, __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(4, nk_i32x16_load_cdna5_(a, 8), 4,
                                                                                  nk_i32x16_load_cdna5_(b, 8), (short)0,
                                                                                  nk_f32x8_load_cdna5_(accumulator)));
}

/** Adds the four signed byte products of @p a and @p b to @p sum with CDNA5's @c v_dot4_i32_iu8 ,
 *  both operands marked signed. */
NUMKONG_DEVICE nk_i32_t nk_dot_i8x4_cdna5_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) {
    return __builtin_amdgcn_sudot4(1, (int)a, 1, (int)b, sum, 0);
}

#pragma endregion Instructions

/*  The I8 and I4 norms through MI400's own signed dot; the U8 and U4 norms are CDNA3's, and the
 *  Float ones CDNA4's. */
#pragma region Norms

NUMKONG_DEVICE void nk_i8_norm_update_cdna5_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word)
        *integer_sum = (nk_u32_t)nk_dot_i8x4_cdna5_(words[word], words[word], (nk_i32_t)*integer_sum);
}

NUMKONG_DEVICE void nk_i4_norm_update_cdna5_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_i4x8_to_i8x8_simt_(words[word], &low, &high);
        *integer_sum = (nk_u32_t)nk_dot_i8x4_cdna5_(low, low, (nk_i32_t)*integer_sum);
        *integer_sum = (nk_u32_t)nk_dot_i8x4_cdna5_(high, high, (nk_i32_t)*integer_sum);
    }
}

NUMKONG_DEVICE void nk_cross_stage_norms_i8_cdna5_(int row_squares, int column_squares,
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
            nk_i8_norm_update_cdna5_(words, &integer_norms[operand][step], &real_norms[operand][step]);
        }
}

NUMKONG_DEVICE void nk_cross_stage_norms_i4_cdna5_(int row_squares, int column_squares,
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
            nk_i4_norm_update_cdna5_(words, &integer_norms[operand][step], &real_norms[operand][step]);
        }
}

#pragma endregion Norms

#pragma region Tile

/** The 64 bytes of half @p half of staged row @p row: a WMMA operand for its lane. */
NUMKONG_DEVICE void nk_cross_load_fragment_cdna5_(unsigned char const *stage, unsigned row, unsigned half,
                                                  nk_u32_t fragment[16]) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        uint4 const bytes = *(uint4 const *)(stage + nk_cross_stage_offset_cdna3_(row, half * 4 + chunk));
        fragment[chunk * 4 + 0] = bytes.x, fragment[chunk * 4 + 1] = bytes.y;
        fragment[chunk * 4 + 2] = bytes.z, fragment[chunk * 4 + 3] = bytes.w;
    }
}

NUMKONG_DEVICE void nk_cross_accumulators_clear_cdna5_(nk_fui32_t accumulators[4][2][8]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
#pragma unroll
            for (unsigned element = 0; element < 8; ++element) accumulators[row_tile][column_tile][element].u = 0;
}

/** Loads the next slab's chunks while this one multiplies, then waits for every store of
 *  this one and every read of the stage the next refills, and loads this wavefront's
 *  fragments of @p stage. */
NUMKONG_DEVICE void nk_cross_advance_slab_cdna5_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                 nk_size_t first_column, nk_size_t slab, nk_size_t slabs,
                                                 unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k],
                                                 uint4 chunks[2][nk_cross_loads_cdna3_k], nk_u32_t a_fragments[4][16],
                                                 nk_u32_t b_fragments[2][16]) {
    unsigned const lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    unsigned const wave_row = (wave >> 2) * 64, wave_column = (wave & 3) * 32;
    unsigned const lane_row = lane & 15, half = lane >> 4;
    if (slab + 1 < slabs) nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, slab + 1);
    __syncthreads();
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
        nk_cross_load_fragment_cdna5_(stage[0], wave_row + row_tile * 16 + lane_row, half, a_fragments[row_tile]);
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
        nk_cross_load_fragment_cdna5_(stage[1], wave_column + column_tile * 16 + lane_row, half,
                                      b_fragments[column_tile]);
}

/**
 *  @brief Writes the statistics of one 128 × 128 output tile and its outputs after every slab.
 *  @sa nk_cross_finish_tile_cdna3_ for the parameters and the statistics.
 *
 *  Lane l of a 16 × 16 WMMA holds rows 8 × (l / 16) + e of column l % 16.
 */
NUMKONG_DEVICE void nk_cross_finish_tile_cdna5_(
    nk_cross_epilogue_t epilogue, nk_f32_t output_scale, nk_cross_norm_t norm, nk_f32_t norm_scale,
    nk_cross_metric_t metric, nk_diagonal_band_t band, nk_cross_tile_arguments_t const *arguments,
    nk_cross_shared_cdna3_t *shared, nk_size_t first_row, nk_size_t first_column, nk_size_t slabs,
    nk_fui32_t accumulators[4][2][8], nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k],
    nk_f32_t real_norms[2][nk_cross_loads_cdna3_k], nk_u32_t sums[2][nk_cross_loads_cdna3_k]) {
    unsigned const lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    unsigned const wave_row = (wave >> 2) * 64, wave_column = (wave & 3) * 32;
    unsigned const lane_row = lane & 15, half = lane >> 4;
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
    nk_u32_t const offset_correction = (nk_u32_t)(slabs * nk_cross_slab_bytes_cdna3_k * 16384u);

#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned element = 0; element < 8; ++element) {
            unsigned const tile_row = wave_row + row_tile * 16 + half * 8 + element;
            nk_size_t const row = first_row + tile_row;
            if (row >= arguments->rows_end) continue;
            nk_size_t column_begin, column_end;
            nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin, &column_end);
            unsigned char *output = (unsigned char *)arguments->c + row * arguments->c_stride;
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile) {
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

/*  Each folds one 128-byte slab. WMMAs over words 0-7 and 8-15 of every fragment cover the slab
 *  once, since A and B share each lane's depth mapping; the outer loop over those halves keeps
 *  dependent WMMAs 8 apart. */
/*  The 16-bit, 8-bit and Float4 formats take one WMMA of 8 words a lane per fragment half. */
#pragma region Multiplies

NUMKONG_DEVICE void nk_dots_bf16_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                 nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_wmma_bf16_cdna5_(accumulators[row_tile][column_tile], a[row_tile] + half * 8,
                                    b[column_tile] + half * 8);
}

NUMKONG_DEVICE void nk_dots_f16_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_wmma_f16_cdna5_(accumulators[row_tile][column_tile], a[row_tile] + half * 8,
                                   b[column_tile] + half * 8);
}

NUMKONG_DEVICE void nk_dots_i8_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                               nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_wmma_i8_cdna5_(accumulators[row_tile][column_tile], a[row_tile] + half * 8,
                                  b[column_tile] + half * 8);
}

NUMKONG_DEVICE void nk_dots_u8_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                               nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_wmma_u8_cdna5_(accumulators[row_tile][column_tile], a[row_tile] + half * 8,
                                  b[column_tile] + half * 8);
}

NUMKONG_DEVICE void nk_dots_e2m1_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                 nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_wmma_e2m1_cdna5_(accumulators[row_tile][column_tile], a[row_tile] + half * 8,
                                    b[column_tile] + half * 8);
}

NUMKONG_DEVICE void nk_dots_e5m2_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                 nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
            nk_wmma_e5m2_cdna5_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e4m3_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                 nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
            nk_wmma_e4m3_cdna5_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile]);
}

/** Packs 64 Float6 codes, one per byte, into the 12 dense words one WMMA operand holds. */
NUMKONG_DEVICE void nk_b8x64_to_b6x64_cdna5_(nk_u32_t const codes[16], nk_u32_t packed[12]) {
    nk_b8x32_to_b6x32_cdna4_(codes, packed);
    nk_b8x32_to_b6x32_cdna4_(codes + 8, packed + 6);
}

NUMKONG_DEVICE void nk_dots_e3m2_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                 nk_u32_t const b[2][16]) {
    nk_u32_t a_packed[4][12], b_packed[2][12];
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) nk_b8x64_to_b6x64_cdna5_(a[row_tile], a_packed[row_tile]);
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
        nk_b8x64_to_b6x64_cdna5_(b[column_tile], b_packed[column_tile]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
            nk_wmma_e3m2_cdna5_(accumulators[row_tile][column_tile], a_packed[row_tile], b_packed[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e2m3_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                 nk_u32_t const b[2][16]) {
    nk_u32_t a_packed[4][12], b_packed[2][12];
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) nk_b8x64_to_b6x64_cdna5_(a[row_tile], a_packed[row_tile]);
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
        nk_b8x64_to_b6x64_cdna5_(b[column_tile], b_packed[column_tile]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
            nk_wmma_e2m3_cdna5_(accumulators[row_tile][column_tile], a_packed[row_tile], b_packed[column_tile]);
}

/*  Nibble fragments widen into 8-bit ones, 4 words of nibbles becoming the 8 words of one WMMA. */

NUMKONG_DEVICE void nk_dots_i4_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                               nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned step = 0; step < 4; ++step) {
        nk_u32_t a_widened[4][8], b_widened[2][8];
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
#pragma unroll
            for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
                nk_i4x8_to_i8x8_simt_(a[row_tile][step * 4 + word], &a_widened[row_tile][word * 2],
                                      &a_widened[row_tile][word * 2 + 1]);
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_i4x8_to_i8x8_simt_(b[column_tile][step * 4 + word], &b_widened[column_tile][word * 2],
                                      &b_widened[column_tile][word * 2 + 1]);
        }
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_wmma_i8_cdna5_(accumulators[row_tile][column_tile], a_widened[row_tile], b_widened[column_tile]);
    }
}

NUMKONG_DEVICE void nk_dots_u4_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                               nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned step = 0; step < 4; ++step) {
        nk_u32_t a_widened[4][8], b_widened[2][8];
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
#pragma unroll
            for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
                nk_u4x8_to_u8x8_simt_(a[row_tile][step * 4 + word], &a_widened[row_tile][word * 2],
                                      &a_widened[row_tile][word * 2 + 1]);
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_u4x8_to_u8x8_simt_(b[column_tile][step * 4 + word], &b_widened[column_tile][word * 2],
                                      &b_widened[column_tile][word * 2 + 1]);
        }
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_wmma_u8_cdna5_(accumulators[row_tile][column_tile], a_widened[row_tile], b_widened[column_tile]);
    }
}

#pragma endregion Multiplies

#pragma region Tiles

/** The GEMM of one 128 × 128 output tile of BF16 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_bf16_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_bf16_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_bf16_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of F16 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_f16_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_f16_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_f16_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of E5M2 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_e5m2_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_e5m2_cdna4_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_e5m2_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of E4M3 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_e4m3_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_e4m3_cdna4_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_e4m3_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, 65536.0f, metric, band,
                                    arguments, &shared, first_row, first_column, slabs, accumulators, integer_norms,
                                    real_norms, sums);
    }
}

/** The GEMM of one 128 × 128 output tile of E3M2 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_e3m2_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_e3m2_cdna4_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_e3m2_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, 16777216.0f, metric, band,
                                    arguments, &shared, first_row, first_column, slabs, accumulators, integer_norms,
                                    real_norms, sums);
    }
}

/** The GEMM of one 128 × 128 output tile of E2M3 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_e2m3_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_e2m3_cdna4_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_e2m3_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, 0.015625f, metric, band,
                                    arguments, &shared, first_row, first_column, slabs, accumulators, integer_norms,
                                    real_norms, sums);
    }
}

/** The GEMM of one 128 × 128 output tile of E2M1 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_e2m1_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_e2m1_cdna4_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_e2m1_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, 0.25f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of I8 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_i8_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_i8_cdna5_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_i8_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_i32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of I4 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_i4_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_i4_cdna5_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_i4_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_i32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of U8 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_u8_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_u8_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_u8_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_u32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

/** The GEMM of one 128 × 128 output tile of U4 inputs on 32-lane wavefronts, staged as
 *  the CDNA3 tile. */
NUMKONG_DEVICE void nk_cross_tile_u4_cdna5_(nk_cross_metric_t metric, nk_diagonal_band_t band,
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

        nk_fui32_t accumulators[4][2][8];
        nk_cross_accumulators_clear_cdna5_(accumulators);
        nk_u32_t integer_norms[2][nk_cross_loads_cdna3_k] = {{0}}, sums[2][nk_cross_loads_cdna3_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna3_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna3_k];
        nk_cross_load_slab_cdna3_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna3_k] = shared.stages[slab & 1];
            nk_cross_store_slab_cdna3_(stage, chunks);
            nk_cross_stage_norms_u4_cdna3_(row_squares, column_squares, chunks, integer_norms, real_norms);
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
            nk_cross_advance_slab_cdna5_(arguments, first_row, first_column, slab, slabs, stage, chunks, a_fragments,
                                         b_fragments);
            nk_dots_u4_multiply_cdna5_(accumulators, a_fragments, b_fragments);
        }
        nk_cross_finish_tile_cdna5_(nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_u32_k, 1.0f, metric, band, arguments,
                                    &shared, first_row, first_column, slabs, accumulators, integer_norms, real_norms,
                                    sums);
    }
}

#pragma endregion Tiles

#pragma region BF16

nk_define_cross_pack_rocm_(bf16, cdna5, bf16, bf16, nk_load_b8_simt_, /*norm_value_type=*/f32, nk_bf16_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, bf16, cdna5, cdna5, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_rocm_(f16, cdna5, f16, f16, nk_load_b8_simt_, /*norm_value_type=*/f32, nk_f16_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, f16, cdna5, cdna5, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_pack_rocm_(e5m2, cdna5, e5m2, e5m2, nk_load_b8_simt_, /*norm_value_type=*/f32, nk_e5m2_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e5m2, cdna5, cdna5, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_pack_rocm_(e4m3, cdna5, e4m3, e4m3, nk_load_b8_simt_, /*norm_value_type=*/f32, nk_e4m3_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e4m3, cdna5, cdna5, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_pack_rocm_(e3m2, cdna5, e3m2, e3m2, nk_load_b8_simt_, /*norm_value_type=*/f32, nk_e3m2_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e3m2, cdna5, cdna5, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_pack_rocm_(e2m3, cdna5, e2m3, e2m3, nk_load_b8_simt_, /*norm_value_type=*/f32, nk_e2m3_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e2m3, cdna5, cdna5, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_rocm_(e2m1, cdna5, e2m1x2, e2m1x2, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           nk_e2m1_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, e2m1, cdna5, cdna5, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_pack_rocm_(i8, cdna5, i8, i8, nk_load_b8_simt_, /*norm_value_type=*/u32, nk_i8_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, i8, cdna5, cdna5, i8, i8, i32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_rocm_(i4, cdna5, i4x2, i4x2, nk_load_b8_simt_, /*norm_value_type=*/u32, nk_i4_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, i4, cdna5, cdna5, i4x2, i4x2, i32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_rocm_(u8, cdna5, u8, u8, nk_load_b8_simt_, /*norm_value_type=*/u32, nk_u8_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, u8, cdna5, cdna5, u8, u8, u32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_rocm_(u4, cdna5, u4x2, u4x2, nk_load_b8_simt_, /*norm_value_type=*/u32, nk_u4_lane_sumsq_simt_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, u4, cdna5, cdna5, u4x2, u4x2, u32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion U4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_CDNA5
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_DOTS_CDNA5_CUH
