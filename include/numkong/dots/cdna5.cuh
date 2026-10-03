/**
 *  @file include/numkong/dots/cdna5.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated Batched Dot Products for AMD Instinct MI400, gfx1250 and gfx1251.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/cdna4.cuh
 *
 *  The CDNA4 tile's staging, norms and epilogue on 32-lane wavefronts: eight of them own the
 *  @b [128,128] output tile, 64 × 32 each, and every lane reads the 64 bytes of one half of its
 *  row's 128-byte slab. The 16 × 16 WMMAs take BF16, F16 and 8-bit integers two to a slab, and the
 *  Float8, Float6 and Float4 codes as they are through @c v_wmma_f32_16x16x128_f8f6f4, Float6
 *  packed into its dense 6-bit stream. The integer WMMA takes each operand's signedness, so U8
 *  needs no offset, and I4 and U4 widen into it. The pack stores rows as they are, as the @c rocm
 *  capability does. Only the MI400 code objects carry these kernels.
 */
#ifndef NUMKONG_DOTS_CDNA5_CUH
#define NUMKONG_DOTS_CDNA5_CUH

#if NUMKONG_TARGET_CDNA5

#include "numkong/dots/cdna4.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

/** The tile side and block size of CDNA4, which CDNA5's tile keeps. */
enum { nk_cross_tile_cdna5_k = nk_cross_tile_cdna4_k, nk_cross_threads_cdna5_k = nk_cross_threads_cdna4_k };

/** Folds one wavefront's fragments of a 128-byte slab: A as 4 row tiles of 16 and B as 2 column
 *  tiles of 16, each lane holding the 64 bytes of one half of its row's slab. */
typedef void (*nk_cross_multiply_cdna5_t)(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                          nk_u32_t const b[2][16]);

#pragma endregion Configuration

#pragma region Instructions

#if defined(__gfx1250__) || defined(__gfx1251__)

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

#else

/*  Other device passes and the host pass get trapping bodies, so a code object picked for the
 *  wrong device fails loudly rather than returning zeros. */
NUMKONG_DEVICE void nk_wmma_bf16_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_wmma_f16_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_wmma_i8_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_wmma_u8_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_wmma_u8i8_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_wmma_e4m3_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[16], nk_u32_t const b[16]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_wmma_e5m2_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[16], nk_u32_t const b[16]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_wmma_e2m3_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[12], nk_u32_t const b[12]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_wmma_e3m2_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[12], nk_u32_t const b[12]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_wmma_e2m1_cdna5_(nk_fui32_t accumulator[8], nk_u32_t const a[8], nk_u32_t const b[8]) {
    __builtin_trap();
}

#endif // defined(__gfx1250__) || defined(__gfx1251__)

#pragma endregion Instructions

#pragma region Tile

/** The 64 bytes of half @p half of staged row @p row: a WMMA operand for its lane. */
NUMKONG_DEVICE void nk_cross_load_fragment_cdna5_(unsigned char const *stage, unsigned row, unsigned half,
                                                  nk_u32_t fragment[16]) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        uint4 const bytes = *(uint4 const *)(stage + nk_cross_stage_offset_cdna4_(row, half * 4 + chunk));
        fragment[chunk * 4 + 0] = bytes.x, fragment[chunk * 4 + 1] = bytes.y;
        fragment[chunk * 4 + 2] = bytes.z, fragment[chunk * 4 + 3] = bytes.w;
    }
}

/**
 *  @brief The whole GEMM of one 128 × 128 output tile on 32-lane wavefronts, shared by every dtype
 *      and every metric.
 *  @sa nk_cross_tile_cdna4_ for the parameters, the staging and the epilogue.
 *
 *  Lane l of a 16 × 16 WMMA holds rows 8 × (l / 16) + e of column l % 16.
 */
NUMKONG_DEVICE void nk_cross_tile_cdna5_(nk_cross_multiply_cdna5_t multiply, nk_cross_epilogue_t epilogue,
                                         nk_f32_t output_scale, nk_cross_norm_t norm,
                                         nk_cross_norm_update_t norm_update, nk_f32_t norm_scale,
                                         nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                         nk_cross_tile_arguments_t const *arguments) {
    __shared__ __attribute__((aligned(16))) unsigned char staged[2][2][nk_cross_stage_bytes_cdna4_k];

    unsigned const lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    unsigned const wave_row = (wave >> 2) * 64, wave_column = (wave & 3) * 32;
    unsigned const lane_row = lane & 15, half = lane >> 4;
    nk_size_t const slabs = nk_size_divide_round_up_(arguments->depth_bytes, nk_cross_slab_bytes_cdna4_k);
    int const offsets = epilogue == nk_cross_epilogue_offset_u32_k, row_squares = metric != nk_cross_metric_dot_k;
    int const column_squares = row_squares && triangle == nk_cross_triangle_upper_k;

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's epilogue reads the first stage, which this tile's first slab refills.
        __syncthreads();
        nk_size_t const first_row = arguments->row_start + tile / arguments->column_tiles * nk_cross_tile_cdna4_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_cdna4_k;
        if (triangle == nk_cross_triangle_upper_k && first_column + nk_cross_tile_cdna4_k <= first_row) continue;

        nk_fui32_t accumulators[4][2][8];
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
#pragma unroll
                for (unsigned element = 0; element < 8; ++element) accumulators[row_tile][column_tile][element].u = 0;
        nk_u32_t integer_norms[2][nk_cross_loads_cdna4_k] = {{0}}, sums[2][nk_cross_loads_cdna4_k] = {{0}};
        nk_f32_t real_norms[2][nk_cross_loads_cdna4_k] = {{0}};
        uint4 chunks[2][nk_cross_loads_cdna4_k];
        nk_cross_load_slab_cdna4_(chunks, arguments, first_row, first_column, 0);
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            unsigned char (*stage)[nk_cross_stage_bytes_cdna4_k] = staged[slab & 1];
            nk_cross_store_slab_cdna4_(stage, chunks);
            nk_cross_stage_statistics_cdna4_(norm_update, row_squares, column_squares, offsets, chunks, integer_norms,
                                             real_norms, sums);
            if (slab + 1 < slabs) nk_cross_load_slab_cdna4_(chunks, arguments, first_row, first_column, slab + 1);
            // Retires the stores above, and every read of the stage the next slab refills.
            __syncthreads();
            nk_u32_t a_fragments[4][16], b_fragments[2][16];
#pragma unroll
            for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
                nk_cross_load_fragment_cdna5_(stage[0], wave_row + row_tile * 16 + lane_row, half,
                                              a_fragments[row_tile]);
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                nk_cross_load_fragment_cdna5_(stage[1], wave_column + column_tile * 16 + lane_row, half,
                                              b_fragments[column_tile]);
            multiply(accumulators, a_fragments, b_fragments);
        }

        // Row, then column, norms and byte sums go to the first stage once fragment reads retire.
        nk_fui32_t *statistics = (nk_fui32_t *)staged[0][0];
        if (row_squares || offsets) {
            __syncthreads();
            nk_cross_write_statistics_cdna4_(norm, norm_scale, row_squares, column_squares, offsets, arguments,
                                             first_column, integer_norms, real_norms, sums, statistics);
            __syncthreads();
        }
        nk_u32_t const offset_correction = (nk_u32_t)(slabs * nk_cross_slab_bytes_cdna4_k * 16384u);

#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned element = 0; element < 8; ++element) {
                unsigned const tile_row = wave_row + row_tile * 16 + half * 8 + element;
                nk_size_t const row = first_row + tile_row;
                if (row >= arguments->row_end) continue;
                unsigned char *output = (unsigned char *)arguments->c + row * arguments->c_stride;
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 2; ++column_tile) {
                    unsigned const tile_column = wave_column + column_tile * 16 + lane_row;
                    nk_size_t const column = first_column + tile_column;
                    if (column >= arguments->column_count || (triangle == nk_cross_triangle_upper_k && column < row))
                        continue;
                    nk_cross_store_cdna4_(epilogue, output_scale, triangle, metric, norm,
                                          accumulators[row_tile][column_tile][element], statistics, offset_correction,
                                          tile_row, tile_column, row, column, output);
                }
            }
    }
}

#pragma endregion Tile

/*  Each folds one 128-byte slab. WMMAs over words 0-7 and 8-15 of every fragment cover the slab
 *  once, since A and B share each lane's depth mapping; the outer loop over those halves keeps
 *  dependent WMMAs 8 apart. */
#pragma region Multiplies

/** One WMMA of 8 words a lane per fragment half, for the 16-bit, 8-bit and Float4 formats. */
NUMKONG_DEVICE void nk_dots_halves_multiply_cdna5_(void (*wmma)(nk_fui32_t *, nk_u32_t const *, nk_u32_t const *),
                                                   nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                   nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                wmma(accumulators[row_tile][column_tile], a[row_tile] + half * 8, b[column_tile] + half * 8);
}

NUMKONG_DEVICE void nk_dots_bf16_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                 nk_u32_t const b[2][16]) {
    nk_dots_halves_multiply_cdna5_(nk_wmma_bf16_cdna5_, accumulators, a, b);
}

NUMKONG_DEVICE void nk_dots_f16_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                nk_u32_t const b[2][16]) {
    nk_dots_halves_multiply_cdna5_(nk_wmma_f16_cdna5_, accumulators, a, b);
}

NUMKONG_DEVICE void nk_dots_i8_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                               nk_u32_t const b[2][16]) {
    nk_dots_halves_multiply_cdna5_(nk_wmma_i8_cdna5_, accumulators, a, b);
}

NUMKONG_DEVICE void nk_dots_u8_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                               nk_u32_t const b[2][16]) {
    nk_dots_halves_multiply_cdna5_(nk_wmma_u8_cdna5_, accumulators, a, b);
}

NUMKONG_DEVICE void nk_dots_e2m1_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                 nk_u32_t const b[2][16]) {
    nk_dots_halves_multiply_cdna5_(nk_wmma_e2m1_cdna5_, accumulators, a, b);
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

/** Widens nibble fragments into 8-bit ones, 4 words of nibbles becoming the 8 words of one WMMA. */
NUMKONG_DEVICE void nk_dots_widened_multiply_cdna5_(nk_cross_widen_t widen,
                                                    void (*wmma)(nk_fui32_t *, nk_u32_t const *, nk_u32_t const *),
                                                    nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                                    nk_u32_t const b[2][16]) {
#pragma unroll
    for (unsigned step = 0; step < 4; ++step) {
        nk_u32_t a_widened[4][8], b_widened[2][8];
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
#pragma unroll
            for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
                widen(a[row_tile][step * 4 + word], &a_widened[row_tile][word * 2], &a_widened[row_tile][word * 2 + 1]);
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                widen(b[column_tile][step * 4 + word], &b_widened[column_tile][word * 2],
                      &b_widened[column_tile][word * 2 + 1]);
        }
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 2; ++column_tile)
                wmma(accumulators[row_tile][column_tile], a_widened[row_tile], b_widened[column_tile]);
    }
}

NUMKONG_DEVICE void nk_dots_i4_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                               nk_u32_t const b[2][16]) {
    nk_dots_widened_multiply_cdna5_(nk_i4x8_to_i8x8_, nk_wmma_i8_cdna5_, accumulators, a, b);
}

NUMKONG_DEVICE void nk_dots_u4_multiply_cdna5_(nk_fui32_t accumulators[4][2][8], nk_u32_t const a[4][16],
                                               nk_u32_t const b[2][16]) {
    nk_dots_widened_multiply_cdna5_(nk_u4x8_to_u8x8_, nk_wmma_u8_cdna5_, accumulators, a, b);
}

#pragma endregion Multiplies

#pragma region BF16

nk_define_cross_pack_rocm_(bf16, cdna5, bf16, bf16, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, bf16, cdna5, cdna5, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_bf16_multiply_cdna5_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_rocm_(f16, cdna5, f16, f16, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, f16, cdna5, cdna5, f16, f16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1,
                      nk_dots_f16_multiply_cdna5_, nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                      NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_pack_rocm_(e5m2, cdna5, e5m2, e5m2, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e5m2, cdna5, cdna5, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e5m2_multiply_cdna5_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_pack_rocm_(e4m3, cdna5, e4m3, e4m3, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e4m3, cdna5, cdna5, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e4m3_multiply_cdna5_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_pack_rocm_(e3m2, cdna5, e3m2, e3m2, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e3m2, cdna5, cdna5, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e3m2_multiply_cdna5_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_pack_rocm_(e2m3, cdna5, e2m3, e2m3, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e2m3, cdna5, cdna5, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e2m3_multiply_cdna5_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_rocm_(e2m1, cdna5, e2m1x2, e2m1x2, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, e2m1, cdna5, cdna5, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_e2m1_multiply_cdna5_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_pack_rocm_(i8, cdna5, i8, i8, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, i8, cdna5, cdna5, i8, i8, i32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1,
                      nk_dots_i8_multiply_cdna5_, nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                      NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_rocm_(i4, cdna5, i4x2, i4x2, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, i4, cdna5, cdna5, i4x2, i4x2, i32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_i4_multiply_cdna5_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_rocm_(u8, cdna5, u8, u8, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, u8, cdna5, cdna5, u8, u8, u32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1,
                      nk_dots_u8_multiply_cdna5_, nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                      NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_rocm_(u4, cdna5, u4x2, u4x2, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, u4, cdna5, cdna5, u4x2, u4x2, u32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_u4_multiply_cdna5_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion U4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_CDNA5
#endif // NUMKONG_DOTS_CDNA5_CUH
