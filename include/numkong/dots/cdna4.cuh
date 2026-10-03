/**
 *  @file include/numkong/dots/cdna4.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated Batched Dot Products for AMD Instinct MI350, gfx950.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/rocm.cuh
 *
 *  Four wavefronts own a @b [128,128] output tile, 64 × 64 each, streaming 128-byte depth slabs
 *  through registers into two swizzled shared-memory stages. Every lane reads the 32 bytes of one
 *  quarter of its row's slab, so dtypes differ only in how those fragments are multiplied: BF16,
 *  F16 and I8 take two 16 × 16 MFMAs per slab, and the Float8, Float6 and Float4 codes go to
 *  @c v_mfma_scale_f32_16x16x128_f8f6f4 as they are, with every block scale at 2⁰. Float6 codes are
 *  packed from their bytes into the dense 6-bit stream the MFMA reads. U8 is offset into I8 and
 *  restored from the row and column byte sums, and I4 and U4 widen into I8. The pack stores rows as
 *  they are, as the @c rocm capability does. Only the gfx950 code object carries these kernels; the
 *  staging, norms and epilogue serve the CDNA5 tile too.
 */
#ifndef NUMKONG_DOTS_CDNA4_CUH
#define NUMKONG_DOTS_CDNA4_CUH

#if NUMKONG_ARCH_ROCM_CDNA4_

#include "numkong/dots/rocm.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_cross_threads_cdna4_k = 256,
    nk_cross_tile_cdna4_k = 128,
    nk_cross_slab_bytes_cdna4_k = 128,
    nk_cross_stage_bytes_cdna4_k = nk_cross_tile_cdna4_k * nk_cross_slab_bytes_cdna4_k,
    nk_cross_loads_cdna4_k = nk_cross_stage_bytes_cdna4_k / 16 / nk_cross_threads_cdna4_k,
};

/** Folds one wavefront's fragments of a 128-byte slab: A as 4 row tiles of 16 and B as 4 column
 *  tiles of 16, each lane holding the 32 bytes of one quarter of its row's slab. */
typedef void (*nk_cross_multiply_cdna4_t)(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                          nk_u32_t const b[4][8]);

#pragma endregion Configuration

#pragma region Instructions

#if defined(__gfx950__)

typedef float nk_f32x4_cdna4_t __attribute__((vector_size(16)));
typedef int nk_i32x4_cdna4_t __attribute__((vector_size(16)));
typedef int nk_i32x8_cdna4_t __attribute__((vector_size(32)));
typedef __bf16 nk_bf16x8_cdna4_t __attribute__((vector_size(16)));
typedef _Float16 nk_f16x8_cdna4_t __attribute__((vector_size(16)));

/** Every E8M0 block scale at 127, 2⁰, whichever byte the MFMA selects. */
enum { nk_unit_scales_cdna4_k = 0x7F7F7F7F };

NUMKONG_DEVICE nk_f32x4_cdna4_t nk_f32x4_load_cdna4_(nk_fui32_t const accumulator[4]) {
    nk_f32x4_cdna4_t const sum = {accumulator[0].f, accumulator[1].f, accumulator[2].f, accumulator[3].f};
    return sum;
}

NUMKONG_DEVICE void nk_f32x4_store_cdna4_(nk_fui32_t accumulator[4], nk_f32x4_cdna4_t sum) {
    accumulator[0].f = sum[0], accumulator[1].f = sum[1], accumulator[2].f = sum[2], accumulator[3].f = sum[3];
}

NUMKONG_DEVICE nk_i32x4_cdna4_t nk_i32x4_load_cdna4_(nk_u32_t const words[4]) {
    nk_i32x4_cdna4_t const vector = {(int)words[0], (int)words[1], (int)words[2], (int)words[3]};
    return vector;
}

/** Up to 8 words of Float8, Float6 or Float4 codes; the MFMA reads as many as the format needs. */
NUMKONG_DEVICE nk_i32x8_cdna4_t nk_i32x8_load_cdna4_(nk_u32_t const *words, unsigned count) {
    nk_i32x8_cdna4_t vector = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (unsigned index = 0; index < 8; ++index)
        if (index < count) vector[index] = (int)words[index];
    return vector;
}

NUMKONG_DEVICE void nk_mfma_bf16_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    nk_f32x4_store_cdna4_(accumulator, __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                                           __builtin_bit_cast(nk_bf16x8_cdna4_t, nk_i32x4_load_cdna4_(a)),
                                           __builtin_bit_cast(nk_bf16x8_cdna4_t, nk_i32x4_load_cdna4_(b)),
                                           nk_f32x4_load_cdna4_(accumulator), 0, 0, 0));
}

NUMKONG_DEVICE void nk_mfma_f16_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    nk_f32x4_store_cdna4_(accumulator, __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                           __builtin_bit_cast(nk_f16x8_cdna4_t, nk_i32x4_load_cdna4_(a)),
                                           __builtin_bit_cast(nk_f16x8_cdna4_t, nk_i32x4_load_cdna4_(b)),
                                           nk_f32x4_load_cdna4_(accumulator), 0, 0, 0));
}

NUMKONG_DEVICE void nk_mfma_i8_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    nk_i32x4_cdna4_t const sum = {accumulator[0].i, accumulator[1].i, accumulator[2].i, accumulator[3].i};
    nk_i32x4_cdna4_t const result = __builtin_amdgcn_mfma_i32_16x16x64_i8(nk_i32x4_load_cdna4_(a),
                                                                          nk_i32x4_load_cdna4_(b), sum, 0, 0, 0);
    accumulator[0].i = result[0], accumulator[1].i = result[1], accumulator[2].i = result[2];
    accumulator[3].i = result[3];
}

/*  One 16 × 16 × 128 step per format: @c cbsz and @c blgp select E4M3 as 0, E5M2 as 1, E2M3 as 2,
 *  E3M2 as 3 and E2M1 as 4, for 8, 8, 6, 6 and 4 words a lane. */

NUMKONG_DEVICE void nk_mfma_e4m3_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_f32x4_store_cdna4_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 8), nk_i32x8_load_cdna4_(b, 8), nk_f32x4_load_cdna4_(accumulator), 0,
                         0, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

NUMKONG_DEVICE void nk_mfma_e5m2_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_f32x4_store_cdna4_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 8), nk_i32x8_load_cdna4_(b, 8), nk_f32x4_load_cdna4_(accumulator), 1,
                         1, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

NUMKONG_DEVICE void nk_mfma_e2m3_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[6], nk_u32_t const b[6]) {
    nk_f32x4_store_cdna4_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 6), nk_i32x8_load_cdna4_(b, 6), nk_f32x4_load_cdna4_(accumulator), 2,
                         2, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

NUMKONG_DEVICE void nk_mfma_e3m2_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[6], nk_u32_t const b[6]) {
    nk_f32x4_store_cdna4_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 6), nk_i32x8_load_cdna4_(b, 6), nk_f32x4_load_cdna4_(accumulator), 3,
                         3, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

NUMKONG_DEVICE void nk_mfma_e2m1_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    nk_f32x4_store_cdna4_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 4), nk_i32x8_load_cdna4_(b, 4), nk_f32x4_load_cdna4_(accumulator), 4,
                         4, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

#else

/*  Other device passes and the host pass get trapping bodies, so a code object picked for the
 *  wrong device fails loudly rather than returning zeros. */
NUMKONG_DEVICE void nk_mfma_bf16_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_mfma_f16_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_mfma_i8_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_mfma_e4m3_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[8], nk_u32_t const b[8]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_mfma_e5m2_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[8], nk_u32_t const b[8]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_mfma_e2m3_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[6], nk_u32_t const b[6]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_mfma_e3m2_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[6], nk_u32_t const b[6]) {
    __builtin_trap();
}
NUMKONG_DEVICE void nk_mfma_e2m1_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    __builtin_trap();
}
#endif // defined(__gfx950__)

/*  The byte permute, which gfx950 and the MI400 parts share. */
#if defined(__gfx950__) || defined(__gfx1250__) || defined(__gfx1251__)

/** Byte @c i of the result is byte @c selector.u8[i] of the 8 bytes @p high : @p low, as
 *  @c v_perm_b32 picks them. */
NUMKONG_DEVICE nk_u32_t nk_byte_permute_cdna4_(nk_u32_t high, nk_u32_t low, nk_u32_t selector) {
    return __builtin_amdgcn_perm(high, low, selector);
}

#else

NUMKONG_DEVICE nk_u32_t nk_byte_permute_cdna4_(nk_u32_t high, nk_u32_t low, nk_u32_t selector) {
    __builtin_trap();
    return 0;
}

#endif // defined(__gfx950__) || defined(__gfx1250__) || defined(__gfx1251__)

#pragma endregion Instructions

#pragma region Conversions

/** Four 6-bit codes, one in the low bits of each byte, as 24 dense bits, code @c i at bit 6 × i. */
NUMKONG_DEVICE nk_u32_t nk_b8x4_to_b6x4_cdna4_(nk_u32_t codes) {
    return (codes & 0x3Fu) | ((codes >> 2) & 0xFC0u) | ((codes >> 4) & 0x3F000u) | ((codes >> 6) & 0xFC0000u);
}

/** Thirty-two 6-bit codes, one per byte, as the 192-bit stream a Float6 MFMA operand holds. */
NUMKONG_DEVICE void nk_b8x32_to_b6x32_cdna4_(nk_u32_t const codes[8], nk_u32_t packed[6]) {
    nk_u32_t dense[8];
#pragma unroll
    for (unsigned word = 0; word < 8; ++word) dense[word] = nk_b8x4_to_b6x4_cdna4_(codes[word]);
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        nk_u32_t const *quad = dense + half * 4;
        packed[half * 3 + 0] = quad[0] | (quad[1] << 24);
        packed[half * 3 + 1] = (quad[1] >> 8) | (quad[2] << 16);
        packed[half * 3 + 2] = (quad[2] >> 16) | (quad[3] << 8);
    }
}

#pragma endregion Conversions

/*  Each adds the squares of one 16-byte chunk of a staged row. Float8 and E3M2 square their scaled
 *  F16 widenings, and E2M3 and E2M1 their scaled integer ones, so the tile scales them back. */
#pragma region Norms

NUMKONG_DEVICE void nk_e5m2_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned byte = 0; byte < 16; ++byte) {
        nk_e5m2_t const code = (nk_e5m2_t)(words[byte / 4] >> (byte % 4 * 8));
        nk_f32_t value;
        nk_e5m2_to_f32_simt_(&code, &value);
        *real_sum = __fmaf_rn(value, value, *real_sum);
    }
}

NUMKONG_DEVICE void nk_e4m3_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned byte = 0; byte < 16; ++byte) {
        nk_u32_t const code = (words[byte / 4] >> (byte % 4 * 8)) & 0xFFu;
        // S.1111.111 is E4M3's NaN, which the F16 widening would read as 480.
        nk_f32_t const value = (code & 0x7Fu) == 0x7Fu ? __int_as_float(0x7FC00000) : nk_e4m3_to_scaled_f32_(code);
        *real_sum = __fmaf_rn(value, value, *real_sum);
    }
}

NUMKONG_DEVICE void nk_e3m2_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned byte = 0; byte < 16; ++byte) {
        nk_f32_t const value = nk_e3m2_to_scaled_f32_((words[byte / 4] >> (byte % 4 * 8)) & 0xFFu);
        *real_sum = __fmaf_rn(value, value, *real_sum);
    }
}

NUMKONG_DEVICE void nk_e2m3_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const magnitudes = nk_e2m3x4_to_u8x4_magnitudes_(words[word]);
        *integer_sum = nk_dot_u8x4_(magnitudes, magnitudes, *integer_sum);
    }
}

NUMKONG_DEVICE void nk_e2m1_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    // Squares of twice each magnitude, {0, 1, 4, 9, 16, 36, 64, 144}, 4 nibbles a lookup.
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const low = nk_byte_permute_cdna4_(0x90402410u, 0x09040100u, words[word] & 0x07070707u);
        nk_u32_t const high = nk_byte_permute_cdna4_(0x90402410u, 0x09040100u, (words[word] >> 4) & 0x07070707u);
        *integer_sum = nk_dot_u8x4_(low, 0x01010101u, *integer_sum);
        *integer_sum = nk_dot_u8x4_(high, 0x01010101u, *integer_sum);
    }
}

NUMKONG_DEVICE void nk_i8_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word)
        *integer_sum = (nk_u32_t)nk_dot_i8x4_(words[word], words[word], (nk_i32_t)*integer_sum);
}

NUMKONG_DEVICE void nk_u8_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) *integer_sum = nk_dot_u8x4_(words[word], words[word], *integer_sum);
}

NUMKONG_DEVICE void nk_i4_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_i4x8_to_i8x8_(words[word], &low, &high);
        *integer_sum = (nk_u32_t)nk_dot_i8x4_(low, low, (nk_i32_t)*integer_sum);
        *integer_sum = (nk_u32_t)nk_dot_i8x4_(high, high, (nk_i32_t)*integer_sum);
    }
}

NUMKONG_DEVICE void nk_u4_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_u4x8_to_u8x8_(words[word], &low, &high);
        *integer_sum = nk_dot_u8x4_(low, low, *integer_sum);
        *integer_sum = nk_dot_u8x4_(high, high, *integer_sum);
    }
}

#pragma endregion Norms

#pragma region Staging

/** Byte offset of 16-byte chunk @p chunk of staged row @p row. The XOR keeps any 16 consecutive
 *  rows' same chunk on distinct banks, whether the LDS serves 8 or 16 lanes of 16 bytes a clock. */
NUMKONG_DEVICE unsigned nk_cross_stage_offset_cdna4_(unsigned row, unsigned chunk) {
    return row * nk_cross_slab_bytes_cdna4_k + ((chunk ^ (row & 7) ^ ((row >> 3) & 1)) << 4);
}

/** Loads the 16 bytes of @p row from @p byte, as zeros past @p depth_bytes, never reading past
 *  them, since A's stride padding may hold anything. */
NUMKONG_DEVICE uint4 nk_cross_load_chunk_cdna4_(unsigned char const *row, nk_size_t byte, nk_size_t depth_bytes) {
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
NUMKONG_DEVICE void nk_cross_load_slab_cdna4_(uint4 chunks[2][nk_cross_loads_cdna4_k],
                                              nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                              nk_size_t first_column, nk_size_t slab) {
    nk_size_t const byte = slab * nk_cross_slab_bytes_cdna4_k + (threadIdx.x & 7) * 16;
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand) {
        unsigned char const *rows = operand ? arguments->b : arguments->a;
        nk_size_t const stride = operand ? arguments->b_stride : arguments->a_stride;
        nk_size_t const first = operand ? first_column : first_row;
        nk_size_t const end = operand ? arguments->column_count : arguments->row_end;
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna4_k; ++step) {
            nk_size_t const row = first + (threadIdx.x >> 3) + step * (nk_cross_threads_cdna4_k / 8);
            chunks[operand][step] = row < end && byte < arguments->depth_bytes
                                        ? nk_cross_load_chunk_cdna4_(rows + row * stride, byte, arguments->depth_bytes)
                                        : make_uint4(0, 0, 0, 0);
        }
    }
}

/** Stores this thread's chunks into @p stage, one swizzled slab per operand. */
NUMKONG_DEVICE void nk_cross_store_slab_cdna4_(unsigned char stage[2][nk_cross_stage_bytes_cdna4_k],
                                               uint4 const chunks[2][nk_cross_loads_cdna4_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna4_k; ++step)
            *(uint4 *)(stage[operand] +
                       nk_cross_stage_offset_cdna4_((threadIdx.x >> 3) + step * (nk_cross_threads_cdna4_k / 8),
                                                    threadIdx.x & 7)) = chunks[operand][step];
}

/** Folds this thread's staged chunks into its running sums: A's squares through @p norm_update if
 *  @p row_squares, B's if @p column_squares, and both operands' byte sums if @p offsets. */
NUMKONG_DEVICE void nk_cross_stage_statistics_cdna4_(nk_cross_norm_update_t norm_update, int row_squares,
                                                     int column_squares, int offsets,
                                                     uint4 const chunks[2][nk_cross_loads_cdna4_k],
                                                     nk_u32_t integer_norms[2][nk_cross_loads_cdna4_k],
                                                     nk_f32_t real_norms[2][nk_cross_loads_cdna4_k],
                                                     nk_u32_t sums[2][nk_cross_loads_cdna4_k]) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna4_k; ++step) {
            uint4 const chunk = chunks[operand][step];
            nk_u32_t const words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            if (operand ? column_squares : row_squares)
                norm_update(words, &integer_norms[operand][step], &real_norms[operand][step]);
            if (!offsets) continue;
#pragma unroll
            for (unsigned word = 0; word < 4; ++word)
                sums[operand][step] = nk_dot_u8x4_(words[word], 0x01010101u, sums[operand][step]);
        }
}

/** Merges the running sums of the 8 threads staging each row, then writes them from its first
 *  thread into @p statistics, a tile's worth each: row norms, column norms, row byte sums and
 *  column byte sums. Without @p column_squares, a metric reads its column norms from @c b_norms. */
NUMKONG_DEVICE void nk_cross_write_statistics_cdna4_(nk_cross_norm_t norm, nk_f32_t norm_scale, int row_squares,
                                                     int column_squares, int offsets,
                                                     nk_cross_tile_arguments_t const *arguments, nk_size_t first_column,
                                                     nk_u32_t integer_norms[2][nk_cross_loads_cdna4_k],
                                                     nk_f32_t real_norms[2][nk_cross_loads_cdna4_k],
                                                     nk_u32_t sums[2][nk_cross_loads_cdna4_k], nk_fui32_t *statistics) {
#pragma unroll
    for (unsigned operand = 0; operand < 2; ++operand)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_cdna4_k; ++step) {
            int const squares = operand ? column_squares : row_squares;
#pragma unroll
            for (unsigned offset = 1; offset < 8; offset <<= 1) {
                if (squares)
                    integer_norms[operand][step] += nk_shuffle_xor_u32_(integer_norms[operand][step], offset),
                        real_norms[operand][step] += nk_shuffle_xor_f32_(real_norms[operand][step], offset);
                if (offsets) sums[operand][step] += nk_shuffle_xor_u32_(sums[operand][step], offset);
            }
            if ((threadIdx.x & 7) != 0) continue;
            unsigned const tile_row = (threadIdx.x >> 3) + step * (nk_cross_threads_cdna4_k / 8);
            if (squares)
                statistics[operand * nk_cross_tile_cdna4_k + tile_row] = nk_cross_norm_finalize_(
                    norm, integer_norms[operand][step], real_norms[operand][step], norm_scale);
            if (offsets) statistics[(2 + operand) * nk_cross_tile_cdna4_k + tile_row].u = sums[operand][step];
        }
    if (!row_squares || column_squares || threadIdx.x >= nk_cross_tile_cdna4_k) return;
    nk_size_t const column = first_column + threadIdx.x;
    statistics[nk_cross_tile_cdna4_k + threadIdx.x].u = column < arguments->column_count
                                                            ? ((nk_u32_t const *)arguments->b_norms)[column]
                                                            : 0;
}

/** Writes one output of a tile: the dot product, restored from its U8 offset where it has one, or
 *  the metric from it and both squared norms, with zeros on a @c symmetric diagonal. @p statistics
 *  holds the row norms, column norms, row byte sums and column byte sums, a tile's worth each. */
NUMKONG_DEVICE void nk_cross_store_cdna4_(nk_cross_epilogue_t epilogue, nk_f32_t output_scale,
                                          nk_cross_triangle_t triangle, nk_cross_metric_t metric, nk_cross_norm_t norm,
                                          nk_fui32_t sum, nk_fui32_t const *statistics, nk_u32_t offset_correction,
                                          unsigned tile_row, unsigned tile_column, nk_size_t row, nk_size_t column,
                                          unsigned char *output) {
    if (epilogue == nk_cross_epilogue_offset_u32_k)
        sum.u += 128u * (statistics[2 * nk_cross_tile_cdna4_k + tile_row].u +
                         statistics[3 * nk_cross_tile_cdna4_k + tile_column].u) -
                 offset_correction;
    if (metric == nk_cross_metric_dot_k) {
        if (epilogue == nk_cross_epilogue_f32_k) ((nk_f32_t *)output)[column] = sum.f * output_scale;
        else if (epilogue == nk_cross_epilogue_i32_to_f32_k)
            ((nk_f32_t *)output)[column] = (nk_f32_t)sum.i * output_scale;
        else ((nk_u32_t *)output)[column] = sum.u;
        return;
    }
    if (triangle == nk_cross_triangle_upper_k && column == row) {
        ((nk_f32_t *)output)[column] = 0.0f;
        return;
    }
    nk_f32_t const dot = nk_cross_dot_to_f32_(sum, epilogue, norm, output_scale);
    nk_f32_t const row_norm = nk_cross_norm_to_f32_(statistics[tile_row], norm);
    nk_f32_t const column_norm = nk_cross_norm_to_f32_(statistics[nk_cross_tile_cdna4_k + tile_column], norm);
    ((nk_f32_t *)output)[column] = metric == nk_cross_metric_angular_k ? nk_f32_angular_(dot, row_norm, column_norm)
                                                                       : nk_f32_euclidean_(dot, row_norm, column_norm);
}

#pragma endregion Staging

#if NUMKONG_TARGET_CDNA4

#pragma region Tile

/** The 32 bytes of quarter @p quarter of staged row @p row: an MFMA operand for its lane. */
NUMKONG_DEVICE void nk_cross_load_fragment_cdna4_(unsigned char const *stage, unsigned row, unsigned quarter,
                                                  nk_u32_t fragment[8]) {
    uint4 const first = *(uint4 const *)(stage + nk_cross_stage_offset_cdna4_(row, quarter * 2));
    uint4 const second = *(uint4 const *)(stage + nk_cross_stage_offset_cdna4_(row, quarter * 2 + 1));
    fragment[0] = first.x, fragment[1] = first.y, fragment[2] = first.z, fragment[3] = first.w;
    fragment[4] = second.x, fragment[5] = second.y, fragment[6] = second.z, fragment[7] = second.w;
}

/**
 *  @brief The whole GEMM of one 128 × 128 output tile, shared by every dtype and every metric.
 *  @param[in] multiply Folds one slab of fragments into the wavefront's accumulators; inlined,
 *      being a constant.
 *  @param[in] epilogue What the accumulators hold and how they reach the output.
 *  @param[in] output_scale Undoes the power of two a widening introduced, or 1.
 *  @param[in] norm How the squared norms are stored and the metric is computed; unused for dots.
 *  @param[in] norm_update Adds staged squares; unused for dots.
 *  @param[in] norm_scale Undoes the power of two the norm update's widening introduced, or 1.
 *  @param[in] triangle Whether tiles and outputs below the diagonal are skipped.
 *  @param[in] metric The dot product itself, or a distance from it and the two squared norms.
 *
 *  Each slab's global loads land in registers while the previous slab multiplies, then go to the
 *  stage no wavefront is reading, so one barrier a slab suffices. Row norms, and for @c symmetric
 *  the column norms, accumulate from the staged chunks, 8 threads per row; @c packed reads its
 *  column norms from @c b_norms. Lane l of each 16 × 16 MFMA output holds its column l % 16 in rows
 *  4 × (l / 16) + e.
 */
NUMKONG_DEVICE void nk_cross_tile_cdna4_(nk_cross_multiply_cdna4_t multiply, nk_cross_epilogue_t epilogue,
                                         nk_f32_t output_scale, nk_cross_norm_t norm,
                                         nk_cross_norm_update_t norm_update, nk_f32_t norm_scale,
                                         nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                         nk_cross_tile_arguments_t const *arguments) {
    __shared__ __attribute__((aligned(16))) unsigned char staged[2][2][nk_cross_stage_bytes_cdna4_k];

    unsigned const lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    unsigned const wave_row = (wave >> 1) * 64, wave_column = (wave & 1) * 64;
    unsigned const lane_row = lane & 15, quarter = lane >> 4;
    nk_size_t const slabs = nk_size_divide_round_up_(arguments->depth_bytes, nk_cross_slab_bytes_cdna4_k);
    int const offsets = epilogue == nk_cross_epilogue_offset_u32_k, row_squares = metric != nk_cross_metric_dot_k;
    int const column_squares = row_squares && triangle == nk_cross_triangle_upper_k;

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's epilogue reads the first stage, which this tile's first slab refills.
        __syncthreads();
        nk_size_t const first_row = arguments->row_start + tile / arguments->column_tiles * nk_cross_tile_cdna4_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_cdna4_k;
        if (triangle == nk_cross_triangle_upper_k && first_column + nk_cross_tile_cdna4_k <= first_row) continue;

        nk_fui32_t accumulators[4][4][4];
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) accumulators[row_tile][column_tile][element].u = 0;
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
            nk_u32_t a_fragments[4][8], b_fragments[4][8];
#pragma unroll
            for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
                nk_cross_load_fragment_cdna4_(stage[0], wave_row + row_tile * 16 + lane_row, quarter,
                                              a_fragments[row_tile]);
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_cross_load_fragment_cdna4_(stage[1], wave_column + column_tile * 16 + lane_row, quarter,
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
        // Every multiplied position of a U8 offset product adds 128² beyond the codes' own product.
        nk_u32_t const offset_correction = (nk_u32_t)(slabs * nk_cross_slab_bytes_cdna4_k * 16384u);

#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element) {
                unsigned const tile_row = wave_row + row_tile * 16 + quarter * 4 + element;
                nk_size_t const row = first_row + tile_row;
                if (row >= arguments->row_end) continue;
                unsigned char *output = (unsigned char *)arguments->c + row * arguments->c_stride;
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 4; ++column_tile) {
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

/*  Each folds one 128-byte slab. Two MFMAs that read words 0-3 and 4-7 of every fragment cover
 *  the slab once, since A and B share each lane's depth mapping; the outer loop over those halves
 *  keeps dependent MFMAs 16 apart. */
#pragma region Multiplies

NUMKONG_DEVICE void nk_dots_bf16_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_bf16_cdna4_(accumulators[row_tile][column_tile], a[row_tile] + half * 4,
                                    b[column_tile] + half * 4);
}

NUMKONG_DEVICE void nk_dots_f16_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_f16_cdna4_(accumulators[row_tile][column_tile], a[row_tile] + half * 4,
                                   b[column_tile] + half * 4);
}

NUMKONG_DEVICE void nk_dots_e5m2_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
            nk_mfma_e5m2_cdna4_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e4m3_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
            nk_mfma_e4m3_cdna4_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e3m2_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
    nk_u32_t a_packed[4][6], b_packed[4][6];
#pragma unroll
    for (unsigned tile = 0; tile < 4; ++tile)
        nk_b8x32_to_b6x32_cdna4_(a[tile], a_packed[tile]), nk_b8x32_to_b6x32_cdna4_(b[tile], b_packed[tile]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
            nk_mfma_e3m2_cdna4_(accumulators[row_tile][column_tile], a_packed[row_tile], b_packed[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e2m3_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
    nk_u32_t a_packed[4][6], b_packed[4][6];
#pragma unroll
    for (unsigned tile = 0; tile < 4; ++tile)
        nk_b8x32_to_b6x32_cdna4_(a[tile], a_packed[tile]), nk_b8x32_to_b6x32_cdna4_(b[tile], b_packed[tile]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
            nk_mfma_e2m3_cdna4_(accumulators[row_tile][column_tile], a_packed[row_tile], b_packed[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e2m1_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_e2m1_cdna4_(accumulators[row_tile][column_tile], a[row_tile] + half * 4,
                                    b[column_tile] + half * 4);
}

NUMKONG_DEVICE void nk_dots_i8_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna4_(accumulators[row_tile][column_tile], a[row_tile] + half * 4,
                                  b[column_tile] + half * 4);
}

/** U8 codes offset by −128 into I8 by flipping each byte's top bit, which the epilogue undoes. */
NUMKONG_DEVICE void nk_dots_u8_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        nk_u32_t a_offset[4][4], b_offset[4][4];
#pragma unroll
        for (unsigned tile = 0; tile < 4; ++tile)
#pragma unroll
            for (unsigned word = 0; word < 4; ++word)
                a_offset[tile][word] = a[tile][half * 4 + word] ^ 0x80808080u,
                b_offset[tile][word] = b[tile][half * 4 + word] ^ 0x80808080u;
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna4_(accumulators[row_tile][column_tile], a_offset[row_tile], b_offset[column_tile]);
    }
}

/** Widens nibble fragments into I8, 2 words of nibbles becoming the 4 words of one MFMA. */
NUMKONG_DEVICE void nk_dots_widened_i8_multiply_cdna4_(nk_cross_widen_t widen, nk_fui32_t accumulators[4][4][4],
                                                       nk_u32_t const a[4][8], nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned step = 0; step < 4; ++step) {
        nk_u32_t a_widened[4][4], b_widened[4][4];
#pragma unroll
        for (unsigned tile = 0; tile < 4; ++tile) {
            widen(a[tile][step * 2], &a_widened[tile][0], &a_widened[tile][1]);
            widen(a[tile][step * 2 + 1], &a_widened[tile][2], &a_widened[tile][3]);
            widen(b[tile][step * 2], &b_widened[tile][0], &b_widened[tile][1]);
            widen(b[tile][step * 2 + 1], &b_widened[tile][2], &b_widened[tile][3]);
        }
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna4_(accumulators[row_tile][column_tile], a_widened[row_tile], b_widened[column_tile]);
    }
}

NUMKONG_DEVICE void nk_dots_i4_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
    nk_dots_widened_i8_multiply_cdna4_(nk_i4x8_to_i8x8_, accumulators, a, b);
}

NUMKONG_DEVICE void nk_dots_u4_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
    nk_dots_widened_i8_multiply_cdna4_(nk_u4x8_to_u8x8_, accumulators, a, b);
}

#pragma endregion Multiplies

#pragma region BF16

nk_define_cross_pack_rocm_(bf16, cdna4, bf16, bf16, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, bf16, cdna4, cdna4, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_bf16_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_rocm_(f16, cdna4, f16, f16, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, f16, cdna4, cdna4, f16, f16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1,
                      nk_dots_f16_multiply_cdna4_, nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                      NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_pack_rocm_(e5m2, cdna4, e5m2, e5m2, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e5m2, cdna4, cdna4, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e5m2_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_pack_rocm_(e4m3, cdna4, e4m3, e4m3, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e4m3, cdna4, cdna4, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e4m3_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_pack_rocm_(e3m2, cdna4, e3m2, e3m2, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e3m2, cdna4, cdna4, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e3m2_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_pack_rocm_(e2m3, cdna4, e2m3, e2m3, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e2m3, cdna4, cdna4, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e2m3_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_rocm_(e2m1, cdna4, e2m1x2, e2m1x2, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, e2m1, cdna4, cdna4, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_e2m1_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_pack_rocm_(i8, cdna4, i8, i8, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, i8, cdna4, cdna4, i8, i8, i32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1,
                      nk_dots_i8_multiply_cdna4_, nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                      NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_rocm_(i4, cdna4, i4x2, i4x2, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, i4, cdna4, cdna4, i4x2, i4x2, i32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_i4_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_rocm_(u8, cdna4, u8, u8, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, u8, cdna4, cdna4, u8, u8, u32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1,
                      nk_dots_u8_multiply_cdna4_, nk_cross_epilogue_offset_u32_k, /*output_scale=*/1.0f,
                      nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_rocm_(u4, cdna4, u4x2, u4x2, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, u4, cdna4, cdna4, u4x2, u4x2, u32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_u4_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion U4

#endif // NUMKONG_TARGET_CDNA4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA4_
#endif // NUMKONG_DOTS_CDNA4_CUH
