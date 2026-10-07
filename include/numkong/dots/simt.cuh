/**
 *  @file include/numkong/dots/simt.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched Dot Products on the SIMT cores of every CUDA and ROCm device.
 *
 *  @sa include/numkong/dots.h
 *
 *  The portable half of the baseline every other GPU capability stands on, as serial is on the CPU:
 *  no matrix unit and no vendor instruction. F64 inputs sum in Dot2 and F32 inputs in F64 on the
 *  F64 tile. Every narrower type folds 32-bit words on the B32 tile: integers into wrapping I32 or
 *  U32 like the serial backends, and the 16-bit and narrower floats into F32. Each of 256 threads
 *  owns a 4 × 4 grid of one 64 × 64 output tile, over 16-word slabs staged in shared memory. The
 *  pack stores rows as they are, padded to 16 bytes, with the serial backends' norm types.
 *
 *  Every tensor tile of the other capabilities builds on the contract and kernel generators here, so
 *  the file runs from the contract to device code, then to the tiles' portable stages. The tiles
 *  themselves live beside this file: each vendor's F64 tile, merging lanes and rounding through its
 *  own primitives, and each capability's B32 tile, folding through its own dot instructions, in
 *  `cuda.cuh`, `rocm.cuh` and `cdna3.cuh`, which also size, launch and export them.
 *
 *  @sa include/numkong/dots/cuda.cuh
 *  @sa include/numkong/dots/rocm.cuh
 *  @sa include/numkong/dots/cdna3.cuh
 */
#ifndef NUMKONG_DOTS_SIMT_CUH
#define NUMKONG_DOTS_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/dots/serial.h"
#include "numkong/cast/simt.cuh" // `nk_e4m3_to_f32_simt_`, `nk_f32_to_f16_simt_`

#if defined(__cplusplus)
extern "C" {
#endif

/*  The baseline tiles' launch shape, the same on every vendor. AMD blocks stay whole multiples of
 *  64 threads, so they hold whole wavefronts of either width. The grid side counts the threads
 *  along each side of a tile, and each block of a device pack holds 8 groups of 32 lanes. */
#pragma region Configuration

enum {
    nk_cross_threads_simt_k = 256,
    nk_cross_tile_simt_k = 64,
    nk_cross_thread_tile_simt_k = 4,
    nk_cross_slab_simt_k = 16,
    nk_cross_loads_simt_k = nk_cross_tile_simt_k * nk_cross_slab_simt_k / nk_cross_threads_simt_k,
    nk_cross_grid_side_simt_k = nk_cross_tile_simt_k / nk_cross_thread_tile_simt_k,
    nk_cross_pack_groups_k = 8,
};

nk_static_assert_(nk_cross_grid_side_simt_k *nk_cross_grid_side_simt_k == nk_cross_threads_simt_k,
                  nk_cross_grid_is_square);
nk_static_assert_(nk_cross_threads_simt_k % 64 == 0, nk_cross_blocks_hold_whole_wavefronts);
nk_static_assert_(nk_cross_threads_simt_k % nk_cross_slab_simt_k == 0 && nk_cross_slab_simt_k <= 32,
                  nk_cross_slab_fits_a_warp);

/** How a tile accumulates, matching the serial backends, and how a B32 tile's word holds depth. */
typedef enum {

    /** One F64 FMA per product, for F32 inputs. */
    nk_cross_accumulation_f64_k,

    /** Ogita-Rump-Oishi Dot2: TwoProd and TwoSum, for F64 inputs. */
    nk_cross_accumulation_dot2_k,

    /** One element per word as F32, folded by one F32 FMA. */
    nk_cross_accumulation_f32_k,

    /** Two elements per word as F16, folded by one dot2 into F32. */
    nk_cross_accumulation_f16x2_k,

    /** Four signed bytes per word, folded by one dot4 into wrapping I32. */
    nk_cross_accumulation_i8x4_k,

    /** Four unsigned bytes per word, folded by one dot4 into wrapping U32. */
    nk_cross_accumulation_u8x4_k,

    /** Four signed nibbles widened to bytes, folded as @c i8x4. */
    nk_cross_accumulation_i4x4_k,

    /** Four unsigned nibbles widened to bytes, folded as @c u8x4. */
    nk_cross_accumulation_u4x4_k,

    /** Eight signed nibbles as packed, folded by one dot8 into wrapping I32. */
    nk_cross_accumulation_i4x8_k,

    /** Eight unsigned nibbles as packed, folded by one dot8 into wrapping U32. */
    nk_cross_accumulation_u4x8_k,
} nk_cross_accumulation_t;

/** Which outputs a tile writes. */
typedef enum {

    /** Every output, for @c packed. */
    nk_cross_triangle_full_k,

    /** The upper triangle with its diagonal, for @c symmetric. */
    nk_cross_triangle_upper_k,
} nk_cross_triangle_t;

/** What a tile turns each dot product into. */
typedef enum {

    /** The dot product itself. */
    nk_cross_metric_dot_k,

    /** 1 − dot / (‖a‖ ‖b‖), clamped at 0. */
    nk_cross_metric_angular_k,

    /** √(‖a‖² + ‖b‖² − 2 · dot), clamped at 0. */
    nk_cross_metric_euclidean_k,
} nk_cross_metric_t;

/** Everything one launch shares, passed by value as the kernels' only argument. */
typedef struct {

    /** Row-major A, or the vectors for @c symmetric. */
    unsigned char const *a;

    /** Packed B rows past the header, or the vectors again for @c symmetric. */
    unsigned char const *b;

    /** Row-major output, indexed by absolute row. */
    void *c;

    /** First output row. */
    nk_size_t row_start;

    /** One past the last output row. */
    nk_size_t row_end;

    /** Output columns, the rows of B. */
    nk_size_t column_count;

    /** Bytes of depth per row, past which A and B read as zeros. */
    nk_size_t depth_bytes;

    /** Elements of depth per row, which the baseline tiles count in. */
    nk_size_t depth;

    /** Bytes between rows of A. */
    nk_size_t a_stride;

    /** Bytes between rows of B. */
    nk_size_t b_stride;

    /** Bytes between rows of C. */
    nk_size_t c_stride;

    /** One scale byte per block of each A row, or NULL for plain dtypes. */
    unsigned char const *a_scales;

    /** One scale byte per block of each B row: the pack's scale rows, or the vectors' for
     *  @c symmetric. */
    unsigned char const *b_scales;

    /** Bytes between scale rows of A and of B. */
    nk_size_t a_scales_stride, b_scales_stride;

    /** Tensor scales of A and of B in device memory, or NULL for 1. */
    nk_f32_t const *a_tensor_scale, *b_tensor_scale;

    /** 64-byte slabs of depth, which the tensor tiles count in. */
    nk_size_t depth_slabs;

    /** Output tiles per row of tiles. */
    nk_size_t column_tiles;

    /** Output tiles in all, which the blocks walk with a stride of the grid. */
    nk_size_t tiles;

    /** Column norms past the packed rows, which only a @c packed metric reads. */
    void const *b_norms;
} nk_cross_tile_arguments_t;

/** What the accumulators hold and how they reach the output. */
typedef enum {

    /** F32 sums, stored times the output scale. */
    nk_cross_epilogue_f32_k,

    /** Integer sums, stored as they are. */
    nk_cross_epilogue_i32_k,

    /** Integer sums of scaled codes, converted and stored times the output scale. */
    nk_cross_epilogue_i32_to_f32_k,

    /** Integer sums of U8 codes offset by −128, restored from the byte sums. */
    nk_cross_epilogue_offset_u32_k,
} nk_cross_epilogue_t;

/** How squared norms are stored, which also fixes the precision a metric is computed in. */
typedef enum {

    /** F32 in true units. */
    nk_cross_norm_f32_k,

    /** F64. */
    nk_cross_norm_f64_k,

    /** Wrapping U32 sums read as I32. */
    nk_cross_norm_i32_k,

    /** Wrapping U32 sums. */
    nk_cross_norm_u32_k,
} nk_cross_norm_t;

/** Adds the squares of 16 staged bytes: exact integer codes into @p integer_sum, the others
 *  into @p real_sum. */
typedef void (*nk_cross_norm_update_t)(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum);

/** Splits one fragment register into two narrower-typed ones: 4 codes into F16 pairs, or 8 nibbles
 *  into I8 quads. */
typedef void (*nk_cross_widen_t)(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high);

#pragma endregion Configuration

#pragma region Pack Layout

/** Storage values in one packed GPU row: @c nk_cross_padded_values_ without its power-of-two
 *  break, since the GPU loaders never read the padding and only lose bandwidth to it. */
NUMKONG_INLINE nk_size_t nk_cross_padded_values_simt_(nk_size_t depth, nk_size_t depth_simd_dimensions,
                                                      nk_size_t dimensions_per_value, nk_size_t packed_value_bytes) {
    nk_unused_(packed_value_bytes);
    return nk_size_round_up_to_multiple_(depth, depth_simd_dimensions) / dimensions_per_value;
}

#pragma endregion Pack Layout

/*  Portable folds of one 32-bit word of each operand, for capabilities lacking a dot instruction of
 *  that shape; the capabilities that have one name it in their own helpers. */
#pragma region Folds

/** Adds the four signed byte products of @p a and @p b to @p sum, wrapping like serial I32. */
NUMKONG_DEVICE nk_i32_t nk_dot_i8x4_simt_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) {
    nk_i32_t products = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        products += (nk_i32_t)(signed char)(a >> shift) * (nk_i32_t)(signed char)(b >> shift);
    return (nk_i32_t)((nk_u32_t)sum + (nk_u32_t)products);
}

/** Adds the four unsigned byte products of @p a and @p b to @p sum, wrapping like serial U32. */
NUMKONG_DEVICE nk_u32_t nk_dot_u8x4_simt_(nk_u32_t a, nk_u32_t b, nk_u32_t sum) {
    for (unsigned shift = 0; shift < 32; shift += 8) sum += ((a >> shift) & 0xFFu) * ((b >> shift) & 0xFFu);
    return sum;
}

/** Adds the eight signed nibble products of @p a and @p b, paired by position, to @p sum. */
NUMKONG_DEVICE nk_i32_t nk_dot_i4x8_simt_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) {
    nk_i32_t products = 0;
    for (unsigned shift = 0; shift < 32; shift += 4)
        products += ((nk_i32_t)(((a >> shift) & 0xFu) ^ 8u) - 8) * ((nk_i32_t)(((b >> shift) & 0xFu) ^ 8u) - 8);
    return (nk_i32_t)((nk_u32_t)sum + (nk_u32_t)products);
}

/** Adds the eight unsigned nibble products of @p a and @p b, paired by position, to @p sum. */
NUMKONG_DEVICE nk_u32_t nk_dot_u4x8_simt_(nk_u32_t a, nk_u32_t b, nk_u32_t sum) {
    for (unsigned shift = 0; shift < 32; shift += 4) sum += ((a >> shift) & 0xFu) * ((b >> shift) & 0xFu);
    return sum;
}

/** Adds both F16 products of @p a and @p b to @p sum, each exact in F32, low pair first. */
NUMKONG_DEVICE nk_f32_t nk_dot_f16x2_simt_(nk_u32_t a, nk_u32_t b, nk_f32_t sum) {
    nk_f32_t const a_low = __half2float(__ushort_as_half((unsigned short)(a & 0xFFFFu)));
    nk_f32_t const b_low = __half2float(__ushort_as_half((unsigned short)(b & 0xFFFFu)));
    nk_f32_t const a_high = __half2float(__ushort_as_half((unsigned short)(a >> 16)));
    nk_f32_t const b_high = __half2float(__ushort_as_half((unsigned short)(b >> 16)));
    return __fmaf_rn(a_high, b_high, __fmaf_rn(a_low, b_low, sum));
}

#pragma endregion Folds

/*  Widenings every vendor's tiles share, then element decoders that widen one element exactly: F64
 *  and F32 to F64 for the F64 tile, and every narrower float to F32. The Float8 and Float6 ones go
 *  through F16, which holds E5M2 exactly, E4M3 over 256 and E3M2 over 4096. Nibble types hold
 *  element 2 × i in the high nibble of byte i and element 2 × i + 1 in the low one. 16-bit types
 *  read bytewise, since a pack's source row stride need not be a whole number of elements. */
#pragma region Conversions

/** Eight I4 nibbles become two registers of four sign-extended I8, each (v ^ 8) − 8 without
 *  cross-byte borrows. */
NUMKONG_DEVICE void nk_i4x8_to_i8x8_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_nibbles = codes & 0x0F0F0F0Fu, high_nibbles = (codes >> 4) & 0x0F0F0F0Fu;
    *low = (((low_nibbles ^ 0x08080808u) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
    *high = (((high_nibbles ^ 0x08080808u) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}

/** Eight U4 nibbles become two registers of four U8. */
NUMKONG_DEVICE void nk_u4x8_to_u8x8_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    *low = codes & 0x0F0F0F0Fu, *high = (codes >> 4) & 0x0F0F0F0Fu;
}

/** Four E2M3 magnitudes times 8 as U8: @c m where e = 0, otherwise (8 + m) << (e − 1), at most
 *  60. */
NUMKONG_DEVICE nk_u32_t nk_e2m3x4_to_u8x4_magnitudes_(nk_u32_t codes) {
    nk_u32_t const exponent_low = (codes >> 3) & 0x01010101u, exponent_high = (codes >> 4) & 0x01010101u;
    nk_u32_t const significand = (codes & 0x07070707u) | ((exponent_low | exponent_high) << 3);
    nk_u32_t const doubled_mask = exponent_high * 0xFFu, quadrupled_mask = (exponent_high & exponent_low) * 0xFFu;
    return significand + (significand & doubled_mask) + ((significand << 1) & quadrupled_mask);
}

/** One E3M2 code over 4096. */
NUMKONG_DEVICE nk_f32_t nk_e3m2_to_scaled_f32_(nk_u32_t code) {
    return __half2float(__ushort_as_half((unsigned short)(((code & 0x1Fu) << 8) | ((code & 0x20u) << 10))));
}

/** Keeps a packed byte as it is. */
NUMKONG_DEVICE unsigned char nk_load_b8_(unsigned char value) { return value; }

NUMKONG_DEVICE nk_f64_t nk_f64_load_f64_(unsigned char const *row, nk_size_t index) {
    return ((nk_f64_t const *)row)[index];
}

NUMKONG_DEVICE nk_f64_t nk_f32_load_f64_(unsigned char const *row, nk_size_t index) {
    return (nk_f64_t)((nk_f32_t const *)row)[index];
}

NUMKONG_DEVICE nk_f32_t nk_bf16_load_f32_(unsigned char const *row, nk_size_t index) {
    return __uint_as_float(((nk_u32_t)row[index * 2] | ((nk_u32_t)row[index * 2 + 1] << 8)) << 16);
}

NUMKONG_DEVICE nk_f32_t nk_f16_load_f32_(unsigned char const *row, nk_size_t index) {
    return __half2float(__ushort_as_half((unsigned short)(row[index * 2] | (row[index * 2 + 1] << 8))));
}

NUMKONG_DEVICE nk_f32_t nk_e5m2_load_f32_(unsigned char const *row, nk_size_t index) {
    nk_f32_t value;
    nk_e5m2_to_f32_simt_(row + index, &value);
    return value;
}

NUMKONG_DEVICE nk_f32_t nk_e4m3_load_f32_(unsigned char const *row, nk_size_t index) {
    nk_f32_t value;
    nk_e4m3_to_f32_simt_(row + index, &value);
    return value;
}

NUMKONG_DEVICE nk_f32_t nk_e3m2_load_f32_(unsigned char const *row, nk_size_t index) {
    return nk_e3m2_to_scaled_f32_(row[index]) * 4096.0f;
}

NUMKONG_DEVICE nk_f32_t nk_e2m3_load_f32_(unsigned char const *row, nk_size_t index) {
    unsigned const code = row[index];
    nk_f32_t const magnitude = (nk_f32_t)nk_e2m3x4_to_u8x4_magnitudes_(code) * 0.125f;
    return code & 0x20u ? -magnitude : magnitude;
}

NUMKONG_DEVICE unsigned nk_b4_load_(unsigned char const *row, nk_size_t index) {
    return (index & 1) ? (row[index / 2] & 0x0Fu) : (row[index / 2] >> 4);
}

NUMKONG_DEVICE nk_f32_t nk_e2m1_load_f32_(unsigned char const *row, nk_size_t index) {
    unsigned const code = nk_b4_load_(row, index), exponent = (code >> 1) & 3u;
    nk_f32_t const magnitude = exponent ? (nk_f32_t)((2u + (code & 1u)) << (exponent - 1)) * 0.5f
                                        : (nk_f32_t)(code & 1u) * 0.5f;
    return code & 8u ? -magnitude : magnitude;
}

/** Element @p index of an F64 or F32 row, as the F64 tile reads it. */
NUMKONG_DEVICE nk_f64_t nk_cross_load_f64_(nk_dtype_t dtype, unsigned char const *row, nk_size_t index) {
    return dtype == nk_f64_k ? nk_f64_load_f64_(row, index) : nk_f32_load_f64_(row, index);
}

/** Element @p index of a 16-bit or narrower float row, exactly as F32. */
NUMKONG_DEVICE nk_f32_t nk_cross_load_f32_(nk_dtype_t dtype, unsigned char const *row, nk_size_t index) {
    switch (dtype) {
    case nk_bf16_k: return nk_bf16_load_f32_(row, index);
    case nk_f16_k: return nk_f16_load_f32_(row, index);
    case nk_e5m2_k: return nk_e5m2_load_f32_(row, index);
    case nk_e4m3_k: return nk_e4m3_load_f32_(row, index);
    case nk_e3m2_k: return nk_e3m2_load_f32_(row, index);
    case nk_e2m3_k: return nk_e2m3_load_f32_(row, index);
    default: return nk_e2m1_load_f32_(row, index);
    }
}

#pragma endregion Conversions

/*  Each lane share returns a lane's part of a column's sum of squares, over indices `lane + 32 × k`
 *  below @c depth, which the pack merges across the 32 lanes: F64 for floats and exact 64-bit sums
 *  for integers. The Dot2-compensated shares of F64 and F32 inputs, and the merges, round through
 *  each vendor's primitives in its own header. Rows read bytewise, since the source's stride need
 *  not be element-aligned. */
#pragma region Norms

#define nk_define_lane_sumsq_simt_(input_type_name)                                                       \
    NUMKONG_DEVICE nk_f64_t nk_##input_type_name##_lane_sumsq_(unsigned char const *row, nk_size_t depth, \
                                                               unsigned lane) {                           \
        nk_f64_t sum = 0;                                                                                 \
        for (nk_size_t index = lane; index < depth; index += 32) {                                        \
            nk_f64_t const value = nk_##input_type_name##_load_f32_(row, index);                          \
            sum = __fma_rn(value, value, sum);                                                            \
        }                                                                                                 \
        return sum;                                                                                       \
    }

nk_define_lane_sumsq_simt_(bf16)
nk_define_lane_sumsq_simt_(f16)
nk_define_lane_sumsq_simt_(e5m2)
nk_define_lane_sumsq_simt_(e4m3)
nk_define_lane_sumsq_simt_(e3m2)
nk_define_lane_sumsq_simt_(e2m3)
nk_define_lane_sumsq_simt_(e2m1)

/** The sum of squares of a block-scaled row of @p dtype over its blocks first + step × k below
 *  @p depth, each block summed in F32 and taking its scale times @p tensor_scale squared once, in
 *  F64 so no UE8M0 extreme overflows: a lane's share for the pack, or a whole row. */
NUMKONG_DEVICE nk_f64_t nk_cross_scaled_sumsq_simt_(nk_dtype_t dtype, unsigned char const *row,
                                                    unsigned char const *scales, nk_f64_t tensor_scale, nk_size_t depth,
                                                    nk_size_t first, nk_size_t step) {
    nk_block_scaled_format_t const format = nk_block_scaled_format_of_dtype(dtype);
    nk_f64_t sum = 0;
    for (nk_size_t block = first; block * format.block_size < depth; block += step) {
        nk_f32_t block_sum = 0;
        for (nk_size_t index = block * format.block_size; index != (block + 1) * format.block_size; ++index) {
            nk_f32_t const value = nk_cross_load_f32_(format.element_dtype, row, index);
            block_sum = __fmaf_rn(value, value, block_sum);
        }
        nk_f64_t const scale = nk_block_scaled_decode_scale_serial_(scales[block], format.scale_dtype) * tensor_scale;
        sum += block_sum * scale * scale;
    }
    return sum;
}

#undef nk_define_lane_sumsq_simt_

NUMKONG_DEVICE nk_u64_t nk_i8_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_i32_t const value = (signed char)row[index];
        sum += (nk_u64_t)(value * value);
    }
    return sum;
}

NUMKONG_DEVICE nk_u64_t nk_u8_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) sum += (nk_u64_t)row[index] * row[index];
    return sum;
}

NUMKONG_DEVICE nk_u64_t nk_i4_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_i32_t const value = (nk_i32_t)(nk_b4_load_(row, index) ^ 8u) - 8;
        sum += (nk_u64_t)(value * value);
    }
    return sum;
}

NUMKONG_DEVICE nk_u64_t nk_u4_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_u64_t const value = nk_b4_load_(row, index);
        sum += value * value;
    }
    return sum;
}

/*  The tensor tiles' row norms: each update adds the squares of one 16-byte chunk of a staged row,
 *  in any order, since a sum of squares has none, and the finish stores them in 32 bits. */

/** Adds the squares of both F16 halves of @p halves to @p real_sum. */
NUMKONG_DEVICE void nk_f16x2_norm_update_(nk_u32_t halves, nk_f32_t *real_sum) {
    nk_f32_t const low = __half2float(__ushort_as_half((unsigned short)(halves & 0xFFFFu)));
    nk_f32_t const high = __half2float(__ushort_as_half((unsigned short)(halves >> 16)));
    *real_sum = __fmaf_rn(high, high, __fmaf_rn(low, low, *real_sum));
}

NUMKONG_DEVICE void nk_bf16_norm_update_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_f32_t const low = __uint_as_float(words[word] << 16), high = __uint_as_float(words[word] & 0xFFFF0000u);
        *real_sum = __fmaf_rn(high, high, __fmaf_rn(low, low, *real_sum));
    }
}

NUMKONG_DEVICE void nk_f16_norm_update_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) nk_f16x2_norm_update_(words[word], real_sum);
}

/** A thread's accumulated squared norm in the 32 bits the epilogue reads: integers as summed,
 *  floats in true units. */
NUMKONG_DEVICE nk_fui32_t nk_cross_norm_finalize_(nk_cross_norm_t norm, nk_u32_t integer_sum, nk_f32_t real_sum,
                                                  nk_f32_t norm_scale) {
    nk_fui32_t result;
    if (norm == nk_cross_norm_f32_k) result.f = ((nk_f32_t)integer_sum + real_sum) * norm_scale;
    else result.u = integer_sum;
    return result;
}

/** A norm's 32 bits as F32, integers read signed or unsigned as @p norm says. */
NUMKONG_DEVICE nk_f32_t nk_cross_norm_to_f32_(nk_fui32_t bits, nk_cross_norm_t norm) {
    if (norm == nk_cross_norm_i32_k) return (nk_f32_t)bits.i;
    if (norm == nk_cross_norm_u32_k) return (nk_f32_t)bits.u;
    return bits.f;
}

#pragma endregion Norms

#pragma region Metrics

/** 1 − dot / (‖a‖ ‖b‖) clamped at 0; with a zero norm, 0 when the dot is 0 and 1 otherwise. */
NUMKONG_DEVICE nk_f64_t nk_f64_angular_(nk_f64_t dot, nk_f64_t row_norm, nk_f64_t column_norm) {
    if (!(row_norm > 0 && column_norm > 0)) return dot == 0 ? 0.0 : 1.0;
    nk_f64_t const unclipped = 1.0 - dot * (rsqrt(row_norm) * rsqrt(column_norm));
    return unclipped > 0 ? unclipped : 0.0;
}

/** √(‖a‖² + ‖b‖² − 2 · dot), with a negative radicand from rounding clamped to 0. */
NUMKONG_DEVICE nk_f64_t nk_f64_euclidean_(nk_f64_t dot, nk_f64_t row_norm, nk_f64_t column_norm) {
    nk_f64_t const squared = row_norm + column_norm - 2.0 * dot;
    return squared > 0 ? sqrt(squared) : 0.0;
}

/** 1 − dot / (‖a‖ ‖b‖) clamped at 0; with a zero norm, 0 when the dot is 0 and 1 otherwise. */
NUMKONG_DEVICE nk_f32_t nk_f32_angular_(nk_f32_t dot, nk_f32_t row_norm, nk_f32_t column_norm) {
    if (!(row_norm > 0 && column_norm > 0)) return dot == 0 ? 0.0f : 1.0f;
    nk_f32_t const unclipped = 1.0f - dot * (rsqrtf(row_norm) * rsqrtf(column_norm));
    return unclipped > 0 ? unclipped : 0.0f;
}

/** √(‖a‖² + ‖b‖² − 2 · dot), with a negative radicand from rounding clamped to 0. */
NUMKONG_DEVICE nk_f32_t nk_f32_euclidean_(nk_f32_t dot, nk_f32_t row_norm, nk_f32_t column_norm) {
    nk_f32_t const squared = row_norm + column_norm - 2.0f * dot;
    return squared > 0 ? sqrtf(squared) : 0.0f;
}

/** A tensor-core dot product as F32, the way the serial metrics read it: F32 sums and scaled
 *  integer sums times @p output_scale, other integers signed unless their norms are unsigned. */
NUMKONG_DEVICE nk_f32_t nk_cross_dot_to_f32_(nk_fui32_t sum, nk_cross_epilogue_t epilogue, nk_cross_norm_t norm,
                                             nk_f32_t output_scale) {
    if (epilogue == nk_cross_epilogue_f32_k) return sum.f * output_scale;
    if (epilogue == nk_cross_epilogue_i32_to_f32_k) return (nk_f32_t)sum.i * output_scale;
    return norm == nk_cross_norm_u32_k ? (nk_f32_t)sum.u : (nk_f32_t)sum.i;
}

#pragma endregion Metrics

/*  The two baseline tiles share one shape: each of 256 threads owns a 4 × 4 grid of one 64 × 64
 *  output tile, strided by 16, so shared-memory reads broadcast along one axis and sweep the banks
 *  along the other. The F64 tile stages F64 values and sums F64 or Dot2; the B32 tile stages 32-bit
 *  words and folds each pair with one instruction. For a metric, each thread also squares the A
 *  and, for @c symmetric, B elements it stages, and the threads staging one row merge them. Here
 *  are the stages needing no vendor instruction, which each capability's tile runs around its own
 *  folds and lane merges. */
#pragma region Baseline Tile

/** A B32 sum as F32: floats as they are, integers read signed or unsigned as serial backends do. */
NUMKONG_DEVICE nk_f32_t nk_cross_b32_to_f32_(nk_cross_accumulation_t accumulation, nk_fui32_t sum) {
    switch (accumulation) {
    case nk_cross_accumulation_f32_k:
    case nk_cross_accumulation_f16x2_k: return sum.f;
    case nk_cross_accumulation_i8x4_k:
    case nk_cross_accumulation_i4x4_k:
    case nk_cross_accumulation_i4x8_k: return (nk_f32_t)sum.i;
    default: return (nk_f32_t)sum.u;
    }
}

/** Elements of depth one staged word holds under @p accumulation. */
NUMKONG_DEVICE unsigned nk_cross_b32_dimensions_(nk_cross_accumulation_t accumulation) {
    switch (accumulation) {
    case nk_cross_accumulation_f32_k: return 1;
    case nk_cross_accumulation_f16x2_k: return 2;
    case nk_cross_accumulation_i4x8_k:
    case nk_cross_accumulation_u4x8_k: return 8;
    default: return 4;
    }
}

/** Word @p word of a row as the B32 tile stages it for @p accumulation, with every element at or
 *  past @p depth zeroed. Rows start on 16 bytes, so whole words load aligned. */
NUMKONG_DEVICE nk_u32_t nk_cross_stage_b32_(nk_cross_accumulation_t accumulation, nk_dtype_t dtype,
                                            unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * nk_cross_b32_dimensions_(accumulation);
    nk_u32_t bits = 0;
    switch (accumulation) {
    case nk_cross_accumulation_f32_k: return __float_as_uint(nk_cross_load_f32_(dtype, row, first));
    case nk_cross_accumulation_f16x2_k: {
        unsigned short halves[2] = {0, 0};
        nk_f32_t value = nk_cross_load_f32_(dtype, row, first);
        nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[0]);
        if (first + 1 < depth) {
            value = nk_cross_load_f32_(dtype, row, first + 1);
            nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[1]);
        }
        return (nk_u32_t)halves[0] | ((nk_u32_t)halves[1] << 16);
    }
    case nk_cross_accumulation_i8x4_k:
    case nk_cross_accumulation_u8x4_k:
        if (first + 4 <= depth) return *(nk_u32_t const *)(row + first);
        for (unsigned byte = 0; byte < 4; ++byte)
            if (first + byte < depth) bits |= (nk_u32_t)row[first + byte] << (byte * 8);
        return bits;
    case nk_cross_accumulation_i4x4_k:
    case nk_cross_accumulation_u4x4_k:
        // Sign- or zero-extends each nibble into its own byte, for NVIDIA's byte-wise `dp4a`.
        for (unsigned nibble = 0; nibble < 4; ++nibble) {
            if (first + nibble >= depth) continue;
            nk_u32_t const code = nk_b4_load_(row, first + nibble);
            nk_u32_t const byte = accumulation == nk_cross_accumulation_i4x4_k ? ((0u - (code & 8u)) | code) & 0xFFu
                                                                               : code;
            bits |= byte << (nibble * 8);
        }
        return bits;
    default:
        if (first + 8 <= depth) return *(nk_u32_t const *)(row + first / 2);
        for (unsigned nibble = 0; nibble < 8; ++nibble)
            if (first + nibble < depth)
                bits |= nk_b4_load_(row, first + nibble) << (nibble / 2 * 8 + (nibble & 1 ? 0 : 4));
        return bits;
    }
}

/** Stages one slab of F64 values for the tile at @p first_row and @p first_column, returning the
 *  values this thread staged in @p a_values and @p b_values for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_f64_simt_(nk_dtype_t dtype, nk_cross_tile_arguments_t const *arguments,
                                                  nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
                                                  nk_f64_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_f64_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_f64_t a_values[nk_cross_loads_simt_k],
                                                  nk_f64_t b_values[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        unsigned const element = threadIdx.x + step * nk_cross_threads_simt_k;
        unsigned const tile_row = element / nk_cross_slab_simt_k, offset = element % nk_cross_slab_simt_k;
        nk_size_t const index = slab + offset, row = first_row + tile_row, column = first_column + tile_row;
        nk_f64_t const a_value = row < arguments->row_end && index < arguments->depth
                                     ? nk_cross_load_f64_(dtype, arguments->a + row * arguments->a_stride, index)
                                     : 0;
        nk_f64_t const b_value = column < arguments->column_count && index < arguments->depth
                                     ? nk_cross_load_f64_(dtype, arguments->b + column * arguments->b_stride, index)
                                     : 0;
        a_slab[offset][tile_row] = a_value, b_slab[offset][tile_row] = b_value;
        a_values[step] = a_value, b_values[step] = b_value;
    }
}

/** Publishes the tile's row norms, which the threads staging each row merged and rounded, and its
 *  column norms, merged likewise for @c symmetric and read from @p b_norms for @c packed. */
NUMKONG_DEVICE void nk_cross_publish_norms_f64_simt_(nk_cross_triangle_t triangle,
                                                     nk_cross_tile_arguments_t const *arguments, nk_size_t first_column,
                                                     nk_f64_t const a_norms[nk_cross_loads_simt_k],
                                                     nk_f64_t const b_norms[nk_cross_loads_simt_k],
                                                     nk_f64_t norms[2][nk_cross_tile_simt_k]) {
    // Zero-depth tiles skip the slab loop's barriers, so this one retires the last epilogue reads.
    __syncthreads();
    if (threadIdx.x % nk_cross_slab_simt_k == 0)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
            unsigned const tile_row = threadIdx.x / nk_cross_slab_simt_k +
                                      step * (nk_cross_threads_simt_k / nk_cross_slab_simt_k);
            norms[0][tile_row] = a_norms[step];
            if (triangle == nk_cross_triangle_upper_k) norms[1][tile_row] = b_norms[step];
        }
    if (triangle == nk_cross_triangle_full_k && threadIdx.x < nk_cross_tile_simt_k) {
        nk_size_t const column = first_column + threadIdx.x;
        norms[1][threadIdx.x] = column < arguments->column_count ? ((nk_f64_t const *)arguments->b_norms)[column] : 0;
    }
    __syncthreads();
}

/** Writes this thread's outputs of the tile as F64: its rounded @p dots, or metrics with zeros on
 *  the diagonal of @c symmetric. */
NUMKONG_DEVICE void nk_cross_store_tile_f64_simt_(
    nk_cross_triangle_t triangle, nk_cross_metric_t metric, nk_cross_tile_arguments_t const *arguments,
    nk_size_t first_row, nk_size_t first_column,
    nk_f64_t dots[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k], nk_f64_t norms[2][nk_cross_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step) {
        unsigned const tile_row = thread_row + nk_cross_grid_side_simt_k * row_step;
        nk_size_t const row = first_row + tile_row;
        if (row >= arguments->row_end) continue;
        nk_f64_t *output = (nk_f64_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
        for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step) {
            unsigned const tile_column = thread_column + nk_cross_grid_side_simt_k * column_step;
            nk_size_t const column = first_column + tile_column;
            if (column >= arguments->column_count || (triangle == nk_cross_triangle_upper_k && column < row)) continue;
            nk_f64_t const dot = dots[row_step][column_step];
            if (metric == nk_cross_metric_dot_k) output[column] = dot;
            else if (triangle == nk_cross_triangle_upper_k && column == row) output[column] = 0;
            else if (metric == nk_cross_metric_angular_k)
                output[column] = nk_f64_angular_(dot, norms[0][tile_row], norms[1][tile_column]);
            else output[column] = nk_f64_euclidean_(dot, norms[0][tile_row], norms[1][tile_column]);
        }
    }
}

/** Stages one slab of 32-bit words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_b32_simt_(nk_dtype_t dtype, nk_cross_accumulation_t accumulation,
                                                  nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                  nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                  nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_u32_t a_words[nk_cross_loads_simt_k],
                                                  nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        unsigned const element = threadIdx.x + step * nk_cross_threads_simt_k;
        unsigned const tile_row = element / nk_cross_slab_simt_k, offset = element % nk_cross_slab_simt_k;
        nk_size_t const word = slab + offset, row = first_row + tile_row, column = first_column + tile_row;
        nk_u32_t const a_word = row < arguments->row_end && word < words
                                    ? nk_cross_stage_b32_(accumulation, dtype, arguments->a + row * arguments->a_stride,
                                                          word, arguments->depth)
                                    : 0;
        nk_u32_t const b_word = column < arguments->column_count && word < words
                                    ? nk_cross_stage_b32_(accumulation, dtype,
                                                          arguments->b + column * arguments->b_stride, word,
                                                          arguments->depth)
                                    : 0;
        a_slab[offset][tile_row] = a_word, b_slab[offset][tile_row] = b_word;
        a_words[step] = a_word, b_words[step] = b_word;
    }
}

/** Publishes the tile's row norms, which the threads staging each row merged, and its column norms,
 *  merged likewise for @c symmetric and read from the pack for @c packed. */
NUMKONG_DEVICE void nk_cross_publish_norms_b32_simt_(nk_cross_triangle_t triangle,
                                                     nk_cross_tile_arguments_t const *arguments, nk_size_t first_column,
                                                     nk_fui32_t const a_norms[nk_cross_loads_simt_k],
                                                     nk_fui32_t const b_norms[nk_cross_loads_simt_k],
                                                     nk_fui32_t norms[2][nk_cross_tile_simt_k]) {
    // Zero-depth tiles skip the slab loop's barriers, so this one retires the last epilogue reads.
    __syncthreads();
    if (threadIdx.x % nk_cross_slab_simt_k == 0)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
            unsigned const tile_row = threadIdx.x / nk_cross_slab_simt_k +
                                      step * (nk_cross_threads_simt_k / nk_cross_slab_simt_k);
            norms[0][tile_row] = a_norms[step];
            if (triangle == nk_cross_triangle_upper_k) norms[1][tile_row] = b_norms[step];
        }
    if (triangle == nk_cross_triangle_full_k && threadIdx.x < nk_cross_tile_simt_k) {
        nk_size_t const column = first_column + threadIdx.x;
        norms[1][threadIdx.x].u = column < arguments->column_count ? ((nk_u32_t const *)arguments->b_norms)[column] : 0;
    }
    __syncthreads();
}

/** Writes this thread's outputs of the tile: dots as their 32 bits, or metrics in F32 with zeros
 *  on the diagonal of @c symmetric. */
NUMKONG_DEVICE void nk_cross_store_tile_b32_simt_(
    nk_cross_accumulation_t accumulation, nk_cross_triangle_t triangle, nk_cross_metric_t metric,
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column,
    nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k],
    nk_fui32_t norms[2][nk_cross_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step) {
        unsigned const tile_row = thread_row + nk_cross_grid_side_simt_k * row_step;
        nk_size_t const row = first_row + tile_row;
        if (row >= arguments->row_end) continue;
        nk_fui32_t *output = (nk_fui32_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
        for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step) {
            unsigned const tile_column = thread_column + nk_cross_grid_side_simt_k * column_step;
            nk_size_t const column = first_column + tile_column;
            if (column >= arguments->column_count || (triangle == nk_cross_triangle_upper_k && column < row)) continue;
            if (metric == nk_cross_metric_dot_k) {
                output[column] = sums[row_step][column_step];
                continue;
            }
            nk_f32_t const dot = nk_cross_b32_to_f32_(accumulation, sums[row_step][column_step]);
            nk_f32_t const row_norm = nk_cross_b32_to_f32_(accumulation, norms[0][tile_row]);
            nk_f32_t const column_norm = nk_cross_b32_to_f32_(accumulation, norms[1][tile_column]);
            if (triangle == nk_cross_triangle_upper_k && column == row) output[column].f = 0;
            else if (metric == nk_cross_metric_angular_k)
                output[column].f = nk_f32_angular_(dot, row_norm, column_norm);
            else output[column].f = nk_f32_euclidean_(dot, row_norm, column_norm);
        }
    }
}

/** Decodes the scale of block @p block of row `first + threadIdx.x` times @p tensor_scale into
 *  @p scales, zero past @p rows, for the block-scaled tiles. */
NUMKONG_DEVICE void nk_cross_stage_scale_simt_(nk_block_scaled_format_t format, unsigned char const *row_scales,
                                               nk_size_t scales_stride, nk_f32_t tensor_scale, nk_size_t first,
                                               nk_size_t rows, nk_size_t block, nk_f32_t *scales) {
    nk_size_t const row = first + threadIdx.x;
    scales[threadIdx.x] = row < rows ? nk_block_scaled_decode_scale_serial_(row_scales[row * scales_stride + block],
                                                                            format.scale_dtype) *
                                           tensor_scale
                                     : 0;
}

#pragma endregion Baseline Tile

/*  Every tile, baseline or tensor, takes its own leading arguments, then the triangle, the metric
 *  and the launch arguments, so one generator per shape serves them all: the site passes the tile's
 *  own arguments last, through the variadic tail. */
#pragma region Cross Macros

/**
 *  @brief Generates the bytes of a device pack: the header, then rows padded by
 *      @c nk_cross_padded_values_simt_, then one norm per column.
 *  @param[in] depth_simd_dimensions The dimensions each row rounds up to, 16 bytes' worth.
 *  @sa nk_define_cross_pack_size_ for the host original.
 */
#define nk_define_cross_pack_size_simt_(input_type_name, isa_suffix, packed_value_type, norm_value_type,             \
                                        depth_simd_dimensions, dimensions_per_value)                                 \
    NUMKONG_API nk_status_t nk_dots_pack_size_##input_type_name##_##isa_suffix(nk_size_t column_count,               \
                                                                               nk_size_t depth, nk_size_t *bytes) {  \
        nk_assert_(depth % dimensions_per_value == 0);                                                               \
        nk_size_t const row_bytes = nk_cross_padded_values_simt_(depth, depth_simd_dimensions, dimensions_per_value, \
                                                                 sizeof(nk_##packed_value_type##_t)) *               \
                                    sizeof(nk_##packed_value_type##_t);                                              \
        *bytes = sizeof(nk_cross_packed_buffer_header_t) +                                                           \
                 column_count * (row_bytes + nk_cross_scales_stride_(nk_##input_type_name##_k, depth) +              \
                                 sizeof(nk_##norm_value_type##_t));                                                  \
        return nk_success_k;                                                                                         \
    }

#pragma endregion Cross Macros

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_DOTS_SIMT_CUH
