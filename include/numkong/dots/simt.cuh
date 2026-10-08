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
#include "numkong/simt.cuh"      // `nk_diagonal_band_row_range_simt_`
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

/** Elements of depth one staged 32-bit word holds, by the shape of the word. */
enum {
    nk_cross_dimensions_f32x1_simt_k = 1,
    nk_cross_dimensions_f16x2_simt_k = 2,
    nk_cross_dimensions_b8x4_simt_k = 4,
    nk_cross_dimensions_b4x8_simt_k = 8,
};

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
    nk_size_t rows_begin;

    /** One past the last output row. */
    nk_size_t rows_end;

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

    /** Wrapping U32 sums of a signed integer type, whose dots are I32. */
    nk_cross_norm_i32_k,

    /** Wrapping U32 sums of an unsigned integer type, whose dots are U32. */
    nk_cross_norm_u32_k,
} nk_cross_norm_t;

#pragma endregion Configuration

#pragma region Pack Layout

/** Storage values in one packed GPU row: @c nk_cross_padded_values_serial_ without its power-of-two
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
NUMKONG_DEVICE void nk_i4x8_to_i8x8_simt_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_nibbles = codes & 0x0F0F0F0Fu, high_nibbles = (codes >> 4) & 0x0F0F0F0Fu;
    *low = (((low_nibbles ^ 0x08080808u) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
    *high = (((high_nibbles ^ 0x08080808u) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}

/** Eight U4 nibbles become two registers of four U8. */
NUMKONG_DEVICE void nk_u4x8_to_u8x8_simt_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    *low = codes & 0x0F0F0F0Fu, *high = (codes >> 4) & 0x0F0F0F0Fu;
}

/** Four E2M3 magnitudes times 8 as U8: @c m where e = 0, otherwise (8 + m) << (e − 1), at most
 *  60. */
NUMKONG_DEVICE nk_u32_t nk_e2m3x4_to_u8x4_magnitudes_simt_(nk_u32_t codes) {
    nk_u32_t const exponent_low = (codes >> 3) & 0x01010101u, exponent_high = (codes >> 4) & 0x01010101u;
    nk_u32_t const significand = (codes & 0x07070707u) | ((exponent_low | exponent_high) << 3);
    nk_u32_t const doubled_mask = exponent_high * 0xFFu, quadrupled_mask = (exponent_high & exponent_low) * 0xFFu;
    return significand + (significand & doubled_mask) + ((significand << 1) & quadrupled_mask);
}

/** One E3M2 code over 4096. */
NUMKONG_DEVICE nk_f32_t nk_e3m2_to_scaled_f32_simt_(nk_u32_t code) {
    return __half2float(__ushort_as_half((unsigned short)(((code & 0x1Fu) << 8) | ((code & 0x20u) << 10))));
}

/** Keeps a packed byte as it is. */
NUMKONG_DEVICE unsigned char nk_load_b8_simt_(unsigned char value) { return value; }

NUMKONG_DEVICE nk_f64_t nk_f64_load_f64_simt_(unsigned char const *row, nk_size_t index) {
    nk_f64_t const *elements = (nk_f64_t const *)row;
    return elements[index];
}

NUMKONG_DEVICE nk_f64_t nk_f32_load_f64_simt_(unsigned char const *row, nk_size_t index) {
    nk_f32_t const *elements = (nk_f32_t const *)row;
    return (nk_f64_t)elements[index];
}

NUMKONG_DEVICE nk_f32_t nk_bf16_load_f32_simt_(unsigned char const *row, nk_size_t index) {
    return __uint_as_float(((nk_u32_t)row[index * 2] | ((nk_u32_t)row[index * 2 + 1] << 8)) << 16);
}

NUMKONG_DEVICE nk_f32_t nk_f16_load_f32_simt_(unsigned char const *row, nk_size_t index) {
    return __half2float(__ushort_as_half((unsigned short)(row[index * 2] | (row[index * 2 + 1] << 8))));
}

NUMKONG_DEVICE nk_f32_t nk_e5m2_load_f32_simt_(unsigned char const *row, nk_size_t index) {
    nk_f32_t value;
    nk_e5m2_to_f32_simt_(row + index, &value);
    return value;
}

NUMKONG_DEVICE nk_f32_t nk_e4m3_load_f32_simt_(unsigned char const *row, nk_size_t index) {
    nk_f32_t value;
    nk_e4m3_to_f32_simt_(row + index, &value);
    return value;
}

NUMKONG_DEVICE nk_f32_t nk_e3m2_load_f32_simt_(unsigned char const *row, nk_size_t index) {
    return nk_e3m2_to_scaled_f32_simt_(row[index]) * 4096.0f;
}

NUMKONG_DEVICE nk_f32_t nk_e2m3_load_f32_simt_(unsigned char const *row, nk_size_t index) {
    unsigned const code = row[index];
    nk_f32_t const magnitude = (nk_f32_t)nk_e2m3x4_to_u8x4_magnitudes_simt_(code) * 0.125f;
    return code & 0x20u ? -magnitude : magnitude;
}

NUMKONG_DEVICE unsigned nk_b4_load_simt_(unsigned char const *row, nk_size_t index) {
    return (index & 1) ? (row[index / 2] & 0x0Fu) : (row[index / 2] >> 4);
}

NUMKONG_DEVICE nk_f32_t nk_e2m1_load_f32_simt_(unsigned char const *row, nk_size_t index) {
    unsigned const code = nk_b4_load_simt_(row, index), exponent = (code >> 1) & 3u;
    nk_f32_t const magnitude = exponent ? (nk_f32_t)((2u + (code & 1u)) << (exponent - 1)) * 0.5f
                                        : (nk_f32_t)(code & 1u) * 0.5f;
    return code & 8u ? -magnitude : magnitude;
}

#pragma endregion Conversions

/*  Each lane share returns a lane's part of a column's sum of squares, over indices `lane + 32 × k`
 *  below @c depth, which the pack merges across the 32 lanes: F64 for floats and exact 64-bit sums
 *  for integers, while block-scaled columns store `nk_cross_scaled_pack_norm_<dtype>_simt_` in F32
 *  rebased form. The Dot2-compensated shares of F64 and F32 inputs, and the merges, round through
 *  each vendor's primitives in its own header. Rows read bytewise, since the source's stride need
 *  not be element-aligned. */
#pragma region Norms

#define nk_define_lane_sumsq_simt_(input_type_name)                                                                   \
    NUMKONG_DEVICE nk_f64_t nk_##input_type_name##_lane_sumsq_simt_(unsigned char const *row, nk_size_t depth,        \
                                                                    unsigned lane, unsigned char const *, nk_f32_t) { \
        nk_f64_t sum = 0;                                                                                             \
        for (nk_size_t index = lane; index < depth; index += 32) {                                                    \
            nk_f64_t const value = nk_##input_type_name##_load_f32_simt_(row, index);                                 \
            sum = __fma_rn(value, value, sum);                                                                        \
        }                                                                                                             \
        return sum;                                                                                                   \
    }

nk_define_lane_sumsq_simt_(bf16)
nk_define_lane_sumsq_simt_(f16)
nk_define_lane_sumsq_simt_(e5m2)
nk_define_lane_sumsq_simt_(e4m3)
nk_define_lane_sumsq_simt_(e3m2)
nk_define_lane_sumsq_simt_(e2m3)
nk_define_lane_sumsq_simt_(e2m1)

#undef nk_define_lane_sumsq_simt_

NUMKONG_DEVICE nk_u64_t nk_i8_lane_sumsq_simt_(unsigned char const *row, nk_size_t depth, unsigned lane,
                                               unsigned char const *, nk_f32_t) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_i32_t const value = (signed char)row[index];
        sum += (nk_u64_t)(value * value);
    }
    return sum;
}

NUMKONG_DEVICE nk_u64_t nk_u8_lane_sumsq_simt_(unsigned char const *row, nk_size_t depth, unsigned lane,
                                               unsigned char const *, nk_f32_t) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) sum += (nk_u64_t)row[index] * row[index];
    return sum;
}

NUMKONG_DEVICE nk_u64_t nk_i4_lane_sumsq_simt_(unsigned char const *row, nk_size_t depth, unsigned lane,
                                               unsigned char const *, nk_f32_t) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_i32_t const value = (nk_i32_t)(nk_b4_load_simt_(row, index) ^ 8u) - 8;
        sum += (nk_u64_t)(value * value);
    }
    return sum;
}

NUMKONG_DEVICE nk_u64_t nk_u4_lane_sumsq_simt_(unsigned char const *row, nk_size_t depth, unsigned lane,
                                               unsigned char const *, nk_f32_t) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_u64_t const value = nk_b4_load_simt_(row, index);
        sum += value * value;
    }
    return sum;
}

/*  The tensor tiles' row norms: each update adds the squares of one 16-byte chunk of a staged row,
 *  in any order, since a sum of squares has none, and the finish stores them in 32 bits. */

/** Adds the squares of both F16 halves of @p halves to @p real_sum. */
NUMKONG_DEVICE void nk_f16x2_norm_update_simt_(nk_u32_t halves, nk_f32_t *real_sum) {
    nk_f32_t const low = __half2float(__ushort_as_half((unsigned short)(halves & 0xFFFFu)));
    nk_f32_t const high = __half2float(__ushort_as_half((unsigned short)(halves >> 16)));
    *real_sum = __fmaf_rn(high, high, __fmaf_rn(low, low, *real_sum));
}

NUMKONG_DEVICE void nk_bf16_norm_update_simt_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_f32_t const low = __uint_as_float(words[word] << 16), high = __uint_as_float(words[word] & 0xFFFF0000u);
        *real_sum = __fmaf_rn(high, high, __fmaf_rn(low, low, *real_sum));
    }
}

NUMKONG_DEVICE void nk_f16_norm_update_simt_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) nk_f16x2_norm_update_simt_(words[word], real_sum);
}

/** A thread's accumulated squared norm in the 32 bits the epilogue reads: integers as summed,
 *  floats in true units. */
NUMKONG_DEVICE nk_fui32_t nk_cross_norm_finalize_simt_(nk_cross_norm_t norm, nk_u32_t integer_sum, nk_f32_t real_sum,
                                                       nk_f32_t norm_scale) {
    nk_fui32_t result;
    if (norm == nk_cross_norm_f32_k) result.f = ((nk_f32_t)integer_sum + real_sum) * norm_scale;
    else result.u = integer_sum;
    return result;
}

#pragma endregion Norms

#pragma region Metrics

/** 1 − dot / (‖a‖ ‖b‖) clamped at 0, by the rule of `spatials.h`: 0 for two zero norms, 1 for one
 *  zero norm or a zero dot, and NaN for a NaN dot. */
NUMKONG_DEVICE nk_f64_t nk_f64_angular_simt_(nk_f64_t dot, nk_f64_t row_norm, nk_f64_t column_norm) {
    if (dot != dot) return dot;
    if (row_norm == 0 && column_norm == 0) return 0.0;
    if (dot == 0 || row_norm == 0 || column_norm == 0) return 1.0;
    nk_f64_t const unclipped = 1.0 - dot * rsqrt(row_norm) * rsqrt(column_norm);
    return unclipped < 0 ? 0.0 : unclipped;
}

/** √(‖a‖² + ‖b‖² − 2 · dot), with a negative radicand from rounding clamped to 0 and NaN kept. */
NUMKONG_DEVICE nk_f64_t nk_f64_euclidean_simt_(nk_f64_t dot, nk_f64_t row_norm, nk_f64_t column_norm) {
    nk_f64_t const squared = row_norm + column_norm - 2.0 * dot;
    return squared < 0 ? 0.0 : sqrt(squared);
}

/** 1 − dot / (‖a‖ ‖b‖) clamped at 0, by the rule of `spatials.h`: 0 for two zero norms, 1 for one
 *  zero norm or a zero dot, and NaN for a NaN dot. */
NUMKONG_DEVICE nk_f32_t nk_f32_angular_simt_(nk_f32_t dot, nk_f32_t row_norm, nk_f32_t column_norm) {
    if (dot != dot) return dot;
    if (row_norm == 0 && column_norm == 0) return 0.0f;
    if (dot == 0 || row_norm == 0 || column_norm == 0) return 1.0f;
    nk_f32_t const unclipped = 1.0f - dot * rsqrtf(row_norm) * rsqrtf(column_norm);
    return unclipped < 0 ? 0.0f : unclipped;
}

/** √(‖a‖² + ‖b‖² − 2 · dot), with a negative radicand from rounding clamped to 0 and NaN kept. */
NUMKONG_DEVICE nk_f32_t nk_f32_euclidean_simt_(nk_f32_t dot, nk_f32_t row_norm, nk_f32_t column_norm) {
    nk_f32_t const squared = row_norm + column_norm - 2.0f * dot;
    return squared < 0 ? 0.0f : sqrtf(squared);
}

/** The angular or euclidean distance of an integer pair from its exact @p dot bits, I32 or U32 as
 *  @p norm says, and wrapping U32 norms, as the serial backends finish it: ab − d² and a + b − 2d
 *  stay exact in 64 bits, so equal rows are exactly 0 apart, and one F64 tail rounds into F32. */
NUMKONG_DEVICE nk_f32_t nk_cross_integer_metric_simt_(nk_cross_metric_t metric, nk_cross_norm_t norm, nk_u32_t dot_bits,
                                                      nk_u32_t row_norm, nk_u32_t column_norm) {
    nk_i64_t const dot = norm == nk_cross_norm_u32_k ? (nk_i64_t)dot_bits : (nk_i64_t)(nk_i32_t)dot_bits;
    if (metric == nk_cross_metric_euclidean_k) {
        nk_i64_t const distance_sq = (nk_i64_t)row_norm + column_norm - 2 * dot;
        return distance_sq > 0 ? (nk_f32_t)sqrt((nk_f64_t)distance_sq) : 0.0f;
    }
    nk_u64_t const product = (nk_u64_t)row_norm * column_norm;
    if (product == 0) return (row_norm | column_norm) != 0 ? 1.0f : 0.0f;
    nk_f64_t const product_f64 = (nk_f64_t)product, root = sqrt(product_f64);
    if (dot <= 0) return (nk_f32_t)(1.0 - (nk_f64_t)dot / root);
    nk_u64_t const dot_sq = (nk_u64_t)dot * (nk_u64_t)dot;
    return (nk_f32_t)((nk_f64_t)(product - dot_sq) / (product_f64 + (nk_f64_t)dot * root));
}

/** A tensor-core dot product of a floating type as F32: F32 sums, or integer sums of its scaled
 *  codes, times @p output_scale. */
NUMKONG_DEVICE nk_f32_t nk_cross_dot_to_f32_simt_(nk_fui32_t sum, nk_cross_epilogue_t epilogue, nk_f32_t output_scale) {
    return epilogue == nk_cross_epilogue_f32_k ? sum.f * output_scale : (nk_f32_t)sum.i * output_scale;
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

/** Whether @p band asks for a Gram matrix: the diagonal and above, whose lower half is mirrored. */
NUMKONG_DEVICE int nk_cross_symmetric_simt_(nk_diagonal_band_t band) { return band.subdiagonals == 0; }

/** One thread's cell of a staged slab: its row in the tile, its word in the slab, and the rows of A
 *  and B to read it from, null past the operand's rows or past @c words . */
typedef struct {
    unsigned tile_row, offset;
    nk_size_t word;
    unsigned char const *a_row, *b_row;
} nk_cross_cell_simt_t;

NUMKONG_DEVICE nk_cross_cell_simt_t nk_cross_cell_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                        nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                        unsigned step) {
    unsigned const element = threadIdx.x + step * nk_cross_threads_simt_k;
    nk_cross_cell_simt_t cell;
    cell.tile_row = element / nk_cross_slab_simt_k, cell.offset = element % nk_cross_slab_simt_k;
    cell.word = slab + cell.offset;
    nk_size_t const row = first_row + cell.tile_row, column = first_column + cell.tile_row;
    cell.a_row = row < arguments->rows_end && cell.word < words ? arguments->a + row * arguments->a_stride
                                                                : NUMKONG_NULL;
    cell.b_row = column < arguments->column_count && cell.word < words ? arguments->b + column * arguments->b_stride
                                                                       : NUMKONG_NULL;
    return cell;
}

/** Stages one slab of F64 values of an F64 row for the tile at @p first_row and @p first_column,
 *  returning the values this thread staged in @p a_values and @p b_values for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_f64_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                  nk_size_t first_column, nk_size_t slab,
                                                  nk_f64_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_f64_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_f64_t a_values[nk_cross_loads_simt_k],
                                                  nk_f64_t b_values[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab,
                                                              arguments->depth, step);
        nk_f64_t const a_value = cell.a_row ? nk_f64_load_f64_simt_(cell.a_row, cell.word) : 0;
        nk_f64_t const b_value = cell.b_row ? nk_f64_load_f64_simt_(cell.b_row, cell.word) : 0;
        a_slab[cell.offset][cell.tile_row] = a_value, b_slab[cell.offset][cell.tile_row] = b_value;
        a_values[step] = a_value, b_values[step] = b_value;
    }
}

/** @c nk_cross_stage_slab_f64_simt_ for F32 rows, widened to F64. */
NUMKONG_DEVICE void nk_cross_stage_slab_f32_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                  nk_size_t first_column, nk_size_t slab,
                                                  nk_f64_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_f64_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_f64_t a_values[nk_cross_loads_simt_k],
                                                  nk_f64_t b_values[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab,
                                                              arguments->depth, step);
        nk_f64_t const a_value = cell.a_row ? nk_f32_load_f64_simt_(cell.a_row, cell.word) : 0;
        nk_f64_t const b_value = cell.b_row ? nk_f32_load_f64_simt_(cell.b_row, cell.word) : 0;
        a_slab[cell.offset][cell.tile_row] = a_value, b_slab[cell.offset][cell.tile_row] = b_value;
        a_values[step] = a_value, b_values[step] = b_value;
    }
}

/** Publishes the tile's row norms, which the threads staging each row merged and rounded, and its
 *  column norms, merged likewise for a Gram matrix and read from @p b_norms for a packed one. */
NUMKONG_DEVICE void nk_cross_publish_norms_f64_simt_(nk_diagonal_band_t band,
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
            if (nk_cross_symmetric_simt_(band)) norms[1][tile_row] = b_norms[step];
        }
    if (!nk_cross_symmetric_simt_(band) && threadIdx.x < nk_cross_tile_simt_k) {
        nk_size_t const column = first_column + threadIdx.x;
        nk_f64_t const *column_norms = (nk_f64_t const *)arguments->b_norms;
        norms[1][threadIdx.x] = column < arguments->column_count ? column_norms[column] : 0;
    }
    __syncthreads();
}

/** Writes this thread's outputs of the tile as F64: its rounded @p dots, or metrics with zeros on
 *  the diagonal of a Gram matrix. */
NUMKONG_DEVICE void nk_cross_store_tile_f64_simt_(
    nk_diagonal_band_t band, nk_cross_metric_t metric, nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
    nk_size_t first_column, nk_f64_t dots[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k],
    nk_f64_t norms[2][nk_cross_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step) {
        unsigned const tile_row = thread_row + nk_cross_grid_side_simt_k * row_step;
        nk_size_t const row = first_row + tile_row;
        if (row >= arguments->rows_end) continue;
        nk_size_t column_begin, column_end;
        nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin, &column_end);
        nk_f64_t *output = (nk_f64_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
        for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step) {
            unsigned const tile_column = thread_column + nk_cross_grid_side_simt_k * column_step;
            nk_size_t const column = first_column + tile_column;
            if (column < column_begin || column >= column_end) continue;
            nk_f64_t const dot = dots[row_step][column_step];
            if (metric == nk_cross_metric_dot_k) output[column] = dot;
            else if (nk_cross_symmetric_simt_(band) && column == row) output[column] = 0;
            else if (metric == nk_cross_metric_angular_k)
                output[column] = nk_f64_angular_simt_(dot, norms[0][tile_row], norms[1][tile_column]);
            else output[column] = nk_f64_euclidean_simt_(dot, norms[0][tile_row], norms[1][tile_column]);
        }
    }
}

/** Stores the staged words of a cell, and returns them for the metric's norms. */
NUMKONG_DEVICE void nk_cross_store_cell_simt_(nk_cross_cell_simt_t const *cell, nk_u32_t a_word, nk_u32_t b_word,
                                              nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                              nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                              nk_u32_t *a_staged, nk_u32_t *b_staged) {
    a_slab[cell->offset][cell->tile_row] = a_word, b_slab[cell->offset][cell->tile_row] = b_word;
    *a_staged = a_word, *b_staged = b_word;
}

/** The word of a row of bytes: 4 consecutive ones, zeros past @p depth. Rows start on 16 bytes, so
 *  whole words load aligned. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_b8x4_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 4;
    if (first + 4 <= depth) return *(nk_u32_t const *)(row + first);
    nk_u32_t bits = 0;
    for (unsigned byte = 0; byte < 4; ++byte)
        if (first + byte < depth) bits |= (nk_u32_t)row[first + byte] << (byte * 8);
    return bits;
}

/** The word of a row of nibbles as packed: 8 consecutive ones, zeros past @p depth. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_b4x8_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 8;
    if (first + 8 <= depth) return *(nk_u32_t const *)(row + first / 2);
    nk_u32_t bits = 0;
    for (unsigned nibble = 0; nibble < 8; ++nibble)
        if (first + nibble < depth)
            bits |= nk_b4_load_simt_(row, first + nibble) << (nibble / 2 * 8 + (nibble & 1 ? 0 : 4));
    return bits;
}

/** The word of a row of signed nibbles, each sign-extended into its own byte for byte-wise dots. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_i4x4_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 4;
    nk_u32_t bits = 0;
    for (unsigned nibble = 0; nibble < 4; ++nibble) {
        if (first + nibble >= depth) continue;
        nk_u32_t const code = nk_b4_load_simt_(row, first + nibble);
        bits |= (((0u - (code & 8u)) | code) & 0xFFu) << (nibble * 8);
    }
    return bits;
}

/** The word of a row of unsigned nibbles, each zero-extended into its own byte. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_u4x4_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 4;
    nk_u32_t bits = 0;
    for (unsigned nibble = 0; nibble < 4; ++nibble)
        if (first + nibble < depth) bits |= nk_b4_load_simt_(row, first + nibble) << (nibble * 8);
    return bits;
}

/** The word of a row of F16: 2 consecutive elements as F16 pairs, low first, a second one past
 *  @p depth as zero. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_f16_f16x2_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 2;
    unsigned short halves[2] = {0, 0};
    nk_f32_t value = nk_f16_load_f32_simt_(row, first);
    nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[0]);
    if (first + 1 < depth) {
        value = nk_f16_load_f32_simt_(row, first + 1);
        nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[1]);
    }
    return (nk_u32_t)halves[0] | ((nk_u32_t)halves[1] << 16);
}

/** The word of a row of E5M2: 2 consecutive elements as F16 pairs, low first, a second one past
 *  @p depth as zero. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_e5m2_f16x2_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 2;
    unsigned short halves[2] = {0, 0};
    nk_f32_t value = nk_e5m2_load_f32_simt_(row, first);
    nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[0]);
    if (first + 1 < depth) {
        value = nk_e5m2_load_f32_simt_(row, first + 1);
        nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[1]);
    }
    return (nk_u32_t)halves[0] | ((nk_u32_t)halves[1] << 16);
}

/** The word of a row of E4M3: 2 consecutive elements as F16 pairs, low first, a second one past
 *  @p depth as zero. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_e4m3_f16x2_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 2;
    unsigned short halves[2] = {0, 0};
    nk_f32_t value = nk_e4m3_load_f32_simt_(row, first);
    nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[0]);
    if (first + 1 < depth) {
        value = nk_e4m3_load_f32_simt_(row, first + 1);
        nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[1]);
    }
    return (nk_u32_t)halves[0] | ((nk_u32_t)halves[1] << 16);
}

/** The word of a row of E3M2: 2 consecutive elements as F16 pairs, low first, a second one past
 *  @p depth as zero. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_e3m2_f16x2_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 2;
    unsigned short halves[2] = {0, 0};
    nk_f32_t value = nk_e3m2_load_f32_simt_(row, first);
    nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[0]);
    if (first + 1 < depth) {
        value = nk_e3m2_load_f32_simt_(row, first + 1);
        nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[1]);
    }
    return (nk_u32_t)halves[0] | ((nk_u32_t)halves[1] << 16);
}

/** The word of a row of E2M3: 2 consecutive elements as F16 pairs, low first, a second one past
 *  @p depth as zero. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_e2m3_f16x2_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 2;
    unsigned short halves[2] = {0, 0};
    nk_f32_t value = nk_e2m3_load_f32_simt_(row, first);
    nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[0]);
    if (first + 1 < depth) {
        value = nk_e2m3_load_f32_simt_(row, first + 1);
        nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[1]);
    }
    return (nk_u32_t)halves[0] | ((nk_u32_t)halves[1] << 16);
}

/** The word of a row of E2M1: 2 consecutive elements as F16 pairs, low first, a second one past
 *  @p depth as zero. */
NUMKONG_DEVICE nk_u32_t nk_cross_word_e2m1_f16x2_simt_(unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * 2;
    unsigned short halves[2] = {0, 0};
    nk_f32_t value = nk_e2m1_load_f32_simt_(row, first);
    nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[0]);
    if (first + 1 < depth) {
        value = nk_e2m1_load_f32_simt_(row, first + 1);
        nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[1]);
    }
    return (nk_u32_t)halves[0] | ((nk_u32_t)halves[1] << 16);
}

/** Stages one slab of BF16 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_bf16_f32x1_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? __float_as_uint(nk_bf16_load_f32_simt_(cell.a_row, cell.word)) : 0;
        nk_u32_t const b_word = cell.b_row ? __float_as_uint(nk_bf16_load_f32_simt_(cell.b_row, cell.word)) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of F16 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_f16_f32x1_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                        nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                        nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                        nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                        nk_u32_t a_words[nk_cross_loads_simt_k],
                                                        nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? __float_as_uint(nk_f16_load_f32_simt_(cell.a_row, cell.word)) : 0;
        nk_u32_t const b_word = cell.b_row ? __float_as_uint(nk_f16_load_f32_simt_(cell.b_row, cell.word)) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E5M2 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e5m2_f32x1_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? __float_as_uint(nk_e5m2_load_f32_simt_(cell.a_row, cell.word)) : 0;
        nk_u32_t const b_word = cell.b_row ? __float_as_uint(nk_e5m2_load_f32_simt_(cell.b_row, cell.word)) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E4M3 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e4m3_f32x1_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? __float_as_uint(nk_e4m3_load_f32_simt_(cell.a_row, cell.word)) : 0;
        nk_u32_t const b_word = cell.b_row ? __float_as_uint(nk_e4m3_load_f32_simt_(cell.b_row, cell.word)) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E3M2 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e3m2_f32x1_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? __float_as_uint(nk_e3m2_load_f32_simt_(cell.a_row, cell.word)) : 0;
        nk_u32_t const b_word = cell.b_row ? __float_as_uint(nk_e3m2_load_f32_simt_(cell.b_row, cell.word)) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E2M3 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e2m3_f32x1_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? __float_as_uint(nk_e2m3_load_f32_simt_(cell.a_row, cell.word)) : 0;
        nk_u32_t const b_word = cell.b_row ? __float_as_uint(nk_e2m3_load_f32_simt_(cell.b_row, cell.word)) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E2M1 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e2m1_f32x1_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? __float_as_uint(nk_e2m1_load_f32_simt_(cell.a_row, cell.word)) : 0;
        nk_u32_t const b_word = cell.b_row ? __float_as_uint(nk_e2m1_load_f32_simt_(cell.b_row, cell.word)) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of F16 words as F16 pairs for the tile at @p first_row and @p first_column,
 *  returning the words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_f16_f16x2_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                        nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                        nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                        nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                        nk_u32_t a_words[nk_cross_loads_simt_k],
                                                        nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_f16_f16x2_simt_(cell.a_row, cell.word, arguments->depth) : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_f16_f16x2_simt_(cell.b_row, cell.word, arguments->depth) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E5M2 words as F16 pairs for the tile at @p first_row and @p first_column,
 *  returning the words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e5m2_f16x2_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_e5m2_f16x2_simt_(cell.a_row, cell.word, arguments->depth)
                                           : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_e5m2_f16x2_simt_(cell.b_row, cell.word, arguments->depth)
                                           : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E4M3 words as F16 pairs for the tile at @p first_row and @p first_column,
 *  returning the words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e4m3_f16x2_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_e4m3_f16x2_simt_(cell.a_row, cell.word, arguments->depth)
                                           : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_e4m3_f16x2_simt_(cell.b_row, cell.word, arguments->depth)
                                           : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E3M2 words as F16 pairs for the tile at @p first_row and @p first_column,
 *  returning the words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e3m2_f16x2_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_e3m2_f16x2_simt_(cell.a_row, cell.word, arguments->depth)
                                           : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_e3m2_f16x2_simt_(cell.b_row, cell.word, arguments->depth)
                                           : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E2M3 words as F16 pairs for the tile at @p first_row and @p first_column,
 *  returning the words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e2m3_f16x2_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_e2m3_f16x2_simt_(cell.a_row, cell.word, arguments->depth)
                                           : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_e2m3_f16x2_simt_(cell.b_row, cell.word, arguments->depth)
                                           : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of E2M1 words as F16 pairs for the tile at @p first_row and @p first_column,
 *  returning the words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_e2m1_f16x2_simt_(
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
    nk_size_t words, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1], nk_u32_t a_words[nk_cross_loads_simt_k],
    nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_e2m1_f16x2_simt_(cell.a_row, cell.word, arguments->depth)
                                           : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_e2m1_f16x2_simt_(cell.b_row, cell.word, arguments->depth)
                                           : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of I8 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_i8_b8x4_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                      nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                      nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t a_words[nk_cross_loads_simt_k],
                                                      nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_b8x4_simt_(cell.a_row, cell.word, arguments->depth) : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_b8x4_simt_(cell.b_row, cell.word, arguments->depth) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of U8 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_u8_b8x4_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                      nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                      nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t a_words[nk_cross_loads_simt_k],
                                                      nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_b8x4_simt_(cell.a_row, cell.word, arguments->depth) : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_b8x4_simt_(cell.b_row, cell.word, arguments->depth) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of I4 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_i4_i4x4_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                      nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                      nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t a_words[nk_cross_loads_simt_k],
                                                      nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_i4x4_simt_(cell.a_row, cell.word, arguments->depth) : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_i4x4_simt_(cell.b_row, cell.word, arguments->depth) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of U4 words for the tile at @p first_row and @p first_column, returning the
 *  words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_u4_u4x4_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                      nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                      nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t a_words[nk_cross_loads_simt_k],
                                                      nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_u4x4_simt_(cell.a_row, cell.word, arguments->depth) : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_u4x4_simt_(cell.b_row, cell.word, arguments->depth) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of I4 words as packed nibbles for the tile at @p first_row and @p first_column,
 *  returning the words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_i4_b4x8_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                      nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                      nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t a_words[nk_cross_loads_simt_k],
                                                      nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_b4x8_simt_(cell.a_row, cell.word, arguments->depth) : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_b4x8_simt_(cell.b_row, cell.word, arguments->depth) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Stages one slab of U4 words as packed nibbles for the tile at @p first_row and @p first_column,
 *  returning the words this thread staged in @p a_words and @p b_words for a metric's norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_u4_b4x8_simt_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                      nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                      nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                      nk_u32_t a_words[nk_cross_loads_simt_k],
                                                      nk_u32_t b_words[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        nk_cross_cell_simt_t const cell = nk_cross_cell_simt_(arguments, first_row, first_column, slab, words, step);
        nk_u32_t const a_word = cell.a_row ? nk_cross_word_b4x8_simt_(cell.a_row, cell.word, arguments->depth) : 0;
        nk_u32_t const b_word = cell.b_row ? nk_cross_word_b4x8_simt_(cell.b_row, cell.word, arguments->depth) : 0;
        nk_cross_store_cell_simt_(&cell, a_word, b_word, a_slab, b_slab, &a_words[step], &b_words[step]);
    }
}

/** Publishes the tile's row norms, which the threads staging each row merged, and its column norms,
 *  merged likewise for a Gram matrix and read from the pack for a packed one. */
NUMKONG_DEVICE void nk_cross_publish_norms_b32_simt_(nk_diagonal_band_t band,
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
            if (nk_cross_symmetric_simt_(band)) norms[1][tile_row] = b_norms[step];
        }
    if (!nk_cross_symmetric_simt_(band) && threadIdx.x < nk_cross_tile_simt_k) {
        nk_size_t const column = first_column + threadIdx.x;
        nk_u32_t const *column_norms = (nk_u32_t const *)arguments->b_norms;
        norms[1][threadIdx.x].u = column < arguments->column_count ? column_norms[column] : 0;
    }
    __syncthreads();
}

/** Writes this thread's outputs of a tile of F32 sums: dots as their 32 bits, or metrics in F32
 *  with zeros on the diagonal of a Gram matrix. */
NUMKONG_DEVICE void nk_cross_store_tile_f32_simt_(
    nk_diagonal_band_t band, nk_cross_metric_t metric, nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
    nk_size_t first_column, nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k],
    nk_fui32_t norms[2][nk_cross_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step) {
        unsigned const tile_row = thread_row + nk_cross_grid_side_simt_k * row_step;
        nk_size_t const row = first_row + tile_row;
        if (row >= arguments->rows_end) continue;
        nk_size_t column_begin, column_end;
        nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin, &column_end);
        nk_fui32_t *output = (nk_fui32_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
        for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step) {
            unsigned const tile_column = thread_column + nk_cross_grid_side_simt_k * column_step;
            nk_size_t const column = first_column + tile_column;
            if (column < column_begin || column >= column_end) continue;
            nk_fui32_t const sum = sums[row_step][column_step];
            nk_fui32_t const row_norm = norms[0][tile_row], column_norm = norms[1][tile_column];
            if (metric == nk_cross_metric_dot_k) output[column] = sum;
            else if (nk_cross_symmetric_simt_(band) && column == row) output[column].f = 0;
            else if (metric == nk_cross_metric_angular_k)
                output[column].f = nk_f32_angular_simt_(sum.f, row_norm.f, column_norm.f);
            else output[column].f = nk_f32_euclidean_simt_(sum.f, row_norm.f, column_norm.f);
        }
    }
}

/** Writes this thread's outputs of a tile of wrapping integer sums, signed or unsigned as @p norm
 *  says: dots as their 32 bits, or metrics in F32 with zeros on the diagonal of a Gram matrix. */
NUMKONG_DEVICE void nk_cross_store_tile_integer_simt_(
    nk_cross_norm_t norm, nk_diagonal_band_t band, nk_cross_metric_t metric, nk_cross_tile_arguments_t const *arguments,
    nk_size_t first_row, nk_size_t first_column,
    nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k],
    nk_fui32_t norms[2][nk_cross_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step) {
        unsigned const tile_row = thread_row + nk_cross_grid_side_simt_k * row_step;
        nk_size_t const row = first_row + tile_row;
        if (row >= arguments->rows_end) continue;
        nk_size_t column_begin, column_end;
        nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin, &column_end);
        nk_fui32_t *output = (nk_fui32_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
        for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step) {
            unsigned const tile_column = thread_column + nk_cross_grid_side_simt_k * column_step;
            nk_size_t const column = first_column + tile_column;
            if (column < column_begin || column >= column_end) continue;
            nk_fui32_t const sum = sums[row_step][column_step];
            if (metric == nk_cross_metric_dot_k) output[column] = sum;
            else if (nk_cross_symmetric_simt_(band) && column == row) output[column].f = 0;
            else
                output[column].f = nk_cross_integer_metric_simt_(metric, norm, sum.u, norms[0][tile_row].u,
                                                                 norms[1][tile_column].u);
        }
    }
}

#pragma endregion Baseline Tile

/*  The block-scaled tiles follow the serial backends' rebased F32: every row and column rebases its
 *  scales to its largest exponent, relative block products stay normal F32, the epilogue applies
 *  both bases and the tensor factor once, and outputs whose spreads leave the window take exact
 *  wide sums. The scaling, splitting, wide-sum and metric helpers mirror `dots/serial.h`. */
#pragma region Block Scales

/** @p value times two to the power of @p exponent, for any exponent, rounding twice only when the
 *  result is subnormal. */
NUMKONG_DEVICE nk_f32_t nk_f32_scale_simt_(nk_f32_t value, nk_i32_t exponent) {
    exponent = exponent < -378 ? -378 : exponent > 381 ? 381 : exponent;
    for (; exponent > 127; exponent -= 127) value = __fmul_rn(value, __uint_as_float(254u << 23));
    for (; exponent < -126; exponent += 126) value = __fmul_rn(value, __uint_as_float(1u << 23));
    return __fmul_rn(value, __uint_as_float((nk_u32_t)(exponent + 127) << 23));
}

/** Splits @p value into a mantissa in [1, 2) and a power of two, subnormals included; zeros,
 *  infinities and NaNs return themselves with a zero exponent. */
NUMKONG_DEVICE nk_f32_t nk_f32_split_simt_(nk_f32_t value, nk_i32_t *exponent) {
    nk_u32_t bits = __float_as_uint(value);
    nk_u32_t const biased = (bits >> 23) & 0xFF;
    *exponent = 0;
    if (biased == 0xFF || (bits & 0x7FFFFFFFu) == 0) return value;
    nk_i32_t shift = 0;
    if (biased == 0) bits = __float_as_uint(__fmul_rn(value, 16777216.0f)), shift = 24;
    *exponent = (nk_i32_t)((bits >> 23) & 0xFF) - 127 - shift;
    return __uint_as_float((bits & 0x807FFFFFu) | 0x3F800000u);
}

/** The product of two F32 tensor scales as a mantissa product and a power of two. */
NUMKONG_DEVICE nk_cross_tensor_factor_t nk_cross_tensor_factor_simt_(nk_f32_t first, nk_f32_t second) {
    nk_cross_tensor_factor_t factor;
    nk_i32_t first_exponent, second_exponent;
    nk_f32_t const first_mantissa = nk_f32_split_simt_(first, &first_exponent);
    factor.mantissa = __fmul_rn(first_mantissa, nk_f32_split_simt_(second, &second_exponent));
    factor.exponent = first_exponent + second_exponent;
    return factor;
}

NUMKONG_DEVICE void nk_cross_wide_add_simt_(nk_cross_wide_sum_t *state, nk_f32_t value, nk_i32_t exponent) {
    if (value == 0) return;
    if (state->sum == 0 || exponent > state->exponent)
        state->sum = nk_f32_scale_simt_(state->sum, state->exponent - exponent), state->exponent = exponent;
    else value = nk_f32_scale_simt_(value, exponent - state->exponent);
    state->sum = __fadd_rn(state->sum, value);
}

/** @p value times a tensor @p factor, its mantissa into the sum and its power into the exponent. */
NUMKONG_DEVICE nk_cross_wide_sum_t nk_cross_wide_times_simt_(nk_cross_wide_sum_t value,
                                                             nk_cross_tensor_factor_t factor) {
    value.sum = __fmul_rn(value.sum, factor.mantissa), value.exponent += factor.exponent;
    return value;
}

/** One UE8M0 scale @p code as one at 2^(code − 127), zero for 0x00 and a NaN for 0xFF, both at
 *  exponent zero. */
NUMKONG_DEVICE nk_f32_t nk_ue8m0_split_simt_(unsigned code, nk_i32_t *exponent) {
    *exponent = code == 0 || code == 0xFF ? 0 : (nk_i32_t)code - 127;
    return __uint_as_float(code == 0 ? 0 : code == 0xFF ? 0x7FC00000u : 0x3F800000u);
}

/** One UE4M3 scale @p code as its value at exponent zero, a NaN for 0x7F. */
NUMKONG_DEVICE nk_f32_t nk_ue4m3_split_simt_(unsigned code, nk_i32_t *exponent) {
    unsigned char const magnitude = (unsigned char)(code & 0x7F);
    nk_f32_t value;
    nk_e4m3_to_f32_simt_(&magnitude, &value);
    *exponent = 0;
    return value;
}

/** The rebasing exponent of @p blocks UE8M0 scale codes: their largest exponent less the headroom,
 *  over the codes that are neither zero nor NaN, or zero for none; @p spread receives the spread
 *  of their exponents. */
NUMKONG_DEVICE nk_i32_t nk_cross_scaled_base_ue8m0_simt_(unsigned char const *scales, nk_size_t blocks,
                                                         nk_i32_t *spread) {
    nk_i32_t low = 0x7FFFFFFF, high = -0x7FFFFFFF;
    for (nk_size_t block = 0; block != blocks; ++block) {
        nk_i32_t exponent;
        nk_f32_t const mantissa = nk_ue8m0_split_simt_(scales[block], &exponent);
        if (mantissa == 0 || mantissa != mantissa) continue;
        low = exponent < low ? exponent : low, high = exponent > high ? exponent : high;
    }
    *spread = low > high ? 0 : high - low;
    return low > high ? 0 : high - nk_cross_scaled_headroom_k;
}

/** UE8M0 scale code @p code relative to a row's @p base, exact while the row's spread fits the
 *  window. */
NUMKONG_DEVICE nk_f32_t nk_cross_relative_scale_ue8m0_simt_(unsigned code, nk_i32_t base) {
    nk_i32_t exponent;
    nk_f32_t const mantissa = nk_ue8m0_split_simt_(code, &exponent);
    return nk_f32_scale_simt_(mantissa, exponent - base);
}

/** The rebasing exponent of @p blocks UE4M3 scale codes: their largest exponent less the headroom,
 *  over the codes that are neither zero nor NaN, or zero for none; @p spread receives the spread
 *  of their exponents. */
NUMKONG_DEVICE nk_i32_t nk_cross_scaled_base_ue4m3_simt_(unsigned char const *scales, nk_size_t blocks,
                                                         nk_i32_t *spread) {
    nk_i32_t low = 0x7FFFFFFF, high = -0x7FFFFFFF;
    for (nk_size_t block = 0; block != blocks; ++block) {
        nk_i32_t exponent;
        nk_f32_t const mantissa = nk_ue4m3_split_simt_(scales[block], &exponent);
        if (mantissa == 0 || mantissa != mantissa) continue;
        low = exponent < low ? exponent : low, high = exponent > high ? exponent : high;
    }
    *spread = low > high ? 0 : high - low;
    return low > high ? 0 : high - nk_cross_scaled_headroom_k;
}

/** UE4M3 scale code @p code relative to a row's @p base, exact while the row's spread fits the
 *  window. */
NUMKONG_DEVICE nk_f32_t nk_cross_relative_scale_ue4m3_simt_(unsigned code, nk_i32_t base) {
    nk_i32_t exponent;
    nk_f32_t const mantissa = nk_ue4m3_split_simt_(code, &exponent);
    return nk_f32_scale_simt_(mantissa, exponent - base);
}

/** Whether a row and a column spreading @p row_spread and @p column_spread binades leave the
 *  rebased window: their products past its width, and @p normalized norms past twice the larger. */
NUMKONG_DEVICE int nk_cross_scaled_exceeds_simt_(nk_i32_t row_spread, nk_i32_t column_spread, int normalized) {
    nk_i32_t const larger = row_spread > column_spread ? row_spread : column_spread;
    return (normalized ? 2 * larger : row_spread + column_spread) > nk_cross_scaled_spread_k;
}

/** The squared norm of a nvfp4 row of codes relative to its @p base: each block's squares summed in
 *  F32 times its relative scale squared. */
NUMKONG_DEVICE nk_f32_t nk_cross_scaled_norm_nvfp4_simt_(unsigned char const *row, unsigned char const *scales,
                                                         nk_size_t depth, nk_i32_t base) {
    nk_f32_t sum = 0;
    for (nk_size_t block = 0; block * 16 < depth; ++block) {
        nk_f32_t block_sum = 0;
        for (nk_size_t index = block * 16; index != (block + 1) * 16; ++index) {
            nk_f32_t const value = nk_e2m1_load_f32_simt_(row, index);
            block_sum = __fmaf_rn(value, value, block_sum);
        }
        nk_f32_t const scale = nk_cross_relative_scale_ue4m3_simt_(scales[block], base);
        sum = __fadd_rn(sum, __fmul_rn(block_sum, __fmul_rn(scale, scale)));
    }
    return sum;
}

/** The squared norm of a mxfp4 row of codes relative to its @p base: each block's squares summed in
 *  F32 times its relative scale squared. */
NUMKONG_DEVICE nk_f32_t nk_cross_scaled_norm_mxfp4_simt_(unsigned char const *row, unsigned char const *scales,
                                                         nk_size_t depth, nk_i32_t base) {
    nk_f32_t sum = 0;
    for (nk_size_t block = 0; block * 32 < depth; ++block) {
        nk_f32_t block_sum = 0;
        for (nk_size_t index = block * 32; index != (block + 1) * 32; ++index) {
            nk_f32_t const value = nk_e2m1_load_f32_simt_(row, index);
            block_sum = __fmaf_rn(value, value, block_sum);
        }
        nk_f32_t const scale = nk_cross_relative_scale_ue8m0_simt_(scales[block], base);
        sum = __fadd_rn(sum, __fmul_rn(block_sum, __fmul_rn(scale, scale)));
    }
    return sum;
}

/** The squared norm of a mxfp8e4m3 row of codes relative to its @p base: each block's squares
 *  summed in F32 times its relative scale squared. */
NUMKONG_DEVICE nk_f32_t nk_cross_scaled_norm_mxfp8e4m3_simt_(unsigned char const *row, unsigned char const *scales,
                                                             nk_size_t depth, nk_i32_t base) {
    nk_f32_t sum = 0;
    for (nk_size_t block = 0; block * 32 < depth; ++block) {
        nk_f32_t block_sum = 0;
        for (nk_size_t index = block * 32; index != (block + 1) * 32; ++index) {
            nk_f32_t const value = nk_e4m3_load_f32_simt_(row, index);
            block_sum = __fmaf_rn(value, value, block_sum);
        }
        nk_f32_t const scale = nk_cross_relative_scale_ue8m0_simt_(scales[block], base);
        sum = __fadd_rn(sum, __fmul_rn(block_sum, __fmul_rn(scale, scale)));
    }
    return sum;
}

/** The squared norm of a mxfp8e5m2 row of codes relative to its @p base: each block's squares
 *  summed in F32 times its relative scale squared. */
NUMKONG_DEVICE nk_f32_t nk_cross_scaled_norm_mxfp8e5m2_simt_(unsigned char const *row, unsigned char const *scales,
                                                             nk_size_t depth, nk_i32_t base) {
    nk_f32_t sum = 0;
    for (nk_size_t block = 0; block * 32 < depth; ++block) {
        nk_f32_t block_sum = 0;
        for (nk_size_t index = block * 32; index != (block + 1) * 32; ++index) {
            nk_f32_t const value = nk_e5m2_load_f32_simt_(row, index);
            block_sum = __fmaf_rn(value, value, block_sum);
        }
        nk_f32_t const scale = nk_cross_relative_scale_ue8m0_simt_(scales[block], base);
        sum = __fadd_rn(sum, __fmul_rn(block_sum, __fmul_rn(scale, scale)));
    }
    return sum;
}

/** A packed nvfp4 column's squared norm relative to four to the power of its base, at twice the
 *  base as the exponent and without the tensor scale; lane 0 sums it whole, as the tiles do, so
 *  both agree bit for bit. */
NUMKONG_DEVICE nk_cross_wide_sum_t nk_cross_scaled_pack_norm_nvfp4_simt_(unsigned char const *row, nk_size_t depth,
                                                                         unsigned lane, unsigned char const *scales,
                                                                         nk_f32_t) {
    nk_cross_wide_sum_t norm = {0, 0};
    if (lane != 0) return norm;
    nk_i32_t spread;
    nk_i32_t const base = nk_cross_scaled_base_ue4m3_simt_(scales, depth / 16, &spread);
    norm.sum = nk_cross_scaled_norm_nvfp4_simt_(row, scales, depth, base), norm.exponent = 2 * base;
    return norm;
}

/** A packed mxfp4 column's squared norm relative to four to the power of its base, at twice the
 *  base as the exponent and without the tensor scale; lane 0 sums it whole, as the tiles do, so
 *  both agree bit for bit. */
NUMKONG_DEVICE nk_cross_wide_sum_t nk_cross_scaled_pack_norm_mxfp4_simt_(unsigned char const *row, nk_size_t depth,
                                                                         unsigned lane, unsigned char const *scales,
                                                                         nk_f32_t) {
    nk_cross_wide_sum_t norm = {0, 0};
    if (lane != 0) return norm;
    nk_i32_t spread;
    nk_i32_t const base = nk_cross_scaled_base_ue8m0_simt_(scales, depth / 32, &spread);
    norm.sum = nk_cross_scaled_norm_mxfp4_simt_(row, scales, depth, base), norm.exponent = 2 * base;
    return norm;
}

/** A packed mxfp8e4m3 column's squared norm relative to four to the power of its base, at twice the
 *  base as the exponent and without the tensor scale; lane 0 sums it whole, as the tiles do, so
 *  both agree bit for bit. */
NUMKONG_DEVICE nk_cross_wide_sum_t nk_cross_scaled_pack_norm_mxfp8e4m3_simt_(unsigned char const *row, nk_size_t depth,
                                                                             unsigned lane, unsigned char const *scales,
                                                                             nk_f32_t) {
    nk_cross_wide_sum_t norm = {0, 0};
    if (lane != 0) return norm;
    nk_i32_t spread;
    nk_i32_t const base = nk_cross_scaled_base_ue8m0_simt_(scales, depth / 32, &spread);
    norm.sum = nk_cross_scaled_norm_mxfp8e4m3_simt_(row, scales, depth, base), norm.exponent = 2 * base;
    return norm;
}

/** A packed mxfp8e5m2 column's squared norm relative to four to the power of its base, at twice the
 *  base as the exponent and without the tensor scale; lane 0 sums it whole, as the tiles do, so
 *  both agree bit for bit. */
NUMKONG_DEVICE nk_cross_wide_sum_t nk_cross_scaled_pack_norm_mxfp8e5m2_simt_(unsigned char const *row, nk_size_t depth,
                                                                             unsigned lane, unsigned char const *scales,
                                                                             nk_f32_t) {
    nk_cross_wide_sum_t norm = {0, 0};
    if (lane != 0) return norm;
    nk_i32_t spread;
    nk_i32_t const base = nk_cross_scaled_base_ue8m0_simt_(scales, depth / 32, &spread);
    norm.sum = nk_cross_scaled_norm_mxfp8e5m2_simt_(row, scales, depth, base), norm.exponent = 2 * base;
    return norm;
}

/** Adds one block's exact sums and TwoSum errors of the dot and of both squared norms into the wide
 *  sums, at the exponents of the block's two scale mantissas. */
NUMKONG_DEVICE void nk_cross_wide_block_simt_(nk_cross_wide_sum_t *dot, nk_cross_wide_sum_t *a_squares,
                                              nk_cross_wide_sum_t *b_squares, nk_f32_t const sums[3],
                                              nk_f32_t const errors[3], nk_f32_t a_mantissa, nk_i32_t a_exponent,
                                              nk_f32_t b_mantissa, nk_i32_t b_exponent) {
    nk_cross_wide_add_simt_(dot, __fmul_rn(__fadd_rn(sums[0], errors[0]), __fmul_rn(a_mantissa, b_mantissa)),
                            a_exponent + b_exponent);
    nk_cross_wide_add_simt_(a_squares, __fmul_rn(__fadd_rn(sums[1], errors[1]), __fmul_rn(a_mantissa, a_mantissa)),
                            2 * a_exponent);
    nk_cross_wide_add_simt_(b_squares, __fmul_rn(__fadd_rn(sums[2], errors[2]), __fmul_rn(b_mantissa, b_mantissa)),
                            2 * b_exponent);
}

/** TwoSum of the three products of one element pair into the running sums and their errors, so
 *  cancelling blocks give zero. */
NUMKONG_DEVICE void nk_cross_two_sum_products_simt_(nk_f32_t sums[3], nk_f32_t errors[3], nk_f32_t a_value,
                                                    nk_f32_t b_value) {
    nk_f32_t const products[3] = {__fmul_rn(a_value, b_value), __fmul_rn(a_value, a_value),
                                  __fmul_rn(b_value, b_value)};
    for (unsigned term = 0; term != 3; ++term) {
        nk_f32_t const sum = __fadd_rn(sums[term], products[term]);
        nk_f32_t const split_sum = __fsub_rn(sum, sums[term]);
        errors[term] = __fadd_rn(errors[term], __fadd_rn(__fsub_rn(sums[term], __fsub_rn(sum, split_sum)),
                                                         __fsub_rn(products[term], split_sum)));
        sums[term] = sum;
    }
}

/** The dot of an A row and a B row of nvfp4 codes as an unrounded wide sum of exact block sums,
 *  summing both rows' squared norms the same way into @p a_sumsq and @p b_sumsq. */
NUMKONG_DEVICE nk_cross_wide_sum_t nk_cross_scaled_exact_wide_nvfp4_simt_(
    unsigned char const *a_row, unsigned char const *a_scales, unsigned char const *b_row,
    unsigned char const *b_scales, nk_size_t depth, nk_cross_wide_sum_t *a_sumsq, nk_cross_wide_sum_t *b_sumsq) {
    nk_cross_wide_sum_t dot = {0, 0}, a_squares = {0, 0}, b_squares = {0, 0};
    for (nk_size_t block = 0; block * 16 < depth; ++block) {
        nk_f32_t sums[3] = {0, 0, 0}, errors[3] = {0, 0, 0};
        for (nk_size_t index = block * 16; index != (block + 1) * 16; ++index)
            nk_cross_two_sum_products_simt_(sums, errors, nk_e2m1_load_f32_simt_(a_row, index),
                                            nk_e2m1_load_f32_simt_(b_row, index));
        nk_i32_t a_exponent, b_exponent;
        nk_f32_t const a_mantissa = nk_ue4m3_split_simt_(a_scales[block], &a_exponent);
        nk_f32_t const b_mantissa = nk_ue4m3_split_simt_(b_scales[block], &b_exponent);
        nk_cross_wide_block_simt_(&dot, &a_squares, &b_squares, sums, errors, a_mantissa, a_exponent, b_mantissa,
                                  b_exponent);
    }
    *a_sumsq = a_squares, *b_sumsq = b_squares;
    return dot;
}

/** The dot of an A row and a B row of mxfp4 codes as an unrounded wide sum of exact block sums,
 *  summing both rows' squared norms the same way into @p a_sumsq and @p b_sumsq. */
NUMKONG_DEVICE nk_cross_wide_sum_t nk_cross_scaled_exact_wide_mxfp4_simt_(
    unsigned char const *a_row, unsigned char const *a_scales, unsigned char const *b_row,
    unsigned char const *b_scales, nk_size_t depth, nk_cross_wide_sum_t *a_sumsq, nk_cross_wide_sum_t *b_sumsq) {
    nk_cross_wide_sum_t dot = {0, 0}, a_squares = {0, 0}, b_squares = {0, 0};
    for (nk_size_t block = 0; block * 32 < depth; ++block) {
        nk_f32_t sums[3] = {0, 0, 0}, errors[3] = {0, 0, 0};
        for (nk_size_t index = block * 32; index != (block + 1) * 32; ++index)
            nk_cross_two_sum_products_simt_(sums, errors, nk_e2m1_load_f32_simt_(a_row, index),
                                            nk_e2m1_load_f32_simt_(b_row, index));
        nk_i32_t a_exponent, b_exponent;
        nk_f32_t const a_mantissa = nk_ue8m0_split_simt_(a_scales[block], &a_exponent);
        nk_f32_t const b_mantissa = nk_ue8m0_split_simt_(b_scales[block], &b_exponent);
        nk_cross_wide_block_simt_(&dot, &a_squares, &b_squares, sums, errors, a_mantissa, a_exponent, b_mantissa,
                                  b_exponent);
    }
    *a_sumsq = a_squares, *b_sumsq = b_squares;
    return dot;
}

/** The dot of an A row and a B row of mxfp8e4m3 codes as an unrounded wide sum of exact block sums,
 *  summing both rows' squared norms the same way into @p a_sumsq and @p b_sumsq. */
NUMKONG_DEVICE nk_cross_wide_sum_t nk_cross_scaled_exact_wide_mxfp8e4m3_simt_(
    unsigned char const *a_row, unsigned char const *a_scales, unsigned char const *b_row,
    unsigned char const *b_scales, nk_size_t depth, nk_cross_wide_sum_t *a_sumsq, nk_cross_wide_sum_t *b_sumsq) {
    nk_cross_wide_sum_t dot = {0, 0}, a_squares = {0, 0}, b_squares = {0, 0};
    for (nk_size_t block = 0; block * 32 < depth; ++block) {
        nk_f32_t sums[3] = {0, 0, 0}, errors[3] = {0, 0, 0};
        for (nk_size_t index = block * 32; index != (block + 1) * 32; ++index)
            nk_cross_two_sum_products_simt_(sums, errors, nk_e4m3_load_f32_simt_(a_row, index),
                                            nk_e4m3_load_f32_simt_(b_row, index));
        nk_i32_t a_exponent, b_exponent;
        nk_f32_t const a_mantissa = nk_ue8m0_split_simt_(a_scales[block], &a_exponent);
        nk_f32_t const b_mantissa = nk_ue8m0_split_simt_(b_scales[block], &b_exponent);
        nk_cross_wide_block_simt_(&dot, &a_squares, &b_squares, sums, errors, a_mantissa, a_exponent, b_mantissa,
                                  b_exponent);
    }
    *a_sumsq = a_squares, *b_sumsq = b_squares;
    return dot;
}

/** The dot of an A row and a B row of mxfp8e5m2 codes as an unrounded wide sum of exact block sums,
 *  summing both rows' squared norms the same way into @p a_sumsq and @p b_sumsq. */
NUMKONG_DEVICE nk_cross_wide_sum_t nk_cross_scaled_exact_wide_mxfp8e5m2_simt_(
    unsigned char const *a_row, unsigned char const *a_scales, unsigned char const *b_row,
    unsigned char const *b_scales, nk_size_t depth, nk_cross_wide_sum_t *a_sumsq, nk_cross_wide_sum_t *b_sumsq) {
    nk_cross_wide_sum_t dot = {0, 0}, a_squares = {0, 0}, b_squares = {0, 0};
    for (nk_size_t block = 0; block * 32 < depth; ++block) {
        nk_f32_t sums[3] = {0, 0, 0}, errors[3] = {0, 0, 0};
        for (nk_size_t index = block * 32; index != (block + 1) * 32; ++index)
            nk_cross_two_sum_products_simt_(sums, errors, nk_e5m2_load_f32_simt_(a_row, index),
                                            nk_e5m2_load_f32_simt_(b_row, index));
        nk_i32_t a_exponent, b_exponent;
        nk_f32_t const a_mantissa = nk_ue8m0_split_simt_(a_scales[block], &a_exponent);
        nk_f32_t const b_mantissa = nk_ue8m0_split_simt_(b_scales[block], &b_exponent);
        nk_cross_wide_block_simt_(&dot, &a_squares, &b_squares, sums, errors, a_mantissa, a_exponent, b_mantissa,
                                  b_exponent);
    }
    *a_sumsq = a_squares, *b_sumsq = b_squares;
    return dot;
}

/** The angular distance of wide sums: 0 for two zero vectors, else 1 for a zero dot, else
 *  max(0, 1 − cosine), the norms' product halved through an even exponent; NaNs propagate. */
NUMKONG_DEVICE nk_f32_t nk_angular_f32_from_wide_simt_(nk_cross_wide_sum_t dot, nk_cross_wide_sum_t a_sumsq,
                                                       nk_cross_wide_sum_t b_sumsq) {
    if (a_sumsq.sum == 0 && b_sumsq.sum == 0) return 0;
    if (dot.sum == 0) return 1;
    nk_i32_t dot_exponent, a_exponent, b_exponent;
    nk_f32_t const dot_mantissa = nk_f32_split_simt_(dot.sum, &dot_exponent);
    nk_f32_t const a_mantissa = nk_f32_split_simt_(a_sumsq.sum, &a_exponent);
    nk_f32_t norms_mantissa = __fmul_rn(a_mantissa, nk_f32_split_simt_(b_sumsq.sum, &b_exponent));
    nk_i32_t norms_exponent = a_exponent + a_sumsq.exponent + b_exponent + b_sumsq.exponent;
    if (norms_exponent & 1) norms_mantissa = __fmul_rn(norms_mantissa, 2.0f), norms_exponent -= 1;
    nk_f32_t const cosine = nk_f32_scale_simt_(__fmul_rn(dot_mantissa, rsqrtf(norms_mantissa)),
                                               dot_exponent + dot.exponent - norms_exponent / 2);
    nk_f32_t const angular = 1 - cosine;
    return angular < 0 ? 0 : angular;
}

/** The euclidean distance of wide sums: ‖a‖² + ‖b‖² − 2 · dot as a wide sum of split terms,
 *  clamped at zero, its root taken through an even exponent; NaNs propagate. */
NUMKONG_DEVICE nk_f32_t nk_euclidean_f32_from_wide_simt_(nk_cross_wide_sum_t dot, nk_cross_wide_sum_t a_sumsq,
                                                         nk_cross_wide_sum_t b_sumsq) {
    nk_cross_wide_sum_t squares = {0, 0};
    nk_i32_t exponent;
    nk_f32_t mantissa = nk_f32_split_simt_(a_sumsq.sum, &exponent);
    nk_cross_wide_add_simt_(&squares, mantissa, exponent + a_sumsq.exponent);
    mantissa = nk_f32_split_simt_(b_sumsq.sum, &exponent);
    nk_cross_wide_add_simt_(&squares, mantissa, exponent + b_sumsq.exponent);
    mantissa = nk_f32_split_simt_(dot.sum, &exponent);
    nk_cross_wide_add_simt_(&squares, -mantissa, exponent + dot.exponent + 1);
    if (squares.sum != squares.sum) return squares.sum;
    if (squares.sum <= 0) return 0;
    if (squares.exponent & 1) squares.sum = __fmul_rn(squares.sum, 2.0f), squares.exponent -= 1;
    return nk_f32_scale_simt_(sqrtf(squares.sum), squares.exponent / 2);
}

/** The @p metric of one pair from its wide dot and wide squared norms, tensor factors applied,
 *  rounding once. */
NUMKONG_DEVICE nk_f32_t nk_cross_scaled_metric_simt_(nk_cross_metric_t metric, nk_cross_wide_sum_t dot,
                                                     nk_cross_wide_sum_t a_sumsq, nk_cross_wide_sum_t b_sumsq) {
    switch (metric) {
    case nk_cross_metric_dot_k: return nk_f32_scale_simt_(dot.sum, dot.exponent);
    case nk_cross_metric_angular_k: return nk_angular_f32_from_wide_simt_(dot, a_sumsq, b_sumsq);
    default: return nk_euclidean_f32_from_wide_simt_(dot, a_sumsq, b_sumsq);
    }
}

#pragma endregion Block Scales

/*  Every tile, baseline or tensor, takes the band, the metric and the launch arguments, so one
 *  generator per shape serves them all. */
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
                 column_count * (row_bytes + nk_cross_scales_stride_serial_(nk_##input_type_name##_k, depth) +       \
                                 sizeof(nk_##norm_value_type##_t));                                                  \
        return nk_success_k;                                                                                         \
    }

#pragma endregion Cross Macros

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_DOTS_SIMT_CUH
