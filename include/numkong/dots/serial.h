/**
 *  @file include/numkong/dots/serial.h
 *  @author Ash Vardanian
 *  @date September 14, 2024
 *  @brief SWAR-accelerated Batched Dot Products for SIMD-free CPUs.
 *
 *  @sa include/numkong/dots.h for API overview and use cases
 *
 *  This file provides two macro families for generating GEMM kernels:
 *
 *  - nk_define_cross_packed_: vectorized inner-products between rows of A and Bᵀ
 *  - nk_define_cross_symmetric_: vectorized inner-products between rows and columns of A
 *
 *  Both use the same B packing format (see below), enabling pack-once-use-anywhere.
 *
 *  @section dots_serial_packing B Matrix Packing Format
 *
 *  Computing C = A × Bᵀ where:
 *
 *  - A[row_count, depth] row-major: A[i, k] at address A + i × lda + k
 *  - B[column_count, depth] row-major (pre-transposed): B[j, k] at address B + j × ldb + k
 *  - C[row_count, column_count] row-major: C[i, j] at address C + i × ldc + j
 *
 *  The API convention stores B as Bᵀ for efficient SIMD access:
 *
 *  - A[i, k:k+4] is contiguous in row-major (good)
 *  - B[j, k:k+4] is contiguous in row-major (good - already transposed)
 *
 *  Packing adds row grouping (group_size = 16) for:
 *
 *  - Zero-padding on edges (avoids boundary checks in inner loop)
 *  - Cache-friendly blocking in outer loops
 *
 *  Memory layout example - B[8, 8] with 8 output columns (j), 8 depth (k):
 *
 *  @verbatim
 *          k=0   k=1   k=2   k=3   k=4   k=5   k=6   k=7
 *       ┌─────────────────────────────────────────────────┐
 *  j=0  │  a0    a1    a2    a3    a4    a5    a6    a7   │
 *  j=1  │  b0    b1    b2    b3    b4    b5    b6    b7   │
 *  j=2  │  c0    c1    c2    c3    c4    c5    c6    c7   │
 *  j=3  │  d0    d1    d2    d3    d4    d5    d6    d7   │
 *  j=4  │  e0    e1    e2    e3    e4    e5    e6    e7   │
 *  j=5  │  f0    f1    f2    f3    f4    f5    f6    f7   │
 *  j=6  │  g0    g1    g2    g3    g4    g5    g6    g7   │
 *  j=7  │  h0    h1    h2    h3    h4    h5    h6    h7   │
 *       └─────────────────────────────────────────────────┘
 *  @endverbatim
 *
 *  Packed as B_packed[column_count_padded, depth] (grouped for alignment):
 *
 *  @verbatim
 *  Group 0 (j=0..7, padded to 16):
 *    ┌───────────────────────────────────┐
 *    │ a0 a1 a2 a3 a4 a5 a6 a7 │  j=0    │  ← row 0 copied as-is
 *    │ b0 b1 b2 b3 b4 b5 b6 b7 │  j=1    │
 *    │ c0 c1 c2 c3 c4 c5 c6 c7 │  j=2    │
 *    │ d0 d1 d2 d3 d4 d5 d6 d7 │  j=3    │
 *    │ e0 e1 e2 e3 e4 e5 e6 e7 │  j=4    │
 *    │ f0 f1 f2 f3 f4 f5 f6 f7 │  j=5    │
 *    │ g0 g1 g2 g3 g4 g5 g6 g7 │  j=6    │
 *    │ h0 h1 h2 h3 h4 h5 h6 h7 │  j=7    │
 *    │ 00 00 00 00 00 00 00 00 │ padding │
 *    │ ...                     │ ...     │
 *    └───────────────────────────────────┘
 *  @endverbatim
 *
 *  Addressing formula for B_packed[j, k]:
 *
 *  @verbatim
 *  group = j / group_size
 *  j_in_group = j % group_size
 *  B_packed[j, k] = packed[group * group_size * depth + j_in_group * depth + k]
 *  @endverbatim
 *
 *  The inner loop accesses B_packed[j, k:k+simd], which is contiguous, at just `pointer + k`.
 */

#ifndef NUMKONG_DOTS_SERIAL_H
#define NUMKONG_DOTS_SERIAL_H

#include "numkong/types.h"
#include "numkong/capabilities.h"   // `nk_capability_t`
#include "numkong/cast/serial.h"    // `nk_partial_load_b32x4_serial_`
#include "numkong/dot/serial.h"     // `nk_dot_f32x4_state_serial_t`
#include "numkong/spatial/serial.h" // `nk_f32_sqrt_`
#include "numkong/reduce.h"         // `nk_reduce_moments_f32_best`, so packs norm with their own capability

/*  GCC's -Wstringop-overflow produces false positives on the padded accumulator arrays in
 *  nk_define_cross_symmetric_ macro expansions — accumulators[4][7] with runtime indexing. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/*  Packed buffer header, 64-byte aligned. Used by all packed matmul backends — serial, NEON,
 *  AVX-512, SVE.
 *
 *  Important units clarification:
 *  - For types where dimensions_per_value = 1 (f32, i8, u8, etc.): dimensions == values
 *  - For sub-byte types (i4x2, u4x2): dimensions ≠ values
 *    - dimensions = individual 4-bit nibbles (e.g., 128 nibbles)
 *    - values = storage bytes containing nibbles (e.g., 64 bytes for 128 nibbles)
 *    - dimensions_per_value = 2 (2 nibbles per byte) */
typedef struct {

    /** Columns, not padded. */
    nk_u32_t column_count;

    /** Logical depth in dimensions: nibbles for i4 and u4, values for i8 and f32. */
    nk_u32_t depth_dimensions;

    /** Padded depth in storage values: bytes for i4 and u4, values for i8 and f32. */
    nk_u32_t depth_padded_values;

    /** Bytes between the scale rows that follow the packed rows of a block-scaled dtype, else 0. */
    nk_u32_t scales_stride;

    /** The packed operand's tensor scale, 1 when it has none, which every product multiplies in. */
    nk_f32_t tensor_scale;

    /** Byte offset from the buffer start to the per-column norms, as in SME packs. */
    nk_u32_t norms_offset;

    /** Zeroed; pads the header to 64 bytes. */
    nk_u32_t reserved[8];

    /** The capability that packed the buffer, which every consumer checks. */
    nk_capability_t capability;
} nk_cross_packed_buffer_header_t;

/** Values one packed row spans: @p depth rounded up to @p depth_simd_dimensions, in values of
 *  @p dimensions_per_value, plus one more step when that lands on a power-of-two byte stride, as
 *  @c nk_define_cross_pack_size_ pads. The Metal packs size their rows with it, while the CUDA and
 *  ROCm packs skip the extra step through @c nk_cross_padded_values_simt_. */
NUMKONG_INLINE nk_size_t nk_cross_padded_values_(nk_size_t depth, nk_size_t depth_simd_dimensions,
                                                 nk_size_t dimensions_per_value, nk_size_t value_bytes) {
    nk_size_t values = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions) / dimensions_per_value;
    nk_size_t const stride = values * value_bytes;
    if ((stride & (stride - 1)) == 0 && stride > 0) values += depth_simd_dimensions / dimensions_per_value;
    return values;
}

/** Blocks of scales in a row of @p depth elements of @p dtype, zero for plain dtypes. */
NUMKONG_CONSTEXPR nk_size_t nk_cross_scale_blocks_(nk_dtype_t dtype, nk_size_t depth) {
    nk_size_t const block_size = nk_block_scaled_format_of_dtype(dtype).block_size;
    return block_size ? depth / block_size : 0;
}

/** Whether @p depth spans whole blocks of @p dtype, as every block-scaled kernel requires; true for
 *  plain dtypes. */
NUMKONG_CONSTEXPR int nk_cross_whole_blocks_(nk_dtype_t dtype, nk_size_t depth) {
    nk_size_t const block_size = nk_block_scaled_format_of_dtype(dtype).block_size;
    return block_size == 0 || depth % block_size == 0;
}

/** Bytes between scale rows of @p depth elements of @p dtype, one byte per block rounded up to 16
 *  so every row starts where a tensor-memory copy can load it; zero for plain dtypes. */
NUMKONG_CONSTEXPR nk_size_t nk_cross_scales_stride_(nk_dtype_t dtype, nk_size_t depth) {
    nk_size_t const blocks = nk_cross_scale_blocks_(dtype, depth);
    return blocks ? nk_size_round_up_to_multiple_(blocks, 16) : 0;
}

/*  The operand each generated entry point takes, by the input type name it is generated for: an
 *  element pointer for plain dtypes, and a reference to codes and scales for block-scaled ones. */
typedef nk_f64_t nk_cross_f64_operand_t;
typedef nk_f32_t nk_cross_f32_operand_t;
typedef nk_bf16_t nk_cross_bf16_operand_t;
typedef nk_f16_t nk_cross_f16_operand_t;
typedef nk_e4m3_t nk_cross_e4m3_operand_t;
typedef nk_e5m2_t nk_cross_e5m2_operand_t;
typedef nk_e2m3_t nk_cross_e2m3_operand_t;
typedef nk_e3m2_t nk_cross_e3m2_operand_t;
typedef nk_e2m1x2_t nk_cross_e2m1_operand_t;

typedef nk_mxfp8e5m2_cref_t nk_cross_mxfp8e5m2_operand_t;
typedef nk_mxfp8e4m3_cref_t nk_cross_mxfp8e4m3_operand_t;
typedef nk_mxfp6e3m2_cref_t nk_cross_mxfp6e3m2_operand_t;
typedef nk_mxfp6e2m3_cref_t nk_cross_mxfp6e2m3_operand_t;
typedef nk_mxfp4_cref_t nk_cross_mxfp4_operand_t;
typedef nk_nvfp4_cref_t nk_cross_nvfp4_operand_t;

typedef nk_i8_t nk_cross_i8_operand_t;
typedef nk_i4x2_t nk_cross_i4_operand_t;

typedef nk_u8_t nk_cross_u8_operand_t;
typedef nk_u4x2_t nk_cross_u4_operand_t;
typedef nk_u1x8_t nk_cross_u1_operand_t;

/** One operand unpacked: its codes, and for block-scaled dtypes its scales, their row stride and
 *  its tensor scale, which stays a pointer so GPU entry points never read device memory. */
typedef struct {
    void const *elements;
    nk_u8_t const *scales;
    nk_size_t scales_stride;
    nk_f32_t const *tensor_scale;
} nk_cross_operand_t;

/** Unpacks @p operand of @p dtype, whose rows of codes are @p stride bytes apart, folding to a
 *  plain pointer copy for plain dtypes. Every MX reference shares @c nk_mxfp4_cref_t's layout. */
NUMKONG_INLINE nk_cross_operand_t nk_cross_operand_(nk_dtype_t dtype, void const *operand, nk_size_t stride) {
    nk_block_scaled_format_t const format = nk_block_scaled_format_of_dtype(dtype);
    nk_cross_operand_t unpacked = {operand, NUMKONG_NULL, 0, NUMKONG_NULL};
    if (dtype == nk_nvfp4_k) {
        nk_nvfp4_cref_t const *reference = (nk_nvfp4_cref_t const *)operand;
        unpacked.elements = reference->elements, unpacked.scales = reference->scales;
        unpacked.tensor_scale = reference->tensor_scale;
    }
    else if (format.block_size) {
        nk_mxfp4_cref_t reference;
        nk_copy_bytes_(&reference, operand, sizeof(reference));
        unpacked.elements = reference.elements, unpacked.scales = reference.scales;
    }
    nk_assert_(!format.block_size || stride % format.block_bytes == 0);
    if (format.block_size) unpacked.scales_stride = stride / format.block_bytes;
    return unpacked;
}

/** The value behind @p tensor_scale, 1 when it is null. */
NUMKONG_INLINE nk_f32_t nk_cross_tensor_scale_(nk_f32_t const *tensor_scale) NUMKONG_STREAMABLE_ {
    return tensor_scale ? *tensor_scale : 1;
}

/** Multiplies @p count F32 results by @p factor, skipped when it is 1, as for plain dtypes. */
NUMKONG_INLINE void nk_cross_scale_results_(nk_f32_t *results, nk_size_t count, nk_f32_t factor) {
    if (factor != 1)
        for (nk_size_t index = 0; index != count; ++index) results[index] *= factor;
}

/** Rebased MX scales lie in 2^(31 − spread) … 2³¹; a row and a column may span 156 binades
 *  together while every block product, E5M2's 2⁻³² granules included, stays a normal F32. */
enum { nk_cross_scaled_headroom_k = 31, nk_cross_scaled_spread_k = 156 };

/** The rebasing exponent of @p blocks UE8M0 scales: their largest finite exponent less the
 *  headroom; @p spread receives their spread. Zero for rows of only 0 and 0xFF codes; NVFP4 rows
 *  never call it and rebase by zero. */
NUMKONG_INLINE nk_i32_t nk_cross_scaled_base_(nk_u8_t const *scales, nk_size_t blocks,
                                              nk_i32_t *spread) NUMKONG_STREAMABLE_ {
    nk_i32_t low = 255, high = 0;
    *spread = 0;
    for (nk_size_t block = 0; block != blocks; ++block) {
        nk_i32_t const code = scales[block];
        if (code == 0 || code == 255) continue;
        low = code < low ? code : low, high = code > high ? code : high;
    }
    if (low > high) return 0;
    *spread = high - low;
    return high - 127 - nk_cross_scaled_headroom_k;
}

/** Whether rows and columns spreading @p row_spread and @p column_spread binades leave the rebased
 *  window: their products past the sum, and @p normalized norms past twice the larger, as a norm
 *  squares one side's scales. */
NUMKONG_INLINE int nk_cross_scaled_exceeds_(nk_i32_t row_spread, nk_i32_t column_spread, int normalized) {
    nk_i32_t const larger = row_spread > column_spread ? row_spread : column_spread;
    return (normalized ? 2 * larger : row_spread + column_spread) > nk_cross_scaled_spread_k;
}

/** One UE8M0 scale relative to @p base: 2^(code − 127 − base), 0x00 to zero, 0xFF to a NaN. Codes
 *  more than 157 binades under the base decode garbage, which the exact overwrite replaces. */
NUMKONG_INLINE nk_f32_t nk_ue8m0_to_f32_relative_(nk_u8_t code, nk_i32_t base) NUMKONG_STREAMABLE_ {
    nk_fui32_t bits;
    bits.u = code == 0 ? 0 : code == 0xFF ? 0x7FC00000u : (nk_u32_t)((nk_i32_t)code - base) << 23;
    return bits.f;
}

/** @p value times two to the power of @p exponent, for any exponent, rounding twice only when the
 *  result is subnormal. */
NUMKONG_INLINE nk_f32_t nk_f32_scale_(nk_f32_t value, nk_i32_t exponent) NUMKONG_STREAMABLE_ {
    nk_fui32_t step;
    exponent = exponent < -378 ? -378 : exponent > 381 ? 381 : exponent;
    for (; exponent > 127; exponent -= 127) step.u = 254u << 23, value *= step.f;
    for (; exponent < -126; exponent += 126) step.u = 1u << 23, value *= step.f;
    step.u = (nk_u32_t)(exponent + 127) << 23;
    return value * step.f;
}

/** Splits @p value into a mantissa in [1, 2) and a power of two, subnormals included; zeros,
 *  infinities and NaNs return themselves with a zero exponent. */
NUMKONG_INLINE nk_f32_t nk_f32_split_(nk_f32_t value, nk_i32_t *exponent) NUMKONG_STREAMABLE_ {
    nk_fui32_t bits;
    bits.f = value;
    nk_u32_t const biased = (bits.u >> 23) & 0xFF;
    *exponent = 0;
    if (biased == 0xFF || (bits.u & 0x7FFFFFFFu) == 0) return value;
    nk_i32_t shift = 0;
    if (biased == 0) bits.f = value * 16777216.0f, shift = 24;
    *exponent = (nk_i32_t)((bits.u >> 23) & 0xFF) - 127 - shift;
    bits.u = (bits.u & 0x807FFFFFu) | 0x3F800000u;
    return bits.f;
}

/** The product of two F32 tensor scales as a mantissa product and a power of two, so tiny or huge
 *  scales never round through a subnormal or overflow before the result does. */
typedef struct nk_cross_tensor_factor_t {
    nk_f32_t mantissa;
    nk_i32_t exponent;
} nk_cross_tensor_factor_t;

NUMKONG_INLINE nk_cross_tensor_factor_t nk_cross_tensor_factor_(nk_f32_t first, nk_f32_t second) NUMKONG_STREAMABLE_ {
    nk_cross_tensor_factor_t factor;
    nk_i32_t first_exponent, second_exponent;
    factor.mantissa = nk_f32_split_(first, &first_exponent) * nk_f32_split_(second, &second_exponent);
    factor.exponent = first_exponent + second_exponent;
    return factor;
}

/** An F32 sum scaled by 2^exponent: exact block terms of any scales add without overflow or
 *  underflow, shedding only bits 2⁻¹⁴⁹ below the largest term, and round once at the end. */
typedef struct nk_cross_wide_sum_t {
    nk_f32_t sum;
    nk_i32_t exponent;
} nk_cross_wide_sum_t;

NUMKONG_INLINE void nk_cross_wide_add_(nk_cross_wide_sum_t *state, nk_f32_t value,
                                       nk_i32_t exponent) NUMKONG_STREAMABLE_ {
    if (value == 0) return;
    if (state->sum == 0 || exponent > state->exponent)
        state->sum = nk_f32_scale_(state->sum, state->exponent - exponent), state->exponent = exponent;
    else value = nk_f32_scale_(value, exponent - state->exponent);
    state->sum += value;
}

/** @p value times a tensor @p factor, its mantissa into the sum and its power into the exponent. */
NUMKONG_INLINE nk_cross_wide_sum_t nk_cross_wide_times_(nk_cross_wide_sum_t value,
                                                        nk_cross_tensor_factor_t factor) NUMKONG_STREAMABLE_ {
    value.sum *= factor.mantissa, value.exponent += factor.exponent;
    return value;
}

/*  Metrics of one pair from its wide dot and wide squared norms, tensor scales applied, rounding
 *  once: the exact paths of block-scaled kernels finish through them. */

/** The dot product, rounded once; the norms go unused. */
NUMKONG_INLINE nk_f32_t nk_dot_f32_from_wide_(nk_cross_wide_sum_t dot, nk_cross_wide_sum_t a_sumsq,
                                              nk_cross_wide_sum_t b_sumsq) {
    nk_unused_(a_sumsq), nk_unused_(b_sumsq);
    return nk_f32_scale_(dot.sum, dot.exponent);
}

/** The angular distance: 0 for two zero vectors, else 1 for a zero dot, else max(0, 1 − cosine),
 *  the norms' product halved through an even exponent; NaNs propagate. */
NUMKONG_INLINE nk_f32_t nk_angular_f32_from_wide_(nk_cross_wide_sum_t dot, nk_cross_wide_sum_t a_sumsq,
                                                  nk_cross_wide_sum_t b_sumsq) {
    if (a_sumsq.sum == 0 && b_sumsq.sum == 0) return 0;
    if (dot.sum == 0) return 1;
    nk_i32_t dot_exponent, a_exponent, b_exponent;
    nk_f32_t const dot_mantissa = nk_f32_split_(dot.sum, &dot_exponent);
    nk_f32_t norms_mantissa = nk_f32_split_(a_sumsq.sum, &a_exponent) * nk_f32_split_(b_sumsq.sum, &b_exponent);
    nk_i32_t norms_exponent = a_exponent + a_sumsq.exponent + b_exponent + b_sumsq.exponent;
    if (norms_exponent & 1) norms_mantissa *= 2, norms_exponent -= 1;
    nk_f32_t const cosine = nk_f32_scale_(dot_mantissa * nk_f32_rsqrt_(norms_mantissa),
                                          dot_exponent + dot.exponent - norms_exponent / 2);
    nk_f32_t const angular = 1 - cosine;
    return angular < 0 ? 0 : angular;
}

/** The euclidean distance: ‖a‖² + ‖b‖² − 2 · dot as a wide sum of split terms,
 *  clamped at zero, its root taken through an even exponent; NaNs propagate. */
NUMKONG_INLINE nk_f32_t nk_euclidean_f32_from_wide_(nk_cross_wide_sum_t dot, nk_cross_wide_sum_t a_sumsq,
                                                    nk_cross_wide_sum_t b_sumsq) {
    nk_cross_wide_sum_t squares = {0, 0};
    nk_i32_t exponent;
    nk_f32_t mantissa = nk_f32_split_(a_sumsq.sum, &exponent);
    nk_cross_wide_add_(&squares, mantissa, exponent + a_sumsq.exponent);
    mantissa = nk_f32_split_(b_sumsq.sum, &exponent);
    nk_cross_wide_add_(&squares, mantissa, exponent + b_sumsq.exponent);
    mantissa = nk_f32_split_(dot.sum, &exponent);
    nk_cross_wide_add_(&squares, -mantissa, exponent + dot.exponent + 1);
    if (squares.sum != squares.sum) return squares.sum;
    if (squares.sum <= 0) return 0;
    if (squares.exponent & 1) squares.sum *= 2, squares.exponent -= 1;
    return nk_f32_scale_(nk_f32_sqrt_(squares.sum), squares.exponent / 2);
}

/** The scale row of packed column @p column, among the scale rows between a pack's packed rows and
 *  its norms. */
NUMKONG_INLINE nk_u8_t const *nk_cross_packed_scales_(void const *b_packed, nk_size_t packed_value_bytes,
                                                      nk_size_t column) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    return (nk_u8_t const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
           (nk_size_t)header->column_count * header->depth_padded_values * packed_value_bytes +
           column * header->scales_stride;
}

/** Copies @p blocks raw scale bytes into a packed row of @p stride bytes, zeroing the rest; raw
 *  packs keep codes, so they ignore the row's rebasing @p base. */
NUMKONG_INLINE void nk_cross_pack_scales_bytes_(nk_u8_t const *source, nk_size_t blocks, nk_u8_t *destination,
                                                nk_size_t stride, nk_i32_t base) {
    nk_unused_(base);
    for (nk_size_t byte = 0; byte != stride; ++byte) destination[byte] = byte < blocks ? source[byte] : 0;
}

/** Writes @p blocks UE4M3 scales as F32 times the element @p lift, @p copies times each, then the
 *  raw codes the exact path reads. A scale row of @p stride bytes holds 4 · copies + 1 bytes per
 *  block, and the rest stays zero. */
NUMKONG_INLINE void nk_cross_pack_scales_ue4m3_f32_(nk_u8_t const *source, nk_size_t blocks, nk_u8_t *destination,
                                                    nk_size_t stride, nk_f32_t lift, nk_size_t copies) {
    nk_size_t const padded_blocks = stride / (4 * copies + 1);
    nk_f32_t *scales = (nk_f32_t *)destination;
    nk_u8_t *codes = destination + padded_blocks * copies * sizeof(nk_f32_t);
    for (nk_size_t block = 0; block != padded_blocks; ++block) {
        nk_u8_t const code = block < blocks ? source[block] : 0;
        nk_f32_t scale;
        nk_ue4m3_to_f32_(&code, &scale);
        for (nk_size_t copy = 0; copy != copies; ++copy) scales[block * copies + copy] = lift * scale;
        codes[block] = code;
    }
}

/** Writes @p blocks UE8M0 scales as F32 relative to @p base, laid out as
 *  @c nk_cross_pack_scales_ue4m3_f32_ lays out UE4M3 ones. */
NUMKONG_INLINE void nk_cross_pack_scales_ue8m0_f32_(nk_u8_t const *source, nk_size_t blocks, nk_u8_t *destination,
                                                    nk_size_t stride, nk_i32_t base, nk_f32_t lift, nk_size_t copies) {
    nk_size_t const padded_blocks = stride / (4 * copies + 1);
    nk_f32_t *scales = (nk_f32_t *)destination;
    nk_u8_t *codes = destination + padded_blocks * copies * sizeof(nk_f32_t);
    for (nk_size_t block = 0; block != padded_blocks; ++block) {
        nk_u8_t const code = block < blocks ? source[block] : 0;
        nk_f32_t const scale = lift * nk_ue8m0_to_f32_relative_(code, base);
        for (nk_size_t copy = 0; copy != copies; ++copy) scales[block * copies + copy] = scale;
        codes[block] = code;
    }
}

/**
 *  @brief Generates the exact path of one block-scaled dtype over one layout of B.
 *
 *  Emits @c nk_cross_scaled_exact_wide_<type>_<isa>_, the dot of an A row and a B row as an
 *  unrounded wide sum of exact block sums, summing both rows' squared norms the same way into
 *  @p a_sumsq and @p b_sumsq, so a row against itself gives three equal sums; and
 *  @c nk_cross_scaled_exact_dot_<type>_<isa>_, that dot times a tensor factor, rounded once.
 *
 *  @param[in] a_load_fn nk_f32_t fn(nk_u8_t const *row, nk_size_t index), element @p index of an A
 *      row of raw codes, exactly.
 *  @param[in] b_load_fn The same over a B row in its own layout, raw or packed.
 *  @param[in] scale_split_fn nk_f32_t fn(nk_u8_t code, nk_i32_t *exponent), a scale code as a
 *      mantissa and a power of two.
 *  @param[in] block_size Elements per scale.
 */
#define nk_define_cross_scaled_exact_(input_type_name, isa_suffix, a_load_fn, b_load_fn, scale_split_fn, block_size)   \
    NUMKONG_INLINE nk_cross_wide_sum_t nk_cross_scaled_exact_wide_##input_type_name##_##isa_suffix##_(                 \
        nk_u8_t const *a_row, nk_u8_t const *a_scales, nk_u8_t const *b_row, nk_u8_t const *b_scales, nk_size_t depth, \
        nk_cross_wide_sum_t *a_sumsq, nk_cross_wide_sum_t *b_sumsq) NUMKONG_STREAMABLE_ {                              \
        nk_cross_wide_sum_t dot = {0, 0}, a_squares = {0, 0}, b_squares = {0, 0};                                      \
        for (nk_size_t block = 0; block * (block_size) < depth; ++block) {                                             \
            /* TwoSum keeps what each sum of exact products sheds, so cancelling blocks give zero */                   \
            nk_f32_t sums[3] = {0, 0, 0}, errors[3] = {0, 0, 0};                                                       \
            for (nk_size_t index = block * (block_size); index != (block + 1) * (block_size); ++index) {               \
                nk_f32_t const a_value = a_load_fn(a_row, index), b_value = b_load_fn(b_row, index);                   \
                nk_f32_t const products[3] = {a_value * b_value, a_value * a_value, b_value * b_value};                \
                for (nk_size_t term = 0; term != 3; ++term) {                                                          \
                    nk_f32_t const sum = sums[term] + products[term], split = sum - sums[term];                        \
                    errors[term] += (sums[term] - (sum - split)) + (products[term] - split);                           \
                    sums[term] = sum;                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            nk_i32_t a_exponent, b_exponent;                                                                           \
            nk_f32_t const a_mantissa = scale_split_fn(a_scales[block], &a_exponent);                                  \
            nk_f32_t const b_mantissa = scale_split_fn(b_scales[block], &b_exponent);                                  \
            nk_cross_wide_add_(&dot, (sums[0] + errors[0]) * (a_mantissa * b_mantissa), a_exponent + b_exponent);      \
            nk_cross_wide_add_(&a_squares, (sums[1] + errors[1]) * (a_mantissa * a_mantissa), 2 * a_exponent);         \
            nk_cross_wide_add_(&b_squares, (sums[2] + errors[2]) * (b_mantissa * b_mantissa), 2 * b_exponent);         \
        }                                                                                                              \
        *a_sumsq = a_squares, *b_sumsq = b_squares;                                                                    \
        return dot;                                                                                                    \
    }                                                                                                                  \
    NUMKONG_INLINE nk_f32_t nk_cross_scaled_exact_dot_##input_type_name##_##isa_suffix##_(                             \
        nk_u8_t const *a_row, nk_u8_t const *a_scales, nk_u8_t const *b_row, nk_u8_t const *b_scales, nk_size_t depth, \
        nk_cross_tensor_factor_t factor) NUMKONG_STREAMABLE_ {                                                         \
        nk_cross_wide_sum_t a_sumsq, b_sumsq;                                                                          \
        nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_##input_type_name##_##isa_suffix##_(                \
            a_row, a_scales, b_row, b_scales, depth, &a_sumsq, &b_sumsq);                                              \
        return nk_f32_scale_(dot.sum * factor.mantissa, dot.exponent + factor.exponent);                               \
    }

/*  The exact paths over raw codes on both sides, which unpacked operands and the serial and Skylake
 *  packs share. */
nk_define_cross_scaled_exact_(nvfp4, serial, nk_e2m1_load_f32_serial_, nk_e2m1_load_f32_serial_, nk_ue4m3_split_serial_,
                              /*block_size=*/16)
nk_define_cross_scaled_exact_(mxfp4, serial, nk_e2m1_load_f32_serial_, nk_e2m1_load_f32_serial_, nk_ue8m0_split_serial_,
                              /*block_size=*/32)
nk_define_cross_scaled_exact_(mxfp6e2m3, serial, nk_e2m3_load_f32_serial_, nk_e2m3_load_f32_serial_,
                              nk_ue8m0_split_serial_, /*block_size=*/32)
nk_define_cross_scaled_exact_(mxfp6e3m2, serial, nk_e3m2_load_f32_serial_, nk_e3m2_load_f32_serial_,
                              nk_ue8m0_split_serial_, /*block_size=*/32)
nk_define_cross_scaled_exact_(mxfp8e4m3, serial, nk_e4m3_load_f32_serial_, nk_e4m3_load_f32_serial_,
                              nk_ue8m0_split_serial_, /*block_size=*/32)
nk_define_cross_scaled_exact_(mxfp8e5m2, serial, nk_e5m2_load_f32_serial_, nk_e5m2_load_f32_serial_,
                              nk_ue8m0_split_serial_, /*block_size=*/32)

/** Generates the serial packed and symmetric block-scaled entries of @p api_name, every
 *  output from the wide sums of @c nk_cross_scaled_exact_wide_<type>_serial_ through
 *  @p wide_metric_fn, shaped like @c nk_dot_f32_from_wide_; packs keep raw codes and raw scale
 *  rows in @c nk_define_cross_pack_ layout, and symmetric outputs fill each row from its
 *  diagonal to the last column. */
#define nk_define_cross_scaled_serial_(api_name, input_type_name, wide_metric_fn)                                      \
    NUMKONG_API nk_status_t nk_##api_name##_packed_##input_type_name##_serial(                                         \
        nk_##input_type_name##_cref_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns,  \
        nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, void *stream) {                                       \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;             \
        nk_cross_operand_t const a_unpacked = nk_cross_operand_(nk_##input_type_name##_k, a, a_stride);                \
        nk_u8_t const *b_values = (nk_u8_t const *)b_packed + sizeof(nk_cross_packed_buffer_header_t);                 \
        nk_u8_t const *b_scales = nk_cross_packed_scales_(b_packed, 1, 0);                                             \
        nk_f32_t const a_tensor_scale = nk_cross_tensor_scale_(a_unpacked.tensor_scale);                               \
        nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_(a_tensor_scale, header->tensor_scale);         \
        nk_cross_tensor_factor_t const a_factor = nk_cross_tensor_factor_(a_tensor_scale, a_tensor_scale);             \
        nk_cross_tensor_factor_t const b_factor = nk_cross_tensor_factor_(header->tensor_scale, header->tensor_scale); \
        for (nk_size_t row = 0; row < rows; ++row) {                                                                   \
            nk_u8_t const *a_row = (nk_u8_t const *)a_unpacked.elements + row * a_stride;                              \
            nk_u8_t const *a_scales = a_unpacked.scales + row * a_unpacked.scales_stride;                              \
            nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);                                                \
            for (nk_size_t column = 0; column < columns; ++column) {                                                   \
                nk_cross_wide_sum_t a_sumsq, b_sumsq;                                                                  \
                nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_##input_type_name##_serial_(                \
                    a_row, a_scales, b_values + column * header->depth_padded_values,                                  \
                    b_scales + column * header->scales_stride, depth, &a_sumsq, &b_sumsq);                             \
                c_row[column] = wide_metric_fn(nk_cross_wide_times_(dot, factor),                                      \
                                               nk_cross_wide_times_(a_sumsq, a_factor),                                \
                                               nk_cross_wide_times_(b_sumsq, b_factor));                               \
            }                                                                                                          \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##api_name##_symmetric_##input_type_name##_serial(                                      \
        nk_##input_type_name##_cref_t const *vectors, nk_size_t count, nk_size_t depth, nk_size_t stride,              \
        nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {           \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        nk_cross_operand_t const unpacked = nk_cross_operand_(nk_##input_type_name##_k, vectors, stride);              \
        nk_f32_t const tensor_scale = nk_cross_tensor_scale_(unpacked.tensor_scale);                                   \
        nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_(tensor_scale, tensor_scale);                   \
        nk_size_t const row_end = row_start + row_count < count ? row_start + row_count : count;                       \
        for (nk_size_t row = row_start; row < row_end; ++row) {                                                        \
            nk_u8_t const *row_codes = (nk_u8_t const *)unpacked.elements + row * stride;                              \
            nk_u8_t const *row_scales = unpacked.scales + row * unpacked.scales_stride;                                \
            nk_f32_t *result_row = (nk_f32_t *)((char *)result + row * result_stride);                                 \
            for (nk_size_t column = row; column < count; ++column) {                                                   \
                nk_cross_wide_sum_t row_sumsq, column_sumsq;                                                           \
                nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_##input_type_name##_serial_(                \
                    row_codes, row_scales, (nk_u8_t const *)unpacked.elements + column * stride,                       \
                    unpacked.scales + column * unpacked.scales_stride, depth, &row_sumsq, &column_sumsq);              \
                result_row[column] = wide_metric_fn(nk_cross_wide_times_(dot, factor),                                 \
                                                    nk_cross_wide_times_(row_sumsq, factor),                           \
                                                    nk_cross_wide_times_(column_sumsq, factor));                       \
            }                                                                                                          \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }

/*  Norm helpers that @c nk_define_cross_pack_ uses to append per-column norms to packed buffers.
 *  Each computes the norm, sum-of-squares or popcount, of a row whose values sit @p stride bytes
 *  apart with the serial reduction, which every packing capability shares. */
NUMKONG_INLINE nk_f64_t nk_dots_reduce_sumsq_f64_(nk_f64_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f64_t sum, sumsq;
    nk_reduce_moments_f64_strided_(data, count, stride, &sum, &sumsq);
    return sumsq;
}
NUMKONG_INLINE nk_f64_t nk_dots_reduce_sumsq_f32_(nk_f32_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f64_t sum, sumsq;
    nk_reduce_moments_f32_strided_(data, count, stride, &sum, &sumsq);
    return sumsq;
}
NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_f16_(nk_f16_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_f16_strided_(data, count, stride, &sum, &sumsq);
    return sumsq;
}
NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_bf16_(nk_bf16_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_bf16_strided_(data, count, stride, &sum, &sumsq);
    return sumsq;
}
NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e4m3_(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e4m3_strided_(data, count, stride, &sum, &sumsq);
    return sumsq;
}
NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e5m2_(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e5m2_strided_(data, count, stride, &sum, &sumsq);
    return sumsq;
}
NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m3_(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e2m3_strided_(data, count, stride, &sum, &sumsq);
    return sumsq;
}
NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m1_(nk_e2m1x2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e2m1_strided_(data, count, stride, &sum, &sumsq);
    return sumsq;
}
NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e3m2_(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e3m2_strided_(data, count, stride, &sum, &sumsq);
    return sumsq;
}
NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i8_(nk_i8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_i64_t sum;
    nk_u64_t sumsq;
    nk_reduce_moments_i8_strided_(data, count, stride, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}
NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u8_(nk_u8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_u64_t sum, sumsq;
    nk_reduce_moments_u8_strided_(data, count, stride, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}
NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i4_(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_i64_t sum;
    nk_u64_t sumsq;
    nk_reduce_moments_i4_strided_(data, count, stride, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}
NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u4_(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_u64_t sum, sumsq;
    nk_reduce_moments_u4_strided_(data, count, stride, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}
NUMKONG_INLINE nk_u32_t nk_dots_reduce_sum_u1_(nk_u1x8_t const *data, nk_size_t count_bits, nk_size_t stride) {
    nk_u64_t sum, sumsq;
    nk_reduce_moments_u1_strided_(data, count_bits, stride, &sum, &sumsq);
    return (nk_u32_t)sum;
}

/*  Combined moment trampolines for compensated GEMM, each computing a sum and a norm, the sum of
 *  squares, of a row whose values sit @p stride bytes apart, in a single @c nk_reduce_moments_*
 *  call for @c nk_define_cross_compensated_pack_ to store both in the packed buffer. */
NUMKONG_INLINE void nk_dots_reduce_moments_i8_(nk_i8_t const *data, nk_size_t count, nk_size_t stride, nk_i32_t *sum,
                                               nk_u32_t *norm) {
    nk_i64_t s;
    nk_u64_t sq;
    nk_reduce_moments_i8_strided_(data, count, stride, &s, &sq);
    *sum = (nk_i32_t)s;
    *norm = (nk_u32_t)sq;
}
NUMKONG_INLINE void nk_dots_reduce_moments_u8_(nk_u8_t const *data, nk_size_t count, nk_size_t stride, nk_u32_t *sum,
                                               nk_u32_t *norm) {
    nk_u64_t s, sq;
    nk_reduce_moments_u8_strided_(data, count, stride, &s, &sq);
    *sum = (nk_u32_t)s;
    *norm = (nk_u32_t)sq;
}
NUMKONG_INLINE void nk_dots_reduce_moments_i4_(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride, nk_i32_t *sum,
                                               nk_u32_t *norm) {
    nk_i64_t s;
    nk_u64_t sq;
    nk_reduce_moments_i4_strided_(data, count, stride, &s, &sq);
    *sum = (nk_i32_t)s;
    *norm = (nk_u32_t)sq;
}

/*  A-row sum helpers for compensated GEMM finalization. i8/u8: no A-side correction needed, stubs
 *  return 0. i4: needs A-side sum for correction term. */
NUMKONG_INLINE nk_i32_t nk_dots_reduce_sum_i8_stub_(nk_i8_t const *d, nk_size_t c) {
    nk_unused_(d);
    nk_unused_(c);
    return 0;
}
NUMKONG_INLINE nk_i32_t nk_dots_reduce_sum_u8_stub_(nk_u8_t const *d, nk_size_t c) {
    nk_unused_(d);
    nk_unused_(c);
    return 0;
}
NUMKONG_INLINE nk_i32_t nk_dots_reduce_sum_i4_(nk_i4x2_t const *data, nk_size_t count) {
    nk_i64_t sum;
    nk_u64_t sumsq;
    nk_reduce_moments_i4_strided_(data, count, sizeof(nk_i4x2_t), &sum, &sumsq);
    return (nk_i32_t)sum;
}

/**
 *  @brief Generates function to calculate packed B matrix buffer size for GEMM micro-kernels.
 *
 *  Memory layout: B_packed[column_count, depth_padded] with header storing metadata.
 *  Buffer size: sizeof(header) + column_count × depth_padded × sizeof(packed_value_type) +
 *  column_count × sizeof(norm).
 *  Depth padding logic: Round up to @c depth_simd_dimensions multiple, then add
 *  @c depth_simd_dimensions if stride is power-of-2.
 *
 *  @param[in] api_name Operation name, hammings or dots.
 *  @param[in] input_type_name B matrix's original type name, e.g. i4, f16, bf16, e4m3, e5m2, f32.
 *  @param[in] isa_suffix Platform ISA suffix, e.g. serial, haswell, icelake.
 *  @param[in] input_value_type B matrix's original type, e.g. i4x2, f16, bf16, e4m3, e5m2, f32.
 *  @param[in] packed_value_type Packed storage type, often bf16 or f32 for mixed precision.
 *  @param[in] norm_value_type Per-column norm type, f32/f64/u32, appended after packed data.
 *  @param[in] depth_simd_dimensions SIMD vector width in values for this platform/type combination.
 *  @param[in] dimensions_per_value Logical dimensions per single input_type_name value.
 *  @param[in] scale_bytes Bytes one packed block scale takes, 1 for raw scale bytes.
 */
#define nk_define_cross_pack_size_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type, \
                                   norm_value_type, depth_simd_dimensions, dimensions_per_value, scale_bytes)  \
    NUMKONG_API nk_status_t nk_##api_name##_pack_size_##input_type_name##_##isa_suffix(                        \
        nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {                                           \
        nk_assert_(depth % dimensions_per_value == 0);                                                         \
        /* depth_simd_dimensions is also in logical dimensions */                                              \
                                                                                                               \
        /* Pad depth in dimensions */                                                                          \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);       \
                                                                                                               \
        /* Convert dimensions to storage values */                                                             \
        nk_size_t depth_values_padded = depth_dimensions_padded / dimensions_per_value;                        \
                                                                                                               \
        /* Calculate stride in bytes for power-of-2 check */                                                   \
        nk_size_t const stride = depth_values_padded * sizeof(nk_##packed_value_type##_t);                     \
                                                                                                               \
        /* Break power-of-2 strides for cache associativity */                                                 \
        if ((stride & (stride - 1)) == 0 && stride > 0) {                                                      \
            /* Add one SIMD step worth of storage values */                                                    \
            depth_values_padded += depth_simd_dimensions / dimensions_per_value;                               \
        }                                                                                                      \
                                                                                                               \
        /* Return total buffer size (packed data + block scales + per-column norms) */                         \
        *bytes = sizeof(nk_cross_packed_buffer_header_t) +                                                     \
                 column_count * depth_values_padded * sizeof(nk_##packed_value_type##_t) +                     \
                 column_count * nk_cross_scales_stride_(nk_##input_type_name##_k, depth) * (scale_bytes) +     \
                 column_count * sizeof(nk_##norm_value_type##_t);                                              \
        return nk_success_k;                                                                                   \
    }

/**
 *  @brief Generates a packed-shape accessor reading a cross-backend packed buffer header.
 *
 *  Mirrors nk_define_cross_pack_size_ but reads the exact dims back out of the header.
 *
 *  @param[in] api_name Family name (dots, maxsim).
 *  @param[in] input_type_name Data type suffix (bf16, f16, i8, u1, etc.)
 *  @param[in] isa_suffix Backend suffix (serial, haswell, neon, etc.)
 */
#define nk_define_cross_packed_shape_(api_name, input_type_name, isa_suffix)                               \
    NUMKONG_API nk_status_t nk_##api_name##_packed_shape_##input_type_name##_##isa_suffix(                 \
        void const *b_packed, nk_size_t *columns, nk_size_t *depth, void *stream) {                        \
        nk_assert_(stream == NUMKONG_NULL);                                                                \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed; \
        if (header->capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;                      \
        *columns = header->column_count;                                                                   \
        *depth = header->depth_dimensions;                                                                 \
        return nk_success_k;                                                                               \
    }

/** Stores in @p norm the squared norm of row @p row_index of @p operand: through @p compute_norm_fn
 *  for plain dtypes, and for block-scaled ones through it block by block, each block times its
 *  squared scale, in a wide F32 sum rounded once relative to four to the power of the row's
 *  @c nk_cross_scaled_base_ and without the tensor scale, as rebased kernels pair it with
 *  their relative dots. */
#define nk_cross_row_norm_(norm, norm_value_type, compute_norm_fn, input_type_name, row, depth, dimensions_per_value, \
                           operand, row_index)                                                                        \
    do {                                                                                                              \
        nk_block_scaled_format_t const norm_format = nk_block_scaled_format_of_dtype(nk_##input_type_name##_k);       \
        nk_cross_wide_sum_t norm_sum = {0, 0};                                                                        \
        for (nk_size_t norm_block = 0; norm_format.block_size && norm_block * norm_format.block_size < (depth);       \
             ++norm_block) {                                                                                          \
            nk_u8_t const norm_code = (operand).scales[(row_index) * (operand).scales_stride + norm_block];           \
            nk_i32_t norm_exponent;                                                                                   \
            nk_f32_t const norm_mantissa = norm_format.scale_dtype == nk_ue4m3_k                                      \
                                               ? nk_ue4m3_split_serial_(norm_code, &norm_exponent)                    \
                                               : nk_ue8m0_split_serial_(norm_code, &norm_exponent);                   \
            nk_f32_t const norm_block_sumsq = (nk_f32_t)compute_norm_fn(                                              \
                (row) + norm_block * norm_format.block_size / (dimensions_per_value), norm_format.block_size,         \
                sizeof(*(row)));                                                                                      \
            nk_cross_wide_add_(&norm_sum, norm_block_sumsq * norm_mantissa * norm_mantissa, 2 * norm_exponent);       \
        }                                                                                                             \
        if (norm_format.block_size) {                                                                                 \
            nk_i32_t norm_base = 0, norm_spread;                                                                      \
            if (norm_format.scale_dtype == nk_ue8m0_k)                                                                \
                norm_base = nk_cross_scaled_base_((operand).scales + (row_index) * (operand).scales_stride,           \
                                                  (depth) / norm_format.block_size, &norm_spread);                    \
            norm = (nk_##norm_value_type##_t)nk_f32_scale_(norm_sum.sum, norm_sum.exponent - 2 * norm_base);          \
        }                                                                                                             \
        else norm = compute_norm_fn(row, depth, sizeof(*(row)));                                                      \
    } while (0)

/**
 *  @brief Generates pack function using SIMD load/store helpers.
 *
 *  Packs the B matrix into padded row-major layout with optional type conversion, using vectorized
 *  load/store for the bulk copy and a small scalar tail for padding.
 *
 *  @param[in] vec_type SIMD vector type, nk_b512_vec_t, nk_b256_vec_t, or nk_b128_vec_t.
 *  @param[in] load_fn Full load: void fn(void const*, vec_type*).
 *  @param[in] partial_load_fn Masked/partial load: void fn(void const*, vec_type*, nk_size_t).
 *  @param[in] store_fn Full store: void fn(vec_type const*, void*).
 *  @param[in] partial_store_fn Masked/partial store: void fn(vec_type const*, void*, nk_size_t).
 *  @param[in] simd_width Elements per SIMD load/store operation.
 *  @param[in] pack_scales_fn Scale row writer, shaped like @c nk_cross_pack_scales_bytes_.
 *  @param[in] scale_bytes Packed bytes per block scale, matching @c nk_define_cross_pack_size_.
 */
#define nk_define_cross_pack_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type, vec_type,   \
                              load_fn, partial_load_fn, store_fn, partial_store_fn, simd_width, norm_value_type,      \
                              compute_norm_fn, depth_simd_dimensions, dimensions_per_value, pack_scales_fn,           \
                              scale_bytes)                                                                            \
    /* Packs rows [rows_begin, rows_end) of source over dims [depth_first, depth_first + depth) into                  \
     * rows of values_stride values and scale rows of scales_stride bytes, rebasing row r by                          \
     * bases[r - rows_begin]; packs pass one column at a time, and block-scaled kernels stage                         \
     * panels of A through it. */                                                                                     \
    NUMKONG_INLINE void nk_##api_name##_pack_rows_##input_type_name##_##isa_suffix##_(                                \
        nk_cross_operand_t source, nk_size_t source_stride, nk_size_t rows_begin, nk_size_t rows_end,                 \
        nk_size_t depth_first, nk_size_t depth, nk_i32_t const *bases, nk_##packed_value_type##_t *values,            \
        nk_size_t values_stride, nk_u8_t *scales, nk_size_t scales_stride) {                                          \
        nk_size_t const block_size = nk_block_scaled_format_of_dtype(nk_##input_type_name##_k).block_size;            \
        nk_size_t const depth_in_values = depth / dimensions_per_value;                                               \
        nk_size_t const full_chunks = depth_in_values / (simd_width);                                                 \
        nk_size_t const remainder = depth_in_values % (simd_width);                                                   \
        for (nk_size_t row = rows_begin; row < rows_end; ++row) {                                                     \
            nk_##input_value_type##_t const *source_row =                                                             \
                (nk_##input_value_type##_t const *)((char const *)source.elements + row * source_stride) +            \
                depth_first / dimensions_per_value;                                                                   \
            nk_##packed_value_type##_t *destination_row = values + (row - rows_begin) * values_stride;                \
            for (nk_size_t chunk = 0; chunk < full_chunks; ++chunk) {                                                 \
                vec_type vec;                                                                                         \
                load_fn(source_row + chunk * (simd_width), &vec);                                                     \
                store_fn(&vec, destination_row + chunk * (simd_width));                                               \
            }                                                                                                         \
            /* Zero the padding first: a tail store may spill decoded values past the depth */                        \
            for (nk_size_t pad = depth_in_values; pad < values_stride; ++pad) destination_row[pad] = 0;               \
            if (remainder > 0) {                                                                                      \
                vec_type vec;                                                                                         \
                partial_load_fn(source_row + full_chunks * (simd_width), &vec, remainder);                            \
                partial_store_fn(&vec, destination_row + full_chunks * (simd_width), remainder);                      \
            }                                                                                                         \
            if (block_size)                                                                                           \
                pack_scales_fn(source.scales + row * source.scales_stride + depth_first / block_size,                 \
                               depth / block_size, scales + (row - rows_begin) * scales_stride, scales_stride,        \
                               bases ? bases[row - rows_begin] : 0);                                                  \
        }                                                                                                             \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_##api_name##_pack_##input_type_name##_##isa_suffix(                                    \
        nk_cross_##input_type_name##_operand_t const *b_operand, nk_size_t column_count, nk_size_t depth,             \
        nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, void *stream) {           \
        nk_cross_operand_t const b_unpacked = nk_cross_operand_(nk_##input_type_name##_k, b_operand, b_stride);       \
        nk_block_scaled_format_t const format = nk_block_scaled_format_of_dtype(nk_##input_type_name##_k);            \
        nk_size_t const scales_stride = nk_cross_scales_stride_(nk_##input_type_name##_k, depth) * (scale_bytes);     \
        nk_size_t const blocks = nk_cross_scale_blocks_(nk_##input_type_name##_k, depth);                             \
        nk_f32_t const tensor_scale = nk_cross_tensor_scale_(b_unpacked.tensor_scale);                                \
        nk_assert_(stream == NUMKONG_NULL);                                                                           \
        nk_assert_(depth % dimensions_per_value == 0 && nk_cross_whole_blocks_(nk_##input_type_name##_k, depth));     \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);              \
        nk_size_t depth_values_padded = depth_dimensions_padded / dimensions_per_value;                               \
        nk_size_t const stride = depth_values_padded * sizeof(nk_##packed_value_type##_t);                            \
        if ((stride & (stride - 1)) == 0 && stride > 0)                                                               \
            depth_values_padded += depth_simd_dimensions / dimensions_per_value;                                      \
                                                                                                                      \
        nk_##packed_value_type##_t *packed = (nk_##packed_value_type##_t *)((char *)b_packed +                        \
                                                                            sizeof(nk_cross_packed_buffer_header_t)); \
        nk_u8_t *scales = (nk_u8_t *)(packed + column_count * depth_values_padded);                                   \
        nk_##norm_value_type##_t *norms = (nk_##norm_value_type##_t *)(scales + column_count * scales_stride);        \
        if (columns_begin == 0) {                                                                                     \
            nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;                    \
            header->column_count = (nk_u32_t)column_count;                                                            \
            header->depth_dimensions = (nk_u32_t)depth;                                                               \
            header->depth_padded_values = (nk_u32_t)depth_values_padded;                                              \
            header->scales_stride = (nk_u32_t)scales_stride;                                                          \
            header->tensor_scale = tensor_scale;                                                                      \
            header->norms_offset = (nk_u32_t)((char *)norms - (char *)b_packed);                                      \
            header->capability = nk_cap_##isa_suffix##_k;                                                             \
            for (nk_size_t reserved_index = 0; reserved_index < 8; reserved_index++)                                  \
                header->reserved[reserved_index] = 0;                                                                 \
        }                                                                                                             \
        for (nk_size_t column_index = columns_begin; column_index < columns_end; ++column_index) {                    \
            nk_##input_value_type##_t const *source_row =                                                             \
                (nk_##input_value_type##_t const *)((char const *)b_unpacked.elements + column_index * b_stride);     \
            nk_u8_t const *source_scales = blocks ? b_unpacked.scales + column_index * b_unpacked.scales_stride       \
                                                  : b_unpacked.scales;                                                \
            nk_i32_t base = 0, spread;                                                                                \
            if (format.scale_dtype == nk_ue8m0_k) base = nk_cross_scaled_base_(source_scales, blocks, &spread);       \
            nk_##api_name##_pack_rows_##input_type_name##_##isa_suffix##_(                                            \
                b_unpacked, b_stride, column_index, column_index + 1, 0, depth, &base,                                \
                packed + column_index * depth_values_padded, depth_values_padded,                                     \
                scales + column_index * scales_stride, scales_stride);                                                \
            nk_cross_row_norm_(norms[column_index], norm_value_type, compute_norm_fn, input_type_name, source_row,    \
                               depth, dimensions_per_value, b_unpacked, column_index);                                \
        }                                                                                                             \
        return nk_success_k;                                                                                          \
    }

/**
 *  @brief Generates function to calculate packed B matrix buffer size for compensated GEMM.
 *
 *  Like nk_define_cross_pack_size_, but the buffer stores both norms and column sums, laid out as
 *  `[header 64B] [packed data] [norms, norm_type] [column sums, sum_type]`. Norms come first, so
 *  nk_define_cross_normalized_packed_ reads them at the same offset.
 */
#define nk_define_cross_compensated_pack_size_(api_name, input_type_name, isa_suffix, input_value_type,            \
                                               packed_value_type, sum_value_type, norm_value_type,                 \
                                               depth_simd_dimensions, dimensions_per_value)                        \
    NUMKONG_API nk_status_t nk_##api_name##_pack_size_##input_type_name##_##isa_suffix(                            \
        nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {                                               \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);           \
        nk_size_t depth_values_padded = depth_dimensions_padded / dimensions_per_value;                            \
        nk_size_t const stride = depth_values_padded * sizeof(nk_##packed_value_type##_t);                         \
        if ((stride & (stride - 1)) == 0 && stride > 0) {                                                          \
            depth_values_padded += depth_simd_dimensions / dimensions_per_value;                                   \
        }                                                                                                          \
        *bytes = sizeof(nk_cross_packed_buffer_header_t) +                                                         \
                 column_count * depth_values_padded * sizeof(nk_##packed_value_type##_t) +                         \
                 column_count * sizeof(nk_##norm_value_type##_t) + column_count * sizeof(nk_##sum_value_type##_t); \
        return nk_success_k;                                                                                       \
    }

/**
 *  @brief Like nk_define_cross_pack_ but stores both per-column norms and column sums.
 *
 *  Layout: [ Header 64B ] [ Packed data ] [ Norms (norm_type) ] [ Column sums (sum_type) ]
 */
#define nk_define_cross_compensated_pack_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type, \
                                          vec_type, load_fn, partial_load_fn, store_fn, partial_store_fn, simd_width, \
                                          sum_value_type, norm_value_type, compute_moments_fn, depth_simd_dimensions, \
                                          dimensions_per_value)                                                       \
    NUMKONG_API nk_status_t nk_##api_name##_pack_##input_type_name##_##isa_suffix(                                    \
        nk_##input_value_type##_t const *b, nk_size_t column_count, nk_size_t depth, nk_size_t b_stride,              \
        void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, void *stream) {                               \
        nk_assert_(stream == NUMKONG_NULL);                                                                           \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);              \
        nk_size_t depth_values_padded = depth_dimensions_padded / dimensions_per_value;                               \
        nk_size_t const stride = depth_values_padded * sizeof(nk_##packed_value_type##_t);                            \
        if ((stride & (stride - 1)) == 0 && stride > 0)                                                               \
            depth_values_padded += depth_simd_dimensions / dimensions_per_value;                                      \
        nk_size_t const depth_in_values = depth / dimensions_per_value;                                               \
        nk_##packed_value_type##_t *packed = (nk_##packed_value_type##_t *)((char *)b_packed +                        \
                                                                            sizeof(nk_cross_packed_buffer_header_t)); \
                                                                                                                      \
        if (columns_begin == 0) {                                                                                     \
            nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;                    \
            header->column_count = (nk_u32_t)column_count;                                                            \
            header->depth_dimensions = (nk_u32_t)depth;                                                               \
            header->depth_padded_values = (nk_u32_t)depth_values_padded;                                              \
            header->capability = nk_cap_##isa_suffix##_k;                                                             \
            header->scales_stride = 0;                                                                                \
            header->tensor_scale = 1;                                                                                 \
            header->norms_offset = (nk_u32_t)((char *)(packed + column_count * depth_values_padded) -                 \
                                              (char *)b_packed);                                                      \
            for (nk_size_t reserved_index = 0; reserved_index < 8; reserved_index++)                                  \
                header->reserved[reserved_index] = 0;                                                                 \
        }                                                                                                             \
                                                                                                                      \
        nk_size_t const full_chunks = depth_in_values / (simd_width);                                                 \
        nk_size_t const remainder = depth_in_values % (simd_width);                                                   \
                                                                                                                      \
        for (nk_size_t column_index = columns_begin; column_index < columns_end; ++column_index) {                    \
            nk_##input_value_type##_t const *source_row =                                                             \
                (nk_##input_value_type##_t const *)((char const *)b + column_index * b_stride);                       \
            nk_##packed_value_type##_t *destination_row = packed + column_index * depth_values_padded;                \
            for (nk_size_t chunk = 0; chunk < full_chunks; ++chunk) {                                                 \
                vec_type vec;                                                                                         \
                load_fn(source_row + chunk * (simd_width), &vec);                                                     \
                store_fn(&vec, destination_row + chunk * (simd_width));                                               \
            }                                                                                                         \
            if (remainder > 0) {                                                                                      \
                vec_type vec;                                                                                         \
                partial_load_fn(source_row + full_chunks * (simd_width), &vec, remainder);                            \
                partial_store_fn(&vec, destination_row + full_chunks * (simd_width), remainder);                      \
            }                                                                                                         \
            for (nk_size_t pad = depth_in_values; pad < depth_values_padded; ++pad) destination_row[pad] = 0;         \
        }                                                                                                             \
                                                                                                                      \
        nk_size_t const total_values = column_count * depth_values_padded;                                            \
        nk_##norm_value_type##_t *norms = (nk_##norm_value_type##_t *)(packed + total_values);                        \
        nk_##sum_value_type##_t *col_sums = (nk_##sum_value_type##_t *)(norms + column_count);                        \
        for (nk_size_t column_index = columns_begin; column_index < columns_end; ++column_index) {                    \
            nk_##input_value_type##_t const *source_row =                                                             \
                (nk_##input_value_type##_t const *)((char const *)b + column_index * b_stride);                       \
            compute_moments_fn(source_row, depth, sizeof(nk_##input_value_type##_t), &col_sums[column_index],         \
                               &norms[column_index]);                                                                 \
        }                                                                                                             \
        return nk_success_k;                                                                                          \
    }

/**
 *  @brief Generates optimized GEMM implementation: C = A × Bᵀ with pre-packed B matrix.
 *
 *  This macro creates a complete batched matrix multiplication kernel with three specialized code
 *  paths that are automatically selected based on the remaining work at each blocking level. The
 *  kernel requires B to be pre-packed using nk_define_cross_pack_ before invocation.
 *
 *  Mathematically, the kernel computes:
 *
 *  @verbatim
 *  C[row_count, column_count] = A[row_count, depth] × Bᵀ[column_count, depth] where operation can
 *  be dot product, Hamming distance, Jaccard similarity, etc.
 *  @endverbatim
 *
 *  Three kernel variants adapt to the work that remains:
 *
 *  1. @b 4×4 @b register @b tile @b kernel (primary path, ~80% of work):
 *     - Processes 4 rows of A × 4 columns of B simultaneously
 *     - Maintains 16 independent accumulators in registers (state_type[4][4])
 *     - Achieves maximum instruction-level parallelism (16 FMAs per depth iteration)
 *     - Used when: row_count ≥ 4 and column_count ≥ 4
 *     - Performance: Peak throughput, optimal register utilization
 *
 *  2. @b 1×8 @b register @b tile @b kernel (edge case, ~15% of work):
 *     - Processes 1 row of A × 8 columns of B when remaining rows < 4
 *     - Maintains 8 independent accumulators (state_type[1][8])
 *     - Balances vectorization with low row count
 *     - Used when: row_count < 4 and column_count ≥ 8
 *     - Performance: Better throughput than generic fallback for wide matrices
 *
 *  3. @b Generic @b fallback @b kernel (edge cases, ~5% of work):
 *     - Handles all irregular cases (row_count < 4 and column_count < 8)
 *     - Single accumulator, minimal unrolling
 *     - Used for: Small tiles, remainder handling
 *     - Performance: Lower throughput but handles all edge cases correctly
 *
 *  The cache blocking strategy skips depth. Unlike traditional GEMM, which blocks all three
 *  dimensions (M, N, K), this implementation omits depth (K) blocking for several reasons:
 *
 *  1. @b Streaming @b access @b pattern: A and B are read sequentially along depth dimension
 *     - Prefetcher-friendly access (hardware prefetch works well)
 *     - No cache reuse along depth within a single C[i,j] computation
 *
 *  2. @b Depth @b is @b typically @b small: For ML inference, depth is often 128-4096 values
 *     - Fits in L2/L3 cache for single row of A
 *     - B is pre-packed for optimal spatial locality
 *
 *  3. @b Simplicity @b and @b instruction @b cache @b efficiency:
 *     - Fewer nested loops = better instruction cache utilization
 *     - Simpler control flow = easier for compiler to optimize
 *
 *  Pre-packing the B matrix with @c nk_define_cross_pack_ before kernel invocation pays off in
 *  three ways:
 *  - @b Type @b conversion @b amortization: Convert B values once, bf16 → f32 for example, rather
 *    than per A row access. Saves (row_count - 1) × column_count conversions.
 *  - @b Cache @b line @b optimization: Pad depth to break power-of-2 strides that cause cache
 *    associativity conflicts (e.g., 8192 → 8200 values).
 *  - @b Spatial @b locality: Transpose B so columns are contiguous, enabling efficient SIMD loads.
 *
 *  The loop structure, in Python-like pseudocode:
 *
 *  @code{.py}
 *  for column_block in columns:        # step varies based on available columns
 *      for row_block in rows:          # step varies based on available rows
 *          for row_tile in row_block:      # step 4 or 1 depending on variant
 *              for column_tile in column_block:  # step 4 or 8 depending on variant
 *                  accumulator_tiles[row_tile][column_tile] = init_accumulator_fn()
 *                  for depth_index in depth:     # step depth_simd_dimensions
 *                      a_vectors = load_a_vec_fn(A[row_tile, depth_index])
 *                      b_vectors = load_b_vec_fn(B_packed[column_tile, depth_index])
 *                      accumulator_tiles = inner_product_fn(accumulator_tiles, a_vectors, b_vectors)
 *                  results = reduce_accumulators_fn(accumulator_tiles)
 *                  partial_store_fn(results, C[row_tile, column_tile])
 *  @endcode
 *
 *  The generated function has this signature:
 *
 *  @code{.c}
 *  nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(
 *      A_matrix, B_packed_buffer, C_matrix, row_count, column_count, depth,
 *      A_stride, C_stride)
 *  @endcode
 *
 *  @param[in] api_name Operation family, dots/hammings/jaccards, for codegen namespace.
 *  @param[in] input_type_name Type identifier for codegen, e.g. f32, bf16, i8, u1.
 *  @param[in] isa_suffix ISA backend identifier, e.g. serial, haswell, neon, sve, icelake.
 *  @param[in] input_value_type C type of input matrix values, e.g. f32, bf16, i8, u1x8.
 *  @param[in] packed_value_type Packed B storage type, often bf16 or f32 for mixed precision.
 *  @param[in] result_value_type C type of output matrix C values, e.g. f32, u32, f64.
 *  @param[in] vec_type SIMD vector type for depth dimension, e.g. __m256, nk_b256_vec_t.
 *  @param[in] state_type Accumulator state type, often vec_type or wider, e.g. __m256 or __m512.
 *  @param[in] result_vec_type Reduction-result SIMD vector type, e.g. __m128 for 4 f32 results.
 *  @param[in] init_accumulator_fn Initialize accumulator: void fn(state_type*).
 *  @param[in] load_a_vec_fn Full A load: vec_type fn(input_value_type const*, nk_size_t offset).
 *  @param[in] partial_load_a_vec_fn Partial A load for remainder.
 *  @param[in] load_b_vec_fn Full B load: vec_type fn(packed_value_type const*, nk_size_t offset).
 *  @param[in] partial_load_b_vec_fn Partial B load for remainder.
 *  @param[in] inner_product_fn Inner product accumulate.
 *  @param[in] reduce_accumulators_fn Reduce 4 accumulators.
 *  @param[in] store_fn Full-columns store for results.
 *  @param[in] partial_store_fn Partial store for results.
 *  @param[in] depth_simd_dimensions SIMD vector width in logical dimensions, e.g. 8 for f32 on
 *      AVX2, 128 for u1 on serial.
 *  @param[in] dimensions_per_value Packing ratio: dimensions per storage value, 1 for f32, 2 for
 *      i4x2, 8 for u1x8.
 *
 *  @sa nk_define_cross_symmetric_ for symmetric C = A × Aᵀ computation, upper triangle only.
 *  @sa nk_define_cross_pack_size_ for calculating B_packed buffer size
 *  @sa nk_define_cross_pack_ for packing B matrix into optimized layout
 *  @sa include/numkong/set/serial.h for state type definitions
 *  @sa include/numkong/cast/serial.h for load/store function implementations
 */
#define nk_define_cross_packed_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type,            \
                                result_value_type, vec_type, state_type, result_vec_type, init_accumulator_fn,         \
                                load_a_vec_fn, partial_load_a_vec_fn, load_b_vec_fn, partial_load_b_vec_fn,            \
                                inner_product_fn, reduce_accumulators_fn, store_fn, partial_store_fn,                  \
                                depth_simd_dimensions, dimensions_per_value)                                           \
    NUMKONG_INLINE void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(                            \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {        \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
                                                                                                                       \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
                                                                                                                       \
        /* Cache blocking parameters (no depth_block blocking - full depth accumulated per tile) */                    \
        nk_size_t const row_block_size = 128;      /* L2 cache blocking over rows */                                   \
        nk_size_t const column_block_size = 2048;  /* L3 cache blocking over columns */                                \
        nk_size_t const register_row_count = 4;    /* Rows per register tile */                                        \
        nk_size_t const register_column_count = 4; /* Columns per register tile */                                     \
        /* Correct aligned_depth calculation for sub-byte types */                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = depth_dimensions_aligned / dimensions_per_value;                               \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = depth_simd_dimensions / dimensions_per_value;                              \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride);                  \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        for (nk_size_t column_block_start_index = 0; column_block_start_index < column_count;                          \
             column_block_start_index += column_block_size) {                                                          \
            nk_size_t column_block_end_index = column_block_start_index + column_block_size;                           \
            if (column_block_end_index > column_count) column_block_end_index = column_count;                          \
                                                                                                                       \
            /* Loop 2: L2 cache blocking over rows */                                                                  \
            for (nk_size_t row_block_start_index = 0; row_block_start_index < row_count;                               \
                 row_block_start_index += row_block_size) {                                                            \
                nk_size_t row_block_end_index = row_block_start_index + row_block_size;                                \
                if (row_block_end_index > row_count) row_block_end_index = row_count;                                  \
                                                                                                                       \
                /* Loop 3: Register tiling over columns (register_column_count columns per batch) */                   \
                for (nk_size_t tile_column_start_index = column_block_start_index;                                     \
                     tile_column_start_index < column_block_end_index;                                                 \
                     tile_column_start_index += register_column_count) {                                               \
                                                                                                                       \
                    /* Compute B pointers once per column tile - direct column-major addressing */                     \
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data +                                    \
                                                                      (tile_column_start_index + 0) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 = packed_data +                                    \
                                                                      (tile_column_start_index + 1) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 = packed_data +                                    \
                                                                      (tile_column_start_index + 2) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 = packed_data +                                    \
                                                                      (tile_column_start_index + 3) * depth_padded;    \
                                                                                                                       \
                    /* Loop 4: Register tiling over rows (register_row_count rows per tile) */                         \
                    for (nk_size_t tile_row_start_index = row_block_start_index;                                       \
                         tile_row_start_index < row_block_end_index; tile_row_start_index += register_row_count) {     \
                                                                                                                       \
                        /* Initialize the register_row_count × register_column_count grid                             \
                         * of accumulator states */                                                                    \
                        state_type accumulator_tiles[4][4];                                                            \
                        init_accumulator_fn(&accumulator_tiles[0][0]), init_accumulator_fn(&accumulator_tiles[0][1]),  \
                            init_accumulator_fn(&accumulator_tiles[0][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[0][3]);                                             \
                        init_accumulator_fn(&accumulator_tiles[1][0]), init_accumulator_fn(&accumulator_tiles[1][1]),  \
                            init_accumulator_fn(&accumulator_tiles[1][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[1][3]);                                             \
                        init_accumulator_fn(&accumulator_tiles[2][0]), init_accumulator_fn(&accumulator_tiles[2][1]),  \
                            init_accumulator_fn(&accumulator_tiles[2][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[2][3]);                                             \
                        init_accumulator_fn(&accumulator_tiles[3][0]), init_accumulator_fn(&accumulator_tiles[3][1]),  \
                            init_accumulator_fn(&accumulator_tiles[3][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[3][3]);                                             \
                                                                                                                       \
                        /* A row pointers */                                                                           \
                        nk_##input_value_type##_t const *a_row_ptr_0 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 0) * a_stride);                \
                        nk_##input_value_type##_t const *a_row_ptr_1 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 1) * a_stride);                \
                        nk_##input_value_type##_t const *a_row_ptr_2 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 2) * a_stride);                \
                        nk_##input_value_type##_t const *a_row_ptr_3 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 3) * a_stride);                \
                                                                                                                       \
                        /* Tight inner loop: full depth with simple depth_index addressing */                          \
                        vec_type a_vector_0, a_vector_1, a_vector_2, a_vector_3;                                       \
                        vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3;                                       \
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth;                                   \
                             depth_index += depth_step_values) {                                                       \
                            /* Load next few values from 4 rows from A (unpacked, may upcast) */                       \
                            load_a_vec_fn(a_row_ptr_0 + depth_index, &a_vector_0);                                     \
                            load_a_vec_fn(a_row_ptr_1 + depth_index, &a_vector_1);                                     \
                            load_a_vec_fn(a_row_ptr_2 + depth_index, &a_vector_2);                                     \
                            load_a_vec_fn(a_row_ptr_3 + depth_index, &a_vector_3);                                     \
                                                                                                                       \
                            /* Load next few values from 4 rows from B (packed, already upcasted) */                   \
                            load_b_vec_fn(b_depth_ptr_0 + depth_index, &b_vector_0);                                   \
                            load_b_vec_fn(b_depth_ptr_1 + depth_index, &b_vector_1);                                   \
                            load_b_vec_fn(b_depth_ptr_2 + depth_index, &b_vector_2);                                   \
                            load_b_vec_fn(b_depth_ptr_3 + depth_index, &b_vector_3);                                   \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            inner_product_fn(&accumulator_tiles[0][0], a_vector_0, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[0][1], a_vector_0, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[0][2], a_vector_0, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[0][3], a_vector_0, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][0], a_vector_1, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][1], a_vector_1, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][2], a_vector_1, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][3], a_vector_1, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][0], a_vector_2, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][1], a_vector_2, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][2], a_vector_2, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][3], a_vector_2, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][0], a_vector_3, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][1], a_vector_3, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][2], a_vector_3, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][3], a_vector_3, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                        }                                                                                              \
                        /* Finalize and store register_rows x register_columns results                                 \
                         * using a batched 4-way reduction */                                                          \
                        result_vec_type result_vector;                                                                 \
                        nk_##result_value_type##_t *c_row_ptr_0 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix + (tile_row_start_index + 0) * c_stride);  \
                        reduce_accumulators_fn(&accumulator_tiles[0][0], &accumulator_tiles[0][1],                     \
                                               &accumulator_tiles[0][2], &accumulator_tiles[0][3], depth,              \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr_0 + tile_column_start_index);                               \
                        nk_##result_value_type##_t *c_row_ptr_1 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix + (tile_row_start_index + 1) * c_stride);  \
                        reduce_accumulators_fn(&accumulator_tiles[1][0], &accumulator_tiles[1][1],                     \
                                               &accumulator_tiles[1][2], &accumulator_tiles[1][3], depth,              \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr_1 + tile_column_start_index);                               \
                        nk_##result_value_type##_t *c_row_ptr_2 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix + (tile_row_start_index + 2) * c_stride);  \
                        reduce_accumulators_fn(&accumulator_tiles[2][0], &accumulator_tiles[2][1],                     \
                                               &accumulator_tiles[2][2], &accumulator_tiles[2][3], depth,              \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr_2 + tile_column_start_index);                               \
                        nk_##result_value_type##_t *c_row_ptr_3 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix + (tile_row_start_index + 3) * c_stride);  \
                        reduce_accumulators_fn(&accumulator_tiles[3][0], &accumulator_tiles[3][1],                     \
                                               &accumulator_tiles[3][2], &accumulator_tiles[3][3], depth,              \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr_3 + tile_column_start_index);                               \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_INLINE void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                        \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {        \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values; /* in storage values */                            \
                                                                                                                       \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
                                                                                                                       \
        /* Cache blocking parameters (no depth_block blocking - full depth accumulated per tile) */                    \
        nk_size_t const row_block_size = 128;      /* L2 cache blocking over rows */                                   \
        nk_size_t const column_block_size = 2048;  /* L3 cache blocking over columns */                                \
        nk_size_t const register_row_count = 1;    /* Rows per register tile */                                        \
        nk_size_t const register_column_count = 8; /* Columns per register tile (2 × 4) */                             \
        /* Correct aligned_depth calculation for sub-byte types */                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = depth_dimensions_aligned / dimensions_per_value;                               \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = depth_simd_dimensions / dimensions_per_value;                              \
        nk_unused_(register_row_count); /* Used in comments, loop uses 1 directly */                                   \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride);                  \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        for (nk_size_t column_block_start_index = 0; column_block_start_index < column_count;                          \
             column_block_start_index += column_block_size) {                                                          \
            nk_size_t column_block_end_index = column_block_start_index + column_block_size;                           \
            if (column_block_end_index > column_count) column_block_end_index = column_count;                          \
                                                                                                                       \
            /* Loop 2: L2 cache blocking over rows */                                                                  \
            for (nk_size_t row_block_start_index = 0; row_block_start_index < row_count;                               \
                 row_block_start_index += row_block_size) {                                                            \
                nk_size_t const row_block_end_index = row_block_start_index + row_block_size < row_count               \
                                                          ? row_block_start_index + row_block_size                     \
                                                          : row_count;                                                 \
                                                                                                                       \
                /* Loop 3: Register tiling over columns (register_column_count columns per batch) */                   \
                for (nk_size_t tile_column_start_index = column_block_start_index;                                     \
                     tile_column_start_index < column_block_end_index;                                                 \
                     tile_column_start_index += register_column_count) {                                               \
                                                                                                                       \
                    /* Compute B pointers once per column tile - direct column-major addressing */                     \
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data +                                    \
                                                                      (tile_column_start_index + 0) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 = packed_data +                                    \
                                                                      (tile_column_start_index + 1) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 = packed_data +                                    \
                                                                      (tile_column_start_index + 2) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 = packed_data +                                    \
                                                                      (tile_column_start_index + 3) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_4 = packed_data +                                    \
                                                                      (tile_column_start_index + 4) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_5 = packed_data +                                    \
                                                                      (tile_column_start_index + 5) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_6 = packed_data +                                    \
                                                                      (tile_column_start_index + 6) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_7 = packed_data +                                    \
                                                                      (tile_column_start_index + 7) * depth_padded;    \
                                                                                                                       \
                    /* Loop 4: Process 1 row at a time */                                                              \
                    for (nk_size_t row_index = row_block_start_index; row_index < row_block_end_index; ++row_index) {  \
                                                                                                                       \
                        /* Initialize 1 × 8 accumulator states */                                                      \
                        state_type accumulator_0, accumulator_1, accumulator_2, accumulator_3, accumulator_4,          \
                            accumulator_5, accumulator_6, accumulator_7;                                               \
                        init_accumulator_fn(&accumulator_0), init_accumulator_fn(&accumulator_1),                      \
                            init_accumulator_fn(&accumulator_2), init_accumulator_fn(&accumulator_3),                  \
                            init_accumulator_fn(&accumulator_4), init_accumulator_fn(&accumulator_5),                  \
                            init_accumulator_fn(&accumulator_6), init_accumulator_fn(&accumulator_7);                  \
                                                                                                                       \
                        /* A row pointer */                                                                            \
                        nk_##input_value_type##_t const *a_row_ptr =                                                   \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix + row_index * a_stride);        \
                                                                                                                       \
                        /* Tight inner loop: full depth with simple depth_index addressing */                          \
                        vec_type a_vector;                                                                             \
                        vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3, b_vector_4, b_vector_5, b_vector_6,   \
                            b_vector_7;                                                                                \
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth;                                   \
                             depth_index += depth_step_values) {                                                       \
                            /* Load A vector (1 row) */                                                                \
                            load_a_vec_fn(a_row_ptr + depth_index, &a_vector);                                         \
                                                                                                                       \
                            /* Load B vectors (8 columns) */                                                           \
                            load_b_vec_fn(b_depth_ptr_0 + depth_index, &b_vector_0);                                   \
                            load_b_vec_fn(b_depth_ptr_1 + depth_index, &b_vector_1);                                   \
                            load_b_vec_fn(b_depth_ptr_2 + depth_index, &b_vector_2);                                   \
                            load_b_vec_fn(b_depth_ptr_3 + depth_index, &b_vector_3);                                   \
                            load_b_vec_fn(b_depth_ptr_4 + depth_index, &b_vector_4);                                   \
                            load_b_vec_fn(b_depth_ptr_5 + depth_index, &b_vector_5);                                   \
                            load_b_vec_fn(b_depth_ptr_6 + depth_index, &b_vector_6);                                   \
                            load_b_vec_fn(b_depth_ptr_7 + depth_index, &b_vector_7);                                   \
                                                                                                                       \
                            /* 8 FMAs: 1 A row × 8 B columns */                                                        \
                            inner_product_fn(&accumulator_0, a_vector, b_vector_0, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_1, a_vector, b_vector_1, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_2, a_vector, b_vector_2, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_3, a_vector, b_vector_3, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_4, a_vector, b_vector_4, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_5, a_vector, b_vector_5, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_6, a_vector, b_vector_6, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_7, a_vector, b_vector_7, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store 1 × 8 results using two 4-way reductions */                              \
                        result_vec_type result_vector;                                                                 \
                        nk_##result_value_type##_t *c_row_ptr = (nk_##result_value_type##_t *)((char *)c_matrix +      \
                                                                                               row_index * c_stride);  \
                        /* First 4 columns */                                                                          \
                        reduce_accumulators_fn(&accumulator_0, &accumulator_1, &accumulator_2, &accumulator_3, depth,  \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr + tile_column_start_index);                                 \
                        /* Second 4 columns */                                                                         \
                        reduce_accumulators_fn(&accumulator_4, &accumulator_5, &accumulator_6, &accumulator_7, depth,  \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr + tile_column_start_index + 4);                             \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_INLINE nk_status_t nk_##api_name##_packed_##input_type_name##_##isa_suffix##_(                             \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {        \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        if (header->capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;                                  \
        nk_assert_(header->column_count == column_count && header->depth_dimensions == depth &&                        \
                   depth % dimensions_per_value == 0 && nk_cross_whole_blocks_(nk_##input_type_name##_k, depth));      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
                                                                                                                       \
        /* Cache blocking parameters (hardcoded for optimal L1/L2/L3 utilization) */                                   \
        nk_size_t const row_block_size = 128;      /* L2 cache blocking over rows */                                   \
        nk_size_t const column_block_size = 2048;  /* L3 cache blocking over columns */                                \
        nk_size_t const register_row_count = 4;    /* Rows per register tile */                                        \
        nk_size_t const register_column_count = 4; /* Columns per register tile */                                     \
        nk_unused_(register_column_count);         /* Suppress unused warnings */                                      \
        /* Use 1 × 8 kernel when columns are aligned to 8 and many columns relative to rows */                         \
        if (column_count % 8 == 0 && column_count >= row_count * 2 && depth % depth_simd_dimensions == 0) {            \
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                                    \
                a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth, a_stride, c_stride);              \
            return nk_success_k;                                                                                       \
        }                                                                                                              \
        /* Use 4 × 4 kernel when dimensions are 4-aligned */                                                           \
        if (row_count % 4 == 0 && column_count % 4 == 0 && depth % depth_simd_dimensions == 0) {                       \
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(                                        \
                a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth, a_stride, c_stride);              \
            return nk_success_k;                                                                                       \
        }                                                                                                              \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride);                  \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Compute aligned/remainder depth for partial loads (correct for sub-byte types) */                           \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = depth_dimensions_aligned / dimensions_per_value;                               \
        nk_size_t const depth_in_values = depth / dimensions_per_value;                                                \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = depth_simd_dimensions / dimensions_per_value;                              \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
        for (nk_size_t column_block_start_index = 0; column_block_start_index < column_count;                          \
             column_block_start_index += column_block_size) {                                                          \
            nk_size_t column_block_end_index = column_block_start_index + column_block_size;                           \
            if (column_block_end_index > column_count) column_block_end_index = column_count;                          \
                                                                                                                       \
            /* Loop 2: L2 cache blocking over rows */                                                                  \
            for (nk_size_t row_block_start_index = 0; row_block_start_index < row_count;                               \
                 row_block_start_index += row_block_size) {                                                            \
                nk_size_t row_block_end_index = row_block_start_index + row_block_size;                                \
                if (row_block_end_index > row_count) row_block_end_index = row_count;                                  \
                                                                                                                       \
                /* Loop 4: Register tiling over columns (register_column_count columns per batch) */                   \
                for (nk_size_t tile_column_start_index = column_block_start_index;                                     \
                     tile_column_start_index < column_block_end_index;                                                 \
                     tile_column_start_index += register_column_count) {                                               \
                    nk_size_t tile_column_count = register_column_count;                                               \
                    if (tile_column_start_index + tile_column_count > column_block_end_index)                          \
                        tile_column_count = column_block_end_index - tile_column_start_index;                          \
                                                                                                                       \
                    /* Compute B pointers once per column tile - direct column-major addressing */                     \
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data +                                    \
                                                                      (tile_column_start_index + 0) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 =                                                  \
                        (tile_column_count > 1) ? packed_data + (tile_column_start_index + 1) * depth_padded           \
                                                : b_depth_ptr_0;                                                       \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 =                                                  \
                        (tile_column_count > 2) ? packed_data + (tile_column_start_index + 2) * depth_padded           \
                                                : b_depth_ptr_0;                                                       \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 =                                                  \
                        (tile_column_count > 3) ? packed_data + (tile_column_start_index + 3) * depth_padded           \
                                                : b_depth_ptr_0;                                                       \
                                                                                                                       \
                    /* Loop 5: Register tiling over rows (register_rows rows per tile) */                              \
                    for (nk_size_t tile_row_start_index = row_block_start_index;                                       \
                         tile_row_start_index < row_block_end_index; tile_row_start_index += register_row_count) {     \
                        nk_size_t tile_row_count = register_row_count;                                                 \
                        if (tile_row_start_index + tile_row_count > row_block_end_index)                               \
                            tile_row_count = row_block_end_index - tile_row_start_index;                               \
                                                                                                                       \
                        /* Initialize register_rows x register_columns accumulator states */                           \
                        state_type accumulator_tiles[4][4];                                                            \
                        for (nk_size_t r = 0; r < tile_row_count; ++r) {                                               \
                            init_accumulator_fn(&accumulator_tiles[r][0]);                                             \
                            init_accumulator_fn(&accumulator_tiles[r][1]);                                             \
                            init_accumulator_fn(&accumulator_tiles[r][2]);                                             \
                            init_accumulator_fn(&accumulator_tiles[r][3]);                                             \
                        }                                                                                              \
                                                                                                                       \
                        /* A row pointers */                                                                           \
                        nk_##input_value_type##_t const *a_row_ptr_0 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 0) * a_stride);                \
                        nk_##input_value_type##_t const *a_row_ptr_1 =                                                 \
                            (tile_row_count > 1)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix +                         \
                                                                      (tile_row_start_index + 1) * a_stride)           \
                                : a_row_ptr_0;                                                                         \
                        nk_##input_value_type##_t const *a_row_ptr_2 =                                                 \
                            (tile_row_count > 2)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix +                         \
                                                                      (tile_row_start_index + 2) * a_stride)           \
                                : a_row_ptr_0;                                                                         \
                        nk_##input_value_type##_t const *a_row_ptr_3 =                                                 \
                            (tile_row_count > 3)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix +                         \
                                                                      (tile_row_start_index + 3) * a_stride)           \
                                : a_row_ptr_0;                                                                         \
                                                                                                                       \
                        /* Tight inner loop: k values with simple ptr+k addressing */                                  \
                        vec_type a_first_vec, a_second_vec, a_third_vec, a_fourth_vec;                                 \
                        vec_type b_first_vec, b_second_vec, b_third_vec, b_fourth_vec;                                 \
                        for (nk_size_t k = 0; k < aligned_depth; k += depth_step_values) {                             \
                            /* Load next few values from 4 rows from A */                                              \
                            load_a_vec_fn(a_row_ptr_0 + k, &a_first_vec);                                              \
                            load_a_vec_fn(a_row_ptr_1 + k, &a_second_vec);                                             \
                            load_a_vec_fn(a_row_ptr_2 + k, &a_third_vec);                                              \
                            load_a_vec_fn(a_row_ptr_3 + k, &a_fourth_vec);                                             \
                                                                                                                       \
                            /* Load next few values from 4 rows from B */                                              \
                            load_b_vec_fn(b_depth_ptr_0 + k, &b_first_vec);                                            \
                            load_b_vec_fn(b_depth_ptr_1 + k, &b_second_vec);                                           \
                            load_b_vec_fn(b_depth_ptr_2 + k, &b_third_vec);                                            \
                            load_b_vec_fn(b_depth_ptr_3 + k, &b_fourth_vec);                                           \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            inner_product_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                        }                                                                                              \
                                                                                                                       \
                        /* Handle remainder k positions with partial loads */                                          \
                        if (remainder_depth > 0) {                                                                     \
                            /* Load next few values from 4 rows from A */                                              \
                            partial_load_a_vec_fn(a_row_ptr_0 + aligned_depth, &a_first_vec, remainder_dimensions);    \
                            partial_load_a_vec_fn(a_row_ptr_1 + aligned_depth, &a_second_vec, remainder_dimensions);   \
                            partial_load_a_vec_fn(a_row_ptr_2 + aligned_depth, &a_third_vec, remainder_dimensions);    \
                            partial_load_a_vec_fn(a_row_ptr_3 + aligned_depth, &a_fourth_vec, remainder_dimensions);   \
                                                                                                                       \
                            /* Load next few values from 4 rows from B */                                              \
                            partial_load_b_vec_fn(b_depth_ptr_0 + aligned_depth, &b_first_vec, remainder_dimensions);  \
                            partial_load_b_vec_fn(b_depth_ptr_1 + aligned_depth, &b_second_vec, remainder_dimensions); \
                            partial_load_b_vec_fn(b_depth_ptr_2 + aligned_depth, &b_third_vec, remainder_dimensions);  \
                            partial_load_b_vec_fn(b_depth_ptr_3 + aligned_depth, &b_fourth_vec, remainder_dimensions); \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            inner_product_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store register_rows x register_columns results                                 \
                         * using a batched 4-way reduction */                                                          \
                        for (nk_size_t r = 0; r < tile_row_count; ++r) {                                               \
                            result_vec_type result_vector;                                                             \
                            reduce_accumulators_fn(&accumulator_tiles[r][0], &accumulator_tiles[r][1],                 \
                                                   &accumulator_tiles[r][2], &accumulator_tiles[r][3], depth,          \
                                                   &result_vector);                                                    \
                                                                                                                       \
                            nk_##result_value_type##_t *c_row =                                                        \
                                (nk_##result_value_type##_t *)((char *)c_matrix +                                      \
                                                               (tile_row_start_index + r) * c_stride);                 \
                            partial_store_fn(&result_vector, c_row + tile_column_start_index, tile_column_count);      \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##api_name##_packed_##input_type_name##_##isa_suffix(                                   \
        nk_cross_##input_type_name##_operand_t const *a_operand, void const *b_packed_buffer,                          \
        nk_##result_value_type##_t *c_matrix, nk_size_t row_count, nk_size_t column_count, nk_size_t depth,            \
        nk_size_t a_stride, nk_size_t c_stride, void *stream) {                                                        \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        return nk_##api_name##_packed_##input_type_name##_##isa_suffix##_(                                             \
            (nk_##input_value_type##_t const *)a_operand, b_packed_buffer, c_matrix, row_count, column_count, depth,   \
            a_stride, c_stride);                                                                                       \
    }

/**
 *  @brief Generates compensated GEMM: C = A × Bᵀ with precomputed B column sums.
 *
 *  Like nk_define_cross_packed_ but the finalize function receives precomputed B column sums and
 *  per-row A sums to apply algebraic correction inline. This eliminates correction accumulators
 *  from the inner loop state, halving register pressure for integer dot products.
 *
 *  The compensated_finalize_fn signature differs from the standard reduce_accumulators_fn:
 *    compensated_finalize_fn(state_a, state_b, state_c, state_d, depth, a_sum, b_sums_vec, result)
 *  where a_sum is a scalar A row sum and b_sums_vec contains 4 B column sums as SIMD vector.
 *
 *  Buffer layout: [ Header ] [ Packed data ] [ Norms ] [ Column sums ]. The norms occupy the same
 *  position as in non-compensated packs, so spatial functions work.
 */
#define nk_define_cross_compensated_packed_(                                                                           \
    api_name, input_type_name, isa_suffix, input_value_type, packed_value_type, result_value_type, sum_value_type,     \
    norm_value_type, vec_type, state_type, result_vec_type, init_accumulator_fn, load_a_vec_fn, partial_load_a_vec_fn, \
    load_b_vec_fn, partial_load_b_vec_fn, inner_product_fn, compensated_finalize_fn, store_fn, partial_store_fn,       \
    load_sum_fn, partial_load_sum_fn, compute_a_sum_fn, depth_simd_dimensions, dimensions_per_value)                   \
    NUMKONG_INLINE void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(                            \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {        \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
        /* Locate column sums: after packed data + norms */                                                            \
        nk_size_t const total_packed_values = column_count * depth_padded;                                             \
        nk_##norm_value_type##_t const *b_norms = (nk_##norm_value_type##_t const *)(packed_data +                     \
                                                                                     total_packed_values);             \
        nk_##sum_value_type##_t const *b_sums = (nk_##sum_value_type##_t const *)(b_norms + column_count);             \
        nk_unused_(b_norms);                                                                                           \
        nk_size_t const row_block_size = 128;                                                                          \
        nk_size_t const column_block_size = 2048;                                                                      \
        nk_size_t const register_row_count = 4;                                                                        \
        nk_size_t const register_column_count = 4;                                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = depth_dimensions_aligned / dimensions_per_value;                               \
        nk_size_t const depth_step_values = depth_simd_dimensions / dimensions_per_value;                              \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride);                  \
            for (nk_size_t ci = 0; ci < column_count; ++ci) c_row[ci] = 0;                                             \
        }                                                                                                              \
        for (nk_size_t cb = 0; cb < column_count; cb += column_block_size) {                                           \
            nk_size_t ce = cb + column_block_size;                                                                     \
            if (ce > column_count) ce = column_count;                                                                  \
            for (nk_size_t rb = 0; rb < row_count; rb += row_block_size) {                                             \
                nk_size_t re = rb + row_block_size;                                                                    \
                if (re > row_count) re = row_count;                                                                    \
                for (nk_size_t tc = cb; tc < ce; tc += register_column_count) {                                        \
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data + (tc + 0) * depth_padded;           \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 = packed_data + (tc + 1) * depth_padded;           \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 = packed_data + (tc + 2) * depth_padded;           \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 = packed_data + (tc + 3) * depth_padded;           \
                    /* Load 4 B column sums as SIMD vector */                                                          \
                    result_vec_type b_sum_vec;                                                                         \
                    load_sum_fn(b_sums + tc, &b_sum_vec);                                                              \
                    for (nk_size_t tr = rb; tr < re; tr += register_row_count) {                                       \
                        state_type acc[4][4];                                                                          \
                        init_accumulator_fn(&acc[0][0]), init_accumulator_fn(&acc[0][1]),                              \
                            init_accumulator_fn(&acc[0][2]), init_accumulator_fn(&acc[0][3]);                          \
                        init_accumulator_fn(&acc[1][0]), init_accumulator_fn(&acc[1][1]),                              \
                            init_accumulator_fn(&acc[1][2]), init_accumulator_fn(&acc[1][3]);                          \
                        init_accumulator_fn(&acc[2][0]), init_accumulator_fn(&acc[2][1]),                              \
                            init_accumulator_fn(&acc[2][2]), init_accumulator_fn(&acc[2][3]);                          \
                        init_accumulator_fn(&acc[3][0]), init_accumulator_fn(&acc[3][1]),                              \
                            init_accumulator_fn(&acc[3][2]), init_accumulator_fn(&acc[3][3]);                          \
                        nk_##input_value_type##_t const *a_row_ptr_0 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix + (tr + 0) * a_stride);         \
                        nk_##input_value_type##_t const *a_row_ptr_1 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix + (tr + 1) * a_stride);         \
                        nk_##input_value_type##_t const *a_row_ptr_2 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix + (tr + 2) * a_stride);         \
                        nk_##input_value_type##_t const *a_row_ptr_3 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix + (tr + 3) * a_stride);         \
                        /* Precompute A row sums (no-op for i8/u8, real for i4) */                                     \
                        nk_##sum_value_type##_t a_sums[4];                                                             \
                        a_sums[0] = compute_a_sum_fn(a_row_ptr_0, depth);                                              \
                        a_sums[1] = compute_a_sum_fn(a_row_ptr_1, depth);                                              \
                        a_sums[2] = compute_a_sum_fn(a_row_ptr_2, depth);                                              \
                        a_sums[3] = compute_a_sum_fn(a_row_ptr_3, depth);                                              \
                        vec_type av0, av1, av2, av3, bv0, bv1, bv2, bv3;                                               \
                        for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                          \
                            load_a_vec_fn(a_row_ptr_0 + di, &av0);                                                     \
                            load_a_vec_fn(a_row_ptr_1 + di, &av1);                                                     \
                            load_a_vec_fn(a_row_ptr_2 + di, &av2);                                                     \
                            load_a_vec_fn(a_row_ptr_3 + di, &av3);                                                     \
                            load_b_vec_fn(b_depth_ptr_0 + di, &bv0);                                                   \
                            load_b_vec_fn(b_depth_ptr_1 + di, &bv1);                                                   \
                            load_b_vec_fn(b_depth_ptr_2 + di, &bv2);                                                   \
                            load_b_vec_fn(b_depth_ptr_3 + di, &bv3);                                                   \
                            inner_product_fn(&acc[0][0], av0, bv0, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[0][1], av0, bv1, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[0][2], av0, bv2, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[0][3], av0, bv3, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[1][0], av1, bv0, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[1][1], av1, bv1, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[1][2], av1, bv2, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[1][3], av1, bv3, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[2][0], av2, bv0, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[2][1], av2, bv1, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[2][2], av2, bv2, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[2][3], av2, bv3, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[3][0], av3, bv0, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[3][1], av3, bv1, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[3][2], av3, bv2, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[3][3], av3, bv3, di * dimensions_per_value, depth_simd_dimensions);  \
                        }                                                                                              \
                        /* Compensated finalize: apply correction with precomputed sums */                             \
                        result_vec_type result_vector;                                                                 \
                        for (nk_size_t r = 0; r < register_row_count; ++r) {                                           \
                            compensated_finalize_fn(&acc[r][0], &acc[r][1], &acc[r][2], &acc[r][3], depth, a_sums[r],  \
                                                    &b_sum_vec, &result_vector);                                       \
                            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +      \
                                                                                               (tr + r) * c_stride);   \
                            store_fn(&result_vector, c_row + tc);                                                      \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_INLINE void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                        \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {        \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
        nk_size_t const total_packed_values = column_count * depth_padded;                                             \
        nk_##norm_value_type##_t const *b_norms = (nk_##norm_value_type##_t const *)(packed_data +                     \
                                                                                     total_packed_values);             \
        nk_##sum_value_type##_t const *b_sums = (nk_##sum_value_type##_t const *)(b_norms + column_count);             \
        nk_unused_(b_norms);                                                                                           \
        nk_size_t const row_block_size = 128;                                                                          \
        nk_size_t const column_block_size = 2048;                                                                      \
        nk_size_t const register_column_count = 8;                                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = depth_dimensions_aligned / dimensions_per_value;                               \
        nk_size_t const depth_step_values = depth_simd_dimensions / dimensions_per_value;                              \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride);                  \
            for (nk_size_t ci = 0; ci < column_count; ++ci) c_row[ci] = 0;                                             \
        }                                                                                                              \
        for (nk_size_t cb = 0; cb < column_count; cb += column_block_size) {                                           \
            nk_size_t ce = cb + column_block_size;                                                                     \
            if (ce > column_count) ce = column_count;                                                                  \
            for (nk_size_t rb2 = 0; rb2 < row_count; rb2 += row_block_size) {                                          \
                nk_size_t re2 = rb2 + row_block_size < row_count ? rb2 + row_block_size : row_count;                   \
                for (nk_size_t tc = cb; tc < ce; tc += register_column_count) {                                        \
                    nk_##packed_value_type##_t const *bp0 = packed_data + (tc + 0) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp1 = packed_data + (tc + 1) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp2 = packed_data + (tc + 2) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp3 = packed_data + (tc + 3) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp4 = packed_data + (tc + 4) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp5 = packed_data + (tc + 5) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp6 = packed_data + (tc + 6) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp7 = packed_data + (tc + 7) * depth_padded;                     \
                    result_vec_type b_sum_low, b_sum_high;                                                             \
                    load_sum_fn(b_sums + tc, &b_sum_low);                                                              \
                    load_sum_fn(b_sums + tc + 4, &b_sum_high);                                                         \
                    for (nk_size_t ri = rb2; ri < re2; ++ri) {                                                         \
                        state_type s0, s1, s2, s3, s4, s5, s6, s7;                                                     \
                        init_accumulator_fn(&s0), init_accumulator_fn(&s1), init_accumulator_fn(&s2),                  \
                            init_accumulator_fn(&s3), init_accumulator_fn(&s4), init_accumulator_fn(&s5),              \
                            init_accumulator_fn(&s6), init_accumulator_fn(&s7);                                        \
                        nk_##input_value_type##_t const *a_row =                                                       \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix + ri * a_stride);               \
                        nk_##sum_value_type##_t a_sum_val = compute_a_sum_fn(a_row, depth);                            \
                        vec_type av;                                                                                   \
                        vec_type bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7;                                               \
                        for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                          \
                            load_a_vec_fn(a_row + di, &av);                                                            \
                            load_b_vec_fn(bp0 + di, &bv0), load_b_vec_fn(bp1 + di, &bv1);                              \
                            load_b_vec_fn(bp2 + di, &bv2), load_b_vec_fn(bp3 + di, &bv3);                              \
                            load_b_vec_fn(bp4 + di, &bv4), load_b_vec_fn(bp5 + di, &bv5);                              \
                            load_b_vec_fn(bp6 + di, &bv6), load_b_vec_fn(bp7 + di, &bv7);                              \
                            inner_product_fn(&s0, av, bv0, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s1, av, bv1, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s2, av, bv2, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s3, av, bv3, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s4, av, bv4, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s5, av, bv5, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s6, av, bv6, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s7, av, bv7, di * dimensions_per_value, depth_simd_dimensions);          \
                        }                                                                                              \
                        result_vec_type rv;                                                                            \
                        nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +          \
                                                                                           ri * c_stride);             \
                        compensated_finalize_fn(&s0, &s1, &s2, &s3, depth, a_sum_val, &b_sum_low, &rv);                \
                        store_fn(&rv, c_row + tc);                                                                     \
                        compensated_finalize_fn(&s4, &s5, &s6, &s7, depth, a_sum_val, &b_sum_high, &rv);               \
                        store_fn(&rv, c_row + tc + 4);                                                                 \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##api_name##_packed_##input_type_name##_##isa_suffix(                                   \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,          \
        void *stream) {                                                                                                \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        if (header->capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;                                  \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
        nk_size_t const row_block_size = 128;                                                                          \
        nk_size_t const column_block_size = 2048;                                                                      \
        nk_size_t const register_row_count = 4;                                                                        \
        nk_size_t const register_column_count = 4;                                                                     \
        nk_unused_(register_column_count);                                                                             \
        if (column_count % 8 == 0 && column_count >= row_count * 2 && depth % depth_simd_dimensions == 0) {            \
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                                    \
                a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth, a_stride, c_stride);              \
            return nk_success_k;                                                                                       \
        }                                                                                                              \
        if (row_count % 4 == 0 && column_count % 4 == 0 && depth % depth_simd_dimensions == 0) {                       \
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(                                        \
                a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth, a_stride, c_stride);              \
            return nk_success_k;                                                                                       \
        }                                                                                                              \
        /* Generic fallback with partial loads and compensated finalize */                                             \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
        nk_size_t const total_packed_values = column_count * depth_padded;                                             \
        nk_##norm_value_type##_t const *b_norms = (nk_##norm_value_type##_t const *)(packed_data +                     \
                                                                                     total_packed_values);             \
        nk_##sum_value_type##_t const *b_sums = (nk_##sum_value_type##_t const *)(b_norms + column_count);             \
        nk_unused_(b_norms);                                                                                           \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = depth_dimensions_aligned / dimensions_per_value;                               \
        nk_size_t const depth_in_values = depth / dimensions_per_value;                                                \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        nk_size_t const depth_step_values = depth_simd_dimensions / dimensions_per_value;                              \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride);                  \
            for (nk_size_t ci = 0; ci < column_count; ++ci) c_row[ci] = 0;                                             \
        }                                                                                                              \
        for (nk_size_t cb = 0; cb < column_count; cb += column_block_size) {                                           \
            nk_size_t ce = cb + column_block_size;                                                                     \
            if (ce > column_count) ce = column_count;                                                                  \
            for (nk_size_t rb = 0; rb < row_count; rb += row_block_size) {                                             \
                nk_size_t re = rb + row_block_size;                                                                    \
                if (re > row_count) re = row_count;                                                                    \
                for (nk_size_t tc = cb; tc < ce; tc += register_column_count) {                                        \
                    nk_size_t tile_col_count = register_column_count;                                                  \
                    if (tc + tile_col_count > ce) tile_col_count = ce - tc;                                            \
                    nk_##packed_value_type##_t const *bdp0 = packed_data + (tc + 0) * depth_padded;                    \
                    nk_##packed_value_type##_t const *bdp1 = (tile_col_count > 1)                                      \
                                                                 ? packed_data + (tc + 1) * depth_padded               \
                                                                 : bdp0;                                               \
                    nk_##packed_value_type##_t const *bdp2 = (tile_col_count > 2)                                      \
                                                                 ? packed_data + (tc + 2) * depth_padded               \
                                                                 : bdp0;                                               \
                    nk_##packed_value_type##_t const *bdp3 = (tile_col_count > 3)                                      \
                                                                 ? packed_data + (tc + 3) * depth_padded               \
                                                                 : bdp0;                                               \
                    result_vec_type b_sum_vec;                                                                         \
                    partial_load_sum_fn(b_sums + tc, &b_sum_vec, tile_col_count);                                      \
                    for (nk_size_t tr = rb; tr < re; tr += register_row_count) {                                       \
                        nk_size_t tile_row_count = register_row_count;                                                 \
                        if (tr + tile_row_count > re) tile_row_count = re - tr;                                        \
                        state_type acc[4][4];                                                                          \
                        for (nk_size_t rr = 0; rr < tile_row_count; ++rr) {                                            \
                            init_accumulator_fn(&acc[rr][0]);                                                          \
                            init_accumulator_fn(&acc[rr][1]);                                                          \
                            init_accumulator_fn(&acc[rr][2]);                                                          \
                            init_accumulator_fn(&acc[rr][3]);                                                          \
                        }                                                                                              \
                        nk_##input_value_type##_t const *arp0 =                                                        \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix + (tr + 0) * a_stride);         \
                        nk_##input_value_type##_t const *arp1 =                                                        \
                            (tile_row_count > 1)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix + (tr + 1) * a_stride)    \
                                : arp0;                                                                                \
                        nk_##input_value_type##_t const *arp2 =                                                        \
                            (tile_row_count > 2)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix + (tr + 2) * a_stride)    \
                                : arp0;                                                                                \
                        nk_##input_value_type##_t const *arp3 =                                                        \
                            (tile_row_count > 3)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix + (tr + 3) * a_stride)    \
                                : arp0;                                                                                \
                        nk_##sum_value_type##_t a_sums[4];                                                             \
                        a_sums[0] = compute_a_sum_fn(arp0, depth);                                                     \
                        a_sums[1] = (tile_row_count > 1) ? compute_a_sum_fn(arp1, depth) : 0;                          \
                        a_sums[2] = (tile_row_count > 2) ? compute_a_sum_fn(arp2, depth) : 0;                          \
                        a_sums[3] = (tile_row_count > 3) ? compute_a_sum_fn(arp3, depth) : 0;                          \
                        vec_type av0, av1, av2, av3, bv0, bv1, bv2, bv3;                                               \
                        for (nk_size_t k = 0; k < aligned_depth; k += depth_step_values) {                             \
                            load_a_vec_fn(arp0 + k, &av0);                                                             \
                            load_a_vec_fn(arp1 + k, &av1);                                                             \
                            load_a_vec_fn(arp2 + k, &av2);                                                             \
                            load_a_vec_fn(arp3 + k, &av3);                                                             \
                            load_b_vec_fn(bdp0 + k, &bv0);                                                             \
                            load_b_vec_fn(bdp1 + k, &bv1);                                                             \
                            load_b_vec_fn(bdp2 + k, &bv2);                                                             \
                            load_b_vec_fn(bdp3 + k, &bv3);                                                             \
                            inner_product_fn(&acc[0][0], av0, bv0, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[0][1], av0, bv1, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[0][2], av0, bv2, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[0][3], av0, bv3, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[1][0], av1, bv0, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[1][1], av1, bv1, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[1][2], av1, bv2, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[1][3], av1, bv3, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[2][0], av2, bv0, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[2][1], av2, bv1, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[2][2], av2, bv2, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[2][3], av2, bv3, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[3][0], av3, bv0, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[3][1], av3, bv1, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[3][2], av3, bv2, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[3][3], av3, bv3, k * dimensions_per_value, depth_simd_dimensions);   \
                        }                                                                                              \
                        if (remainder_depth > 0) {                                                                     \
                            partial_load_a_vec_fn(arp0 + aligned_depth, &av0, remainder_dimensions);                   \
                            partial_load_a_vec_fn(arp1 + aligned_depth, &av1, remainder_dimensions);                   \
                            partial_load_a_vec_fn(arp2 + aligned_depth, &av2, remainder_dimensions);                   \
                            partial_load_a_vec_fn(arp3 + aligned_depth, &av3, remainder_dimensions);                   \
                            partial_load_b_vec_fn(bdp0 + aligned_depth, &bv0, remainder_dimensions);                   \
                            partial_load_b_vec_fn(bdp1 + aligned_depth, &bv1, remainder_dimensions);                   \
                            partial_load_b_vec_fn(bdp2 + aligned_depth, &bv2, remainder_dimensions);                   \
                            partial_load_b_vec_fn(bdp3 + aligned_depth, &bv3, remainder_dimensions);                   \
                            inner_product_fn(&acc[0][0], av0, bv0, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[0][1], av0, bv1, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[0][2], av0, bv2, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[0][3], av0, bv3, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[1][0], av1, bv0, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[1][1], av1, bv1, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[1][2], av1, bv2, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[1][3], av1, bv3, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[2][0], av2, bv0, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[2][1], av2, bv1, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[2][2], av2, bv2, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[2][3], av2, bv3, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[3][0], av3, bv0, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[3][1], av3, bv1, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[3][2], av3, bv2, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[3][3], av3, bv3, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                        }                                                                                              \
                        for (nk_size_t rr = 0; rr < tile_row_count; ++rr) {                                            \
                            result_vec_type rv;                                                                        \
                            compensated_finalize_fn(&acc[rr][0], &acc[rr][1], &acc[rr][2], &acc[rr][3], depth,         \
                                                    a_sums[rr], &b_sum_vec, &rv);                                      \
                            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +      \
                                                                                               (tr + rr) * c_stride);  \
                            partial_store_fn(&rv, c_row + tc, tile_col_count);                                         \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }

/**
 *  @brief Generates compensated symmetric Gram matrix: C = A × Aᵀ with inline correction.
 *
 *  Like nk_define_cross_symmetric_ but the finalize function receives precomputed sums. For
 *  symmetric computation, both row and column vectors come from the same matrix A, so A sums serve
 *  as both row and column sums.
 *
 *  The off-diagonal helper uses 4×4 tiling, matching nk_define_cross_symmetric_, with progressive
 *  sum accumulation: SAD runs on port 5 alongside DPBUSD on ports 0+1 for zero throughput overhead
 *  on Alder Lake and Ice Lake.
 */
#define nk_define_cross_compensated_symmetric_(                                                                        \
    api_name, input_type_name, isa_suffix, input_value_type, result_value_type, sum_value_type, norm_value_type,       \
    vec_type, state_type, result_vec_type, init_accumulator_fn, load_vec_fn, partial_load_vec_fn, inner_product_fn,    \
    compensated_finalize_fn, store_fn, partial_store_fn, load_sum_fn, partial_load_sum_fn, sum_state_type,             \
    init_sum_fn, update_sum_fn, finalize_sum_fn, depth_simd_dimensions, dimensions_per_value)                          \
    NUMKONG_INLINE void nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                        \
        nk_##input_value_type##_t const **vector_base_ptrs, nk_size_t i_macro, nk_size_t macro_size,                   \
        nk_size_t aligned_depth, nk_size_t remainder_depth, nk_size_t remainder_dimensions,                            \
        nk_size_t depth_step_values, nk_size_t dimensions_per_value_runtime, nk_##result_value_type##_t *result,       \
        nk_size_t result_stride_values, nk_size_t finalizer_batch_size, nk_size_t depth) {                             \
        nk_unused_(finalizer_batch_size);                                                                              \
        nk_unused_(dimensions_per_value_runtime);                                                                      \
        /* Compute sums via stateful helpers — a separate loop is fine,                                              \
         * since the diagonal is ~1.6% of work */                                                                      \
        nk_size_t padded_depth_dimensions = aligned_depth * dimensions_per_value +                                     \
                                            (remainder_depth > 0 ? depth_simd_dimensions : 0);                         \
        nk_##sum_value_type##_t precomputed_sums[32];                                                                  \
        for (nk_size_t s = 0; s < macro_size; s++) {                                                                   \
            sum_state_type ss;                                                                                         \
            init_sum_fn(&ss);                                                                                          \
            for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                                      \
                vec_type v;                                                                                            \
                load_vec_fn(vector_base_ptrs[s] + di, &v);                                                             \
                update_sum_fn(&ss, v);                                                                                 \
            }                                                                                                          \
            if (remainder_depth > 0) {                                                                                 \
                vec_type v;                                                                                            \
                partial_load_vec_fn(vector_base_ptrs[s] + aligned_depth, &v, remainder_dimensions);                    \
                update_sum_fn(&ss, v);                                                                                 \
            }                                                                                                          \
            precomputed_sums[s] = finalize_sum_fn(&ss, padded_depth_dimensions);                                       \
        }                                                                                                              \
        for (nk_size_t tile_row_start = 0; tile_row_start < macro_size; tile_row_start += 4) {                         \
            for (nk_size_t tile_col_start = tile_row_start; tile_col_start < macro_size; tile_col_start += 4) {        \
                nk_size_t tile_rows = (tile_row_start + 4 <= macro_size) ? 4 : (macro_size - tile_row_start);          \
                nk_size_t tile_columns = (tile_col_start + 4 <= macro_size) ? 4 : (macro_size - tile_col_start);       \
                int is_diag = (tile_row_start == tile_col_start);                                                      \
                nk_align_(64) state_type accumulators[4][7];                                                           \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    nk_size_t init_start = is_diag ? row : 0;                                                          \
                    nk_size_t init_end = is_diag ? (row + 4) : 4;                                                      \
                    for (nk_size_t col = init_start; col < init_end; col++) {                                          \
                        init_accumulator_fn(&accumulators[row][col]);                                                  \
                    }                                                                                                  \
                }                                                                                                      \
                nk_##input_value_type##_t const *row_ptrs[4], *col_ptrs[4];                                            \
                row_ptrs[0] = vector_base_ptrs[tile_row_start + 0];                                                    \
                row_ptrs[1] = (tile_rows > 1) ? vector_base_ptrs[tile_row_start + 1] : row_ptrs[0];                    \
                row_ptrs[2] = (tile_rows > 2) ? vector_base_ptrs[tile_row_start + 2] : row_ptrs[0];                    \
                row_ptrs[3] = (tile_rows > 3) ? vector_base_ptrs[tile_row_start + 3] : row_ptrs[0];                    \
                if (is_diag) {                                                                                         \
                    col_ptrs[0] = row_ptrs[0];                                                                         \
                    col_ptrs[1] = row_ptrs[1];                                                                         \
                    col_ptrs[2] = row_ptrs[2];                                                                         \
                    col_ptrs[3] = row_ptrs[3];                                                                         \
                }                                                                                                      \
                else {                                                                                                 \
                    col_ptrs[0] = vector_base_ptrs[tile_col_start + 0];                                                \
                    col_ptrs[1] = (tile_columns > 1) ? vector_base_ptrs[tile_col_start + 1] : col_ptrs[0];             \
                    col_ptrs[2] = (tile_columns > 2) ? vector_base_ptrs[tile_col_start + 2] : col_ptrs[0];             \
                    col_ptrs[3] = (tile_columns > 3) ? vector_base_ptrs[tile_col_start + 3] : col_ptrs[0];             \
                }                                                                                                      \
                vec_type row_vecs[4], col_vecs[4];                                                                     \
                for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                                  \
                    load_vec_fn(row_ptrs[0] + di, &row_vecs[0]);                                                       \
                    load_vec_fn(row_ptrs[1] + di, &row_vecs[1]);                                                       \
                    load_vec_fn(row_ptrs[2] + di, &row_vecs[2]);                                                       \
                    load_vec_fn(row_ptrs[3] + di, &row_vecs[3]);                                                       \
                    if (!is_diag) {                                                                                    \
                        load_vec_fn(col_ptrs[0] + di, &col_vecs[0]);                                                   \
                        load_vec_fn(col_ptrs[1] + di, &col_vecs[1]);                                                   \
                        load_vec_fn(col_ptrs[2] + di, &col_vecs[2]);                                                   \
                        load_vec_fn(col_ptrs[3] + di, &col_vecs[3]);                                                   \
                    }                                                                                                  \
                    else {                                                                                             \
                        col_vecs[0] = row_vecs[0];                                                                     \
                        col_vecs[1] = row_vecs[1];                                                                     \
                        col_vecs[2] = row_vecs[2];                                                                     \
                        col_vecs[3] = row_vecs[3];                                                                     \
                    }                                                                                                  \
                    if (tile_rows == 4 && tile_columns == 4 && is_diag) {                                              \
                        /* Upper triangle: 10 FMAs */                                                                  \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                    }                                                                                                  \
                    else if (tile_rows == 4 && tile_columns == 4) {                                                    \
                        /* Full 4×4 rectangle: 16 FMAs */                                                              \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][0], row_vecs[1], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][0], row_vecs[2], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][1], row_vecs[2], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][0], row_vecs[3], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][1], row_vecs[3], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][2], row_vecs[3], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                    }                                                                                                  \
                    else {                                                                                             \
                        for (nk_size_t row = 0; row < tile_rows; row++) {                                              \
                            nk_size_t col_start = is_diag ? row : 0;                                                   \
                            nk_size_t col_end = is_diag ? (row < 4 ? 4 : tile_columns) : tile_columns;                 \
                            for (nk_size_t col = col_start; col < col_end; col++)                                      \
                                inner_product_fn(&accumulators[row][col], row_vecs[row], col_vecs[col],                \
                                                 di * dimensions_per_value, depth_simd_dimensions);                    \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                if (remainder_depth > 0) {                                                                             \
                    partial_load_vec_fn(row_ptrs[0] + aligned_depth, &row_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[1] + aligned_depth, &row_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[2] + aligned_depth, &row_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[3] + aligned_depth, &row_vecs[3], remainder_dimensions);              \
                    if (!is_diag) {                                                                                    \
                        partial_load_vec_fn(col_ptrs[0] + aligned_depth, &col_vecs[0], remainder_dimensions);          \
                        partial_load_vec_fn(col_ptrs[1] + aligned_depth, &col_vecs[1], remainder_dimensions);          \
                        partial_load_vec_fn(col_ptrs[2] + aligned_depth, &col_vecs[2], remainder_dimensions);          \
                        partial_load_vec_fn(col_ptrs[3] + aligned_depth, &col_vecs[3], remainder_dimensions);          \
                    }                                                                                                  \
                    else {                                                                                             \
                        col_vecs[0] = row_vecs[0];                                                                     \
                        col_vecs[1] = row_vecs[1];                                                                     \
                        col_vecs[2] = row_vecs[2];                                                                     \
                        col_vecs[3] = row_vecs[3];                                                                     \
                    }                                                                                                  \
                    if (tile_rows == 4 && tile_columns == 4 && is_diag) {                                              \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                    }                                                                                                  \
                    else if (tile_rows == 4 && tile_columns == 4) {                                                    \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][0], row_vecs[1], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][0], row_vecs[2], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][1], row_vecs[2], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][0], row_vecs[3], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][1], row_vecs[3], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][2], row_vecs[3], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                    }                                                                                                  \
                    else {                                                                                             \
                        for (nk_size_t row = 0; row < tile_rows; row++) {                                              \
                            nk_size_t col_start = is_diag ? row : 0;                                                   \
                            nk_size_t col_end = is_diag ? (row < 4 ? 4 : tile_columns) : tile_columns;                 \
                            for (nk_size_t col = col_start; col < col_end; col++)                                      \
                                inner_product_fn(&accumulators[row][col], row_vecs[row], col_vecs[col],                \
                                                 aligned_depth * dimensions_per_value, remainder_dimensions);          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                nk_##sum_value_type##_t row_sums[4] = {0}, col_sums_arr[4] = {0};                                      \
                for (nk_size_t r = 0; r < tile_rows; r++) row_sums[r] = precomputed_sums[tile_row_start + r];          \
                for (nk_size_t c = 0; c < tile_columns; c++)                                                           \
                    col_sums_arr[c] = is_diag ? row_sums[c] : precomputed_sums[tile_col_start + c];                    \
                /* Build column sums as SIMD vector — for diagonal tiles, shift per row */                             \
                result_vec_type col_sum_vec;                                                                           \
                if (!is_diag) partial_load_sum_fn(col_sums_arr, &col_sum_vec, tile_columns);                           \
                /* Finalize with compensation */                                                                       \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    if (is_diag) {                                                                                     \
                        nk_##sum_value_type##_t shifted[4] = {0};                                                      \
                        for (nk_size_t c = 0; c < 4 && (row + c) < tile_columns; c++)                                  \
                            shifted[c] = col_sums_arr[row + c];                                                        \
                        partial_load_sum_fn(shifted, &col_sum_vec, 4);                                                 \
                    }                                                                                                  \
                    result_vec_type rv;                                                                                \
                    compensated_finalize_fn(                                                                           \
                        &accumulators[row][is_diag ? row : 0], &accumulators[row][(is_diag ? row : 0) + 1],            \
                        &accumulators[row][(is_diag ? row : 0) + 2], &accumulators[row][(is_diag ? row : 0) + 3],      \
                        depth, row_sums[row], &col_sum_vec, &rv);                                                      \
                    nk_size_t global_row = i_macro + tile_row_start + row;                                             \
                    nk_size_t global_col_start = i_macro + tile_col_start + (is_diag ? row : 0);                       \
                    nk_size_t store_count = is_diag ? (tile_columns - row) : tile_columns;                             \
                    nk_##result_value_type##_t *dest = result + global_row * result_stride_values + global_col_start;  \
                    partial_store_fn(&rv, dest, store_count);                                                          \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    /* Off-diagonal helper: 4×4 tiling, inline sums — 16 FMAs + up to 8 SADs per depth step */                         \
    NUMKONG_INLINE void nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_(                     \
        nk_##input_value_type##_t const **row_ptrs_macro, nk_##input_value_type##_t const **col_ptrs_macro,            \
        nk_size_t i_macro, nk_size_t j_macro, nk_size_t macro_i_size, nk_size_t macro_j_size, nk_size_t aligned_depth, \
        nk_size_t remainder_depth, nk_size_t remainder_dimensions, nk_size_t depth_step_values,                        \
        nk_size_t dimensions_per_value_runtime, nk_##result_value_type##_t *result, nk_size_t result_stride_values,    \
        nk_size_t finalizer_batch_size, nk_size_t depth) {                                                             \
        nk_unused_(finalizer_batch_size);                                                                              \
        nk_unused_(dimensions_per_value_runtime);                                                                      \
        nk_size_t padded_depth_dimensions = aligned_depth * dimensions_per_value +                                     \
                                            (remainder_depth > 0 ? depth_simd_dimensions : 0);                         \
        /* Sum caches for this macro-tile pair — computed once, reused across tiles */                                 \
        nk_##sum_value_type##_t row_sums[32], col_sums[32];                                                            \
        for (nk_size_t tile_row_start = 0; tile_row_start < macro_i_size; tile_row_start += 4) {                       \
            for (nk_size_t tile_col_start = 0; tile_col_start < macro_j_size; tile_col_start += 4) {                   \
                nk_size_t tile_rows = (tile_row_start + 4 <= macro_i_size) ? 4 : (macro_i_size - tile_row_start);      \
                nk_size_t tile_columns = (tile_col_start + 4 <= macro_j_size) ? 4 : (macro_j_size - tile_col_start);   \
                /* Determine if this tile should compute sums — predictable branches */                                \
                int compute_row_sums_flag = (tile_col_start == 0);                                                     \
                int compute_col_sums_flag = (tile_row_start == 0);                                                     \
                /* Initialize 4×4 dot accumulators */                                                                  \
                nk_align_(64) state_type accumulators[4][4];                                                           \
                for (nk_size_t row = 0; row < tile_rows; row++)                                                        \
                    for (nk_size_t col = 0; col < 4; col++) init_accumulator_fn(&accumulators[row][col]);              \
                /* Initialize sum accumulators (only when needed) */                                                   \
                sum_state_type rsum[4], csum[4];                                                                       \
                if (compute_row_sums_flag)                                                                             \
                    for (nk_size_t r = 0; r < tile_rows; r++) init_sum_fn(&rsum[r]);                                   \
                if (compute_col_sums_flag)                                                                             \
                    for (nk_size_t c = 0; c < tile_columns; c++) init_sum_fn(&csum[c]);                                \
                /* Setup pointers (hoist outside depth loop) */                                                        \
                nk_##input_value_type##_t const *row_ptrs[4], *col_ptrs[4];                                            \
                row_ptrs[0] = row_ptrs_macro[tile_row_start + 0];                                                      \
                row_ptrs[1] = (tile_rows > 1) ? row_ptrs_macro[tile_row_start + 1] : row_ptrs[0];                      \
                row_ptrs[2] = (tile_rows > 2) ? row_ptrs_macro[tile_row_start + 2] : row_ptrs[0];                      \
                row_ptrs[3] = (tile_rows > 3) ? row_ptrs_macro[tile_row_start + 3] : row_ptrs[0];                      \
                col_ptrs[0] = col_ptrs_macro[tile_col_start + 0];                                                      \
                col_ptrs[1] = (tile_columns > 1) ? col_ptrs_macro[tile_col_start + 1] : col_ptrs[0];                   \
                col_ptrs[2] = (tile_columns > 2) ? col_ptrs_macro[tile_col_start + 2] : col_ptrs[0];                   \
                col_ptrs[3] = (tile_columns > 3) ? col_ptrs_macro[tile_col_start + 3] : col_ptrs[0];                   \
                /* Depth loop — innermost, 16 FMAs + up to 8 SADs per iteration */                                     \
                vec_type row_vecs[4], col_vecs[4];                                                                     \
                for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                                  \
                    load_vec_fn(row_ptrs[0] + di, &row_vecs[0]);                                                       \
                    load_vec_fn(row_ptrs[1] + di, &row_vecs[1]);                                                       \
                    load_vec_fn(row_ptrs[2] + di, &row_vecs[2]);                                                       \
                    load_vec_fn(row_ptrs[3] + di, &row_vecs[3]);                                                       \
                    load_vec_fn(col_ptrs[0] + di, &col_vecs[0]);                                                       \
                    load_vec_fn(col_ptrs[1] + di, &col_vecs[1]);                                                       \
                    load_vec_fn(col_ptrs[2] + di, &col_vecs[2]);                                                       \
                    load_vec_fn(col_ptrs[3] + di, &col_vecs[3]);                                                       \
                    nk_size_t vector_offset = di * dimensions_per_value;                                               \
                    if (tile_rows == 4 && tile_columns == 4) {                                                         \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][0], row_vecs[1], col_vecs[0], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][0], row_vecs[2], col_vecs[0], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][1], row_vecs[2], col_vecs[1], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][0], row_vecs[3], col_vecs[0], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][1], row_vecs[3], col_vecs[1], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][2], row_vecs[3], col_vecs[2], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                    }                                                                                                  \
                    else {                                                                                             \
                        for (nk_size_t row = 0; row < tile_rows; row++)                                                \
                            for (nk_size_t col = 0; col < tile_columns; col++)                                         \
                                inner_product_fn(&accumulators[row][col], row_vecs[row], col_vecs[col], vector_offset, \
                                                 depth_simd_dimensions);                                               \
                    }                                                                                                  \
                    /* Progressive sum accumulation, with SADs on port 5 running                                       \
                     * in parallel with DPBUSD on ports 0+1 */                                                         \
                    if (compute_row_sums_flag) {                                                                       \
                        update_sum_fn(&rsum[0], row_vecs[0]);                                                          \
                        if (tile_rows > 1) update_sum_fn(&rsum[1], row_vecs[1]);                                       \
                        if (tile_rows > 2) update_sum_fn(&rsum[2], row_vecs[2]);                                       \
                        if (tile_rows > 3) update_sum_fn(&rsum[3], row_vecs[3]);                                       \
                    }                                                                                                  \
                    if (compute_col_sums_flag) {                                                                       \
                        update_sum_fn(&csum[0], col_vecs[0]);                                                          \
                        if (tile_columns > 1) update_sum_fn(&csum[1], col_vecs[1]);                                    \
                        if (tile_columns > 2) update_sum_fn(&csum[2], col_vecs[2]);                                    \
                        if (tile_columns > 3) update_sum_fn(&csum[3], col_vecs[3]);                                    \
                    }                                                                                                  \
                }                                                                                                      \
                /* Handle remainder depth */                                                                           \
                if (remainder_depth > 0) {                                                                             \
                    partial_load_vec_fn(row_ptrs[0] + aligned_depth, &row_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[1] + aligned_depth, &row_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[2] + aligned_depth, &row_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[3] + aligned_depth, &row_vecs[3], remainder_dimensions);              \
                    partial_load_vec_fn(col_ptrs[0] + aligned_depth, &col_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(col_ptrs[1] + aligned_depth, &col_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(col_ptrs[2] + aligned_depth, &col_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(col_ptrs[3] + aligned_depth, &col_vecs[3], remainder_dimensions);              \
                    nk_size_t vector_offset = aligned_depth * dimensions_per_value;                                    \
                    for (nk_size_t row = 0; row < tile_rows; row++)                                                    \
                        for (nk_size_t col = 0; col < tile_columns; col++)                                             \
                            inner_product_fn(&accumulators[row][col], row_vecs[row], col_vecs[col], vector_offset,     \
                                             remainder_dimensions);                                                    \
                    if (compute_row_sums_flag) {                                                                       \
                        update_sum_fn(&rsum[0], row_vecs[0]);                                                          \
                        if (tile_rows > 1) update_sum_fn(&rsum[1], row_vecs[1]);                                       \
                        if (tile_rows > 2) update_sum_fn(&rsum[2], row_vecs[2]);                                       \
                        if (tile_rows > 3) update_sum_fn(&rsum[3], row_vecs[3]);                                       \
                    }                                                                                                  \
                    if (compute_col_sums_flag) {                                                                       \
                        update_sum_fn(&csum[0], col_vecs[0]);                                                          \
                        if (tile_columns > 1) update_sum_fn(&csum[1], col_vecs[1]);                                    \
                        if (tile_columns > 2) update_sum_fn(&csum[2], col_vecs[2]);                                    \
                        if (tile_columns > 3) update_sum_fn(&csum[3], col_vecs[3]);                                    \
                    }                                                                                                  \
                }                                                                                                      \
                /* Finalize and cache sums */                                                                          \
                if (compute_row_sums_flag)                                                                             \
                    for (nk_size_t r = 0; r < tile_rows; r++)                                                          \
                        row_sums[tile_row_start + r] = finalize_sum_fn(&rsum[r], padded_depth_dimensions);             \
                if (compute_col_sums_flag)                                                                             \
                    for (nk_size_t c = 0; c < tile_columns; c++)                                                       \
                        col_sums[tile_col_start + c] = finalize_sum_fn(&csum[c], padded_depth_dimensions);             \
                /* Build col_sum SIMD vector once (constant across rows) */                                            \
                nk_##sum_value_type##_t cs_arr[4] = {0};                                                               \
                for (nk_size_t c = 0; c < tile_columns; c++) cs_arr[c] = col_sums[tile_col_start + c];                 \
                result_vec_type cs_vec;                                                                                \
                partial_load_sum_fn(cs_arr, &cs_vec, tile_columns);                                                    \
                /* Compensated finalize + store */                                                                     \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    result_vec_type rv;                                                                                \
                    compensated_finalize_fn(&accumulators[row][0], &accumulators[row][1], &accumulators[row][2],       \
                                            &accumulators[row][3], depth, row_sums[tile_row_start + row], &cs_vec,     \
                                            &rv);                                                                      \
                    nk_##result_value_type##_t *dest = result +                                                        \
                                                       (i_macro + tile_row_start + row) * result_stride_values +       \
                                                       (j_macro + tile_col_start);                                     \
                    partial_store_fn(&rv, dest, tile_columns);                                                         \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##api_name##_symmetric_##input_type_name##_##isa_suffix(                                \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,          \
        nk_##result_value_type##_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,         \
        void *stream) {                                                                                                \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        nk_assert_(stride % sizeof(nk_##input_value_type##_t) == 0 &&                                                  \
                   stride >=                                                                                           \
                       nk_size_divide_round_up_(depth, dimensions_per_value) * sizeof(nk_##input_value_type##_t));     \
        nk_size_t const macro_tile_size = 32;                                                                          \
        nk_size_t const row_block_size = 128;     /* L2 cache blocking */                                              \
        nk_size_t const column_block_size = 2048; /* L3 cache blocking */                                              \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = depth_dimensions_aligned / dimensions_per_value;                               \
        nk_size_t const depth_in_values = depth / dimensions_per_value;                                                \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        nk_size_t const depth_step = depth_simd_dimensions / dimensions_per_value;                                     \
        nk_size_t const result_stride_values = result_stride / sizeof(nk_##result_value_type##_t);                     \
        nk_size_t const row_end = (row_start + row_count < vectors_count) ? (row_start + row_count) : vectors_count;   \
                                                                                                                       \
        /* Process the upper triangle with L3/L2/L1 blocking: column                                                   \
         * blocks → row blocks → 32×32 macro-tiles */                                                             \
        for (nk_size_t j_block = 0; j_block < vectors_count; j_block += column_block_size) {                           \
            nk_size_t j_block_end = (j_block + column_block_size < vectors_count) ? j_block + column_block_size        \
                                                                                  : vectors_count;                     \
                                                                                                                       \
            for (nk_size_t i_block = row_start; i_block < row_end; i_block += row_block_size) {                        \
                nk_size_t i_block_end = (i_block + row_block_size < row_end) ? i_block + row_block_size : row_end;     \
                                                                                                                       \
                /* Skip blocks entirely below diagonal. Blocks fully above the diagonal are still                      \
                 * part of the upper triangle and must be computed. */                                                 \
                if (i_block >= j_block_end) continue;                                                                  \
                                                                                                                       \
                for (nk_size_t i_macro = i_block, macro_i_size; i_macro < i_block_end; i_macro += macro_i_size) {      \
                    /* Row tiles stop at column-block edges, so only `j_macro == i_macro` meets the diagonal */        \
                    nk_size_t const column_edge = (i_macro / column_block_size + 1) * column_block_size;               \
                    macro_i_size = nk_min_of_two(macro_tile_size, nk_min_of_two(i_block_end, column_edge) - i_macro);  \
                    nk_size_t j_start = (i_macro > j_block) ? i_macro : j_block;                                       \
                    for (nk_size_t j_macro = j_start; j_macro < j_block_end; j_macro += macro_tile_size) {             \
                        nk_size_t macro_j_size = (j_macro + macro_tile_size <= j_block_end) ? macro_tile_size          \
                                                                                            : (j_block_end - j_macro); \
                                                                                                                       \
                        nk_##input_value_type##_t const *vec_ptrs_i[32];                                               \
                        nk_##input_value_type##_t const *vec_ptrs_j[32];                                               \
                        for (nk_size_t k = 0; k < macro_i_size; k++)                                                   \
                            vec_ptrs_i[k] = (nk_##input_value_type##_t const *)((char const *)vectors +                \
                                                                                (i_macro + k) * stride);               \
                        for (nk_size_t k = macro_i_size; k < 32; k++) vec_ptrs_i[k] = vec_ptrs_i[0];                   \
                        for (nk_size_t k = 0; k < macro_j_size; k++)                                                   \
                            vec_ptrs_j[k] = (nk_##input_value_type##_t const *)((char const *)vectors +                \
                                                                                (j_macro + k) * stride);               \
                        for (nk_size_t k = macro_j_size; k < 32; k++) vec_ptrs_j[k] = vec_ptrs_j[0];                   \
                                                                                                                       \
                        /* A diagonal tile is a triangle, plus a rectangle when the row range ends early */            \
                        nk_size_t const skipped_columns = (i_macro == j_macro) ? macro_i_size : 0;                     \
                        if (i_macro == j_macro)                                                                        \
                            nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                    \
                                vec_ptrs_i, i_macro, macro_i_size, aligned_depth, remainder_depth,                     \
                                remainder_dimensions, depth_step, dimensions_per_value, result, result_stride_values,  \
                                4, depth);                                                                             \
                        if (skipped_columns < macro_j_size)                                                            \
                            nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_(                 \
                                vec_ptrs_i, vec_ptrs_j + skipped_columns, i_macro, j_macro + skipped_columns,          \
                                macro_i_size, macro_j_size - skipped_columns, aligned_depth, remainder_depth,          \
                                remainder_dimensions, depth_step, dimensions_per_value, result, result_stride_values,  \
                                4, depth);                                                                             \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }

/**
 *  @brief Generates optimized symmetric Gram matrix computation: C = A × Aᵀ (upper triangle only).
 *
 *  This macro creates a complete symmetric cross-product implementation with two specialized
 *  internal helper functions (diagonal and off-diagonal) that are called by a public wrapper.
 *  Symmetric computation exploits the property that C[i,j] = C[j,i], computing only the upper
 *  triangle and avoiding redundant computation and storage.
 *
 *  Mathematically, for each pair @b (i,j) where i ≤ j, it computes:
 *
 *  @verbatim
 *  C[i,j] = operation(A[i,:], A[j,:]) where operation can be dot product, Hamming distance,
 *  Jaccard similarity, etc.
 *  @endverbatim
 *
 *  The architecture is a three-level tiling hierarchy:
 *
 *  1. @b 32×32 @b macro-tiles (outermost): Divides the upper triangle into 32×32 blocks
 *     - Rationale: Fits well in L1 cache (32 vectors × depth × value_size)
 *     - Enables diagonal vs off-diagonal specialization
 *     - Amortizes vector loads across all depth iterations
 *     - Pre-loads and upcasts all 32 vectors once per depth iteration (not per FMA)
 *
 *  2. @b 4×4 @b register @b tiles (middle): Within each macro-tile, process 4×4 sub-blocks
 *     - Rationale: Maximizes register reuse (4 A vectors × 4 A vectors = 16 accumulators)
 *     - Enables full FMA unrolling (16 FMAs for off-diagonal, 10 for diagonal)
 *     - Balances register pressure with instruction-level parallelism
 *
 *  3. @b Depth @b loop (innermost): For each depth chunk, accumulate outer products
 *     - Depth loop is inside macro-tile, outside register tiles
 *     - Type conversion (e.g., bf16 → f32) happens at macro-tile level (once per vector)
 *
 *  Diagonal and off-diagonal macro-tiles are optimized separately:
 *
 *  - @b Diagonal @b macro-tiles (i_macro == j_macro): Computes C[i:i+32, i:i+32]
 *    - Loads 32 vectors once (50% load reduction vs off-diagonal)
 *    - Computes upper triangle only within the tile (10 FMAs per 4×4 block)
 *    - Uses nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_ helper
 *
 *  - @b Off-diagonal @b macro-tiles (i_macro < j_macro): Computes C[i:i+32, j:j+32]
 *    - Loads vec_i[32] + vec_j[32] (full 64 vectors for two sets)
 *    - Computes full 32×32 block (16 FMAs per 4×4 block)
 *    - Uses nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_ helper
 *
 *  Choose between the symmetric and packed variants by whether both sides are the same matrix:
 *
 *  - Use symmetric (this macro) when: A is the same matrix for both sides (C = A × Aᵀ)
 *    - Saves 50% computation and storage (upper triangle only)
 *    - Automatic diagonal optimization (50% fewer loads on diagonal tiles)
 *    - Ideal for: distance matrices, correlation matrices, Gram matrices
 *
 *  - Use packed variant when: Computing C = A × Bᵀ where A ≠ B
 *    - Full matrix computation (no symmetry to exploit)
 *    - B can be pre-packed for cache efficiency
 *
 *  This macro generates two NUMKONG_INLINE helpers and one NUMKONG_API wrapper:
 *  1. nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_
 *  2. nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_
 *  3. nk_##api_name##_symmetric_##input_type_name##_##isa_suffix
 *
 *  @param[in] api_name Operation family, dots/hammings/jaccards, for codegen namespace.
 *  @param[in] input_type_name Type identifier for codegen, e.g. f32, bf16, i8, u1.
 *  @param[in] isa_suffix ISA backend identifier, e.g. serial, haswell, neon, sve, icelake.
 *  @param[in] input_value_type C type of input matrix values, e.g. f32, bf16, i8, u1x8.
 *  @param[in] result_value_type C type of output matrix values, e.g. f32, u32, f64.
 *  @param[in] vec_type SIMD vector type for input vectors, e.g. __m256, nk_b256_vec_t.
 *  @param[in] state_type Accumulator state type, often vec_type or wider, e.g. __m256 or __m512.
 *  @param[in] result_vec_type Reduction-result SIMD vector type, e.g. __m128 for 4 f32 results.
 *  @param[in] init_accumulator_fn Initialize accumulator: void fn(state_type*).
 *  @param[in] load_vec_fn Full vector load: vec_type fn(input_value_type const*, nk_size_t offset).
 *  @param[in] partial_load_vec_fn Partial vector load for remainder.
 *  @param[in] inner_product_fn Inner product accumulate.
 *  @param[in] reduce_accumulators_fn Reduce 4 accumulators.
 *  @param[in] partial_store_fn Partial store for results.
 *  @param[in] depth_simd_dimensions SIMD vector width in logical dimensions, e.g. 8 for f32 on
 *      AVX2, 128 for u1 on serial.
 *  @param[in] dimensions_per_value Packing ratio: dimensions per storage value, 1 for f32, 2 for
 *      i4x2, 8 for u1x8.
 *
 *  @sa nk_define_cross_packed_ for asymmetric C = A × Bᵀ computation.
 *  @sa nk_define_cross_pack_size_ for calculating packed buffer size
 *  @sa nk_define_cross_pack_ for packing B matrix
 *  @sa include/numkong/set/serial.h for state type definitions
 *  @sa include/numkong/cast/serial.h for load/store function implementations
 */
#define nk_define_cross_symmetric_(api_name, input_type_name, isa_suffix, input_value_type, result_value_type,         \
                                   vec_type, state_type, result_vec_type, init_accumulator_fn, load_vec_fn,            \
                                   partial_load_vec_fn, inner_product_fn, reduce_accumulators_fn, store_fn,            \
                                   partial_store_fn, depth_simd_dimensions, dimensions_per_value)                      \
    NUMKONG_INLINE void nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                        \
        nk_##input_value_type##_t const **vector_base_ptrs, nk_size_t i_macro, nk_size_t macro_size,                   \
        nk_size_t aligned_depth, nk_size_t remainder_depth, nk_size_t remainder_dimensions,                            \
        nk_size_t depth_step_values, nk_size_t dimensions_per_value_runtime, nk_##result_value_type##_t *result,       \
        nk_size_t result_stride_values, nk_size_t finalizer_batch_size, nk_size_t depth) {                             \
                                                                                                                       \
        nk_unused_(dimensions_per_value_runtime);                                                                      \
        nk_unused_(finalizer_batch_size);                                                                              \
        /* Tile-first architecture: process a 32×32 macro-tile as                                                     \
         * 4×4 register tiles, depth innermost */                                                                     \
        for (nk_size_t tile_row_start = 0; tile_row_start < macro_size; tile_row_start += 4) {                         \
            for (nk_size_t tile_column_start = tile_row_start; tile_column_start < macro_size;                         \
                 tile_column_start += 4) {                                                                             \
                                                                                                                       \
                nk_size_t tile_rows = (tile_row_start + 4 <= macro_size) ? 4 : (macro_size - tile_row_start);          \
                nk_size_t tile_columns = (tile_column_start + 4 <= macro_size) ? 4 : (macro_size - tile_column_start); \
                int is_diagonal_tile = (tile_row_start == tile_column_start);                                          \
                                                                                                                       \
                /* Register-resident accumulators, padded to [4][7] so that the reduce call, */                        \
                /* which always reads 4 entries from column_start on, stays in bounds */                               \
                nk_align_(64) state_type accumulators[4][7];                                                           \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    nk_size_t init_start = is_diagonal_tile ? row : 0;                                                 \
                    nk_size_t init_end = is_diagonal_tile ? (row + 4) : 4;                                             \
                    for (nk_size_t column = init_start; column < init_end; column++) {                                 \
                        init_accumulator_fn(&accumulators[row][column]);                                               \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Setup pointers (hoist outside depth loop) - always safe even for partial tiles */                   \
                nk_##input_value_type##_t const *row_ptrs[4];                                                          \
                nk_##input_value_type##_t const *column_ptrs[4];                                                       \
                row_ptrs[0] = vector_base_ptrs[tile_row_start + 0];                                                    \
                row_ptrs[1] = (tile_rows > 1) ? vector_base_ptrs[tile_row_start + 1] : row_ptrs[0];                    \
                row_ptrs[2] = (tile_rows > 2) ? vector_base_ptrs[tile_row_start + 2] : row_ptrs[0];                    \
                row_ptrs[3] = (tile_rows > 3) ? vector_base_ptrs[tile_row_start + 3] : row_ptrs[0];                    \
                                                                                                                       \
                if (is_diagonal_tile) {                                                                                \
                    column_ptrs[0] = row_ptrs[0];                                                                      \
                    column_ptrs[1] = row_ptrs[1];                                                                      \
                    column_ptrs[2] = row_ptrs[2];                                                                      \
                    column_ptrs[3] = row_ptrs[3];                                                                      \
                }                                                                                                      \
                else {                                                                                                 \
                    column_ptrs[0] = vector_base_ptrs[tile_column_start + 0];                                          \
                    column_ptrs[1] = (tile_columns > 1) ? vector_base_ptrs[tile_column_start + 1] : column_ptrs[0];    \
                    column_ptrs[2] = (tile_columns > 2) ? vector_base_ptrs[tile_column_start + 2] : column_ptrs[0];    \
                    column_ptrs[3] = (tile_columns > 3) ? vector_base_ptrs[tile_column_start + 3] : column_ptrs[0];    \
                }                                                                                                      \
                                                                                                                       \
                /* Depth loop is now innermost - key optimization */                                                   \
                vec_type row_vecs[4];                                                                                  \
                vec_type column_vecs[4];                                                                               \
                                                                                                                       \
                for (nk_size_t depth_offset = 0; depth_offset < aligned_depth; depth_offset += depth_step_values) {    \
                    /* Always load all 4 vectors - aliasing is cheaper than branches */                                \
                    load_vec_fn(row_ptrs[0] + depth_offset, &row_vecs[0]);                                             \
                    load_vec_fn(row_ptrs[1] + depth_offset, &row_vecs[1]);                                             \
                    load_vec_fn(row_ptrs[2] + depth_offset, &row_vecs[2]);                                             \
                    load_vec_fn(row_ptrs[3] + depth_offset, &row_vecs[3]);                                             \
                                                                                                                       \
                    /* For diagonal tiles, column vectors alias row vectors (same memory) */                           \
                    load_vec_fn(column_ptrs[0] + depth_offset, &column_vecs[0]);                                       \
                    load_vec_fn(column_ptrs[1] + depth_offset, &column_vecs[1]);                                       \
                    load_vec_fn(column_ptrs[2] + depth_offset, &column_vecs[2]);                                       \
                    load_vec_fn(column_ptrs[3] + depth_offset, &column_vecs[3]);                                       \
                                                                                                                       \
                    nk_size_t vector_offset = depth_offset * dimensions_per_value;                                     \
                                                                                                                       \
                    /* Compute: always unroll for full 4×4, use loops only for partial tiles */                        \
                    if (tile_rows == 4 && tile_columns == 4) {                                                         \
                        if (is_diagonal_tile) {                                                                        \
                            /* Full 4×4 diagonal tile - upper triangle only (10 FMAs) */                               \
                            inner_product_fn(&accumulators[0][0], row_vecs[0], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][1], row_vecs[0], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][2], row_vecs[0], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][3], row_vecs[0], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][1], row_vecs[1], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][2], row_vecs[1], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][3], row_vecs[1], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][2], row_vecs[2], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][3], row_vecs[2], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][3], row_vecs[3], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                        }                                                                                              \
                        else {                                                                                         \
                            /* Full 4×4 off-diagonal tile (16 FMAs) */                                                 \
                            inner_product_fn(&accumulators[0][0], row_vecs[0], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][1], row_vecs[0], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][2], row_vecs[0], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][3], row_vecs[0], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][0], row_vecs[1], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][1], row_vecs[1], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][2], row_vecs[1], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][3], row_vecs[1], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][0], row_vecs[2], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][1], row_vecs[2], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][2], row_vecs[2], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][3], row_vecs[2], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][0], row_vecs[3], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][1], row_vecs[3], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][2], row_vecs[3], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][3], row_vecs[3], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                        }                                                                                              \
                    }                                                                                                  \
                    else {                                                                                             \
                        /* Partial tile - use loops (rare edge case) */                                                \
                        for (nk_size_t row = 0; row < tile_rows; row++) {                                              \
                            nk_size_t column_start = is_diagonal_tile ? row : 0;                                       \
                            for (nk_size_t column = column_start; column < tile_columns; column++) {                   \
                                inner_product_fn(&accumulators[row][column], row_vecs[row], column_vecs[column],       \
                                                 vector_offset, depth_simd_dimensions);                                \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Handle remainder depth (happens once per tile, not in hot loop) */                                  \
                if (remainder_depth > 0) {                                                                             \
                    partial_load_vec_fn(row_ptrs[0] + aligned_depth, &row_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[1] + aligned_depth, &row_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[2] + aligned_depth, &row_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[3] + aligned_depth, &row_vecs[3], remainder_dimensions);              \
                    partial_load_vec_fn(column_ptrs[0] + aligned_depth, &column_vecs[0], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[1] + aligned_depth, &column_vecs[1], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[2] + aligned_depth, &column_vecs[2], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[3] + aligned_depth, &column_vecs[3], remainder_dimensions);        \
                                                                                                                       \
                    nk_size_t vector_offset = aligned_depth * dimensions_per_value;                                    \
                    for (nk_size_t row = 0; row < tile_rows; row++) {                                                  \
                        nk_size_t column_start = is_diagonal_tile ? row : 0;                                           \
                        for (nk_size_t column = column_start; column < tile_columns; column++) {                       \
                            inner_product_fn(&accumulators[row][column], row_vecs[row], column_vecs[column],           \
                                             vector_offset, remainder_dimensions);                                     \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Direct finalization and store (no intermediate buffer) */                                           \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    nk_size_t column_start = is_diagonal_tile ? row : 0;                                               \
                    nk_size_t columns_remaining = tile_columns - column_start;                                         \
                    result_vec_type result_vec;                                                                        \
                                                                                                                       \
                    /* Always reduce 4 accumulators (partial_store handles actual count) */                            \
                    reduce_accumulators_fn(&accumulators[row][column_start], &accumulators[row][column_start + 1],     \
                                           &accumulators[row][column_start + 2], &accumulators[row][column_start + 3], \
                                           depth, &result_vec);                                                        \
                                                                                                                       \
                    nk_##result_value_type##_t *output_ptr =                                                           \
                        &result[(i_macro + tile_row_start + row) * result_stride_values +                              \
                                (i_macro + tile_column_start + column_start)];                                         \
                    partial_store_fn(&result_vec, output_ptr, columns_remaining);                                      \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_INLINE void nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_offdiagonal_(                     \
        nk_##input_value_type##_t const **vector_base_ptrs_i, nk_##input_value_type##_t const **vector_base_ptrs_j,    \
        nk_size_t i_macro, nk_size_t j_macro, nk_size_t macro_i_size, nk_size_t macro_j_size, nk_size_t aligned_depth, \
        nk_size_t remainder_depth, nk_size_t remainder_dimensions, nk_size_t depth_step_values,                        \
        nk_size_t dimensions_per_value_runtime, nk_##result_value_type##_t *result, nk_size_t result_stride_values,    \
        nk_size_t finalizer_batch_size, nk_size_t depth) {                                                             \
                                                                                                                       \
        nk_unused_(dimensions_per_value_runtime);                                                                      \
        nk_unused_(finalizer_batch_size);                                                                              \
        /* Tile-first architecture: process a 32×32 macro-tile as                                                     \
         * 4×4 register tiles, depth innermost */                                                                     \
        for (nk_size_t tile_row_start = 0; tile_row_start < macro_i_size; tile_row_start += 4) {                       \
            for (nk_size_t tile_column_start = 0; tile_column_start < macro_j_size; tile_column_start += 4) {          \
                                                                                                                       \
                nk_size_t tile_rows = (tile_row_start + 4 <= macro_i_size) ? 4 : (macro_i_size - tile_row_start);      \
                nk_size_t tile_columns = (tile_column_start + 4 <= macro_j_size) ? 4                                   \
                                                                                 : (macro_j_size - tile_column_start); \
                                                                                                                       \
                /* Initialize 4×4 register-resident accumulators, the full                                            \
                 * rectangle for off-diagonal tiles */                                                                 \
                nk_align_(64) state_type accumulators[4][4];                                                           \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    for (nk_size_t column = 0; column < 4; column++) {                                                 \
                        init_accumulator_fn(&accumulators[row][column]);                                               \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Setup pointers (hoist outside depth loop) - always safe even for partial tiles */                   \
                nk_##input_value_type##_t const *row_ptrs[4];                                                          \
                nk_##input_value_type##_t const *column_ptrs[4];                                                       \
                row_ptrs[0] = vector_base_ptrs_i[tile_row_start + 0];                                                  \
                row_ptrs[1] = (tile_rows > 1) ? vector_base_ptrs_i[tile_row_start + 1] : row_ptrs[0];                  \
                row_ptrs[2] = (tile_rows > 2) ? vector_base_ptrs_i[tile_row_start + 2] : row_ptrs[0];                  \
                row_ptrs[3] = (tile_rows > 3) ? vector_base_ptrs_i[tile_row_start + 3] : row_ptrs[0];                  \
                column_ptrs[0] = vector_base_ptrs_j[tile_column_start + 0];                                            \
                column_ptrs[1] = (tile_columns > 1) ? vector_base_ptrs_j[tile_column_start + 1] : column_ptrs[0];      \
                column_ptrs[2] = (tile_columns > 2) ? vector_base_ptrs_j[tile_column_start + 2] : column_ptrs[0];      \
                column_ptrs[3] = (tile_columns > 3) ? vector_base_ptrs_j[tile_column_start + 3] : column_ptrs[0];      \
                                                                                                                       \
                /* Depth loop is now innermost - key optimization */                                                   \
                vec_type row_vecs[4];                                                                                  \
                vec_type column_vecs[4];                                                                               \
                                                                                                                       \
                for (nk_size_t depth_offset = 0; depth_offset < aligned_depth; depth_offset += depth_step_values) {    \
                    /* Always load all 8 vectors - aliasing is cheaper than branches */                                \
                    load_vec_fn(row_ptrs[0] + depth_offset, &row_vecs[0]);                                             \
                    load_vec_fn(row_ptrs[1] + depth_offset, &row_vecs[1]);                                             \
                    load_vec_fn(row_ptrs[2] + depth_offset, &row_vecs[2]);                                             \
                    load_vec_fn(row_ptrs[3] + depth_offset, &row_vecs[3]);                                             \
                    load_vec_fn(column_ptrs[0] + depth_offset, &column_vecs[0]);                                       \
                    load_vec_fn(column_ptrs[1] + depth_offset, &column_vecs[1]);                                       \
                    load_vec_fn(column_ptrs[2] + depth_offset, &column_vecs[2]);                                       \
                    load_vec_fn(column_ptrs[3] + depth_offset, &column_vecs[3]);                                       \
                                                                                                                       \
                    nk_size_t vector_offset = depth_offset * dimensions_per_value;                                     \
                                                                                                                       \
                    /* Compute: always unroll for full 4×4, use loops only for partial tiles */                        \
                    if (tile_rows == 4 && tile_columns == 4) {                                                         \
                        /* Full 4×4 off-diagonal tile (16 FMAs) */                                                     \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], column_vecs[0], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], column_vecs[1], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], column_vecs[2], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], column_vecs[3], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][0], row_vecs[1], column_vecs[0], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], column_vecs[1], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], column_vecs[2], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], column_vecs[3], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][0], row_vecs[2], column_vecs[0], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][1], row_vecs[2], column_vecs[1], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], column_vecs[2], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], column_vecs[3], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][0], row_vecs[3], column_vecs[0], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][1], row_vecs[3], column_vecs[1], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][2], row_vecs[3], column_vecs[2], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], column_vecs[3], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                    }                                                                                                  \
                    else {                                                                                             \
                        /* Partial tile - use loops (rare edge case) */                                                \
                        for (nk_size_t row = 0; row < tile_rows; row++) {                                              \
                            for (nk_size_t column = 0; column < tile_columns; column++) {                              \
                                inner_product_fn(&accumulators[row][column], row_vecs[row], column_vecs[column],       \
                                                 vector_offset, depth_simd_dimensions);                                \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Handle remainder depth (happens once per tile, not in hot loop) */                                  \
                if (remainder_depth > 0) {                                                                             \
                    partial_load_vec_fn(row_ptrs[0] + aligned_depth, &row_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[1] + aligned_depth, &row_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[2] + aligned_depth, &row_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[3] + aligned_depth, &row_vecs[3], remainder_dimensions);              \
                    partial_load_vec_fn(column_ptrs[0] + aligned_depth, &column_vecs[0], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[1] + aligned_depth, &column_vecs[1], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[2] + aligned_depth, &column_vecs[2], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[3] + aligned_depth, &column_vecs[3], remainder_dimensions);        \
                                                                                                                       \
                    nk_size_t vector_offset = aligned_depth * dimensions_per_value;                                    \
                    for (nk_size_t row = 0; row < tile_rows; row++) {                                                  \
                        for (nk_size_t column = 0; column < tile_columns; column++) {                                  \
                            inner_product_fn(&accumulators[row][column], row_vecs[row], column_vecs[column],           \
                                             vector_offset, remainder_dimensions);                                     \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Direct finalization and store (no intermediate buffer) */                                           \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    result_vec_type result_vec;                                                                        \
                                                                                                                       \
                    /* Always reduce 4 accumulators (partial_store handles actual count) */                            \
                    reduce_accumulators_fn(&accumulators[row][0], &accumulators[row][1], &accumulators[row][2],        \
                                           &accumulators[row][3], depth, &result_vec);                                 \
                                                                                                                       \
                    nk_##result_value_type##_t *output_ptr =                                                           \
                        &result[(i_macro + tile_row_start + row) * result_stride_values +                              \
                                (j_macro + tile_column_start)];                                                        \
                    partial_store_fn(&result_vec, output_ptr, tile_columns);                                           \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_INLINE void nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_(                                 \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,          \
        nk_##result_value_type##_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {       \
        nk_assert_(depth % dimensions_per_value == 0 && nk_cross_whole_blocks_(nk_##input_type_name##_k, depth));      \
        nk_assert_(stride % sizeof(nk_##input_value_type##_t) == 0 &&                                                  \
                   stride >=                                                                                           \
                       nk_size_divide_round_up_(depth, dimensions_per_value) * sizeof(nk_##input_value_type##_t));     \
        nk_size_t const macro_tile_size = 32;                                                                          \
        nk_size_t const finalizer_batch_size = 4;                                                                      \
        nk_size_t const row_block_size = 128;     /* L2 cache blocking */                                              \
        nk_size_t const column_block_size = 2048; /* L3 cache blocking */                                              \
                                                                                                                       \
        /* Stride and depth calculations */                                                                            \
        nk_size_t const vectors_stride_values = stride / sizeof(nk_##input_value_type##_t);                            \
        nk_size_t const result_stride_values = result_stride / sizeof(nk_##result_value_type##_t);                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = depth_dimensions_aligned / dimensions_per_value;                               \
        nk_size_t const depth_in_values = depth / dimensions_per_value;                                                \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        nk_size_t const depth_step_values = depth_simd_dimensions / dimensions_per_value;                              \
        nk_size_t const row_end = (row_start + row_count < vectors_count) ? (row_start + row_count) : vectors_count;   \
                                                                                                                       \
        /* Process the upper triangle with L3/L2/L1 blocking: column                                                   \
         * blocks → row blocks → 32×32 macro-tiles */                                                             \
        for (nk_size_t j_block = 0; j_block < vectors_count; j_block += column_block_size) {                           \
            nk_size_t j_block_end = (j_block + column_block_size < vectors_count) ? j_block + column_block_size        \
                                                                                  : vectors_count;                     \
                                                                                                                       \
            for (nk_size_t i_block = row_start; i_block < row_end; i_block += row_block_size) {                        \
                nk_size_t i_block_end = (i_block + row_block_size < row_end) ? i_block + row_block_size : row_end;     \
                                                                                                                       \
                /* Skip blocks entirely below diagonal. Blocks fully above the diagonal are still                      \
                 * part of the upper triangle and must be computed. */                                                 \
                if (i_block >= j_block_end) continue;                                                                  \
                                                                                                                       \
                for (nk_size_t i_macro = i_block, macro_i_size; i_macro < i_block_end; i_macro += macro_i_size) {      \
                    /* Row tiles stop at column-block edges, so only `j_macro == i_macro` meets the diagonal */        \
                    nk_size_t const column_edge = (i_macro / column_block_size + 1) * column_block_size;               \
                    macro_i_size = nk_min_of_two(macro_tile_size, nk_min_of_two(i_block_end, column_edge) - i_macro);  \
                    nk_size_t j_start = (i_macro > j_block) ? i_macro : j_block;                                       \
                    for (nk_size_t j_macro = j_start; j_macro < j_block_end; j_macro += macro_tile_size) {             \
                        nk_size_t macro_j_size = (j_macro + macro_tile_size <= j_block_end) ? macro_tile_size          \
                                                                                            : (j_block_end - j_macro); \
                                                                                                                       \
                        /* Hoist pointer computation outside depth loop */                                             \
                        nk_##input_value_type##_t const *vector_base_ptrs_i[32];                                       \
                        nk_##input_value_type##_t const *vector_base_ptrs_j[32];                                       \
                        for (nk_size_t i = 0; i < macro_i_size; i++) {                                                 \
                            vector_base_ptrs_i[i] = vectors + (i_macro + i) * vectors_stride_values;                   \
                        }                                                                                              \
                        for (nk_size_t j = 0; j < macro_j_size; j++) {                                                 \
                            vector_base_ptrs_j[j] = vectors + (j_macro + j) * vectors_stride_values;                   \
                        }                                                                                              \
                                                                                                                       \
                        /* A diagonal tile is a triangle, plus a rectangle when the row range ends early */            \
                        nk_size_t const skipped_columns = (i_macro == j_macro) ? macro_i_size : 0;                     \
                        if (i_macro == j_macro)                                                                        \
                            nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                    \
                                vector_base_ptrs_i, i_macro, macro_i_size, aligned_depth, remainder_depth,             \
                                remainder_dimensions, depth_step_values, dimensions_per_value, result,                 \
                                result_stride_values, finalizer_batch_size, depth);                                    \
                        if (skipped_columns < macro_j_size)                                                            \
                            nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_offdiagonal##_(               \
                                vector_base_ptrs_i, vector_base_ptrs_j + skipped_columns, i_macro,                     \
                                j_macro + skipped_columns, macro_i_size, macro_j_size - skipped_columns,               \
                                aligned_depth, remainder_depth, remainder_dimensions, depth_step_values,               \
                                dimensions_per_value, result, result_stride_values, finalizer_batch_size, depth);      \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##api_name##_symmetric_##input_type_name##_##isa_suffix(                                \
        nk_cross_##input_type_name##_operand_t const *vectors_operand, nk_size_t vectors_count, nk_size_t depth,       \
        nk_size_t stride, nk_##result_value_type##_t *result, nk_size_t result_stride, nk_size_t row_start,            \
        nk_size_t row_count, void *stream) {                                                                           \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_(                                                 \
            (nk_##input_value_type##_t const *)vectors_operand, vectors_count, depth, stride, result, result_stride,   \
            row_start, row_count);                                                                                     \
        return nk_success_k;                                                                                           \
    }

/**
 *  @brief Generates block-scaled packed @p api_name over decoded panels of A.
 *
 *  Rows of A are packed 32 at a time, one depth chunk at a time, into stack panels through the
 *  pack's own row helper, so one loader reads both operands with plain loads. Register tiles
 *  accumulate block partials in rebased F32; chunks before the last store raw partials in C, and
 *  the last hands them to @p metric_fn with the bases and tensor scales of both sides and, when
 *  @p normalized, their relative squared norms: the pack's for B, and the panel rows times
 *  themselves for A. Outputs whose spreads leave the rebased window take the wide sums of
 *  @p exact_fn through @p wide_metric_fn instead.
 *
 *  @param[in] operand_type One decoded depth step of either operand.
 *  @param[in] state_type Per-output accumulator.
 *  @param[in] init_fn void fn(state_type *, nk_f32_t seed), seeding a stored partial.
 *  @param[in] load_fn void fn(packed values const *, nk_u8_t const *scales, nk_size_t offset,
 *      operand_type *), @p offset in dimensions.
 *  @param[in] update_fn void fn(state_type *, operand_type, operand_type).
 *  @param[in] reduce_fn void fn(state_type const * × 4, nk_b128_vec_t *).
 *  @param[in] metric_fn void fn(nk_b128_vec_t *values, nk_f32_t mantissa, nk_i32_t row_exponent,
 *      nk_i32_t const *column_exponents, nk_f32_t row_norm, nk_f32_t const *column_norms), as
 *      @c nk_angular_f32x4_from_relative_neon_ finishes four relative sums.
 *  @param[in] exact_fn The @c nk_define_cross_scaled_exact_ wide dot of the dtype over the
 *      element layout of the pack.
 *  @param[in] wide_metric_fn The metric from wide sums, shaped like @c nk_dot_f32_from_wide_.
 *  @param[in] normalized 1 when @p metric_fn reads norms, which square one side's scales: a row or
 *      a column then takes the exact path once its own spread passes half the window.
 *  @param[in] scale_bytes Bytes per block of a pack scale row, the same as the pack size takes.
 *  @param[in] register_rows Rows per register tile.
 *  @param[in] register_columns Columns per register tile, a multiple of 4.
 */
#define nk_define_cross_scaled_packed_(api_name, input_type_name, isa_suffix, packed_value_type, operand_type,         \
                                       state_type, init_fn, load_fn, update_fn, reduce_fn, metric_fn, exact_fn,        \
                                       wide_metric_fn, normalized, depth_simd_dimensions, dimensions_per_value,        \
                                       scale_bytes, register_rows, register_columns)                                   \
    /* Adds to norms[r] the products of panel rows r < rows with themselves over depth dims, the                       \
     * relative squared norms the tiles reproduce on a diagonal. */                                                    \
    NUMKONG_INLINE void nk_##api_name##_scaled_norms_##input_type_name##_##isa_suffix##_(                              \
        nk_##packed_value_type##_t const *values, nk_u8_t const *scales, nk_size_t values_stride,                      \
        nk_size_t scales_stride, nk_size_t rows, nk_size_t depth, nk_f32_t *norms) {                                   \
        for (nk_size_t row = 0; row < rows; row += 4) {                                                                \
            state_type states[4];                                                                                      \
            for (nk_size_t r = 0; r != 4; ++r) init_fn(&states[r], row + r < rows ? norms[row + r] : 0);               \
            for (nk_size_t step = 0; step < depth; step += (depth_simd_dimensions))                                    \
                for (nk_size_t r = 0; r != 4; ++r) {                                                                   \
                    nk_size_t const panel_row = row + r < rows ? row + r : rows - 1;                                   \
                    operand_type operand;                                                                              \
                    load_fn(values + panel_row * values_stride, scales + panel_row * scales_stride, step, &operand);   \
                    update_fn(&states[r], operand, operand);                                                           \
                }                                                                                                      \
            nk_b128_vec_t sums;                                                                                        \
            reduce_fn(&states[0], &states[1], &states[2], &states[3], &sums);                                          \
            for (nk_size_t r = 0; r != 4 && row + r < rows; ++r) norms[row + r] = sums.f32s[r];                        \
        }                                                                                                              \
    }                                                                                                                  \
    /* Tiles rows [a_first, a_first + a_rows) of a panel against columns                                               \
     * [b_first, b_first + b_columns) of a pack or panel over depth dims, reading B from b_offset                      \
     * dims on; upper keeps the columns from each row on. */                                                           \
    NUMKONG_INLINE void nk_##api_name##_scaled_tiles_##input_type_name##_##isa_suffix##_(                              \
        nk_##packed_value_type##_t const *a_values, nk_u8_t const *a_scales, nk_size_t a_values_stride,                \
        nk_size_t a_scales_stride, nk_i32_t const *a_exponents, nk_f32_t const *a_norms, nk_size_t a_first,            \
        nk_size_t a_rows, nk_##packed_value_type##_t const *b_values, nk_u8_t const *b_scales,                         \
        nk_size_t b_values_stride, nk_size_t b_scales_stride, nk_i32_t const *b_exponents, nk_f32_t const *b_norms,    \
        nk_size_t b_first, nk_size_t b_columns, nk_size_t b_offset, nk_size_t depth, int first, int last, int upper,   \
        nk_f32_t mantissa, nk_f32_t *c, nk_size_t c_stride) {                                                          \
        for (nk_size_t column = 0; column < b_columns; column += (register_columns))                                   \
            for (nk_size_t row = 0; row < a_rows; row += (register_rows)) {                                            \
                if (upper && b_first + column + (register_columns) <= a_first + row) continue;                         \
                state_type states[register_rows][register_columns];                                                    \
                for (nk_size_t r = 0; r != (register_rows); ++r)                                                       \
                    for (nk_size_t q = 0; q != (register_columns); ++q) {                                              \
                        int const stored = !first && row + r < a_rows && column + q < b_columns &&                     \
                                           (!upper || b_first + column + q >= a_first + row + r);                      \
                        init_fn(&states[r][q],                                                                         \
                                stored ? ((nk_f32_t const *)((char const *)c +                                         \
                                                             (a_first + row + r) * c_stride))[b_first + column + q]    \
                                       : 0);                                                                           \
                    }                                                                                                  \
                for (nk_size_t step = 0; step < depth; step += (depth_simd_dimensions)) {                              \
                    operand_type a_operands[register_rows];                                                            \
                    for (nk_size_t r = 0; r != (register_rows); ++r) {                                                 \
                        nk_size_t const a_row = row + r < a_rows ? row + r : a_rows - 1;                               \
                        load_fn(a_values + a_row * a_values_stride, a_scales + a_row * a_scales_stride, step,          \
                                &a_operands[r]);                                                                       \
                    }                                                                                                  \
                    for (nk_size_t q = 0; q != (register_columns); ++q) {                                              \
                        nk_size_t const b_column = column + q < b_columns ? column + q : b_columns - 1;                \
                        operand_type b_operand;                                                                        \
                        load_fn(b_values + b_column * b_values_stride, b_scales + b_column * b_scales_stride,          \
                                b_offset + step, &b_operand);                                                          \
                        for (nk_size_t r = 0; r != (register_rows); ++r)                                               \
                            update_fn(&states[r][q], a_operands[r], b_operand);                                        \
                    }                                                                                                  \
                }                                                                                                      \
                for (nk_size_t r = 0; r != (register_rows) && row + r < a_rows; ++r) {                                 \
                    nk_f32_t *c_row = (nk_f32_t *)((char *)c + (a_first + row + r) * c_stride) + b_first;              \
                    for (nk_size_t q = 0; q != (register_columns); q += 4) {                                           \
                        nk_b128_vec_t sums;                                                                            \
                        reduce_fn(&states[r][q], &states[r][q + 1], &states[r][q + 2], &states[r][q + 3], &sums);      \
                        if (last)                                                                                      \
                            metric_fn(&sums, mantissa, a_exponents[row + r], b_exponents + column + q,                 \
                                      a_norms[row + r], b_norms + column + q);                                         \
                        for (nk_size_t lane = 0; lane != 4; ++lane)                                                    \
                            if (column + q + lane < b_columns &&                                                       \
                                (!upper || b_first + column + q + lane >= a_first + row + r))                          \
                                c_row[column + q + lane] = sums.f32s[lane];                                            \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##api_name##_packed_##input_type_name##_##isa_suffix(                                   \
        nk_cross_##input_type_name##_operand_t const *a_operand, void const *b_packed, nk_f32_t *c, nk_size_t rows,    \
        nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, void *stream) {                    \
        /* Panels of 32 rows take up to 64 KB of values, at most 2048 dims per depth chunk */                          \
        enum {                                                                                                         \
            chunk_k = 2048 * (dimensions_per_value) / sizeof(nk_##packed_value_type##_t) < 2048                        \
                          ? 2048 * (dimensions_per_value) / sizeof(nk_##packed_value_type##_t)                         \
                          : 2048                                                                                       \
        };                                                                                                             \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        nk_cross_operand_t const a = nk_cross_operand_(nk_##input_type_name##_k, a_operand, a_stride);                 \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;             \
        nk_block_scaled_format_t const format = nk_block_scaled_format_of_dtype(nk_##input_type_name##_k);             \
        nk_f32_t const a_tensor_scale = nk_cross_tensor_scale_(a.tensor_scale), b_tensor_scale = header->tensor_scale; \
        nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_(a_tensor_scale, b_tensor_scale);               \
        nk_cross_tensor_factor_t const a_factor = nk_cross_tensor_factor_(a_tensor_scale, a_tensor_scale);             \
        nk_cross_tensor_factor_t const b_factor = nk_cross_tensor_factor_(b_tensor_scale, b_tensor_scale);             \
        int const rebased = format.scale_dtype == nk_ue8m0_k;                                                          \
        nk_size_t const blocks = depth / format.block_size, chunk = chunk_k;                                           \
        nk_size_t const b_values_stride = header->depth_padded_values, b_scales_stride = header->scales_stride;        \
        nk_size_t const b_codes = b_scales_stride - b_scales_stride / (scale_bytes);                                   \
        nk_##packed_value_type##_t const *b_values =                                                                   \
            (nk_##packed_value_type##_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t));    \
        nk_u8_t const *b_scales = nk_cross_packed_scales_(b_packed, sizeof(nk_##packed_value_type##_t), 0);            \
        nk_f32_t const *b_norms = (nk_f32_t const *)(b_scales + header->column_count * b_scales_stride);               \
        nk_align_(64) nk_##packed_value_type##_t panel_values[32 * chunk_k / (dimensions_per_value)];                  \
        nk_align_(64) nk_u8_t panel_scales[32 * chunk_k / 16 * (scale_bytes)];                                         \
        nk_i32_t row_bases[32] = {0}, row_exponents[32] = {0}, row_spreads[32] = {0}, column_spreads[256] = {0};       \
        nk_i32_t column_exponents[256 + (register_columns)] = {0};                                                     \
        nk_f32_t row_norms[32] = {0}, column_norms[256 + (register_columns)] = {0};                                    \
        for (nk_size_t row_first = 0; row_first < rows; row_first += 32) {                                             \
            nk_size_t const block_rows = rows - row_first < 32 ? rows - row_first : 32;                                \
            nk_i32_t row_spread_max = 0;                                                                               \
            for (nk_size_t row = 0; row != block_rows; ++row) {                                                        \
                if (rebased)                                                                                           \
                    row_bases[row] = nk_cross_scaled_base_(a.scales + (row_first + row) * a.scales_stride, blocks,     \
                                                           &row_spreads[row]);                                         \
                row_exponents[row] = row_bases[row] + a_factor.exponent / 2, row_norms[row] = 0;                       \
                row_spread_max = row_spreads[row] > row_spread_max ? row_spreads[row] : row_spread_max;                \
            }                                                                                                          \
            /* Runs once for an empty depth too, finishing zero sums */                                                \
            nk_size_t depth_first = 0;                                                                                 \
            do {                                                                                                       \
                nk_size_t const chunk_depth = depth - depth_first < chunk ? depth - depth_first : chunk;               \
                nk_size_t const panel_stride = nk_size_round_up_to_multiple_(chunk_depth, depth_simd_dimensions) /     \
                                               (dimensions_per_value);                                                 \
                nk_size_t const panel_scales_stride = nk_cross_scales_stride_(nk_##input_type_name##_k, chunk_depth) * \
                                                      (scale_bytes);                                                   \
                int const first = depth_first == 0, last = depth_first + chunk_depth >= depth;                         \
                nk_dots_pack_rows_##input_type_name##_##isa_suffix##_(                                                 \
                    a, a_stride, row_first, row_first + block_rows, depth_first, chunk_depth, row_bases, panel_values, \
                    panel_stride, panel_scales, panel_scales_stride);                                                  \
                if (normalized)                                                                                        \
                    nk_##api_name##_scaled_norms_##input_type_name##_##isa_suffix##_(                                  \
                        panel_values, panel_scales, panel_stride, panel_scales_stride, block_rows, chunk_depth,        \
                        row_norms);                                                                                    \
                for (nk_size_t row = 0; row != block_rows && (normalized) && last; ++row)                              \
                    row_norms[row] *= a_factor.mantissa;                                                               \
                for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {                        \
                    nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;       \
                    nk_i32_t column_spread_max = 0;                                                                    \
                    for (nk_size_t column = 0; column != block_columns && last; ++column) {                            \
                        nk_i32_t base = 0;                                                                             \
                        if (rebased)                                                                                   \
                            base = nk_cross_scaled_base_(                                                              \
                                b_scales + (column_first + column) * b_scales_stride + b_codes, blocks,                \
                                &column_spreads[column]);                                                              \
                        column_exponents[column] = base + b_factor.exponent / 2;                                       \
                        if (normalized) column_norms[column] = b_norms[column_first + column] * b_factor.mantissa;     \
                        column_spread_max = column_spreads[column] > column_spread_max ? column_spreads[column]        \
                                                                                       : column_spread_max;            \
                    }                                                                                                  \
                    nk_##api_name##_scaled_tiles_##input_type_name##_##isa_suffix##_(                                  \
                        panel_values, panel_scales, panel_stride, panel_scales_stride, row_exponents, row_norms,       \
                        row_first, block_rows, b_values + column_first * b_values_stride,                              \
                        b_scales + column_first * b_scales_stride, b_values_stride, b_scales_stride, column_exponents, \
                        column_norms, column_first, block_columns, depth_first, chunk_depth, first, last, 0,           \
                        factor.mantissa, c, c_stride);                                                                 \
                    int const exact = rebased && last &&                                                               \
                                      nk_cross_scaled_exceeds_(row_spread_max, column_spread_max, normalized);         \
                    for (nk_size_t row = 0; row != block_rows && exact; ++row)                                         \
                        for (nk_size_t column = 0; column != block_columns; ++column) {                                \
                            if (!nk_cross_scaled_exceeds_(row_spreads[row], column_spreads[column], normalized))       \
                                continue;                                                                              \
                            nk_size_t const a_row = row_first + row, b_column = column_first + column;                 \
                            nk_cross_wide_sum_t a_sumsq, b_sumsq;                                                      \
                            nk_cross_wide_sum_t const dot = exact_fn(                                                  \
                                (nk_u8_t const *)a.elements + a_row * a_stride, a.scales + a_row * a.scales_stride,    \
                                (nk_u8_t const *)(b_values + b_column * b_values_stride),                              \
                                b_scales + b_column * b_scales_stride + b_codes, depth, &a_sumsq, &b_sumsq);           \
                            ((nk_f32_t *)((char *)c + a_row * c_stride))[b_column] = wide_metric_fn(                   \
                                nk_cross_wide_times_(dot, factor), nk_cross_wide_times_(a_sumsq, a_factor),            \
                                nk_cross_wide_times_(b_sumsq, b_factor));                                              \
                        }                                                                                              \
                }                                                                                                      \
                depth_first += chunk;                                                                                  \
            } while (depth_first < depth);                                                                             \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }

/**
 *  @brief Generates block-scaled symmetric @p api_name over decoded panels, from the tiles, norms
 *      and row helper of the @c nk_define_cross_scaled_packed_ and @c nk_define_cross_pack_
 *      instances of the same dtype and ISA.
 *
 *  Rows [row_start, row_start + row_count) and windows of 32 columns from each row block on are
 *  packed into half-size panels per depth chunk, each rebased by its own vectors' bases; diagonal
 *  windows reuse the row panel, and every output keeps the columns from its row on. Norms come from
 *  the panels: a column window's take its earlier chunks packed again at the last one. Outputs past
 *  the window take the serial exact sums through @p wide_metric_fn, both operands being raw codes.
 */
#define nk_define_cross_scaled_symmetric_(api_name, input_type_name, isa_suffix, packed_value_type, wide_metric_fn,    \
                                          normalized, depth_simd_dimensions, dimensions_per_value, scale_bytes,        \
                                          register_rows, register_columns)                                             \
    NUMKONG_API nk_status_t nk_##api_name##_symmetric_##input_type_name##_##isa_suffix(                                \
        nk_cross_##input_type_name##_operand_t const *vectors_operand, nk_size_t count, nk_size_t depth,               \
        nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,         \
        void *stream) {                                                                                                \
        /* Row and column panels of 32 vectors take up to 32 KB of values each, at most 1024 dims                      \
         * per chunk */                                                                                                \
        enum {                                                                                                         \
            chunk_k = 1024 * (dimensions_per_value) / sizeof(nk_##packed_value_type##_t) < 1024                        \
                          ? 1024 * (dimensions_per_value) / sizeof(nk_##packed_value_type##_t)                         \
                          : 1024                                                                                       \
        };                                                                                                             \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        nk_cross_operand_t const vectors = nk_cross_operand_(nk_##input_type_name##_k, vectors_operand, stride);       \
        nk_block_scaled_format_t const format = nk_block_scaled_format_of_dtype(nk_##input_type_name##_k);             \
        nk_f32_t const tensor_scale = nk_cross_tensor_scale_(vectors.tensor_scale);                                    \
        nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_(tensor_scale, tensor_scale);                   \
        int const rebased = format.scale_dtype == nk_ue8m0_k;                                                          \
        nk_size_t const blocks = depth / format.block_size, chunk = chunk_k;                                           \
        nk_size_t const row_end = row_start + row_count < count ? row_start + row_count : count;                       \
        nk_align_(64) nk_##packed_value_type##_t row_values[32 * chunk_k / (dimensions_per_value)];                    \
        nk_align_(64) nk_##packed_value_type##_t column_values[32 * chunk_k / (dimensions_per_value)];                 \
        nk_align_(64) nk_u8_t row_scales[32 * chunk_k / 16 * (scale_bytes)];                                           \
        nk_align_(64) nk_u8_t column_scales[32 * chunk_k / 16 * (scale_bytes)];                                        \
        /* Diagonal windows read the row arrays four lanes at a time as columns */                                     \
        nk_i32_t row_bases[32] = {0}, row_exponents[32 + (register_columns)] = {0}, row_spreads[32] = {0};             \
        nk_i32_t column_bases[32] = {0}, column_exponents[32 + (register_columns)] = {0}, column_spreads[32] = {0};    \
        nk_f32_t row_norms[32 + (register_columns)] = {0}, column_norms[32 + (register_columns)] = {0};                \
        for (nk_size_t row_first = row_start; row_first < row_end; row_first += 32) {                                  \
            nk_size_t const block_rows = row_end - row_first < 32 ? row_end - row_first : 32;                          \
            nk_i32_t row_spread_max = 0;                                                                               \
            for (nk_size_t row = 0; row != block_rows; ++row) {                                                        \
                if (rebased)                                                                                           \
                    row_bases[row] = nk_cross_scaled_base_(vectors.scales + (row_first + row) * vectors.scales_stride, \
                                                           blocks, &row_spreads[row]);                                 \
                row_exponents[row] = row_bases[row] + factor.exponent / 2, row_norms[row] = 0;                         \
                row_spread_max = row_spreads[row] > row_spread_max ? row_spreads[row] : row_spread_max;                \
            }                                                                                                          \
            /* Runs once for an empty depth too, finishing zero sums */                                                \
            nk_size_t depth_first = 0;                                                                                 \
            do {                                                                                                       \
                nk_size_t const chunk_depth = depth - depth_first < chunk ? depth - depth_first : chunk;               \
                nk_size_t const panel_stride = nk_size_round_up_to_multiple_(chunk_depth, depth_simd_dimensions) /     \
                                               (dimensions_per_value);                                                 \
                nk_size_t const panel_scales_stride = nk_cross_scales_stride_(nk_##input_type_name##_k, chunk_depth) * \
                                                      (scale_bytes);                                                   \
                int const first = depth_first == 0, last = depth_first + chunk_depth >= depth;                         \
                nk_dots_pack_rows_##input_type_name##_##isa_suffix##_(                                                 \
                    vectors, stride, row_first, row_first + block_rows, depth_first, chunk_depth, row_bases,           \
                    row_values, panel_stride, row_scales, panel_scales_stride);                                        \
                if (normalized)                                                                                        \
                    nk_##api_name##_scaled_norms_##input_type_name##_##isa_suffix##_(                                  \
                        row_values, row_scales, panel_stride, panel_scales_stride, block_rows, chunk_depth,            \
                        row_norms);                                                                                    \
                for (nk_size_t row = 0; row != block_rows && (normalized) && last; ++row)                              \
                    row_norms[row] *= factor.mantissa;                                                                 \
                for (nk_size_t column_first = row_first; column_first < count; column_first += 32) {                   \
                    nk_size_t const block_columns = count - column_first < 32 ? count - column_first : 32;             \
                    int const shared = column_first == row_first && block_columns == block_rows;                       \
                    nk_i32_t column_spread_max = shared ? row_spread_max : 0;                                          \
                    for (nk_size_t column = 0; column != block_columns && !shared; ++column) {                         \
                        if (rebased)                                                                                   \
                            column_bases[column] = nk_cross_scaled_base_(                                              \
                                vectors.scales + (column_first + column) * vectors.scales_stride, blocks,              \
                                &column_spreads[column]);                                                              \
                        column_exponents[column] = column_bases[column] + factor.exponent / 2;                         \
                        column_spread_max = column_spreads[column] > column_spread_max ? column_spreads[column]        \
                                                                                       : column_spread_max;            \
                    }                                                                                                  \
                    for (nk_size_t column = 0; column != block_columns && !shared && (normalized) && last; ++column)   \
                        column_norms[column] = 0;                                                                      \
                    /* Normalized windows pack every chunk at the last one, ending with it */                          \
                    for (nk_size_t norm_first = (normalized) && last ? 0 : depth_first;                                \
                         norm_first <= depth_first && !shared; norm_first += chunk) {                                  \
                        nk_size_t const norm_depth = depth - norm_first < chunk ? depth - norm_first : chunk;          \
                        nk_size_t const norm_stride =                                                                  \
                            nk_size_round_up_to_multiple_(norm_depth, depth_simd_dimensions) / (dimensions_per_value); \
                        nk_size_t const norm_scales_stride =                                                           \
                            nk_cross_scales_stride_(nk_##input_type_name##_k, norm_depth) * (scale_bytes);             \
                        nk_dots_pack_rows_##input_type_name##_##isa_suffix##_(                                         \
                            vectors, stride, column_first, column_first + block_columns, norm_first, norm_depth,       \
                            column_bases, column_values, norm_stride, column_scales, norm_scales_stride);              \
                        if ((normalized) && last)                                                                      \
                            nk_##api_name##_scaled_norms_##input_type_name##_##isa_suffix##_(                          \
                                column_values, column_scales, norm_stride, norm_scales_stride, block_columns,          \
                                norm_depth, column_norms);                                                             \
                    }                                                                                                  \
                    for (nk_size_t column = 0; column != block_columns && !shared && (normalized) && last; ++column)   \
                        column_norms[column] *= factor.mantissa;                                                       \
                    nk_##api_name##_scaled_tiles_##input_type_name##_##isa_suffix##_(                                  \
                        row_values, row_scales, panel_stride, panel_scales_stride, row_exponents, row_norms,           \
                        row_first, block_rows, shared ? row_values : column_values,                                    \
                        shared ? row_scales : column_scales, panel_stride, panel_scales_stride,                        \
                        shared ? row_exponents : column_exponents, shared ? row_norms : column_norms, column_first,    \
                        block_columns, 0, chunk_depth, first, last, 1, factor.mantissa, result, result_stride);        \
                    int const exact = rebased && last &&                                                               \
                                      nk_cross_scaled_exceeds_(row_spread_max, column_spread_max, normalized);         \
                    for (nk_size_t row = 0; row != block_rows && exact; ++row)                                         \
                        for (nk_size_t column = 0; column != block_columns; ++column) {                                \
                            nk_size_t const row_index = row_first + row, column_index = column_first + column;         \
                            nk_i32_t const column_spread = shared ? row_spreads[column] : column_spreads[column];      \
                            if (column_index < row_index ||                                                            \
                                !nk_cross_scaled_exceeds_(row_spreads[row], column_spread, normalized))                \
                                continue;                                                                              \
                            nk_cross_wide_sum_t row_sumsq, column_sumsq;                                               \
                            nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_##input_type_name##_serial_(    \
                                (nk_u8_t const *)vectors.elements + row_index * stride,                                \
                                vectors.scales + row_index * vectors.scales_stride,                                    \
                                (nk_u8_t const *)vectors.elements + column_index * stride,                             \
                                vectors.scales + column_index * vectors.scales_stride, depth, &row_sumsq,              \
                                &column_sumsq);                                                                        \
                            ((nk_f32_t *)((char *)result + row_index * result_stride))[column_index] = wide_metric_fn( \
                                nk_cross_wide_times_(dot, factor), nk_cross_wide_times_(row_sumsq, factor),            \
                                nk_cross_wide_times_(column_sumsq, factor));                                           \
                        }                                                                                              \
                }                                                                                                      \
                depth_first += chunk;                                                                                  \
            } while (depth_first < depth);                                                                             \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }

#if NUMKONG_TARGET_SERIAL

/*  Keep the serial instantiations below actually scalar, regardless of build type. Without this,
 *  -O3 + LTO can vectorize or clone the serial kernels under AVX-512 callers in dispatch_*.c, which
 *  wastes ~1 MB of binary and — more importantly — breaks the nk_*_serial-as-scalar-oracle contract
 *  that tests and the numerical-stability docs in this header rely on.
 *
 *  Clang gets no blanket region here: one expansion of @c nk_define_cross_packed_ /
 *  @c nk_define_cross_symmetric_ emits @c always_inline @c _aligned_ fast paths next to the kernel,
 *  and clang rejects @c noinline there; `no-ipa-cp-clone` stops the cloning instead. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

/*  Size bias for release. Gated on NDEBUG so Debug builds keep -O0 for stepping. */
#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("s", on)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("Os")
#endif
#endif

/* F64 GEMM: depth_simd_dimensions=2 (2 f64s = 16 bytes) */
nk_define_cross_pack_size_(dots, f64, serial, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, f64, serial)
nk_define_cross_pack_(dots, f64, serial, f64, f64, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                      nk_store_b128_serial_, nk_partial_store_b64x2_serial_,
                      /*simd_width=*/2, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f64_,
                      /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, f64, serial, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                           nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                           nk_dot_f64x2_update_serial, nk_dot_f64x2_finalize_serial, nk_store_b256_serial_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, serial, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                        nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                        nk_load_b128_serial_, nk_partial_load_b64x2_serial_, nk_dot_f64x2_update_serial,
                        nk_dot_f64x2_finalize_serial, nk_store_b256_serial_, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

/* F32 GEMM: depth_simd_dimensions=4 (4 f32s = 16 bytes) */
nk_define_cross_pack_size_(dots, f32, serial, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/4,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, f32, serial)
nk_define_cross_pack_(dots, f32, serial, f32, f32, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                      /*simd_width=*/4, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f32_,
                      /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, f32, serial, f32, f64, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b256_vec_t,
                           nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                           nk_dot_f32x4_update_serial, nk_dot_f32x4_finalize_serial, nk_store_b256_serial_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, serial, f32, f32, f64, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b256_vec_t,
                        nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                        nk_load_b128_serial_, nk_partial_load_b32x4_serial_, nk_dot_f32x4_update_serial,
                        nk_dot_f32x4_finalize_serial, nk_store_b256_serial_, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* F16 packed GEMM: pre-upcast B to f32 and process 4 logical dimensions per 128-bit step. */
nk_define_cross_pack_size_(dots, f16, serial, f16, f32, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/4,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, f16, serial)
nk_define_cross_pack_(dots, f16, serial, f16, f32, nk_b128_vec_t, nk_load_f16x4_to_f32x4_serial_,
                      nk_partial_load_f16x4_to_f32x4_serial_, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                      /*simd_width=*/4, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_f16_,
                      /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, f16, serial, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_serial_t, nk_b128_vec_t,
                           nk_dot_f16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                           nk_dot_f16x8_update_serial, nk_dot_f16x8_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, serial, f16, f32, f32, nk_b128_vec_t, nk_dot_through_f32x4_state_serial_t,
                        nk_b128_vec_t, nk_dot_through_f32x4_init_serial, nk_load_f16x4_to_f32x4_serial_,
                        nk_partial_load_f16x4_to_f32x4_serial_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                        nk_dot_through_f32x4_update_serial, nk_dot_through_f32x4_finalize_serial, nk_store_b128_serial_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* BF16 GEMM: depth_simd_dimensions=8 (8 bf16s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, bf16, serial, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, bf16, serial)
nk_define_cross_pack_(dots, bf16, serial, bf16, bf16, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b16x8_serial_, nk_store_b128_serial_, nk_partial_store_b16x8_serial_,
                      /*simd_width=*/8, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_bf16_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, bf16, serial, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                           nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                           nk_dot_bf16x8_update_serial, nk_dot_bf16x8_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, serial, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                        nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                        nk_load_b128_serial_, nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_serial,
                        nk_dot_bf16x8_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* I8 GEMM: depth_simd_dimensions=16 (16 i8s = 16 bytes), I32 accumulator */
nk_define_cross_pack_size_(dots, i8, serial, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, i8, serial)
nk_define_cross_pack_(dots, i8, serial, i8, i8, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i8_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, i8, serial, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_i8x16_update_serial, nk_dot_i8x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, serial, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_serial,
                        nk_dot_i8x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=16 (16 u8s = 16 bytes), U32 accumulator */
nk_define_cross_pack_size_(dots, u8, serial, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, u8, serial)
nk_define_cross_pack_(dots, u8, serial, u8, u8, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, u8, serial, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_u8x16_update_serial, nk_dot_u8x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, serial, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_serial,
                        nk_dot_u8x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E4M3 GEMM: depth_simd_dimensions=16 (16 e4m3s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e4m3, serial, e4m3, e4m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e4m3, serial)
nk_define_cross_pack_(dots, e4m3, serial, e4m3, e4m3, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e4m3, serial, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e4m3x16_update_serial, nk_dot_e4m3x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, serial, e4m3, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e4m3x16_update_serial,
                        nk_dot_e4m3x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=16 (16 e5m2s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e5m2, serial, e5m2, e5m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e5m2, serial)
nk_define_cross_pack_(dots, e5m2, serial, e5m2, e5m2, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e5m2, serial, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e5m2x16_update_serial, nk_dot_e5m2x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, serial, e5m2, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e5m2x16_update_serial,
                        nk_dot_e5m2x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/*  Block-scaled serial products take the exact path for every block: raw packs, block sums exact
 *  in F32, an F32 sum with an I32 exponent across blocks, and one rounding. */
nk_define_cross_pack_size_(dots, mxfp8e4m3, serial, e4m3, e4m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, mxfp8e4m3, serial)
nk_define_cross_pack_(dots, mxfp8e4m3, serial, e4m3, e4m3, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_scaled_serial_(dots, mxfp8e4m3, nk_dot_f32_from_wide_)

nk_define_cross_pack_size_(dots, mxfp6e2m3, serial, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, mxfp6e2m3, serial)
nk_define_cross_pack_(dots, mxfp6e2m3, serial, e2m3, e2m3, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_scaled_serial_(dots, mxfp6e2m3, nk_dot_f32_from_wide_)

nk_define_cross_pack_size_(dots, mxfp6e3m2, serial, e3m2, e3m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, mxfp6e3m2, serial)
nk_define_cross_pack_(dots, mxfp6e3m2, serial, e3m2, e3m2, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e3m2_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_scaled_serial_(dots, mxfp6e3m2, nk_dot_f32_from_wide_)

nk_define_cross_pack_size_(dots, mxfp8e5m2, serial, e5m2, e5m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, mxfp8e5m2, serial)
nk_define_cross_pack_(dots, mxfp8e5m2, serial, e5m2, e5m2, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_scaled_serial_(dots, mxfp8e5m2, nk_dot_f32_from_wide_)

/* E2M3 GEMM: depth_simd_dimensions=16 (16 e2m3s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e2m3, serial, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e2m3, serial)
nk_define_cross_pack_(dots, e2m3, serial, e2m3, e2m3, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e2m3, serial, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e2m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e2m3x16_update_serial, nk_dot_e2m3x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, serial, e2m3, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e2m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e2m3x16_update_serial,
                        nk_dot_e2m3x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E2M1 GEMM: depth_simd_dimensions=16, 8 bytes = 16 nibbles, doubled values in I32 accumulator */
nk_define_cross_pack_size_(dots, e2m1, serial, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e2m1, serial)
nk_define_cross_pack_(dots, e2m1, serial, e2m1x2, e2m1x2, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e2m1, serial, e2m1x2, f32, nk_b64_vec_t, nk_dot_e2m1x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e2m1x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                           nk_dot_e2m1x16_update_serial, nk_dot_e2m1x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, e2m1, serial, e2m1x2, e2m1x2, f32, nk_b64_vec_t, nk_dot_e2m1x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e2m1x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                        nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_e2m1x16_update_serial,
                        nk_dot_e2m1x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)

/* NVFP4 and MXFP4 GEMM: raw packs and the exact path, as the MX formats above */
nk_define_cross_pack_size_(dots, nvfp4, serial, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, nvfp4, serial)
nk_define_cross_pack_(dots, nvfp4, serial, e2m1x2, e2m1x2, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_scaled_serial_(dots, nvfp4, nk_dot_f32_from_wide_)

nk_define_cross_pack_size_(dots, mxfp4, serial, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, mxfp4, serial)
nk_define_cross_pack_(dots, mxfp4, serial, e2m1x2, e2m1x2, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_scaled_serial_(dots, mxfp4, nk_dot_f32_from_wide_)

/* E3M2 GEMM: depth_simd_dimensions=16 (16 e3m2s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e3m2, serial, e3m2, e3m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e3m2, serial)
nk_define_cross_pack_(dots, e3m2, serial, e3m2, e3m2, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e3m2_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e3m2, serial, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e3m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e3m2x16_update_serial, nk_dot_e3m2x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e3m2, serial, e3m2, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e3m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e3m2x16_update_serial,
                        nk_dot_e3m2x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U4 GEMM: u4x2 for both A and B */
nk_define_cross_pack_size_(dots, u4, serial, u4x2, u4x2, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, u4, serial)
nk_define_cross_pack_(dots, u4, serial, u4x2, u4x2, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u4_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, u4, serial, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                           nk_dot_u4x16_update_serial, nk_dot_u4x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, u4, serial, u4x2, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                        nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_u4x16_update_serial,
                        nk_dot_u4x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)

/* I4 GEMM: i4x2 for both A and B */
nk_define_cross_pack_size_(dots, i4, serial, i4x2, i4x2, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, i4, serial)
nk_define_cross_pack_(dots, i4, serial, i4x2, i4x2, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i4_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, i4, serial, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                           nk_dot_i4x16_update_serial, nk_dot_i4x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, i4, serial, i4x2, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                        nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_i4x16_update_serial,
                        nk_dot_i4x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)

/* U1 GEMM: u1x8 for both A and B */
nk_define_cross_pack_size_(dots, u1, serial, u1x8, u1x8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/8, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, u1, serial)
nk_define_cross_pack_(dots, u1, serial, u1x8, u1x8, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sum_u1_, /*depth_simd_dimensions=*/128,
                      /*dimensions_per_value=*/8, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, u1, serial, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_serial_t, nk_b128_vec_t,
                           nk_dot_u1x128_init_serial, nk_load_b128_serial_, nk_partial_load_b1x128_serial_,
                           nk_dot_u1x128_update_serial, nk_dot_u1x128_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_packed_(dots, u1, serial, u1x8, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_serial_t, nk_b128_vec_t,
                        nk_dot_u1x128_init_serial, nk_load_b128_serial_, nk_partial_load_b1x128_serial_,
                        nk_load_b128_serial_, nk_partial_load_b1x128_serial_, nk_dot_u1x128_update_serial,
                        nk_dot_u1x128_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)

#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("", on)
#elif defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

#endif // NUMKONG_TARGET_SERIAL

#define nk_define_cross_normalized_packed_(metric_name, input_type_name, isa_suffix, input_value_type,              \
                                           packed_value_type, dot_result_type, norm_value_type, final_result_type,  \
                                           vec_type, dots_packed_fn, from_dot_fn, compute_norm_fn, load_fn,         \
                                           partial_load_fn, store_fn, partial_store_fn, dimensions_per_value)       \
    NUMKONG_API nk_status_t nk_##metric_name##s_packed_##input_type_name##_##isa_suffix(                            \
        nk_cross_##input_type_name##_operand_t const *a_operand, void const *b_packed_buffer,                       \
        nk_##final_result_type##_t *c_matrix, nk_size_t row_count, nk_size_t column_count, nk_size_t depth,         \
        nk_size_t a_stride, nk_size_t c_stride, void *stream) {                                                     \
        nk_assert_(stream == NUMKONG_NULL);                                                                         \
        nk_status_t const status = dots_packed_fn(a_operand, b_packed_buffer, (nk_##dot_result_type##_t *)c_matrix, \
                                                  row_count, column_count, depth, a_stride, c_stride, stream);      \
        if (status != nk_success_k) return status;                                                                  \
        nk_cross_operand_t const a = nk_cross_operand_(nk_##input_type_name##_k, a_operand, a_stride);              \
                                                                                                                    \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;   \
        nk_##norm_value_type##_t const *b_norms = (nk_##norm_value_type##_t const *)((char const *)header +         \
                                                                                     header->norms_offset);         \
                                                                                                                    \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                         \
            nk_##input_value_type##_t const *a_row = (nk_##input_value_type##_t const *)((char const *)a.elements + \
                                                                                         row_index * a_stride);     \
            nk_##norm_value_type##_t query_norm;                                                                    \
            nk_cross_row_norm_(query_norm, norm_value_type, compute_norm_fn, input_type_name, a_row, depth,         \
                               dimensions_per_value, a, row_index);                                                 \
            nk_##dot_result_type##_t *r_row_dots = (nk_##dot_result_type##_t *)((char *)c_matrix +                  \
                                                                                row_index * c_stride);              \
            nk_##final_result_type##_t *r_row_out = (nk_##final_result_type##_t *)((char *)c_matrix +               \
                                                                                   row_index * c_stride);           \
                                                                                                                    \
            nk_size_t column_index = 0;                                                                             \
            for (; column_index + 4 <= column_count; column_index += 4) {                                           \
                vec_type dots_vec, norms_vec, results_vec;                                                          \
                load_fn(r_row_dots + column_index, &dots_vec);                                                      \
                load_fn(b_norms + column_index, &norms_vec);                                                        \
                from_dot_fn(&dots_vec, query_norm, &norms_vec, &results_vec);                                       \
                store_fn(&results_vec, r_row_out + column_index);                                                   \
            }                                                                                                       \
            if (column_index < column_count) {                                                                      \
                vec_type dots_vec = {{0}}, norms_vec = {{0}}, results_vec;                                          \
                partial_load_fn(r_row_dots + column_index, &dots_vec, column_count - column_index);                 \
                partial_load_fn(b_norms + column_index, &norms_vec, column_count - column_index);                   \
                from_dot_fn(&dots_vec, query_norm, &norms_vec, &results_vec);                                       \
                partial_store_fn(&results_vec, r_row_out + column_index, column_count - column_index);              \
            }                                                                                                       \
        }                                                                                                           \
        return nk_success_k;                                                                                        \
    }

#define nk_define_cross_normalized_symmetric_(metric_name, input_type_name, isa_suffix, input_value_type,          \
                                              dot_result_type, norm_value_type, final_result_type, vec_type,       \
                                              dots_symmetric_fn, from_dot_fn, compute_norm_fn, load_fn,            \
                                              partial_load_fn, store_fn, partial_store_fn, dimensions_per_value)   \
    NUMKONG_API nk_status_t nk_##metric_name##s_symmetric_##input_type_name##_##isa_suffix(                        \
        nk_cross_##input_type_name##_operand_t const *vectors_operand, nk_size_t vectors_count, nk_size_t depth,   \
        nk_size_t stride, nk_##final_result_type##_t *result, nk_size_t result_stride, nk_size_t row_start,        \
        nk_size_t row_count, void *stream) {                                                                       \
        nk_assert_(stream == NUMKONG_NULL);                                                                        \
        row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;           \
        nk_status_t const status = dots_symmetric_fn(vectors_operand, vectors_count, depth, stride,                \
                                                     (nk_##dot_result_type##_t *)result, result_stride, row_start, \
                                                     row_count, stream);                                           \
        if (status != nk_success_k) return status;                                                                 \
        nk_cross_operand_t const vectors = nk_cross_operand_(nk_##input_type_name##_k, vectors_operand, stride);   \
                                                                                                                   \
        /* Cache row norms in the result diagonal (O(row_count) calls) */                                          \
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {                    \
            nk_##input_value_type##_t const *row_vector =                                                          \
                (nk_##input_value_type##_t const *)((char const *)vectors.elements + row_index * stride);          \
            nk_##norm_value_type##_t *row_diag = (nk_##norm_value_type##_t *)((char *)result +                     \
                                                                              row_index * result_stride);          \
            nk_cross_row_norm_(row_diag[row_index], norm_value_type, compute_norm_fn, input_type_name, row_vector, \
                               depth, dimensions_per_value, vectors, row_index);                                   \
        }                                                                                                          \
                                                                                                                   \
        /* Column-first post-processing with 256-element norm cache */                                             \
        nk_##norm_value_type##_t column_norms[256];                                                                \
        for (nk_size_t column_chunk_start = 0; column_chunk_start < vectors_count; column_chunk_start += 256) {    \
            nk_size_t column_chunk_end = column_chunk_start + 256 < vectors_count ? column_chunk_start + 256       \
                                                                                  : vectors_count;                 \
                                                                                                                   \
            /* Pre-compute norms for this column chunk — each column visited exactly once */                       \
            for (nk_size_t col = column_chunk_start; col < column_chunk_end; ++col) {                              \
                nk_##input_value_type##_t const *column_vector =                                                   \
                    (nk_##input_value_type##_t const *)((char const *)vectors.elements + col * stride);            \
                nk_cross_row_norm_(column_norms[col - column_chunk_start], norm_value_type, compute_norm_fn,       \
                                   input_type_name, column_vector, depth, dimensions_per_value, vectors, col);     \
            }                                                                                                      \
                                                                                                                   \
            /* Sweep assigned rows against this column chunk */                                                    \
            for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {                \
                nk_size_t j_start = row_index + 1 > column_chunk_start ? row_index + 1 : column_chunk_start;       \
                if (j_start >= column_chunk_end) continue;                                                         \
                char *row_ptr = (char *)result + row_index * result_stride;                                        \
                nk_##norm_value_type##_t sumsq_i = ((nk_##norm_value_type##_t *)row_ptr)[row_index];               \
                nk_##dot_result_type##_t *r_dots = (nk_##dot_result_type##_t *)row_ptr;                            \
                nk_##final_result_type##_t *r_out = (nk_##final_result_type##_t *)row_ptr;                         \
                                                                                                                   \
                /* 4-wide vectorized loop */                                                                       \
                nk_size_t j = j_start;                                                                             \
                for (; j + 4 <= column_chunk_end; j += 4) {                                                        \
                    vec_type target_norms_vec;                                                                     \
                    load_fn(&column_norms[j - column_chunk_start], &target_norms_vec);                             \
                    vec_type dots_vec, results_vec;                                                                \
                    load_fn(r_dots + j, &dots_vec);                                                                \
                    from_dot_fn(&dots_vec, sumsq_i, &target_norms_vec, &results_vec);                              \
                    store_fn(&results_vec, r_out + j);                                                             \
                }                                                                                                  \
                /* Remainder */                                                                                    \
                if (j < column_chunk_end) {                                                                        \
                    vec_type dots_vec = {{0}}, norms_vec = {{0}}, results_vec;                                     \
                    partial_load_fn(r_dots + j, &dots_vec, column_chunk_end - j);                                  \
                    partial_load_fn(&column_norms[j - column_chunk_start], &norms_vec, column_chunk_end - j);      \
                    from_dot_fn(&dots_vec, sumsq_i, &norms_vec, &results_vec);                                     \
                    partial_store_fn(&results_vec, r_out + j, column_chunk_end - j);                               \
                }                                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
                                                                                                                   \
        /* Zero diagonals */                                                                                       \
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {                    \
            nk_##final_result_type##_t *r_out = (nk_##final_result_type##_t *)((char *)result +                    \
                                                                               row_index * result_stride);         \
            r_out[row_index] = 0;                                                                                  \
        }                                                                                                          \
        return nk_success_k;                                                                                       \
    }

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif // NUMKONG_DOTS_SERIAL_H
